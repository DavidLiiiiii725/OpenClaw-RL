"""MathTutor-RL API server.

Extends the OpenClaw proxy pattern with four domain-specific additions:

1. SymPy Verifier     — replaces the LLM-as-judge PRM with deterministic
                        symbolic step-level rewards (zero extra model overhead).
2. BKT Student Model  — per-session Bayesian Knowledge Tracing; mastery
                        scores are persisted to MEMORY.md at session end.
3. KC-Aware OPD       — Knowledge Component tagger enriches the teacher
                        context before hint extraction, improving the
                        token-level directive signal.
4. Memory Files       — USER.md, MEMORY.md, YYYY-MM-DD.md, .learnings/
                        under ``{MATH_TUTOR_MEMORY_DIR}/{student_id}/``.

Request conventions (on top of the standard OpenAI chat body):
  • X-Session-Id     — session identifier (also used as student_id)
  • X-Turn-Type      — "main" (training turn) | "side" (no training data)
  • X-Session-Done   — "1" / "true" → flush session and update memory
  • X-Reference-Answer — ground-truth answer for the current problem;
                         used for r_final and pedagogy tracking.
  • Body field ``student_id`` — override student_id (defaults to session_id)
  • Body field ``is_hint_turn`` — "true" if tutor response is a hint only

Reward formula (Eq. 1):
  R(s,a) = r_final + r_step + r_form + r_redun + r_pedagogy

  r_final   : ±1.0  (SymPy final answer check)
  r_step    : +0.5  per verified algebraic step
  r_form    : +0.1  LaTeX well-formed bonus
  r_redun   : −0.2  redundant-step penalty
  r_pedagogy: ±0.3  student correct / incorrect on next problem after hint
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import queue
import re
import threading
import time
from itertools import count
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from bkt_student_model import BKTStudentModel
from kc_tagger import tag_error, build_kc_hint_prefix
import memory_manager as mm
from sympy_verifier import (
    compute_math_reward,
    R_PEDAGOGY_POS,
    R_PEDAGOGY_NEG,
    _extract_final_answer,
)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)

_NON_STANDARD_BODY_KEYS = {
    "session_id", "session_done", "turn_type",
    "student_id", "is_hint_turn",
}


# ---------------------------------------------------------------------------
# Utility functions (shared with existing OpenClaw servers)
# ---------------------------------------------------------------------------

def _flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content) if content is not None else ""


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)
        if m.get("tool_calls"):
            m["tool_calls"] = [_normalize_tool_call(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


def _normalize_tool_call(tc: dict) -> dict:
    tc = dict(tc)
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        tc["function"] = fn
    return tc


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _build_hint_judge_messages(
    response_text: str, next_state_text: str, next_state_role: str = "user"
) -> list[dict]:
    """OPD judge prompt: extract a directive hint from the next state."""
    system = (
        "You are a math tutoring coach extracting a hindsight hint.\n"
        "You see the tutor's response at turn t and the student's reply at t+1.\n\n"
        "Decide whether the student reply reveals useful information that could "
        "have improved the tutor's previous explanation.\n\n"
        "Output rules (strict):\n"
        "- Include exactly one \\boxed{1} (hint useful) or \\boxed{-1} (not useful).\n"
        "- If \\boxed{1}: provide a concise 1-3 sentence math hint inside "
        "[HINT_START] … [HINT_END]. The hint must be mathematically accurate and "
        "directly actionable.\n"
        "- If \\boxed{-1}: do not add a hint block.\n"
    )
    user = (
        f"## Tutor response (turn t)\n{response_text}\n\n"
        f"## Student reply (turn t+1) [role: {next_state_role}]\n{next_state_text}\n\n"
        "Output your decision and (if positive) the hint."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_judge_result(text: str) -> tuple[Optional[int], str]:
    boxed = _BOXED_RE.findall(text)
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1):
        score = None
    hints = _HINT_RE.findall(text)
    hint = hints[-1].strip() if hints else ""
    return score, hint


def _select_best_hint(votes: list[dict]) -> Optional[dict]:
    good = [
        v for v in votes
        if v.get("score") == 1 and isinstance(v.get("hint"), str)
        and len(v["hint"].strip()) > 10
    ]
    if not good:
        return None
    return max(good, key=lambda v: len(v["hint"].strip()))


def _append_hint_to_messages(messages: list[dict], hint: str) -> list[dict]:
    """Inject the KC-enriched hint into the last user message."""
    import copy
    cloned = copy.deepcopy(messages)
    if not cloned:
        return [{"role": "user", "content": f"[hint]\n{hint}"}]

    target_idx = None
    for i in range(len(cloned) - 1, -1, -1):
        if cloned[i].get("role") == "user":
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cloned) - 1

    content = _flatten_message_content(cloned[target_idx].get("content"))
    cloned[target_idx]["content"] = (content + f"\n\n[hint]\n{hint.strip()}").strip()
    return cloned


# ---------------------------------------------------------------------------
# Pass-through reward_func / generate for slime compatibility
# ---------------------------------------------------------------------------

async def reward_func(args, sample_or_samples, **kwargs):
    """Reward function: scores are already stored in sample.reward by the server."""
    if isinstance(sample_or_samples, list):
        return [
            {"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}
            for s in sample_or_samples
        ]
    s = sample_or_samples
    return {"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    """Standalone generate function for slime eval path."""
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    messages = (
        sample.prompt
        if isinstance(sample.prompt, list)
        else [{"role": "user", "content": str(sample.prompt)}]
    )
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    payload = {"input_ids": input_ids, "sampling_params": sampling_params, "return_logprob": True}
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        output = response.json()
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    pairs = meta.get("output_token_logprobs", [])
    if isinstance(pairs, list) and pairs:
        token_ids = [int(p[1]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
        logprobs = [float(p[0]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
    else:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        logprobs = [0.0] * len(token_ids)
    sample.tokens = input_ids + token_ids
    sample.response = text
    sample.response_length = len(token_ids)
    sample.rollout_log_probs = logprobs
    sample.loss_mask = [1] * len(token_ids)
    sample.status = Sample.Status.COMPLETED
    return sample


# ---------------------------------------------------------------------------
# MathTutorAPIServer
# ---------------------------------------------------------------------------

class MathTutorAPIServer:
    """Proxy + reward engine for the MathTutor-RL system.

    Handles:
    - Forwarding requests to SGLang (tutor model)
    - Computing multi-granularity SymPy rewards deterministically
    - Tracking BKT state and pedagogy rewards across turns
    - Optional KC-aware OPD: if prm_enable=True, a separate LLM judge
      extracts KC-enriched hints and provides teacher log-probs
    - Persisting student memory files at session end
    """

    def __init__(self, args, output_queue: queue.Queue, submission_enabled: threading.Event):
        self.args = args
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.sglang_chat_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
        )
        self.sglang_health_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/health"
        )
        self.expected_api_key = os.getenv("SGLANG_API_KEY", "")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "30000"))
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "deepseek-math-7b")

        self._index_counter = count(0)
        self._group_counter = count(0)

        # Per-session state
        self._turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}
        self._session_effective: dict[str, int] = {}

        # BKT: {session_id: BKTStudentModel}
        self._bkt_models: dict[str, BKTStudentModel] = {}
        # Reference answers: {session_id: {turn_num: ref_answer}}
        self._ref_answers: dict[str, dict[int, str]] = {}
        # Hint flags: {session_id: {turn_num: bool}}
        self._hint_flags: dict[str, dict[int, bool]] = {}
        # Pending pedagogy turns: {session_id: {turn_num: None|bool}}
        # None = still waiting, True/False = resolved
        self._pedagogy_pending: dict[str, dict[int, Optional[bool]]] = {}

        # Memory file configuration
        self._memory_dir = os.getenv("MATH_TUTOR_MEMORY_DIR", mm._DEFAULT_MEMORY_DIR)

        # OPD judge (optional — for KC-aware hint extraction)
        self._opd_enabled = getattr(args, "prm_enable", False)
        self._prm_m = int(os.getenv("PRM_M", getattr(args, "prm_m", 3)))
        self._prm_temperature = float(getattr(args, "prm_temperature", 0.6))
        self._prm_max_tokens = int(getattr(args, "prm_max_new_tokens", 4096))
        prm_ip = getattr(args, "prm_router_ip", None)
        prm_port = getattr(args, "prm_router_port", None)
        self._prm_url = (
            f"http://{prm_ip}:{prm_port}/generate" if prm_ip and prm_port else ""
        )
        self._prm_tokenizer = None
        if self._opd_enabled:
            prm_path = getattr(args, "prm_model_path", None) or args.hf_checkpoint
            self._prm_tokenizer = load_tokenizer(prm_path, trust_remote_code=True)
            logger.info("[MathTutor] OPD judge enabled: url=%s m=%d", self._prm_url, self._prm_m)

        # Eval scores (PRM-compat metric for monitoring)
        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()

        # Record file (optional debugging)
        self._record_file = (
            os.getenv("MATH_TUTOR_RECORD_FILE", "")
            if os.getenv("MATH_TUTOR_RECORD_ENABLED", "0") == "1"
            else ""
        )
        if self._record_file:
            os.makedirs(os.path.dirname(self._record_file), exist_ok=True)
            open(self._record_file, "w").close()

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="MathTutor-RL Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
            x_reference_answer: str | None = Header(default=None),
        ):
            owner: MathTutorAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                raise HTTPException(status_code=503, detail="submission paused for weight update")

            body = await request.json()
            session_id = x_session_id or body.get("session_id") or "unknown"
            turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()
            session_done = (
                (x_session_done and x_session_done.strip().lower() in {"1", "true", "yes", "on"})
                or str(body.get("session_done", "")).strip().lower() in {"1", "true", "yes", "on"}
            )
            ref_answer = (
                x_reference_answer
                or body.get("reference_answer", "")
                or ""
            )
            is_hint_turn = str(body.get("is_hint_turn", "false")).strip().lower() in {
                "1", "true", "yes"
            }

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body,
                session_id=session_id,
                turn_type=turn_type,
                session_done=session_done,
                ref_answer=ref_answer,
                is_hint_turn=is_hint_turn,
            )
            if stream:
                return StreamingResponse(
                    owner._stream_response(result), media_type="text/event-stream"
                )
            return JSONResponse(content=result["response"])

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # ------------------------------------------------------------------
    # BKT helpers
    # ------------------------------------------------------------------

    def _get_bkt(self, session_id: str) -> BKTStudentModel:
        if session_id not in self._bkt_models:
            # Try loading from persisted memory
            self._bkt_models[session_id] = mm.load_memory(self._memory_dir, session_id)
        return self._bkt_models[session_id]

    def _update_bkt_from_next_state(
        self,
        session_id: str,
        prev_turn_data: dict,
        next_state: dict,
    ) -> Optional[bool]:
        """Update BKT from student's observed response. Returns correct/incorrect/None."""
        ref_answer = prev_turn_data.get("ref_answer", "")
        if not ref_answer:
            return None

        next_text = _flatten_message_content(next_state.get("content", ""))
        student_ans = _extract_final_answer(next_text)
        if student_ans is None:
            return None

        # SymPy comparison
        from sympy_verifier import verify_final_answer
        score = verify_final_answer(next_text, ref_answer)
        correct = score > 0

        kc = prev_turn_data.get("kc_tag")
        if kc:
            new_mastery = self._get_bkt(session_id).update(kc, correct)
            logger.info(
                "[MathTutor] BKT update session=%s kc=%s correct=%s mastery=%.3f",
                session_id, kc, correct, new_mastery,
            )
        return correct

    # ------------------------------------------------------------------
    # SymPy reward evaluation
    # ------------------------------------------------------------------

    async def _sympy_evaluate(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict,
        next_state: dict,
    ) -> dict:
        """Compute multi-granularity SymPy reward and resolve pedagogy."""
        response_text = turn_data["response_text"]
        ref_answer = turn_data.get("ref_answer", "")
        is_hint = turn_data.get("is_hint", False)

        # Determine pedagogy bonus from student's next answer
        correct = await asyncio.to_thread(
            self._update_bkt_from_next_state, session_id, turn_data, next_state
        )
        pedagogy_bonus = 0.0
        if is_hint and correct is not None:
            pedagogy_bonus = R_PEDAGOGY_POS if correct else R_PEDAGOGY_NEG

        # Compute SymPy reward (CPU-bound → run in thread)
        reward = await asyncio.to_thread(
            compute_math_reward, response_text, ref_answer, pedagogy_bonus
        )

        # Optionally tag error KC for the student's next message (for logging/BKT)
        next_text = _flatten_message_content(next_state.get("content", "")) if next_state else ""
        kc = tag_error(next_text, response_text)
        turn_data["kc_tag"] = kc  # persist for BKT update on following turn

        logger.info(
            "%s[MathTutor] SymPy reward session=%s turn=%d "
            "score=%.3f r_final=%.1f r_step=%.2f r_form=%.2f "
            "r_redun=%.2f r_ped=%.2f kc=%s%s",
            _CYAN, session_id, turn_num,
            reward["score"], reward["r_final"], reward["r_step"],
            reward["r_form"], reward["r_redun"], reward["r_pedagogy"], kc, _RESET,
        )

        # Track eval score for monitoring
        with self._eval_scores_lock:
            self._eval_scores.append(reward["score"])

        return reward

    # ------------------------------------------------------------------
    # OPD judge (optional, KC-aware)
    # ------------------------------------------------------------------

    async def _query_judge_once(self, judge_prompt: str, vote_id: int) -> dict:
        if not self._prm_url:
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}
        payload = {
            "text": judge_prompt,
            "sampling_params": {
                "temperature": self._prm_temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": self._prm_max_tokens,
                "skip_special_tokens": False,
                "no_stop_trim": True,
            },
            "return_logprob": False,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self._prm_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            raw = data.get("text", data) if isinstance(data, dict) else str(data)
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            raw = str(raw)
            score, hint = _parse_judge_result(raw)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": raw}
        except Exception as exc:
            logger.warning("[MathTutor] OPD judge query failed (vote %d): %s", vote_id, exc)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _opd_extract_hint(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict,
        next_state: dict,
    ) -> Optional[str]:
        """Extract a KC-enriched hint via the OPD judge LLM."""
        if not self._opd_enabled or not self._prm_url:
            return None

        next_text = _flatten_message_content(next_state.get("content", ""))
        next_role = next_state.get("role", "user")
        response_text = turn_data["response_text"]

        # Tag the error KC
        kc = tag_error(next_text, response_text)
        kc_prefix = build_kc_hint_prefix(kc)

        judge_msgs = _build_hint_judge_messages(response_text, next_text, next_role)
        if self._prm_tokenizer:
            judge_prompt = self._prm_tokenizer.apply_chat_template(
                judge_msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            judge_prompt = "\n".join(m["content"] for m in judge_msgs)

        votes = await asyncio.gather(
            *[self._query_judge_once(judge_prompt, i) for i in range(self._prm_m)]
        )
        selected = _select_best_hint(list(votes))
        if selected is None:
            return None

        # Prepend KC tag to the extracted hint
        enriched_hint = f"{kc_prefix}\n{selected['hint'].strip()}"
        logger.info(
            "[MathTutor] OPD hint extracted session=%s turn=%d kc=%s hint_len=%d",
            session_id, turn_num, kc, len(enriched_hint),
        )
        return enriched_hint

    async def _compute_teacher_log_probs(
        self,
        input_ids: list[int],
        response_len: int,
    ) -> list[float]:
        """Get teacher log-probs from the OPD judge server for distillation."""
        start_len = max(0, len(input_ids) - response_len)
        payload = {
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": start_len,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(self._prm_url, json=payload)
            resp.raise_for_status()
            result = resp.json()
        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        inp = meta.get("input_token_logprobs")
        if not isinstance(inp, list):
            return [0.0] * response_len
        all_lp = []
        for item in inp:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                val = item[0]
                all_lp.append(float(val) if val is not None else 0.0)
            elif isinstance(item, dict) and "logprob" in item:
                all_lp.append(float(item["logprob"]))
            else:
                all_lp.append(0.0)
        if len(all_lp) > 1:
            all_lp = all_lp[1:]
        if len(all_lp) >= response_len:
            return all_lp[-response_len:]
        return [0.0] * (response_len - len(all_lp)) + all_lp

    # ------------------------------------------------------------------
    # Main request handler
    # ------------------------------------------------------------------

    async def _handle_request(
        self,
        body: dict,
        session_id: str,
        turn_type: str,
        session_done: bool,
        ref_answer: str,
        is_hint_turn: bool,
    ) -> dict:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        tools = body.get("tools")
        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self.served_model_name

        async with httpx.AsyncClient(timeout=None) as client:
            sglang_resp = await client.post(self.sglang_chat_url, json=forward_body)
            if sglang_resp.status_code != 200:
                logger.error(
                    "[MathTutor] SGLang returned %d: %s",
                    sglang_resp.status_code, sglang_resp.text[:1000],
                )
                sglang_resp.raise_for_status()
            output = sglang_resp.json()

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        tool_calls = assistant_msg.get("tool_calls") or []

        logger.info(
            "%s[MathTutor] [%s] session=%s msgs=%d%s",
            _YELLOW, turn_type, session_id, len(messages), _RESET,
        )
        logger.info(
            "%s[MathTutor] [%s] session=%s thinking=%d chars\n%s%s",
            _RED, turn_type, session_id, len(reasoning), content, _RESET,
        )

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)

            # When a new user message arrives, it is the next_state for the previous turn.
            if prev_turn_num > 0 and messages:
                prev_turn_data = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_turn_data is not None:
                    self._fire_reward_task(session_id, prev_turn_num, prev_turn_data, messages[-1])

            # Tokenise current turn
            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]

            prompt_text = self.tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                norm_msgs + [norm_resp], tools=tools, tokenize=False, add_generation_prompt=False
            )
            response_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip():
                logger.info("[MathTutor] MAIN session=%s → empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs += [0.0] * (len(response_ids) - len(response_logprobs))

            self._turn_counts[session_id] = prev_turn_num + 1
            turn_num = self._turn_counts[session_id]

            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "messages": messages,
                "tools": tools,
                "ref_answer": ref_answer,
                "is_hint": is_hint_turn,
                "kc_tag": None,
                "has_next_state": False,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._ref_answers.setdefault(session_id, {})[turn_num] = ref_answer
            self._hint_flags.setdefault(session_id, {})[turn_num] = is_hint_turn

            logger.info(
                "[MathTutor] MAIN session=%s turn=%d prompt=%d response=%d ref=%r hint=%s",
                session_id, turn_num, len(prompt_ids), len(response_ids),
                ref_answer[:20] if ref_answer else "", is_hint_turn,
            )
            self._maybe_submit_ready_samples(session_id)

        else:
            logger.info("[MathTutor] SIDE session=%s → skipped", session_id)

        if session_done:
            # Last turn: no next_state → submit with SymPy-only (no pedagogy)
            self._maybe_submit_ready_samples(session_id, force_last_turn=True)
            self._finalize_session(session_id)

        output["session_id"] = session_id
        return {"response": output}

    # ------------------------------------------------------------------
    # Reward task management
    # ------------------------------------------------------------------

    def _fire_reward_task(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict,
        next_state: dict,
    ):
        """Kick off async reward computation + optional OPD task for a turn."""
        turn_data["has_next_state"] = True
        task = asyncio.create_task(
            self._reward_and_opd(session_id, turn_num, turn_data, next_state)
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task

    async def _reward_and_opd(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict,
        next_state: dict,
    ) -> dict:
        """Compute SymPy reward and (optionally) KC-aware OPD teacher log-probs."""
        # 1. SymPy reward (always)
        reward = await self._sympy_evaluate(session_id, turn_num, turn_data, next_state)

        result: dict = {"reward": reward, "teacher_log_probs": None}

        # 2. OPD hint extraction (optional)
        if self._opd_enabled and self._prm_url:
            hint = await self._opd_extract_hint(session_id, turn_num, turn_data, next_state)
            if hint:
                enhanced_messages = _append_hint_to_messages(turn_data["messages"], hint)
                norm_enhanced = _normalize_messages_for_template(enhanced_messages)
                enhanced_prompt_text = self.tokenizer.apply_chat_template(
                    norm_enhanced,
                    tools=turn_data.get("tools"),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                enhanced_full = enhanced_prompt_text + turn_data["response_text"]
                enhanced_ids = self.tokenizer(enhanced_full, add_special_tokens=False)["input_ids"]
                teacher_lp = await self._compute_teacher_log_probs(
                    enhanced_ids, len(turn_data["response_ids"])
                )
                result["teacher_log_probs"] = teacher_lp

        return result

    def _maybe_submit_ready_samples(
        self,
        session_id: str,
        force_last_turn: bool = False,
    ):
        tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})

        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = tasks.get(turn_num)

            if task is None:
                # No next_state has arrived yet for this turn.
                if force_last_turn:
                    # Submit with SymPy score but no pedagogy (last turn of session).
                    pending.pop(turn_num, None)
                    self._safe_create_task(self._submit_turn_no_next_state(td, session_id))
                continue

            if not task.done():
                continue

            pending.pop(turn_num, None)
            tasks.pop(turn_num, None)
            try:
                result = task.result()
            except Exception as exc:
                logger.warning(
                    "[MathTutor] reward task failed session=%s turn=%d: %s",
                    session_id, turn_num, exc,
                )
                continue

            self._safe_create_task(self._submit_turn_sample(td, session_id, result))

    async def _submit_turn_no_next_state(self, turn_data: dict, session_id: str):
        """Submit the final turn of a session using SymPy-only reward (no next_state)."""
        reward = await asyncio.to_thread(
            compute_math_reward,
            turn_data["response_text"],
            turn_data.get("ref_answer", ""),
            0.0,  # no pedagogy
        )
        result = {"reward": reward, "teacher_log_probs": None}
        await self._submit_turn_sample(turn_data, session_id, result)

    async def _submit_turn_sample(
        self, turn_data: dict, session_id: str, result: dict
    ):
        """Build a Sample and push it onto the training queue."""
        reward = result["reward"]
        score = reward["score"]

        # Exclude turns with score=0 unless this session has produced zero
        # effective samples (at-least-one guarantee from OpenClaw).
        has_next_state = turn_data.get("has_next_state", False)
        exclude = not has_next_state or score == 0.0
        if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
            exclude = False
            logger.info(
                "[MathTutor] promoting session=%s turn score=0 → loss_mask=1 (at-least-one)",
                session_id,
            )

        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = reward  # full dict: score + component breakdown

        # Attach teacher log-probs for OPD distillation (if available)
        teacher_lp = result.get("teacher_log_probs")
        if teacher_lp:
            import torch
            if len(teacher_lp) > len(response_ids):
                teacher_lp = teacher_lp[: len(response_ids)]
            elif len(teacher_lp) < len(response_ids):
                teacher_lp += [0.0] * (len(response_ids) - len(teacher_lp))
            sample.teacher_log_probs = torch.tensor(teacher_lp, dtype=torch.float32)

        if not exclude:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1

        logger.info(
            "[MathTutor] submitted sample session=%s idx=%d score=%.3f "
            "exclude=%s prompt=%d response=%d",
            session_id, sample.index, score, exclude, len(prompt_ids), len(response_ids),
        )

        # Log turn event to daily session file
        mm.log_turn_event(
            self._memory_dir, session_id, session_id,
            turn_num=sample.index,
            reward=reward,
            kc=turn_data.get("kc_tag"),
            correct=None,
            hint_given=turn_data.get("is_hint", False),
        )

        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # ------------------------------------------------------------------
    # Session finalisation: persist BKT + memory files
    # ------------------------------------------------------------------

    def _finalize_session(self, session_id: str):
        """Persist BKT state, update memory files, and clean up session data."""
        bkt = self._bkt_models.get(session_id)
        if bkt:
            mm.save_memory(self._memory_dir, session_id, bkt)
            mm.record_learning_pattern(
                self._memory_dir,
                session_id,
                {
                    "session_id": session_id,
                    "kc_mastery": bkt.all_kcs(),
                    "zpd_kcs": bkt.get_zpd_kcs(),
                    "error_type": "session_summary",
                },
            )

        turn_count = self._turn_counts.get(session_id, 0)
        eff = self._session_effective.get(session_id, 0)
        logger.info(
            "[MathTutor] session=%s done turns=%d effective=%d",
            session_id, turn_count, eff,
        )

        # Cleanup
        self._turn_counts.pop(session_id, None)
        self._session_effective.pop(session_id, None)
        self._bkt_models.pop(session_id, None)
        self._ref_answers.pop(session_id, None)
        self._hint_flags.pop(session_id, None)
        self._pedagogy_pending.pop(session_id, None)
        self._prm_tasks.pop(session_id, None)
        self._pending_turn_data.pop(session_id, None)

    # ------------------------------------------------------------------
    # Eval score drain (for monitoring metric in rollout loop)
    # ------------------------------------------------------------------

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    # ------------------------------------------------------------------
    # Record purge (called when training resumes after weight update)
    # ------------------------------------------------------------------

    def purge_record_files(self):
        if not self._record_file:
            return
        try:
            open(self._record_file, "w").close()
            logger.info("[MathTutor] record file purged: %s", self._record_file)
        except OSError as exc:
            logger.warning("[MathTutor] failed to purge record file: %s", exc)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[MathTutor] background task failed: %s", exc, exc_info=exc)

    async def _stream_response(self, result: dict):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._readiness_thread = threading.Thread(
            target=self._wait_for_sglang_ready, daemon=True
        )
        self._readiness_thread.start()

    def _wait_for_sglang_ready(self):
        while True:
            try:
                r = httpx.get(self.sglang_health_url, timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        logger.info("[MathTutor] policy server ready")
        time.sleep(8)
        opd_line = ""
        if self._opd_enabled:
            opd_line = f"\n  KC-aware OPD enabled: {self._prm_url} (m={self._prm_m})"
        banner = (
            f"\n{'=' * 70}\n"
            f"  [MathTutor-RL] SymPy verifier active | BKT enabled\n"
            f"  proxy {self.host}:{self.port} → SGLang "
            f"{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
            f"{opd_line}\n"
            f"  memory dir: {self._memory_dir}\n"
            f"{'=' * 70}\n"
        )
        logger.info("%s%s%s", _GREEN, banner, _RESET)

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
