# MathTutor-RL

**Online reinforcement learning for a personalised mathematics tutoring agent.**

Extends [OpenClaw-RL](../openclaw-rl) with four domain-specific components:

| Component | Purpose |
|---|---|
| **SymPy Verifier** | Multi-granularity symbolic reward — replaces the LLM-as-judge PRM with deterministic algebraic checking |
| **BKT Student Model** | Bayesian Knowledge Tracing per Knowledge Component (KC); drives ZPD-based problem selection |
| **KC-Aware OPD** | Knowledge-Component tagger enriches the OPD teacher context before hint extraction |
| **Memory Files** | Persistent student profile across sessions (USER.md, MEMORY.md, daily logs, .learnings/) |

## Architecture

```
Student Interface
        │ message / answer
        ▼
OpenClaw Gateway
        │ X-Session-Id, X-Turn-Type, X-Reference-Answer
        ▼
MathTutorAPIServer  (PORT 30000)
  ├── forward ──────────────────────► SGLang (DeepSeek-Math-7B)
  │                                        │ tutor response
  │   ◄─────────────────────────────────────┘
  ├── SymPy Verifier  (deterministic, no model call)
  │     r_final  ±1.0   answer correct/wrong
  │     r_step   +0.5   per verified algebraic step
  │     r_form   +0.1   LaTeX well-formed
  │     r_redun  −0.2   ≥2 redundant steps
  │     r_ped    ±0.3   student correct after hint
  ├── BKT Student Model  →  MEMORY.md
  ├── KC Tagger          →  enriches OPD teacher context (optional)
  └── Sample ──────────────────────► GRPO Trainer (async, every 64 interactions)
```

## Reward Function

```
R(s, a) = r_final + r_step + r_form + r_redun + r_pedagogy
```

| Term | Value | Trigger |
|---|---|---|
| `r_final` | ±1.0 | SymPy confirms/rejects final boxed answer |
| `r_step` | +0.5 / step | Intermediate algebraic line symbolically verified |
| `r_form` | +0.1 | LaTeX notation well-formed (`\boxed{}`, `$$…$$`, etc.) |
| `r_pedagogy` | ±0.3 | Student solves next problem correctly / incorrectly after hint |
| `r_redun` | −0.2 | ≥ 2 redundant steps detected |

## Bayesian Knowledge Tracing

Standard 4-parameter BKT per KC (Eq. 2):

```
P(mastered | obs) = P(obs | mastered) × P(mastered) / P(obs)
P(L_{n+1})        = P(L_n | obs) + (1 − P(L_n | obs)) × p_learn
```

Problems are sampled from the **Zone of Proximal Development**: KCs with mastery ∈ [0.40, 0.70].

## Knowledge Component Taxonomy (42 KCs)

| Category | Count | Examples |
|---|---|---|
| Procedural | 12 | `solve_linear_eq`, `fraction_add_sub`, `exponent_rules` |
| Conceptual | 12 | `function_concept`, `limit_concept`, `inequality_concept` |
| Strategic | 10 | `choose_method`, `substitution_strategy`, `check_solution` |
| Representational | 8 | `latex_notation`, `graph_reading`, `unit_analysis` |

## Memory Files

```
{MATH_TUTOR_MEMORY_DIR}/{student_id}/
  USER.md          — goals, grade level, explanation style preference
  MEMORY.md        — per-KC mastery scores (auto-updated each session)
  YYYY-MM-DD.md    — daily session log: turns, rewards, corrections, hints
  .learnings/      — pending behaviour patterns; promotable by cron job
```

## Module Files

| File | Description |
|---|---|
| `sympy_verifier.py` | Multi-granularity reward computation using SymPy |
| `bkt_student_model.py` | 4-parameter BKT model with ZPD helpers |
| `kc_tagger.py` | Rule-based KC error taxonomy (42 KCs, 4 categories) |
| `memory_manager.py` | Read/write helpers for USER.md, MEMORY.md, session logs |
| `math_tutor_api_server.py` | FastAPI proxy server + reward engine + BKT state |
| `math_tutor_rollout.py` | SLIME rollout entry point |
| `run_deepseek_math_7b.sh` | GRPO training script for DeepSeek-Math-7B |

## Quick Start

### 1. Configure paths

```bash
export HF_CKPT=/path/to/deepseek-math-7b-instruct
export MATH_TUTOR_MEMORY_DIR=/data/students
```

### 2. Launch training

```bash
cd math-tutor-rl
bash run_deepseek_math_7b.sh
```

### 3. Enable KC-aware OPD (optional)

```bash
OPD_ENABLE=1 OPD_MODEL_PATH=/path/to/judge-model bash run_deepseek_math_7b.sh
```

### 4. Send tutoring requests

Use standard OpenAI chat API format with extra headers:

```bash
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: student_alice_session_001" \
  -H "X-Turn-Type: main" \
  -H "X-Reference-Answer: 2" \
  -d '{
    "model": "deepseek-math-7b",
    "messages": [
      {"role": "user", "content": "Please help me solve: 2x + 3 = 7"}
    ]
  }'
```

**Request headers / body fields:**

| Key | Description |
|---|---|
| `X-Session-Id` | Session / student ID (also used as memory file key) |
| `X-Turn-Type` | `main` (training turn) or `side` (skip, no gradient) |
| `X-Session-Done` | `1` to flush session, save memory, and clean up |
| `X-Reference-Answer` | Correct answer for the current problem (for r_final + pedagogy) |
| `is_hint_turn` | `true` if this is a hint-only response (enables r_pedagogy tracking) |

## GPU Requirements

| Configuration | GPUs |
|---|---|
| SymPy-only (no OPD) | 6 (4 actor + 2 rollout) |
| With KC-aware OPD | 8 (4 actor + 2 rollout + 2 judge) |

Runs on a single RTX 4090 (24 GB) with `NUM_GPUS=1 ACTOR_GPUS=1 ROLLOUT_GPUS=1 OPD_ENABLE=0`.

## Differences from openclaw-rl

| Feature | openclaw-rl | math-tutor-rl |
|---|---|---|
| Reward signal | LLM judge (PRM) | SymPy symbolic verifier |
| Student model | None | BKT per-KC mastery |
| Error analysis | None | 42-KC taxonomy |
| OPD enrichment | Standard hint | KC-tagged hint |
| Memory | None | USER.md + MEMORY.md + session logs |
| Target model | Qwen3 / Qwen3.5 | DeepSeek-Math-7B |

## Reference

Based on the MathTutor-RL proposal:
- SymPy reward replaces LLM judge (zero additional model overhead)
- BKT mastery scoring follows the standard 4-parameter model
- GRPO eliminates the Critic network (~40% GPU memory saving vs PPO)
- KC-aware OPD closes the feedback loop between tutoring quality and knowledge transfer
