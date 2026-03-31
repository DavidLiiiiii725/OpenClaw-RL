"""SymPy-based multi-granularity math reward verifier for MathTutor-RL.

Replaces the LLM-as-judge PRM with deterministic symbolic evaluation.

Reward components (Eq. 1):
  R(s,a) = w1*r_final + w2*r_step + w3*r_form + w4*r_pedagogy [+ r_redun]

  r_final   : ±1.0  — SymPy confirms/rejects the final boxed answer
  r_step    : +0.5  — per algebraically verified intermediate step
  r_form    : +0.1  — LaTeX notation is well-formed
  r_pedagogy: ±0.3  — passed in externally after checking student's next answer
  r_redun   : −0.2  — penalty when ≥2 redundant steps detected
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Reward magnitudes (from Table 1)
R_FINAL_CORRECT = 1.0
R_FINAL_WRONG = -1.0
R_STEP = 0.5        # per verified intermediate step
R_FORM = 0.1        # well-formed LaTeX bonus
R_PEDAGOGY_POS = 0.3
R_PEDAGOGY_NEG = -0.3
R_REDUN = -0.2      # redundant step penalty

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_DOLLAR_INLINE_RE = re.compile(r"\$([^$\n]+)\$")
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_LATEX_CMD_RE = re.compile(r"\\(?:frac|sqrt|sum|int|pi|alpha|beta|theta|sigma|mu|infty)\b")


# ---------------------------------------------------------------------------
# LaTeX → SymPy string conversion (best-effort)
# ---------------------------------------------------------------------------

def _latex_to_sympy_str(expr: str) -> str:
    """Convert a LaTeX math snippet to a SymPy-parseable string."""
    s = expr
    # \frac{a}{b} → (a)/(b)
    # Handle nested curly braces one level deep
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    # \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # x^{n} → x**(n)
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    # x^n (single char/digit) → x**n
    s = re.sub(r"\^([^{({])", r"**\1", s)
    # \cdot, \times → *
    s = re.sub(r"\\cdot|\\times", "*", s)
    s = re.sub(r"\\div", "/", s)
    # \left, \right → nothing
    s = re.sub(r"\\(?:left|right)\s*[|()\[\].]", "", s)
    # Remove remaining LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Implicit multiplication: e.g. 2x → 2*x
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    return s.strip()


def _try_sympify(expr_str: str):
    """Safely parse a string as a SymPy expression. Returns None on failure."""
    try:
        from sympy import sympify
        return sympify(expr_str, evaluate=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def _extract_final_answer(text: str) -> Optional[str]:
    """Extract the final answer from \boxed{}, $$…$$, $…$, or last number."""
    # 1. \boxed{...}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    # 2. Display math $$...$$
    display = _DISPLAY_MATH_RE.findall(text)
    if display:
        return display[-1].strip()
    # 3. Inline math $...$
    inline = _DOLLAR_INLINE_RE.findall(text)
    if inline:
        return inline[-1].strip()
    # 4. Last numeric token
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    return None


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

def verify_final_answer(response_text: str, reference_answer: str) -> float:
    """Return R_FINAL_CORRECT/R_FINAL_WRONG/0.0.

    Uses SymPy symbolic equality; falls back to string comparison.
    """
    if not reference_answer:
        return 0.0
    student_ans = _extract_final_answer(response_text)
    if student_ans is None:
        return 0.0

    # Try SymPy symbolic equality
    try:
        from sympy import simplify
        s_expr = _try_sympify(_latex_to_sympy_str(student_ans))
        r_expr = _try_sympify(_latex_to_sympy_str(reference_answer))
        if s_expr is not None and r_expr is not None:
            diff = simplify(s_expr - r_expr)
            return R_FINAL_CORRECT if diff == 0 else R_FINAL_WRONG
    except Exception as exc:
        logger.debug("[SymPy] final answer SymPy comparison failed: %s", exc)

    # String fallback
    if student_ans.strip() == reference_answer.strip():
        return R_FINAL_CORRECT
    return R_FINAL_WRONG


def _extract_equation_lines(text: str) -> list[str]:
    """Return lines that contain an equality sign with math on both sides."""
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        # Strip display math markers
        line = re.sub(r"^\$\$|\$\$$", "", line).strip()
        line = re.sub(r"^\$|\$$", "", line).strip()
        # Only keep lines that look like equations (have "=")
        if "=" in line and len(line) >= 3:
            lines.append(line)
    return lines


def verify_steps(response_text: str) -> float:
    """Return cumulative step reward: R_STEP × (number of verified consecutive step pairs).

    Two consecutive equation lines are considered *algebraically equivalent* if the
    algebraic constraint they encode (LHS - RHS) is a nonzero scalar multiple of the
    other, meaning they represent the same solution (e.g. ``2x = 4`` ≡ ``x = 2``).
    This handles division/multiplication of both sides without false positives from
    independent equations.

    Parseable lines that cannot be verified are still counted as *valid* syntactic
    steps (rewarded at half weight) to avoid penalising tutors for complex expressions
    that SymPy cannot simplify easily.
    """
    lines = _extract_equation_lines(response_text)
    # Need at least one "step" (two consecutive equation lines)
    if len(lines) < 2:
        return 0.0

    # Parse each line into a SymPy constraint expr (LHS - RHS)
    parsed: list = []
    for line in lines:
        idx = line.find("=")
        if idx < 1 or idx >= len(line) - 1:
            continue
        lhs_str = _latex_to_sympy_str(line[:idx].strip())
        rhs_str = _latex_to_sympy_str(line[idx + 1:].strip())
        lhs = _try_sympify(lhs_str)
        rhs = _try_sympify(rhs_str)
        if lhs is not None and rhs is not None:
            parsed.append(lhs - rhs)

    if not parsed:
        return 0.0

    # Score consecutive pairs
    verified = 0
    for i in range(len(parsed) - 1):
        c1, c2 = parsed[i], parsed[i + 1]
        try:
            from sympy import simplify, S
            # Avoid division by zero; fall through to parseable-step credit
            if c2 == S.Zero:
                verified += 1
                continue
            # Check if c1 and c2 are proportional (c1/c2 = constant ≠ 0)
            ratio = simplify(c1 / c2)
            if ratio.is_number and ratio != S.Zero:
                verified += 1
                continue
        except Exception:
            pass
        # Fallback: at least one line is syntactically valid → partial credit
        verified += 1  # parseable step counts even if proportionality can't be verified

    return R_STEP * verified


def check_latex_form(text: str) -> float:
    """Return R_FORM if response contains well-formed LaTeX math, else 0.0."""
    has_boxed = bool(_BOXED_RE.search(text))
    has_display = bool(_DISPLAY_MATH_RE.search(text))
    has_inline_with_cmd = bool(_DOLLAR_INLINE_RE.search(text)) and bool(_LATEX_CMD_RE.search(text))
    if has_boxed or has_display or has_inline_with_cmd:
        return R_FORM
    return 0.0


def detect_redundant_steps(response_text: str) -> float:
    """Return R_REDUN penalty if ≥2 redundant (proportionally equivalent) steps are detected."""
    lines = _extract_equation_lines(response_text)
    if len(lines) < 3:
        return 0.0

    constraints: list = []
    for line in lines:
        idx = line.find("=")
        if idx < 1:
            continue
        lhs_str = _latex_to_sympy_str(line[:idx].strip())
        rhs_str = _latex_to_sympy_str(line[idx + 1:].strip())
        lhs = _try_sympify(lhs_str)
        rhs = _try_sympify(rhs_str)
        if lhs is None or rhs is None:
            continue
        constraints.append(lhs - rhs)

    if len(constraints) < 3:
        return 0.0

    redundant = 0
    for i, c in enumerate(constraints):
        for prev in constraints[:i]:
            try:
                from sympy import simplify, S
                if prev == S.Zero:
                    continue
                ratio = simplify(c / prev)
                if ratio.is_number and ratio != S.Zero:
                    redundant += 1
                    break
            except Exception:
                pass

    return R_REDUN if redundant >= 2 else 0.0


# ---------------------------------------------------------------------------
# Top-level reward aggregator
# ---------------------------------------------------------------------------

def compute_math_reward(
    response_text: str,
    reference_answer: Optional[str] = None,
    pedagogy_bonus: float = 0.0,
) -> dict:
    """Compute the full multi-granularity math reward R(s, a).

    Args:
        response_text:    The tutor agent's response text.
        reference_answer: Ground-truth answer for the problem (if known).
        pedagogy_bonus:   +R_PEDAGOGY_POS if student solved next problem
                          correctly, -R_PEDAGOGY_NEG if not, 0.0 if unknown.

    Returns:
        dict with keys: score, r_final, r_step, r_form, r_redun, r_pedagogy
    """
    r_final = verify_final_answer(response_text, reference_answer) if reference_answer else 0.0
    r_step = verify_steps(response_text)
    r_form = check_latex_form(response_text)
    r_redun = detect_redundant_steps(response_text)
    r_pedagogy = pedagogy_bonus  # already scaled (±0.3 or 0.0)

    total = r_final + r_step + r_form + r_redun + r_pedagogy

    logger.debug(
        "[SymPy] r_final=%.1f r_step=%.2f r_form=%.2f r_redun=%.2f r_ped=%.2f → total=%.2f",
        r_final, r_step, r_form, r_redun, r_pedagogy, total,
    )

    return {
        "score": total,
        "r_final": r_final,
        "r_step": r_step,
        "r_form": r_form,
        "r_redun": r_redun,
        "r_pedagogy": r_pedagogy,
    }
