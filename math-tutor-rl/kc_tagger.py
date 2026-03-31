"""Rule-based Knowledge Component (KC) tagger for MathTutor-RL.

Classifies student errors into one of 42 KCs across 4 categories:
  Procedural (12), Conceptual (12), Strategic (10), Representational (8)

The tagger uses lexical pattern matching against the student's incorrect
response and/or the reference solution.  It returns the single most
likely KC that explains the observed error.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KC taxonomy — 42 knowledge components
# ---------------------------------------------------------------------------

# Format: (kc_id, category, display_name)
KC_TAXONOMY: list[tuple[str, str, str]] = [
    # --- Procedural (12) ---
    ("arithmetic_ops",            "Procedural",       "Basic arithmetic operations"),
    ("fraction_simplify",         "Procedural",       "Simplifying fractions"),
    ("fraction_add_sub",          "Procedural",       "Adding/subtracting fractions"),
    ("fraction_mul_div",          "Procedural",       "Multiplying/dividing fractions"),
    ("solve_linear_eq",           "Procedural",       "Solving linear equations"),
    ("solve_quadratic",           "Procedural",       "Solving quadratic equations"),
    ("expand_simplify",           "Procedural",       "Expanding and simplifying expressions"),
    ("factoring",                 "Procedural",       "Factoring polynomials"),
    ("exponent_rules",            "Procedural",       "Applying exponent rules"),
    ("logarithm_rules",           "Procedural",       "Applying logarithm rules"),
    ("trig_identity",             "Procedural",       "Trigonometric identities"),
    ("derivative_rules",          "Procedural",       "Computing derivatives"),
    # --- Conceptual (12) ---
    ("variable_concept",          "Conceptual",       "Understanding variables and expressions"),
    ("eq_vs_expression",          "Conceptual",       "Distinguishing equations from expressions"),
    ("function_concept",          "Conceptual",       "Understanding functions and mappings"),
    ("limit_concept",             "Conceptual",       "Understanding limits"),
    ("derivative_concept",        "Conceptual",       "Conceptual understanding of derivatives"),
    ("integral_concept",          "Conceptual",       "Conceptual understanding of integrals"),
    ("proportion_concept",        "Conceptual",       "Proportionality and ratios"),
    ("probability_concept",       "Conceptual",       "Basic probability theory"),
    ("set_theory",                "Conceptual",       "Set operations and notation"),
    ("inequality_concept",        "Conceptual",       "Understanding inequalities"),
    ("coordinate_geometry",       "Conceptual",       "Points, lines, and planes"),
    ("vector_concept",            "Conceptual",       "Vector operations"),
    # --- Strategic (10) ---
    ("choose_method",             "Strategic",        "Selecting appropriate solution method"),
    ("problem_decompose",         "Strategic",        "Breaking down complex problems"),
    ("substitution_strategy",     "Strategic",        "Using substitution"),
    ("elimination_strategy",      "Strategic",        "Using elimination for systems"),
    ("proof_strategy",            "Strategic",        "Constructing mathematical proofs"),
    ("estimation_strategy",       "Strategic",        "Using estimation and approximation"),
    ("backward_reasoning",        "Strategic",        "Working backwards from goal"),
    ("pattern_recognition",       "Strategic",        "Identifying mathematical patterns"),
    ("check_solution",            "Strategic",        "Verifying solutions"),
    ("special_cases",             "Strategic",        "Handling edge cases and special values"),
    # --- Representational (8) ---
    ("latex_notation",            "Representational", "Using LaTeX math notation"),
    ("graph_reading",             "Representational", "Interpreting graphs"),
    ("table_interpretation",      "Representational", "Reading and using tables"),
    ("diagram_geometry",          "Representational", "Geometric diagram interpretation"),
    ("number_line",               "Representational", "Number line representation"),
    ("symbolic_notation",         "Representational", "Mathematical symbol conventions"),
    ("unit_analysis",             "Representational", "Unit conversion and dimensional analysis"),
    ("scientific_notation",       "Representational", "Scientific notation"),
]

KC_IDS: list[str] = [kc[0] for kc in KC_TAXONOMY]
KC_CATEGORIES: dict[str, str] = {kc[0]: kc[1] for kc in KC_TAXONOMY}
KC_NAMES: dict[str, str] = {kc[0]: kc[2] for kc in KC_TAXONOMY}

DEFAULT_KC = "arithmetic_ops"  # fallback if no pattern matches


# ---------------------------------------------------------------------------
# Pattern rules — (kc_id, [compiled regexes])
# ---------------------------------------------------------------------------
# Each rule fires when ANY of its patterns matches the combined error context.

def _p(*patterns: str) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_RULES: list[tuple[str, list[re.Pattern]]] = [
    # Procedural
    ("arithmetic_ops",        _p(r"arithmetic", r"\d+\s*[+\-*/]\s*\d+", r"add|subtract|multipl|divid",
                                  r"sum|difference|product|quotient")),
    ("fraction_simplify",     _p(r"simplif.*fraction", r"reduce.*fraction", r"common factor.*fraction",
                                  r"lowest terms", r"GCF|GCD")),
    ("fraction_add_sub",      _p(r"common denominator", r"add.*fraction", r"subtract.*fraction",
                                  r"LCD|LCM.*fraction")),
    ("fraction_mul_div",      _p(r"multipl.*fraction", r"divid.*fraction", r"reciprocal",
                                  r"invert.*multipl")),
    ("solve_linear_eq",       _p(r"linear equation", r"solve.*for\s+[a-z]", r"isolate.*variable",
                                  r"balance.*equation", r"one.step equation")),
    ("solve_quadratic",       _p(r"quadratic", r"x\^2|x²", r"discriminant",
                                  r"quadratic formula", r"completing the square")),
    ("expand_simplify",       _p(r"expand", r"distribute", r"FOIL", r"like terms",
                                  r"combine.*terms")),
    ("factoring",             _p(r"factor", r"difference of squares", r"perfect square trinomial",
                                  r"greatest common factor")),
    ("exponent_rules",        _p(r"exponent", r"power rule", r"product rule.*exponent",
                                  r"quotient rule.*exponent", r"negative exponent",
                                  r"zero exponent", r"a\^m.*a\^n")),
    ("logarithm_rules",       _p(r"log(?:arithm)?", r"ln\b", r"natural log",
                                  r"change of base", r"log.*product|log.*quotient")),
    ("trig_identity",         _p(r"sin|cos|tan|sec|csc|cot", r"trig", r"pythagorean identity",
                                  r"double angle", r"half angle")),
    ("derivative_rules",      _p(r"derivative", r"differentiat", r"chain rule",
                                  r"product rule.*deriv", r"quotient rule.*deriv",
                                  r"d/dx|dy/dx")),
    # Conceptual
    ("variable_concept",      _p(r"variable", r"unknown", r"parameter", r"expression.*vs")),
    ("eq_vs_expression",      _p(r"equation.*expression", r"expression.*equation",
                                  r"equals sign", r"statement.*true")),
    ("function_concept",      _p(r"function", r"domain", r"range", r"f\(x\)",
                                  r"input.*output", r"mapping")),
    ("limit_concept",         _p(r"\blimit\b", r"lim\b", r"approach", r"tends to",
                                  r"infinity", r"indeterminate")),
    ("derivative_concept",    _p(r"rate of change", r"slope.*tangent", r"instantaneous",
                                  r"concavity", r"increasing.*decreasing")),
    ("integral_concept",      _p(r"integral", r"area under", r"accumulation",
                                  r"anti.?deriv", r"fundamental theorem")),
    ("proportion_concept",    _p(r"proportion", r"ratio", r"scale", r"percent",
                                  r"cross.multipl")),
    ("probability_concept",   _p(r"probabilit", r"likelihood", r"sample space",
                                  r"event", r"random")),
    ("set_theory",            _p(r"\bset\b", r"union|intersection", r"element of",
                                  r"subset", r"complement")),
    ("inequality_concept",    _p(r"inequalit", r"greater than|less than", r"[<>]",
                                  r"at least|at most", r"between.*and")),
    ("coordinate_geometry",   _p(r"coordinate", r"x.axis|y.axis", r"slope.*line",
                                  r"y.intercept", r"distance formula", r"midpoint")),
    ("vector_concept",        _p(r"vector", r"magnitude", r"dot product", r"cross product",
                                  r"component", r"unit vector")),
    # Strategic
    ("choose_method",         _p(r"which method", r"approach to use", r"strategy",
                                  r"how to solve", r"best way")),
    ("problem_decompose",     _p(r"break.*down", r"sub.?problem", r"step.*by.*step",
                                  r"decompos", r"smaller.*parts")),
    ("substitution_strategy", _p(r"substitut", r"replace.*with", r"let\s+[a-z]\s*=",
                                  r"u.substitut")),
    ("elimination_strategy",  _p(r"eliminat", r"system.*equation", r"cancel.*out",
                                  r"add.*equation", r"subtract.*equation")),
    ("proof_strategy",        _p(r"proof|prove|theorem", r"contradict", r"induction",
                                  r"if.*then", r"given.*show")),
    ("estimation_strategy",   _p(r"estimat", r"approximat", r"round", r"about|roughly",
                                  r"order of magnitude")),
    ("backward_reasoning",    _p(r"work.*backward", r"from.*answer", r"reverse",
                                  r"goal.*start")),
    ("pattern_recognition",   _p(r"pattern", r"sequence", r"arithmetic.*series",
                                  r"geometric.*series", r"nth term")),
    ("check_solution",        _p(r"check|verify", r"plug.*in|substitute.*back",
                                  r"does it satisfy", r"valid.*answer")),
    ("special_cases",         _p(r"special case", r"edge case", r"exception",
                                  r"when.*zero", r"undefined")),
    # Representational
    ("latex_notation",        _p(r"\\frac|\\sqrt|\\sum|\\int", r"LaTeX", r"typeset",
                                  r"\\boxed|\\cdot")),
    ("graph_reading",         _p(r"graph", r"plot", r"curve", r"x.intercept",
                                  r"y.intercept", r"parabola")),
    ("table_interpretation",  _p(r"table", r"row|column", r"lookup", r"tabular")),
    ("diagram_geometry",      _p(r"diagram", r"figure", r"angle", r"triangle",
                                  r"circle", r"perpendicular|parallel")),
    ("number_line",           _p(r"number line", r"left.*right.*number",
                                  r"positive.*negative.*position")),
    ("symbolic_notation",     _p(r"notation", r"symbol", r"∑|∏|∫", r"Σ|Π",
                                  r"subscript|superscript")),
    ("unit_analysis",         _p(r"unit", r"conversion", r"dimensional", r"meter|gram|second",
                                  r"miles per hour|km/h")),
    ("scientific_notation",   _p(r"scientific notation", r"\d+\.\d+\s*[eE]",
                                  r"10\^", r"power of ten")),
]


def tag_error(
    student_response: str,
    correct_response: Optional[str] = None,
    problem_text: Optional[str] = None,
) -> str:
    """Classify the student's error into the most likely KC.

    Args:
        student_response:  The student's incorrect response text.
        correct_response:  The ground-truth / reference solution (if available).
        problem_text:      The problem statement (if available).

    Returns:
        KC identifier string (one of KC_IDS).
    """
    combined = " ".join(
        t for t in [student_response, correct_response, problem_text] if t
    )

    scores: dict[str, int] = {}
    for kc_id, patterns in _RULES:
        count = sum(1 for pat in patterns if pat.search(combined))
        if count > 0:
            scores[kc_id] = count

    if not scores:
        logger.debug("[KCTagger] no pattern matched, using fallback kc=%s", DEFAULT_KC)
        return DEFAULT_KC

    best_kc = max(scores, key=lambda k: scores[k])
    logger.debug("[KCTagger] kc=%s (score=%d) from %d candidates", best_kc, scores[best_kc], len(scores))
    return best_kc


def tag_all_errors(
    student_response: str,
    correct_response: Optional[str] = None,
    problem_text: Optional[str] = None,
    top_k: int = 3,
) -> list[tuple[str, int]]:
    """Return the top-k (kc_id, match_count) pairs for the given context.

    Useful for multi-label error analysis.
    """
    combined = " ".join(
        t for t in [student_response, correct_response, problem_text] if t
    )

    scores: list[tuple[str, int]] = []
    for kc_id, patterns in _RULES:
        count = sum(1 for pat in patterns if pat.search(combined))
        if count > 0:
            scores.append((kc_id, count))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def build_kc_hint_prefix(kc_id: str) -> str:
    """Return a structured KC tag string to prepend to an OPD hint.

    This enriches the teacher's re-generation context with the specific
    error category, guiding the distillation signal.
    """
    name = KC_NAMES.get(kc_id, kc_id)
    category = KC_CATEGORIES.get(kc_id, "Unknown")
    return f"[KC: {kc_id} | {category}: {name}]"
