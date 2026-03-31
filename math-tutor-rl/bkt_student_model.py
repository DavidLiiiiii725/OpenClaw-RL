"""Bayesian Knowledge Tracing (BKT) for per-KC mastery tracking.

Standard 4-parameter BKT model per Knowledge Component (KC):
  p_know0  — prior probability of mastery before any practice
  p_learn  — probability of learning (not-known → known) after one practice
  p_guess  — P(correct | not mastered)
  p_slip   — P(incorrect | mastered)

Bayesian update (Eq. 2 from problem statement):
  P(mastered | obs) = P(obs | mastered) * P(mastered) / P(obs)

Followed by a learning update:
  P(Ln+1) = P(Ln | obs) + (1 - P(Ln | obs)) * p_learn

Zone of Proximal Development (ZPD): 0.40 ≤ mastery ≤ 0.70
Problems should be sampled from KCs in this range.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

# Zone of Proximal Development bounds
ZPD_LO = 0.40
ZPD_HI = 0.70

_DENOMINATOR_EPSILON = 1e-12  # Minimum denominator magnitude for BKT Bayes update
_P_KNOW0 = 0.30   # low prior: most KCs start un-mastered
_P_LEARN = 0.10   # 10% chance of learning per practice trial
_P_GUESS = 0.20   # 20% chance of guessing correctly when not mastered
_P_SLIP = 0.10    # 10% chance of slipping when mastered


@dataclass
class KCState:
    """Per-KC Bayesian state."""
    p_mastery: float = _P_KNOW0
    p_learn: float = _P_LEARN
    p_guess: float = _P_GUESS
    p_slip: float = _P_SLIP
    attempts: int = 0
    correct: int = 0


class BKTStudentModel:
    """Per-student Bayesian Knowledge Tracing model.

    Maintains a separate KCState for every KC encountered.
    Thread-safety: not guaranteed — callers should serialize updates.
    """

    def __init__(self, kc_params: Dict[str, dict] | None = None):
        """
        Args:
            kc_params: Optional dict of {kc_name: {"p_know0": …, "p_learn": …, …}}
                       to override default hyper-parameters per KC.
        """
        self._kcs: Dict[str, KCState] = {}
        self._default_overrides: Dict[str, dict] = kc_params or {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, kc_name: str) -> KCState:
        if kc_name not in self._kcs:
            overrides = self._default_overrides.get(kc_name, {})
            self._kcs[kc_name] = KCState(
                p_mastery=overrides.get("p_know0", _P_KNOW0),
                p_learn=overrides.get("p_learn", _P_LEARN),
                p_guess=overrides.get("p_guess", _P_GUESS),
                p_slip=overrides.get("p_slip", _P_SLIP),
            )
        return self._kcs[kc_name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, kc_name: str, correct: bool) -> float:
        """Observe one practice trial and return the updated mastery probability.

        Implements the standard BKT update (Eq. 2 from paper):
          1. Bayes posterior  P(mastered | obs)
          2. Learning prior   P(Ln+1) = P(Ln|obs) + (1 - P(Ln|obs)) * p_learn

        Args:
            kc_name: Knowledge component identifier.
            correct: Whether the student answered correctly.

        Returns:
            Updated mastery probability in [0, 1].
        """
        kc = self._get_or_create(kc_name)
        p = kc.p_mastery

        # Likelihoods
        if correct:
            p_obs_m = 1.0 - kc.p_slip          # P(correct | mastered)
            p_obs_nm = kc.p_guess               # P(correct | not mastered)
        else:
            p_obs_m = kc.p_slip                 # P(incorrect | mastered)
            p_obs_nm = 1.0 - kc.p_guess         # P(incorrect | not mastered)

        # Bayes update
        numerator = p_obs_m * p
        denominator = numerator + p_obs_nm * (1.0 - p)
        p_posterior = numerator / denominator if denominator > _DENOMINATOR_EPSILON else p

        # Learning transition
        p_new = p_posterior + (1.0 - p_posterior) * kc.p_learn
        p_new = min(max(p_new, 0.0), 1.0)

        kc.p_mastery = p_new
        kc.attempts += 1
        if correct:
            kc.correct += 1

        logger.debug(
            "[BKT] kc=%s obs=%s prior=%.3f posterior=%.3f new=%.3f",
            kc_name, "correct" if correct else "wrong", p, p_posterior, p_new,
        )
        return p_new

    def get_mastery(self, kc_name: str) -> float:
        """Return current mastery probability for a KC (0 if never seen)."""
        return self._get_or_create(kc_name).p_mastery

    def is_in_zpd(self, kc_name: str) -> bool:
        """Return True if KC is in Zone of Proximal Development [0.40, 0.70]."""
        p = self.get_mastery(kc_name)
        return ZPD_LO <= p <= ZPD_HI

    def get_zpd_kcs(self) -> List[str]:
        """Return all tracked KCs currently in the ZPD, sorted by mastery."""
        return sorted(
            [k for k in self._kcs if self.is_in_zpd(k)],
            key=lambda k: self._kcs[k].p_mastery,
        )

    def all_kcs(self) -> Dict[str, float]:
        """Return {kc_name: mastery_prob} for every tracked KC."""
        return {k: v.p_mastery for k, v in self._kcs.items()}

    def accuracy(self, kc_name: str) -> float:
        """Return raw accuracy (correct / attempts) for a KC, or 0.0 if never seen."""
        kc = self._kcs.get(kc_name)
        if kc is None or kc.attempts == 0:
            return 0.0
        return kc.correct / kc.attempts

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "kcs": {k: asdict(v) for k, v in self._kcs.items()},
        }

    def add_kc_state(self, kc_name: str, state: "KCState") -> None:
        """Add or overwrite a KC state directly (used for deserialisation)."""
        self._kcs[kc_name] = state

    @classmethod
    def from_dict(cls, data: dict) -> "BKTStudentModel":
        model = cls()
        for kc_name, kc_data in data.get("kcs", {}).items():
            model.add_kc_state(kc_name, KCState(**kc_data))
        return model

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BKTStudentModel":
        return cls.from_dict(json.loads(text))
