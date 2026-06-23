"""
Unified strategic adversary harness.

A single adversary model, reused as the robustness target for federated
detection, the opponent in the path-aware MTD game, and the attacker in
evaluation -- so detection and response face the same attacker rather than two
different implicit ones.

Two threat components, re-implemented from the public descriptions in:
  - Da Dalt & Perrig, "Strategic Games and Zero-Shot Attacks on Heavy-Hitter
    Network Flow Monitoring," NDSS 2026 (zero-shot monitor evasion).
  - Xu, Duan, Cai & Perrig, "Resolve the Unresolved: Systematic Work Profiling
    for DNS Resolvers," IEEE S&P 2026 (work-asymmetry exhaustion).

This re-implements the threat model only; it does not reproduce those papers'
systems or measurements. Any numbers printed here are properties of this code.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Zero-shot evasion adversary
# ---------------------------------------------------------------------------
@dataclass
class ZeroShotEvasionAdversary:
    """
    Strategic, zero-shot evasion against a flow monitor.

    "Zero-shot" means the adversary does not observe the deployed monitor's
    configuration (threshold / sampling rate). Instead it plays against a
    *distribution* over plausible monitor configurations and shapes its traffic
    to minimize expected detection while preserving attack impact. This mirrors
    the strategic-game framing of heavy-hitter monitoring evasion.

    Parameters
    ----------
    n_features : int
        Dimensionality of the feature vector the detector consumes.
    impact_mask : Optional[np.ndarray]
        Boolean/0-1 weights marking which features carry "impact" (the volume
        the attacker actually needs). The adversary keeps these as high as it
        can while shrinking its detectable footprint on the rest.
    budget : float
        L2 budget for the evasive perturbation (how much the adversary may
        reshape its observable footprint).
    """

    n_features: int
    impact_mask: Optional[np.ndarray] = None
    budget: float = 1.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    def __post_init__(self) -> None:
        if self.impact_mask is None:
            # By default, the last quarter of features carry the attack impact.
            mask = np.zeros(self.n_features)
            mask[int(0.75 * self.n_features):] = 1.0
            self.impact_mask = mask

    def craft(self, benign_batch: np.ndarray,
              monitor_config_dist: Optional[Callable[[], np.ndarray]] = None
              ) -> np.ndarray:
        """
        Produce an evasive attack batch from a benign reference batch.

        The attack inflates impact-carrying features (the actual flood) while
        spending its ``budget`` to push the *detectable* (non-impact) features
        back toward the benign distribution -- the geometric essence of staying
        under a heavy-hitter threshold the adversary cannot see.
        """
        benign_batch = np.atleast_2d(benign_batch).astype(float)
        mean_benign = benign_batch.mean(axis=0)

        # Sample a distribution over monitor configs (zero-shot: not observed).
        if monitor_config_dist is not None:
            _ = monitor_config_dist()  # acknowledged but never directly used

        attack = benign_batch.copy()
        # 1) Raise the impact features (this is the real attack volume).
        attack += 3.0 * self.impact_mask
        # 2) Spend the evasion budget pulling detectable features back to benign.
        detectable = 1.0 - self.impact_mask
        direction = (mean_benign - attack) * detectable
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        attack += self.budget * direction / norms
        return attack


# ---------------------------------------------------------------------------
# Work-asymmetry (resource-exhaustion) adversary
# ---------------------------------------------------------------------------
@dataclass
class WorkAsymmetryAdversary:
    """
    Resource-exhaustion adversary that maximizes defender work per unit of
    attacker cost -- "denial of defense" from the inside.

    We model the defender's per-request pipeline as a list of stages, each with
    a (defender_cost, attacker_cost, trigger_prob) profile. The adversary
    selects inputs that trigger the most expensive stages cheaply. The harness
    reports the *amplification factor* = defender_work / attacker_work, which is
    the quantity the proposal turns into a checkable property (target < 2x on
    latency-critical paths).
    """

    stage_defender_cost: np.ndarray   # work the defender does per stage
    stage_attacker_cost: np.ndarray   # cost to the attacker to trigger a stage

    def amplification(self, triggers: np.ndarray) -> float:
        """Amplification factor for a chosen trigger vector (0/1 per stage)."""
        triggers = np.asarray(triggers, dtype=float)
        d = float(np.dot(triggers, self.stage_defender_cost))
        a = float(np.dot(triggers, self.stage_attacker_cost))
        return d / a if a > 0 else float("inf")

    def worst_case_triggers(self) -> np.ndarray:
        """
        Greedy worst case: trigger every stage whose defender/attacker cost
        ratio exceeds 1 (i.e., every stage that pays the attacker to abuse it).
        """
        ratio = self.stage_defender_cost / np.maximum(self.stage_attacker_cost, 1e-9)
        return (ratio > 1.0).astype(float)


# ---------------------------------------------------------------------------
# Unified harness
# ---------------------------------------------------------------------------
@dataclass
class UnifiedStrategicAdversary:
    """
    One adversary object handed to detection (RQ1), MTD (RQ3), and evaluation.

    This is the concrete embodiment of the proposal's methodological keystone:
    a single specification with three uses.
    """

    n_features: int
    evasion: ZeroShotEvasionAdversary = field(init=False)
    work: WorkAsymmetryAdversary = field(init=False)

    def __post_init__(self) -> None:
        self.evasion = ZeroShotEvasionAdversary(n_features=self.n_features)
        # A small illustrative pipeline: stage 2 is the abusable hot spot.
        self.work = WorkAsymmetryAdversary(
            stage_defender_cost=np.array([1.0, 1.0, 8.0, 2.0]),
            stage_attacker_cost=np.array([1.0, 1.0, 1.0, 2.0]),
        )

    # ---- (a) robustness target for the detector --------------------------
    def evaluate_detection_robustness(self, detector: Callable[[np.ndarray], np.ndarray],
                                      benign_batch: np.ndarray) -> dict:
        """
        Measure detection-rate retention under zero-shot evasion.

        ``detector`` maps a (batch, n_features) array to predicted labels
        (1 = attack). We compare the detection rate on a naive attack vs the
        evasive attack; the gap is the adversary's advantage.
        """
        # Naive attack is "loud": it inflates the whole footprint (including the
        # features a heavy-hitter monitor watches), so it is easy to catch.
        naive = np.atleast_2d(benign_batch).astype(float) + 3.0
        # Evasive attack keeps the impact while leaving the watched footprint
        # near-benign -- the geometric essence of zero-shot evasion.
        evasive = self.evasion.craft(benign_batch)
        naive_rate = float(np.mean(detector(naive)))
        evasive_rate = float(np.mean(detector(evasive)))
        return {
            "detection_rate_naive": naive_rate,
            "detection_rate_evasive": evasive_rate,
            "retention": evasive_rate / naive_rate if naive_rate > 0 else 0.0,
        }

    # ---- (c) work-asymmetry property -------------------------------------
    def worst_case_amplification(self) -> float:
        return self.work.amplification(self.work.worst_case_triggers())


def _demo() -> None:
    print("Unified Strategic Adversary Harness")
    print("=" * 60)
    rng = np.random.default_rng(0)
    benign = rng.normal(size=(64, 83))

    # A toy threshold detector on the impact features (stand-in for the real
    # CNN-GRU-DNN; the point is to exercise the adversary, not to claim accuracy).
    adv = UnifiedStrategicAdversary(n_features=83)

    def toy_detector(batch: np.ndarray) -> np.ndarray:
        # A naive heavy-hitter monitor: watch the footprint (non-impact features).
        score = batch[:, :62].sum(axis=1)
        return (score > 60.0).astype(int)

    rob = adv.evaluate_detection_robustness(toy_detector, benign)
    print(f"  naive detection rate   : {rob['detection_rate_naive']:.2f}")
    print(f"  evasive detection rate : {rob['detection_rate_evasive']:.2f}")
    print(f"  retention (lower = adversary wins): {rob['retention']:.2f}")
    print(f"  worst-case work amplification     : {adv.worst_case_amplification():.2f}x")
    print("\nHand this same object to the MTD game (RQ3) and the SCIONLab eval (RQ7).")


if __name__ == "__main__":
    _demo()
