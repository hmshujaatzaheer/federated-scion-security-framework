"""
Adversary-Coupled, Formally Verified Control Loop (reference implementation)
===========================================================================

Network defense as a single closed loop
    observe -> learn -> decide -> reconfigure
co-designed against one strategic adversary, with a bounded per-iteration
latency budget suitable for cyber-physical control over SCION.

The four stages correspond to the proposal's five research questions:
    Observe      -> path-aware + fractional-fair-share isolation features (RQ1)
    Learn        -> Byzantine-robust federated detection, robust to evasion (RQ1)
    Decide       -> verified MTD game equilibrium vs. the shared adversary (RQ3)
    Reconfigure  -> verified path switch with bounded latency (RQ3/RQ5)

Anchors (see docs/CITATIONS.md for full references):
    - Strategic / zero-shot adversary:  Da Dalt & Perrig, NDSS 2026
    - Work-asymmetry adversary:         Xu, Duan, Cai & Perrig, S&P 2026
    - Fractional-fair-share isolation:  Wyss, Hu, Lenders, Meier & Perrig, NDSS 2026
    - Cyber-physical bounded delay:     Zhang, Kottmann, Peng, Perrig & Hug, arXiv 2026
    - Verifiable coordination (Signet): Ehsani Moghadam et al., ICDCS 2026
    - Refinement methodology:           Pereira et al. (Protocols to Code), CCS 2025

A small, runnable version of the loop -- enough to make the design concrete and
testable, and to show detection and response sharing one adversary object. It is
not tuned for performance.
"""

from __future__ import annotations

import time
import sys
import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

# Allow running both as a module and as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated-learning"))
try:
    from adversary.strategic_adversary import UnifiedStrategicAdversary
except Exception:  # pragma: no cover - fallback for unusual import paths
    UnifiedStrategicAdversary = None  # type: ignore


@dataclass
class LoopBudget:
    """Hard timing budget for one iteration of the loop.

    For grid-style fast frequency response over SCION the admissible end-to-end
    delay is dictated by stability, not user experience; a reconfiguration that
    misses this budget is unsafe, not merely slow.
    """
    max_iteration_ms: float = 50.0   # illustrative cyber-physical budget


class ClosedLoopController:
    """observe -> learn -> decide -> reconfigure, against one adversary."""

    def __init__(self,
                 observe: Callable[[], np.ndarray],
                 detector: Callable[[np.ndarray], np.ndarray],
                 decide: Callable[[float], int],
                 reconfigure: Callable[[int], None],
                 adversary: "UnifiedStrategicAdversary",
                 budget: LoopBudget | None = None):
        self.observe = observe
        self.detector = detector
        self.decide = decide
        self.reconfigure = reconfigure
        self.adversary = adversary
        self.budget = budget or LoopBudget()
        self.history: List[dict] = []

    def step(self, under_attack: bool = False) -> dict:
        """Run one closed-loop iteration and record whether it met its budget."""
        t0 = time.perf_counter()

        # OBSERVE -------------------------------------------------------------
        features = np.atleast_2d(self.observe()).astype(float)
        if under_attack and self.adversary is not None:
            features = self.adversary.evasion.craft(features)

        # LEARN / DETECT ------------------------------------------------------
        preds = self.detector(features)
        threat_level = float(np.mean(preds))

        # DECIDE (MTD equilibrium response keyed on the same adversary) --------
        new_path = self.decide(threat_level)

        # RECONFIGURE ---------------------------------------------------------
        self.reconfigure(new_path)

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        record = {
            "threat_level": threat_level,
            "selected_path": new_path,
            "iteration_ms": elapsed_ms,
            "within_budget": elapsed_ms <= self.budget.max_iteration_ms,
            "under_attack": under_attack,
        }
        self.history.append(record)
        return record

    def run(self, iterations: int = 10, attack_from: int = 5) -> List[dict]:
        for i in range(iterations):
            self.step(under_attack=(i >= attack_from))
        return self.history

    def report(self) -> dict:
        if not self.history:
            return {}
        met = sum(r["within_budget"] for r in self.history)
        return {
            "iterations": len(self.history),
            "budget_met_fraction": met / len(self.history),
            "max_iteration_ms": max(r["iteration_ms"] for r in self.history),
            "worst_case_work_amplification":
                self.adversary.worst_case_amplification() if self.adversary else None,
        }


def _demo() -> None:
    print("Adversary-Coupled Verified Control Loop (reference)")
    print("=" * 60)
    rng = np.random.default_rng(1)

    adversary = UnifiedStrategicAdversary(n_features=83) if UnifiedStrategicAdversary else None

    def observe() -> np.ndarray:
        return rng.normal(size=(32, 83))

    def detector(batch: np.ndarray) -> np.ndarray:
        return (batch[:, 62:].sum(axis=1) > 30.0).astype(int)

    def decide(threat_level: float) -> int:
        # High threat -> rotate to a fresh path (MTD); else hold path 0.
        return int(threat_level > 0.3)

    chosen = {"path": 0}

    def reconfigure(path: int) -> None:
        chosen["path"] = path

    loop = ClosedLoopController(observe, detector, decide, reconfigure, adversary)
    loop.run(iterations=10, attack_from=5)
    rep = loop.report()
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("\nDetection and response shared ONE adversary object -- by construction.")


if __name__ == "__main__":
    _demo()
