"""
Unit tests for the federated-learning framework and the 2026 upgrade modules.

Source directories use hyphens (federated-learning, control-loop, ...), which are
not importable as packages, so we load modules directly from their file paths.
"""

import os
import importlib.util

BASE = os.path.join(os.path.dirname(__file__), "..")


def _load(relpath, name):
    import sys
    path = os.path.join(BASE, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses (with `from __future__ import
    # annotations`) can resolve the module namespace.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_repository_structure():
    dirs_to_check = [
        "src/federated-learning",
        "src/federated-learning/adversary",
        "src/control-loop",
        "src/formal-verification",
        "src/zero-knowledge",
        "src/zero-knowledge/coordination",
        "experiments/frequency_response_demo",
        "data",
        "docs",
    ]
    for dir_path in dirs_to_check:
        assert os.path.exists(os.path.join(BASE, dir_path)), f"missing {dir_path}"


def test_citations_doc_exists():
    assert os.path.exists(os.path.join(BASE, "docs", "CITATIONS.md"))


# ---------------------------------------------------------------------------
# Strategic adversary (RQ0.1)
# ---------------------------------------------------------------------------
def test_strategic_adversary_evasion_lowers_detection():
    import numpy as np
    mod = _load("src/federated-learning/adversary/strategic_adversary.py", "strategic_adversary")
    rng = np.random.default_rng(0)
    benign = rng.normal(size=(64, 83))
    adv = mod.UnifiedStrategicAdversary(n_features=83)

    def detector(batch):
        return (batch[:, 62:].sum(axis=1) > 30.0).astype(int)

    rob = adv.evaluate_detection_robustness(detector, benign)
    # Evasion should not make detection *easier*.
    assert rob["detection_rate_evasive"] <= rob["detection_rate_naive"] + 1e-9


def test_work_amplification_is_reported():
    mod = _load("src/federated-learning/adversary/strategic_adversary.py", "strategic_adversary")
    adv = mod.UnifiedStrategicAdversary(n_features=83)
    amp = adv.worst_case_amplification()
    assert amp >= 1.0  # by construction the defender never does *less* work


# ---------------------------------------------------------------------------
# Closed-loop controller (RQ0)
# ---------------------------------------------------------------------------
def test_closed_loop_runs_and_reports_budget():
    import numpy as np
    mod = _load("src/control-loop/closed_loop.py", "closed_loop")
    rng = np.random.default_rng(1)

    loop = mod.ClosedLoopController(
        observe=lambda: rng.normal(size=(16, 83)),
        detector=lambda b: (b[:, 62:].sum(axis=1) > 30.0).astype(int),
        decide=lambda lvl: int(lvl > 0.3),
        reconfigure=lambda p: None,
        adversary=None,
    )
    loop.run(iterations=6, attack_from=3)
    rep = loop.report()
    assert rep["iterations"] == 6
    assert 0.0 <= rep["budget_met_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Signet-style verifiable notification (RQ2.1)
# ---------------------------------------------------------------------------
def test_signet_chain_integrity():
    mod = _load("src/zero-knowledge/coordination/signet_notification.py", "signet_notification")
    log = mod.SignetLog()
    for i in range(3):
        log.notify(f"commit-{i}")
    assert log.verify_chain() is True
    assert log.proof_of_notification(1)["epoch"] == 1
    assert log.proof_of_notification(99) is None


# ---------------------------------------------------------------------------
# Fractional-fair-share feature (RQ1.1)
# ---------------------------------------------------------------------------
def test_fair_share_deviation_feature():
    mod = _load("src/federated-learning/models/scion_features.py", "scion_features")
    ex = mod.SCIONFeatureExtractor()
    pkt = mod.SCIONPacket(
        src_as=1, dst_as=10, path_id=1, hop_count=5,
        segment_types=["up", "core", "down"], timestamp=1.0, packet_size=1200,
        fair_share_entitlement=10.0, realized_service=4.0,
    )
    feats = ex.extract_features(pkt)
    # (10 - 4)/10 = 0.6 starvation signal at index 15
    assert abs(feats[15] - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# MTD loop stability (RQ3.2)
# ---------------------------------------------------------------------------
def test_mtd_loop_stability_reported():
    mod = _load("src/moving-target-defense/game_theory/mtd_game.py", "mtd_game")
    game = mod.MTDGameTheory(num_paths=5)
    s = game.assess_loop_stability(iterations=300, tail=50)
    assert "tail_movement" in s and "stable" in s


def test_basic():
    assert 1 + 1 == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("All tests passed!")
