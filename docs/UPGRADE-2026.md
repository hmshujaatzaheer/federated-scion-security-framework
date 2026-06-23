# June 2026 Revision — What Changed and Why

The original framework (Jan 2026) proposed five research questions integrating
formal verification, federated learning, ZK privacy, path-aware MTD, a digital
twin, and lightweight IoT-SCION. This revision keeps all five but **binds them
into one object**: an *adversary-coupled, formally verified control loop*.

## Why revise now

The SCION group's 2026 results make a single thread visible (full references in
[`CITATIONS.md`](CITATIONS.md)):

- **Monitoring is now an explicit target.** Da Dalt & Perrig (NDSS 2026) show
  strategic, *zero-shot* evasion of heavy-hitter flow monitoring; Xu, Duan, Cai
  & Perrig (S&P 2026) profile *work-asymmetry* exhaustion of core infrastructure.
- **The data plane is lightweight, isolation-bearing, and verified.** Wyss et al.
  (NDSS 2026) deliver fractional-fair-share allocation with provable isolation;
  Pereira et al. (CCS 2025) verify routers by refinement; Ehsani Moghadam et al.
  (ICDCS 2026, *Signet*) give verifiable network-driven coordination.
- **SCION now carries cyber-physical control.** Zhang, Kottmann, Peng, Perrig &
  Hug (2026) run grid fast frequency response over SCION under bounded delay.

**The gap:** detection and response are designed against *different, implicit*
adversaries, so an attacker optimal against the monitor can steer the defense.
On a cyber-physical path a late/oscillating reconfiguration is unsafe, not slow.

## The organizing idea

Defense becomes a closed loop — `observe → learn → decide → reconfigure` — with
three invariants:

1. **One adversary, three uses** (detection target = MTD opponent = eval attacker).
2. **Bounded iteration latency** keyed to physical stability.
3. **End-to-end refinement** from Isabelle/HOL to Gobra-verified code.

## Concrete changes in this repo

| File / dir | Change |
|------------|--------|
| `src/federated-learning/adversary/strategic_adversary.py` | **new** — unified zero-shot-evasion + work-asymmetry adversary (RQ0.1) |
| `src/control-loop/` | **new** — runnable closed-loop controller with a hard latency budget (RQ0) |
| `src/zero-knowledge/coordination/signet_notification.py` | **new** — Signet-style verifiable notification stub (RQ2.1) |
| `src/zero-knowledge/circuits/ownership.circom` | placeholder → real ownership circuit binding the fair-share entitlement |
| `src/federated-learning/models/scion_features.py` | added fractional-fair-share deviation feature (idx 15) |
| `src/moving-target-defense/game_theory/mtd_game.py` | adversary-coupled game + control-theoretic loop-stability check (RQ3.2) |
| `src/digital-twin/synchronization/twin_sync.py` | added `meets_control_deadline` for cyber-physical use (RQ4.4) |
| `src/formal-verification/isabelle/adversary/StrategicAdversary.thy` | **new** — formal threat model |
| `src/formal-verification/isabelle/control_loop/ControlLoop.thy` | **new** — end-to-end loop correctness obligation |
| `src/formal-verification/isabelle/federated_protocols/FedAvg.thy` | stub → adversary-coupled robustness statement |
| `experiments/frequency_response_demo/` | **new** — cyber-physical latency-budget simulation |
| `tests/test_federated.py` | tests for all new modules |
| `requirements.txt` | fixed stray markdown fences that broke `pip install` |
| `docs/CITATIONS.md` | **new** — the real 2026 references this builds on |

## Scope

The citations are to real, published or announced works; this repository does
not reproduce their systems or results. The new components are reference
implementations and small simulations, labeled in their source headers, and no
number here is presented as a measurement of an external system.
