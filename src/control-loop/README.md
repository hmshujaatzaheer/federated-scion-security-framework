# Adversary-Coupled Verified Control Loop (RQ0)

This module is the **spine** of the framework. It treats network defense as a
single closed loop rather than a catalog of point mechanisms:

```
        ┌──────────────────────────────────────────────┐
        │   Refinement: Isabelle/HOL ⇒ Gobra            │
        │   (Protocols-to-Code style, Pereira CCS 2025) │
        │                                                │
        │   ┌─────────┐   evade    ┌──────────┐          │
        │   │ Observe │◄───────────│          │          │
        │   └────┬────┘            │ Unified  │          │
        │        ▼                 │ strategic│          │
        │   ┌─────────┐   evade    │ adversary│          │
        │   │  Learn  │◄───────────│    𝒜     │          │
        │   └────┬────┘            │          │          │
        │        ▼                 │ (Da Dalt │          │
        │   ┌─────────┐   steer    │  + Xu,   │          │
        │   │ Decide  │◄───────────│   2026)  │          │
        │   └────┬────┘            │          │          │
        │        ▼                 │          │          │
        │   ┌──────────────┐ delay │          │          │
        │   │ Reconfigure  │◄──────│          │          │
        │   └──────────────┘       └──────────┘          │
        └──────────────────────────────────────────────┘
```

## Why a loop?

The 2026 SCION literature shows that **detection and response now share one
fate**. A strategic adversary that is provably good at evading flow monitoring
(Da Dalt & Perrig, NDSS 2026) inherits, for free, the ability to *steer the
response that the monitor triggers* — unless the response was designed against
that same adversary. On a cyber-physical path (grid frequency response over
SCION, Zhang et al. 2026) a late or oscillating reconfiguration is a safety
violation, not a performance dip.

## How the five research questions map to the loop

The five RQs are not replaced by the loop — they *are* its parts:

| Research question | Role in the loop |
|---|---|
| RQ1 Federated learning | the **Observe → Learn** stage (the detector) |
| RQ3 Moving Target Defense | the **Decide → Reconfigure** stage (the response) |
| RQ4 Federated digital twin | predictive **sensing** feeding the loop |
| RQ2 ZK bandwidth markets | the private, verifiable **coordination substrate** beneath it |
| RQ5 Lightweight IoT-SCION | the constrained, latency-critical **environment** it runs in |

So this loop enforces three invariants:

1. **One adversary, three uses.** The same `UnifiedStrategicAdversary` object is
   the robustness target for the detector *and* the opponent in the MTD game
   *and* the attacker in evaluation. See
   [`strategic_adversary.py`](../federated-learning/adversary/strategic_adversary.py).
2. **Bounded iteration latency.** `LoopBudget.max_iteration_ms` is a hard budget
   keyed to cyber-physical stability, not best-effort targets.
3. **End-to-end refinement.** The loop’s correctness statement is intended to be
   discharged by refinement from Isabelle/HOL to Gobra-verified code, following
   the SCION router verification methodology.

## Run

```bash
python src/control-loop/closed_loop.py
```

## Note on scope

This is a *reference* loop: it makes the construction concrete and testable. It
does not claim production throughput or reproduce any external system’s
numbers. See [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md) for the works
this builds on.
