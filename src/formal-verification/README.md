# Formal Verification Module (RQ0.2, RQ1.2, RQ1.3)

Isabelle/HOL and Gobra verification, following the **Protocols-to-Code**
refinement methodology (Pereira et al., CCS 2025).

## Theories
- `isabelle/adversary/StrategicAdversary.thy` — the **unified threat model**:
  zero-shot evasion + work-asymmetry, with `robust_against` / `work_safe`
- `isabelle/control_loop/ControlLoop.thy` — the **end-to-end obligation**: if the
  detector is robust to adversary `A`, the MTD policy is an equilibrium against
  the *same* `A`, and each step meets the latency budget, then the loop preserves
  the physical safety invariant (`loop_preserves_safety`)
- `isabelle/federated_protocols/FedAvg.thy` — Byzantine-robust aggregation whose
  deployed detector is robust to `A` (the adversary-coupling, stated together)
- `isabelle/federated_protocols/FedAvg_Convergence.thy` — convergence + DP lemmas

## Progress
- [x] Threat model and loop correctness **stated** (obligations as `sorry`)
- [ ] Discharge `loop_preserves_safety` and `fedavg_byzantine_and_strategic_robust`
- [ ] Isabelle formalization (target 8,000+ LoC)
- [ ] Gobra verification (target 5,000+ LoC), refining the above

> The contribution at this stage is the **precise statement** of what the loop
> must guarantee; the proofs are Phase-1 work. See
> [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md).
