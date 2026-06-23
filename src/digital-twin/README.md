# Digital Twin Module (RQ4)

Federated digital twin for SCION networks — the predictive feed into the
**`Decide`** stage of the [closed loop](../control-loop/README.md).

## What's here
- `synchronization/twin_sync.py`
  - vector-clock causal-consistency synchronization across AS-local twins
  - federated anomaly detection and bandwidth-exhaustion forecasting
  - **`meets_control_deadline`**: frames synchronization latency as a **hard
    safety budget** for cyber-physical use (grid frequency response over SCION,
    Zhang et al. 2026), not a best-effort target

## Targets
- Forecast horizon: 5–10 min · Sync lag: <5 s (best-effort)
- **Cyber-physical deadline:** within the stability budget (new, RQ4.4)

See [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md).
