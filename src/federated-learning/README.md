# Federated Learning Module (RQ1)

Byzantine-robust federated learning for SCION DDoD detection — and the
**`Learn`** stage of the [closed loop](../control-loop/README.md).

## What's here
- `protocols/` — FedAvg, Krum / Multi-Krum (Byzantine robustness)
- `privacy/` — `(ε=1.0, δ=10⁻⁵)`-differential privacy
- `models/` — CNN-GRU-DNN, plus SCION feature extraction with a
  **fractional-fair-share isolation deviation** signal (Wyss et al., NDSS 2026)
- `adversary/` — **unified strategic adversary harness** (RQ0.1): zero-shot
  evasion (Da Dalt & Perrig, NDSS 2026) + work-asymmetry exhaustion
  (Xu et al., S&P 2026), reused by detection *and* MTD *and* evaluation

## Targets
- Accuracy: 99%+
- Latency: <60s
- Privacy: `(ε=1.0, δ=10⁻⁵)`-DP
- **Detection retention under zero-shot evasion: high** (new, RQ0.1)
- **Defender-work amplification under attack: < 2×** (new, RQ0.1)

See [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md) for references.
