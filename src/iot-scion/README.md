# IoT-SCION Integration Module (RQ5)

Lightweight SCION protocols for resource-constrained critical infrastructure
(128–512 KB RAM devices).

## Direction
Gateway-mediated bandwidth reservation is built on **fractional-fair-share**
isolation (Wyss et al., NDSS 2026), and must hold end-to-end QoS tight enough
for **cyber-physical control** — the regime where SCION now carries grid fast
frequency response (Zhang et al., 2026) and where digital assets meet the
physical world (GECKO, Krähenbühl et al., ICNC 2026).

## Targets
- Crypto overhead: 60% reduction (aggregate MACs, batch verification)
- Energy: 40% reduction vs. standard SCION
- Latency: suitable for real-time control (sub-budget on the cyber-physical path)
- Formally verified security equivalence to full SCION under a bounded adversary

See [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md).
