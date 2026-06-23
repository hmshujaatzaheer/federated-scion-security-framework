# Zero-Knowledge Module (RQ2)

ZK-SNARK circuits for **privacy over** the 2026 bandwidth substrate — not a
replacement for it.

## What's here
- `circuits/` — Circom circuits
  - `ownership.circom` — prove ownership of a **fractional-fair-share** isolated
    asset (Wyss et al., NDSS 2026) without revealing key, value, or entitlement
  - `bandwidth_market.circom` — ownership + payment validity + Sybil resistance
- `coordination/signet_notification.py` — **Signet-style** verifiable,
  network-driven proof of notification (Ehsani Moghadam et al., ICDCS 2026), so
  the market settles cross-AS **without a trusted sequencer**

## Targets
- Proof generation: <10s · Verification: <1s · Proof size: 128–288 B (Groth16/BN254)
- ~1,000 R1CS constraints (ownership)

## Design stance
Privacy is *layered over* provable isolation (fair shares) and verifiable
coordination (Signet); its correctness is to be proven **jointly** with that
substrate, not in isolation. See [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md).
