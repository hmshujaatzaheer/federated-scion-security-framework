# Formally Verified Federated Learning Framework for Privacy-Preserving Anomaly Detection in Path-Aware Networks

[![Tests](https://github.com/hmshujaatzaheer/federated-scion-security-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/hmshujaatzaheer/federated-scion-security-framework/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> PhD research project — work in progress.
> Contact: shujabis@gmail.com

---

## Research Vision

This repository treats network defense as one closed control loop —
`observe → learn → decide → reconfigure` — co-designed and co-verified against a
single strategic adversary, rather than as a catalog of separate point defenses.
It integrates five domains as the five stages of that loop:

1. Formal verification (Isabelle/HOL, Gobra) — provable security guarantees, end to end
2. Federated machine learning (Byzantine-robust, privacy-preserving) — distributed detection
3. Zero-knowledge cryptography (ZK-SNARKs) — privacy over the bandwidth-market substrate
4. Path-aware networking (SCION) — multipath routing and isolation
5. Blockchain systems (Sui / Signet-style notification) — verifiable coordination

The motivating observation: detection and response are usually designed against
different, implicit adversaries. A strategic attacker who is good at evading a
flow monitor can therefore also steer the defense that monitor triggers. The aim
here is to fix one adversary for the whole loop and discharge a single
end-to-end correctness statement by refinement.

This June 2026 revision is positioned against the SCION group's 2026 results;
see [`docs/UPGRADE-2026.md`](docs/UPGRADE-2026.md) for what changed and
[`docs/CITATIONS.md`](docs/CITATIONS.md) for the references it builds on.

---

## What's New in the June 2026 Revision

| Area | Addition | Builds on |
|------|----------|-----------|
| Shared threat model | [`strategic_adversary.py`](src/federated-learning/adversary/strategic_adversary.py): one adversary (zero-shot evasion + work-asymmetry) reused by detection, MTD, and evaluation | Da Dalt & Perrig (NDSS'26); Xu et al. (S&P'26) |
| Closed-loop controller | [`src/control-loop/`](src/control-loop/): runnable `observe→learn→decide→reconfigure` loop with a latency budget | Pereira et al. (CCS'25); Zhang et al. (2026) |
| Isolation-aware features | fractional-fair-share deviation signal in [`scion_features.py`](src/federated-learning/models/scion_features.py) | Wyss et al. (NDSS'26) |
| Verifiable coordination | [`signet_notification.py`](src/zero-knowledge/coordination/signet_notification.py): settle the ZK market without a trusted sequencer | Ehsani Moghadam et al. (ICDCS'26) |
| Loop stability | control-theoretic stability check in [`mtd_game.py`](src/moving-target-defense/game_theory/mtd_game.py) | Scherrer, Perrig & Schmid (Perf. Eval.'26) |
| Cyber-physical demo | [`frequency_response_demo`](experiments/frequency_response_demo/): the latency budget as a safety property | Zhang et al. (SCION freq. response, 2026) |
| Formal specs | `StrategicAdversary.thy`, `ControlLoop.thy`, and an adversary-coupled statement in `FedAvg.thy` | Pereira et al. (CCS'25) |

---

## Research Questions

### RQ0: Closing the Detection–Response Loop (the unifying objective)
- RQ0.1: Can one strategic adversary (zero-shot evasion + work-asymmetry) serve as both the detector's robustness target and the MTD opponent?
- RQ0.2: What end-to-end correctness statement can be machine-checked for the `observe→learn→decide→reconfigure` loop via refinement?
- RQ0.3: Is the worst-case loop latency within the budget required for cyber-physical control over SCION?

### RQ1: Formally Verified Federated Learning for SCION
- RQ1.1: How can federated protocols exploit SCION path-aware properties for DDoD detection?
- RQ1.2: Can we formally verify federated aggregation with Byzantine robustness in Isabelle/HOL?
- RQ1.3: How to verify privacy-preserving implementations in Go using Gobra?
- RQ1.4: What are the performance trade-offs between verification completeness and overhead?

### RQ2: Zero-Knowledge Privacy for Bandwidth Markets
- RQ2.1: How to integrate ZK-SNARKs with Hummingbird smart contracts?
- RQ2.2: Can we achieve sub-10s proof generation for bandwidth reservations?
- RQ2.3: How to formally verify ZK circuit and smart contract correctness?
- RQ2.4: What are the privacy-utility trade-offs in bandwidth trading?

### RQ3: Moving Target Defense with Path-Aware Properties
- RQ3.1: How can SCION multipath routing enable dynamic traffic shifting MTD?
- RQ3.2: Can we create formal game-theoretic models verified in Isabelle/HOL?
- RQ3.3: How to enable federated MTD decision-making across ASes?
- RQ3.4: What are the performance bounds during MTD reconfiguration?

### RQ4: Federated Digital Twin for SCION Networks
- RQ4.1: How to design a distributed digital twin with formally verified synchronization?
- RQ4.2: What consistency models (eventual, causal, strong) can be proven in Isabelle/HOL?
- RQ4.3: How to enable federated anomaly detection across AS digital twins?
- RQ4.4: What predictive accuracy can we achieve for bandwidth exhaustion and attacks?

### RQ5: Lightweight SCION-IoT Integration
- RQ5.1: What cryptographic optimizations work for 128-512KB RAM devices?
- RQ5.2: How to implement gateway-mediated bandwidth reservation for IoT?
- RQ5.3: Can we achieve 40% energy efficiency improvements?
- RQ5.4: How to formally verify lightweight protocol security equivalence?

---

## Repository Structure
```
federated-scion-security-framework/
├── .github/workflows/        # CI
├── docs/                     # documentation
│   ├── CITATIONS.md          # 2026 SCION references this builds on
│   └── UPGRADE-2026.md       # what changed in the June 2026 revision
├── src/                      # source (RQ0 spine + 5 RQ areas)
│   ├── control-loop/         # RQ0   adversary-coupled verified loop (the spine)
│   ├── formal-verification/  # RQ1.2, RQ1.3  Isabelle/HOL & Gobra
│   ├── federated-learning/   # RQ1   federated ML + the strategic adversary harness
│   ├── zero-knowledge/       # RQ2   ZK-SNARKs, smart contracts, Signet-style coordination
│   ├── moving-target-defense/# RQ3   path-aware MTD + loop stability
│   ├── digital-twin/         # RQ4   federated digital twin
│   └── iot-scion/            # RQ5   lightweight IoT protocols
├── experiments/              # testbed setup & simulations
│   └── frequency_response_demo/
├── data/                     # datasets & benchmarks
├── tools/                    # scripts
├── tests/                    # tests
└── publications/             # outputs
```

---

## Status and Maturity

This is an early-stage research repository that accompanies the proposal. What is
here today is **specifications, architecture, and runnable demonstrations** — not
finished research results. Concretely:

- The Isabelle/HOL theories state the properties to be proven; the proofs
  themselves are left as obligations (`sorry`) to be discharged.
- The Python modules are reference implementations and small simulations; the ML
  models are defined but not yet trained on real datasets, and no SCIONLab
  measurements have been collected.
- The Circom circuits are written but not yet compiled / proven.

In short: the scaffolding and direction are in place; the proofs, trained models,
compiled circuits, and testbed evaluation are the doctoral work ahead.

---

## Technology Stack

- Formal verification: Isabelle/HOL, Gobra (via Viper), Z3
- Machine learning: PyTorch, TensorFlow Federated, scikit-learn
- Zero-knowledge: Circom, snarkjs, Groth16 / BN254
- Path-aware networking: SCION, SCIONLab, Docker
- Blockchain: Sui, Move

---

## Getting Started

```bash
git clone https://github.com/hmshujaatzaheer/federated-scion-security-framework.git
cd federated-scion-security-framework

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# run the demos
python src/federated-learning/adversary/strategic_adversary.py
python src/control-loop/closed_loop.py
python experiments/frequency_response_demo/freq_response_demo.py

# run the tests
python -m pytest tests/ -q
```

---

## Research Timeline (36 Months)

- Year 1 (Months 1–12): literature integration, Isabelle/HOL foundations,
  initial federated learning, SCION feature engineering, first paper submission.
- Year 2 (Months 13–24): ZK-SNARK circuits and Hummingbird extension, federated
  digital twin with verified synchronization, MTD game-theoretic analysis.
- Year 3 (Months 25–36): IoT-SCION optimization, SCIONLab evaluation, thesis.

---

## Target Venues

Formal verification (CAV/FM/ITP), networking (INFOCOM/CoNEXT/SIGCOMM/NSDI),
security (CCS/S&P/USENIX/NDSS), and machine learning (NeurIPS/ICML).

---

## Performance Targets

These are design targets for the research, not measured results.

| Component | Metric | Target | RQ |
|-----------|--------|--------|-----|
| Closed loop (spine) | Worst-case iteration latency | < cyber-physical budget | RQ0.3 |
| | Detection retention under zero-shot evasion | high | RQ0.1 |
| | Work amplification under attack | < 2× | RQ0.1 |
| Federated DDoD detection | Accuracy | 99%+ | RQ1.1 |
| | Detection latency | <60s | RQ1.1 |
| | False positive rate | <5% | RQ1.1 |
| ZK bandwidth markets | Proof generation | <10s | RQ2.2 |
| | Verification time | <1s | RQ2.2 |
| Moving target defense | Response improvement | 15–20% | RQ3.4 |
| Digital twin | Forecast horizon | 5–10 min | RQ4.4 |
| IoT-SCION | Energy reduction | 40% | RQ5.3 |

---

## Related Work

- [SCION Architecture](https://www.scion-architecture.net/)
- [Network Security Group, ETH Zürich — Publications](https://netsec.ethz.ch/publications/) (source of the 2026 references; see `docs/CITATIONS.md`)
- [Hummingbird](https://github.com/netsys-lab/hummingbird)
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [Circom](https://docs.circom.io/)
- [Isabelle/HOL](https://isabelle.in.tum.de/)

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

This is independent research. It builds on the publicly published work of the
Network Security Group at ETH Zürich (SCION, Hummingbird, the formally verified
router, and the 2026 results listed in `docs/CITATIONS.md`) and on the
open-source SCION, Isabelle/HOL, and Circom ecosystems.

---

Maintained by H M Shujaat Zaheer — contact: shujabis@gmail.com
