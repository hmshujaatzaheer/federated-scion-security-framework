# Foundational Citations

This repository is positioned directly against the 2026 results from the SCION
ecosystem (Network Security Group, ETH Zürich, and collaborators). Every claim
the code makes about *what it builds on* traces to one of the references below.
These are real, published/announced works; the repository does not reproduce
their systems or results, it builds new mechanisms over the directions they open.

## 2026 anchors

| Key | Reference | Role in this repo |
|-----|-----------|-------------------|
| `wyss2026fractional` | M. Wyss, Y.-C. Hu, V. Lenders, R. Meier, A. Perrig. **Lightweight Internet Bandwidth Allocation and Isolation with Fractional Fair Shares.** NDSS 2026. DOI: 10.14722/ndss.2026.240023 | Isolation substrate; fair-share deviation features (`scion_features.py`); bandwidth traded in the ZK market |
| `dadalt2026strategic` | F. Da Dalt, A. Perrig. **Strategic Games and Zero-Shot Attacks on Heavy-Hitter Network Flow Monitoring.** NDSS 2026. | Zero-shot evasion adversary (`strategic_adversary.py`); shared opponent for detection + MTD |
| `xu2026resolve` | L. Xu, H. Duan, Z. Cai, A. Perrig. **Resolve the Unresolved: Systematic Work Profiling for DNS Resolvers.** IEEE S&P 2026. | Work-asymmetry adversary; per-stage amplification metric |
| `moghadam2026signet` | E. Ehsani Moghadam, M. Wyss, J. Kwon, M. Frei, Y.-C. Hu, A. Perrig, A. Sonnino. **Signet: Scalable Network-Driven Proof of Notification for Blockchain Systems.** ICDCS 2026. | Verifiable cross-AS coordination for the bandwidth market (`signet_notification.py`) |
| `zhang2026frequency` | J. Zhang, F. Kottmann, J. C.-H. Peng, A. Perrig, G. Hug. **Fast Frequency Response with Heterogeneous Communication Delay Management under the SCION Internet Architecture.** arXiv:2601.06879, 2026. | Cyber-physical bounded-latency budget; frequency-response demo (`experiments/frequency_response_demo`) |
| `kraehenbuehl2026gecko` | C. Krähenbühl, N. Hauser, C. Gloor, J. A. García-Pardo, A. Perrig. **GECKO: Securing Digital Assets Through(out) the Physical World.** ICNC 2026, pp. 602–608. | Cyber-physical boundary / IoT asset security context |
| `scherrer2026control` | S. Scherrer, A. Perrig, S. Schmid. **A Control-Theoretic Perspective on BBR/CUBIC Congestion-Control Competition.** Performance Evaluation, vol. 171, art. 102529, 2026. | Control-loop stability lens for the MTD reconfiguration loop |

## Earlier foundations

| Key | Reference | Role |
|-----|-----------|------|
| `pereira2025protocols` | N. Pereira et al. **Protocols to Code: Formal Verification of a Secure Next-Generation Internet Router.** ACM CCS 2025. | Refinement methodology (Isabelle/HOL ⇒ Gobra) for the whole loop |
| `wuest2025hummingbird` | K. Wüst et al. **Hummingbird: Fast, Flexible, and Fair Inter-Domain Bandwidth Reservations.** ACM SIGCOMM 2025. | Bandwidth-market baseline extended with ZK privacy |

> If a result is not listed above, the code does not claim it. Simulated
> components are labeled as such in their source headers.
