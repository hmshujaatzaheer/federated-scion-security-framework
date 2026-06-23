# Cyber-Physical Frequency-Response Demo

A minimal, clearly-labeled **simulation** that motivates the bounded-latency
requirement on the closed-loop defense (RQ0.3, RQ3.4, RQ4.4, RQ5.2).

## Context

SCION has been shown to carry **fast frequency response for power grids** under
explicitly managed communication delay (Zhang, Kottmann, Peng, Perrig & Hug,
arXiv:2601.06879, 2026 — see [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md)).
In that regime the admissible end-to-end delay is dictated by grid stability. A
security mechanism on that path inherits a **hard timing budget**: miss it and
the system is unsafe, not merely slow.

## What it shows

`freq_response_demo.py` runs a single-area swing-equation frequency model with a
primary controller that reacts to *delayed* measurements, and sweeps the delay:

```bash
python experiments/frequency_response_demo/freq_response_demo.py
```

You will see frequency stay within the safety band at small delays and breach it
as delay grows — i.e., there is a concrete latency budget the defense loop in
[`src/control-loop`](../../src/control-loop) must respect **even under the
strategic adversary**.

## Note on scope

The numbers are properties of this script, not measurements of a real grid or of
the cited paper. The demo is pedagogical: it pins down *why* the loop budget is a
safety property rather than a performance target.
