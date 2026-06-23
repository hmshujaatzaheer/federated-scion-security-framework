"""
Cyber-Physical Frequency-Response Demo (simulation)
===================================================

Why this exists
---------------
SCION now carries safety-critical cyber-physical control: Zhang, Kottmann,
Peng, Perrig & Hug, "Fast Frequency Response with Heterogeneous Communication
Delay Management under the SCION Internet Architecture," arXiv:2601.06879, 2026
(see docs/CITATIONS.md). In that regime the admissible end-to-end delay is set
by *grid stability*, not user experience -- so a security mechanism that adds
unbounded or oscillating latency is unsafe, not merely slow.

This demo makes the point concrete and testable. It runs a minimal swing-style
grid-frequency model under a primary frequency-response controller whose command
arrives after a communication delay. We sweep the delay to show there is a hard
budget beyond which frequency deviates past a safety band -- which is exactly the
budget our closed-loop defense (src/control-loop) must respect even *under attack*.

Note: this is an illustrative simulation; the numbers are properties of this
script, not measurements of a real grid or of the cited paper.
"""

from __future__ import annotations

import numpy as np


def simulate_frequency(delay_ms: float,
                       horizon_s: float = 12.0,
                       dt_s: float = 0.002,
                       H: float = 4.0,        # inertia constant (s)
                       D: float = 0.8,        # damping (pu/Hz, normalized)
                       Kp: float = 2.2,       # primary control gain (pu/Hz)
                       load_step_pu: float = 0.05,  # 5% load step at t=1s
                       settle_band_hz: float = 0.3,
                       f0_hz: float = 50.0) -> dict:
    """
    Single-area swing model with delayed primary frequency control:
        2H/f0 * df/dt = -(D/f0)(f - f0) - dP_load + Kp * (f0 - f(t - delay))
    The control reacts to a *delayed* frequency measurement. A strong, fast
    controller is well damped at small delay but is driven into growing
    oscillation as the delay increases -- the classic delay-induced instability
    that turns end-to-end latency into a hard safety budget.
    """
    n = int(horizon_s / dt_s)
    delay_steps = int(round((delay_ms / 1000.0) / dt_s))
    f = np.full(n, f0_hz)
    blackout_hz = 5.0   # beyond a few Hz the grid trips; stop and call it unsafe
    diverged = False

    for k in range(1, n):
        load = load_step_pu if (k * dt_s) >= 1.0 else 0.0
        kd = max(0, k - 1 - delay_steps)          # delayed measurement
        control = Kp * (f0_hz - f[kd])
        dfdt = (f0_hz / (2.0 * H)) * (
            -(D / f0_hz) * (f[k - 1] - f0_hz) - load + control
        )
        f[k] = f[k - 1] + dfdt * dt_s
        if abs(f[k] - f0_hz) > blackout_hz:
            f = f[: k + 1]
            diverged = True
            break

    nadir = float(np.min(f))
    max_dev = min(float(np.max(np.abs(f - f0_hz))), blackout_hz)
    safe = bool((not diverged) and max_dev <= settle_band_hz + 1e-9)
    return {
        "delay_ms": delay_ms,
        "frequency_nadir_hz": nadir if not diverged else f0_hz - blackout_hz,
        "max_deviation_hz": max_dev,
        "within_safety_band": safe,
        "diverged": diverged,
    }


def sweep(delays_ms=(5, 20, 50, 100, 200, 400)) -> None:
    print("Cyber-Physical Frequency-Response under Communication Delay (sim)")
    print("=" * 64)
    print(f"{'delay (ms)':>10} | {'nadir (Hz)':>11} | {'max dev (Hz)':>12} | safe?")
    print("-" * 64)
    budget = None
    for d in delays_ms:
        r = simulate_frequency(float(d))
        flag = "yes" if r["within_safety_band"] else "NO"
        if r["within_safety_band"]:
            budget = d
        print(f"{d:>10} | {r['frequency_nadir_hz']:>11.3f} | "
              f"{r['max_deviation_hz']:>12.3f} | {flag}")
    print("-" * 64)
    print(f"Largest safe delay in this sweep: ~{budget} ms")
    print("=> The closed-loop defense must keep its observe->...->reconfigure")
    print("   latency under this budget EVEN under the strategic adversary.")


if __name__ == "__main__":
    sweep()
