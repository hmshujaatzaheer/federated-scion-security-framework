# Moving Target Defense Module (RQ3)

Path-aware MTD with game-theoretic analysis — the **`Decide`/`Reconfigure`**
stages of the [closed loop](../control-loop/README.md).

The defender plays against the **same** unified strategic adversary used to
attack detection (RQ0.1), not a fresh, weaker model — so an attacker who evades
the monitor cannot also freely steer the response.

## What's here
- `game_theory/mtd_game.py`
  - two-player path-switching game + Nash equilibrium via fictitious play
  - **`assess_loop_stability`**: a control-theoretic check that the
    reconfiguration loop *settles* rather than oscillates — essential on a
    bounded-latency cyber-physical path (after Scherrer, Perrig & Schmid,
    Perf. Eval. 2026; Zhang et al., 2026)

## Why stability, not just equilibrium
A Nash point can exist while the iterate keeps churning. On a grid-style path a
churning reconfiguration is a *safety* failure. We therefore require the
time-averaged strategy to converge (tail movement → 0).

See [`../../docs/CITATIONS.md`](../../docs/CITATIONS.md).
