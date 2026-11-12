# Controller details

Moved verbatim from AGENTS.md on 2026-11-12 (git `3d9bcf8`); the text is unchanged. Reformatted for readability the same day: the words are unchanged and long paragraphs became bullets (checks: `docs/agents/_reformat/check_*.py`). It was a sub-item of "Fallback" in AGENTS.md §3. (updated 2026-11-12: stale numbers have since been marked in place with dated notes; the reformat checks read this file at commit 38a1b46.)

- **Exit from the safety layer's fail-safe** (`NairACCConfig.failsafe_recovery`, set with `safelayer`, 2026-11-04, final-check item 7, 0419637).
  - The fail-safe leaves the acceleration below a_min (down to −8).
  - The MPC starts from the issued command as it is and plans the recovery along the fastest envelope at the fail-safe's jerk, e(t) = min(a_min, a_0 + 20 t):
    - the planned acceleration at step k is bounded below by e(k·dt),
    - the jerk bound is 20 where the envelope lies below a_min and 5 elsewhere,
    - and the issued command must reach e(command_dt) (beyond it only at the comfort jerk),
    - so the release is a ramp −8 → −6 → −4 → −3 at 0.1 s (−7, −6, … at 0.05 s) and the MPC's bounds take over at a_min (`failsafe_recovery_envelope`; per-step QP parameters, bit-identical outside a recovery).
    - (updated 2026-11-12: these numbers are for the fork's limits, a_min −3 and a fail-safe of −8 m/s² at jerk 20. At the NGSIM replay's default since 2026-11-12, `amin3.5` and `fsdecel8_fsjerk10`, the envelope is min(a_min, a_0 + 10 t), the jerk bound below a_min is 10, and the ramp reaches a_min −3.5 after 0.45 s; `docs/agents/tokens.md`, `fsdecel<x>`, `fsjerk<x>`.)
  - The command limiter opens the same window around the issued command (`failsafe_recovery_command`).
  - Below the speed from which the envelope cannot keep v ≥ 0 on the model grid (1.6 m/s at the design's limits; updated 2026-11-12: 3.8 m/s at the NGSIM replay's default limits since 2026-11-12, `tokens.md`, `fsdecel<x>`, `fsjerk<x>`) the MPC's start is raised to `failsafe_recovery_floor` (the old −√(2jv) clip, exact on the grid) while the limiter holds the command on the ramp.
  - Before, the MPC started from the previous command clipped to its bounds while the limiter clipped around the unclipped one, so the first command after a −8 fail-safe was −3 whatever the plan (50 m/s³ at 0.1 s, 100 at 0.05 s).
  - Never occurred on the final NGSIM set (the layer issues no fail-safe there)
    - on the lanefix root's ds6_v2080 (8 scored fail-safe ticks) the exit's command j_max 50 → 20 m/s³, min gap unchanged (`replay_out/failsafe_exit_20261104/result.md`).
  - The PC-ACC port has no such exit (its emergency profile lies inside its nominal bounds).
  - Bit-identical where the layer never acts (12 smoke + ds5_v2609, 3 lead-only).
  - `tests/test_failsafe_recovery.py` (10 tests).
  - Not run in CARLA.
