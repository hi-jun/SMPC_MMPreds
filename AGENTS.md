# AGENTS.md — guide for AI assistants working on this repository

Audience: AI coding assistants (Claude, GPT, …) and new contributors who read this repository on GitHub.
It describes the state on 2026-10-07, branch `feat/probability-gated-cutin`. Where this file and the code disagree, the code is right.

**The maintainer works in Korean. Answer in Korean unless asked otherwise.**

---

## 1. What this repository is

- **Origin.** This is a fork of [shn66/SMPC_MMPreds](https://github.com/shn66/SMPC_MMPreds) (Nair et al., stochastic MPC with multimodal predictions). The root `README.md` is the upstream README and describes the original 2-D planner, not this fork's work.
- **What the fork does.** It turns that planner into a **1-D longitudinal ACC (adaptive cruise control) controller**. The controller is evaluated in **CARLA 0.9.13 (Town04)** on cut-in, no-cut-in and cut-out scenarios.
- **Predictors.**
  - **STDAN**, a learned trajectory predictor with three intentions (lane keep, left lane change, right lane change)
  - single-mode **LSTM** baselines
  - **IAIMM-KF** (Zhou et al.), a model-based baseline
- **The contribution ("Proposed", policy token `cutin_chance0.6`)** is a confidence chance constraint. The code describes it as the 1-D counterpart of Benciolini et al. eq. (19a).
  - Inputs: each mode j has probability p_j, and β_ref = 0.6.
  - Definitions: q(β) = Φ⁻¹((1+β)/2) and α_j = q(β_j)/q(β_ref).
  - **Cut-in modes:** β_j = min(p_j, β_ref). The gap to the predicted vehicle must exceed α_j·d_safe(v) + q(β_j)·σ_j.
  - **Lane-keeping (`lk`) modes of the ego-lane lead:** the safe distance is scaled by α_j, with β_j = min(p_j, β_ref) and no σ term.
  - **`cutout` modes:** β_j = min(1 − p_j, β_ref). The scale is floored at 0.35, and there is no σ term.
- **Unlikely cut-in modes are switched off, not removed.** Below p_j = 0.1 (`CUTIN_CHANCE_VANISH_BELOW`; override with `cutin_gate<p>`), the mode's `active_mask` is cleared.
  - The mode then never acts as a lead, but it stays in the scenario set.
  - This gate applies to every STDAN and IAIMM-KF policy, not only to Proposed.
- **Paper files.** `overleaf/formulation_twosided_conditional.tex` is the MPC formulation as written for the paper; it differs from the code in four places (§3). The result tables are `overleaf/table_cutin_final.tex` and `overleaf/table_cutout_final.tex`.

## 2. Where things live

| Path | Role |
|---|---|
| `scripts/carla/utils/acc_nair_smpc.py` | The ACC controller: `NairACCConfig`, `original_nair_acc_config()`, `NairACCSMPC`, the cached QPs (`CachedOpenLoopFixedQP`, `CachedFeedbackScalarChanceQP`), `OldACCReferenceAdapter`, `MultimodalLeadPrediction` |
| `scripts/carla/policies/acc_nair_smpc_agent.py` | The CARLA ego agent. It parses the policy string (§4), builds the predictor and the controller, and writes `policy_log` |
| `scripts/carla/utils/mpc_utils.py` | Upstream 2-D Nair SMPC. Its planner logic is unchanged (the fork only added Gurobi version setup). It is still imported at run time, through the upstream agent `smpc_agent.py` |
| `scripts/carla/utils/low_level_control.py` | `IdealLongitudinalActuator`, used by the ego and by the scripted vehicles |
| `scripts/carla/policies/distance_triggered_lane_change_agent.py`, `distance_triggered_cutout_agent.py`, `fixed_lane_speed_agent.py` | Scripted traffic: the cut-in vehicle (TV), the departing lead, and lane keepers |
| `scripts/carla/policies/lead_speed_cap.py` | `LeadSpeedCapMixin`, which the cut-in and cut-out agents mix in |
| `scripts/carla/scenarios/run_lk_scenario.py` | The scenario runner: spawns actors, steps CARLA, records results |
| `scripts/carla/devel/sweep_acc_scenarios.py` | Grid sweeps. `--kind` selects the scenario family (the kinds are built in `acc_scenario_factory.py`) |
| `scripts/carla/devel/watch_acc_sweep.py` | Watchdog that restarts a sweep after a CARLA crash or stall |
| `scripts/carla/devel/aggregate_cutin_table.py` | Metrics and tables. **Its docstring is the definition of record for every metric** |
| `scripts/carla/devel/animate_acc_stdan_scenario.py` | Renders one run: scene, predicted intentions, mode probabilities, control panel |
| `predictor/stdan_3int_signed_tcross_velint/` | The STDAN used in the paper: model, training (`preprocessing/`), checkpoints. `acc_adapter.py` turns CARLA history into model tensors and Frenet modes. `acc_postprocess.py` maps intentions to ACC modes and applies the confidence scaling |
| `predictor/lstm/`, `predictor/lstm_vel/` | Single-mode LSTM baselines, with position and velocity output respectively |
| `predictor/iaimm_kf/` | IAIMM-KF port (policy token `iaimm`) |
| `predictor/stdan/` | The original 9-mode STDAN. The other adapters still import `FT2M` and `transform_points` from it |
| `predictor/stdan_vel/` | A 9-mode STDAN with velocity output |
| `overleaf/` | Paper tables, the final formulation, and the figure and table generators (`overleaf/figs/scripts/`) |
| `tests/` | 7 unit-test modules (111 tests): controller port, chance/cut-out extension, cut-out trajectory mapping, fixed-lane agent, ideal actuator, LSTM adapter, STDAN integration |
| `results/` | Sweep outputs. `results/*/` is gitignored, so sweep data and launch scripts exist only on the maintainer's machine |

## 3. The controller in one page

- **State and dynamics.** The state is x = [s, v] (Frenet longitudinal position and speed), and the input is u = a. The prediction model is a double integrator (`acc_dynamics_matrices`).
- **Timing.** The MPC step is dt = 0.2 s, and every sweep uses horizon N = 15 (`--ego-horizons 15`; `original_nair_acc_config()` defaults to 10). CARLA runs at 20 Hz, so a new command is issued every 0.05 s.
- **The paper's tex differs from the code in four places.** `overleaf/formulation_twosided_conditional.tex`:
  1. writes a three-state model x = [s, v, a] with a first-order acceleration lag τ_a; the code has no lag
  2. applies the q(β)σ term to every mode; the code applies it only to cut-in modes
  3. does not state the 0.35 floor on the cut-out scale
  4. does not state the 0.1 activation gate

Defaults come from `original_nair_acc_config()` plus the agent's settings:

| Parameter | Value | Note |
|---|---|---|
| speed bounds | 0 ≤ v ≤ 20 m/s | Upstream's 15 was below the scenarios' desired speeds. Because of it, every ego-16 run in the 2026-09-02 sweep collided |
| acceleration bounds | −3 ≤ a ≤ 2 m/s² | |
| jerk limit | 5 m/s³, hard, also applied to the issued command | Upstream's 1.5 was too slow for cut-ins with a TTC of 1.0–1.3 s. It was 10 until 2026-09-09 |
| weights | q_s 0, q_v 1, r_a 1, r_jerk 10, slack 5000 | `a_ref = 0`, so `r_a` penalises command size |
| tightening | 1.64 (ε ≈ 0.05) | Only `scalar_chance` and `optimized_eta` use it; the paper's policies do not |
| solver | Gurobi through CasADi `Opti('conic')` | The agent sets a 0.15 s time limit; the config's own default is no limit |

- **Multimodal structure.**
  - `acc_postprocess.py` turns each target vehicle into ACC modes:
    - an adjacent-lane vehicle gives `lk` and `cutin`
    - the ego-lane vehicle gives `lk`, plus `cutout` only when a predicted lane-change trajectory ends outside the ego lane
    - vehicles farther out give one mode
    - modes with zero probability are skipped
  - Joint modes (the product over targets) are capped by `modes<N>`.
  - Scenarios share their inputs while they have the same effective lead, meaning the same vehicle and the same mode, and the first input u₀ is always shared.
  - The cached QPs are keyed by scenario count and switch the sharing with 0/1 parameters, so a moving split point does not rebuild the problem.
  - The open-loop variant has no sharing tree.
- **Safety constraint** (per mode and step): `s_lead − s_ego − c·(L + d0 + τ·v) − q(β)·σ + slack ≥ 0`, with L = 4.5 m, d0 = 3 m and τ = 1.3 s.
  - Without a chance token (`nominal_safe_distance`), c = 1 and there is no σ term.
  - `cutin_chance` sets c = α_j and adds the σ term on cut-in modes (§1).
  - The constraint is soft: slack ≥ 0, with cost 5000·slack².
  - At a step where a mode has no lead in the ego lane (`active_mask = False`), the cached QPs keep the row but place the lead 10⁶ m ahead, so the row never binds.
- **Uncertainty.**
  - Unlike upstream, the TV's prediction noise is not propagated through closed-loop dynamics.
  - σ combines the predicted lead-position variance with the ego's process noise, so σ ≥ 0.164 m even when the predictor reports no variance.
  - σ enters the constraint above, and also the reference gap that shapes `s_ref` and `v_ref`.
  - With `with_k`, ±1σ lead samples also feed the K feedback.
- **Fallback.**
  - If the solve raises (e.g. Gurobi `INF_OR_UNBD`), `_fallback_policy` issues a jerk-limited rule-based command, and the step logs `solve_path == "fallback"`.
  - Other Gurobi statuses do not mark a fallback; they appear only in `solver_message`.
  - Without a working Gurobi, every step falls back.

## 4. The policy string is the configuration

Sweeps pass `--ego-policy <string>`. The `_parse_*` methods of `ACCNairSMPCAgent` (in `acc_nair_smpc_agent.py`, starting around line 1186) read tokens from it by substring or regex match.

| Token | Effect |
|---|---|
| `const_lead` | No predictor: constant-velocity lead ("SCC"), 1 mode |
| `lstm`, `lstm_vel` | Single-mode LSTM predictor, 1 mode |
| `stdan_3int` | 3-intention STDAN |
| `iaimm` (written `iaimm_kf` in policy names) | IAIMM-KF. Its tuning knobs come from `ACC_NAIR_IAIMM_CFG` |
| `modes<N>` | Cap on joint modes, e.g. `modes8` |
| `best_mode` | Keep only the most likely mode |
| `cutin_chance<β_ref>` | **Proposed** confidence chance constraint, e.g. `cutin_chance0.6` |
| `cutin_gate<p>` | Probability below which a cut-in mode stops acting as a lead (default 0.1 for STDAN and IAIMM-KF) |
| `rmove<w>`, `rjerk<w>` | Weight on the step-0 command change; jerk weight |
| `open_loop`, `multimodal_ol`, any `_ol` | Open-loop variant: one input sequence shared by all scenarios, and one scalar slack |
| `no_k`, `with_k` | Disable or enable K-feedback optimisation (off by default) |
| `scalar_chance`, `optimized_eta`, `probability_weighted` | Chance-constraint and risk-allocation variants not used in the paper |
| `brake_distance`, `brake_hard_band` | Braking-distance safety bound (experimental) |
| `cutin_ramp<β>`, `cutin_tlc<s>`, `gap_recover<s>`, `intent_lpf<s>`, `track_smooth<n>` | Other options, off by default and not used in the paper. Combining `cutin_ramp` with `cutin_chance` raises `ValueError` |
| `fixed_risk`, `nominal_safe_distance` | Labels only. `fixed_risk` sets the variant name and changes nothing in the solve. `nominal_safe_distance` is never parsed: the deterministic safety distance is simply what you get when no chance token is present. Every paper policy carries both |

Matching is by substring. The predictor is chosen in the order `const_lead` > `iaimm` > `lstm_vel` > `lstm` > `stdan_vel` > `stdan_3int` > synthetic, and any `_ol` in the string selects the open-loop variant.

The four controllers in the paper (final sweeps `cutin_final_20260910` and `cutout_final_20260910`):

| Paper label | Policy string |
|---|---|
| SCC | `acc_nair_smpc_const_lead_fixed_risk_nominal_safe_distance_rmove1000` |
| SCC + LSTM | `acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance_rmove1000` |
| Multimodal SCC | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_rmove1000` |
| Proposed | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_rmove1000` |

The IAIMM-KF comparison (`iaimm_cutin_final_20260930`, groups 01–04 only) ran the last two controllers with `stdan_3int` replaced by `iaimm_kf`, under this tuned setting:

```bash
export ACC_NAIR_IAIMM_CFG='{"lat_speedup": 2.0, "proj_horizon_s": 3.0, "q_lat_scale": 1.0, "trans_eps": 0.25, "weight": [0.1, 0.3, 10.0, 0.1]}'
```

When the variable is unset, the adapter's built-in defaults apply.

## 5. Environment

- **CARLA 0.9.13, map Town04.**
  - Export `CARLA_ROOT`, e.g. `/home/core-dev/CARLA/carla_0_9_13`. It has no default: the runner, the ACC agent and `fixed_lane_speed_agent.py` raise `ValueError` without it, and so do the unit tests.
  - The sweep's `--carla-root` flag defaults to the same path, but it only locates `CarlaUE4.sh`.
- **Python.**
  - The conda env `mmpreds-c0913` runs every sweep and test: Python 3.7, the CARLA 0.9.13 client, PyTorch 1.13 on CPU.
  - The maintainer also has a GPU env `hmg` (Python 3.9, PyTorch 2.7 with CUDA 11.8). No script in the repo depends on it.
- **Gurobi** needs a valid license. Without it, every step falls back (§3).
- **IAIMM-KF parameters.** IAIMM-KF reads `Model_Parameters.mat` from Zhou et al.'s repository, which is not committed. Clone that repository into the repository root:
  `git clone https://github.com/JianZhou1212/interaction-safety-aware-motion-planning.git`
  The adapter expects `interaction-safety-aware-motion-planning/Implementation/CASE_1_ISAMPC_SIM/Model_Parameters.mat`.

Environment variables:

| Variable | Use |
|---|---|
| `ACC_NAIR_STDAN_CKPT` | STDAN checkpoint. **Cut-in experiments must set `predictor/stdan_3int_signed_tcross_velint/checkpoints_out30_no_tcross/best_model.pt`.** If it is unset, the code tries `ACC_NAIR_STDAN_3INT_CKPT` and then silently loads `checkpoints_out30/best_model.pt`. With that checkpoint, early cut-in detection is lost: p > 0.5 arrives only around the moment the TV starts steering, instead of about 2 s before |
| `ACC_NAIR_LSTM_CKPT` | LSTM checkpoint. The default is `predictor/lstm/ckpt/best_model.pt`; the paper uses `predictor/lstm/ckpt/epoch_72.tar` |
| `ACC_NAIR_LSTM_VEL_CKPT` | Velocity-LSTM checkpoint (default `predictor/lstm_vel/ckpt/best_model.pt`) |
| `ACC_NAIR_IAIMM_CFG` | JSON of `IAIMMKFACCAdapter` keyword arguments (§4) |
| `CARLA_ROOT` | CARLA install (required) |

## 6. Running things

All commands run from the repository root. The paths are the maintainer's; adjust them to your machine.

```bash
export CARLA_ROOT=/home/core-dev/CARLA/carla_0_9_13
export ACC_NAIR_STDAN_CKPT=predictor/stdan_3int_signed_tcross_velint/checkpoints_out30_no_tcross/best_model.pt
export ACC_NAIR_LSTM_CKPT=predictor/lstm/ckpt/epoch_72.tar
PY=/home/core-dev/anaconda3/envs/mmpreds-c0913/bin/python
POL=acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_rmove1000
OUT=results/acc_scenario_sweep/my_sweep/$POL/01_cutin_normal

# One group of the paper sweep (normal cut-in), wrapped in the watchdog
$PY scripts/carla/devel/watch_acc_sweep.py --no-cleanup-carla-before-start --poll-s 60 --stall-s 1500 \
    --max-restarts 6 --restart-delay-s 60 --output-root "$OUT" -- \
  $PY scripts/carla/devel/sweep_acc_scenarios.py --kind cutin --ego-policy $POL --output-root "$OUT" \
    --port 2030 --fresh-server-per-run --port-step 4 --server-timeout 180 --ego-horizons 15 \
    --spawn-settle-s 6.0 --no-spawn-settle-realtime --cruise-warmup-s 4.0 --lane-shift-right 1 \
    --no-stop-on-run-failure --resume --max-sim-time 19 \
    --target-start-gaps 35 --target-lead-gaps 35 --same-lane-distances 0 \
    --outer-blocker-counts 3 --outer-blocker-spacings 12.0 --outer-blocker-speed-deltas=-4.5 \
    --ego-speeds 14,17 --target-speeds 11,13,15 --target-lead-speed-deltas=-6 \
    --trigger-distances 17 --ego-gap-at-trigger 22 --lane-change-times 3.0,3.3333,3.6667
```

The paper sweep runs seven groups per policy, all with the runner and watchdog flags above.

- **Groups 01–04** add the "CIN" flags: `--target-start-gaps 35 --target-lead-gaps 35 --same-lane-distances 0 --outer-blocker-counts 3 --outer-blocker-spacings 12.0 --outer-blocker-speed-deltas=-4.5`.
- **Groups 05–07** add `--ego-speeds 14,17 --target-speeds 11,13 --target-start-gaps 30 --same-lane-distances 0 --cutout-directions left`.

Beyond those shared flags, the groups differ only in:

| Group | `--kind` | `--max-sim-time` | Group-specific flags |
|---|---|---|---|
| `01_cutin_normal` | `cutin` | 19 | `--ego-speeds 14,17 --target-speeds 11,13,15 --target-lead-speed-deltas=-6 --trigger-distances 17 --ego-gap-at-trigger 22 --lane-change-times 3.0,3.3333,3.6667` |
| `02_cutin_aggressive` | `aggressive_cutin` | 19 | the speeds and lead Δv of 01, with `--trigger-distances 13 --ego-gap-at-trigger 14 --lane-change-times 1.6667,2.0,2.3333` |
| `03_no_cutin_decel` | `no_cutin_decel` | 19 | `--ego-speeds 14,17 --target-speeds 15 --target-lead-speed-deltas=-4,-6,-8 --trigger-distances 22 --ego-gap-at-trigger 22 --lane-change-times 3.0` |
| `04_no_cutin_decel_onset` | `no_cutin_decel` | 19 | `--ego-speeds 14,17 --target-speeds 13,15 --target-lead-speed-deltas=-6 --trigger-distances 22,34 --ego-gap-at-trigger 22 --lane-change-times 3.0` |
| `05_cutout_no_sublv` | `cutout_no_sublv` | 25 | `--lane-change-times 2.0,3.0 --trigger-times 12.0` |
| `06_cutout_sublv_normal` | `cutout_sublv` | 25 | `--lane-change-times 3.0 --trigger-distances 17 --target-lead-gaps 89 --target-lead-speed-deltas=-6` |
| `07_cutout_sublv_aggr` | `cutout_sublv` | 25 | `--lane-change-times 2.0 --trigger-distances 13 --target-lead-gaps 89 --target-lead-speed-deltas=-6` |

What happens in each scenario:

- **Cut-in kinds.** The TV drives in the adjacent lane behind its own lead, which is slower by `--target-lead-speed-deltas`. The TV starts its lane change when its gap to that lead reaches `--trigger-distances`.
- **`no_cutin_decel`.** The adjacent vehicle decelerates but never cuts in.
- **`cutout_no_sublv`.** The ego-lane lead leaves its lane at a fixed time.
- **`cutout_sublv`.** A second lead drives 89 m ahead of the ego-lane lead, 6 m/s slower than it. The ego-lane lead leaves when its gap to that second lead reaches `--trigger-distances`.

Metrics, tables and figures:

```bash
$PY scripts/carla/devel/aggregate_cutin_table.py results/acc_scenario_sweep/cutin_final_20260910 --window-s 17.5
$PY scripts/carla/devel/aggregate_cutin_table.py results/acc_scenario_sweep/cutout_final_20260910 --window-start-s 3
$PY overleaf/figs/scripts/make_tex.py   # no arguments; rewrites overleaf/table_cutin_final.tex and table_cutout_final.tex
$PY scripts/carla/devel/animate_acc_stdan_scenario.py <run_dir> --accel-ylim MIN MAX --speed-ylim MIN MAX
```

The aggregator writes `METRICS.md`, `table_cutin_runs.csv`, `table_cutin_summary.csv` and LaTeX rows into the sweep root.

The `tests/` folder has no `__init__.py`, so name the test modules explicitly when running the unit tests. `CARLA_ROOT` must be set.

```bash
$PY -m unittest $(ls tests/test_*.py | sed 's#/#.#; s#\.py$##')
```

## 7. Output layout

```
results/acc_scenario_sweep/<sweep>/<policy>/<group>/
    sweep_config.json, sweep_manifest.json
    watchdog.log, watchdog_attempt_NNN.log   # the attempt logs hold the sweep's stdout
    <kind>_<NNNN>/
        scenario_result.pkl                  # the full log
        summary.json, metrics.json, resolved_config.json
        carla_server.log                     # only with --fresh-server-per-run
```

`scenario_result.pkl` is a dict.

- **Per-actor keys:** `ego_<i>`, `target_cutin_<i>`, `traffic_cutin_lane_lead_<i>` and so on.
- **Run-level keys** (prefixed with `_`): `_collision_log`, `_collision_count`, `_collision_sensor_errors`, `_prediction_log`, `_spawn_settle_log`, `_cruise_warmup_log`, `_retired_actors`.

Inside it:

- **`state_trajectory`.** Every actor has rows `[t, x, y, yaw, v]`. They are in a right-handed frame: CARLA's y and yaw are sign-flipped, and yaw is in radians.
- **`policy_log["steps"]`.** The ego also has this list, one entry per 0.05 s tick. Each entry has `time_s`, `ego_s`, `ego_v`, `accel_cmd`, `solve_path`, `predictor_type` and `stdan_debug`.
  - In predictor runs, `stdan_debug.processed_targets` holds the mode probabilities, the predicted Frenet trajectories and ego-lane membership.
  - Since 2026-09-04, `stdan_debug.stdan_ckpt` records the checkpoint path.
  - SCC (`const_lead`) runs have neither key.
- **`_collision_log`.** Each entry has `time_s`, `actor_role`, `other_actor_role` and `normal_impulse_norm`.

## 8. Pitfalls — read before interpreting any sweep

1. **Count fallbacks first.** A run full of fallback steps (§3) still ends with `ran_successfully: true`.
   - The classic case: the measured ego speed sat above `v_max` by more than one jerk-limited step could remove. (The ideal actuator held an ego at 20.11 m/s against `v_max = 20`.) Every step was then infeasible, and the ego stayed on the fallback command, which was zero acceleration in that case.
   - Check the aggregator's `fallback_ratio` before any other metric. It is 0 in every run of the final sweeps.
2. **STDAN checkpoint.** Forgetting `ACC_NAIR_STDAN_CKPT` silently changes cut-in detection timing (§5). Compare runs only if their `stdan_debug.stdan_ckpt` matches.
3. **Time origin.** `time_s` counts from CARLA server start. The scoring origin `t0` is the first ego `policy_log` step (about 15 s), not the sum of spawn settle and cruise warm-up.
4. **CARLA is not deterministic.** Re-running an identical cell moves the event-window `j_max` by about ±0.5. Do not judge a change from one or two runs.
5. **Cut-in scoring window.** Aggregate cut-in sweeps with `--window-s 17.5`. After about 32 s of absolute time, route following misidentifies the lead, which produces fake gap collapses and collisions.
6. **Town04 lateral scrape.**
   - Near a Town04 road-segment boundary (y ≈ 47), the ego's path runs only about 2.0 m centre-to-centre from adjacent-lane traffic, instead of the usual 3.5 m.
   - An ego overtaking a slower adjacent car can therefore scrape it. `ego_collision` records the scrape, but it is not a control failure.
7. **Predictor valid domain.**
   - **Grid range.** STDAN only sees neighbours within about ±30 m longitudinally: ±29.9 m in the CARLA adapter, |dy| < 90 ft = 27.4 m in training. This caps how early a cut-in can be detected.
   - In the paper's cut-in groups, the TV's own lead starts 35 m ahead of the TV, outside that range.
   - **Speed.** An ego speed of 20 m/s is outside the NGSIM training distribution.
   - **Prior.** Class balancing (`balance_211`, which undersamples lane keeping) inflates the lane-change prior about 14.5×.
8. **Fix simulator artefacts at the source.** Do not change predictor input features to hide a simulator artefact. Fix the scenario, the vehicle or the conversion code instead; this is the maintainer's rule.
9. **Known caveats in the final cut-in table** (`overleaf/table_cutin_final.tex`).
   - **Proposed normal cut-in, run `cutin_0012`.** The ego spawned 1.63 m sideways of every other run (x −11.27 vs −12.90).
     - Its collision, Proposed's only one in that group (1/18), is a side scrape with the adjacent lane's lead, not a cut-in failure.
     - A rerun of it had no collision.
   - **SCC aggressive cut-in, run `aggressive_cutin_0012`.** It failed to spawn ("Ego lane is not adjacent"), so SCC has 17 aggressive runs.
     - A rerun of it collided, which would make SCC 3/18.
   - The reruns are in `results/acc_scenario_sweep/iaimm_cutin_final_20260930/reference_reruns/`.
   - **δ and Proposed.** The intrusion metric δ is scored against the unscaled safe distance for every controller, so Proposed's α-relaxed standoff counts as intrusion.
10. **Markdown is ignored by git.** `.gitignore` contains `*.md`, with an exception only for `README.md`. Add new docs with `git add -f`.

## 9. Working conventions

- **Git.** Push to the `fork` remote (`hi-jun/SMPC_MMPreds`), never to `origin` (upstream `shn66`). Snapshot-commit any uncommitted work before editing code.
- **Use the existing tools rather than writing new ones.**
  - Add new metrics to `aggregate_cutin_table.py` instead of computing them ad hoc.
  - Make figures with `animate_acc_stdan_scenario.py`, with fixed `--accel-ylim` and `--speed-ylim` when panels sit side by side.
  - Wrap CARLA sweeps in `watch_acc_sweep.py`.
  - Keep one-off analysis scripts out of the repository.
- **Claims.** Report only improvements that were measured. Before comparing with an older sweep, re-aggregate both with the current script, because metric definitions have changed over time.
- **Changes.** Keep them surgical: touch only what the task needs, and match the existing style.

## 10. Open items (2026-10-07)

- **`_fallback_policy` sign.** It commands `a_ref − 0.25·min(0, gap)`. `safety_function` is negative when the gap is violated, so this accelerates toward a lead that is too close. No run in the final sweeps used the fallback (§8.1).
- **Step-0 jerk cost.** All three QP builders price the step-0 command change with `config.dt` (0.2 s) instead of `command_dt` (0.05 s), which makes it 16× too cheap. The hard jerk constraint already uses `command_dt`.
- **Unfiltered x₀.** The MPC takes the raw measured ego speed as x₀. The ideal actuator runs with zero throttle, so CARLA's tyres slip from tick to tick. That slip adds speed noise and inflates command jerk.
- **σ-term cap.** The confidence chance constraint uses one capped β = min(p, β_ref) for both the standoff scale and the σ term, so the σ term never exceeds q(0.6)·σ ≈ 0.84σ. Whether the σ term should get its own uncapped β is undecided.
- **STDAN grid edges.** The CARLA adapter puts its grid edges at ±6.5 cells of 4.6 m (±29.9 m). Training kept neighbours with |dy| < 90 ft (27.43 m), which cuts at the outer cells' centres. Most of the 2.5 m difference comes from this edge convention, not from the cell size.
- **TV re-trigger.** After the ego-proximity guard holds or aborts a TV lane change, the TV restarts it as soon as the ego is no longer alongside. The restart checks only the TV's gap to its own lead, never its gap to the ego.
- **Paper tex vs code.** The four differences are listed in §3.
- **Table caption.** The caption of `overleaf/table_cutin_final.tex` says every controller has 17 aggressive runs; only SCC does.
- **IAIMM-KF tuning.** IAIMM-KF's settings were tuned offline on replays of 58 final-sweep runs, i.e. in-sample. STDAN was never tuned on CARLA.
