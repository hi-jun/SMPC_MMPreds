# AGENTS.md — guide for AI assistants working on this repository

Audience: AI coding assistants (Claude, GPT, …) and new contributors who read this repository on GitHub.
It describes the state on 2026-10-20, branch `feat/probability-gated-cutin`. Where this file and the code disagree, the code is right.

**The maintainer works in Korean. Answer in Korean unless asked otherwise.**

---

## 1. What this repository is

- **Origin.** This is a fork of [shn66/SMPC_MMPreds](https://github.com/shn66/SMPC_MMPreds) (Nair et al., stochastic MPC with multimodal predictions). The root `README.md` is the upstream README and describes the original 2-D planner, not this fork's work.
- **What the fork does.** It turns that planner into a **1-D longitudinal ACC (adaptive cruise control) controller**. The controller is evaluated in **CARLA 0.9.13 (Town04)** on cut-in, no-cut-in and cut-out scenarios.
- **Predictors.**
  - **STDAN**, a learned trajectory predictor with three intentions (lane keep, left lane change, right lane change)
  - single-mode **LSTM** baselines
  - **IAIMM-KF** (Zhou et al.), a model-based baseline
- **The paper's contribution ("Proposed", policy token `cutin_chance0.6`)** is a confidence chance constraint. The code describes it as the 1-D counterpart of Benciolini et al. eq. (19a).
  - Inputs: each mode j has probability p_j, and β_ref = 0.6.
  - Definitions: q(β) = Φ⁻¹((1+β)/2) and α_j = q(β_j)/q(β_ref).
  - **Cut-in modes:** β_j = min(p_j, β_ref). The gap to the predicted vehicle must exceed α_j·d_safe(v) + q(β_j)·σ_j.
  - **Lane-keeping (`lk`) modes of the ego-lane lead:** the safe distance is scaled by α_j, with β_j = min(p_j, β_ref) and no σ term.
  - **`cutout` modes:** β_j = min(1 − p_j, β_ref). The scale is floored at 0.35, and there is no σ term.
- **The current design is D-opt, `dopt100lkr_trajcutin`** (the maintainer's decision on 2026-10-17). Its policy string is `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_rmove1000`: Proposed plus three tokens (§4).
  - `dopt100`: on cut-in cells α_j scales only the footprint L + d0, and the optimizer chooses how much of the time gap τv to keep, at a price 100·P_j·shortfall².
  - `lkr`: the same price on the lane-keeping cells of a vehicle that has had a cut-in cell, until it leads and the gap has held the full standoff for 0.5 s. Every other in-lane lead keeps the hard standoff.
  - `trajcutin`: an adjacent vehicle's lane-keeping trajectory that enters the ego lane is a cut-in mode of its own.
  - Proposed (`cutin_chance0.6` alone) is kept unchanged so that the paper's sweeps reproduce. The paper and its tables (2026-09-10) predate D-opt.
  - Against Proposed on the same code (`cutin_postfix_cmp_20261016`, `cutout_postfix_cmp_20261016`, 66 runs each): aggressive cut-in a_min_cmd −2.73 → −2.09 m/s² (+3.0 SE) and j_max_cmd_filt 4.07 → 3.14 (−2.4 SE); normal cut-in a_min +1.4 SE, j_max −1.6 SE; no-cut-in and cut-out not distinguishable (|diff| ≤ 1.4 SE).
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
| `scripts/carla/policies/mpc_agent.py` | Upstream path-following MPC that the scripted traffic agents inherit (IPOPT, 0.2 s CPU limit). Since 2026-10-19 a solve without a warm start starts from the current state and the reference instead of zeros (§8.13) |
| `scripts/carla/utils/pcacc.py`, `scripts/carla/policies/pcacc_agent.py` | Baseline `acc_pcacc`: Althoff, Maierhofer & Pek's "Provably-Correct and Comfortable ACC" (T-IV 2021), rule-based, no predictor (§4) |
| `scripts/carla/policies/isampc_agent.py` | Baseline `acc_isampc`: Zhou, Olofsson & Frisk's ISA-MPC restricted to lane keeping, on the IAIMM-KF port (§4) |
| `scripts/carla/run_all_scenarios.py` | Maps `--ego-policy` prefixes (`acc_nair_smpc`, `acc_isampc`, `acc_pcacc`) to the runner's `policy_type` |
| `scripts/carla/scenarios/run_lk_scenario.py` | The scenario runner: spawns actors, steps CARLA, records results |
| `scripts/carla/devel/sweep_acc_scenarios.py` | Grid sweeps. `--kind` selects the scenario family (the kinds are built in `acc_scenario_factory.py`) |
| `scripts/carla/devel/watch_acc_sweep.py` | Watchdog that restarts a sweep after a CARLA crash or stall. It counts and cleans up only the CARLA servers in its own sweep's port range (`--carla-port-span`), so several sweeps can share the machine (§6) |
| `scripts/carla/devel/aggregate_cutin_table.py` | Metrics and tables. **Its docstring is the definition of record for every metric** |
| `scripts/carla/devel/animate_acc_stdan_scenario.py` | Renders one run: scene, predicted intentions, mode probabilities, control panel |
| `predictor/stdan_3int_signed_tcross_velint/` | The STDAN used in the paper: model, training (`preprocessing/`), checkpoints. `acc_adapter.py` turns CARLA history into model tensors and Frenet modes. `acc_postprocess.py` maps intentions to ACC modes and applies the confidence scaling |
| `predictor/lstm/`, `predictor/lstm_vel/` | Single-mode LSTM baselines, with position and velocity output respectively |
| `predictor/iaimm_kf/` | IAIMM-KF port (policy token `iaimm`) |
| `predictor/stdan/` | The original 9-mode STDAN. The other adapters still import `FT2M` and `transform_points` from it |
| `predictor/stdan_vel/` | A 9-mode STDAN with velocity output |
| `predictor/uq/` | Conformal novelty detector and a 5-member deep ensemble for the `stdan_vel` checkpoint, ported from the maintainer's thesis repository (added 2026-10-13). Not wired into the controller, and not valid for `stdan_3int`. Read `predictor/uq/AGENTS.md` first |
| `overleaf/` | Paper tables, the final formulation, and the figure and table generators (`overleaf/figs/scripts/`) |
| `tests/` | 10 unit-test modules (193 tests): controller port, chance/cut-out extension, cut-out trajectory mapping, fixed-lane agent, ideal actuator, LSTM adapter, STDAN integration, context trust `ctx`, PC-ACC, ISA-MPC |
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
| `sigmacap<c>` | With `cutin_chance` only: the σ term takes min(p_j, c) instead of min(p_j, β_ref); the standoff factor keeps β_ref (added 2026-10-07, not in the paper) |
| `dopt<w>` | With `cutin_chance` only: on cut-in cells the probability factor α_j scales the footprint L + d0 only, and the optimizer picks how much of the time gap τv to keep, paying w·P_j·shortfall² (`NairACCConfig.standoff_shortfall_weight`); the standoff lies in [α_j(L + d0), α_j(L + d0) + τv]. Cut-out cells keep Proposed's factor. Cached feedback QP only (added 2026-10-07 with a fixed L + d0 floor, which 2026-10-15 replaced by α_j(L + d0); not in the paper) |
| `dopt<w>lk` | `dopt<w>` that also prices the shortfall on the ego-lane lead's lane-keeping cells (`NairACCConfig.standoff_shortfall_on_lk`, cells marked by `MultimodalLeadPrediction.lead_mode_lk`), so a cut-in keeps its price once it is in the lane instead of snapping back to the hard d_safe and the emergency slack. Cut-out cells still keep Proposed's factor. `stdan_3int` only (added 2026-10-16, not in the paper) |
| `dopt<w>lkr` | `dopt<w>lk` whose lane-keeping cells are priced only for a vehicle that has had a cut-in cell (from then on, also while it is still crossing the line), until it leads at step 0 and the gap to it has held the full standoff for 0.5 s (`NairACCConfig.standoff_shortfall_lk_cutin_only`, `NairACCSMPC._restrict_lk_pricing`); every other in-lane lead keeps the hard standoff as in Proposed. Priced on every in-lane lead, `dopt<w>lk` settled 6–13 % inside d_safe behind the cut-out groups' slow second lead (added 2026-10-16, not in the paper) |
| `trajcutin` | An adjacent vehicle's lane-keeping-labelled trajectory that enters the ego lane within the horizon becomes a cut-in mode of its own (own probability and trajectory; σ term, cut-in factor, gate) instead of a lane-keeping cell with the full, unscaled standoff. While a vehicle crosses the lane line its LK trajectory drifts into the ego lane 0.4–0.5 s before the vehicle is reclassified as ego-lane. The labelled cut-in keeps its mode; nothing changes while no LK trajectory enters. A vehicle whose every non-cut-in trajectory enters has no lane-keeping mode (that raised until 2026-10-17, §8.12). The entering trajectory's mode is still a `cutin` for every constraint but is labelled `cutin/LK` (`ACCModePrediction.scenario_label`) in scenario names and effective-lead keys, so a vehicle's two cut-in modes are distinct scenarios (`acc_postprocess._mode_specs_for_relation`, `stdan_3int` only, added 2026-10-16, not in the paper) |
| `branchdec<a>` | Splits a cut-in hypothesis from lane keeping at its vehicle's decision point instead of at the step its trajectory enters the ego lane. The decision step is Nair et al.'s traffic-light rule carried over: the TV closing on its own lead at Δv must decide by the gap `Δv²/(2a) + 7 m`, so `k_dec = ceil((gap − g_dec)/(Δv·dt))`, floored at 1 (`acc_postprocess.decision_point_branch_step`). From that step a still-inactive cut-in mode adds `(vehicle, label, "decided")` to the scenario's effective-lead key, which splits the policy tree there; constraint cells are unchanged. The agent collects every other vehicle's Frenet state to find each target's own lead and logs `stdan_debug.branch_decision`. Measured 2026-10-18 with a = 7 on the paper grid: the split moves as designed (13/8/6/3/1 vs 13/12/10/6/3 at t_cross −3…−0.5 s) but every metric is unchanged. For Proposed, by the time `k_dec` precedes lane entry the cut-in probability already exceeds β_ref (α = 1). For `dopt100lkr_trajcutin`, α only scales the footprint and the w = 100 price demands 87–100 % of d_safe at the entry step almost regardless of P; in a two-scenario re-solve the shared u₀ then sits on the command jerk limit whether the split is at entry or at step 1, unless P < 0.3, and with `rmove1000` the split moves u₀ by ≤ 0.07 m/s². The executed braking only shifts in time (0.2–0.3 m/s² less at t_cross −2…−1.5 s, up to 0.14 more after −1 s). `stdan_3int` only, off by default, not in the paper |
| `ctx<λ>` | Context trust λ ∈ [0, 1] (e.g. `ctx0.5`), `stdan_3int` only: an adjacent vehicle's cut-in-direction intention is mixed with its measured lateral motion, `p_λ = p_S·[(1 − λ) + λ·o_v]`, before the ACC-mode mapping, so the gate, α, the `dopt` price, `trajcutin` and `branchdec` all read p_λ; the removed mass goes to the vehicle's lane-keeping intention and every trajectory stays STDAN's. `o_v = clip((y_v − 0.10)/(0.85 − 0.10), 0, 1)` with y_v the vehicle's offset from its own lane centre (CARLA waypoint) toward the ego lane (`acc_postprocess.mix_cutin_probability`, `ACCNairSMPCAgent._lateral_offset_toward_ego_lane`; logged as `stdan_debug.processed_targets[*].ctx_*`). λ = 0 is the plain policy, λ = 1 believes no cut-in before the vehicle moves laterally. At p_λ = 0 the mode leaves the scenario set (zero-probability modes are skipped) rather than being gated. The aggregator counts the design's two loss events per run, `ctx_miss` (entered and δ > `--ctx-delta-thr` within 3 s after t_cross; default 0.2, not fixed yet) and `ctx_unnec` (did not enter, a_cmd ≤ −0.3 while no measured in-lane lead within d_safe), for every policy. No λ is calibrated yet: `ctx0.6`, from a predictor-level NGSIM calibration, made D-opt worse in CARLA, and a calibration on closed-loop control outcomes is planned (§10) (added 2026-10-19, not in the paper) |
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

The current design (§1), not in the paper:

| Label | Policy string |
|---|---|
| D-opt | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_rmove1000` |

Compare it with the paper's controllers only on runs made with the same code (§8.11). The same-code roots `cutin_postfix_cmp_20261016` and `cutout_postfix_cmp_20261016` hold it next to Proposed, Multimodal SCC and the other D-opt variants.

Two external baselines (added 2026-10-19, smoke-tested on one run each, no sweep yet). They are separate ego policies, not `acc_nair_smpc` tokens; `run_all_scenarios.py` maps the prefix to the runner's `policy_type`, and the aggregator scores them like SCC runs:

| Label | Policy string | Notes |
|---|---|---|
| PC-ACC | `acc_pcacc` | Althoff et al.: safe distance = ego braking along the paper's jerk profile minus the lead braking at −10.5 m/s²; nominal QP (2)–(6); emergency jerk profile when (7) fails; lane membership by the paper's rule (lateral > 0.4 m and heading > 0.035 rad for 0.5 s, δ_leave 1 s); cut-in recapture QP (Prop. V.1, a_cut-in −2 m/s²). Measured states only. Solver qpOASES. **By default (maintainer's decision, 2026-10-19 evening) the standoff is the fork's d_safe = L + d0 + τ·v on the aggregator's gap** — in the nominal reference and constraint (3) (per step, linear in v_k), the safety check (7) (next-state gap below the standoff engages the emergency profile), the cut-in definition (Def. II.3) and the recapture terminal constraint; the inevitable-collision test stays physical — and the limits are the fork's (a ∈ [−3, 2], |j| ≤ 5). `paperdsafe` restores the paper's braking-based d_safe and `amin10_amax3_jlim10` Table I's limits, so `acc_pcacc_paperdsafe_amin10_amax3_jlim10` is the faithful variant. The clearing time t_c is not in the paper; default 4 s (`tc<x>`). Smoke run (ego 17 / TV 15, fork standoff): at entry the recapture QP is infeasible (25 m standoff vs a 10 m entry gap within t_c), so the bounded fallback brakes at −3 m/s² for about 1.7 s; the standoff is restored 2.75 s after t_cross, δ_max 57 % (the entry gap itself) |
| ISA-MPC (cruise only) | `acc_isampc` | Zhou et al.'s planner with the lateral motion frozen: per step, a Gaussian mixture over the IAIMM-KF modes cut at ε = 0.8 of its peak gives an occupancy box (a mode survives iff p ≥ 0.8·p_max; kept modes are a worst case; no chance constraint, single input sequence), a speed projection through their closed-loop tracker, then a snap-input QP with a soft standoff (slack weight 0.055, terminal speed cost). **Both standoffs are the fork's d_safe = L + d0 + τ·v on Frenet centre gaps** (maintainer's decision, 2026-10-19 evening; Zhou's Th_QP = 2 s and Th_MPC = 1.5 s headways are gone): the projection becomes v_ref ≤ (x_rear,k − B_k − (L + d0))/(A_k + τ), the MPC row x_rear,k − s_k − (L + d0 + τ v_k) + ρ_k ≥ 0, with x_rear the level-set box's rear edge without the longitudinal vehicle growth (ζ_l = 0, since L covers the vehicle; the lateral growth and the lane test are theirs). Subclass of `ACCNairSMPCAgent`, Gurobi. Deviations: N = 15 at 0.2 s (theirs 25 × 0.32 s; `--ego-horizons 40` restores 8 s), the fork's limits [−3, 2] / 5 m/s³ (theirs |a| ≤ 6, no jerk bound), command = the plan advanced 0.05 s along the snap chain. Solve 8 ms + IAIMM-KF 28 ms per tick (p99 48 ms of the 50 ms budget). **Appendix ablation `acc_isampc_ainput_rmove1000`** (2026-10-20, not the baseline): acceleration input and the fork's own jerk terms (r_jerk 10 on ((a_k − a_{k−1})/dt)², r_move 1000 on (a_0 − u_prev)², the fork's hard jerk limits) in place of Zhou's snap input and snap cost, all other Zhou terms kept — to show how much of the smoothness comes from the snap input. Without `rmove<w>` the step-0 weight is the config's r_jerk/dt² = 250. D-opt's QP sums these terms over three identical samples, so its absolute jerk price is 3× this one's |

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

**Several sweeps at once.** Give each line its own `--port`, 200 apart (2030, 2230, 2430, …), and its own output root. The watchdog counts and cleans up only the CARLA servers in `[--port, --port + 200)` (`--carla-port-span`), so lines never kill each other's servers; before 2026-10-15 it killed every CarlaUE4 on the machine, and lines that shared `--port 2030` crashed each other's servers. Set `OMP_NUM_THREADS=6` per line so the predictors do not oversubscribe the CPU. Measured on 2026-10-15 with four lines: a run takes 69 s instead of 59 s, so throughput is about 3.5×, and the results match serial runs (§8.11).

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
11. **Runs made before 2026-10-15 carry a command/tick race.** `set_target_velocity` and `apply_control` are asynchronous RPCs. The runner used to send the ego's command right before `world.tick()`, and on 1.4–1.8 % of the ego's ticks the server processed the tick first: the command landed one tick late and the measured speed dipped by about 0.2 m/s for one tick.
    - Since 2026-10-15 the runner queues every vehicle command and applies the batch with `client.apply_batch_sync` before each tick (`_apply_tick_commands`), which returns only after the server applied it. The dips are gone (0 in 11,400 ego ticks, serial and parallel alike).
    - Measured-speed metrics (raw `j_max` and `a_min`, `j_max_filt`) and some δ values differ systematically between runs made before and after the fix; the command-based metrics (`j_max_cmd_filt`, `a_min_cmd`), Δt_ant and collisions do not. The paper sweeps (`*_final_20260910`) predate the fix. Do not mix the two in one table.
12. **A predictor exception does not show up as a fallback.** When `predict_acc` raises, `_stdan_prediction_bundle` catches it and runs the tick on the synthetic two-mode prediction (`lane_keeping` 0.6 / `cutin` 0.4, no vehicle ids), logging `stdan_debug.fallback = True` and `stdan_debug.reason`; `solve_path` stays normal, so `fallback_ratio` is 0. The aggregator's `pred_fallback_ratio` and `pred_error_steps` columns count these (since 2026-10-17; `pred_error_steps > 0` makes the run invalid). The first `trajcutin` grid (2026-10-16, code `621d5ec`) hid such an exception in 108/108 cut-in runs, 6–9 ticks each, right before the TV was reclassified as ego-lane; the synthetic prediction also cleared `dopt<w>lkr`'s armed set, which is what made those runs look worse. `fallback = True` without a `reason` means the predictor returned nothing (no vehicle in the grid), which is normal.
13. **Runs made before 2026-10-19 carry two run-start lateral artefacts.** (a) The scripted agents' first IPOPT solve (`mpc_agent.py`) started from an all-zero initial guess, often hit the 0.2 s CPU limit, and its unconverged steering (at the ±0.5 rad bound) was applied: the cut-in TV swerved up to 1.2 m sideways within the first second (7/50 TVs > 0.3 m under four parallel lines, 2–4/50 in the paper sweep, 0 after the fix). STDAN read the swerve as a cut-in (p 0.78 at t0 + 0.25 s in 01/cutin_0003) and the ego braked for it. (b) The 4 s cruise warm-up applied no steering, so a vehicle with a spawn heading error drifted straight: one right-lane car started t0 1.47 m off its lane centre. Since 2026-10-19 the first solve starts from the current state and the reference (converges in 0.02–0.03 s under load) and the warm-up steers every vehicle onto its lane centre (`_tick_vehicles_cruising`, Stanley form; the shared `lane_keeping_steer` was lightly damped under the velocity pin). Verified on 12 single runs (`startfix_check_20261019`): no infeasible TV tick, lateral motion ≤ 0.001 m, every vehicle ≤ 0.02 m off centre at t0, triggers unchanged, t_cross 0.05–0.15 s later. The first second's predictor behaviour differs between old and new runs, so do not mix them in one table; every sweep root before this date, the paper's included, is "old".

## 9. Working conventions

- **Git.** Push to the `fork` remote (`hi-jun/SMPC_MMPreds`), never to `origin` (upstream `shn66`). Snapshot-commit any uncommitted work before editing code.
- **Use the existing tools rather than writing new ones.**
  - Add new metrics to `aggregate_cutin_table.py` instead of computing them ad hoc.
  - Make figures with `animate_acc_stdan_scenario.py`, with fixed `--accel-ylim` and `--speed-ylim` when panels sit side by side.
  - Wrap CARLA sweeps in `watch_acc_sweep.py`.
  - Keep one-off analysis scripts out of the repository.
- **Claims.** Report only improvements that were measured. Before comparing with an older sweep, re-aggregate both with the current script, because metric definitions have changed over time.
- **Changes.** Keep them surgical: touch only what the task needs, and match the existing style.

## 10. Open items (2026-10-20)

- **Context trust λ: calibrate on control outcomes (next work item).** `ctx<λ>` exists; no λ is calibrated. History:
  - 2026-10-19/20, predictor-level calibration on NGSIM Val+Test (`results/ngsim_lambda_cal_20261019/`, s0–s7; inputs bit-exact with the training loader; 1,233 cut-in / 67,078 no-cut-in episodes). Loss "share of the 2.8 s label window with p_λ ≥ 0.1" gave λ̂(ε 0.2) = 0.6; loss "label-window mean of min(p_λ/0.6, 1)" gave none (STDAN alone already loses 0.218).
  - In CARLA (`cutin_verify_20261020`) `ctx0.6` weakened D-opt: prediction lead 2.83 → 1.30 s (normal) / 2.04 → 0.82 s (aggressive), a_min_cmd 0.78 / 0.92 m/s² harder (3.3 / 5.4 SE), δ_max +7 / +15 %p, aggressive `ctx_miss` 6 → 17 of 18; unnecessary decelerations only 12 → 8 of 14. The proxy losses never saw what the controller does with P: before lateral motion λ = 0.6 cuts P to 0.4·p_S, which still clears the 0.1 gate but scales α and the `dopt` price.
  - Decision (maintainer): calibrate on the controller's realised losses, as Conformal Decision Theory does. The CommonRoad scenarios in `/home/core-dev/HJ/ACC/safe-acc` cannot serve: 25 of 26 US-101 snippets match NGSIM US-101 07:50 (first 3.2 min) and every vehicle in them is a STDAN Train vehicle; there is no I-80 scenario. Plan (route B, not built yet): a standalone NGSIM closed-loop replay in `mmpreds-c0913` — the Val+Test episodes, the target-lane follower removed and replaced by the simulated ego (written into its rows so STDAN's inputs stay bit-exact), all other vehicles replayed (non-reactive), STDAN-3int and D-opt + `ctx<λ>` in the loop with the CARLA code, Frenet = (Local_Y, Local_X) with the measured lane centres; per episode the realised loss (front-gap intrusion against d_safe after t_cross; rear contacts excluded), then CRC with a monotonised loss on held-out TVs. Undecided: the loss threshold δ_thr, the subset (main-lane TVs, one episode per TV), the calibration/report split, ε.
- **Verification sweep (2026-10-20).** Decided: both baselines hold the fork's d_safe and the table scores δ against it; PC-ACC uses the fork's limits; ISA-MPC keeps the 3 s horizon and its original snap input and cost (method internals stay the authors'; only the task, the safe distance, the limits and the horizon are common conditions). Roots `cutin_verify_20261020` / `cutout_verify_20261020` (launcher `launch_verify_all_20261020.sh`), post-fix code (§8.13), 66 runs per policy, all valid. Cut-in normal / aggressive δ_max: SCC 55 / 84 %, PC-ACC 43 / 79 %, ISA-MPC 31 / 54 %, D-opt 2 / 19 %, D-opt + `ctx0.6` 9 / 34 %; ego collisions SCC 2 / 3, PC-ACC 0 / 2, the rest 0 (all in the ego 17 / TV 11 cell); j_max_cmd_filt ISA-MPC 1.19 / 1.75 vs D-opt 2.18 / 2.99; unnecessary decelerations in the 14 no-cut-in runs: SCC 0, PC-ACC 0, ISA-MPC 4, D-opt 12, `ctx0.6` 8. PC-ACC's `aggressive_cutin_0012` failed to spawn ("Ego lane is not adjacent", the same cell SCC lost in the paper sweep, §8.9), so PC-ACC has 17 aggressive runs; SCC 02/0012 touched a guardrail with 0.3 % fallback (unchecked). The appendix ablation `acc_isampc_ainput_rmove1000` was smoother still on cut-in (j_max 0.73 / 1.14, a_min −1.21 / −1.76, δ_max 36 / 60 %), so ISA-MPC's smoothness is not due to the snap input; in cut-out group 05 it undershot to 0.3–4.6 m/s, then passed v_max = 20 m/s, fell back and hit a guardrail in 2 of 8 runs. Still open for ISA-MPC: ε (0.8 vs CASE_3/4's 0.1) and an 8 s horizon check (`--ego-horizons 40`).

- **`_fallback_policy` sign.** It commands `a_ref − 0.25·min(0, gap)`. `safety_function` is negative when the gap is violated, so this accelerates toward a lead that is too close. No run in the final sweeps used the fallback (§8.1).
- **Step-0 jerk cost.** All three QP builders price the step-0 command change with `config.dt` (0.2 s) instead of `command_dt` (0.05 s), which makes it 16× too cheap. The hard jerk constraint already uses `command_dt`.
- **Unfiltered x₀.** The MPC takes the raw measured ego speed as x₀. Most of the tick-to-tick speed noise that was attributed to tyre slip under zero throttle was the command/tick race (§8.11), fixed on 2026-10-15. Whether any slip noise remains has not been re-measured since.
- **σ-term cap.** The confidence chance constraint uses one capped β = min(p, β_ref) for both the standoff scale and the σ term, so the σ term never exceeds q(0.6)·σ ≈ 0.84σ. `sigmacap<c>` decouples them; `cutin_chance0.6` itself is unchanged so that the paper's sweeps reproduce. Whether the paper policy should switch is undecided.
- **`slack_steps` misses slack use in chance policies.** It counts `feasible == False`, and in chance modes `feasible` already adds the slack, so a solve that leans on the 5000-weighted slack still counts as feasible. The raw per-step `slack_max` (logged since 2026-10-07) shows it.
- **STDAN grid edges.** The CARLA adapter puts its grid edges at ±6.5 cells of 4.6 m (±29.9 m). Training kept neighbours with |dy| < 90 ft (27.43 m), which cuts at the outer cells' centres. Most of the 2.5 m difference comes from this edge convention, not from the cell size.
- **TV re-trigger.** After the ego-proximity guard holds or aborts a TV lane change, the TV restarts it as soon as the ego is no longer alongside. The restart checks only the TV's gap to its own lead, never its gap to the ego.
- **Paper tex vs code.** The four differences are listed in §3.
- **Table caption.** The caption of `overleaf/table_cutin_final.tex` says every controller has 17 aggressive runs; only SCC does.
- **IAIMM-KF tuning.** IAIMM-KF's settings were tuned offline on replays of 58 final-sweep runs, i.e. in-sample. STDAN was never tuned on CARLA.
- **Branch point vs. conservatism.** The tree's conservatism on this grid comes from how much the cut-in cell demands, not from where the policy splits: `branchdec7` moved the split and changed nothing (2026-10-18). For Proposed the demand saturates through α at p ≥ β_ref about 2 s before lane entry; for D-opt (w = 100) the price holds 87–100 % of d_safe at entry almost regardless of P. The maintainer's 2026-10-07 observation that an earlier split was less conservative is not reproduced by the decision-point rule; whether a first-input-only split (k_b = 1) reproduces it is untested.
