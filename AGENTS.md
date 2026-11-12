# AGENTS.md — guide for AI assistants working on this repository

Audience: AI coding assistants (Claude, GPT, …) and new contributors who read this repository on GitHub.
It describes the state on 2026-11-12, branch `feat/probability-gated-cutin`. Where this file and the code disagree, the code is right. Long records live in `docs/agents/` (§2), moved there unchanged on 2026-11-12 (updated 2026-11-12: moved verbatim, then reformatted for readability the same day with the words unchanged, and since then marked in place where stale, with dated "updated", "withdrawn", "clarified" and "pointer updated" notes; checks in §2).

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
- **Unlikely cut-in modes are switched off, not removed.** Below p_j = 0.1 (`CUTIN_CHANCE_VANISH_BELOW`; override with `cutin_gate<p>`), the mode's `active_mask` is cleared.
  - The mode then never acts as a lead, but it stays in the scenario set.
  - This gate applies to every STDAN and IAIMM-KF policy, not only to Proposed.
- **Paper files.** `overleaf/formulation_twosided_conditional.tex` is the MPC formulation as written for the paper; it differs from the code in four places (§3). The result tables are `overleaf/table_cutin_final.tex` and `overleaf/table_cutout_final.tex`.

### Current design (2026-11-12)

**NGSIM design (since the night of 2026-11-12): the final design, `onerow` with w 10 + `zconf0.95` + `nocap` + `snap160`, the NGSIM replay's default** (maintainer's decision, 2026-11-12 night: "C + snap160 is the final NGSIM design"). The string is `policy_example` of `results/ngsim_lambda_cal_20261019/replay_out/fin_npw3_iso_side_onerow_w10_z95_nocap_snap160_20261112/run_info.json` (λ = 0; the last row of §4's current-design table with `ctx<λ>` = `ctx0`):

```
acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold_entrant2_bcts_occin0.1_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_nocap_snap160
```

Its own token groups (details §4, full records `docs/agents/tokens.md`); the others are P1's, below:
- `onerow` with `dopt10`: one row on every active lead cell, cut-in hypotheses and lane-keeping leads alike, gap ≥ L + d0 + τv − short, the shortfall priced at 10·P_j·short²; no floor row, no separate hard lane-keeping row, no `lkr`; the reference asks for the plain d_safe.
- `zconf0.95`: the row's σ term z·σ, z = Φ⁻¹(min(p, 95 %)), p the cell's own hypothesis probability: 1.645σ on the lead's lane keeping, negative for a cut-in hypothesis below 50 %.
- `nocap`: no upper bound on the shortfall, so no row is hard; the ℓ1 slack stays in the string but was 0 on every solved tick of the final set, and the safety layer (`safelayer`, `entrant2`) is the only hard protection.
- `snap160`: a cost 3·P_j·160·Σ(j_k − j_{k−1})² on the jerk's changes, j_{−1} being the jerk the plant got on the last tick.
- Not in it, against P1: `dopt3lkr` (now `dopt10`), `lksigma`, `lkrsigma`, `floorrisk0.1`, `floorhyp`, `nopricesigma`, `lkonesided0.05` (`onerow` refuses them).

Against P1 on the NGSIM final set (paired, SE in brackets; `replay_out/onerow_20261112/tables_onerow_snap.md`, `headline_onerow_snap.md`, `result.md` addendum 5):
- critical cut-in (209 runs): T_rec 1.69 → 1.27 s (−0.50, SE 0.08; not recovered 17 → 1), runs at the falling jerk bound 50 → 12; j_max 1.31 → 1.14 (−0.17, SE 0.14) and δ_max 32.1 → 29.6 % (−2.6 %p, SE 1.9) within 2 SE; floor breaches 21 → 22; one fallback run (ds4_v3035, 1 tick at 0.10 m/s after a fail-safe; P1 none).
- mild cut-in (280): j_max 1.23 → 0.88 (−0.35, SE 0.05), a_min +0.15 (0.03), T_rec −0.26 s (0.05), runs at the falling jerk bound 56 → 2; δ_max 18.0 → 16.9 % (within SE); floor breaches 14 → 20.
- no cut-in (393): j_max −0.24 (0.03), phantom_ep 0.095 → 0.043 m/s² (−0.051, SE 0.014), v_avg +0.07 m/s (0.01), contacts 7 → 5.
- Contacts are side entries only, none rear-end (critical / mild 13 / 12 → 14 / 12); among the new ones the TV side entry ds4_v3230 (mild) at −0.33 m, where P1 kept 3.34 m.
- Not run in CARLA.

How the replay was switched and gated, and how to reproduce P1 and the other roots of 2026-11-12: `docs/agents/ngsim_replay_defaults.md` (2026-11-12) and `replay_out/onerow_20261112/adopt_snap160.log`. The path to this design (w sweep, zσ, `nocap`, `snap`): `docs/agents/design_history.md` (2026-11-12).

**Previous NGSIM design: P1** (updated 2026-11-12 night: it read "NGSIM design: P1, the NGSIM replay's default."; P1 was both from 2026-11-12 until that night; `NGSIM_POLICY_DOPT=_dopt3lkr` with its tail as `NGSIM_POLICY_TAIL` reproduces it). The string is `policy_example` of `results/ngsim_lambda_cal_20261019/replay_out/fin_npw3_iso_side_20261112/run_info.json` (λ = 0; the second-to-last row of §4's current-design table with `ctx<λ>` = `ctx0`; pointer updated 2026-11-12 night: it read "the last row" until the final design's row was added):

```
acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx0_latgate_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10
```

The replay's default condition (no variable set; `results/ngsim_lambda_cal_20261019/replay/ngsim_replay.py`, `baselines.py`) (updated 2026-11-12 night: unchanged by the switch to the final design, and the condition of its runs):
- set speed = the logged follower's (`NGSIM_V_SET=logged`),
- start gap d_safe + 3 m (`NGSIM_SPAWN_MARGIN=3`),
- previous command 0 at the start (`NGSIM_START_CMD`),
- ideal ego without lag (`NGSIM_EGO_LAG_TAU` 0),
- speed bound 20 m/s (`NGSIM_V_MAX`),
- `NGSIM_LANE_TEST=path`,
- `NGSIM_SIDE_ENTRY=live`,
- and the unified ISO limits for every controller (`NGSIM_ISAMPC_POLICY=acc_isampc_amin3.5_amax2_jlim2.5_jlimup5`, `NGSIM_PCACC_POLICY=acc_pcacc_amin3.5_amax2_jlim2.5_jlimup5`).

Older roots and their settings: `docs/agents/ngsim_replay_defaults.md`.

What the token groups do (details §4, full records `docs/agents/tokens.md`) (updated 2026-11-12 night: P1's groups; the final design above keeps every group except the three lines marked below):
- `stdan_3int`, `modes8`: STDAN with three intentions; at most 8 joint modes. `fixed_risk`, `nominal_safe_distance`: labels only.
- `chancecells`: the cut-in cells are chance cells, the cells the price and the floor below act on (`cutin_chance0.6` renamed).
- `dopt3lkr`: on a cut-in cell the time gap τv is priced at 3·P_j·shortfall² instead of hard; `lkr` prices a vehicle's lane-keeping cells the same way once it has had a cut-in cell, until it leads and the gap has held the standoff for 0.5 s. (updated 2026-11-12 night: P1 only; the final design runs plain `dopt10` under `onerow`, every active cell priced.)
- `fullfoot`, `floorrisk0.1`, `floorhyp`, `nopricesigma`: the priced row asks for d_safe on the mean gap; the hard floor is L + d0 + z·σ with z = Φ⁻¹(1 − 0.1/p), so p·P(gap < L + d0) ≤ 10 % per hypothesis (no floor where p ≤ 10 %). (updated 2026-11-12 night: P1 only, except `fullfoot`, which the final design keeps; it has no floor row.)
- `bcts`, `occin0.1`: STDAN's intention probabilities calibrated by BCTS; a cut-in cell is active at step k where p·P_in(k) > 10 %.
- `trajcutin`, `nolkentry`: an adjacent vehicle's lane-keeping and away trajectories never lead, so only its lane change toward the ego lane makes cut-in cells (`trajcutin` then finds no entering lane-keeping trajectory).
- `ctx0_latgate`: context trust at λ = 0, i.e. STDAN's probability unchanged.
- `nocutout`, `lksigma`, `lkonesided0.05`, `lkhold`, `lkrsigma`: the ego-lane lead has no cut-out mode; its lane-keeping row is d_safe + 1.645σ (one-sided, 5 %) and stays active through lateral wobbles; `lkr`'s release also asks for that σ term. (updated 2026-11-12 night: the final design keeps `nocutout` and `lkhold`; its lead row is `onerow`'s, d_safe + 1.645σ − short from `zconf0.95` at p = 1, priced.)
- `rmove1000_rjerk40`, `jinput_ra0`: the jerk is the MPC input, priced 40·j² (×3, §3); no cost on the acceleration's size; `rmove` has no role with `jinput`.
- `l1slack1e6`: the emergency slack is an exact ℓ1 penalty, so ξ > 0 means that the hard rows have no solution.
- `leadmerge`, `mergefirst`: joint modes with the same lead sequence are one scenario, merged before the `modes8` cap.
- `safelayer`, `entrant2`, `fsdecel8_fsjerk10`: PC-ACC's safety layer around the command (fail-safe −8 m/s² at jerk 10 m/s³, path members assumed to brake at no more than −10.5 m/s²), also covering close entrants braking at −2 m/s².
- `amin3.5_amax2_jlim2.5_jlimup5`: comfort limits a ∈ [−3.5, 2] m/s², falling jerk ≤ 2.5 m/s³, rising jerk ≤ 5 m/s³ (ISO 15622).

**CARLA design** (the row "D-opt, current (2026-11-03 night, with `lkhold`)" of §4; every new CARLA run is to use it): `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6_leadmerge_mergefirst_lkrsigma_lkhold`.
- It has never run in CARLA as a whole.
- CARLA has run the paper's four controllers, D-opt (`dopt100lkr_trajcutin`) with its `ctx<λ>`, `latgate`, `rmove1000_rjerk40` and `rmove0_jlim1.5` variants, and the baselines `acc_pcacc` and `acc_isampc` (sweeps up to 2026-10-22).
- Every token added since then has run on the NGSIM replay only:
  - `entrypass0`, `fullfoot`, `l1slack1e6`, `leadmerge`, `mergefirst`, `lkrsigma`, `lkhold` (each "not run in CARLA" in its record)
  - and `nocutout`, `lksigma` (no CARLA sweep root carries them).
- The NGSIM design's other tokens are not decided for CARLA (`jinput`, `safelayer`, `entrant2`) or refused by the CARLA agent (`bcts`, `occin`). (updated 2026-11-12 night: the final NGSIM design's `onerow`, `zconf0.95`, `nocap` and `snap160` have not run in CARLA either; the agent parses them.)

**Pending decisions** (the maintainer's):
- `l1slack100` in place of `l1slack1e6`: run on the final set on 2026-11-12 (`replay_out/l1w_20261112/result.md`). (updated 2026-11-12 night: run on P1; the final design keeps `l1slack1e6`, and `l1slack100` has not been run with it.)
- `onerow` (updated 2026-11-12): one priced row on every active lead cell, no σ term, no floor row, no hard lane-keeping row (3d9bcf8). The maintainer decided on 2026-11-12 that it will be the final design; its weight w is being tuned (w 3 and w 100 run; w sweep in progress, 2026-11-12). Not the replay's default yet. Record: `docs/agents/tokens.md` (`onerow`), `replay_out/onerow_20261112/result.md`. (updated 2026-11-12 night: decided, not pending: the final design above, `onerow` with w 10 + `zconf0.95` + `nocap` + `snap160`, the replay's default; it settles the w choice, w 10, and arm A against arm C, C with `snap160`.)
- `passmiqp<m>` / `passnocap`: not adopted, decision pending (`replay_out/passmiqp_20261104/result.md`, `replay_out/passnocap_20261104/result.md`, `replay_out/passmiqp_20261104/discussion_20261104.md`).
- `occin` ε: 10 % in the design since NL (2026-11-10 evening); the ε comparisons are `replay_out/bcts_20261106/result.md` (0.02 / 0.05 / 0.1) and `replay_out/pcal_o5_20261106/result.md` (O10 vs O5) (updated 2026-11-12: decided, not pending: the maintainer fixed ε at 10 % with NL on the evening of 2026-11-10, `docs/agents/design_history.md` 2026-11-10; the two files are the comparisons behind it).
- `cvar<ε>_<c>`: not adopted, decision pending (`replay_out/cvar_final_20261106/result.md`, `replay_out/cvar_wide_20261107/result.md`).
- `termcost<c>`: every c > 0 failed the pre-registered rule; not adopted, decision pending (`replay_out/termcost_20261102/result.md`).
- `entrypass9` in place of `entrypass0` (the CARLA string; entrypass left the NGSIM design on 2026-11-11): decision pending (`replay_out/entrypass9_20261101/result.md`).
- The ds3_v1780 cut-in contact of the calibrated designs (it stays in NL): the maintainer's review before the paper's tables switch (`replay_out/pcal_cvar_20261106/result.md`, `replay_out/nps_lk_20261110/result.md`).
- CARLA: whether `jinput`, `safelayer` and `entrant2` (and the NGSIM design as a whole) enter the CARLA design.
- `aheadonly` (2026-11-13): decided by the maintainer (obstacles are only the vehicles ahead of the ego, as in ISA-MPC) and implemented (793aba3, `docs/agents/tokens.md`); pending: the final-set run and the replay default switch. The CRC calibration of the BCTS shift λ (`replay_out/bcts_shift_20261112/`) runs on the replay default, so its grid is to be re-run after the switch.
- Side entries of the open-loop replay (2026-11-13, analysis only): of the final root's 43 side entries (40 runs), 19 are TVs that entered ahead of the logged human but beside a simulated ego 2.5–25 m ahead of the human, and 24 (21 not the TV) entered behind or beside the human beside an ego 2–23 m behind it. A time-varying set speed equal to the logged speed was judged against: it would bind on ≥ 1 s in 36 % of cut-ins before t_cross and 62 % of no-cut-ins. The candidates are a gap-acceptance hold for entering vehicles, the start at the logged speed, or reporting only; the choice is the maintainer's.
- Smaller open choices (R1, `floorrisk0.3`, q_v, the entry-drift fixes, the PC-ACC appendix presentation, the t_comp timing): §10 (updated 2026-11-12: `floorrisk0.3` is moot, `floorrisk<ε>` with `floorhyp` has been in the NGSIM design since 2026-11-10, at ε 0.1 since NL) (updated 2026-11-12 night: P1 had `floorrisk0.1`; the final design has no floor row).

The record of every design decision since 2026-10-17 (D-opt's definition, each adoption with its measured effect, every older design string): `docs/agents/design_history.md`.

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
| `scripts/carla/utils/safety_layer.py` | Token `safelayer`: PC-ACC's safety layer (Prop. IV.5) around the MPC's command, on measured states only (§4). The perception is PC-ACC's: `pcacc_agent.observe_preceding_vehicles` in CARLA, `PCACCReplay.observe` in the replay. `SafetyLayer.step` = `classify` (membership history and cut-in set) + `decide` (check (7), fail-safe, ICS), split on 2026-11-03 for `safebranch`, which classifies before the solve; bit-identical |
| `scripts/carla/policies/isampc_agent.py` | Baseline `acc_isampc`: Zhou, Olofsson & Frisk's ISA-MPC restricted to lane keeping, on the IAIMM-KF port (§4) |
| `scripts/carla/run_all_scenarios.py` | Maps `--ego-policy` prefixes (`acc_nair_smpc`, `acc_isampc`, `acc_pcacc`) to the runner's `policy_type` |
| `scripts/carla/scenarios/run_lk_scenario.py` | The scenario runner: spawns actors, steps CARLA, records results |
| `scripts/carla/devel/sweep_acc_scenarios.py` | Grid sweeps. `--kind` selects the scenario family (the kinds are built in `acc_scenario_factory.py`). `--cells-json` runs a given list of scenarios instead of the grid, e.g. randomly drawn ones for calibration; the other grid flags must then be single-valued |
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
| `tests/` | 39 unit-test modules (487 tests): controller port, chance/cut-out extension, cut-out trajectory mapping, fixed-lane agent, ideal actuator, LSTM adapter, STDAN integration, context trust `ctx`, PC-ACC, ISA-MPC, acceleration lag `alag`, jerk input `jinput`, `relbrake`, `l1slack`, `termcost`, `safelayer`, `floorrisk`, `leadmerge`, `safebranch`, `mergefirst`, the fallback sign, `lkrsigma`, `lkhold`, the fail-safe exit (`failsafe_recovery`), close entrants (`entrant<b>`), the pass / yield MIQP (`passmiqp<m>`), the pass / yield MIQP without the cap (`passnocap`), the occupancy activation (`occin<eps>`), the CVaR bound (`cvar<eps>_<c>`), the probability calibration (`pcal`), its replacement BCTS (`bcts`), the lane-keeping entry (`nolkentry`), the switch-on by P_in alone (`pingate`), the floor quantile of the hypothesis probability (`floorhyp`), D's σ term at the floor's quantile (`floortop`), the lane-keeping σ term one-sided (`lkonesided<ε>`), the priced row without σ (`nopricesigma`), the chance switch's new name (`chancecells`), the unified limits (`amin<x>` `amax<x>` `jlimup<x>` `fsdecel<x>` `fsjerk<x>`, ISA-MPC's and PC-ACC's, `test_iso_limits.py`) (updated 2026-11-12: 40 modules and 494 tests since `tests/test_onerow.py`, the one-row design `onerow`, 3d9bcf8; the full suite ran 494 tests OK, `replay_out/onerow_20261112/full_test_suite.out`) (updated 2026-11-12 night: 43 modules and 509 tests since `tests/test_zconf.py`, `tests/test_nocap.py` and `tests/test_snap.py`, the tokens `zconf<β>` 7ef3da2, `nocap` 506bfcb and `snap<r>` f1bfbc4; the full suite ran 509 tests OK, `replay_out/onerow_20261112/full_test_suite_snap.out`) (updated 2026-11-13: 44 modules and 516 tests since `tests/test_aheadonly.py`, the token `aheadonly` 793aba3; the full suite ran 516 tests OK, `replay_out/aheadonly_20261113/full_test_suite.out`) |
| `results/` | Sweep outputs. `results/*/` is gitignored, so sweep data and launch scripts exist only on the maintainer's machine |
| `docs/agents/` | Full records moved out of this file on 2026-11-12, unchanged: `tokens.md` (every §4 token row and the baselines' notes), `design_history.md` (the design decisions since 2026-10-17, every row of the current-design table), `open_items.md` (§10), `ngsim_replay_defaults.md` (the replay's default history and how to reproduce older roots), `controller.md` (§3's fail-safe exit), `check_split.py` (checks that every line of the old file was kept) (updated 2026-11-12: moved verbatim, then reformatted for readability the same day with the words unchanged, and since then marked in place where stale; `_reformat/check_*.py` check that the readability pass kept every sentence, reading the files at commit 38a1b46) |

## 3. The controller in one page

- **State and dynamics.** The state is x = [s, v] (Frenet longitudinal position and speed), and the input is u = a. The prediction model is a double integrator (`acc_dynamics_matrices`).
- **Timing.** The MPC step is dt = 0.2 s, and every sweep uses horizon N = 15 (`--ego-horizons 15`; `original_nair_acc_config()` defaults to 10). CARLA runs at 20 Hz, so a new command is issued every 0.05 s.
- **The paper's tex differs from the code in four places.** `overleaf/formulation_twosided_conditional.tex`:
  1. writes a three-state model x = [s, v, a] with a first-order acceleration lag τ_a; the code has it only with `alag<τ>` (§4, since 2026-10-29), off by default
  2. applies the q(β)σ term to every mode; the code applies it only to cut-in modes (with `lksigma`, also to the ego-lane lead's lane-keeping cells)
  3. does not state the 0.35 floor on the cut-out scale
  4. does not state the 0.1 activation gate

Defaults come from `original_nair_acc_config()` plus the agent's settings:

| Parameter | Value | Note |
|---|---|---|
| speed bounds | 0 ≤ v ≤ 20 m/s | Upstream's 15 was below the scenarios' desired speeds. Because of it, every ego-16 run in the 2026-09-02 sweep collided |
| acceleration bounds | −3 ≤ a ≤ 2 m/s² | |
| jerk limit | 5 m/s³, hard, also applied to the issued command | Upstream's 1.5 was too slow for cut-ins with a TTC of 1.0–1.3 s. It was 10 until 2026-09-09 |
| weights | q_s 0, q_v 1, r_a 1, r_jerk 10, slack 5000 | `a_ref = 0`, so `r_a` penalises command size; the cached feedback QP counts the tracking and input terms three times (without K-feedback: one rollout weighted ×3 since 2026-10-26, three identical rollouts before; same commands, max |Δa_cmd| 8e-4 m/s² over 400 NGSIM runs, QP time median 77 → 56 ms), so against slack 5000 and the `dopt` price w the effective weights are 3·q_v, 3·r_a, 3·r_jerk, 3·r_move |
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
  - The constraint is soft: slack ≥ 0, with cost 5000·slack² (with `l1slack<w>`, in the current design as `l1slack1e6`, the exact penalty w·slack, §4).
  - At a step where a mode has no lead in the ego lane (`active_mask = False`), the cached QPs keep the row but place the lead 10⁶ m ahead, so the row never binds.
- **Uncertainty.**
  - Unlike upstream, the TV's prediction noise is not propagated through closed-loop dynamics.
  - σ combines the predicted lead-position variance with the ego's process noise, so σ ≥ 0.164 m even when the predictor reports no variance.
  - σ enters the constraint above, and also the reference gap that shapes `s_ref` and `v_ref`.
  - With `with_k`, ±1σ lead samples also feed the K feedback.
- **Fallback.**
  - If the solve raises (e.g. Gurobi `INF_OR_UNBD`), `_fallback_policy` issues a jerk-limited rule-based command, and the step logs `solve_path == "fallback"`.
  - Every solve that does not end OPTIMAL raises in CasADi's `Opti.solve` and falls back: checked on 2026-11-03 for `TIME_LIMIT` (the agent's 0.15 s limit) and `INF_OR_UNBD` with CasADi 3.7.2 (until then this file said other statuses only appear in `solver_message`).
  - `_fallback_policy` aims at `a_ref + 0.25·min(0, gap)`, braking 0.25 m/s² per metre of standoff shortfall, since 2026-11-03 (cf48239, `tests/test_fallback_sign.py`); before, the sign was flipped and it accelerated toward a lead that was too close. It was not used on the final sets, so their roots are unchanged.
  - **Exit from the safety layer's fail-safe** (`NairACCConfig.failsafe_recovery`, set with `safelayer`, 2026-11-04): the MPC plans the release from the fail-safe's acceleration at the fail-safe's jerk. Full record: `docs/agents/controller.md`.
  - Without a working Gurobi, every step falls back.

## 4. The policy string is the configuration

Sweeps pass `--ego-policy <string>`. The `_parse_*` methods of `ACCNairSMPCAgent` (in `acc_nair_smpc_agent.py`, starting around line 1186) read tokens from it by substring or regex match.

Every token's full record (results, checks, commits) is its subsection, under the same name, in `docs/agents/tokens.md`. Status: **design** = in the NGSIM design (§1); CARLA string = in the CARLA design (§1). (updated 2026-11-12 night: the NGSIM design is the final design of §1; "P1 only" = in P1, the previous NGSIM design, and not in the final one.)

| Token | What it does | Status |
|---|---|---|
| `const_lead` | No predictor: constant-velocity lead (SCC), 1 mode | paper (SCC) |
| `lstm`, `lstm_vel` | Single-mode LSTM predictor (position / velocity output) | paper (`lstm`) |
| `stdan_3int` | 3-intention STDAN predictor | **design** |
| `iaimm` | IAIMM-KF predictor, knobs from `ACC_NAIR_IAIMM_CFG` | IAIMM-KF comparison |
| `modes<N>` | Cap on joint modes | **design** (`modes8`) |
| `best_mode` | Keep only the most likely mode | option |
| `cutin_chance<β_ref>` | Proposed confidence chance constraint (§1) | paper (`cutin_chance0.6`); CARLA string |
| `chancecells` | `cutin_chance0.6` under another name | **design** |
| `cutin_gate<p>` | Probability below which a cut-in mode stops leading (default 0.1) | option; `occin` replaces it in the design |
| `sigmacap<c>` | σ term takes min(p_j, c) instead of min(p_j, β_ref) | option |
| `dopt<w>` | Cut-in cells: α_j scales only L + d0; the time gap τv is priced at w·P_j·shortfall² | **design** (as `dopt10`, with `onerow`) (updated 2026-11-12 night: it read "as `dopt3lkr`", P1's) |
| `dopt<w>lk` | `dopt<w>` that also prices the ego-lane lead's lane-keeping cells | superseded by `dopt<w>lkr` |
| `dopt<w>lkr` | Prices lane-keeping cells only of a vehicle that has had a cut-in cell, until it leads and has held the standoff 0.5 s | P1 only (`dopt3lkr`); CARLA string (`dopt100lkr`) (updated 2026-11-12 night: **design** until the final design; `onerow` has no `lkr`) |
| `lkrsigma` | `lkr`'s release test also asks for the step-0 σ term | P1 only; CARLA string (updated 2026-11-12 night: **design** until the final design, which refuses it) |
| `lkhold` | The lead's lane-keeping mode stays active through lateral wobbles | **design**; CARLA string |
| `pricep2` | Price w·P_j·p_cell·short² instead of w·P_j·short² | not adopted (dropped 2026-10-27) |
| `trajcutin` | An adjacent lane-keeping trajectory that enters the ego lane is a cut-in mode of its own | **design** (finds nothing under `nolkentry`); CARLA string |
| `entrypass`, `entrypass<m>` | A cut-in mode entering beside or behind the planned ego never leads | CARLA string (`entrypass0`); out of the NGSIM design since 2026-11-11; `entrypass9` pending |
| `passmiqp<m>` | Pass / yield of an entering hypothesis as a binary of one MIQP per tick | not adopted, pending |
| `passnocap` | `passmiqp<m>` without the acceleration cap | not adopted, pending |
| `pcal` | Calibrated lane-change probabilities (logistic on the lateral state) | superseded by `bcts` |
| `bcts` | Bias-corrected temperature scaling of STDAN's intention probabilities | **design** (NGSIM only) |
| `nolkentry` | Adjacent lane-keeping and away trajectories never lead | **design** |
| `occin<ε>` | A cut-in cell is active at step k iff p·P_in(k) > ε | **design** (`occin0.1`, NGSIM only) |
| `pingate` | Switch-on by P_in(k) > ε alone; p sets the floor | dropped 2026-11-11 |
| `cvar<ε>_<c>` | Per-step CVaR bound on the cut-in shortfall in place of the price | not adopted, pending |
| `nocutout` | No cut-out mode for an ego-lane vehicle | **design**; CARLA string |
| `lksigma` | σ term on the ego-lane lead's lane-keeping cells | P1 only; CARLA string (updated 2026-11-12 night: **design** until the final design, whose lead row gets 1.645σ from `zconf0.95`) |
| `fullfoot` | A cut-in cell keeps the whole footprint L + d0 (factor 1, not α_j) | **design**; CARLA string |
| `relbrake<b>` | Relative-speed (RSS-type) safe distance on every lead cell | not adopted |
| `l1slack<w>` | Emergency slack costs w·Σξ (exact penalty) instead of 5000·Σξ² | **design** (`l1slack1e6`); CARLA string; `l1slack100` pending |
| `termcost<c>` | DARE terminal cost-to-go, with `jinput` | not adopted, pending |
| `safelayer` | PC-ACC's safety layer around the command, on measured states | **design** (NGSIM); not decided for CARLA |
| `safebranch` | The safety layer's fail-safe as a branch of the scenario tree | dropped 2026-11-11 |
| `entrant<b>` | The safety layer also checks close entrants, assumed braking at −b m/s² | **design** (`entrant2`) |
| `floorrisk<ε>` | A cut-in floor as the chance constraint P·P(gap < L + d0) ≤ ε | P1 only (`floorrisk0.1`) (updated 2026-11-12 night: **design** until the final design, which has no floor row) |
| `floorhyp` | The floor's quantile uses the hypothesis probability p | P1 only (updated 2026-11-12 night: **design** until the final design, which has no floor row) |
| `floortop` | The priced row's σ term at the floor's one-sided quantile | not adopted |
| `lkonesided<ε>` | The lead's lane-keeping σ term one-sided, z = Φ⁻¹(1 − ε/p) | P1 only (`lkonesided0.05`) (updated 2026-11-12 night: **design** until the final design, whose lead row gets 1.645σ from `zconf0.95`) |
| `nopricesigma` | Priced rows ask for D on the mean gap; σ only in the floors | P1 only (updated 2026-11-12 night: **design** until the final design, which refuses it) |
| `leadmerge` | Joint modes with the same lead sequence are one scenario | **design**; CARLA string |
| `mergefirst` | Merge before the `modes<N>` cap | **design**; CARLA string |
| `branchdec<a>` | Splits a cut-in from lane keeping at the TV's decision point | option (no metric changed) |
| `ctx<λ>` | Mixes the cut-in intention with the measured lateral motion | in both strings; λ = 0 on NGSIM (no effect) |
| `latgate` | Latched lateral gate in place of `ctx`'s offset ramp | in both strings (inert at λ 0) |
| `rmove<w>`, `rjerk<w>` | Weight on the step-0 command change; jerk weight | **design** and CARLA string (`rmove1000_rjerk40`) |
| `ra<w>` | Weight r_a on the command's size (default 1) | **design** (`ra0`) |
| `alag<τ>` | First-order acceleration lag in the prediction model | option (NGSIM table 2026-10-29 afternoon only) |
| `jinput` | The jerk as the MPC input | **design** (NGSIM); not decided for CARLA |
| `jlim<x>` | Hard jerk bound (the falling side when `jlimup` is set) | **design** (`jlim2.5`) |
| `amin<x>`, `amax<x>`, `jlimup<x>` | Unified comfort limits (ISO 15622) | **design** (replay default since 2026-11-12) |
| `fsdecel<x>`, `fsjerk<x>` | The safety layer's fail-safe deceleration and jerk | **design** (`fsdecel8_fsjerk10`) |
| `open_loop`, `multimodal_ol`, any `_ol` | Open-loop variant | option |
| `no_k`, `with_k` | K-feedback off / on | option (off by default) |
| `scalar_chance`, `optimized_eta`, `probability_weighted` | Chance-constraint and risk-allocation variants | not used in the paper |
| `brake_distance`, `brake_hard_band` | Braking-distance safety bound | experimental |
| `cutin_ramp<β>`, `cutin_tlc<s>`, `gap_recover<s>`, `intent_lpf<s>`, `track_smooth<n>` | Other options | off by default, not in the paper |
| `fixed_risk`, `nominal_safe_distance` | Labels only | labels only |
| `onerow` | One priced row gap ≥ L + d0 + τv − short (w·P_j·short², `dopt<w>`'s w) on every active lead cell, cut-in hypotheses and lane-keeping leads alike; no floor row, no separate hard lane-keeping row, no `lkr`; the reference at plain d_safe | **design** (with `dopt10`, since 2026-11-12 night) |
| `zconf<β>` | With `onerow`: the row's σ term z·σ, z = Φ⁻¹(min(p, β)), p the cell's own hypothesis probability | **design** (`zconf0.95`) |
| `nocap` | With `onerow`: no upper bound on the shortfall, so no row is hard | **design** |
| `snap<r>` | With `jinput`: cost 3·P_j·r·Σ(j_k − j_{k−1})² on the jerk's changes, j_{−1} from the issued commands | **design** (`snap160`) |
| `aheadonly` | ISA-MPC's obstacle rule: only a vehicle whose current position is at or ahead of the ego's is an obstacle; one beside or behind never leads | decided 2026-11-13; implemented, not in the replay default yet (final set not run) |

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
| D-opt, current (2026-11-03 night, with `lkhold`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6_leadmerge_mergefirst_lkrsigma_lkhold` |
| D-opt, NGSIM design (2026-11-11 night, NP w3 without `safebranch`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx<λ>_latgate_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05` |
| D-opt, NGSIM design (2026-11-12, unified limits; P1) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx<λ>_latgate_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10` |
| D-opt, NGSIM final design (2026-11-12 night: `onerow` w 10 + `zconf0.95` + `nocap` + `snap160`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx<λ>_latgate_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold_entrant2_bcts_occin0.1_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_nocap_snap160` |

Every new CARLA run uses the row "D-opt, current (2026-11-03 night, with `lkhold`)" and every new NGSIM run the last row (the replay's default, §1). All 25 rows of this table (updated 2026-11-12 night: 26 with the final design's row, the last one, which every new NGSIM run uses since that night; P1 is the second-to-last), and the note on which row identifies which older root, are in `docs/agents/design_history.md`.

Compare it with the paper's controllers only on runs made with the same code (§8.11). The same-code roots `cutin_postfix_cmp_20261016` and `cutout_postfix_cmp_20261016` hold it next to Proposed, Multimodal SCC and the other D-opt variants.

Two external baselines (added 2026-10-19, smoke-tested on one run each, no sweep yet) (updated 2026-11-12: both have since run in the CARLA verification sweep of 2026-10-20 and on the NGSIM final set; `docs/agents/open_items.md`, "Verification sweep" and "Final NGSIM paper set and checkpoint"). They are separate ego policies, not `acc_nair_smpc` tokens; `run_all_scenarios.py` maps the prefix to the runner's `policy_type`, and the aggregator scores them like SCC runs:

| Label | Policy string | Notes |
|---|---|---|
| PC-ACC | `acc_pcacc` | Althoff et al.'s rule-based PC-ACC on the fork's d_safe and limits, no predictor; `acc_pcacc_paperdsafe_amin10_amax3_jlim10` is the faithful variant; the NGSIM replay reads its string from `NGSIM_PCACC_POLICY`. Full notes: `docs/agents/tokens.md` |
| ISA-MPC (cruise only) | `acc_isampc` | Zhou et al.'s planner with the lateral motion frozen, on the IAIMM-KF port, with the fork's d_safe; `qslack<w>` sets the slack weight; the NGSIM replay reads its string from `NGSIM_ISAMPC_POLICY`. Full notes: `docs/agents/tokens.md` |

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

**Several sweeps at once.**
- Give each line its own `--port`, 200 apart (2030, 2230, 2430, …), and its own output root.
- The watchdog counts and cleans up only the CARLA servers in `[--port, --port + 200)` (`--carla-port-span`), so lines never kill each other's servers; before 2026-10-15 it killed every CarlaUE4 on the machine, and lines that shared `--port 2030` crashed each other's servers.
- Set `OMP_NUM_THREADS=6` per line so the predictors do not oversubscribe the CPU.
- Measured on 2026-10-15 with four lines: a run takes 69 s instead of 59 s, so throughput is about 3.5×, and the results match serial runs (§8.11).

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

Runs made before 2026-10-27 no longer carry `summary.json` (deleted on 2026-10-27 to free disk, 94 GiB; it was `{error, ran_successfully, policy_logs}`, i.e. the pickle's per-actor `policy_log` after a JSON round trip plus the two flags that `metrics.json` also holds; 4 spawn-failure runs without a pickle keep theirs).
- The tools read `scenario_result.pkl` + `metrics.json` instead:
  - `analyze_acc_scenario_result.load_summary` rebuilds the same dict,
  - the animation tool and the older devel readers go through it,
  - and `sweep_acc_scenarios.py --resume` counts a run with `resolved_config.json` + `metrics.json` + `scenario_result.pkl` as finished (38ed197).
- New runs still write `summary.json`; the watchdog keeps it as a live activity marker.

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
12. **A predictor exception does not show up as a fallback.**
    - When `predict_acc` raises, `_stdan_prediction_bundle` catches it and runs the tick on the synthetic two-mode prediction (`lane_keeping` 0.6 / `cutin` 0.4, no vehicle ids), logging `stdan_debug.fallback = True` and `stdan_debug.reason`.
    - `solve_path` stays normal, so `fallback_ratio` is 0.
    - The aggregator's `pred_fallback_ratio` and `pred_error_steps` columns count these (since 2026-10-17; `pred_error_steps > 0` makes the run invalid).
    - The first `trajcutin` grid (2026-10-16, code `621d5ec`) hid such an exception in 108/108 cut-in runs, 6–9 ticks each, right before the TV was reclassified as ego-lane; the synthetic prediction also cleared `dopt<w>lkr`'s armed set, which is what made those runs look worse.
    - `fallback = True` without a `reason` means the predictor returned nothing (no vehicle in the grid), which is normal.
13. **Runs made before 2026-10-19 carry two run-start lateral artefacts.**
    - (a) The scripted agents' first IPOPT solve (`mpc_agent.py`) started from an all-zero initial guess, often hit the 0.2 s CPU limit, and its unconverged steering (at the ±0.5 rad bound) was applied:
      - the cut-in TV swerved up to 1.2 m sideways within the first second (7/50 TVs > 0.3 m under four parallel lines, 2–4/50 in the paper sweep, 0 after the fix).
      - STDAN read the swerve as a cut-in (p 0.78 at t0 + 0.25 s in 01/cutin_0003) and the ego braked for it.
    - (b) The 4 s cruise warm-up applied no steering, so a vehicle with a spawn heading error drifted straight: one right-lane car started t0 1.47 m off its lane centre.
    - Since 2026-10-19 the first solve starts from the current state and the reference (converges in 0.02–0.03 s under load) and the warm-up steers every vehicle onto its lane centre (`_tick_vehicles_cruising`, Stanley form; the shared `lane_keeping_steer` was lightly damped under the velocity pin).
      - Verified on 12 single runs (`startfix_check_20261019`): no infeasible TV tick, lateral motion ≤ 0.001 m, every vehicle ≤ 0.02 m off centre at t0, triggers unchanged, t_cross 0.05–0.15 s later.
    - The first second's predictor behaviour differs between old and new runs, so do not mix them in one table; every sweep root before this date, the paper's included, is "old".

## 9. Working conventions

- **Git.** Push to the `fork` remote (`hi-jun/SMPC_MMPreds`), never to `origin` (upstream `shn66`). Snapshot-commit any uncommitted work before editing code.
- **Use the existing tools rather than writing new ones.**
  - Add new metrics to `aggregate_cutin_table.py` instead of computing them ad hoc.
  - Make figures with `animate_acc_stdan_scenario.py`, with fixed `--accel-ylim` and `--speed-ylim` when panels sit side by side.
  - Wrap CARLA sweeps in `watch_acc_sweep.py`.
  - Keep one-off analysis scripts out of the repository.
- **Claims.** Report only improvements that were measured. Before comparing with an older sweep, re-aggregate both with the current script, because metric definitions have changed over time.
- **Changes.** Keep them surgical: touch only what the task needs, and match the existing style.
- **Sweep roots deleted on 2026-10-27 to free disk.**
  - 88 of the 120 roots under `results/acc_scenario_sweep/` (every superseded, invalid, aborted or uncited root of the inventory, 112 GiB) and the `summary.json` of the 32 remaining roots (94 GiB) were deleted; the disk went from 98 % to 74 %.
  - The numbers quoted in this file stay as text; the list of what was deleted and what was kept, and why, is in `results/acc_scenario_sweep/CLEANUP_PLAN_20261027.md` (§8, with `cleanup_executed_20261027.sh`).
  - `cutin_final_20260905` and `cutout_baselines_20260905` were kept because `overleaf/자소서_연구경력.md` cites them.

## 10. Open items

Full text: `docs/agents/open_items.md`; the NGSIM replay's default history: `docs/agents/ngsim_replay_defaults.md`. Decisions waiting for the maintainer: §1, "Pending decisions".

- **Safety and recursive feasibility.**
  - No formal property: the safety rows are soft and act on predicted hypotheses only, and the hypothesis set neither covers the real cut-in before the point of no return nor stays nested between ticks. (updated 2026-11-12 night: in the final design, `onerow` + `nocap`, no row has a hard part at all; the safety layer is the only hard protection.)
  - Implemented around it: `safelayer`, `entrant2`, the fail-safe exit (`safebranch` was dropped on 2026-11-11).
  - Open:
    - `passmiqp` / `passnocap` and how p and σ enter the pass condition
    - R1 (a hard first-step stopping row), `entrypass9`, `termcost`, `floorrisk0.3`, q_v, the entry-drift fixes, the PC-ACC appendix presentation, the t_comp timing (updated 2026-11-12: `entrypass9` matters only for the CARLA string, entrypass having left the NGSIM design on 2026-11-11; `floorrisk0.3` is moot, `floorrisk<ε>` with `floorhyp` being in the NGSIM design since 2026-11-10).
- **Entry timing of cut-in hypotheses.** Open loop, the TV leads a cell at its real entry step on only 31–59 % of pre-entry ticks; not fixed (`replay_out/entry_coverage_20261104/result.md`).
- **Calibrated lane-change probabilities.** `pcal` (2026-11-05) was superseded by `bcts` (2026-11-06 evening); open: the maintainer's review of the ds3_v1780 contact before the paper's tables switch.
- **Side entries (ghosts).** Since 2026-11-12 a lane change into the ego lane beside the ego stays live (`NGSIM_SIDE_ENTRY=live`); `ghost` gives the old rule (`replay_out/side_live_20261112/`).
- **Floor breach on NGSIM.** Bumper gap < 3 m to that tick's lead after t_cross, whichever vehicle it is (a definition, settled).
- **Computation time.** Quote only single-process timing (`replay_out/timing_noload_20261103/result.md`); whether the paper's t_comp column takes it is the maintainer's decision.
- **NGSIM replay plant lag.** `NGSIM_EGO_LAG_TAU` (default 0, the ideal ego); `alag` runs' standstill fallback on the ideal plant is not fixed, by the maintainer's decision.
- **NGSIM replay defaults.** Every default change since 2026-10-27 and how to reproduce older roots: `docs/agents/ngsim_replay_defaults.md`.
- **Final NGSIM paper set.** 889 episodes (489 cut-ins, 400 no-cut-ins), checkpoint `checkpoints_out30_no_tcross`, λ̂ = 0; open: the budget 0.5274 cannot be re-derived on the restricted λ-calibration population.
- **Context trust λ.** No λ adopted; no λ met both budgets on any data set; `phantom_cf` was withdrawn in favour of `phantom_ep` (clarified 2026-11-12: the paper's NGSIM runs and the replay's default run λ 0, `ctx0`, the plain policy; λ̂ = 0 on the final paper set).
- **Jerk pricing.** `rmove1000_rjerk40` adopted on 2026-10-22, chosen after reading test episodes (the test set is not strictly untouched).
- **CARLA verification sweep (2026-10-20).** Still open for ISA-MPC: ε (0.8 vs 0.1) and an 8 s horizon check (updated 2026-11-12: ε is settled for NGSIM at Zhou's published 0.8, ISA-MPC as published in the main table, maintainer 2026-11-05/06; still open for CARLA).
- **Code issues.**
  - The step-0 jerk cost uses `config.dt` instead of `command_dt` (16× too cheap)
  - unfiltered x₀
  - `slack_steps` misses slack use in chance policies
  - σ-term cap undecided (`sigmacap<c>`) (updated 2026-11-12: moot for the NGSIM design, which uses none of β_ref's effects since NL; still applies to the CARLA string's `cutin_chance0.6`) (updated 2026-11-12 night: still moot for the final design, whose only σ term is `zconf0.95`'s)
  - the TV re-trigger ignores its gap to the ego
  - paper tex vs code (§3)
  - the caption of `overleaf/table_cutin_final.tex`
  - IAIMM-KF tuned in-sample
  - `branchdec` with k_b = 1 untested.
  - The fallback sign was fixed on 2026-11-03 (updated 2026-11-12: closed, listed as a record only).
