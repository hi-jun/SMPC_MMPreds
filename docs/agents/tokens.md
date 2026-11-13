# Policy tokens: full records

Moved verbatim from AGENTS.md on 2026-11-12 (git `3d9bcf8`); the text is unchanged. Reformatted for readability the same day: the words are unchanged and long paragraphs became bullets (checks: `docs/agents/_reformat/check_*.py`). One subsection per row of the token table of AGENTS.md §4, then one per external baseline. The one-line summary of every token is AGENTS.md §4.

(updated 2026-11-12: stale statements have since been marked in place with dated "updated", "withdrawn", "clarified" and "pointer updated" notes, and the records of `onerow` and, that night, `zconf<β>`, `nocap` and `snap<r>` were added; the reformat checks read this file at commit 38a1b46. Clarified 2026-11-12: in these records "not in the paper" means not in the paper of 2026-09-10, `overleaf/` (its formulation and CARLA tables, AGENTS.md §1), whose tracked `.tex` files name none of the tokens added since; "the paper's NGSIM tables" are the NGSIM replay runs made for the paper from 2026-10-28 on (`open_items.md`, "Final NGSIM paper set and checkpoint"). A token can be in those tables and "not in the paper".)

## Tokens

### `const_lead`

No predictor: constant-velocity lead ("SCC"), 1 mode

### `lstm`, `lstm_vel`

Single-mode LSTM predictor, 1 mode

### `stdan_3int`

3-intention STDAN

### `iaimm` (written `iaimm_kf` in policy names)

IAIMM-KF. Its tuning knobs come from `ACC_NAIR_IAIMM_CFG`

### `modes<N>`

Cap on joint modes, e.g. `modes8`

### `best_mode`

Keep only the most likely mode

### `cutin_chance<β_ref>`

**Proposed** confidence chance constraint, e.g. `cutin_chance0.6`

### `chancecells`

- **What:** `cutin_chance0.6` under another name (2026-11-10 night, maintainer): the cut-in cells become chance cells (the cells that `dopt<w>` prices and `floorrisk<ε>` floors) and the σ terms are on, with β_ref 0.6.
- **Why:** The NGSIM design (NL) uses none of β_ref's effects (`nopricesigma` drops the priced row's σ term, `floorrisk<ε>` sets the floor's and `lkonesided<ε>` the lane-keeping one), so the number was dropped from the name.
- **Parsing:**
  - Whole-token match in `_parse_cutin_chance_ref` and `_parse_safety_constraint_mode`
  - ValueError together with `cutin_chance<value>`
  - every other string parser of the agent reads the two names alike (`tests/test_chancecells.py`, 3 tests).
- **Status:** The replay's default chance token since then (`NGSIM_POLICY_CHANCE`), bit-identical to the NL root on 13 smoke + 3 lead-only episodes (`replay_out/nps_lk_20261110/adopt_chancecells.log`)

### `cutin_gate<p>`

Probability below which a cut-in mode stops acting as a lead (default 0.1 for STDAN and IAIMM-KF)

### `sigmacap<c>`

With `cutin_chance` only: the σ term takes min(p_j, c) instead of min(p_j, β_ref); the standoff factor keeps β_ref (added 2026-10-07, not in the paper)

### `dopt<w>`

- **What:** With `cutin_chance` only: on cut-in cells the probability factor α_j scales the footprint L + d0 only, and the optimizer picks how much of the time gap τv to keep, paying w·P_j·shortfall² (`NairACCConfig.standoff_shortfall_weight`)
  - the standoff lies in [α_j(L + d0), α_j(L + d0) + τv].
- Cut-out cells keep Proposed's factor.
- **Status:** Cached feedback QP only (added 2026-10-07 with a fixed L + d0 floor, which 2026-10-15 replaced by α_j(L + d0); not in the paper)

### `dopt<w>lk`

- **What:** `dopt<w>` that also prices the shortfall on the ego-lane lead's lane-keeping cells (`NairACCConfig.standoff_shortfall_on_lk`, cells marked by `MultimodalLeadPrediction.lead_mode_lk`), so a cut-in keeps its price once it is in the lane instead of snapping back to the hard d_safe and the emergency slack.
- Cut-out cells still keep Proposed's factor.
- **Status:** `stdan_3int` only (added 2026-10-16, not in the paper)

### `dopt<w>lkr`

- **What:** `dopt<w>lk` whose lane-keeping cells are priced only for a vehicle that has had a cut-in cell (from then on, also while it is still crossing the line), until it leads at step 0 and the gap to it has held the full standoff for 0.5 s (`NairACCConfig.standoff_shortfall_lk_cutin_only`, `NairACCSMPC._restrict_lk_pricing`)
  - every other in-lane lead keeps the hard standoff as in Proposed.
- Priced on every in-lane lead, `dopt<w>lk` settled 6–13 % inside d_safe behind the cut-out groups' slow second lead (added 2026-10-16, not in the paper)

### `lkrsigma`

- **What:** With `dopt<w>lkr` and `lksigma` only (ValueError otherwise): lkr's release test asks for the full standoff plus the step-0 cell's σ term q(β)σ (`NairACCConfig.standoff_shortfall_lk_release_sigma`), the gap the released cell's hard row requires
- **Why:** without it a release could leave that row short by q(0.6)·0.164 = 0.14 m or more (maintainer, 2026-11-03, final-check item 6).
- **Checks:**
  - without the token bit-identical (12 smoke + 3 lead-only)
  - the lead-only loop is unchanged with it (no cut-in mode, lkr never arms)
  - 4 tests (`tests/test_lkr_release_sigma.py`).
- **Result:** NGSIM final set (λ 0, ideal plant, `replay_out/lkrsigma_20261103/result.md`, root `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_20261103`, paired against the adopted design):
  - printed table cells unchanged except critical j_max 2.05 → 2.03 (SE 0.02)
  - step-0-only slack ticks 38 → 35 (critical) and 195 → 193 (mild)
  - no episode moves
  - 93 of 889 episodes change a command (at most 0.45 m/s²)
  - no time cost at equal load.
- **Status:**
  - Off by default
  - whole-token match (`_parse_lk_release_sigma`)
  - not run in CARLA, not in the paper
    (added 2026-11-03, dcad977;
    part of the current design since the night of 2026-11-03, maintainer's decision, §1;
    the replay's default tail ends `_lkrsigma`;
    paper root with the ghost fix `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_ghost_20261103`, tables `replay_out/lkrsigma_20261103/tables_paper_lkrsigma.md`)
    (updated 2026-11-12: the default tail has grown since and ends `_fsjerk10` since 2026-11-12, with `lkrsigma` still in it, `ngsim_replay_defaults.md`; "not in the paper" means the paper of 2026-09-10, see this file's header, while the paper's NGSIM tables have run with it since the night of 2026-11-03)
    (updated 2026-11-12 night: not in the final NGSIM design, the replay's default since that night, which `onerow` builds without `lkr` and refuses `lkrsigma` with; P1, the design before it, and the CARLA string keep it)

### `lkhold`

- **What:** The ego-lane lead's lane-keeping mode stays active at every step when its representative trajectory is no departure by the 3-step rule (`ends_outside_ego_lane`): a wobble out of the lane at a middle step or at the last one or two no longer switches the lead's cells off (`acc_postprocess.process_vehicle_prediction(lk_hold=)`).
- **Reason** (maintainer, 2026-11-03 night, final-check item 4): the rule's own comment says such a wobble must not relax the standoff, yet the cells at those steps were off, so the lead vanished from the plan there.
- **Checks:**
  - without the token bit-identical (12 smoke + ds5_v2609 + 3 lead-only episodes)
  - 5 tests (`tests/test_lkhold.py`).
- **Result:** NGSIM final set, only with the replay's `NGSIM_LANE_TEST=markings` (§10; `replay_out/lanefix_20261103/result.md`): the token changes no printed table cell against the same switch without it, and cuts the ticks where the lead's lane-keeping cells are off while it keeps its lane from 383 to 46.
- **Status:**
  - `stdan_3int` only, off by default
  - whole-token match (`_parse_lk_hold`)
  - not run in CARLA, not in the paper
    (added 2026-11-03, caa9b64;
    part of the current design since the night of 2026-11-03 with the replay's `NGSIM_LANE_TEST=path`, maintainer's decision, §1;
    the replay's default tail ends `_lkhold`)
    (updated 2026-11-12: the default tail has grown since and ends `_fsjerk10` since 2026-11-12, with `lkhold` still in it, `ngsim_replay_defaults.md`; "not in the paper" means the paper of 2026-09-10, see this file's header, while the paper's NGSIM tables have run with it since the night of 2026-11-03, `design_history.md`)

### `pricep2`

- **What:** With `dopt<w>`: a cell's shortfall costs w·P_j·p_cell·short² instead of w·P_j·short², p_cell being the probability of the cell's lead mode for its own vehicle (`MultimodalLeadPrediction.lead_mode_probability`, filled by `acc_postprocess.build_multitarget_lead_prediction`; `NairACCConfig.standoff_shortfall_cell_price`)
  - with one multi-mode target it is w·P_j².
- **Reason:** P_j also weights the scenario's tracking and jerk terms, so under `dopt<w>` it cancels inside a branch and a cut-in branch plans the same standoff however unlikely the cut-in is.
- **NGSIM probe**
  (2026-10-23, `replay_out/pricep2_probe_20261023` vs `pricep2_probe_base_20261023`;
  16 calibration episodes of `latgate_cal_rj40_20261022`: 8 S2b no-cut-ins with phantom_ep ≥ 0.3 at λ 0.4, 4 critical + 4 mild cut-ins;
  tail `_rmove1000_rjerk40`):
  - phantom_ep 1.04 → 0.97 (λ 0) / 0.97 → 0.90 m/s² (λ 0.4), per episode from 0.38 → 0.00 to 0.78 → 1.05
  - cut-ins unchanged (δ_max +0.0–0.7 %p, no collision).
  - One-tick re-solves on those 8 braking ramps (same state and controller memory): the cut-in modes add 0.48 m/s² to the planned braking over the next 0.6 s, `pricep2` removes 0.08 of it, and making the reference ignore the priced cells removes none.
  - In 4 of the 8 the hypothesised vehicle enters ≈ 2.8 s ahead at a point the ego would already have passed at its current speed (free gap −2.4 to −5.5 m, below the unpriced floor α(L + d0) + qσ of 1.3–6.9 m):
    the lead filter of `build_multitarget_lead_prediction` keeps every mode ahead of the ego's *current* position (`s_val < ego_s`), so the ego yields to a car it would pass, whatever the price.
- **With lower w** (2026-10-27, `replay_out/pricep2_wsweep_20261027/`: the 400 calibration episodes, λ 0.4, no entrypass, logged start command):
  - phantom_ep 0.466 m/s² at w 100 without `pricep2` → 0.434 / 0.396 / 0.392 / 0.386 at `pricep2` w 100 / 10 / 3 / 1,
    while critical δ_max goes 18.5 → 21.1 / 30.3 / 35.1 / 38.8 % and T_rec 0.42 → 1.34 s at w 1
  - in the same setting `entrypass0` alone gives 0.196.
  - The planned standoff does not become p-dependent: low-p cells often have no shortfall to price.
- **With `entrypass0` on** (2026-10-27, `replay_out/ep0_pricep2_wsweep_20261027/`, same setting, both entrypass anchors re-run with the current code) the price holds most of what remains, at a cut-in cost as large as without entrypass:
  - phantom_ep 0.196 m/s² at w 100 without `pricep2` → 0.148 / 0.065 / 0.051 / 0.049 at `pricep2` w 100 / 10 / 3 / 1 (−75 % at w 1, against −17 % without entrypass),
    while critical δ_max goes 20.0 → 22.4 / 34.1 / 39.7 / 43.4 % and, at w 1, T_rec 0.50 → 1.46 s and floor breaches (critical / mild) 6 / 1 → 9 / 9.
- **Status:** `stdan_3int` only, off by default (added 2026-10-23, commit 2db0c6c, not in the paper)

### `trajcutin`

- **What:** An adjacent vehicle's lane-keeping-labelled trajectory that enters the ego lane within the horizon becomes a cut-in mode of its own (own probability and trajectory; σ term, cut-in factor, gate) instead of a lane-keeping cell with the full, unscaled standoff.
- **Details:**
  - While a vehicle crosses the lane line its LK trajectory drifts into the ego lane 0.4–0.5 s before the vehicle is reclassified as ego-lane.
  - The labelled cut-in keeps its mode; nothing changes while no LK trajectory enters.
  - A vehicle whose every non-cut-in trajectory enters has no lane-keeping mode (that raised until 2026-10-17, §8.12).
  - The entering trajectory's mode is still a `cutin` for every constraint but is labelled `cutin/LK` (`ACCModePrediction.scenario_label`) in scenario names and effective-lead keys, so a vehicle's two cut-in modes are distinct scenarios
    (`acc_postprocess._mode_specs_for_relation`, `stdan_3int` only, added 2026-10-16, not in the paper)

### `entrypass`, `entrypass<m>`

- **What:** A cut-in mode (`trajcutin`'s `cutin/LK` included) whose vehicle enters the ego lane beside or behind the ego's *planned* position never leads:
  at the mode's first in-lane step k_e its bumper gap to the planned ego front, s_v(k_e) − (s_0 + Δs(k_e)) − L, must be ≥ −m (`entrypass` alone: m = 0; `entrypass9`: the whole vehicle behind the ego's rear, L + L_v).
- **Details:** Δs integrates the previous tick's planned speed of the most likely scenario in which that vehicle was not a cut-in, read from the time since that plan and held past its end
  - with no previous plan the current speed is held (`acc_postprocess.entry_ego_displacement`, `build_multitarget_lead_prediction(entry_plan=, entry_clearance=)`).
- **Why:** Without it the lead filter compares with the ego's *current* position (`s_val < ego_s`), so the ego yields to a car it would pass (see `pricep2`).
- **Logged** as `scenario_metadata[*].entry_pass` (agent) and the replay's tick arrays `n_entry_excl` / `entry_gap_min`.
- **NGSIM probe** (2026-10-26, `replay_out/entrypass_20261026/`: 400 calibration episodes, set speed 20, start at d_safe, λ 0.4 `latgate`, tail `_rmove1000_rjerk40`, paired with `vset20_base_cal_20261026`):
  - exclusions on 3.9 % / 2.4 % of scored cut-in / no-cut-in ticks
  - no-cut-in a_min −1.14 → −0.85 m/s² (+0.29, SE 0.05), runs ≤ −2 m/s² 37 → 14, j_max 2.71 → 2.03
  - mild cut-in a_min +0.35 (SE 0.09), j_max −0.59, floor breaches 6 → 1
  - critical cut-in δ_max +1.5 %p (SE 1.9, median 0), a_min +0.03, T_rec +0.08 s
  - no collision.
  - But in critical ds3_v1358 it dropped the real cut-in vehicle's hypothesis 0.8–0.3 s before t_cross (STDAN put the entry 2.6 s out; it crossed 0.8 s later) while the previous plan accelerated, min gap 3.11 → 2.47 m
    - with `pricep2` added, contact (−0.30 m).
  - `entrypass9` keeps that vehicle (min gap 3.11 m) but also gives up most of the gain: no-cut-in a_min +0.09 (SE 0.03), ≤ −2 m/s² runs 37 → 31, exclusions on 0.5 / 0.6 % of ticks.
  - `pricep2` adds little on top (one-tick re-solves: even a free time gap moves the first input by only 0.13–0.17 m/s² on a tick with a hypothesis).
- **Status:**
  - Not run in CARLA.
  - `stdan_3int` only, off by default in the code (added 2026-10-26, not in the paper)
  - since 2026-10-27 `entrypass0` is the default option for every new CARLA and NGSIM run (maintainer's decision, §1).
- **On the current design** (2026-11-01, maintainer's request, `replay_out/entrypass9_20261101/result.md`, root `fin_ep9_jinput_ra0_fullfoot_l1slack1e6_20261101`),
  `entrypass9` in place of `entrypass0` gave δ_max −0.8 / −1.3 %p (critical / mild, SE 1.1 / 0.7), no-cut-in phantom_ep 0.149 → 0.390 m/s², runs a_min ≤ −2 8 → 45, floor breaches 13 → 15 (5 fixed, 7 new, 5 of them pass → yield flips), contacts 0
  - not adopted (decision pending).
- Removed from the NGSIM design on the night of 2026-11-11 (NP, maintainer; `replay_out/noep_nl_20261111/result.md`)

### `passmiqp<m>`

- **What:** The pass / yield choice for an entering cut-in hypothesis made **inside the optimization** (final-check item 8, 2026-11-04):
  one MIQP per tick in the cached feedback QP (Gurobi through CasADi's `discrete` flags, `Threads` 1), one binary z per **candidate** hypothesis (vehicle + trajectory label, at most `PASS_MIQP_SLOTS` = 3 per tick), shared by every scenario the hypothesis leads in (one decision now).
- **The two branches:**
  - z = 1 yields: its cells are today's rows.
  - z = 0 passes: its cells are relaxed by big-M (`PASS_MIQP_BIG_M` 500 m), and instead
    - (i) the **pass row** at the first cell the candidate leads in each scenario, the ego's front at least m metres ahead of the vehicle's rear, s_k − s_v,k + L − m ≥ 0, hard (no slack)
    - (ii) the **next-lead rows** on those cells, the standoff to the nearest non-candidate lead (the ego-lane lead as a rule; the `next_*` cell arrays of `MultimodalLeadPrediction`), the same row form with that lead's factor and σ term, priced through the cell's own shortfall where that lead is a dopt cell (`lkr`'s rule applies to it), hard otherwise
    - (iii) the **acceleration cap**, no planned acceleration above max(a_0, 0) before the pass row's cell, so a pass is never bought with more acceleration than the ego already has (the cap trajectory bounds the plan's position step by step).
  - The optimizer takes the cheaper branch under the design's own costs
  - where the pass row cannot be met the solve yields, with the ℓ1 slack if the yield rows cannot be met either.
- **Candidates** (`acc_postprocess.build_multitarget_lead_prediction(pass_margin=, pass_slots=)`):
  the entry test of `entrypass` becomes a pre-filter on the ego's **no-acceleration displacement** (`pass_cap_displacement`: the current speed, the current acceleration held where positive, capped at v_max) in place of the remembered plan:
  - a cut-in mode whose entry gap is below −`PASS_MIQP_FAR_BEHIND_M` (9 m, the whole vehicle behind the ego's rear, `entrypass9`'s geometry) passes without a row (excluded as `entrypass` excludes),
  - one above −m yields (the pass row could not be met; today's handling, no binary),
  - one in between is a candidate (`pass_slot` on its cells)
  - beyond 3 candidates the 3 nearest the ego's front get the slots and the rest pass without a row (`overflow`).
- **Details:**
  - The reference follows the previous tick's decision: a passed hypothesis's cells take the next lead's reference (`NairACCSMPC._pass_reference_prediction`); the rows never read that memory.
  - `entrypass0` may stay in the string: its test is replaced on every cut-in hypothesis.
  - Ticks without a candidate solve the design's QP bit for bit (a separate cache entry, `_pass_rows`, as `entrant<b>`'s rows).
  - `leadmerge` / `mergefirst`: the slot and next-lead arrays are part of the class signature.
  - `safebranch` / `entrant<b>`: unchanged (the branch shares u₀ only; the cap rows are on the scenarios' states).
- **Code:**
  - `NairACCConfig.pass_miqp_margin`
  - whole-token match (`_parse_pass_miqp`; `passmiqp` alone m = 0)
  - needs `stdan_3int`, `cutin_chance` and `dopt<w>`
  - ValueError with `relbrake<b>`, `floorrisk<ε>` (fixed 2026-11-12: it read `floorrisk<eps>`), `alag<τ>`, `with_k`.
- **Logged:**
  - `NairACCSolution.pass_decisions` (vehicle, label, entry step, entry gap under the cap, slot, pass), `pass_n_pass`,
    `pass_accel_max` (the largest planned acceleration over the passed candidates' capped steps, above 0 only where a_0 > 0 was kept), `pass_solve_s`,
    `pass_iter_count` (Gurobi's node count is not exposed by CasADi)
  - `policy_log[*].pass_miqp` (agent)
  - NGSIM tick arrays `pm_n`, `pm_pass`, `pm_tv`, `pm_tv_gap`, `pm_accel`, `pm_ms`, `pm_iter` and results columns `n_pm_*`, `pm_ms_*`.
- **Timing:** MIQP solve time on a warm cache: 6.5 ms against the plain QP's 2.4 ms (one process, one scene).
- **Checks:**
  - without the token bit-identical (12 smoke + ds5_v2609, 3 lead-only, max \|Δa_cmd\| = 0)
  - 15 tests (`tests/test_passmiqp.py`).
- **What it guarantees:**
  - nothing new
  - on a pass the plan is certified against the hypothesis's *predicted* trajectory only (the TV does not react, STDAN's trajectory)
  - the safety layer's guarantee is unchanged.
- **Result:** NGSIM final set (λ 0, ideal plant, `NGSIM_LANE_TEST=path`, `replay_out/passmiqp_20261104/result.md`, root `fin_..._lkrsigma_entrant2_passmiqp_20261104` + `lo_`, paired against the paper root with `entrant2`):
  - critical δ_max 9.1 → 8.1 %, floor breaches 9 → 7, mild δ_max 5.9 → 5.2, floor breaches 3 → 1, no-cut-in phantom_ep 0.188 → 0.179 (SE 0.008), runs a_min ≤ −2 11 → 12, contacts 0, j_max −0.09 / −0.08
  - but all four removed breaches are runs where the passed TV merged beside the ego and the replay's ghost rule hides it (item 10):
    re-scored with side entries live once clear, critical 11.4 → 10.8 % and 14 → 13 breaches, mild 7.8 → 7.4 % and 8 → 7, no cut-in contacts 1 → 2
    (clarified 2026-11-12: "contacts 0" is the replay's own scoring; "no cut-in contacts 1 → 2" is the open-loop recount, in which the ego never reacted to the re-scored vehicles: the paper root has 1 such contact and this root 2, the added one ds4_v3159; critical stays 1 → 1, mild 0 → 0; `replay_out/passmiqp_20261104/result.md`)
    (updated 2026-11-12: since 2026-11-12 the replay keeps lane-change side entries live by default, `NGSIM_SIDE_ENTRY=live`; this root ran under the ghost rule and has not been re-run)
  - one no-cut-in near miss (ds4_v3015, 0.07 m) where the cap made the ego slower
  - MIQP ticks 3.0 % of ticks, TV candidates 731 pass / 10 yield ticks
  - warm MIQP ticks 136 ms against 60 ms (load-inflated), the first-use builds 1.8 s
  - 17 time-limit fallback ticks in 10 runs, all on plain-QP ticks under the added load.
- **Status:**
  - Not run in CARLA
  - off by default
  - not adopted (decision pending) (added 2026-11-04, 6e2e4dd; not in the paper)

### `passnocap`

- **What:** With `passmiqp<m>` only (ValueError otherwise; maintainer's request after the `passmiqp` result, 2026-11-04): no acceleration-cap rows on the pass branch, so the MIQP may plan to accelerate past a merging car within the design's own limits (a ≤ a_max, \|j\| ≤ jerk_limit, v ≤ v_max)
  - the pass row stays hard, so an unreachable pass still yields.
- **Details:** The candidate pre-filter reads the ego's **maximum-acceleration displacement**
  (`acc_postprocess.pass_maxacc_displacement`: the jerk limit from the current acceleration up to a_max on the controller's grid, v capped at v_max; `build_multitarget_lead_prediction(pass_plan=)`),
  so a pass reachable only by accelerating gets a binary (with the cap it yielded without one)
  - the far-behind exclusion keeps the no-acceleration displacement
  - the decisions carry `pass_gap_m` (the gap under the pre-filter's displacement) and the slots go by it.
- **Code:**
  - `NairACCConfig.pass_miqp_no_cap` (part of the cached-QP key)
  - whole-token match (`_parse_pass_no_cap`).
- **Checks:**
  - `passmiqp` alone bit-identical to its root (12 smoke + ds5_v2609, 3 lead-only)
  - 10 tests (`tests/test_passnocap.py`).
- **Result:** NGSIM final set (λ 0, ideal plant, `NGSIM_LANE_TEST=path`, `replay_out/passnocap_20261104/result.md`, root `fin_..._entrant2_passmiqp_passnocap_20261104` + `lo_`, paired against the paper root and the `passmiqp` root):
  - no-cut-in phantom_ep 0.188 → 0.138 (−0.050, SE 0.013), runs ≥ 0.3 71 → 56, runs a_min ≤ −2 11 → 8, v_avg +0.11 m/s
  - critical δ_max 9.1 → 8.1, floor breaches 9 → 3, mild 5.9 → 6.1 and 3 → 3 (verified 2026-11-12: critical δ_max is 8.1 % with `passnocap` as with `passmiqp`, not a copy error, `replay_out/passnocap_20261104/result.md`)
    (3 removed, 3 new: ds2_v1873 0.24 m and ds5_v2583 0.26 m, where the ego was accelerating at +1.5 / +1.1 m/s² toward the TV when it entered and the safety layer's entrant check issued the fail-safe on the entry tick, the first scored-tick action of the layer on the final set;
    ds1_v2879 2.71 m)
  - contacts 0
  - j_max −0.01 / +0.04 / −0.05
  - every removed breach is a passed TV that merged beside the ego and became a ghost (side-entry ghosts with the TV 4 / 16 → 13 / 22, critical / mild)
  - re-scored with side entries live once clear (item 10): critical 14 → 11 breaches, mild 8 → 10 and one re-scored contact, so pre-registered rule 2 fails
    (updated 2026-11-12: since 2026-11-12 the replay keeps lane-change side entries live by default, `NGSIM_SIDE_ENTRY=live`; this root ran under the ghost rule and has not been re-run)
  - MIQP ticks 5.7 % of ticks, pass / yield 4,859 / 1,548 ticks, the TV 1,218 / 495
  - the plan above a_0 on 60 % of pass ticks (median +0.26, p90 +0.94 m/s²), the executed peak on a TV pass median +0.54 (with the cap +0.25)
  - 18 load time-limit fallback ticks in 14 runs, all plain-QP ticks
  - warm MIQP ticks 119 ms against 60 ms (load-inflated).
- **Status:**
  - Not run in CARLA
  - off by default
  - not adopted (decision pending) (added 2026-11-04, 01720bc; not in the paper)

### `pcal`

- **What:** Every target's intention probabilities (LK, LLC, RLC) are replaced, before the mode mapping, by calibrated probabilities of each lane change within 3 s
  (`acc_postprocess.calibrate_intention_probabilities`, through `predict_acc(intention_calibration=)`; the predictor's own stay in `raw_intention_prob_uncal`):
  - per side a logistic model on logit p and the vehicle's measured lateral state, signed toward the side
    (the offset of the causal SG Local_X from its lane centre and the lateral speed over 0.5 s, restricted cubic splines; the net Lane_ID change over the last 3 s toward / away; logit p times each),
    fitted subject to the calibrated probability never falling as p rises
  - lane keeping takes the rest, 0 toward a side without a lane, a sum above 1 scaled to 1.
- **Reason** (maintainer, 2026-11-05): the chance-constraint designs assume a calibrated P_j, and STDAN's is odds-inflated 13–70× by its balanced training set and over-confident (`replay_out/pcal_reliability_20261105/result.md`)
  - calibrated with the lateral state, not by a constant, so that STDAN's post-lane-change carry-over (raw p ≥ 0.9 with 3–4 % observed) is handled.
- **Maps:** `predictor/stdan_3int_signed_tcross_velint/pcal_ngsim_20261105.json` (the paper checkpoint) and `predictor/iaimm_kf/pcal_ngsim_20261105.json` (the paper's ISA-MPC IAIMM-KF setting; the replay's `ISAMPC_PCAL=1` calibrates ISA-MPC's probabilities the same way),
  fitted on every 10 Hz frame of the 2,954 natural Val+Test vehicles of `out/p_S.npz` and cross-fitted by recording (the replay applies to an episode of recording k the map fitted without k).
- **Calibration quality:** Leave-one-recording-out, STDAN left / right:
  - NLL 0.141 / 0.097 raw, 0.045 / 0.025 p-only isotonic, 0.034 / 0.021 this map
  - ECE within lateral situations 0.010 / 0.004 p-only, 0.002 / 0.001 this map (`replay_out/pcal_model_20261105/result.md`).
- **Caveat:** Every probability threshold of the design (the 0.1 gate, β_ref 0.6 of the σ term, `occin`'s ε, `cvar`'s ε and c, `floorrisk`'s ε) was set on the uncalibrated scale
  - at rest a calibrated 0.1 is a raw p of about 0.93–0.97, and `trajcutin`'s entering lane-keeping trajectories now carry the large calibrated lane-keeping probability.
- **Checks:**
  - without the token (and without `ISAMPC_PCAL`) bit-identical (12 smoke + ds5_v2609 and 3 lead-only against the entrant2 roots, ISA-MPC against its paper roots)
  - with it, the probability used equals the map applied offline on every adjacent-TV tick of the 13 episodes
  - 11 tests (`tests/test_pcal.py`).
- **Smoke result:** 13 smoke episodes with the thresholds unchanged (an indication): no-cut-in a_min −1.62 → −0.25 at most, cut-in a_min −0.92 → −3.00 at most and one new floor breach, active cut-in hypotheses 449 → 111 target-ticks.
- **Code:**
  - NGSIM replay only (the agent refuses it)
  - the replay's tick array `p_uncal`
  - whole-token match (`_parse_probability_calibration`).
- **Status:**
  - Not run on the tuning or final set
  - off by default
  - not adopted (decisions pending, §10)
  - not in the paper (added 2026-11-05, 68d3829)
  - (updated 2026-11-12: run on the tuning set, `replay_out/pcal_tune_20261105/result.md`, and on the final set, `replay_out/pcal_cvar_20261106/result.md`; adopted with `occin0.1` into the NGSIM design on 2026-11-06 and replaced by `bcts` that evening; `design_history.md` 2026-11-06, `bcts` below)

### `bcts`

- **What:** Every target's intention probabilities (LK, LLC, RLC) are replaced, before the mode mapping, by bias-corrected temperature scaling (BCTS; Alexandari et al., ICML 2020):
  q = softmax(log p / T + (0, b_L, b_R)), fitted by the three-class log loss to STDAN's own training label (±28 frames of a Lane_ID transition, the nearest wins, US-101 Lane_IDs ≥ 6 one lane) on the 2,378,400 full-window frames of the 2,954 natural Val+Test vehicles, leave-one-recording-out (the replay applies the fold of the episode's recording)
  - no lane mask (the label keeps the direction for 2.8 s after a crossing into an edge lane, 3,692 frames; a first version zeroed that side and was corrected before any closed-loop run)
  - `acc_postprocess.bcts_intention_probabilities` / `load_bcts_calibration` through `predict_acc(intention_calibration=)` (the predictor's own stay in `raw_intention_prob_uncal`, the replay's `p_uncal`).
- **Files:** `predictor/stdan_3int_signed_tcross_velint/bcts_ngsim_20261106.json` (STDAN, all recordings T 1.047, b −3.031 / −3.850) and `predictor/iaimm_kf/bcts_ngsim_20261106.json` (the paper's ISA-MPC IAIMM-KF setting, T 1.578, b −2.291 / −3.491; the replay's `ISAMPC_BCTS=1`).
- **Held out**, STDAN, all lanes, left / right:
  - O/E 0.24 / 0.12 raw → 1.00 / 1.00, log-loss skill −0.06 / −0.84 → 0.62 / 0.57, AUC 0.962 / 0.961 → 0.964 / 0.962
  - TS alone leaves O/E 0.31 / 0.17
  - merge lanes left O/E 1.33.
- **Caveat:** The calibrated probability is that of the ±2.8 s label, not a forward-looking one (median 72 / 57 % 2 s after the crossing), and `trajcutin`'s entering lane-keeping modes carry q_LK.
- Checked against the official `abstention` `TempScaling(bias_positions='all')` (fold 1: T 1.0395 vs 1.0394, held-out max |Δq| 1.4e-4).
- **Checks:**
  - without the token (and `ISAMPC_BCTS`) bit-identical (12 smoke + ds5_v2609, 3 lead-only, ISA-MPC against its paper roots)
  - with it the probabilities read equal the calibration applied offline on every target and tick (13 episodes, 5,457 / 5,430 target-ticks, max error 0)
  - 10 tests (`tests/test_bcts.py`).
- Pipeline archived in `replay_out/bcts_20261106/fit/`
  - record `final_paper/appendix/calibration.md`.
- **Code:**
  - NGSIM replay only (the agent refuses it)
  - not with `pcal`
  - whole-token match (`_parse_bcts_calibration`).
- **Status:**
  - Replaces `pcal` in the NGSIM design (maintainer, 2026-11-06 evening, §1)
  - final set with `occin` ε 0.02 / 0.05 / 0.1 in `replay_out/bcts_20261106/result.md` (roots `fin_bcts_occin<ε>_20261106`, lead-only `lo_fin_bcts_occin0.05_20261106`, equal for every ε on the 3 check episodes)
  - ε 0.05 since 2026-11-10 (maintainer, with `pingate`), 0.1 since NL that evening (§1) (added 2026-11-06, db14f03; not in the paper yet)

### `nolkentry`

- **What:** An adjacent vehicle's trajectories other than its lane change toward the ego lane (lane keeping, the change away) are out of the ego's path at steps 1.. whatever the lane test says (`acc_postprocess.process_vehicle_prediction(no_lk_entry=)`), so they never lead and `trajcutin` finds no entering trajectory.
- **Reason** (maintainer, 2026-11-06 evening): with `bcts` + `occin` a lane-keeping trajectory whose mean never reaches the path became a 92 % cut-in mode through q_LK × P_in > ε (`out/anim_trajcutin_lkentry_20261106/`)
  - a lane-keeping hypothesis means the vehicle stays in its lane.
- **Checks:**
  - without the token bit-identical (13 smoke, 3 lead-only, ISA-MPC)
  - 7 tests (`tests/test_nolkentry.py`).
- **Result:** NGSIM final set with `bcts` + `occin` ε 0.02 / 0.05 / 0.1 (`replay_out/nolkentry_20261106/result.md`, paired against the same ε without it):
  - critical δ_max +5.1 / +4.7 / +3.5 %p (SE 0.7 / 0.7 / 0.6), j_max −0.03 / −0.00 / −0.04 (n.s.), floor breaches 6 / 7 / 9 → 11 / 10 / 11, contact ds3_v1780 in every arm, phantom_ep −0.055 / −0.031 / −0.014:
    it removes the lane-keeping entry (0 jump runs with it) but not the late calibrated labelled probability, so the reaction comes later.
- **Status:**
  - `stdan_3int` only, off by default, whole-token match (`_parse_no_lk_entry`)
  - part of the NGSIM design since the night of 2026-11-06 (maintainer's decision, §1; added 2026-11-06, 0b8e3b3; not in the paper)

### `occin<ε>`

- **What:** For an adjacent target, every raw trajectory i (LK, LLC, RLC) gets P_in,i(k), the Gaussian probability that the vehicle's body overlaps the ego's path laterally at step k
  (the path interval of `NGSIM_LANE_TEST=path`, the trajectory's predicted lateral position as the mean, the lateral std of STDAN's predicted position covariance)
  - a cell at step k ≥ 1 is active iff p_i·P_in,i(k) > ε, in place of the mean-trajectory test and the 0.1 gate (`acc_postprocess.occupancy_memberships`, `predict_acc(lane_membership_prob_fn=)`)
  - `trajcutin`'s rule reads it, step 0 keeps the Lane_ID test, ego-lane targets are unchanged.
- **Code:**
  - Needs `stdan_3int`, `trajcutin` and `NGSIM_LANE_TEST=path`
  - NGSIM replay only (the agent refuses it)
  - whole-token match (`_parse_occupancy_risk`)
  - tick arrays `oc_*`
  - tests `tests/test_occin.py` (added 2026-11-05, 7a85797).
- **Tuning:**
  - On uncalibrated p (tuning set, `replay_out/occin_cvar_20261105/`) `occin0.1` raised phantom braking
  - with `pcal` (`replay_out/pcal_tune_20261105/`) ε 0.02 / 0.05 / 0.10 gave phantom_ep 0.268 / 0.136 / 0.074 against 0.346 at critical δ_max +1.0 / +3.0 / +5.6 %p.
- **Status:**
  - `pcal` + `occin0.1` is part of the NGSIM design since 2026-11-06 (maintainer's decision, §1)
  - `bcts` replaced `pcal` that evening, since 2026-11-10 the design runs `occin0.05` with `pingate`, and `occin0.1` since NL that evening (§1)

### `pingate`

- **What:** With `occin<ε>` only (ValueError otherwise): a trajectory with p_i > ε is in the ego's path at step k ≥ 1 iff P_in,i(k) > ε, in place of p_i·P_in,i(k) > ε, so the switch-on step does not depend on the probability
  (`acc_postprocess.occupancy_memberships(pin_gate=)`, `process_vehicle_prediction(occupancy_pin_gate=)`, `predict_acc(occupancy_pin_gate=)`).
- **Reason** (maintainer, 2026-11-09): the intention probability and the entry step are handled apart
  - the probability sets the requirement through `floorrisk<ε>` + `floorhyp`.
- **Details:**
  - The bound P(in the path at k and gap < L + d0) ≤ ε still holds at the steps left out (p·P_in ≤ P_in ≤ ε); it is the more conservative rule.
  - Without `nolkentry` it also makes lane-keeping trajectories whose spread reaches the path `trajcutin` hypotheses (tests).
- **Checks:**
  - without the token bit-identical (13 smoke against N5's root)
  - lead-only with it identical (3 episodes)
  - 5 tests (`tests/test_pingate.py`).
- **Result:** NGSIM final set with `floorrisk0.05` + `floorhyp` on N5 (arm PG, `replay_out/pingate_20261109/result.md`, root `fin_bcts_occin0.05_nolkentry_pingate_floorhyp0.05_20261109`, paired against N5):
  - the TV's hypothesis switches on 0.33 / 0.29 s earlier (critical / mild, SE 0.07 / 0.06; most runs at the same tick, with its active cells from step 6.6 instead of 10.2), the floor is raised on 18 % of critical scored ticks,
    but critical δ_max +0.1 %p (SE 0.8), mild −1.1 %p (SE 0.5), Δt_ant, T_rec and j_max unchanged
  - floor breaches critical 10 → 13 (4 new, all pass → yield flips: the N5 run's TV merged beside the ego as a side ghost that never got clear ahead), mild 9 → 6
  - no-cut-in phantom_ep 0.031 → 0.058 (+0.027, SE 0.008), runs ≥ 0.3 13 → 24, runs at the jerk bound 5 → 11
  - the ds3_v1780 contact stays (−3.50 → −2.11 m)
  - the response by probability stays flat.
- **Status:**
  - NGSIM replay only (as `occin`)
  - whole-token match (`_parse_occupancy_pin_gate`)
  - off by default
  - part of the NGSIM design since 2026-11-10 with `floorrisk0.05` + `floorhyp` (maintainer's decision, §1; ε 0.1 since NL that evening)
  - removed from the NGSIM design on the night of 2026-11-11 (NP, maintainer)
  - not in the paper (added 2026-11-09, 91a2d0e)

### `cvar<ε>_<c>`

- **What:** With `cutin_chance`, `dopt<w>` and `fullfoot` only: on the cut-in chance cells the dopt price w·P_j·short² is replaced by the per-step bound CVaR_ε(short_k) ≤ c [m] over the scenarios
  (Rockafellar–Uryasev rows with their own ℓ1 slack; a lone hypothesis may give up c·max(1, ε/P_j) of its time gap, capped at τv; the hard floor stays; `lkr`'s priced cells keep the price; ticks without such a cell solve the design's QP).
- **Code:**
  - `NairACCConfig.cutin_cvar_risk / cutin_cvar_bound`
  - whole-token match (`_parse_cutin_cvar`)
  - tests `tests/test_cvar.py` (added 2026-11-05, 7a85797).
- **Tuning:** Tuned only on uncalibrated p (`replay_out/occin_cvar_20261105/`, B0.7_1 selected there).
- **Caveat:** For P to change the allowance, ε must lie above the active cut-in probabilities: at or below `occin`'s ε every active hypothesis gets c (the 2026-11-06 calibrated runs below).
- **Final set** (2026-11-06 night, maintainer's request, `replay_out/cvar_final_20261106/result.md`, without `nolkentry`):
  - with `bcts` + `occin0.05` and ε 0.05 every active hypothesis has P_j > ε, so the row reduces to a hard per-hypothesis bound (each may give up c of its time gap; the probability only switches it on)
  - against the same without CVaR (B5), c 1 / 2 / 4 m: critical δ_max −3.2 / −0.6 / +5.9 %p (SE 0.4 / 0.4 / 0.7), j_max +0.70 / +0.54 / +0.46 (SE 0.10),
    critical runs at the jerk bound 42 → 116 / 101 / 81 (103 of C1's 152 kept-TV jumps within 0.3 s of the hypothesis switching on, 72 of them `trajcutin`'s lane-keeping entry),
    no-cut-in phantom_ep 0.062 → 0.097 / 0.085 / 0.068
  - c 1 against the paper design: critical δ_max +0.5 %p (SE 1.2), phantom_ep 0.188 → 0.097, j_max 2.19 → 3.59.
- **Uncalibrated** (`occin0.1_cvar0.7_1`, the tuning set's choice) against the paper design: critical δ_max 9.1 → 5.3 % (SE 1.0), floor breaches 9 → 5, phantom_ep 0.188 → 0.282 (SE 0.021), runs a_min ≤ −2 11 → 31, j_max +1.32.
- No contact or fallback in any arm (server load fallbacks re-run on the workstation and spliced).
- **Wide tail** (2026-11-07, maintainer's request, `replay_out/cvar_wide_20261107/result.md`, with `bcts` + `occin0.05` + `nolkentry`, ε 0.5 / 0.7 × c 1 / 2 m, paired against the same without CVaR, N5):
  - the allowance follows P (ε 0.5, c 1 m: median shortfall used 5.2 m at p ≤ 0.1, 1.0 m at p ≥ 0.5), no-cut-in phantom_ep 0.031 → 0.028–0.053,
    critical δ_max −4.2 / −0.5 / −3.4 / +0.5 %p (SE 0.4–0.5; ε 0.5 / 0.5 / 0.7 / 0.7, c 1 / 2 / 1 / 2),
    but j_max +1.08 to +1.14 (SE 0.08) and critical runs at the jerk bound 36 → 126–129:
    the bound is hard, and at switch-on the calibrated p has a median of 0.28–0.42, where the allowance is 1.8–3.3 m against a median τv of 9.7 m
  - floor breaches and the ds3_v1780 contact as N5.
- **Status:** Not adopted (decision pending)

### `nocutout`

- **What:** An ego-lane vehicle gets no `cutout` mode.
  - Its trajectories that end outside the ego lane are dropped, and its lane-keeping mode, renormalised to probability 1, keeps the most likely staying trajectory and the full standoff (`acc_postprocess._mode_specs_for_relation(no_cutout=)`).
- **Reason** (maintainer, 2026-10-28): the paper treats cut-in scenarios only.
  - On NGSIM no-cut-ins a cut-out hypothesis relaxed a staying lead's standoff and then collapsed: 2 of the paper checkpoint's 9 hard brakes (a_min ≤ −2) on full_20261020 at λ 0, plus one partial case.
  - The only no-cut-in contact, ds1_v2886 (−3.23 m), came through the lead's lk cell scaled to 0.16, because the 0.35 floor exists on the cut-out cell only (`replay_out/tvt_hardbrake_20261028/result.md`).
- **Status:** `stdan_3int` only, off by default in the code, part of the current design since 2026-10-28 (§1; added 2026-10-28, not in the paper)

### `lksigma`

- **What:** With `cutin_chance` only: the ego-lane lead's lane-keeping cells also get the σ term q(min(p_lk, β_ref))·σ.
- **Details:**
  - The path is `ACCModePrediction.sigma_confidence` → `MultimodalLeadPrediction.sigma_confidence`, read by `cell_sigma_confidence` where `chance_confidence` is NaN.
  - The standoff stays hard: those cells keep a NaN `chance_confidence`, which keeps them out of the `dopt` cells.
- **Reason** (maintainer, 2026-10-28): a slow lead predicted to speed up causes hard brakes.
  - On slow-lead ticks STDAN's +3 s lane-keeping position is more than 2 m too far on 7 % of ticks, and the t_v_t checkpoint shows the same.
- With `nocutout` p_lk = 1, so the term is q(0.6)·σ = 0.84σ.
- **Status:** `stdan_3int` only, off by default in the code, part of the current design since 2026-10-28 (§1; added 2026-10-28, not in the paper) (updated 2026-11-12 night: not in the final NGSIM design, whose lead row takes its σ term from `zconf0.95`, z = 1.645 at p = 1, and with whose `onerow` the agent refuses `lksigma`; P1 and the CARLA string keep it)

### `fullfoot`

- **What:** With `cutin_chance` and `dopt<w>` only: a cut-in chance cell (`cutin` and `trajcutin`'s `cutin/LK`) keeps the whole footprint, factor 1 in place of α_j on L + d0 (`acc_postprocess.chance_cutin_clearance(full_footprint=)`).
- **Details:**
  - The controller reads that one factor for the hard floor, the top of the priced range and the reference, so they become L + d0 + qσ, d_safe(v) and the same
  - the σ term keeps β_j = min(p, β_ref), and the gate, the price, `lkr` and `entrypass` are unchanged.
  - It is `dopt`'s form of 2026-10-07 to 2026-10-15 (before 9c0cecd).
- **Reason** (maintainer, 2026-10-29): `replay_out/alpha_floor_20261029/judgment.md` found α not required (the 2026-10-15 slack spikes came from the lead-order filter, which `entrypass0` now removes) but not inert.
- **Result** (NGSIM final set, λ 0, ideal plant, paired against the design re-run with the same code, `replay_out/fullfoot_20261029/result.md`):
  - critical / mild δ_max −2.1 %p (SE 0.7) / −2.0 %p (0.4), a_min +0.13 (0.03) / +0.08 (0.02), T_rec −0.05 s
  - no-cut-in phantom_ep 0.097 → 0.149 m/s² (+0.052, SE 0.010), runs ≥ 0.3 40 → 62, j_max +0.14 (0.03), v_avg −0.12 m/s
  - contacts 0 → 0
  - floor breaches 15 → 14.
  - That is 0.41 / 0.38 %p of δ_max per +0.01 m/s² of phantom_ep (lowering w traded 1.1–4.8 %p per 0.01 on the older design).
  - The 4 new breaches and the largest δ_max rises (+40 to +90 %p) are 6 pass → yield flips through `entrypass0`:
    the design passed the real cut-in vehicle (its hypothesis excluded as entering behind the planned ego), while `fullfoot`'s slightly earlier braking turned the entry gap positive, kept the hypothesis and yielded late.
- **Checks:** Without the token bit-identical (12 NGSIM smoke episodes, max \|Δa_cmd\| = 0).
- **Status:**
  - `stdan_3int` only, off by default
  - ValueError without `cutin_chance` and `dopt<w>`
  - whole-token match (added 2026-10-29, 9441e0a, not in the paper; part of the current design since 2026-10-30, maintainer's decision, §1)

### `relbrake<b>`

- **What:** A relative-speed safe distance on every lead cell, `b` in m/s² (run as `relbrake3`; maintainer's decision 2026-10-30).
- **Rule:**
  - The required centre gap to a lead with speed v_l is d_rel = L + d0 + max(0, τ·v_e + (v_e² − v_l²)/(2b)): the gap at which, if the lead brakes at b and the ego brakes at b after the reaction time τ, d0 still remains (RSS with reaction time τ, no acceleration during it, equal decelerations).
  - At v_e = v_l it is today's d_safe
  - a faster lead lowers it to the floor L + d0, a slower one raises it (bumper terms, today → new: ego 9.1 / lead 11.1 m/s 14.8 → 8.1 m; 15 / 10: 22.5 → 43.3; 20 / 23: 29.0 → 7.5).
  - v_l is the cell's predicted lead speed (`means[.., 1]`, clipped at 0).
- **Where it applies:** It replaces τv everywhere the controller uses the deterministic standoff (`acc_nair_smpc.speed_gap`):
  - the cached feedback QP's safety rows, where v_e² is replaced by its tangent at the planned speed v̄_k the reference already uses (`speed_gap_tangent`, two per-cell QP parameters, the cached structure stays; the tangent understates v_e² by (v_e − v̄_k)²)
    and max(0, ·) is a second row per cell, gap ≥ c(L + d0) + qσ − ξ with the cell's factor, tightening and slack
  - the `dopt` priced part
    (the whole speed-dependent part h; the floor α(L + d0) and the price w·P_j·short² unchanged; the floor row replaces `short ≤ τv`, infeasible where h < 0, and is equivalent to short ≤ max(h, 0) at the optimum, where the solver leaves a priced shortfall at its tolerance, a few mm, on cells with h < 0)
  - the reference gap and the remembered dopt standoff (`OldACCReferenceAdapter.planned_speed`, so reference and rows ask for the same standoff)
  - `lkr`'s recovery test (d_rel to the lead at step 0)
  - `required_standoff` / `safety_function` / `optimized_standoff`, hence the logged margins, `_fallback_policy` and the agent's min gap.
- The σ term is unchanged (`safety_std` keeps h_ego = [−1, −τ]).
- **Code:**
  - `NairACCConfig.relative_brake_decel` (None = off), `NairACCSolution.speed_gap_tangent_error_max`
  - cached feedback QP only (ValueError elsewhere and with `brake_distance`)
  - combines with `jinput`
  - whole-token match (`_parse_relative_brake_decel`).
- **Checks:**
  - Without it bit-identical (12 NGSIM smoke episodes vs `fin_jinput_ra0_clip_20261029` and 3 lead-only episodes, max \|Δa_cmd\| = 0)
  - 241 tests (8 new, `tests/test_relbrake.py`).
- **Result:** NGSIM final set, λ 0, ideal plant, paired against the design (`replay_out/relbrake_20261030/result.md`, roots `fin_jinput_ra0_relbrake3_20261030` + its own lead-only root):
  - critical cut-in a_min +0.21 (SE 0.04), a_avg −0.13 (0.01), j_max +0.47 (0.11), δ_max +3.6 %p (1.1) on the plain d_safe and −1.5 %p (1.1) on the new rule, T_rec +0.41 s (0.07) with 6 runs not recovered, floor breaches 9 → 11
  - mild cut-in a_min +0.07 (0.04), j_max +0.76 (0.10), δ_max +3.7 %p (0.7) / −1.3 (0.8), T_rec +0.42 s (0.05), breaches 6 → 5
  - no cut-in phantom_ep +0.004 m/s² (0.011; the full and the lead-only loop both brake harder behind closing leads, +0.06 / +0.08), a_min −0.07 (0.02), j_max +0.79 (0.07), v_avg −0.12 m/s, δ_max on the new rule 6.4 → 1.0 %
  - contacts 0 in every group
  - 6 pass → yield flips through `entrypass0` (ds1_v2629 to 0.35 m).
- **Mechanics:**
  - behind a faster lead the ego takes the shorter gap the rule permits (gentler braking around the crossing, and the plain-d_safe δ / T_rec count it as intrusion),
  - behind a closing lead both loops brake harder,
  - and the jerk is the rule's sensitivity to the lead's *predicted* speed, not the logged speed's noise (checked 2026-10-30):
    - a row compares the two stopping points, s_l + v_l²/(2b) − s_e ≥ L + d0 + τv_e + v_e²/(2b),
      so STDAN's revision of a k ≥ 1 lead speed (0.13–0.28 m/s per 0.2 s, as large as the logged speed's own change; 10-27 prediction dump) moves the 0.2 / 0.6 / 1.0 s rows 18 / 9 / 5× more than under d_safe,
      while the logged speed enters only the k = 0 row, which has no decision variable, and the ego jerk does not follow it (corr ≤ 0.12, R² ≤ 0.01).
  - The relbrake3 runs log no predictions, so the link to their j_max ticks is not measured.
- Tables score it with the new column δ_max (rel) (`ngsim_metrics.delta_rel`, every row from its tick logs).
- **Status:**
  - Not run in CARLA
  - λ not re-calibrated (added 2026-10-30, not in the paper; **not adopted**, maintainer's decision 2026-10-30)

### `l1slack<w>`

- **What:** The emergency slack ξ ≥ 0 of every safety row costs w·Σξ in place of 5000·Σξ² (`NairACCConfig.slack_l1_weight`; run as `l1slack1e6`).
- **An exact penalty:** for w above the largest multiplier of the safety rows the solve equals the hard-constrained one whenever that has a solution, so ξ > 0 means that no input sequence keeps every row
  - the quadratic cost has zero slope at 0 and cuts every binding row by about multiplier / (2·5000).
- **Logged:**
  - `NairACCSolution.slack` holds ξ per (mode, step) for every solve
  - the NGSIM replay logs the cells with ξ > 1e-4 m as the tick file's `slack_cells` (frame, mode, step, ξ, lead id, kind, priced, P).
- **Reason** (maintainer, 2026-10-31): item 1 of the 2026-10-30 recursive-feasibility review, so that "no hard solution" can be measured.
- **Result:** NGSIM final set, λ 0, ideal plant, paired against the design (`replay_out/l1slack_20261031/result.md`, root `fin_jinput_ra0_fullfoot_l1slack1e6_20261031` + its own lead-only root):
  - scored ticks with ξ > 0.01 m 15.4 / 13.0 / 4.9 % → 1.1 / 1.0 / 1.0 % (critical / mild / no cut-in), > 0.5 m 0.82 / 0.42 / 0.11 → 0.68 / 0.38 / 0.10 %
  - critical δ_max −0.9 %p (SE 0.3), mild −0.2 (0.1), j_max +0.06 / +0.09 / +0.07 (SE 0.03 / 0.04 / 0.02), phantom_ep +0.000 (0.004), runs ≥ 0.3 62 → 66, floor breaches 14 → 13, contacts 0, fallbacks 0.
  - No hard solution on 0.91 / 0.40 / 0.70 % of scored ticks, in 15 / 20 / 18 runs, among them 9 of the 10 critical floor-breach runs and all 3 mild ones.
  - Of the 51 ticks where the hard solution is lost after a tick that had one, 41 are on cut-in hypothesis cells
    (34 on a hypothesis already active and feasible whose prediction then moved, 5 through an `entrypass0` exclusion lifted, 2 through the 0.1 gate)
    and 10 on the ego-lane lead's hard lane-keeping cell
  - the no-cut-in step-0 violations are mostly `lksigma`'s q·σ term (90 % with the measured gap ≥ d_safe).
- **Checks:**
  - Without the token bit-identical (12 NGSIM smoke episodes vs `fin_jinput_ra0_fullfoot_20261029`, 3 lead-only episodes)
  - w 1e5 and 1e6 differ by ≤ 8.4e-4 m/s² on the smoke.
- **Code:**
  - Cached feedback QP only (ValueError elsewhere)
  - whole-token match (`_parse_slack_l1_weight`)
  - 6 tests (`tests/test_l1slack.py`).
- **Status:**
  - λ not re-calibrated
  - not run in CARLA (added 2026-10-31, 23cf33f, not in the paper; part of the current design since 2026-10-31, maintainer's decision, §1)

### `termcost<c>`

- **What:** A terminal cost-to-go on the cached feedback QP with `jinput`: per scenario c·3·P_j·[e_N, a_N](P − Q)[e_N, a_N]ᵀ, with e_N = v_N − v_ref,N and P the DARE solution of the following-error dynamics (e, a) with the jerk as input over dt
  (Q = diag(q_v, 0), R = r_jerk; at q_v 1, r_jerk 40, dt 0.2, P − Q = [[17.29, 31.62], [31.62, 112.51]]; `acc_nair_smpc.terminal_cost_matrix`, `NairACCConfig.terminal_cost_scale`).
- **Reason** (maintainer, 2026-11-01): without a terminal term the plan ends still braking, so the next solves release the braking late.
- **Code and checks:**
  - ValueError without `jinput`, with `with_k` or outside the cached feedback QP
  - whole-token match (`_parse_terminal_cost_scale`)
  - without it bit-identical (12 NGSIM smoke episodes, 3 lead-only)
  - 7 tests (`tests/test_termcost.py`).
- **Tuning**
  (2026-11-02, `replay_out/termcost_20261102/result.md`, pre-registered rule: critical Δδ_max ≤ +1.0 %p and no-cut-in Δphantom_ep ≤ +0.02;
  94 ramp-merge cut-ins and the 500 λ-calibration no-cut-ins, since no tv_main cut-in is disjoint from the final set):
  - c 0.25–4 against 0 cut over-deceleration by 0.15–0.34 m/s and the brake release by 0.5–1.2 s, at critical δ_max +1.0 to +2.6 %p and phantom_ep +0.010 to +0.049
  - contacts and floor breaches unchanged.
  - Every c > 0 failed the rule (c 0.25: +1.02 %p), so the final set was not run.
- **Status:** Off by default, not adopted (decision pending), not in the paper (added 2026-11-02, ca7183b)

### `safelayer`

- **What:** PC-ACC's safety layer around the MPC's command (`utils/safety_layer.py`; Althoff et al. T-IV 2021, Alg. 1 / Prop. IV.5, with the MPC as the nominal controller, which the paper allows without re-certification).
  - It runs every command interval on measured states only, through PC-ACC's perception: CARLA `policies/pcacc_agent.observe_preceding_vehicles`, NGSIM `replay/baselines.SafetyLayerReplay` = `PCACCReplay.observe`.
- **Membership** is the ego's path: a vehicle ahead (bumper gap > 0) that is in the same lane, or adjacent with its body overlapping the ego's laterally, with PC-ACC's 1 s leave lag.
  - The paper's lane occupancy, its lateral cut-in rule and vehicles alongside the ego are not used.
  - On NGSIM each engaged the layer for vehicles that the longitudinal controller cannot or need not avoid:
    - lane occupancy for vehicles touching the marking 0.4–1.4 m beside the ego (4 of 15 check episodes),
    - the lateral rule for a slow adjacent queue vehicle (to −7.9 m/s²),
    - and alongside vehicles for TVs merging beside the ego (bumper gap −1 to −12 m; full braking turned passes into yields with new floor breaches, first final-set run v1).
- **Cut-ins:** a vehicle that joins the path, or is first seen in it, inside the safe distance of the current state is a cut-in (Def. II.3) until the gap reaches that distance
  - in an ICS against it (it braking at −2 m/s²) the layer brakes fully.
- **Every other member** gets condition (7) for the limited MPC command held over the interval: the fail-safe must then stop behind the vehicle braking at −10.5 m/s² from now.
  - The fail-safe is a constant jerk of −20 m/s³ down to −8 m/s² (updated 2026-11-12: the NGSIM replay's default runs it at jerk 10, `fsjerk10`, since 2026-11-12, which makes the guaranteed bumper gaps below 6.3 / 10.6 / 15.7 m; `fsdecel<x>`, `fsjerk<x>`)
  - the requirement adds \|a_em\|·dt²/8 for the plant's stopping tick (1 cm at 0.1 s).
  - If one member fails, the fail-safe is issued until the MPC command passes again (the MPC command instead, if it brakes harder).
- **Command path:** the fail-safe bypasses the command limiter.
  - After the layer acted, the MPC starts from the issued command and plans the recovery at the fail-safe's jerk (`failsafe_recovery`, §3, 2026-11-04)
  - at low speed the start is raised to `failsafe_recovery_floor` (the −√(2jv) clip of 11-02, exact on the model grid).
  - Until 2026-11-04 it started from the previous command clipped to [max(a_min, −√(2·j·v)), a_max] while the limiter clipped around the unclipped one (v1 had 3 fallback ticks at 0.15–0.6 m/s when the clip ended with the layer).
- **Guarantee:** no collision with a path member that joined the path, or was first seen, outside the safe distance, whatever the predictor or the QP do, if members brake no harder than 10.5 m/s², the plant executes the fail-safe and the states are exact.
  - Guaranteed bumper gap at equal speeds: 4.4 / 7.8 / 11.9 m at 10 / 15 / 20 m/s (the fork's d_safe is 16 / 22.5 / 29 m).
- **Logging:**
  - `policy_log[*].safety_layer` in CARLA
  - on NGSIM the tick arrays `a_nominal`, `safety_path` (0 nominal, 1 fail-safe, 2 ICS), `safety_vid`, `safety_margin`, `safety_n_cutin` and the results columns `n_safety_*`.
- **Checks:**
  - without the token bit-identical (12 NGSIM smoke episodes, 3 lead-only)
  - with it, identical on the same 15 (no tick acted).
- **NGSIM final set** (λ 0, ideal plant, `replay_out/safelayer_20261102/result.md`, roots `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_20261102` + `lo_…`, code 96bfb98):
  - the layer acted on 2 ticks in 2 of 889 runs, both in the unscored warm-up against a non-TV vehicle the replay's start placed near the ego
  - the lead-only loop is identical.
  - Every column of the paper table is unchanged to its precision (mild j_max +0.01) (clarified 2026-11-12: every paired difference is 0.00 at the table's precision except mild j_max, +0.01 (SE 0.01), one unit of its last digit; `replay_out/safelayer_20261102/result.md`, item 3)
  - 0 fallbacks, 0 contacts, floor breaches 10 / 3 as the design.
  - The margin to the layer's requirement on scored ticks is ≥ 0.01 m (critical / mild) and ≥ 2.99 m (no cut-in).
- **Membership fix:** The membership rule was fixed after a first final-set run (v1, 1d0a3a2, roots `*_safelayer_v1_20261102`).
  - v1 counted vehicles alongside the ego and first-sight-inside vehicles as covered, and acted in 22 runs with full braking for TVs merging alongside (5 new floor breaches, one removed)
  - the paper must disclose this.
- **Tests:** 14 tests (`tests/test_safety_layer.py`): an always-accelerating nominal behind a lead braking at −10.5, and 400 random closed loops with lead braking within [−10.5, 2], all without contact.
- **Status:**
  - Whole-token match (`_parse_safety_layer`).
  - Not run in CARLA (added 2026-11-02, maintainer's decision; in the paper's NGSIM tables since 2026-11-02, maintainer's decision, §1: `replay_out/safelayer_20261102/tables_paper_safelayer.md`)

### `safebranch`

- **What:** With `safelayer` only (ValueError otherwise; cached feedback QP only; not with `alag<τ>` or `with_k`):
  the safety layer's fail-safe as one more branch of the scenario tree (maintainer, 2026-11-03: the safety plan belongs in the contingency plan, not in a layer the MPC does not know about).
- **Details:**
  - The layer classifies before the solve (`SafetyLayer.classify`) and its checked members (path members that are not Def. II.3 cut-ins) lead the branch, each braking at −10.5 m/s² from its measured state (`failsafe_branch_lead`; the nearest position per step binds).
  - The branch shares the first input with every scenario and from step 1 obeys the fail-safe's limits (jerk 20, a ≥ −8) instead of the comfort limits.
  - Its rows are hard with the ℓ1 slack:
    - bumper gap ≥ 0.1 m at steps 1..N (covers the 0.2 s sampling dip, 0.09 m, and the layer's 1 cm stop-tick term),
    - and a terminal row s_N + c_T·v_N + D_0 ≤ s_stop − L − 0.1 that bounds the fail-safe's remaining stopping distance (c_T = (a_max + 8)/20 + v_max/16 = 1.75 s, D_0 = 1.3 cm; `failsafe_terminal_coefficients`, derivation in its docstring).
  - It has no time gap, σ term, chance factor or `dopt` price.
  - Its cost is scenario 0's weighted 1e-3; the other scenarios' weights are unchanged.
  - After the solve the layer decides on the same classification (`decide`), unchanged, so the guarantee is the layer's.
  - Without a checked member the branch's lead sits 10⁶ m ahead.
- **Logged:** `NairACCSolution.failsafe_*`, `policy_log[*].failsafe_branch`, NGSIM tick arrays `fs_present`, `fs_members`, `fs_margin_min`, `fs_slack`, `fs_binding`.
- **Checks:**
  - without the token bit-identical (12 smoke + 3 lead-only episodes)
  - 13 tests (`tests/test_safebranch.py`)
  - in the tests' synthetic loops a lead easing at −2 m/s² from 30 m makes the plain design's layer act 3 ticks (a_min −4.63, min gap 7.95 m), while with the branch the command brakes 0.2 s earlier, the layer never acts (a_min −3.00, min gap 9.29 m)
  - against a lead braking at −4 m/s² the comfort-limited first input cannot keep the branch feasible and the fail-safe still runs (13 ticks against 17).
- **Result:** NGSIM final set, λ 0, ideal plant, paired against the design with `leadmerge` (`replay_out/safebranch_20261103/result.md`, roots `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_20261103` + `lo_`):
  - every printed paper-table cell moves by at most one unit of its last digit (critical j_max 2.07 → 2.06, a_min −1.21 → −1.20; mild δ_max 7.2 → 7.3; no-cut-in phantom_ep 0.149 → 0.152),
  - and the pre-registered 2-SE rule fails on systematic third-decimal shifts (critical δ_max −0.08 %p, a_min +0.006, j_max −0.010; no-cut-in v_avg −0.005 m/s),
    which come through the branch's 1e-3 cost (tick 0, identical states: first command −1.6e-4 m/s² on average where the branch is active)
  - contacts, fallbacks and floor breaches (10 / 3) unchanged
  - one `entrypass0` pass → yield flip (ds1_v2629, δ_max 0 → 59 %, 2 cm state difference).
  - The branch is present on 98 % of ticks, has no hard solution on 0.01–0.03 % of them, and never had a hard solution while the layer issued the fail-safe (0 ticks in 889 + 400 runs).
  - Of the design's two layer-acted warm-up runs it removes the fail-safe one (ds4_v2935: braking from 0.4 s earlier, peak −1.23 against the layer's −1.77 m/s² after a 2.0 m/s² one-tick step); the ICS one against a cut-in stays.
  - QP time +11 ms per tick under parallel load (+25 % in the smoke).
- **Status:**
  - Off by default
  - whole-token match (`_parse_failsafe_branch`)
  - not run in CARLA
    (added 2026-11-03, b327b03, implemented by a Fable subagent;
    in the paper's NGSIM tables since 2026-11-03, adopted as is, maintainer's decision, §1: `replay_out/safebranch_20261103/tables_paper_safebranch.md`;
    the replay's default tail ends `_safebranch`;
    dropped from the NGSIM design and the replay's default tail on the night of 2026-11-11, maintainer, §1, §10)

### `entrant<b>`

- **What:** With `safelayer` only (ValueError otherwise; run as `entrant2`, a_c = −2 m/s², PC-ACC's a_cut-in):
  the layer covers its **close entrants** (the code's cut-in set: a vehicle that joined the ego's path, or was first seen in it, inside the layer's safe distance D, whether or not it changed lanes; Althoff et al. Def. II.3),
  which `safelayer` alone leaves outside the guarantee (ICS test only).
- **Per-tick check:**
  - Every tick each close entrant E is checked with `safety_layer.entrant_margin`:
    after the nominal command is held over the interval, the fail-safe (jerk −20 to −8) must keep the bumper gap ≥ 0 at every time against E braking at a_c from its measured state until standstill
    (a trajectory comparison: Prop. IV.4's stopping-point test needs the lead to brake harder than the ego, which a_c does not; exact, the gap being a cubic between breakpoints),
    minus the stopping-tick term
  - if a check fails the fail-safe is issued (path code 3, `PATH_ENTRANT`), and the ICS test is dropped (an unavoidable entrant fails the check for every command, so the fail-safe is issued as mitigation).
  - E leaves the set when its gap reaches D, as before.
- **Fail-safe branch:**
  - With `safebranch`, close entrants lead the fail-safe branch too, braking at a_c (`FailsafeBranch.from_members(entrants=, entrant_decel=)`, per-member bounds in `failsafe_branch_lead`),
    with two terminal rows v_N ≤ min_E v_E(N) and a_N ≤ a_c (`failsafe_entrant_terminal`; `NairACCConfig.failsafe_entrant_decel`), under which the ego is never faster than an entrant after the horizon
  - the rows live in a second cached QP used only on ticks whose branch carries an entrant (two inactive rows had moved Gurobi's solution by up to 7e-4 m/s² on entrant-free ticks), so every other tick solves the design's QP bit for bit.
- **Guarantee** (by Prop. IV.5's induction with a_c in place of a_p):
  no collision with a close entrant that the fail-safe could avoid when it entered,
  provided it brakes no harder than a_c while it is an entrant and no harder than a_p = −10.5 once it is a checked member, the plant executes the fail-safe and the measured states are exact.
- **Not guaranteed:** an entry the fail-safe cannot avoid at the entry tick (logged: `new_unavoidable_vids`), and vehicles not yet in the path.
- **Logged:**
  - `SafetyDecision.entrant_margin / unavoidable_vids / new_entrant_vids / new_unavoidable_vids`, `NairACCSolution.failsafe_entrants / failsafe_entrant_margin`, `policy_log[*].safety_layer` and `.failsafe_branch` fields of the same names
  - NGSIM tick arrays `entrant_margin`, `entrant_unavoid`, `entrant_new`, `entrant_new_unavoid`, `fs_entrants`, `fs_entrant_margin`
    and results columns `n_safety_entrant`, `n_entrant_ticks`, `n_entrant_ticks_sc`, `n_entrant_unavoid_ticks`, `n_entrant_new`, `n_entrant_new_unavoid`.
- **Checks:**
  - without the token bit-identical (12 smoke + ds5_v2609, 3 lead-only)
  - with it on the same episodes (no entrant) bit-identical, 0 fallbacks
  - 15 tests (`tests/test_entrant.py`:
    the exact margin against a fine simulation,
    300 random entrant loops within the bound with no contact whenever the entry was avoidable,
    the adversarial accelerating nominal against an entrant braking at a_c (covered; the ICS-only layer loses gap),
    the transition to a checked member,
    the branch's terminal rows,
    a closed loop with the MPC).
- **Result:** NGSIM final set (λ 0, ideal plant, `replay_out/entrant_20261104/result.md`, root `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_entrant2_20261104` + `lo_`, paired against the paper root with the path test):
  - close entrants in 3 of 889 runs (20 ticks; 1 scored run, ds3_v1828, 4 scored ticks), 3 entries, 0 unavoidable at entry, 0 entrant-path ticks, the layer acting on no tick, 0 fallbacks / contacts
  - the command differs from the paper root at the 1e-11 level on the 3 entrant runs and on ds4_v3035 only through the paper root's own time-limit fallback tick (frame 8315, QP 389 ms under load)
  - every printed paper-table cell unchanged except critical δ_avg 0.73 → 0.72 and fallback runs 1 → 0, both from ds4_v3035 (`tables_paper_entrant2.md`).
- **Status:**
  - `stdan_3int` only (through `safelayer`)
  - off by default
  - whole-token match (`_parse_entrant_decel`)
  - not run in CARLA, not in the paper
    (added 2026-11-04, 4c95000;
    in the paper's NGSIM tables since 2026-11-04, maintainer's decision, §1: `replay_out/entrant_20261104/tables_paper_entrant2.md`;
    the replay's default tail ends `_entrant2`)
    (clarified 2026-11-12: "not in the paper" means the paper of 2026-09-10, see this file's header, so it does not contradict "in the paper's NGSIM tables"; updated 2026-11-12: the default tail has grown since and ends `_fsjerk10` since 2026-11-12, with `entrant2` still in it, `ngsim_replay_defaults.md`)

### `floorrisk<ε>`

- **What:** With `cutin_chance`, `dopt<w>` and `fullfoot` only (ValueError otherwise; cached feedback QP only; not with `relbrake<b>`):
  the hard floor of a cut-in chance cell (`cutin`, `cutin/LK`) becomes the chance constraint P_j · P(gap < L + d0) ≤ ε,
  i.e. the row gap ≥ L + d0 + z_j·σ_jk − ξ with z_j = Φ⁻¹(1 − ε/P_j) (one-sided; σ the cell's `safety_std`, P_j the scenario probability as the QP weights it),
  and no floor where P_j ≤ ε (maintainer's idea, 2026-11-02, to relax the following distance by the mode probability).
- **Details:**
  - The priced row, its top D = L + d0 + τv + q(β_j)σ and the price are unchanged, but the shortfall's cap τv is lifted on these cells, so the price keeps growing below the old floor and the new row (sharing the cell's ℓ1 slack) bounds the gap
  - lane-keeping cells (also the `lkr`-priced ones) and the reference are unchanged (`floor_risk_quantile`, `floor_risk_cells`, `NairACCConfig.floor_risk`; three per-cell QP parameters).
- **Logged:** `NairACCSolution.n_floor_free / n_floor_lowered / n_floor_raised` and `floor_old_margin_min` (the plan's smallest margin to the old floor L + d0 + q(β)σ), also as NGSIM tick arrays.
- **Checks:**
  - without the token bit-identical (12 smoke + 3 lead-only + 5 episodes)
  - the two-row form with z = q matches the old row to 1.4e-7 m/s² at ftol 1e-9
  - lead-only identical with the token
  - 11 tests (`tests/test_floorrisk.py`).
- **Tuning set** (pre-registered, `replay_out/floorrisk_20261102/result.md`; 94 ramp-merge cut-ins + 500 λ-calibration no-cut-ins, ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5} against the design):
  - ε\* = 0.3 passes (no-cut-in phantom_ep 0.292 → 0.281, −0.012, SE 0.003; critical δ_max +0.03 %p; contacts, floor breaches, safety-layer runs unchanged; no-cut-in j_max −0.08),
  - but the floor rarely binds: the plan went below the old floor on 0.25–0.49 % of scored ticks although the rule removed or lowered the floor on about half of the critical ticks.
  - One fallback tick at ε 0.3 (ds3_v1428, a solver numerical failure).
  - Caveat: 82 % of the QP's scenarios are duplicates (same effective-lead sequence), so one event's probability is split over several P_j and the rule relaxes more than the per-event formula intends.
- **Status:**
  - Final set alone not run
  - with `floorhyp` and `pingate` (ε 0.05, 0.1 since NL that evening) part of the NGSIM design since 2026-11-10 (maintainer's decision, §1)
    (updated 2026-11-12: `pingate` left the NGSIM design on the night of 2026-11-11, NP; `floorrisk0.1` with `floorhyp` stays in it and in the replay's default string; `onerow`, which the maintainer decided on 2026-11-12 will be the final design, refuses `floorrisk<ε>`, `onerow` below)
    (updated 2026-11-12 night: no longer in the NGSIM design or the replay's default string: the final design, `onerow` with `zconf0.95`, `nocap` and `snap160`, has no floor row; P1 keeps `floorrisk0.1`)
  - off by default, not in the paper (added 2026-11-02, 0fa1c50, implemented by a Fable subagent)

### `floorhyp`

- **What:** With `floorrisk<ε>` only (ValueError otherwise):
  the floor's z = Φ⁻¹(1 − ε/p) takes p, the probability of the cell's own hypothesis (its lead mode's probability for its vehicle, `MultimodalLeadPrediction.lead_mode_probability`),
  in place of the scenario probability P_j, which splits p over the scenarios that carry the hypothesis and relaxes each (`floor_risk_cell_quantile`, `NairACCConfig.floor_risk_hypothesis`)
  - then P(hypothesis in the path and gap < L + d0) ≤ ε holds per hypothesis where the floor row holds.
  - A prediction with neither those probabilities nor a cut-in chance cell (the replay's no-target placeholder) needs none; a first smoke raised there and was fixed before the run.
- **Checks:** `floorrisk` alone unchanged (its tests); 4 tests (`tests/test_floorhyp.py`).
- **Result:** Final set only with `pingate` (`pingate`'s section above, `replay_out/pingate_20261109/result.md`; pointer updated 2026-11-12: it read "row above").
- **Status:**
  - Whole-token match (`_parse_floor_risk_hypothesis`)
  - off by default
  - part of the NGSIM design since 2026-11-10 with `pingate` and `floorrisk0.05` (maintainer's decision, §1; `floorrisk0.1` since NL that evening)
    (updated 2026-11-12: `pingate` left the NGSIM design on the night of 2026-11-11, NP; `floorhyp` with `floorrisk0.1` stays in it and in the replay's default string; `onerow`, which the maintainer decided on 2026-11-12 will be the final design, refuses `floorhyp`, `onerow` below)
    (updated 2026-11-12 night: no longer in the NGSIM design or the replay's default string: the final design, `onerow` with `zconf0.95`, `nocap` and `snap160`, has no floor row; P1 keeps `floorhyp`)
  - not in the paper (added 2026-11-09, 91a2d0e)

### `floortop`

- **What:** With `floorhyp` only (ValueError otherwise):
  a cut-in chance cell's σ term in its priced row and in the reference is the floor's one-sided z·σ, z = Φ⁻¹(1 − ε/p), in place of q(min(p, β_ref))·σ (`acc_nair_smpc.cell_sigma_term`, `NairACCConfig.floor_risk_top`),
  so the top of the priced range is the floor plus the time gap, D = L + d0 + τv + z·σ, the gap at which p·P(gap < L + d0 + τv) ≤ ε.
- **Details:**
  - Against q, z − q = −1.04 / −0.13 / +0.58 / +0.75 / +0.80 at p 6 / 10 / 30 / 90 / 100 %: below p = 2ε z is negative, and D falls below d_safe as the floor falls below L + d0.
  - The ego-lane lead's lane-keeping cells (`lksigma`) keep q(0.6)·σ = 0.84σ, so a TV's D drops by up to 0.8σ when it becomes the lead (not with `lkonesided<ε>` at the same ε, row below); the lead-only loop is unchanged.
  - The diagnostic `floor_old_margin_min` stays against L + d0 + q(β)·σ.
  - A cut-in cell with p ≤ ε raises (none is active under `occin<ε>` at the same ε).
- **Reason** (maintainer, 2026-11-10): the σ term of D one-sided as well.
- **Checks:** 5 tests (`tests/test_floortop.py`); the full suite passes (439 tests).
- **Result:** Final set not run.
- **Status:** Whole-token match (`_parse_floor_risk_top`); off by default, not adopted, not in the paper (added 2026-11-10, e80b3e6)

### `lkonesided<ε>`

- **What:** With `lksigma` and `nocutout` (ValueError otherwise, from the agent and the replay; run as `lkonesided0.05`):
  a lane-keeping cell's σ term (`lksigma`'s, the ego-lane lead's) is the one-sided z·σ, z = Φ⁻¹(1 − ε/p) with p its lead mode's probability, 1 under `nocutout`, so z = 1.645 at ε 0.05,
  in place of q(min(p, β_ref))·σ = 0.84σ (no β_ref cap; `acc_nair_smpc.cell_sigma_term`, `NairACCConfig.lane_keep_risk`).
- **Details:**
  - Its ε is its own, apart from `floorrisk<ε>`'s (maintainer, 2026-11-10: the cut-in constraint at 10 %; the lane keeping stays at 5 %, an assumption stated to the maintainer).
  - It applies wherever that term is used:
    - the hard row, which then keeps p·P(gap < d_safe) ≤ ε (before: P(gap < d_safe) ≤ 20 %),
    - `lkr`'s priced cells (with `nopricesigma` their floor only),
    - the reference,
    - and `lkrsigma`'s release test.
  - With the margins measured on 2026-11-04
    (`replay_out/sigma_coverage_20261104/result.md`, lead lane keeping 0.15 / 0.32 / 0.67 / 1.03 m at 0.2–0.6 / 0.8–1.4 / 1.6–2.2 / 2.4–3.0 s)
    the row asks for +0.14 / +0.31 / +0.64 / +0.98 m more (an estimate)
  - the same table's 95th percentiles of the error (0.70 / 1.50 / 2.43 m at 1 / 2 / 3 s) lie above 1.645σ, so the Gaussian z likely under-covers there (not measured).
  - It changes the lead-only loop, so `phantom_ep` needs a new lead-only root.
  - With `floortop` a TV's σ term is z·σ before and after it becomes the lead (equal at p = 1 and the same ε).
- **Reason** (maintainer, 2026-11-10): the lane-keeping σ term one-sided, without the 60 % cap.
- **Checks:**
  - 7 tests (`tests/test_lkonesided.py`: the σ term with and without `floortop`, its own ε, the reference, the hard row's step-0 slack (z − 1.2)σ against 0, the release test)
  - the full suite passes (452 tests)
  - PG's string without the token bit-identical (13 smoke + 3 lead-only).
- **Result:** Final set only with `nopricesigma` and ε 10 % (arm NL, row below).
- **Status:**
  - Whole-token match (`_parse_lane_keep_risk`)
  - off by default
  - part of the NGSIM design since the evening of 2026-11-10 (NL, maintainer's decision, §1) (updated 2026-11-12 night: until that night; not in the final NGSIM design, whose `onerow` refuses it and whose lead row gets 1.645σ from `zconf0.95`; P1 keeps it)
  - not in the paper (added 2026-11-10, fb9e916)

### `nopricesigma`

- **What:** With `floorrisk<ε>` only (ValueError otherwise):
  every priced cell (`standoff_shortfall_mask`: the cut-in chance cells and `lkr`'s priced lane-keeping cells) asks for its standoff D on the mean gap, gap ≥ D with cost 100·P·(d_safe − D)² (updated 2026-11-12: the factor is `dopt<w>`'s w, 100 when this was written; the NGSIM design uses w 3, `dopt3lkr`, since the night of 2026-11-11),
  in its priced row and in the reference, without a σ term (`NairACCSMPC._cell_tightening`, `NairACCConfig.price_row_no_sigma`)
  - the floor keeps the σ term in its own hard row (`floor_risk_cells`):
    - a cut-in cell's floorrisk floor L + d0 + z·σ,
    - and a priced lane-keeping cell's L + d0 plus its own σ term (`cell_sigma_term`; its τv cap is lifted as on the cut-in cells, so the floor it had through the cap stays).
- **Reason** (maintainer, 2026-11-10): a σ term belongs only where a probability bound is converted into a row; the priced row carried Proposed's q(min(p, 60 %))σ, inherited when D-opt turned Proposed's hard row g ≥ α·d_safe + q(β)σ into a price.
- The diagnostic `floor_old_margin_min` stays against L + d0 + q(β)σ.
- **Checks:**
  - 6 tests (`tests/test_nopricesigma.py`: the rows' σ terms, the floor cells, the reference, the cut-in shortfall on the mean gap, a priced lane-keeping cell's floor and top)
  - PG's string without the token bit-identical (13 smoke + 3 lead-only).
- **Result:** NGSIM final set with `lkonesided0.05`, `occin0.1` and `floorrisk0.1` on PG
  (arm NL, `replay_out/nps_lk_20261110/result.md`, root `fin_bcts_occin0.1_nolkentry_pingate_floorhyp0.1_nopricesigma_lkonesided0.05_20261110` + its own `lo_`; three changes at once, the maintainer's request, not decomposed),
  paired against PG:
  - critical / mild
    - a_min +0.13 / +0.12 (SE 0.03),
    - j_max −0.27 / −0.30 (SE 0.09 / 0.06),
    - δ_max −0.6 / +0.6 %p (SE 0.8 / 0.3),
    - T_rec +0.12 s (SE 0.03 / 0.02),
    - floor breaches 13 / 6 → 12 / 5
  - no-cut-in phantom_ep 0.058 → 0.029 (SE 0.009), runs ≥ 0.3 24 → 12, v_avg −0.07 m/s
  - the TV's hypothesis on 0.62 / 0.45 s later (ε 10 %)
  - the ds3_v1780 contact stays (−2.11 → −1.30 m)
  - 0 fallbacks.
  - The lead-only loop isolates `lkonesided0.05`: mean gap − d_safe +0.92 m, braking B −0.048 m/s², j_max −0.07.
- **Status:**
  - Whole-token match (`_parse_price_row_no_sigma`)
  - off by default
  - part of the NGSIM design since the evening of 2026-11-10 (NL, maintainer's decision, §1) (updated 2026-11-12 night: until that night; not in the final NGSIM design, whose `onerow` refuses it; P1 keeps it)
  - not in the paper (added 2026-11-10, fb9e916)

### `leadmerge`

- **What:** One scenario per class of joint modes with the same effective lead at every step and the same cell arrays (`acc_postprocess._merge_duplicate_scenarios`, `build_multitarget_lead_prediction(lead_merge=)`):
  - the class takes the summed probability and its most likely member's arrays, and is named after its leads in order (`"<vehicle>:<label>+..."`, `_lead_sequence_name`), so the name-keyed dopt reference memory and `entrypass`'s previous-plan choice follow the leads
  - if two classes would share that name (branchdec markers, inherited cut-out cells) every class keeps its most likely member's name.
- **Reason** (maintainer, 2026-11-03): 82 % of the QP's scenarios were exact duplicates in 1-D (joint modes differing only in modes that never lead: adjacent lane-keeping modes, gated, excluded or shadowed cut-ins), and the QP time follows the scenario count.
- **Checks:**
  - without the token bit-identical (12 NGSIM smoke + 3 lead-only episodes)
  - with it on the smoke, max |Δa_cmd| 3.9e-4 m/s²
  - 5 tests (`tests/test_leadmerge.py`, one solves a 4-scenario problem and its merged 2-scenario version: same first command, |Δu0| < 1e-4).
- **Result:** NGSIM final set (λ 0, ideal plant, `replay_out/leadmerge_20261103/result.md`, roots `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_20261103` + `lo_…`, paired against the paper design with `safelayer`, pre-registered rule passed):
  - every paper-table column unchanged to its precision (mild j_max 1.93 → 1.92) (clarified 2026-11-12: every paired difference rounds to 0 at the printed precision; mild j_max's printed mean still moved by one unit of its last digit, paired −0.00 (SE 0.00); `replay_out/leadmerge_20261103/result.md`)
  - contacts, fallbacks, floor breaches (10 / 3) and safety-layer runs (2) unchanged
  - no episode moves δ_max by > 1 %p
  - 9 of 889 episodes have a tick with |Δa_cmd| > 0.01 (at most 0.17 m/s²).
  - Scenarios per tick 1.21 (1 on 81.2 % of ticks)
  - QP time per tick mean 121.0 → 39.9 ms
  - table t_comp 171 / 178 / 188 → 138 / 137 / 150 ms (critical / mild / no cut-in).
  - The t_comp change understates the QP saving: STDAN's time rose 80 → 117 ms between the two runs with an unchanged predictor
    (run-to-run load, 36 processes on 36 logical CPUs; at the smoke runs' equal load it was 43 ms in both arms),
    so the table's absolute times depend on how many replays ran in parallel.
- **Status:**
  - `stdan_3int` only, off by default
  - whole-token match (`_parse_lead_merge`)
  - the replay logs the QP's scenario count as the tick array `n_scen`.
  - Part of the current design since 2026-11-03 (maintainer's decision, §1; the replay's default tail ends `_leadmerge`)
  - not run in CARLA, not in the paper (added 2026-11-03, 12b8c1d)
  - (updated 2026-11-12: the default tail has grown since and ends `_fsjerk10` since 2026-11-12, with `leadmerge` still in it, `ngsim_replay_defaults.md`; clarified 2026-11-12: "not in the paper" means the paper of 2026-09-10, see this file's header, while the paper's NGSIM tables have run with it since 2026-11-03, row "D-opt, NGSIM paper (2026-11-03)" of `design_history.md`)

### `mergefirst`

- **What:** With `leadmerge` only (ValueError otherwise): every joint mode is merged before the `modes<N>` cap, the classes are ordered by their summed probability and the N most likely kept
  (`acc_postprocess.build_multitarget_lead_prediction(merge_first=)`, `_pool_never_leading_modes`, `_rank_scenarios`).
- **Details:**
  - Before the product, the modes of a non-ego-lane target that are a lead candidate at no step (inactive, entry-excluded, or behind the ego wherever active) are pooled into one inactive copy with their summed probability
  - that gives the same classes and class probabilities as merging every joint mode (brute-force test) from far fewer joint modes (`MERGE_FIRST_MAX_JOINT_MODES` 512 guards the worst case).
  - Ego-lane targets keep their modes (`_vacating_modes` reads them).
- **Reason** (maintainer, 2026-11-03, final-check items 2 and 3):
  `leadmerge` merged only the 8 most likely joint modes, so duplicates could take the 8 places,
  and it kept each class at its most likely member's rank, so scenario 0 (the planned-speed memory, the warm start, `safebranch`'s reference) need not be the most likely class.
- **Logged:** The replay logs `mf_joint`, `mf_joint_raw`, `mf_classes`, `mf_s0_changed`, `mf_new`, `mf_lost`, `mf_dp` against the `leadmerge` way on the same targets.
- **Checks:** without the token bit-identical (12 NGSIM smoke + 3 lead-only episodes vs the `safebranch` root); 6 tests (`tests/test_mergefirst.py`).
- **Result:** NGSIM final set (λ 0, ideal plant, `replay_out/mergefirst_20261103/result.md`, root `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_20261103` + `lo_`, paired against the `safebranch` root):
  - `leadmerge`'s cap bound on 50.9 % of scored ticks (joint modes median 16, max 2,048) but cut a distinct lead sequence on 0.38 % only (321 ticks, 81 runs)
  - scenario 0 changed on 0.04 % (30 ticks, 16 runs)
  - after pooling 1.3 joint modes and 1.24 classes per tick (the new cap never bound).
  - Printed table cells unchanged except critical j_max 2.06 → 2.05 and no-cut-in runs a_min ≤ −2 8 → 7 (one run at −2.002 → −1.999)
  - contacts, fallbacks and floor breaches unchanged
  - one episode moves (ds6_v2292, mild, δ_max 11.8 → 7.6 %, through the class probabilities).
- **Status:**
  - `stdan_3int` only, off by default
  - whole-token match (`_parse_merge_first`)
  - not run in CARLA, not in the paper
    (added 2026-11-03, cf48239;
    part of the current design since the evening of 2026-11-03, maintainer's decision, §1;
    the replay's default tail ends `_mergefirst`)
    (updated 2026-11-12: the default tail has grown since and ends `_fsjerk10` since 2026-11-12, with `mergefirst` still in it, `ngsim_replay_defaults.md`; clarified 2026-11-12: "not in the paper" means the paper of 2026-09-10, see this file's header, while the paper's NGSIM tables have run with it since the evening of 2026-11-03, `replay_out/mergefirst_20261103/tables_paper_mergefirst.md`)

### `branchdec<a>`

- **What:** Splits a cut-in hypothesis from lane keeping at its vehicle's decision point instead of at the step its trajectory enters the ego lane.
- **Details:**
  - The decision step is Nair et al.'s traffic-light rule carried over: the TV closing on its own lead at Δv must decide by the gap `Δv²/(2a) + 7 m`, so `k_dec = ceil((gap − g_dec)/(Δv·dt))`, floored at 1 (`acc_postprocess.decision_point_branch_step`).
  - From that step a still-inactive cut-in mode adds `(vehicle, label, "decided")` to the scenario's effective-lead key, which splits the policy tree there; constraint cells are unchanged.
  - Since 2026-11-03 only a cut-in mode that leads at some step of the horizon gets the marker:
    a gated (p < 0.1), entry-excluded, never-entering or behind-the-ego one has lane keeping's cells, so its branch changed no solution and only added variables
    (maintainer's request; floors are set for every adjacent target with a lead, whatever its cut-in probability).
  - The agent collects every other vehicle's Frenet state to find each target's own lead and logs `stdan_debug.branch_decision`.
- **Result:** Measured 2026-10-18 with a = 7 on the paper grid: the split moves as designed (13/8/6/3/1 vs 13/12/10/6/3 at t_cross −3…−0.5 s) but every metric is unchanged.
  - For Proposed, by the time `k_dec` precedes lane entry the cut-in probability already exceeds β_ref (α = 1).
  - For `dopt100lkr_trajcutin`, α only scales the footprint and the w = 100 price demands 87–100 % of d_safe at the entry step almost regardless of P
    - in a two-scenario re-solve the shared u₀ then sits on the command jerk limit whether the split is at entry or at step 1, unless P < 0.3, and with `rmove1000` the split moves u₀ by ≤ 0.07 m/s².
  - The executed braking only shifts in time (0.2–0.3 m/s² less at t_cross −2…−1.5 s, up to 0.14 more after −1 s).
- **Status:** `stdan_3int` only, off by default, not in the paper

### `ctx<λ>`

- **What:** Context trust λ ∈ [0, 1] (e.g. `ctx0.5`), `stdan_3int` only:
  an adjacent vehicle's cut-in-direction intention is mixed with its measured lateral motion, `p_λ = p_S·[(1 − λ) + λ·o_v]`, before the ACC-mode mapping,
  so the gate, α, the `dopt` price, `trajcutin` and `branchdec` all read p_λ
  - the removed mass goes to the vehicle's lane-keeping intention and every trajectory stays STDAN's.
  - `o_v = clip((y_v − 0.10)/(0.85 − 0.10), 0, 1)` with y_v the vehicle's offset from its own lane centre (CARLA waypoint) toward the ego lane
    (`acc_postprocess.mix_cutin_probability`, `ACCNairSMPCAgent._lateral_offset_toward_ego_lane`; logged as `stdan_debug.processed_targets[*].ctx_*`).
  - λ = 0 is the plain policy, λ = 1 believes no cut-in before the vehicle moves laterally.
  - At p_λ = 0 the mode leaves the scenario set (zero-probability modes are skipped) rather than being gated.
- **Loss events:** The aggregator counts the design's two loss events per run,
  - `ctx_miss` (entered and δ > `--ctx-delta-thr` within 3 s after t_cross; default 0.2, not fixed yet)
  - and `ctx_unnec` (did not enter, a_cmd ≤ −0.3 while no measured in-lane lead within d_safe), for every policy,
  - and since 2026-10-20 (13ef7c5) the two run values the λ calibration's losses read:
    - `floor_breach` (the TV entered and the bumper gap fell below d0 = 3 m after t_cross, i.e. centre gap < L + d0, the floor D-opt holds hard for a vehicle in the lane; False without an entry)
    - and `phantom_brake` (runs without an entry: the largest braking command max(0, −a_cmd) over the ticks with no measured in-lane lead within d_safe, in m/s²; `ctx_unnec` is `phantom_brake ≥ 0.3`; None with an entry).
  - Both are None in cut-out groups.
- **Status:** No λ is calibrated yet: `ctx0.6`, from a predictor-level NGSIM calibration, made D-opt worse in CARLA; a calibration on closed-loop control outcomes (conformal risk control over a CARLA sweep) is running (§10) (added 2026-10-19, not in the paper)
  - (updated 2026-11-12: that CARLA calibration ran on 2026-10-20, `ctxcal_carla_20261020`, and no λ met both budgets; the NGSIM calibrations followed (Route B); `open_items.md`, "Context trust λ". The paper's NGSIM runs and the replay's default use λ 0, `ctx0`, the λ̂ of the final paper set)

### `latgate`

- **What:** With `ctx<λ>` only (`stdan_3int`): the offset ramp o_v is replaced by a latched lateral gate g_v ∈ {0, 1}, p_λ = p_S·[(1 − λ) + λ·g_v], the removed mass still going to lane keeping.
- **Gate:**
  - trigger_v(t) = (y_v > y0) ∧ (v_y ≥ 0.2 m/s) ∧ (p_S ≥ 0.9)
  - latched_v(t) = (latched_v(t − 1) ∨ trigger_v(t)) ∧ (y_v > y0)
  - the latch also clears at any tick the vehicle is not processed as an adjacent target, so a vehicle that becomes adjacent again starts unlatched.
- **Inputs:**
  - y0 = `CTX_LATERAL_NOISE_M` (0.10 m in CARLA, 0.760 m in the NGSIM replay),
  - p_S the raw cut-in-direction intention (`ctx_p_raw`),
  - v_y the lateral speed toward the ego lane averaged over the last 0.5 s from a *continuous* lateral position — never a difference of y_v, which jumps by a lane width when the lane assignment flips (fake 7–11 m/s in the offline NGSIM analysis).
- **Lateral speed:**
  - CARLA: every tick each target's velocity is projected onto the left normal of its own-lane waypoint heading in the right-handed frame (`ACCNairSMPCAgent._lateral_speed_left`, recorded whatever the vehicle's relation),
    and v_y is the mean over (t − 0.5 s, t] with y_v's sign flip (`_lateral_speed_toward_ego_lane`; unknown, i.e. no trigger, with < 0.45 s of history)
  - neither the ego path's Frenet d (it shifts ~1.5 m against the adjacent lanes near the Town04 segment boundary, §8.6) nor a world-position chord (R(1 − cos θ) on curves) is used.
  - NGSIM replay (`run_replay.py --latgate`): v_y = side·(x_causal[f] − x_causal[f − 5])·FT2M / 0.5 s from `geom_causal_ds*.npz`, unknown without a row at f − 5.
- **Constants** `CTX_GATE_P_MIN` 0.9, `CTX_GATE_VY_MIN_MPS` 0.2, `CTX_GATE_VY_WINDOW_S` 0.5, `CTX_GATE_VY_MIN_HISTORY_S` 0.45 (`acc_postprocess.py`, `mix_cutin_probability` stays pure: the agent owns the per-vehicle speed history and latch)
  - logged as `ctx_v_y`, `ctx_trigger`, `ctx_latched` next to the other `ctx_*` fields, with `ctx_o_v` the factor used (0/1).
- **Checks:** Without the token `ctx<λ>` is bit-identical to before (regression test over a grid of inputs).
- **Offline on NGSIM** (open-loop p_S, y0 0.760, 2026-10-21):
  - the trigger fires before the crossing in 382/400 cut-ins (lead p10/p50 0.6/1.3 s) and in 126/2,175 S2b and 4/400 of all no-cut-in episodes, against 399/400, 391 and 32 for the ramp's y_v > y0 alone
  - the open-loop replay reproduces the offline first-trigger tick in 60/60 checked episodes (p_raw = p_S to 1e-6).
- **CARLA smoke** (`latgate_smoke_20261021`, ego 17 / TV 15, λ = 1, 0 fallbacks):
  - the TV triggers 1.30 s before t_cross, 0.4 s after it starts steering (y_v 0.13 m, v_y 0.34 m/s, p_S 0.96), and stays latched until it is reclassified ego-lane
  - a_min_cmd −3.0 m/s², δ_max 46 %
  - the no-cut-in run never triggers (y_v ≤ 0.00 m, p_S ≤ 0.73).
- **Status:** No λ calibrated with it yet (added 2026-10-21, not in the paper)

### `rmove<w>`, `rjerk<w>`

Weight on the step-0 command change; jerk weight

### `ra<w>`

- **What:** The weight r_a of r_a·(u − a_ref)², with a_ref = 0 the only cost on the command's size (default 1, `original_nair_acc_config`); `ra0` drops it.
- **Status:** Whole-token match, so `trajcutin` and `cutin_ramp` never read as it (added 2026-10-29; `ra0` is in the paper's NGSIM table, with `jinput` since 2026-10-29 evening, §1)

### `alag<τ>`

- **What:** A first-order lag from the commanded to the actual acceleration in the prediction model, τ·da/dt + a = u (`NairACCConfig.accel_lag_tau`; Yoon et al., IEEE Access 2021, eq. 25, use τ = 1 s at dt 0.1 s with forward Euler).
- **Details:**
  - The cached feedback QP propagates [s, v, a] with the exact zero-order-hold discretisation over dt (`acc_lag_coefficients`)
  - a₀ is `solve(ego_accel=)`, the previous command without it (exact for CARLA's ideal actuator, which has no lag).
  - The bounds, the jerk rows and every cost stay on the command u
  - `_rollout_policy` and `_fallback_policy` use the same model, so `x_nominal` (entrypass, the planned-speed reference) is consistent.
- **Checks:** Without the token the controller is bit-identical (12 NGSIM final-set episodes, max |Δa_cmd| = 0).
- The NGSIM replay gives its simulated ego the same lag with `NGSIM_EGO_LAG_TAU` (§10).
- **Status:** Cached feedback QP only (added 2026-10-29, b629db4; `alag0.5` was in the paper's NGSIM table on 2026-10-29 afternoon only, §1)

### `jinput`

- **What:** The jerk as the MPC input (`NairACCConfig.jerk_input`; whole-token match, so `_jinput` only).
- **Model:** The state is [s, v, a],
  the decision variable per step and scenario is the jerk j_k held over dt = 0.2 s with the exact triple-integrator discretisation (`acc_jerk_step`),
  and the cached QP's sharing tree is on j (j_0 shared, so the issued command is common to every scenario).
- **Constraints:**
  - |j_k| ≤ jerk_limit replaces the jerk rows on the input differences (the step-0 row against the previous command included)
  - the acceleration bounds become state constraints on a_1..a_N
  - the speed bounds, the safety rows, the slack, the `dopt` / `lkr` prices, the chance and σ terms and `entrypass` are unchanged.
- **Costs:** r_jerk·j_k² on every step in place of r_jerk·((u_k − u_{k−1})/dt)² and r_move·(u_0 − u_prev)²
  — `rmove<w>` has no role with it (the string may still carry it) —
  and r_a·(a_k − a_ref)² on the planned accelerations, keeping the cached QP's ×3 convention (§3).
- **Command:** The output stays an acceleration command, a_0 + j_0·command_dt (PC-ACC's `_command_from_jerk`):
  - a_0 is `solve(ego_accel=)`, the previous command without it, clamped to ≥ 0 at v_min (a stopped car is not asked to keep decelerating; from v = 0 with a_0 < −0.5, v_1 ≥ 0 would need j_0 > 5, the fallback the `alag` runs showed at standstill)
  - not at v_max, where a car is not held (removed 2026-10-29 evening, maintainer).
- **Plans and logs:**
  - `u_nominal` keeps the planned accelerations a_1..a_N so its readers still see accelerations,
  - `NairACCSolution.j_nominal` holds the jerk plan,
  - the agent logs `jerk_cmd` (= j_0) per step and `j_nominal` in the reference debug,
  - and the warm start is the shifted previous jerk plan
  - `_rollout_policy` and `_fallback_policy` use the same model, so `x_nominal` stays consistent.
  - The agent's and the replay's command limiters stay (on the 12-episode NGSIM smoke they never bind).
- **Checks:** Without the token the controller is bit-identical (12 NGSIM final-set episodes, max |Δa_cmd| = 0 against `final_v20s0_ep0_ncls_paper_20261028`).
- **Reason** (maintainer, 2026-10-29): on the ideal replay the design's one-tick j_max was above ISA-MPC's,
  and the issued command change was priced r_move·Δu² = 10·j² per 0.1 s tick against r_jerk·j² = 40·j² inside the plan;
  PC-ACC (jerk input) and ISA-MPC (snap input) already plan this way.
- **Spec, prereg and result:** `replay_out/jinput_20261029/`.
- **Status:** Cached feedback QP only; ValueError with `alag<τ>` (added 2026-10-29; in the paper's NGSIM table since that evening, §1)

### `jlim<x>`

- **What:** The hard jerk bound [m/s³] (`NairACCConfig.jerk_limit`, default 5) on the first input over the command period, on every planned input change and on the issued command.
- `rmove0_jlim1.5` replaces the step-0 move cost with upstream Nair's ±1.5 m/s³ rate bound (hard here, soft upstream).
- **Probe:** A 2-episode NGSIM probe (2026-10-22) showed that `rmove1000` (step 0 priced at 4× an in-plan change, 1000 vs r_jerk/dt² = 250) postpones both the braking build-up and its release, so the ego re-accelerates late after a cut-in
  - the bound keeps the mild case's release prompt but builds an aggressive cut-in's −3 m/s² 0.4 s later.
- **Status:** The maintainer chose it for the 2026-10-22 re-run (§10) (added 2026-10-22, not in the paper)
  - (updated 2026-11-12: superseded the same day: the NGSIM calibration of `rmove0_jlim1.5` was stopped at 360 of 5,600 jobs, `latgate_cal_jlim_20261022`, and `rmove1000_rjerk40` was adopted instead; `open_items.md`, "Jerk pricing and the re-run". Since 2026-11-12 `jlim2.5` is in the NGSIM design as the falling-side bound of the unified limits, `amin<x>`, `amax<x>`, `jlimup<x>` below)

### `amin<x>`, `amax<x>`, `jlimup<x>`

- **What:** Unified limits (2026-11-11, maintainer: every controller's bounds by ISO 15622 in the NGSIM replay; `replay_out/iso_limits_20261111/iso15622_limits.md` and `plan.md`).
  - `amin<x>` / `amax<x>`: the comfort acceleration bounds `a_min = -x` / `a_max = x` (default −3 / 2, `original_nair_acc_config`; PC-ACC's token names).
  - `jlimup<x>`: `NairACCConfig.jerk_limit_up`, the bound on the rising side of the acceleration (positive jerk: building up acceleration, releasing the brake); `jlim<x>` then bounds the falling side only (ISO 15622 limits only the negative jerk).
- **Scope:** The up bound replaces `jerk_limit` as the upper bound
  - of `jinput`'s `j_k`,
  - of the input-change rows,
  - of the warm start, rollout and fallback,
  - of the comfort part of `failsafe_recovery`'s envelope and window,
  - and of the callers' command limiter, now one function, `acc_nair_smpc.limit_accel_command` (`ACCNairSMPCAgent._limit_accel_command` delegates to it, and so does the replay's D-opt limiter since its edit of the night of 2026-11-11).
  - ISA-MPC reads the same tokens through its parent's `controller_config` (`ISAMPCPlanner(jerk_limit_up=)`: the snap chain's jerk state, the acceleration-input rows, `_clip_jerk`).
- **Parsing:**
  - `ACCNairSMPCAgent._parse_limit_tokens` (whole tokens, magnitudes; also the fail-safe tokens below)
  - the NGSIM replay reads them since the night of 2026-11-11 (`iso_limits_20261111/replay_edits.patch`, applied after the merge):
    - D-opt from its policy string,
    - ISA-MPC from `NGSIM_ISAMPC_POLICY` (default `acc_isampc`; recorded as `isampc_policy` in run_info.json) (updated 2026-11-12: the default is `acc_isampc_amin3.5_amax2_jlim2.5_jlimup5` since 2026-11-12, `replay/baselines.py`)
  - without the tokens and the variable the patched replay reproduces the 2026-11-11 roots of
    D-opt (with `NGSIM_DOPT_PREDICTOR=iaimm` too),
    ISA-MPC (slack 5 and 0.055),
    PC-ACC (port, as published, `physcheck_reachfix`)
    bit for bit on the 13 smoke episodes (3 lead-only),
    and so does every token set to today's numbers (`iso_limits_20261111/check_after_merge.out`)
    (clarified 2026-11-12: "today's numbers" are the fork's limits of 2026-11-11: D-opt with `_amin3_amax2_jlim5_jlimup5_fsdecel8_fsjerk20`, ISA-MPC `acc_isampc_amin3_amax2_jlim5_jlimup5`, PC-ACC `acc_pcacc_amin3_amax2_jlim5_jlimup5` and `acc_pcacc_physcheck_reachfix_jlimup5_emamin10_emjlim10`; `iso_limits_20261111/check_after_merge.sh`).
- **Checks:** Without the tokens every problem is the one of before: the full suite, and every command and plan of the controllers against 23b2866 as float hex (`iso_limits_20261111/check_units_bitident.py`, `bitident_units.log`).
- **Status:**
  - Off by default in the code
  - the NGSIM replay's default since 2026-11-12 (D-opt's default tail, `NGSIM_ISAMPC_POLICY`, `NGSIM_PCACC_POLICY`; §10)
  - first runs on the final set on the night of 2026-11-11 (`replay_out/iso_limits_20261111/`: P1 and the baselines at the unified limits) (added 2026-11-11, not in the paper)

### `fsdecel<x>`, `fsjerk<x>`

- **What:** With `safelayer` only (ValueError otherwise): the safety layer's fail-safe, deceleration `a_em = -x` [m/s²] and constant jerk `x` [m/s³], in place of `FAILSAFE_DECEL` −8 / `FAILSAFE_JERK` 20
  (`NairACCConfig.failsafe_decel` / `failsafe_jerk`, `SafetyLayer(failsafe_decel=, failsafe_jerk=)`, `safety_layer_config`;
  Spec. III.5 still needs a_p = −10.5 ≤ a_em, and `failsafe_decel` ≤ `a_min`).
- The MPC's uses follow (`_failsafe_limits(config)`):
  - the start acceleration's lower clip,
  - `failsafe_recovery`'s envelope, floor and window,
  - and `safebranch`'s branch limits and terminal row (`failsafe_terminal_coefficients`, which needs a_max ≤ |a_em|/2; checked with `safebranch`).
- **Numbers:** At −8 / 10
  - the layer's guaranteed bumper gap at equal speeds is 6.3 / 10.6 / 15.7 m at 10 / 15 / 20 m/s (4.4 / 7.8 / 11.9 at jerk 20),
  - the recovery ramp from −8 to a_min −3.5 takes 0.45 s,
  - the recovery floor equals a_em from 3.8 m/s up (1.6),
  - and the branch's c_T / D_0 are 2.25 s / 5.3 cm (1.75 / 1.3).
- The CARLA agent builds the layer with them, and the replay since its edit (`iso_limits_20261111/replay_edits.patch`).
- **Status:**
  - Off by default in the code
  - the NGSIM replay's default since 2026-11-12 (D-opt's default tail, §10)
  - first run on the final set on the night of 2026-11-11 (P1 of `replay_out/iso_limits_20261111/`) (added 2026-11-11, not in the paper)

### `open_loop`, `multimodal_ol`, any `_ol`

Open-loop variant: one input sequence shared by all scenarios, and one scalar slack

### `no_k`, `with_k`

Disable or enable K-feedback optimisation (off by default)

### `scalar_chance`, `optimized_eta`, `probability_weighted`

Chance-constraint and risk-allocation variants not used in the paper

### `brake_distance`, `brake_hard_band`

Braking-distance safety bound (experimental)

### `cutin_ramp<β>`, `cutin_tlc<s>`, `gap_recover<s>`, `intent_lpf<s>`, `track_smooth<n>`

- Other options, off by default and not used in the paper.
- Combining `cutin_ramp` with `cutin_chance` raises `ValueError`

### `fixed_risk`, `nominal_safe_distance`

- Labels only.
- `fixed_risk` sets the variant name and changes nothing in the solve.
- `nominal_safe_distance` is never parsed: the deterministic safety distance is simply what you get when no chance token is present.
- Every paper policy carries both

### `onerow`

(Record added 2026-11-12 from `replay_out/onerow_20261112/agents_row.md` and `result.md`.)

- **What:** One row on every active lead cell, cut-in hypotheses and lane-keeping leads alike (maintainer, 2026-11-12).
  - The row is gap ≥ L + d0 + τv_k − short_k − ξ_k with 0 ≤ short_k ≤ τv_k, priced at w·P_j·short² (`dopt<w>`'s weight), so L + d0 is the hard floor through the cap.
  - The factor on L + d0 is 1 on every cell (`NairACCSMPC._one_row_prediction`), and no row has a σ term (`_cell_tightening` returns 0).
  - There is no floor row, no separate hard lane-keeping row and no `lkr` logic; every active cell is priced (`NairACCSMPC._priced_cells`). The ℓ1 slack stays.
  - The reference asks every cell for the plain d_safe at the planned speed, L + d0 + τv̄_k, with no remembered standoff and no σ term (`OldACCReferenceAdapter.generate`).
- **Code:**
  - `NairACCConfig.one_row`; whole-token match (`_parse_one_row`).
  - Needs plain `dopt<w>` (not `lk` / `lkr`), the chance cells (`chancecells` or `cutin_chance<β>`) and `fullfoot`.
  - ValueError, from the agent and the config, with `floorrisk<ε>`, `floorhyp`, `floortop`, `nopricesigma`, `lksigma` (agent only), `lkonesided<ε>`, `lkrsigma`, `cvar<ε>_<c>`, `passmiqp<m>`, `relbrake<b>` or `safebranch`.
  - The NGSIM replay reads the token since a one-line edit of `build_controller` (`cfg.one_row = A._parse_one_row(policy)`, `replay_out/onerow_20261112/replay_edit.patch`; `ngsim_replay_defaults.md`, 2026-11-12).
- **Checks:**
  - without the token bit-identical (13 smoke + 3 lead-only episodes against `fin_npw3_iso_side_20261112` / `lo_`, every non-timing tick array, on the workstation and on the lab server)
  - the onerow smoke bit-identical between the two machines
  - 7 tests (`tests/test_onerow.py`); the full suite passes (494 tests).
- **Result, w 3:** NGSIM final set (`NGSIM_POLICY_DOPT=_dopt3`, the default tail without `_lksigma`, `_lkrsigma`, `_floorrisk0.1`, `_floorhyp`, `_nopricesigma`, `_lkonesided0.05`, plus `_onerow` (pointer updated 2026-11-12 night: P1's tail, the default when this was written); roots `fin_npw3_iso_side_onerow_20261112` + `lo_`; `replay_out/onerow_20261112/result.md`), paired against the current design (P1, live side entries):
  - the ego follows closer: behind non-TV leads 3.1 / 12.8 / 14.2 % of lead ticks inside d_safe against 0.6 / 0.1 / 0.3 % (critical / mild / no cut-in; median deficit 0.9 / 0.7 / 0.5 m), no-cut-in v_avg +0.18 m/s (SE 0.01)
  - critical / mild δ_max +6.4 / +6.3 %p (SE 0.7 / 0.8), floor breaches 21 → 26 / 14 → 19, T_rec +0.14 / +0.19 s
  - j_max −0.04 (SE 0.11) / −0.12 (0.04), runs at the falling jerk bound 50 → 38 / 56 → 51
  - phantom_ep −0.007 (SE 0.008)
  - contacts 13 / 12 / 7 → 14 / 11 / 4 (all side entries)
  - one new safety-layer fail-safe run behind a hard-braking lead (ds1_v2886, no cut-in: the gap fell from d_safe to 4.0 m at −0.6 to −1.2 m/s² before the layer acted, min gap 1.28 m against 5.84 m).
  - Against the ℓ1-slack-100 arm it brakes harder in every group (critical a_min −0.24, SE 0.04; falling-bound runs 8 → 38).
- **Result, w 100:** `NGSIM_POLICY_DOPT=_dopt100`, nothing else changed (roots `fin_npw3_iso_side_onerow_w100_20261112` + `lo_`; `result.md` addendum), paired against the current design:
  - critical / mild j_max +0.66 / +0.35 (SE 0.13 / 0.07), a_min −0.47 / −0.22 (0.05 / 0.03), runs at the falling jerk bound 50 → 86 / 56 → 75,
    mostly before t_cross while the TV's hypothesis is active (`bound_split.md`: those bound ticks 151 → 397 / 172 → 326; post-entry recovery 135 → 148 / 134 → 100)
  - δ_max −3.5 / −3.7 %p (SE 1.8 / 0.9), T_rec −1.09 / −0.57 s (0.09 / 0.07; not recovered 17 → 1 / 13 → 1), floor breaches 21 → 26 / 14 → 16
  - no cut-in about unchanged (phantom_ep −0.001, SE 0.008; v_avg +0.13 m/s, SE 0.01)
  - behind non-TV leads 1.9 / 6.0 / 6.0 % of lead ticks inside d_safe (median deficit 0.58 / 0.29 / 0.18 m).
- **Result, w sweep** (added 2026-11-12 night): `NGSIM_POLICY_DOPT=_dopt10` / `_dopt30`, nothing else changed (roots `fin_npw3_iso_side_onerow_w10_20261112` / `_w30_` + `lo_`; `result.md` addendum 2, `tables_onerow_wsweep.md`, `headline_onerow_wsweep.md`):
  - with w 3 → 10 → 30 → 100: critical δ_max 38.6 → 34.3 → 30.9 → 28.6 % (P1 32.1), T_rec 1.86 → 1.38 → 1.00 → 0.69 s (1.69), j_max 1.27 → 1.50 → 1.74 → 1.98 (1.31), runs at the falling jerk bound 38 → 41 → 59 → 86 (50), floor breaches 26 / 25 / 25 / 26 (21)
  - mild δ_max 24.3 → 20.5 → 17.3 → 14.3 % (18.0); no-cut-in phantom_ep 0.088–0.093 (0.095), v_avg +0.13 to +0.18 m/s (paired, w 100 to w 3)
  - behind non-TV leads in no-cut-in runs 14.2 / 10.2 / 8.0 / 6.0 % of lead ticks inside d_safe (0.3)
  - against P1, w 10: j_max within SE, δ_max +2.1 / +2.5 %p (SE 1.0 / 0.7), critical T_rec −0.44 s (0.06); w 30: δ_max within SE (−1.3 / −0.7 %p), T_rec −0.81 / −0.37 s (0.08 / 0.07), critical j_max +0.43 (0.12), a_min −0.28 (0.04)
  - the same five TV entries breach the 3 m floor for every w (ds1_v2629, ds1_v2785, ds3_v1481, ds4_v3230, ds5_v2583), where P1 kept 3.1–4.9 m.
  - The maintainer then chose w 10, kept `nocutout`, and put the σ term back (`zconf<β>`, below).
- **Status:**
  - The maintainer decided on 2026-11-12, after the w 3 roots, that `onerow` will be the final design (the old row structure is not clean); `result.md`'s conclusion "Not adopted; decision pending" was written before that decision, which its addendum records.
  - Its weight w is being tuned: w 10 and w 30 (w sweep in progress, 2026-11-12; `replay_out/onerow_20261112/run_server_onerow_wsweep_20261112.sh`) (updated 2026-11-12 night: done, w 10 chosen; "Result, w sweep" above).
  - Not the replay's default yet (that is still P1) (updated 2026-11-12 night: the replay's default since the night of 2026-11-12, in the final design below).
  - In the final NGSIM design since the night of 2026-11-12 (maintainer: "C + snap160 is the final NGSIM design"), as `dopt10` with `zconf0.95`, `nocap` and `snap160` (records below; string AGENTS.md §1; `ngsim_replay_defaults.md`, 2026-11-12).
  - Not run in CARLA; off by default; not in the paper (added 2026-11-12, 3d9bcf8).

### `zconf<β>`

(Record added 2026-11-12 night from `replay_out/onerow_20261112/agents_row.md` and `result.md`, addendum 3; run as `zconf0.95`, β the cap of the confidence level.)

- **What:** With `onerow` only (ValueError otherwise): each active cell's row gets back a σ term (maintainer, 2026-11-12: w 10, keep `nocutout`, the σ term back in the one row).
  - z = Φ⁻¹(min(p, β)), with p the probability of the cell's own lead mode (`MultimodalLeadPrediction.lead_mode_probability`, BCTS-calibrated in the design) and σ the cell's `safety_std`.
  - The row becomes gap ≥ L + d0 + τv − short − ξ + zσ; with the cap 0 ≤ short ≤ τv the floor is L + d0 + zσ.
  - Lane-keeping leads (p = 1 under `nocutout`) get z = 1.645 at β 0.95. A cut-in hypothesis with p < 0.5 gets a negative z, not clamped (p 0.3 → −0.524).
  - The reference keeps the plain d_safe (`zref`, the same term in the reference, was dropped before any run).
  - A prediction without cell probabilities and without a cut-in chance cell (the replay's no-target placeholder) uses p = 1; one with a cut-in chance cell and no probabilities raises. A first workstation sanity run raised on 8 ticks of that placeholder before this rule.
- **Code:**
  - `NairACCConfig.one_row_z_conf`, `acc_nair_smpc.one_row_sigma_term` (used by `_cell_tightening`), the agent's `_parse_one_row_z_conf` (whole-token match; β in (0, 1)).
  - The NGSIM replay reads it since a one-line edit of `build_controller` (`cfg.one_row_z_conf = A._parse_one_row_z_conf(policy)`, `replay_out/onerow_20261112/replay_edit_zconf.patch`; `ngsim_replay_defaults.md`, 2026-11-12).
- **Checks:**
  - 7 tests (`tests/test_zconf.py`: z for p 1 and 0.3, the lane-keeping row and its floor, the cut-in row with negative z, the reference unchanged, the placeholder, the ValueErrors); the full suite passes (501 tests).
  - Without the token bit-identical to P1 (13 smoke + 3 lead-only episodes, strict) and tick-level identical to onerow w10 (results.csv logged: a 1-ulp splice round trip, and one root value of ds6_v2044 that is not the maximum of its own tick array), on the workstation and on the lab server.
  - Arm A's smoke (5 + 3 episodes) bit-identical between the two machines.
- **Result** (arm A = onerow w10 + `zconf0.95`; roots `fin_npw3_iso_side_onerow_w10_z95_20261112` + `lo_`; `result.md` addendum 3, `tables_onerow_z95.md`, `headline_onerow_z95.md`), against onerow w10:
  - behind non-TV leads the ticks inside d_safe fall from 2.7 / 10.1 / 10.2 % to 1.1 / 1.4 / 0.8 % (critical / mild / no cut-in)
  - mild δ_max −4.2 %p (SE 0.4); critical / mild T_rec −0.26 / −0.27 s; critical j_max +0.13 (0.04)
  - the pre-entry braking load stays onerow w10's (falling-bound ticks before t_cross with the TV hypothesis active 172 / 176 against 173 / 170).
  - Against P1: critical j_max +0.31 (0.11), a_min −0.17 (0.03), δ_max +0.5 %p (within SE), T_rec −0.66 s (0.06); mild δ_max −1.7 %p (0.6), T_rec −0.36 s (0.05); floor breaches 21 → 25 / 14 → 16; no cut-in phantom_ep +0.001 (0.004), v_avg +0.08 m/s.
  - Of the five TV entries that every earlier onerow arm breached, A no longer breaches ds1_v2785 and ds5_v2583; the other three are at 2.76–2.87 m.
- **Status:** in the final NGSIM design since the night of 2026-11-12 (maintainer), as `zconf0.95`; not run in CARLA; off by default; not in the paper (added 2026-11-12, 7ef3da2).

### `nocap`

(Record added 2026-11-12 night from `replay_out/onerow_20261112/agents_row.md` and `result.md`, addendum 4.)

- **What:** With `onerow` only (ValueError otherwise): no upper bound on an active cell's shortfall (maintainer, 2026-11-12: remove the cap short ≤ τv).
  - The row is gap ≥ L + d0 + τv + zσ − short − ξ with short ≥ 0, priced w·P_j·short²; inactive cells keep short 0. No row is hard any more.
  - The ℓ1 slack (`l1slack1e6`) stays in the string, but no slack cell (ξ > 1e-4 m) was logged on any solved tick of the `nocap` roots (C, `snap40`, `snap160` and their lead-only roots); the only ones are on ds4_v3035's fallback ticks under `snap<r>`, where no QP solution exists (checked 2026-11-12 from the tick logs).
- **Code:**
  - `NairACCConfig.one_row_no_cap`: in the cached QP the cap row becomes (1 − mask)·short ≤ 0.
  - The agent's `_parse_one_row_no_cap` (whole-token match; `passnocap` does not match).
  - The NGSIM replay reads it since a one-line edit of `build_controller` (`replay_out/onerow_20261112/replay_edit_nocap.patch`).
- **Checks:**
  - 4 tests (`tests/test_nocap.py`: the cap gone with the token and kept without it, inactive cells, the ValueError); the full suite passes (505 tests).
  - Without the token bit-identical to P1 (13 + 3 episodes, strict) and tick-level identical to arm A (results.csv: 1-ulp differences from the splice round trip), on the workstation and on the lab server; arm C's smoke bit-identical between the machines.
- **Result** (arm C = A + `nocap`; roots `fin_npw3_iso_side_onerow_w10_z95_nocap_20261112` + `lo_`; `result.md` addendum 4, `tables_onerow_nocap.md`, `headline_onerow_nocap.md`, `belowfloor_nocap.md`):
  - against A: falling-bound runs 43 → 23 / 47 → 19 / 35 → 8 (critical / mild / no cut-in); a_min +0.15 / +0.15 / +0.07; no-cut-in phantom_ep 0.096 → 0.042 (SE 0.014); T_rec +0.08 / +0.06 s; mild floor breaches 16 → 22
  - the falling-bound ticks before t_cross with the TV hypothesis active fell from 172 / 176 to 43 / 27 (critical / mild): most of A's braking at the jerk bound came from the hard floor L + d0 + zσ that the cap created
  - against P1: T_rec −0.57 / −0.29 s, mild j_max −0.22 (0.09), phantom_ep −0.052 (0.014), δ_max within SE, floor breaches 21 → 25 / 14 → 22
  - the measured step-0 bumper gap fell below 3.27 m (the lowest lane-keeping floor, d0 + 1.645·0.164 m) on 283 / 200 / 173 scored ticks, against 228 / 77 / 50 under A (a lower bound).
  - Behind a TV at its entry the ego now accepts the shortfall instead of braking; the safety layer is the only hard protection.
- **Status:** in the final NGSIM design since the night of 2026-11-12 (maintainer); not run in CARLA; off by default; not in the paper (added 2026-11-12, 506bfcb).

### `snap<r>`

(Record added 2026-11-12 night from `replay_out/onerow_20261112/agents_row.md` and `result.md`, addendum 5.)

- **What:** With `jinput` only (ValueError otherwise): a cost on the jerk's changes, 3·P_j·r·Σ_{k=0}^{N−1} (j_k − j_{k−1})² per scenario (the cached QP's ×3 convention; in (m/s³)², not divided by dt).
  - j_{−1} is the jerk the plant got on the last tick, (a_cmd[t−1] − a_cmd[t−2]) / command_dt, from the issued commands given to `solve`; 0 until there are two.
  - At r = 40 = r_jerk a jerk step costs as much as holding that jerk for one step. Run as `snap40` and `snap160`.
  - Why (coordinator, 2026-11-12): in arm C the issued jerk stepped at probability jumps (24 pre-entry falling-bound bursts where the TV's p jumps about 50 → 76 % 0.8 s before t_cross, 16 behind the TV at or after its entry); the run maximum of the one-tick snap was 5.6 / 18.6 m/s⁴ (critical median / p90), against 2.1 / 3.6 for ISA-MPC.
- **Code:**
  - `NairACCConfig.snap_weight`, a QP parameter for j_{−1} (`NairACCSMPC._set_command_context` computes it from the last issued commands), the agent's `_parse_snap` (whole-token match).
  - The NGSIM replay reads it since a one-line edit of `build_controller` (`replay_out/onerow_20261112/replay_edit_snap.patch`).
- **Checks:**
  - 4 tests (`tests/test_snap.py`: the objective at the solved plan equals the base terms plus 3·r·Σ(j_k − j_{k−1})², j_{−1} from the issued commands and 0 on the first tick, the first jerk pulled toward j_{−1} as r grows, the parser and the ValueErrors); the full suite passes (509 tests).
  - Without the token bit-identical to P1 (strict) and tick-level identical to arm C, on 13 + 3 episodes, on the workstation and on the lab server; both smokes bit-identical between the machines.
- **Result** (roots `fin_npw3_iso_side_onerow_w10_z95_nocap_snap40_20261112` / `_snap160_` + `lo_`; `result.md` addendum 5, `tables_onerow_snap.md`, `headline_onerow_snap.md`, `snap_diag.md`), `snap40` / `snap160`:
  - against C: critical j_max −0.23 / −0.35 (SE 0.07), a_min +0.03 / +0.07; mild j_max −0.11 (0.04) / −0.13 (0.09, within 2 SE); no cut-in j_max −0.05 / −0.09; `snap160` critical T_rec +0.05 s (0.02), no-cut-in v_avg −0.05; δ_max, T_rec and phantom braking otherwise at C's levels
  - run maximum of the one-tick snap, critical median / p90: 5.6 / 18.6 (C) → 3.6 / 11.3 / 2.3 / 7.1 m/s⁴ (ISA-MPC 2.1 / 3.6); the critical j_max gap to ISA-MPC +0.53 → +0.31 / +0.18
  - critical falling-bound bursts (pre-entry / at TV-becomes-lead / behind the TV later / other): 11 / 6 / 4 / 8 (C), 6 / 6 / 5 / 6, 4 / 4 / 2 / 3
  - floor breaches (critical / mild): 25 / 22 (C) → 23 / 21 / 22 / 20 (P1 21 / 14)
  - contacts against C: `snap40` 3 new and 2 removed, `snap160` 4 new and 3 removed, all side entries; the TV side entry ds4_v3230 (mild, at t_cross at 1.0 m/s): min gap 0.20 m (C), −0.03 m, −0.33 m (P1 3.34 m).
- **Status:** `snap160` in the final NGSIM design since the night of 2026-11-12 (maintainer); not run in CARLA; off by default; not in the paper (added 2026-11-12, f1bfbc4).

### `aheadonly`

(Record added 2026-11-13 from `replay_out/aheadonly_20261113/`.)

- **What:** ISA-MPC's obstacle rule (Zhou et al.'s `ISA_MPC.py`, `Obst_k[i][0,0] >= x_EV_k`; the port's `isampc_agent.py` `plan()`): a target is an obstacle only while its current position is at or ahead of the ego's (in the replay its rear + L against the ego's front, the ISA-MPC port's comparison). Every mode of a target behind it never leads, whatever its prediction.
  - Without the token the lead filter compares each predicted step with the ego's *current* position, so a faster vehicle beside or behind the ego leads at the steps it is predicted past that position. In ds1_v2379 the TV's hypothesis, 1.6 m behind the planned ego at step 3, asked for a 29.5 m shortfall.
  - Why (maintainer, 2026-11-13, after the ds1_v2379 analysis): obstacles are only the vehicles ahead of the ego, as in ISA-MPC.
- **Code:**
  - `acc_postprocess.build_multitarget_lead_prediction(ahead_only=)`: the target's modes join the never-leading set (`entry_excluded`, so `mergefirst` pools them); every scenario's metadata lists the targets under `ahead_only`.
  - `STDAN3IntACCAdapter.predict_acc(ahead_only=)`; the agent's `_parse_ahead_only` (whole-token match; needs `stdan_3int`).
  - The replay's `predictor_options` and its three direct calls (lead-only, the `mergefirst` diagnostic, the counterfactual) pass it; with the token only, tick keys `ao_n` (targets behind the ego now) and `ao_tv` (the TV one of them). Backup `replay_backup_20261113_pre_aheadonly/`.
- **Checks:**
  - 7 tests (`tests/test_aheadonly.py`: a TV beside the ego never leads and is listed; the rule equals that target's modes switched off by hand, plain, `leadmerge` and `mergefirst`; a target at or ahead of the ego is unchanged; release once ahead; off by default; the parser and the ValueError); the full suite passes (516 tests, `full_test_suite.out`).
  - Without the token bit-identical to the final root on 13 + 3 episodes on the workstation (max |Δa_cmd| 0, every non-timing tick array equal; `gate_aheadonly.log`).
  - With it, 14 smoke episodes (the 13 + ds1_v2379), valid (0 errors, exceptions, fallbacks).
- **Smoke result (14 episodes, not a measurement):**
  - 12 unchanged, and ds4_v3215 (no cut-in) a_min −0.15 → −0.09.
  - ds1_v2379: the TV is excluded until 0.1 s before t_cross; braking starts +1.0 s after t_cross instead of −1.8 s, then still at the falling jerk bound (a_cmd min −3.47 against −3.42); the open-loop TV still enters beside the ego, so the side contact stays.
- **Final set (2026-11-13):** root `fin_npw3_iso_side_onerow_w10_z95_nocap_snap160_aheadonly_20261113` (889 episodes on the workstation, 683 s; 0 errors, 1 fallback run as the final root; `run_final_aheadonly.{sh,log}`). Tables `tables_aheadonly.md` (a copy of `onerow_20261112/tables_onerow_snap.py`; its final-design and ISA rows equal the old pickle on 56 columns).
  - Paired against the final design over 882 episodes (SE in brackets):
    - critical: j_max −0.05 (0.02), δ_max −3.2 %p (1.4), floor breaches 22 → 17, contacts 14 → 10;
    - mild: j_max −0.04 (0.01), contacts 12 → 11;
    - no cut-in: contacts 5 → 2, phantom_ep −0.025 (0.008), v_avg +0.05 m/s (0.02);
    - T_rec +0.04 s (0.03), within SE.
  - No contact and no floor breach appears only with the token. The 8 contacts it removes are all side entries, and in its runs the ego was clear of those vehicles at that frame.
  - Side-overlap runs went 15 / 16 / 7 → 14 / 15 / 5, and TV side entries stayed 8 / 14. My prediction that the token would add side entries was wrong.
  - The lead-only loop was not re-run, because the token cannot drop its single target, the in-lane lead ahead.
- **Status:** decided by the maintainer on 2026-11-13; implemented (793aba3); final set run (2026-11-13), the design row of the NGSIM main table (AGENTS.md §1); off by default and not in the replay default yet; not run in CARLA.

## External baselines

Two external baselines (added 2026-10-19, smoke-tested on one run each, no sweep yet) (updated 2026-11-12: both have since run in the CARLA verification sweep of 2026-10-20, `cutin_verify_20261020` / `cutout_verify_20261020`, and on the NGSIM final set, e.g. `full_v20s0_{isampc,pcacc}_20261028`; `open_items.md`, "Verification sweep" and "Final NGSIM paper set and checkpoint").
They are separate ego policies, not `acc_nair_smpc` tokens; `run_all_scenarios.py` maps the prefix to the runner's `policy_type`, and the aggregator scores them like SCC runs:

### PC-ACC (`acc_pcacc`)

- **Label:** PC-ACC
- **Policy string:** `acc_pcacc`
- **Method:** Althoff et al.:
  - safe distance = ego braking along the paper's jerk profile minus the lead braking at −10.5 m/s²
  - nominal QP (2)–(6)
  - emergency jerk profile when (7) fails
  - lane membership by the paper's rule (lateral > 0.4 m and heading > 0.035 rad for 0.5 s, δ_leave 1 s)
  - cut-in recapture QP (Prop. V.1, a_cut-in −2 m/s²).
  - Measured states only.
  - Solver qpOASES.
- **Standoff and limits:**
  - **By default (maintainer's decision, 2026-10-19 evening) the standoff is the fork's d_safe = L + d0 + τ·v on the aggregator's gap**
    — in the nominal reference and constraint (3) (per step, linear in v_k),
    the safety check (7) (next-state gap below the standoff engages the emergency profile),
    the cut-in definition (Def. II.3) and the recapture terminal constraint;
    the inevitable-collision test stays physical —
    and the limits are the fork's (a ∈ [−3, 2], |j| ≤ 5).
  - `paperdsafe` restores the paper's braking-based d_safe and `amin10_amax3_jlim10` Table I's limits, so `acc_pcacc_paperdsafe_amin10_amax3_jlim10` is the faithful variant.
  - The port carries no guarantee: the fork's standoff in (7) is not a braking distance (from d0 + τv at 15–20 m/s with the fork's −3 / jerk 5, a lead braking at −4 m/s² already cannot be avoided).
- The NGSIM replay reads the PC-ACC string from `NGSIM_PCACC_POLICY` (default `acc_pcacc` until 2026-11-11, bit-identical; `acc_pcacc_amin3.5_amax2_jlim2.5_jlimup5` since 2026-11-12, §10).
- **The faithful variant on the final set** (2026-11-02, maintainer's choice for an appendix row, `replay_out/pcacc_paper_20261102/result.md`) had no contact while moving in 882 runs (the port: 2).
  - Its costs: commands down to −10 m/s² in 125 runs, one-tick j_max +4 to +7 m/s³, and δ_max +10 to +15 %p, because its d_safe is below the fork's at equal speeds and 0 at standstill.
- The clearing time t_c is not in the paper; default 4 s (`tc<x>`).
- **Fixes:** Two fixes, off by default (2026-11-11, maintainer, `replay_out/pcacc_fix_20261111/`):
  - `physcheck` puts (7) and (3) on Prop. IV.5's distance with Table I's fail-safe (a_min −10, Fig. 6 profile at |j| 10; the command may reach −10 m/s²; NGSIM replay only, the agent refuses it), keeping the nominal reference d0 + τv and the comfort limits,
  - and `reachfix` starts rule 2's one-step reach from max(a, 0) so that a braking ego does not drop its lead (cruise ↔ emergency flicker).
  - The pre-registered rule chose `acc_pcacc_physcheck_reachfix` on the 500 λ-calibration no-cut-ins (0 contacts, runs a_min ≤ −2 202 → 184)
  - on the final set (`fin_pcacc_physcheck_reachfix_sm3_vlog_20261111`, logged set speed, start margin 3 m) against the port:
    - critical / mild δ_max −1.5 / −2.6 %p,
    - critical floor breaches 21 → 16,
    - no-cut-in runs a_min ≤ −2 86 → 80,
    - emergency ticks 17 → 1.7 %,
    - but actual braking beyond −3 m/s² in 45 runs (to −10), mostly for adjacent vehicles that PC-ACC's membership counts as path members (`replay_out/pcacc_fix_20261111/result.md`).
- **Smoke run** (ego 17 / TV 15, fork standoff):
  - at entry the recapture QP is infeasible (25 m standoff vs a 10 m entry gap within t_c), so the bounded fallback brakes at −3 m/s² for about 1.7 s
  - the standoff is restored 2.75 s after t_cross, δ_max 57 % (the entry gap itself).
- **Unified limits** (2026-11-11, off by default, `utils/pcacc.py` docstring):
  - `jlimup<x>` bounds the rising jerk apart from `jlim<x>` (the QPs (6), rule 2's reach, the recovery ramp; the issued command through `PCACCController.command_jerk_up`, which goes with `command_bounds`' falling-side bound)
  - `emamin<x>` / `emjlim<x>` split the emergency limits from the comfort ones as `physcheck` does, with these numbers (each defaults to Table I's),
    also with `paperdsafe`, where Def. II.1's safe distance is then the fail-safe's (`_d_safe` on `em_cfg`),
    so `acc_pcacc_paperdsafe_amin3.5_amax2_jlim2.5_emamin8_emjlim10` is the published controller with ISO comfort limits and a −8 / 10 fail-safe
    (updated 2026-11-12: the replay's PC-ACC-as-published string is `acc_pcacc_paperdsafe_amin3.5_amax2_jlim2.5_jlimup5_emamin8_emjlim10_emics`, with `jlimup5` and `emics`; run_info.json of `fin_pcacc_paper_iso_side_20261112`, `ngsim_replay_defaults.md` 2026-11-12)
  - `emics` runs the ICS test and its full braking on the emergency limits (the port and `physcheck` keep the comfort limits there).
  - The split variants are NGSIM replay only (the agent refuses them, as `physcheck`); every existing string gives the same commands (`iso_limits_20261111/bitident_units.log`)

### ISA-MPC, cruise only (`acc_isampc`)

- **Label:** ISA-MPC (cruise only)
- **Policy string:** `acc_isampc`
- **Method:** Zhou et al.'s planner with the lateral motion frozen: per step,
  - a Gaussian mixture over the IAIMM-KF modes cut at ε = 0.8 of its peak gives an occupancy box (a mode survives iff p ≥ 0.8·p_max; kept modes are a worst case; no chance constraint, single input sequence),
  - a speed projection through their closed-loop tracker,
  - then a snap-input QP with a soft standoff (slack weight 0.055, terminal speed cost).
- **Both standoffs are the fork's d_safe = L + d0 + τ·v on Frenet centre gaps** (maintainer's decision, 2026-10-19 evening; Zhou's Th_QP = 2 s and Th_MPC = 1.5 s headways are gone):
  - the projection becomes v_ref ≤ (x_rear,k − B_k − (L + d0))/(A_k + τ),
  - the MPC row x_rear,k − s_k − (L + d0 + τ v_k) + ρ_k ≥ 0,
  - with x_rear the level-set box's rear edge without the longitudinal vehicle growth (ζ_l = 0, since L covers the vehicle; the lateral growth and the lane test are theirs).
- **Implementation:** Subclass of `ACCNairSMPCAgent`, Gurobi.
- **Deviations:**
  - N = 15 at 0.2 s (theirs 25 × 0.32 s; `--ego-horizons 40` restores 8 s),
  - the fork's limits [−3, 2] / 5 m/s³ (theirs |a| ≤ 6, no jerk bound),
  - command = the plan advanced 0.05 s along the snap chain.
- **Timing:** Solve 8 ms + IAIMM-KF 28 ms per tick (p99 48 ms of the 50 ms budget).
- **Appendix ablation `acc_isampc_ainput_rmove1000`** (2026-10-20, not the baseline):
  acceleration input and the fork's own jerk terms (r_jerk 10 on ((a_k − a_{k−1})/dt)², r_move 1000 on (a_0 − u_prev)², the fork's hard jerk limits) in place of Zhou's snap input and snap cost, all other Zhou terms kept
  — to show how much of the smoothness comes from the snap input.
  - Without `rmove<w>` the step-0 weight is the config's r_jerk/dt² = 250.
  - D-opt's QP counts these terms three times (see §3, weights), so its absolute jerk price is 3× this one's.
- **Optional road model** (2026-10-21, `1075f1c`): `IAIMMKFACCAdapter(lane_geometry=)` (a vehicle's lane index and the centres of its own and both neighbouring lanes) and `ISAMPCPlanner.plan(ego_lane_bounds=)` (the ego lane's lateral bounds for the lane test).
  - The NGSIM replay passes both because NGSIM's lane markings sit 1.2–2.4 m from the lane-keeper centres and its lanes are 3.2–3.8 m wide
  - with the uniform 12 ft lanes the IAIMM-KF kept a vehicle that had already changed into the ego lane in its old lane for up to 1.3 s, and the planner never bound on it.
  - CARLA passes neither and is bit-identical to before.
- **`qslack<w>`** (2026-10-22): the standoff slack weight w in place of Zhou's 0.055 (`ISAMPCAgent._parse_q_slack`, logged in `weights`); the NGSIM lab table uses 5, so the CARLA comparison runs `acc_isampc_qslack5` (updated 2026-11-12: since 2026-11-12 the replay's default leaves the weight unset, i.e. Zhou's 0.055 (`ISAMPC_Q_SLACK` unset, `replay/baselines.py`; `ngsim_replay_defaults.md` 2026-11-12); 5 was the setting of that lab table and of the 2026-11-11 comparisons, `design_history.md` 2026-11-11).
- **Limits** (2026-11-11):
  - its planner takes the parent's `controller_config`, so `amin<x>`, `amax<x>`, `jlim<x>` and `jlimup<x>` set them in CARLA
  - the replay builds it from the fork's config (a ∈ [−3, 2], |j| ≤ 5) unless `NGSIM_ISAMPC_POLICY` names a string with them (since the night of 2026-11-11, `iso_limits_20261111/replay_edits.patch`)
  - its default is `acc_isampc_amin3.5_amax2_jlim2.5_jlimup5` since 2026-11-12 (§10).
  - It has no emergency mode: its braking never goes beyond `a_min`
- **`slackcap`, with `l1slack<w>`** (2026-11-13, maintainer's experiment "ISA-MPC and Proposed both with the short's upper bound τv"; `765de6b`):
  - `slackcap` bounds the standoff slack by the time gap, ρ_k ≤ τ v_k, so L + d0 is hard, as the design's onerow cap.
  - `l1slack<w>` (only with `slackcap`) adds the row's exact-penalty emergency slack ξ ≥ 0 priced w·Σξ, as the design's. Without ξ the cap alone makes a close lead infeasible, and ISA falls back.
  - Off by default, and the QP is then built as before: 5 reproduction episodes were bit-identical to the v3 ISA root (`replay_out/cap_20261113/cap.log`).
  - The replay reads both from `NGSIM_ISAMPC_POLICY` (`baselines.py`, backup `replay_backup_20261113_pre_isacap/`).
  - Run on the final set v3 as `acc_isampc_amin3.5_amax2_jlim2.5_jlimup5_slackcap_l1slack1e6`, next to the design without `nocap` (`replay_out/cap_20261113/`; `main_cap.md` on the fixed 940-event main set, made by `tables_v3.py` / `main_v3.py --arm-set cap --keep-csv episodes_main_v3.csv`).
  - Result against ISA as published (critical / mild / no cut-in):
    - rear-end contacts 9 / 1 / 8 → 0 / 0 / 0;
    - δ_max −10.7 / −6.3 %p;
    - floor breaches 44 / 23 → 19 / 8;
    - fair j_max +0.60 / +0.61 / +0.55;
    - phantom_ref 0.252 → 0.626;
    - runs with a_min ≤ −2 in no cut-in 18 → 65;
    - 16 fallback runs (18 ticks; the QP fails in about 11 ms, also with 1 worker; cause not investigated).
  - Round 2 (2026-11-13 evening, maintainer), `cap.sh run2`, `main_cap2.md`:
    - The same ISA string at ξ 5,500, the design's ratio of ξ to its time-gap slack weight (1e6 / 10 × 0.055).
      - It is not the same as ξ 1e6: of 1,014 runs, 290 differ in the command by ≥ 0.1 m/s² (`cmp_isa55_vs_isa1e6.txt`).
      - On the fixed 940 set, against ξ 1e6 (critical / mild / no cut-in): fair j_max −0.13 / −0.12 / −0.11; phantom_ref 0.626 → 0.518.
      - δ_max and floor breaches are within SE; 18 fallback runs.
    - The design with the cap and the mean-trajectory entry (`occin0.1` dropped as well as `nocap`), against the cap with `occin`:
      - critical δ_max +0.7 %p (SE 0.2), floor breaches 7 → 10;
      - mild fair j_max −0.08;
      - no cut-in the same.
  - Not adopted; the maintainer's decision.
