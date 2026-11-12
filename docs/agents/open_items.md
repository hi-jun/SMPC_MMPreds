# Open items: full text

Moved verbatim from AGENTS.md on 2026-11-12 (git `3d9bcf8`); the text is unchanged. Reformatted for readability the same day: the words are unchanged and long paragraphs became bullets (checks: `docs/agents/_reformat/check_*.py`). The item "NGSIM replay defaults" is in `ngsim_replay_defaults.md`. The short list is AGENTS.md §10. (updated 2026-11-12: stale statements have since been marked in place with dated "updated", "withdrawn" and "clarified" notes; the reformat checks read this file at commit 38a1b46.)

## 10. Open items (last updated 2026-11-05)

(updated 2026-11-12: entries were added or changed after 2026-11-05, the last on 2026-11-12, e.g. live side entries in "Ghosts that enter beside the ego" and Route B.)

### Safety and recursive feasibility

(2026-10-30 to 11-02). Record: `paper/notes/safety_rf_record_20261102.md` (local; `paper/` is untracked).

- **No formal property.** The design has neither recursive feasibility nor a safety guarantee. Its safety rows are soft (ℓ1 since 10-31) and act on predicted hypotheses only.
- **Coverage fails.** The hypothesis set does not cover the real cut-in before the point of no return: 11 of the 13 floor breaches had no active TV hypothesis there.
- **Nesting fails.** The set does not stay nested between ticks: 41 of the 51 losses of the hard solution are cut-in cells whose prediction moved.
- **Implemented** (11-02, maintainer's decision): token `safelayer` (§4), PC-ACC's safety layer around the design.
  - It is Prop. IV.5 with the design as the nominal controller, on vehicles in the ego's path, with a fail-safe of −8 m/s² at jerk 20 and a lead bound of −10.5 m/s² (updated 2026-11-12: the NGSIM replay's default runs the fail-safe at jerk 10, `fsdecel8_fsjerk10`, since 2026-11-12; `tokens.md`, `fsdecel<x>`, `fsjerk<x>`).
  - On the final set (`replay_out/safelayer_20261102/result.md`) it acted on 2 warm-up ticks in 889 runs and left every paper-table column unchanged.
  - No stress test with injected hard lead braking (maintainer).
  - The paper's NGSIM tables run with it (maintainer, 2026-11-02 evening; `replay_out/safelayer_20261102/tables_paper_safelayer.md`).
- **Implemented** (11-03, maintainer's request): token `safebranch` (§4), the layer's fail-safe as a branch of the scenario tree, so that the MPC plans the contingency itself and the layer only executes it.
  - On the final set it changes behaviour only where the layer would have acted (one warm-up run), and otherwise through its 1e-3 cost at the third decimal (`replay_out/safebranch_20261103/result.md`).
  - Adopted as is for the paper's NGSIM tables (maintainer, 2026-11-03) (updated 2026-11-12: dropped from the NGSIM design and the replay's default tail on the night of 2026-11-11, maintainer; `design_history.md` 2026-11-11).
- **Implemented** (11-04, maintainer's request, final-check items 7 and 12): the fail-safe's exit is the MPC's recovery ramp at the fail-safe's jerk (§3, 0419637), and token `entrant<b>` (§4) covers the layer's close entrants under a braking bound (−2 with `entrant2`).
  - On the paper root the layer never acts and close entrants occur in 3 runs (20 ticks, 1 scored run with 4 scored ticks), so the final set shows the token's cost only on those runs: none (every paper-table cell unchanged, the 3 entries all avoidable at entry, no layer action; `replay_out/entrant_20261104/result.md`).
  - The remaining uncovered cases are entries that are unavoidable under a_c at the entry tick (0 of 3 on the final set) and vehicles not yet in the path;
    - a pre-entry (lateral reachability) rule was not implemented: the 11-02 estimate found a 2 m/s² lateral bound binding on 1.4–2.0 % of ticks and acting at the point of no return in 0 of 13 floor-breach runs, and the earliest-entry prototype cut breaches 13 → 7 at phantom_ep +0.32 m/s² and v_avg −0.75 m/s.
  - Adopted for the paper's NGSIM tables (maintainer, 2026-11-04; the replay's default tail ends `_entrant2`; tables `replay_out/entrant_20261104/tables_paper_entrant2.md`) (updated 2026-11-12: the default tail has grown since and ends `_fsjerk10` since 2026-11-12, with `entrant2` still in it; `ngsim_replay_defaults.md`);
  - the name "cut-in" in the layer's code and logs (`cutin_vids`, `safety_n_cutin`) kept for the logs' compatibility, documented as close entrants.
- **Implemented** (11-04 night, maintainer's request, final-check item 8): token `passmiqp<m>` (§4), the pass / yield choice inside the optimization.
  - Result on the final set:
    - critical δ_max 9.1 → 8.1 %, floor breaches 9 → 7, mild δ_max 5.9 → 5.2, floor breaches 3 → 1, no-cut-in phantom_ep 0.188 → 0.179 (SE 0.008), runs a_min ≤ −2 11 → 12, contacts 0, j_max −0.09 / −0.08;
    - but all four removed breaches are runs where the passed TV merged beside the ego and the replay's ghost rule hides it (item 10): re-scored with side entries live once clear, critical 11.4 → 10.8 % and 14 → 13 breaches, mild 7.8 → 7.4 % and 8 → 7, no cut-in contacts 1 → 2;
      - (clarified 2026-11-12: "contacts 0" above is the replay's own scoring; "no cut-in contacts 1 → 2" is the open-loop recount, in which the ego never reacted to the re-scored vehicles: the paper root has 1 such contact and the `passmiqp` root 2, the added one ds4_v3159; `replay_out/passmiqp_20261104/result.md`)
      - (updated 2026-11-12: since 2026-11-12 the replay keeps lane-change side entries live by default, `NGSIM_SIDE_ENTRY=live`; the `passmiqp` and `passnocap` roots ran under the ghost rule and have not been re-run)
    - one no-cut-in near miss (ds4_v3015, 0.07 m) where the cap made the ego slower;
    - MIQP ticks 3.0 % of ticks, TV candidates 731 pass / 10 yield ticks;
    - warm MIQP ticks 136 ms against 60 ms (load-inflated), the first-use builds 1.8 s;
    - 17 time-limit fallback ticks in 10 runs, all on plain-QP ticks under the added load.
  - Decision pending (adopt for the paper's NGSIM tables, keep as an option, or drop);
  - the margin m (0 in the run) and the reference memory are the open design choices.
- **Implemented** (11-04, maintainer's request): token `passnocap` (§4), the pass / yield MIQP without the acceleration cap.
  - Result: the no-cut-in phantom braking falls by 27 % (0.188 → 0.138), but
    - the passes are bought with acceleration (median +0.26 m/s² above a_0 on 60 % of pass ticks),
    - more TVs merge beside the ego (hidden by the ghost rule; updated 2026-11-12: such side entries are live by default since 2026-11-12, `NGSIM_SIDE_ENTRY=live`, and this root has not been re-run), and
    - where the TV comes in front anyway the ego arrives faster: two entries at 0.24 / 0.26 m caught by `entrant2`'s check (no contact), three new mild floor breaches, the un-ghosted recount worse in the mild group (8 → 10).
  - Decision pending (keep the cap, drop it, or a middle setting not run: a cap above a_0 by a fixed amount, or m > 0 without the cap).
- **Discussed** (11-04 afternoon, maintainer; no new run; record `replay_out/passmiqp_20261104/discussion_20261104.md`): the cut-in probability does not enter the pass / yield decision.
  - The yield's main cost is the braking the hard floor forces on the shared pre-entry steps (weight 1); the P-weighted terms (price, σ term, the cut-in branch's post-entry tracking) sit on the yield side only, so a higher P only pushes toward passing.
  - One-tick re-solves on the test scene changed the decision in 0 of 42 scenes (P_A 0.1 → 0.95);
  - on the `passnocap` logs the yield share at equal entry gap does not move with p.
  - The capped MIQP is its own pre-filter on 731 of 741 TV candidate ticks.
  - The wrong passes: in 22 of 25 TV side-entry runs it kept passing with p ≥ 0.9 and the TV latched, but by then the TV was alongside (comfort yield possible in 2 of 22, estimate);
    - the first pass came 2.3–2.7 s earlier at p 0.22–0.5, before any lateral motion.
  - Agreed: σ belongs in the pass condition.
  - Pending: the MIQP or the pre-solve rule (c) (exclude if the TV's rear is behind the ego's front at entry under the current-state rollout, entrypass with `pass_cap_displacement` in place of the previous plan), and how p and σ enter the pass condition.
- **Open:** the final-check findings of 2026-11-03 (`replay_out/lkrsigma_20261103/result.md`) (updated 2026-11-12: items 4 and 5 were closed later in this bullet, H adopted on the night of 2026-11-03; only the "Still open" list at its end stays open):
  - the ego-lane lead's lane-keeping trajectory leaves the lane at some step on 1.42 % of scored ticks while the lead keeps its lane on 65 % of them (item 4), and
  - `trajcutin`'s away-direction cut-in modes lead in a kept scenario on 0.18 % of ticks with probabilities up to 1.0 (item 5).
  - **Cause of both** (2026-11-03 night, `replay_out/lkrsigma_20261103/result.md`, section "Items 4 and 5: cause"): the replay tests lane membership with two definitions, the logged Lane_ID at step 0 and the midpoint between binned lane centres at steps 1-15 (`ngsim_replay.py` `lane_membership_fn`);
    - a vehicle in the strip between them is a lead at step 0 but absent at the first predicted steps (item 4, mostly a TV that has just entered), or adjacent at step 0 but in the lane at the first steps, so `trajcutin` turns its lane-keeping and away-direction trajectories into cut-in modes (item 5).
  - **Fix run on the final set** (2026-11-03 night, maintainer's request; `replay_out/lanefix_20261103/result.md`): `NGSIM_LANE_TEST=markings` (§10) + token `lkhold` (§4) removed the cases (away-direction modes leading 152 → 14 scored ticks; lead's lane-keeping cells off while it keeps its lane 775 → 46) but moved the loop:
    - no-cut-in phantom_ep 0.152 → 0.196 (+0.044, SE 0.015), runs a_min ≤ −2 7 → 11,
    - critical δ_avg −0.19 %p (SE 0.07), floor breaches critical 10 → 9 and mild 3 → 6,
    - 180 cut-in episodes moving δ_max by > 1 %p, and
    - one −8 m/s² fail-safe (8 scored ticks, ds6_v2080) against a vehicle whose body overlapped the ego lane, which the away-direction mode used to keep the MPC away from (item-5 vehicles sit 0.02 / 0.07 / 0.30 m beyond the marking, p10 / p50 / p90).
  - Cause: the marking line moves every cut-in hypothesis's entry step.
  - Not adopted (decision pending); a narrower fix (midpoint band kept, only ego-lane vehicles anchored, `lkhold`, item 5's modes documented) is not run.
  - CommonRoad's US-101 lanelet boundaries (a hand-made map) lie −0.01 m (median, p5-p95 −0.20 to +0.24) from the marking map and −0.33 m from the midpoint (10,762 points).
  - **The body test** (`NGSIM_LANE_TEST=body` + `lkhold`, `replay_out/lanefix_20261103/result_body.md`):
    - critical δ_max 11.7 → 3.3 % and floor breaches 10 → 2, mild 7.3 → 1.9 % and 3 → 1,
    - but no-cut-in phantom_ep 0.152 → 0.509, runs a_min ≤ −2 7 → 56, j_max +0.8 in every group, v_avg −0.66 m/s, and one contact (ds4_v2935: a non-TV vehicle merging beside the ego, excluded by `entrypass0`);
    - every adjacent vehicle whose body touches the ego lane becomes a hypothesis entering at step 1.
  - **The ego-path test** (`NGSIM_LANE_TEST=path` + `lkhold`, `replay_out/lanefix_20261103/result_path.md`):
    - critical δ_max 11.7 → 9.1 % (−2.5 %p, SE 0.5), δ_avg −0.27 (0.07), T_rec −0.10 s, floor breaches 10 → 9;
    - mild δ_max 7.3 → 5.9 %, floor breaches 3 → 3;
    - no-cut-in phantom_ep 0.152 → 0.188 (+0.036, SE 0.011), runs a_min ≤ −2 7 → 11, j_max +0.10 to +0.16;
    - no contact, no safety-layer tick, one time-limit fallback tick under parallel load;
    - the lead's lane-keeping cells off while it stays 775 → 48 ticks, away-direction modes kept for vehicles overlapping the ego (150 ticks).
  - **Adopted (maintainer, 2026-11-03 night): H is the design** (§1), so items 4 and 5 are closed.
  - **Still open:**
    - R1 (a hard first-step stopping row) in addition,
    - `entrypass9` (updated 2026-11-12: moot for the NGSIM design, which dropped entrypass on the night of 2026-11-11; still a pending choice for the CARLA string, AGENTS.md §1),
    - `termcost`,
    - `floorrisk0.3` (passes the tuning rule with a 4 % phantom gain) (updated 2026-11-12: moot; `floorrisk<ε>` with `floorhyp` entered the NGSIM design on 2026-11-10, at ε 0.1 since NL that evening; `design_history.md` 2026-11-10),
    - the speed weight q_v (tuning 2026-11-02, `replay_out/qvtune_20261102/result.md`: no q_v in 2–16 passed; critical δ_max +2.9 to +9.3 %p, phantom_ep −0.007 to −0.061; q_v × k is the same problem as w / k and r_jerk / k),
    - the entry-drift fixes,
    - the PC-ACC appendix presentation, and
    - whether the paper table's t_comp column takes the single-process timing below (maintainer's decision; that timing predates `safebranch`, which added 3–5 ms of t_comp under parallel load and 25 % QP time in the smoke runs).

### Entry timing of cut-in hypotheses

(measured 2026-11-04; the maintainer: a real problem, kept as its own item).

- Open-loop NGSIM replay (the paper design at λ 0, final set 889 + tuning set 594, `replay_out/entry_coverage_20261104/result.md`; truth = the replay's own `NGSIM_LANE_TEST=path` rule on the logged future).
- When the TV is actually in the ego's path at a predicted step, the controller has a cell led by it at that step on 59 / 36 / 32 / 31 % of pre-entry ticks (entry 0.2–0.6 / 0.8–1.4 / 1.6–2.2 / 2.4–3.0 s ahead); a nearer lead binds on another 20 %, so the gap is 20 / 43 / 47 / 49 %.
- Near the entry the gap is a late predicted entry (the hypothesis exists; lateness p80 0.2–0.6 s, p95 0.6–1.2 s); from about 1.6 s the hypothesis is absent (gated, p < 0.1: 18–31 %; no entry within 3 s: 7–11 %).
- In no-cut-ins a TV cell is active while the TV is not in its path on 3.1 % (final) / 7.3 % (tuning) of ticks.
- The same run's longitudinal residuals (`replay_out/sigma_coverage_20261104/result.md`) show the σ term near its nominal coverage (more than nominal up to 1.5 s, 0.74 against 0.77 at 2.4–3.0 s, short at speeds below 5 m/s): the position given the mode is predicted well, the entry is not.
- Not fixed; no closed-loop measurement.

### Calibrated lane-change probabilities

(2026-11-05, maintainer's request; `replay_out/pcal_model_20261105/result.md`).

- Token `pcal` (§4) and the replay's `ISAMPC_PCAL=1` calibrate STDAN's and the IAIMM-KF's lane-change probabilities to P(lane change within 3 s) on natural NGSIM traffic, conditioned on the lateral state and cross-fitted by recording.
- Decided (maintainer, 2026-11-05/06):
  - ε candidates {0.02, 0.05, 0.10};
  - the σ term and `trajcutin`'s rule kept as coded;
  - ISA-MPC as published in the main table and calibrated (`ISAMPC_PCAL=1`, ε 0.8) in an appendix row;
  - after the tuning run (`replay_out/pcal_tune_20261105/result.md`) **`pcal` + `occin0.1` adopted** (§1).
- The list of open decisions as it stood before:
  - (1) the probability thresholds on the calibrated scale (the 0.1 gate, β_ref 0.6, `occin`'s ε, `cvar`'s ε and c, `floorrisk`'s ε);
  - (2) which designs to tune with `pcal` on the tuning set and the pre-registered rule;
  - (3) ISA-MPC: as published (uncalibrated), calibrated with Zhou's ε 0.8, or calibrated with ε re-tuned.
    - With the lateral-state map ε 0.8 does not switch anticipation off (open loop, vehicles blocked by a slower lead, final set excluded, `replay_out/pcal_model_20261105/result.md`, section "Blocked vehicles"): lane changes detected 74 → 75 %, first alarm 1.3 → 1.2 s before the change, false-alarm spells 32 → 1.5 %.
    - The "almost off" of `replay_out/iaimm_reliability_20261105/result.md` holds for its p-only isotonic map only;
  - (4) `trajcutin`'s lane-keeping → cut-in rule, whose modes now carry the large calibrated lane-keeping probability;
  - (5) whether `pcal` enters the paper's design.
- The `occin` / `cvar` tuning run of 2026-11-05 (`replay_out/occin_cvar_20261105/`, stopped by the maintainer) used the uncalibrated probabilities.
- Literature precedents for the calibration: `paper/notes/calibration_precedents_20261105.md` (local).
- Final set of `pcal` + `occin0.1` (2026-11-06): §1 and `replay_out/pcal_cvar_20261106/result.md`;
  - ISA-MPC calibrated (appendix row, `fin_isampc_pcal_20261106`):
    - critical / mild δ_max 24.1 / 15.1 → 23.5 / 15.8 %,
    - critical Δt_ant 0.45 s (23 runs) → 0.28 s (62 runs),
    - floor breaches critical / mild 14 / 17 → 16 / 19 (recounted 22 / 29 → 26 / 30),
    - no-cut-in runs a_min ≤ −2 73 → 52,
    - contacts 3 → 2.
- Open: the maintainer's review of the new contact (ds3_v1780) before the paper's tables switch.
- **Superseded on the evening of 2026-11-06 by `bcts`** (§4; `final_paper/appendix/calibration.md`; final-set runs `replay_out/bcts_20261106/`).

### Ghosts that enter beside the ego

(final-check item 10, counted 2026-11-04 from the final-set tick logs; one-off script, re-plays Rule A on each root's own ego trajectory and reproduces its `n_ghosts`, `ghost_ahead` and `ghost_overlap` exactly).

- A vehicle that changes into the ego lane overlapping the ego (rear behind its front, front ahead of its rear) becomes a ghost for good (Route B below).
- Proposed (paper root `..._lkrsigma_path_20261103`, 882 valid runs): 40 runs (critical 14 / mild 18 / no cut-in 8), the TV in 20 (4 / 16);
  - in 18 of those 20 `entrypass0` had excluded the TV's hypothesis (p_cutin up to 1.0) in the second before the entry, by the same test (rear behind the planned ego front), so the replay cannot score a wrong pass.
- The vehicle later got clear ahead in the ego lane, nearer than the live lead, in 11 runs (4 / 5 / 2), the TV in 5 (mild).
- TV side entries (critical / mild) of the baselines, which anticipate less: ISA-MPC 11 / 22 (clear later 8 / 13), PC-ACC 11 / 25 (4 / 10), PC-ACC as published 15 / 26 (8 / 8).
- Recounted with side entries live from their first clear tick (same trajectories; the run is identical up to that tick, whose gap is 0-2 m, so delta_max and floor breaches are what a re-run would give; contacts would not be, since the ego never reacted), delta_max % and floor breaches, critical / mild:

  | Controller | delta_max % critical | delta_max % mild | floor breaches critical | floor breaches mild |
  |---|---|---|---|---|
  | Proposed | 9.2 -> 10.9 | 5.9 -> 7.4 | 9 -> 13 | 3 -> 7 |
  | ISA-MPC | 24.1 -> 27.9 | 15.1 -> 19.2 | 14 -> 22 | 17 -> 29 |
  | PC-ACC | 28.3 -> 30.3 | 26.9 -> 30.4 | 24 -> 27 | 36 -> 46 |
  | PC-ACC as published | 39.7 -> 43.0 | 41.3 -> 43.9 | 29 -> 36 | 58 -> 66 |

- Ghosts that entered fully behind and later passed through the ego (non-reactive followers) are a different case and were not recounted.
- The rule is kept (maintainer, 2026-11-04); the paper describes it as coded.
- Changed on 2026-11-12 (maintainer): side entries are live by default (`NGSIM_SIDE_ENTRY`, Route B; `replay_out/side_live_20261112/`).

### Floor breach on NGSIM

(final-check item 13, maintainer 2026-11-04).

- The replay's `floor_breach` (`ngsim_replay.summarize`) is a run in which, at a scored tick at or after t_cross, the bumper gap to that tick's lead, whichever vehicle it is, falls below d0 = 3 m; a TV that became a ghost (above) is never that lead.
- The paper calls it a floor breach, defined that way, not an intrusion by the cut-in vehicle, so no TV-only column is needed.

### Computation time without parallel load

(2026-11-03, maintainer's request).

- `replay_out/timing_noload_20261103/result.md`: one replay process at a time, 1 thread, i9-10980XE, the same 60 final-set episodes (20 per group) for every controller.
- The table's t_comp (per run the median of prediction + controller time, then the group mean), critical / mild / no cut-in:
  - the design with `safelayer` + `leadmerge` 50 / 51 / 50 ms (1.8 % of ticks above the 100 ms control tick),
  - without `leadmerge` 94 / 100 / 100 ms (70 %),
  - ISA-MPC 77 / 85 / 85 ms (29 %),
  - PC-ACC 3 ms in both settings.
- STDAN takes 39 ms per tick, IAIMM-KF 73 ms.
- The design's p99 (197 ms) is the first-use QP builds, one per scenario count per episode; pre-building them (bit-identical) is not implemented.
- The final-set roots' t_comp was measured with 36 replay processes in parallel (the design 171–188 ms, ISA-MPC 97–105 ms) and is load-inflated; under that load the order of the design and ISA-MPC flips.
- Quote computation time only from single-process runs.

### NGSIM replay plant lag

(2026-10-29).

- `NGSIM_EGO_LAG_TAU=<τ>` gives the replay's simulated ego a first-order lag from the command to its acceleration for every controller (exact integration per 0.1 s tick; at the speed bound the speed is held and the acceleration set to 0; the ego starts settled on its start command);
  - D-opt receives the actual acceleration (`solve(ego_accel=)`) and models the lag only with `alag<τ>`;
  - ISA-MPC and PC-ACC keep their own models.
- Tick logs carry `a_act` (the actual acceleration at the tick start).
- Unset = the ideal double integrator, bit-identical to before.
- First use: `replay_out/alag_eval_20261029/` (prereg there; the maintainer asked for τ from Yoon et al. 2021, 1 s, and r_a 0).
- **Default 0 again since 2026-10-29 evening** (maintainer; 0.5 that afternoon): the paper's NGSIM table uses the ideal ego for every controller (§1). `NGSIM_EGO_LAG_TAU=0.5` gives the lagged ego of `replay_out/alag_eval_20261029/`.
- The paper's table scores a_min, j_avg, j_max, runs a_min ≤ −2 and phantom_ep on the actual acceleration: Δv/Δt per tick on the ideal plant, `a_act` on the lagged one. a_avg is Δv-based and was already actual.
- Jerk is the one-tick difference |a_{k+1} − a_k| / 0.1 s of that acceleration, without the command metric's 0.2 s band (maintainer, 2026-10-29). On the ideal plant the actual acceleration is Δv/Δt per tick.
- On the ideal plant the replay reports `a_act` = the command even while the speed is clipped at 0, so `alag` runs there fall back at standstill (26 of 889 at τ 0.5). `jinput` clamps a₀ at v_min (§4) and has no standstill fallback there.
- Not fixed, by the maintainer's decision (2026-10-29 evening): the fix would change STDAN's ego-acceleration input against the earlier ideal-plant roots, and `jinput`'s v_min clamp already starts the plan from a₀ ≥ 0 at a stop.

### Final NGSIM paper set and checkpoint

(2026-10-28, maintainer).

- **Episodes.** The paper's NGSIM table uses 889 episodes (`replay_out/final_paper_20261028/episodes_all.csv`).
  - They are the 800 of `full_20261020` (400 tv_main cut-ins + 400 random no-cut-ins) plus 89 other lane changes of the same TVs whose replay windows do not overlap.
  - That makes 489 cut-ins (209 critical, 280 mild) and 400 no-cut-ins.
- **λ calibration set.** The S2b set of 500 was made disjoint from the 889 by replacing 168 episodes (`lambda_cal_v2_nocut.csv`). Two episodes overlap when they share a ds, their windows [fw, f1] overlap and they share a vehicle in either role.
- **λ̂ = 0 for both designs.** The risk is 0.342 before the adoption and 0.355 after it, against the budget 0.5274.
- **λ calibration population (maintainer, 2026-10-29 evening).** Only the S2b runs in which an active cut-in hypothesis's predicted trajectory entered the ego lane inside d_safe (rearmost entry bumper gap to the planned ego front < d_safe, `entry_gap_min`, on ≥ 1 scored tick), fixed by the calibrated design's own λ = 0 run:
  - λ decides how much the predictor is used, and that only matters where a hypothesis can constrain the ego.
  - The final test set stays unrestricted; it shows Proposed's performance in NGSIM.
  - On the 10-28 design this keeps 397 of the 500 cells (runs without such an entry have phantom_ep ≈ 0) and raises the λ = 0 risk from 0.355 to 0.445, still under 0.5274 (λ̂ 0).
  - Open: the budget 0.5274 was set on all S2b cells, and the 10-22 root it comes from has no entry arrays, so it cannot be re-derived on the same subset.
- **Disclosure.** full's calibration half stays in the final set; the paper discloses its use in the 10-20/21/22 calibrations.
- **Checkpoint.** The paper's checkpoint, `checkpoints_out30_no_tcross` (2:1:1 balanced + mirrored; the paper describes it as "LK : LC = 1:1, LLC = RLC"). The unmirrored t_v_t checkpoint was not adopted.
- **Roots.**
  - `final_v20s0_ep0_ncls_paper_20261028`: the adopted design, λ 0.
  - `full_v20s0_ep0_paper_20261028` + `add89_v20s0_ep0_paper_20261028`: the design before the adoption.
  - `full_v20s0_{isampc,pcacc}_20261028` + `add89_v20s0_{isampc,pcacc}_20261028`: the baselines.

### Context trust λ: calibration on control outcomes

(first results 2026-10-20). `ctx<λ>` exists (§4); both calibrations below have run, no λ is adopted yet. (clarified 2026-11-12: the paper's NGSIM runs and the replay's default use λ 0, `ctx0`, the plain policy, which is the CRC λ̂ of the final paper set, "Final NGSIM paper set and checkpoint" above.)

- **History.** 2026-10-19/20, a predictor-level calibration on NGSIM Val+Test (`results/ngsim_lambda_cal_20261019/`, s0–s7, `out/SUMMARY.md`; 1,233 cut-in / 67,078 no-cut-in episodes) gave λ̂(ε 0.2) = 0.6 under the loss "share of the 2.8 s label window with p_λ ≥ 0.1", and no λ̂ under the magnitude-aware loss "label-window mean of min(p_λ/0.6, 1)" (STDAN alone already loses 0.218).
  - In CARLA (`cutin_verify_20261020`) `ctx0.6` weakened D-opt:
    - prediction lead 2.83 → 1.30 s (normal) / 2.04 → 0.82 s (aggressive),
    - a_min_cmd 0.78 / 0.92 m/s² harder (3.3 / 5.4 SE),
    - δ_max +7 / +15 %p,
    - aggressive `ctx_miss` 6 → 17 of 18;
    - unnecessary decelerations only 12 → 8 of 14.
  - The proxy losses never saw what the controller does with P: before lateral motion λ = 0.6 cuts P to 0.4·p_S, which still clears the 0.1 gate but scales α and the `dopt` price.
  - The maintainer therefore decided (2026-10-20) to calibrate on the controller's realised losses, as Conformal Decision Theory does.
  - The CommonRoad scenarios in `/home/core-dev/HJ/ACC/safe-acc` cannot serve as data: 25 of 26 US-101 snippets match NGSIM US-101 07:50 (first 3.2 min), every vehicle in them is a STDAN Train vehicle, and there is no I-80 scenario.
- **Loss design (maintainer's decisions, 2026-10-20 afternoon).**
  - Conformal risk control (CRC) picks the largest (or smallest) λ that keeps a risk under its budget, so it spends the whole budget. Whatever is not in the loss gets sacrificed.
  - A floor-only loss (bumper gap < d0, i.e. centre gap < L + d0) cannot see what λ costs.
    - On the verification grid (`results/acc_scenario_sweep/cutin_verify_20261020/table_cutin_runs.csv`, D-opt groups 01 + 02, 36 runs per λ) both λ = 0 and λ = 0.6 had 0/36 floor breaches (`min_bumper_gap` ≥ 8.2 and ≥ 6.4 m), while the median peak braking command `a_min_cmd` went from −1.95 to −3.00 m/s² and the runs at ≤ −2.5 m/s² from 12/36 to 28/36.
  - A single merged loss was rejected. It needs an exchange rate between breaches and braking: dividing braking by 3 makes a 10 % breach rate equal to 0.3 m/s² of mean extra braking, and a larger divisor D inflates CRC's finite-sample term D/(n + 1).
  - **Decision: compare two selection rules on the same runs** (`ctxcal_carla_20261020/crc_select.py`; the losses are the aggregator's `a_min_ev`, `floor_breach` and `phantom_brake`, §4):

    | Rule | Loss per episode (bound B) | Budget | Monotone envelope | λ̂ |
    |---|---|---|---|---|
    | Previous design (bound the cut-in side) | cut-in cells: extra peak braking against the same cell at λ = 0, max(0, a_min_ev(0) − a_min_ev(λ)) [m/s², B = 3]; floor breach [0/1, B = 1] | mean extra braking ≤ X; breach probability ≤ ε | max over λ′ ≤ λ | the largest λ meeting both |
    | Flipped design (bound the no-cut-in side) | no-cut-in cells: `phantom_brake` [m/s², B = 3] | mean ≤ X_u | max over λ′ ≥ λ | the smallest λ meeting it, i.e. the most anticipation: trust the context as much as possible and discount it only as much as needed to keep phantom braking within X_u |

  - **Budgets**, fixed before any result was read (`results/acc_scenario_sweep/ctxcal_carla_20261020/prereg.json`, 2026-10-20 16:30): X = X_u = 0.15 m/s², ε = 0.05.
    - CRC's rule is (n·R̂(λ) + B)/(n + 1) ≤ budget, so with 36 cells the envelope mean must be ≤ 0.071 m/s² and there must be 0 breaches.
  - **Anchors measured on the verification grid** (same CSV, D-opt, λ = 0 → 0.6; that CSV predates the two new columns, so they were recomputed from `a_min_ev`, `a_min_cmd`, `min_bumper_gap` and `ctx_unnec`):
    - cut-in extra peak braking 0 → 0.85 m/s² (mean over the 36 cells, paired by cell);
    - no-cut-in phantom braking 0.96 → 0.53 m/s² (mean over the 14 runs; no in-lane lead was ever within d_safe there, so it equals −a_min_cmd);
    - unnecessary-deceleration runs 12/14 → 8/14.
- **Code.** `sweep_acc_scenarios.py --cells-json` (bbcdcd9, §2) and the aggregator columns `floor_breach` / `phantom_brake` (13ef7c5, §4).
- **CARLA batch calibration** `results/acc_scenario_sweep/ctxcal_carla_20261020/` (launcher `results/acc_scenario_sweep/launch_ctxcal_20261020.sh`, code 13ef7c5, 15:59–18:29, four lines in parallel).
  - D-opt + `ctx<λ>` for λ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0} on 72 cells drawn uniformly from the verification grid's ranges (`make_cells.py`, seed 20261020):
    - 18 normal cut-in (`c1`, trigger 17 m / ego gap 22 m; ego 14–17, TV 11–15 m/s, lane-change time 3.0–3.67 s),
    - 18 aggressive (`c2`, trigger 13 / ego gap 14; lane-change time 1.67–2.33 s),
    - 36 no-cut-in (`c3`, ego 14–17, TV 13–15 m/s, lead Δv −8…−4 m/s, trigger 22–34 m);
  - a fixed grid is not a random sample, so CRC's exchangeability would not hold on it;
  - every λ runs the same cells (paired);
  - the λ = 0 cut-in cells run twice (`*_rep`) for the noise floor.
  - 612 runs, all valid (0 fallback, 0 predictor errors, 0 collisions).
  - Result (`crc_select_prereg.txt`, means over 36 cells):
    - noise floor 0.009 m/s²;
    - cut-in extra braking 0 / 0.108 / 0.270 / 0.504 / 0.708 / 0.893 / 1.016 / 1.041 m/s² at λ 0 … 1 (floor breaches 0 up to λ 0.6, 1 at 0.8 and 1.0);
    - no-cut-in phantom braking 1.106 / 1.048 / 0.951 / 0.847 / 0.742 / 0.546 / 0.179 / 0.003 m/s² (unnecessary-deceleration runs 33 → 0 of 36).
  - **Previous design λ̂ = 0, flipped design λ̂ = 1; no λ meets both budgets.**
  - The exchange is unfavourable at low λ (0 → 0.4: extra +0.71, phantom −0.36) and favourable at high λ (0.6 → 1: phantom −0.54, extra +0.15).
- **Next.**
  - (a) Verification grid 01–04 at λ = 1 (λ = 0 is D-opt, already in `cutin_verify_20261020`).
  - (b) A phantom-braking definition that counts only braking caused by cut-in hypotheses, pre-registered and run on fresh NGSIM no-cut-in TVs (see route B) (updated 2026-11-12: done on 2026-10-21 in Route B below, first as `phantom_cf`, withdrawn the same day, then as `phantom_ep`).
  - Changing a budget or a loss after these results requires fresh calibration data.

#### Route B

**Route B (NGSIM closed-loop replay), built and run 2026-10-20.**

- Code `results/ngsim_lambda_cal_20261019/replay/` (gitignored; `ngsim_replay.py`, `run_replay.py`, `select_episodes.py`, `make_full_lists.py`, `crc_select_replay.py`, `causal_geometry.py`); spec `ROUTE_B_PLAN.md`.
- Simulated ego replaces the target-lane follower (written into its rows, STDAN inputs bit-exact: max |Δp| 1.85e-6 against `out/p_S.npz`);
  - other vehicles replayed;
  - STDAN-3int and D-opt + `ctx<λ>` with the CARLA code;
  - 10 Hz control;
  - set speed = the logged follower's maximum;
  - causal SG(11, 2) lateral offset with y₀ = 0.760 m (its lane-keeping p95; CARLA keeps 0.10 m).
- **Replay rules** (maintainer, 2026-10-20):
  - 3 s warm-up before the scored window;
  - equilibrium start (v = the lead's speed, gap ≥ d_safe);
  - a vehicle whose rear bumper is behind the simulated ego's front bumper at any moment it (re)appears in the ego's lane (the warm-up start, a lane change into it, its track start, a re-entry) is a ghost for the rest of the episode (never lead, target, safety-layer member or contact; still in STDAN's grid),
    - which includes a vehicle that changes in beside the ego with its nose ahead and a ghost that later pulls ahead of the ego (final-check item 10, §10);
  - since 2026-11-12 (`NGSIM_SIDE_ENTRY=live`, the default) a vehicle that changes lanes into the ego lane after the warm-up start beside the ego (a row at the previous frame in another lane; rear behind the ego's front, front ahead of its rear) is live instead, and becomes a ghost only at its first tick in the ego lane with its front at or behind the ego's rear;
  - a track start or re-appearance beside the ego stays a ghost (`replay_out/side_live_20261112/`);
  - episodes with another vehicle entering ≤ 30 m ahead are excluded from the logged geometry before simulation;
  - `phantom_brake` skips ticks followed within 3 s by a vehicle entering ≤ 30 m ahead or overlapping (no effect in CARLA, §4 docstring).
- **Full run** `replay_out/full_20261020/` (prereg 19:47, 19:48–21:49): 400 `tv_main` cut-in TVs + 400 no-cut-in TVs, one episode each, × 8 λ = 6,400 runs;
  - calibration/report 200/200 by TV;
  - 0 predictor or solver exceptions;
  - 7 episodes with fallback at every λ (dropped).
- **Calibration half** (199 cut-in / 197 no-cut-in cells):
  - extra braking 0 / 0.039 / 0.076 / 0.134 / 0.195 / 0.330 / 0.454 / 0.484 m/s²;
  - floor breaches 5–10 of 199 (5 already at λ 0);
  - phantom 0.78 → 0.56 m/s².
- **Previous design λ̂ = 0.3; on the report half at 0.3 the extra-braking risk is 0.137 ≤ 0.15 and the breach risk 0.025 ≤ 0.05.**
- Flipped design: no λ (phantom risk ≥ 0.58).
- Cause, measured at λ = 1: in 172 of the 200 runs with phantom ≥ 0.3 the ego's own lead was just outside d_safe (median 1.01×), 143 of them closing (median 1.2 m/s), and only 3 had a cut-in probability ≥ 0.1 — the phantom definition counts normal lead following, which never occurred in CARLA's lead-free no-cut-in runs.
- **Re-run with a corrected definition** (`replay_out/nocut_cf_20261021/`, prereg 22:08, 22:09–23:21):
  - `phantom_cf` = max over the scored ticks (rule-B ticks excluded) of max(0, min(−a_cmd, u_cf − a_cmd)), u_cf being the command of an identically configured second controller at the same state and previous command with every cut-in mode removed (main loop verified identical, max |Δa_cmd| = 0);
  - 400 fresh no-cut-in TVs (the 739 TVs of the first run excluded, seed 20261021, 200/200), X_u = 0.15 unchanged.
- **Calibration half**: `phantom_cf` 0.106 → 0.011 m/s² from λ = 0 to 1, CRC risk 0.142 already at λ = 0, so **flipped design λ̂ = 0** (withdrawn 2026-10-21, marked here 2026-11-12: it used the flawed `phantom_cf`; the flipped λ̂ with `phantom_ep` is in "Correction (2026-10-21)" below);
  - report half at λ = 0: mean 0.124 m/s² (envelope 0.140).
- On NGSIM, λ ∈ [0, 0.3] therefore meets both budgets (the flipped rule picks 0, the previous rule 0.3), whereas in CARLA no λ did (withdrawn 2026-10-21, marked here 2026-11-12: it rests on `phantom_cf`, see "Correction (2026-10-21)" below; with `phantom_ep` no λ meets both budgets on any data set).
- CARLA's no-cut-in cells are built to look like cut-ins (phantom 1.11 m/s² at λ = 0); in the NGSIM no-cut-in episodes STDAN gave v a cut-in probability ≥ 0.5 in 20 % of episodes and ≥ 0.1 in 14 % of ticks (36 % of ticks on the CARLA dump grid).
- The calibrated λ depends on how often cut-in-looking non-cut-ins occur.
- **Subset calibrations** (2026-10-21, `replay_out/subset_cal_20261021/`; calibration may reuse already-seen runs, only the reporting data must be unseen — maintainer):
  - S1 "v blocked by a slower lead" (v faster than its own lead by ≥ 2 m/s within 30 m, cut-in window t_cross −5…−0.5 s) gives previous λ̂ 0.2 (179 cells) and flipped λ̂ 0.4 (158), no common λ (withdrawn 2026-10-21, marked here 2026-11-12: the flipped λ̂ 0.4 used `phantom_cf`, see "Correction (2026-10-21)" below; with `phantom_ep` it is 1.0);
  - S2b "no-cut-in in which v's open-loop STDAN cut-in intention was the largest mode" calibrated on 2,166 cells (`replay_out/cal_S2b_large_20261021/`, prereg 02:02) gives **flipped λ̂ = 0.8** (phantom_cf 0.204 → 0.105 m/s² from λ 0 to 0.8) (withdrawn 2026-10-21, marked here 2026-11-12: it and the held-out line below use `phantom_cf`, see "Correction (2026-10-21)" below; with `phantom_ep` S2b has no λ̂);
  - 200 held-out fresh TVs: 0.171 → 0.081 (SE 0.010).
- CARLA verification grid at `ctx0.8` (2026-10-21, same controller code as the 2026-10-20 sweep, 66/66 valid): unnecessary decelerations 12 → 4 of 14, but
  - normal / aggressive δ_max 2.1 → 18.5 % / 19.1 → 52.0 %,
  - j_max_cmd_filt 2.18 → 4.23 / 2.99 → 4.98,
  - prediction lead 2.83 → 1.17 / 2.04 → 0.72 s and
  - 3 of 18 aggressive runs below the L + d0 floor (none at λ 0 or 0.6).
- No λ meets both budgets in the cut-in-prone situations; the uniform pre-lateral discount trades cut-in safety for fewer phantom brakes.
- **Correction (2026-10-21): `phantom_cf` was flawed** — it restarts the counterfactual every tick from the main loop's previous command, so it only measures a one-tick difference capped by the 10 Hz jerk step (medians 0.52–0.55, max 0.70 m/s² at every λ, while those runs braked to −1.87 m/s² on average);
  - every flipped-design λ̂ above that used it (NGSIM all 0, S1 0.4, S2 1.0, S2b 0.8) and the statement that NGSIM meets both budgets for λ ∈ [0, 0.3] are withdrawn.
- Replacement `phantom_ep` = max(0, B_λ − B_leadonly), B = a run's largest braking command over its scored ticks (rule-B ticks voided), with a separate closed-loop lead-only run per episode in which every cut-in mode is removed (`replay_out/leadonly_20261021/`, 2,988 episodes, prereg 11:42; equals the CARLA `phantom_brake` when there is no lead).
- Flipped λ̂ with `phantom_ep` (`crc_select_ep_prereg.txt`; λ 0 → 1 mean):
  - NGSIM all 0.8 (197 cells; 0.252 → 0.051; report half at 0.8: 0.108),
  - S1 1.0 (158; 0.262 → 0.089; fresh 200 at 1: 0.104),
  - S2 none (79; 0.463 → 0.184),
  - S2b none (2,166; 0.479 → 0.151).
- With the cut-in side unchanged (NGSIM all 0.3, S1 0.2, CARLA 0), **no λ meets both budgets on any data set**.
- The S2b report set is easier than its calibration set (λ 0.8: 0.136 vs 0.231) because the S2 report run had already taken 200 strong-alarm TVs.

### Jerk pricing and the re-run

(2026-10-22, maintainer's decisions).

- NGSIM probes on three test episodes (ds3_v1467, ds2_v2009, ds3_v1459; scratch) showed that `rmove1000` with r_jerk 10 prices the step-0 command change at 4× an in-plan change (1000 vs r_jerk/dt² = 250), so the plan keeps postponing both braking and its release and the ego re-accelerates late after a cut-in.
- Tried on CARLA's lab-table cells (`cutin_verify_20261020` groups 01–04 + `verify_more_20261022` c1–c3, 50 cut-in + 50 no-cut-in runs per controller):
  - (a) `rmove0_jlim1.5` (§4 `jlim<x>`): normal / aggressive δ_max 3.7 → 9.4 % / 28.1 → 38.6 % at λ 0.4, j_max capped at 1.50;
    - with the baselines also at 1.5 (`acc_pcacc_jlim1.5`, `acc_isampc_qslack5_jlim1.5`) ISA-MPC's aggressive δ_max 23.5 → 52.2 % and PC-ACC's collisions 2 → 4 of 24 (PC-ACC's acceleration-bounded recapture path has no jerk bound, j_max 2.2–2.7);
    - its NGSIM calibration was stopped at 360 of 5,600 jobs (`latgate_cal_jlim_20261022`).
  - (b) **Adopted: `rmove1000_rjerk40`** (r_jerk 40, r_move = r_jerk/dt² = 1000, jerk bound 5; slack weight 5000 and D-opt w 100 unchanged): paired by cell against `rmove1000` at λ 0.4,
    - δ_max −1.5 %p normal (−2.2 SE) / −2.0 %p aggressive (−3.2 SE),
    - a_avg −0.11 / −0.10 m/s²,
    - braking onset 0.25 s earlier (6.3 SE),
    - j_max +0.29 / +0.60 m/s³,
    - no-cut-in phantom braking +0.06 m/s² (3.2 SE).
- Policies `..._trajcutin_ctx<λ>_latgate_rmove1000_rjerk40` (λ 0.4 / 0.6) and `..._trajcutin_rmove1000_rjerk40` (λ 0, groups 03 / 04) in the same CARLA roots.
- NGSIM re-run with `NGSIM_POLICY_TAIL=_rmove1000_rjerk40` (`results/ngsim_lambda_cal_20261019/replay/ngsim_replay.py`): `latgate_cal_rj40_20261022` (700 episodes × 8 λ), `leadonly_rj40_20261022`, `final_heldout_rj40_20261022` (400 × 11 λ);
  - launcher `results/acc_scenario_sweep/launch_rj40_ngsim_20261022.sh`, prereg notes in each root.
- Calibration (499 S2b cells): the flipped design's CRC bounds rise against `rmove1000` (λ 0.4 mean phantom 0.425 → 0.527 m/s², share ≥ 0.3 33.4 → 39.2 %; λ 0.6 0.325 → 0.421, 27.6 → 32.2 %); the previous design's λ̂ goes 0.2 → 0.3.
- The change was chosen after reading test episodes and the CARLA cells, so the test set is not strictly untouched.
- The aggregator's `t_comp_ms` (b6ec3cb) is the per-step computation time used by the lab tables.

### Verification sweep

(2026-10-20).

- Decided:
  - both baselines hold the fork's d_safe and the table scores δ against it;
  - PC-ACC uses the fork's limits;
  - ISA-MPC keeps the 3 s horizon and its original snap input and cost (method internals stay the authors'; only the task, the safe distance, the limits and the horizon are common conditions).
- Roots `cutin_verify_20261020` / `cutout_verify_20261020` (launcher `launch_verify_all_20261020.sh`), post-fix code (§8.13), 66 runs per policy, all valid.
- Cut-in normal / aggressive δ_max: SCC 55 / 84 %, PC-ACC 43 / 79 %, ISA-MPC 31 / 54 %, D-opt 2 / 19 %, D-opt + `ctx0.6` 9 / 34 %;
- ego collisions SCC 2 / 3, PC-ACC 0 / 2, the rest 0 (all in the ego 17 / TV 11 cell);
- j_max_cmd_filt ISA-MPC 1.19 / 1.75 vs D-opt 2.18 / 2.99;
- unnecessary decelerations in the 14 no-cut-in runs: SCC 0, PC-ACC 0, ISA-MPC 4, D-opt 12, `ctx0.6` 8.
- PC-ACC's `aggressive_cutin_0012` failed to spawn ("Ego lane is not adjacent", the same cell SCC lost in the paper sweep, §8.9), so PC-ACC has 17 aggressive runs;
  - SCC 02/0012 touched a guardrail with 0.3 % fallback (unchecked).
- The appendix ablation `acc_isampc_ainput_rmove1000` was smoother still on cut-in (j_max 0.73 / 1.14, a_min −1.21 / −1.76, δ_max 36 / 60 %), so ISA-MPC's smoothness is not due to the snap input;
  - in cut-out group 05 it undershot to 0.3–4.6 m/s, then passed v_max = 20 m/s, fell back and hit a guardrail in 2 of 8 runs.
- Still open for ISA-MPC: ε (0.8 vs CASE_3/4's 0.1) and an 8 s horizon check (`--ego-horizons 40`) (updated 2026-11-12: ε is settled for NGSIM at Zhou's published 0.8, ISA-MPC as published in the main table, "Calibrated lane-change probabilities" above; still open for CARLA).

### Code issues

- **`_fallback_policy` sign: fixed 2026-11-03** (cf48239, maintainer). It commanded `a_ref − 0.25·min(0, gap)` and so accelerated toward a lead that was too close; it now brakes (`+`). No run of the final sweeps or the NGSIM final set used the fallback (§8.1). (updated 2026-11-12: closed; listed here as a record only.)
- **Step-0 jerk cost.** All three QP builders price the step-0 command change with `config.dt` (0.2 s) instead of `command_dt` (0.05 s), which makes it 16× too cheap. The hard jerk constraint already uses `command_dt`.
- **Unfiltered x₀.** The MPC takes the raw measured ego speed as x₀. Most of the tick-to-tick speed noise that was attributed to tyre slip under zero throttle was the command/tick race (§8.11), fixed on 2026-10-15. Whether any slip noise remains has not been re-measured since.
- **σ-term cap.** The confidence chance constraint uses one capped β = min(p, β_ref) for both the standoff scale and the σ term, so the σ term never exceeds q(0.6)·σ ≈ 0.84σ. `sigmacap<c>` decouples them; `cutin_chance0.6` itself is unchanged so that the paper's sweeps reproduce. Whether the paper policy should switch is undecided. (updated 2026-11-12: moot for the NGSIM design, which uses none of β_ref's effects since NL, 2026-11-10: `chancecells`, `nopricesigma`, `floorrisk<ε>`, `lkonesided<ε>`; it still applies to `cutin_chance0.6` and the CARLA string.)
- **`slack_steps` misses slack use in chance policies.** It counts `feasible == False`, and in chance modes `feasible` already adds the slack, so a solve that leans on the 5000-weighted slack still counts as feasible. The raw per-step `slack_max` (logged since 2026-10-07) shows it. (updated 2026-11-12: the designs use the ℓ1 slack, `l1slack1e6`, since 2026-10-31, under which ξ > 0 means that the hard rows have no solution, and the NGSIM replay logs the slack cells as `slack_cells`; whether `slack_steps` still misses slack use under it is unverified as of 2026-11-12.)
- **STDAN grid edges.** The CARLA adapter puts its grid edges at ±6.5 cells of 4.6 m (±29.9 m). Training kept neighbours with |dy| < 90 ft (27.43 m), which cuts at the outer cells' centres. Most of the 2.5 m difference comes from this edge convention, not from the cell size.
- **TV re-trigger.** After the ego-proximity guard holds or aborts a TV lane change, the TV restarts it as soon as the ego is no longer alongside. The restart checks only the TV's gap to its own lead, never its gap to the ego.
- **Paper tex vs code.** The four differences are listed in §3.
- **Table caption.** The caption of `overleaf/table_cutin_final.tex` says every controller has 17 aggressive runs; only SCC does.
- **IAIMM-KF tuning.** IAIMM-KF's settings were tuned offline on replays of 58 final-sweep runs, i.e. in-sample. STDAN was never tuned on CARLA.
- **Branch point vs. conservatism.** The tree's conservatism on this grid comes from how much the cut-in cell demands, not from where the policy splits: `branchdec7` moved the split and changed nothing (2026-10-18).
  - For Proposed the demand saturates through α at p ≥ β_ref about 2 s before lane entry; for D-opt (w = 100) the price holds 87–100 % of d_safe at entry almost regardless of P.
  - The maintainer's 2026-10-07 observation that an earlier split was less conservative is not reproduced by the decision-point rule; whether a first-input-only split (k_b = 1) reproduces it is untested.
