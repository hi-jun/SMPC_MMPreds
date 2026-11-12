# NGSIM replay defaults: history and how to reproduce older roots

Moved verbatim from AGENTS.md on 2026-11-12 (git `3d9bcf8`); the text is unchanged. Reformatted for readability the same day: the words are unchanged and long paragraphs became bullets (checks: `docs/agents/_reformat/check_*.py`). It was an item of AGENTS.md §10. The current defaults are in AGENTS.md §1 (Current design). (updated 2026-11-12: stale statements have since been marked in place with dated "updated" and "pointer updated" notes, and the replay edit for `onerow` was added; the reformat checks read this file at commit 38a1b46.) (updated 2026-11-12 night: the replay edits for `zconf<β>`, `nocap` and `snap<r>` and the final design's default were added, 2026-11-12 below.)

## Defaults, by date

### 2026-10-27

- **NGSIM replay defaults (2026-10-27, maintainer).** `results/ngsim_lambda_cal_20261019/replay/ngsim_replay.py` now starts every episode with
  - set speed 20 m/s (`NGSIM_V_SET`, before: the logged follower's maximum, below the start speed in 12 % of cut-ins and 6 % of no-cut-ins)
  - and previous command 0 (`NGSIM_START_CMD`, before: the lead's logged acceleration, which forced d_safe violations in the warm-up)
  - `logged` restores either.
- Runs before 2026-10-27 used the logged values unless the variable was set, so do not mix them with new runs in one table.
- Since 2026-10-27 evening its default policy tail (`NGSIM_POLICY_TAIL`, the tokens after `ctx<λ>`) is `_entrypass0_rmove1000_rjerk40`
  - every run before that which did not set the variable used `_rmove1000`, and `NGSIM_POLICY_TAIL=_rmove1000_rjerk40` gives the design without entrypass.
- The held-out table is being re-run under the new defaults (`replay_out/launch_final_heldout_v20s0_20261027.sh`, prereg `final_heldout_v20s0_rj40_20261027/prereg.md`) (updated 2026-11-12: done on 2026-10-27, roots `final_heldout_v20s0_*_20261027`, result `replay_out/cal_v20s0_20261027/result.md`).

### 2026-10-28

- Since 2026-10-28 evening the default tail is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40` (the adoption, §1)
  - `NGSIM_POLICY_TAIL=_entrypass0_rmove1000_rjerk40` gives the design before it.

### 2026-10-29

- Since 2026-10-29 afternoon the default tail is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_alag0.5_ra0` (§1)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40` gives the design before it.
- Since 2026-10-29 evening it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0`, with `NGSIM_EGO_LAG_TAU` 0 (§1).

### 2026-10-30

- Since 2026-10-30 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot` (maintainer, §1)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0` gives the design before it.

### 2026-10-31

- Since 2026-10-31 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6` (maintainer, §1; the defaults reproduce `fin_jinput_ra0_fullfoot_l1slack1e6_20261031` on 2 checked episodes, max \|Δa_cmd\| = 0)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot` gives the design before it.

### 2026-11-02

- Since 2026-11-02 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer` (maintainer, §1; the defaults give the policy format of `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_20261102` and its lead-only root exactly)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6` gives the design before it.

### 2026-11-03

- Since 2026-11-03 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge` (maintainer, §1; the defaults give the policy format of `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_20261103`)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer` gives the design before it.
- Since 2026-11-03 12:40 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch` (maintainer, §1; the defaults give the policy format of `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_20261103` exactly)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge` gives the design before it.
- Since the evening of 2026-11-03 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst` (maintainer, §1; the defaults reproduce `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_20261103` bit for bit on 12 smoke + 3 lead-only episodes)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch` gives the design before it.
- Since the night of 2026-11-03 it is `_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma` (maintainer, §1; the defaults reproduce `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_20261103` bit for bit on the 12 smoke episodes and the mergefirst lead-only root on 3, `replay_out/lkrsigma_20261103/adopt_lkrsigma.log`)
  - `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst` gives the design before it.
- Since later that night (the adoption of H) it ends `_lkrsigma_lkhold` and `NGSIM_LANE_TEST` defaults to `path` (the defaults reproduce `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_path_20261103` bit for bit on 12 smoke + ds5_v2609 and its lead-only root on 3, `replay_out/lanefix_20261103/adopt_path.log`)
  - the tail ending `_lkrsigma` with `NGSIM_LANE_TEST=midpoint` gives the design before it.

### 2026-11-04

- Since 2026-11-04 it ends `_lkhold_entrant2` (maintainer, §1; the defaults reproduce `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_entrant2_20261104` bit for bit on 12 smoke + ds5_v2609 and its lead-only root on 3, `replay_out/entrant_20261104/adopt_entrant2.log`)
  - the tail ending `_lkhold` gives the design before it.

### 2026-11-10

- Since the evening of 2026-11-10 it ends `_entrant2_bcts_occin0.1_nolkentry_pingate_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05` (maintainer, NL, §1; the defaults reproduce `fin_bcts_occin0.1_nolkentry_pingate_floorhyp0.1_nopricesigma_lkonesided0.05_20261110` bit for bit on 12 smoke + ds5_v2609 and its lead-only root on 3, `replay_out/nps_lk_20261110/adopt_nl.log`)
  - the tail ending `_entrant2` gives the paper root before calibration.
- Since the night of 2026-11-10 the default chance token (`NGSIM_POLICY_CHANCE`) is `_chancecells` in place of `_cutin_chance0.6`, the same switch (bit-identical on 12 smoke + ds5_v2609 and 3 lead-only episodes, `replay_out/nps_lk_20261110/adopt_chancecells.log`)
  - `NGSIM_POLICY_CHANCE=_cutin_chance0.6` gives the old string.

### 2026-11-11

- Since the night of 2026-11-11 the default tail drops `_entrypass0` and `_pingate`, and `NGSIM_POLICY_DOPT` defaults to `_dopt3lkr` (NP with w 3, maintainer, §1).
  - With `NGSIM_V_SET=logged NGSIM_SPAWN_MARGIN=3` the defaults reproduce `fin_nl_noep_prodgate_w3_sm3_vlog_20261111` bit for bit on the 13 smoke episodes, and its lead-only root `lo_fin_nl_sm3_vlog_20261111` on 3 (`replay_out/startmargin_20261111/adopt_npw3.log`).
  - The NL tail with `NGSIM_POLICY_DOPT=_dopt100lkr` gives the design before it.
- Later that night the default condition changed too: `NGSIM_V_SET` defaults to `logged` (was 20) and `NGSIM_SPAWN_MARGIN` to 3 (was 0).
  - This is the condition of the 2026-11-11 comparisons (maintainer, §1).
  - With no variable set, the defaults reproduce:
    - the NP w3 root and its lead-only root bit for bit on the same episodes;
    - ISA-MPC (with `ISAMPC_Q_SLACK=5` and its usual `ISAMPC_IAIMM_CFG`) and PC-ACC at that condition on the 13 smoke episodes (`adopt_npw3.log`).
- Later that night the default tail also drops `_safebranch` (maintainer: the NGSIM design without the fail-safe branch, §1; `replay_out/iso_limits_20261111/default_tail.patch`).
  - With no variable set the defaults give the policy string of P0 (`fin_npw3_nob_sm3_vlog_20261111`) and reproduce its explicit-tail runs bit for bit on the 13 smoke episodes and the 3 lead-only check episodes.
  - The older roots stay reproducible with `NGSIM_POLICY_TAIL` set to the old default (`_safebranch` between `_leadmerge` and `_mergefirst`), checked bit for bit and with the same policy strings:
    - `fin_nl_noep_prodgate_w3_sm3_vlog_20261111` and `defcond13_20261111` on the 13 smoke episodes,
    - `defcond_lo3_20261111` on the 3,
    - and the predictor-swap roots on the 13, `fin_npw3_iaimm_bcts_sm3_vlog_20261111` (`NGSIM_DOPT_PREDICTOR=iaimm`, the old default tail) and `fin_npw3_iaimm_raw_sm3_vlog_20261111` (the old default tail without `_bcts`, as it ran) (`replay_out/iso_limits_20261111/default_tail_check.out`).

### 2026-11-12

- **Since 2026-11-12 (maintainer) the defaults are the unified limits and live side entries.**
  - The default tail ends `_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10`, so D-opt's default string is P1's (`fin_npw3_nob_iso_sm3_vlog_20261111`) (updated 2026-11-12 night: until that night, when the final design became D-opt's default, below; the limits and the rest of this bullet stay).
  - `NGSIM_ISAMPC_POLICY` defaults to `acc_isampc_amin3.5_amax2_jlim2.5_jlimup5` (q_slack unset, Zhou's 0.055; `ISAMPC_IAIMM_CFG` stays an env setting)
    - and `NGSIM_PCACC_POLICY` to `acc_pcacc_amin3.5_amax2_jlim2.5_jlimup5` (the port; PC-ACC as published stays the env string `acc_pcacc_paperdsafe_amin3.5_amax2_jlim2.5_jlimup5_emamin8_emjlim10_emics`).
  - `NGSIM_SIDE_ENTRY` (new) defaults to `live`: a vehicle that changes lanes into the ego lane beside the ego stays live (Route B, `open_items.md`; pointer updated 2026-11-12: it read "Route B below")
    - a track start or re-appearance beside the ego stays a ghost (a vehicle entering the recording at its upstream edge beside an ego that lags its logged position is an unrecorded follower; a first version that kept them live was stopped on 2026-11-12, `replay_out/side_live_20261112/aborted_trackstart/`)
    - `ghost` gives Rule A as before (pointer updated 2026-11-12: Rule A is the ghost rule of Route B's "Replay rules", `open_items.md`, and of `ngsim_replay.run_episode`).
    - run_info.json records it as `side_entry`.
    - Diagnostics:
      - tick arrays `side_lead`, `side_lead_body`, `side_ovl`, `side_ovl_body`,
      - results columns `n_side_live`, `side_entries`, `n_side_ghosted_behind`, `n_side_ovl_scored`, `n_side_ovl_body_scored`, `side_ghosts`, `n_side_ghost_{lane_change,track_start,reappear}`,
      - and `ngsim_metrics`' `coll_side`, `coll_side_body`, `coll_rear_end`, `d_max_noside`.
  - Gates (`replay_out/side_live_20261112/gates_side.out`):
    - with `NGSIM_SIDE_ENTRY=ghost` and nothing else set, P1 (13 smoke episodes), its lead-only and reference roots (3) and the ISA-MPC 0.055, PC-ACC port and PC-ACC published ISO roots (13) are reproduced bit for bit with the same strings
    - with `ghost`, the tail without the ISO tokens, `NGSIM_ISAMPC_POLICY=acc_isampc` and `NGSIM_PCACC_POLICY=acc_pcacc`, P0 (13 + 3) and the 2026-11-11 ISA-MPC 0.055 and PC-ACC port roots (13) are
    - with `live` every smoke episode without a side entry is unchanged (none of the 13 has one).
  - To reproduce a root made before 2026-11-12, set `NGSIM_SIDE_ENTRY=ghost` and its own explicit strings (for the fork's limits the tail without the ISO tokens, `NGSIM_ISAMPC_POLICY=acc_isampc`, `NGSIM_PCACC_POLICY=acc_pcacc`).
- **Replay edit for `onerow` (2026-11-12; line added 2026-11-12).** One line in `build_controller`, `cfg.one_row = A._parse_one_row(policy)`, so that the replay reads the token as the agent does (`replay_out/onerow_20261112/replay_edit.patch`; backup `replay_backup_pre_onerow/`).
  - md5 of `ngsim_replay.py` 8a5e75a8 → 03718256 (`md5_replay_pre_onerow.txt`, `md5_replay_post_onerow.txt`); the defaults are unchanged.
  - Bit-identical for every string without `onerow`: the parser then returns False, `NairACCConfig.one_row`'s default.
  - Checked on the default string: 13 smoke episodes against `fin_npw3_iso_side_20261112` and 3 lead-only episodes against `lo_fin_npw3_iso_side_20261112`, every non-timing tick array equal (`replay_out/onerow_20261112/ws_gate_20261112.out`).
- **Replay edits for `zconf<β>`, `nocap` and `snap<r>` (2026-11-12; lines added 2026-11-12 night).** One line each in `build_controller`, so that the replay reads the tokens as the agent does (`replay_out/onerow_20261112/replay_edit_zconf.patch`, `replay_edit_nocap.patch`, `replay_edit_snap.patch`; backups `replay_out/onerow_20261112/replay_backup_pre_zconf/`, `replay_backup_pre_nocap/`, `replay_backup_pre_snap/`):
  - `cfg.one_row_z_conf = A._parse_one_row_z_conf(policy)`: md5 of `ngsim_replay.py` 03718256 → 70d986aa;
  - `cfg.one_row_no_cap = A._parse_one_row_no_cap(policy)`: 70d986aa → b803b6b7;
  - `cfg.snap_weight = A._parse_snap(policy)`: b803b6b7 → 2030de45.
  - The defaults were unchanged by them. Each was gated on the workstation and the lab server: the default string bit-identical to `fin_npw3_iso_side_20261112` / `lo_` on 13 + 3 episodes (strict), and the arm before without the new token tick-level identical to its roots (`ws_gate_zconf_20261112.out`, `ws_gate_nocap_20261112.out`, `ws_gate_snap_20261112.out` and the server logs `server_run_zconf.log`, `server_run_nocap.log`, `server_run_snap.log`, all in `replay_out/onerow_20261112/`).
- **Since the night of 2026-11-12 (maintainer: "C + snap160 is the final NGSIM design") D-opt's default is the final design**, `onerow` with w 10 + `zconf0.95` + `nocap` + `snap160` (`replay_out/onerow_20261112/replay_edit_adopt_snap160.patch`; backup `results/ngsim_lambda_cal_20261019/replay_backup_20261112_pre_snap160/`; md5 of `ngsim_replay.py` 2030de45 → e78134fc, `md5_replay_pre_adopt.txt`, `md5_replay_post_adopt.txt`).
  - `NGSIM_POLICY_DOPT` defaults to `_dopt10` (was `_dopt3lkr`).
  - The default tail is `_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold_entrant2_bcts_occin0.1_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_nocap_snap160` (was P1's, `_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10`).
  - With no variable set, D-opt's string (λ 0, `--latgate`) is that of `fin_npw3_iso_side_onerow_w10_z95_nocap_snap160_20261112`: `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold_entrant2_bcts_occin0.1_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_nocap_snap160`.
  - Unchanged: `NGSIM_POLICY_CHANCE`, the condition (`NGSIM_V_SET`, `NGSIM_SPAWN_MARGIN`, `NGSIM_START_CMD`, `NGSIM_EGO_LAG_TAU`, `NGSIM_V_MAX`, `NGSIM_LANE_TEST`, `NGSIM_SIDE_ENTRY`) and the baselines' strings (`NGSIM_ISAMPC_POLICY`, `NGSIM_PCACC_POLICY`); the edit touches `POLICY_DOPT` and `POLICY_TAIL` only, which only D-opt's string reads. The code (f1bfbc4) and the CARLA agent are unchanged.
  - Gates (`replay_out/onerow_20261112/adopt_snap160.log`; `adopt_snap160.sh`, outputs `adopt_snap160_ws.out`, `adopt_snap160_server.out`), on the workstation and, after the setup push, on the lab server:
    - with no variable set, the 13 smoke episodes (`ghostfix_20261103/episodes_smoke12_ds5v2609.csv`, λ 0, `--latgate`) against `fin_npw3_iso_side_onerow_w10_z95_nocap_snap160_20261112` and the 3 lead-only check episodes against its `lo_` root: the same policy strings, max |Δa_cmd| 0, every non-timing tick array equal; results.csv differs only in `wall_s` and, in the lead-only root, in two cells at the last ulp (the splice's pandas round trip);
    - P1 explicit (`NGSIM_POLICY_DOPT=_dopt3lkr`, `NGSIM_POLICY_TAIL` = P1's tail above) on the same 13 + 3 against `fin_npw3_iso_side_20261112` / `lo_fin_npw3_iso_side_20261112`: bit for bit, results.csv only `wall_s`;
    - the server's runs equal the workstation's (every non-timing tick array, results.csv only `wall_s`); 0 errors, exceptions and fallbacks in every gate run.
  - To reproduce a root made before this change, set its strings explicitly:
    - P1 (`fin_npw3_iso_side_20261112`, `lo_`): `NGSIM_POLICY_DOPT=_dopt3lkr` and P1's tail as `NGSIM_POLICY_TAIL` (checked above); the ℓ1-slack-100 roots (`fin_npw3_iso_side_l1w100_20261112`, `lo_`) the same with `_l1slack100` in place of `_l1slack1e6`;
    - the oracle-lead reference `ref_lo_oraclelead_iso_side_20261112` (the reference of `phantom_ref`): `--lead-only --oracle-lead` with `NGSIM_POLICY_DOPT=_dopt3lkr` and P1's tail without `_bcts` (the oracles have no probabilities to calibrate, so the replay refuses `bcts` with them and an oracle-lead run always needs an explicit tail; the final design's own reference would be `NGSIM_POLICY_DOPT=_dopt10` with its tail without `_bcts`, not run);
    - the other `onerow` roots of 2026-11-12: `NGSIM_POLICY_DOPT` and `NGSIM_POLICY_TAIL` from their run_info.json `policy_example`, e.g. arm C (`fin_npw3_iso_side_onerow_w10_z95_nocap_20261112`) = `_dopt10` with the new tail without `_snap160`, arm A without `_nocap_snap160`, onerow w 3 / 10 / 30 / 100 = `_dopt3` / `_dopt10` / `_dopt30` / `_dopt100` with the new tail ending `_fsjerk10_onerow`;
    - the baselines' 2026-11-12 roots keep their settings (their defaults did not change);
    - roots made before 2026-11-12 additionally need `NGSIM_SIDE_ENTRY=ghost` (above).

### Condition of older roots

- `NGSIM_V_SET=20 NGSIM_SPAWN_MARGIN=0` gives the condition of every root from 2026-10-27 to 2026-11-10, including the paper roots and the baseline roots of the earlier tables.

## Other replay settings

### `NGSIM_LANE_TEST` (2026-11-03 night)

- `NGSIM_LANE_TEST` (2026-11-03 night, final-check items 4 and 5): how D-opt's predicted trajectories (steps 1-15) are tested for the ego lane
  - `midpoint` (the default until the adoption of H, every run before) = inside the midpoints to the neighbouring lanes' binned centres,
  - `markings` = the lane-marking map built from the Lane_ID transitions (`baselines.LaneMarkings`), with the marking on a vehicle's side moved to its current position where that position and its Lane_ID disagree
  - `body` = the same markings, in the lane when the vehicle's body (Local_X ± half its logged width) overlaps the lane at all (CommonRoad's lanelet assignment by shape), no anchoring
  - `path` = in when the vehicle's body overlaps the ego's body laterally, |x − x_ego| < (w + w_ego)/2, the ego held at its current offset from its lane's marking centre (the safety layer's path test carried to the predicted points), no lane line
- step 0 is the Lane_ID either way, and the baselines always keep `midpoint` (run_info.json records it; result in §10's open items; pointer updated 2026-11-12: now `open_items.md`, "Safety and recursive feasibility", final-check items 4 and 5).

### Rule A warm-up ghosts (2026-11-03)

- Since the evening of 2026-11-03 the replay's Rule A also makes a ghost, at the warm-up start, of every vehicle in e's lane whose logged front is behind e's logged front (maintainer, final-check item 9): the equilibrium start had moved the ego 10.5 m back behind e's logged follower in ds5_v2609, which then led it on all 79 scored ticks for every controller.
- On the final set it changes only that episode
  - the roots with it re-run are spliced (`*_ghost_20261103`, `replay_out/ghostfix_20261103/result.md`), and with it and `safebranch` the safety layer acts on no tick of the 889 runs.

### Start speed and `NGSIM_V_MAX` (2026-10-29)

- Since 2026-10-29 evening the replay clips the equilibrium start's speed to the speed bound (row field `start_v_clipped`, as the highD copy does): ds4_v2775 (21.97 m/s) and ds1_v3105 (20.09) started above 20 and fell back at tick 0 (D-opt 1 run, ISA-MPC 2).
- With the clip none of the three controllers falls back there, and every other checked episode reproduces the old roots bit for bit (`replay_out/startclip_check_{dopt,isampc,pcacc}_20261029`, the 8 episodes that ever reach 19.5 m/s + 12 smoke).
- `NGSIM_V_MAX` (default 20) sets the speed bound of the D-opt and ISA-MPC configs and the plant clip (PC-ACC keeps its own 51).
- Raised to 30 for an hour that evening, it showed that our controller overshoots the set speed 20 while it cruises without a lead: in ds4_v3355 / ds5_v2794 the ego reaches
  - 23.2 / 21.2 m/s with `jinput_ra0` (still accelerating at the episode end),
  - 22.5 / 20.5 with `jinput`,
  - 21.4 / 21.1 with `ra0`,
  - 20.8 / 20.4 with neither,
  - while ISA-MPC and PC-ACC do not (`vmax30_check_*`, `vmax30_overshoot_*_20261029`).
- The maintainer kept the hard bound at the set speed, which caps it.
