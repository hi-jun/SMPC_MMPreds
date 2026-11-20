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

### 2026-11-13 night

- **D-opt's default is now the current design** (maintainer: "ISA랑 우리 설계 둘다 상한으로 바꾸고 우리 설계 평균 경로로 바꿔", then "재생기 기본값을 지금 설계로 전환"). That is the final design above:
  - without `nocap`: the cap 0 ≤ short ≤ τv is back;
  - without `occin0.1`: the mean-trajectory entry with the 0.1 gate;
  - with `aheadonly`.
- **The edit touches `POLICY_TAIL` only.** Backup `results/ngsim_lambda_cal_20261019/replay_backup_20261113_pre_capmp_default/`; md5 of `ngsim_replay.py` 24a66444 → see `replay_out/default_capmp_20261113/gate.log`.
  - The default tail is `_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold_entrant2_bcts_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_snap160_aheadonly`.
  - `NGSIM_POLICY_DOPT` stays `_dopt10`.
  - With no variable set, D-opt's string (λ 0, `--latgate`) is that of `replay_out/cap_20261113/design_cap_meanpath`, the NGSIM main table's design row (`main_capmp_20261113`).
- **Unchanged:** the condition variables and the baselines' default strings. `NGSIM_ISAMPC_POLICY` still defaults to ISA-MPC as published with the unified limits. The main table's ISA arm, `acc_isampc_amin3.5_amax2_jlim2.5_jlimup5_slackcap_l1slack5500`, is an explicit env string.
- **Gates** (`replay_out/default_capmp_20261113/gate.sh`, `gate.log`), on the 5 reproduction episodes (`finalset_v2_20261113/episodes_repro.csv`):
  - **No variable set, with the v3 episode table**, against `cap_20261113/design_cap_meanpath`: max |Δa_cmd| 0; every non-timing tick array equal; results.csv differs only in `wall_s`.
  - **The previous default tail as `NGSIM_POLICY_TAIL`, no episode table**, against `fin_npw3_iso_side_onerow_w10_z95_nocap_snap160_20261112`: the same, except one results.csv `slack_max` cell, which differs at the last ulp (1e-26, CSV round trip).
- **To reproduce a root made before this change:**
  - The 2026-11-12 final design and the roots spliced from it (`fin_v2_design_aheadonly_20261113`, `fin_v3_design_aheadonly_20261113`) need `NGSIM_POLICY_TAIL` set to the previous default tail. Append `_aheadonly` for the two `_aheadonly` roots.
  - The CRC grid of the BCTS shift λ (`replay_out/bcts_shift_20261112/`) ran on the previous default and is to be re-run on this one.

### 2026-11-19

The student chose "S를 논문 설계에 넣기 (주 표 설계 행·그림·재생기 기본값 갱신)": **D-opt's default is the thesis design.**

**What it contains.** The 11-13 night default plus every change since:
- without the safety layer (`safelayer`, `entrant2`, `fsdecel8_fsjerk10`; 2026-11-16);
- without `nocutout` (2026-11-18);
- `noegonoise` (2026-11-18);
- `pzrisk0.1` in place of `zconf0.95` (2026-11-19, d2db471);
- the lead's own lead S as a predictor target (`NGSIM_SUBLEAD_TARGET`, `replay_out/sublead_20261119/result.md`, `replay_out/pzsub_20261119/result.md`).

**The edit.**
- `ngsim_replay.py` (backup `results/ngsim_lambda_cal_20261019/replay_backup_20261119_pre_pzsub_default/` on the workstation and the lab server; md5 72e2702c → 8aef08a6, `run_replay.py` 8ed4a442 → e14eb621, both machines):
  - **`POLICY_TAIL`** is `_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_leadmerge_mergefirst_lkhold_bcts_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_onerow_pzrisk0.1_snap160_aheadonly_noegonoise`. `NGSIM_POLICY_DOPT` stays `_dopt10`.
  - **`NGSIM_SUBLEAD_TARGET`** is on by default. `0` or empty turns it off.
  - **S's scope:** it applies to D-opt's targets only (STDAN, or IAIMM-KF with `NGSIM_DOPT_PREDICTOR=iaimm`).
    - It never applies to the ISA-MPC baseline, whose obstacles are the replay's target set.
    - It never applies to CV-ACC, which is the in-lane lead only.
    - The replay's metrics (`lead_id`, `gap`) are unchanged.
- `run_replay.py` records `sublead_target` (as in effect) in every D-opt root's `run_info.json`.
- With no variable set, D-opt's string (λ 0, `--latgate`) and condition are those of `replay_out/pzsub_20261119/pzsub_cutin` and `pzsub_cutout`.

**Gates** (`replay_out/default_pzsub_20261119/gate_ws.sh`, `gate_srv.sh`, `gate_ws.log`, `gate_srv.log`). Each passing gate has max |Δa_cmd| 0, every non-timing tick array equal, and results.csv differing only in `wall_s`. Runs are the 5 reproduction episodes (`finalset_v2_20261113/episodes_repro.csv`) unless stated.

| gate | condition | against | result |
|---|---|---|---|
| new default, cut-in | no variable set, the v3 table | `pzsub_20261119/pzsub_cutin` | pass |
| new default, cut-out | the 5 cut-out smoke events | `pzsub_20261119/pzsub_cutout` | pass |
| previous default | its tail as `NGSIM_POLICY_TAIL`, `NGSIM_SUBLEAD_TARGET=0`, the v3 table | `cap_20261113/design_cap_meanpath` | pass |
| PC-ACC | Table I's Q_r, the main table's arm | `pcacc_qr15_20261117/pcacc_qr15` | pass |
| CV-ACC | `NGSIM_DOPT_PREDICTOR=cv`, its tail | `noegonoise_20261118/cv_cutin` | pass |
| ISA-MPC cap | `acc_isampc_amin3.5_amax2_jlim2.5_jlimup5_slackcap_l1slack5500` with `ISAMPC_IAIMM_CFG` as `cap_20261113/cap.sh` set it (`proj_horizon_s` 3.0) | `cap_20261113/isa_cap55` | pass (`ws_isa_iaimmcfg`) |

**ISA-MPC needs the root's IAIMM-KF setting.**
- The main table's ISA-MPC root ran with `ISAMPC_IAIMM_CFG='{"lat_speedup": 2.0, "proj_horizon_s": 3.0, "q_lat_scale": 1.0, "trans_eps": 0.25, "weight": [0.1, 0.3, 10.0, 0.02]}'` (`cap_20261113/cap.sh`; its `run_info.json` `iaimm_cfg`).
  - `baselines.py`'s default is `proj_horizon_s` 8.0.
- Without the variable, ISA-MPC differs from the root from tick 0 on: `ws_isa`, `ws_isa_oldtail` and `ws_isa_presub`, max |Δa_cmd| 4.0.
  - The cut-in probability of the run's TV differs already at tick 0 (0.0685 → 0.0024 on ds1_v2373).
  - This was first misread as a code drift. The replay versions back to 2026-11-13 all give the same, so it was not one.
  - (Corrected 2026-11-19 15:40.)

**To reproduce a D-opt root made before this change:** set `NGSIM_SUBLEAD_TARGET=0`. Also set `NGSIM_POLICY_TAIL` to that root's tail (`run_info.json` `policy_example`). The pre-switch roots of 2026-11-19, `sublead_20261119` and `pzsub_20261119`, set the variable to 1 explicitly.

### 2026-11-19 evening

The student adopted `pzjoint` ("넣고 ㄱㄱ"; `docs/agents/tokens.md`, `replay_out/pzjoint_20261119/result.md`).

**The edit.** The default `POLICY_TAIL` gets `_pzjoint` at its end. That is the only edit.
- `ngsim_replay.py` md5 9327f1d2 → 49a03e10 on the workstation and the lab server.
- Backups: `results/ngsim_lambda_cal_20261019/replay_backup_20261119_pre_pzj_default/` on both machines.
- Before that edit, the same day, `predictor_options` started reading the token (8aef08a6 → 9327f1d2, backup `replay_backup_20261119_pre_pzjoint/`).
- The repo code is eee8104, copied to the server with backup `/data/users/sjun0803/ACC/SMPC_MMPreds/repo_backup_20261119_pre_pzjoint/`. Its `WORKSTATION_HEAD.txt` was updated.

**Gates** (`replay_out/default_pzj_20261119/pzj_default_server.sh`, `.log`; lab server; the 5 reproduction episodes, v3 table). Each has max |Δa_cmd| 0:

| condition | against |
|---|---|
| no variable set | `pzjoint_20261119/pzj_cutin` |
| the previous default tail (without `_pzjoint`) | `pzsub_20261119/pzsub_cutin` |
| that tail with `NGSIM_DOPT_PREDICTOR=iaimm` and the ablation's `ISAMPC_IAIMM_CFG` (`proj_horizon_s` 3.0) | `pzsub_20261119/iaimm_cutin` |

**To reproduce a root of `pzsub_20261119`:** set `NGSIM_POLICY_TAIL` to the default tail without `_pzjoint`.

### 2026-11-19 evening, `noramp`

The student adopted `noramp` ("노램프 채택해"; `docs/agents/tokens.md`, `replay_out/noramp_20261119/result.md`).

**The edit.** The default `POLICY_TAIL` gets `_noramp` after `_pzjoint`. That is the only change to the default.
- `ngsim_replay.py` md5 b877a72c → 372aa4dc on the workstation and the lab server.
- Backups: `results/ngsim_lambda_cal_20261019/replay_backup_20261119_pre_nr_default/` on both machines.
- Before that edit, the same day, the replay's config started reading the token: `cfg.free_cruise_no_ramp = A._parse_no_speed_ramp(policy)`, md5 49a03e10 → b877a72c.
  - Its backup, `replay_backup_20261119_pre_noramp/`, is on the workstation only. It was rebuilt by removing that one line, and its md5 equals the server's copy before the edit.
- The repo code is 1e5284e, copied to the server with backup `/data/users/sjun0803/ACC/SMPC_MMPreds/repo_backup_20261119_pre_noramp/`. Its `WORKSTATION_HEAD.txt` was updated.
- `run_replay.py` is unchanged (e14eb621).

**Gates** (`replay_out/default_nr_20261119/nr_default_server.sh`, `.log`; lab server; the 5 reproduction episodes, v3 table). Each has max |Δa_cmd| 0:

| condition | against |
|---|---|
| no variable set | `noramp_20261119/nr_cutin` |
| the previous default tail (with `_pzjoint`, without `_noramp`) | `pzjoint_20261119/pzj_cutin` |
| that tail with `NGSIM_DOPT_PREDICTOR=iaimm` and the ablation's `ISAMPC_IAIMM_CFG` (`proj_horizon_s` 3.0) | `default_pzj_20261119/iaimm_cutin` |

**To reproduce a root of `pzjoint_20261119` or `default_pzj_20261119`:** set `NGSIM_POLICY_TAIL` to the default tail without `_noramp`.

**CV-ACC ("Conventional ACC" in the thesis) does not take the default tail.**
- Since 2026-11-19 evening its root is `replay_out/cvnr_20261119/cvz_cutin` / `cvz_cutout`: `NGSIM_DOPT_PREDICTOR=cv` with the old CV tail (`noegonoise_20261118/cv_cutin`'s `policy_example` tail) + `_noramp` (`cvnr_20261119/cvz_server.sh`).
- With the default tail, `pzrisk0.1`'s Z of the CV lead's frozen lateral position (zero covariance) switched the in-lane lead's cell off on 1,134 scored ticks in 186 of the 1,014 cut-in runs (`cvnr_20261119/result.md`).

### 2026-11-20, the target window: no band behind, and the in-lane range

The student decided both through session smpc-mmpreds-82:
- "뒤쪽 구간 없애기(코드 수정과 ISA-MPC 재실행 필요) 이걸로 하자 ... 실험은 서버에서 돌려"
- "ego lane에 대해선 200m 안의 차량만 고려한다는 내용도 추가해줘", then, after the check, "9런만 재실행하면 되지? 그럼 진행시켜 서버에서"

Record: `replay_out/adjwin_20261120/result.md`.

**The edits** (`ngsim_replay.py` 372aa4dc → bed4c0cd → 5cbdb6c2; `run_replay.py` e14eb621 → 8df9e4ea → 8f71814e; backups `results/replay_backup_20261120_pre_adjwin/` and `results/replay_backup_20261120_pre_laneahead/`; both machines):
- `NGSIM_ADJ_BEHIND_M`, default 0 (5 before): an adjacent target needs rear bumper + L ≥ the ego's front − this band. At 0 that is the point `aheadonly` compares.
- `NGSIM_ADJ_AHEAD_M`, default 30: d_adj, the adjacent targets' centre gap. Unchanged in value, now settable.
- `NGSIM_LANE_AHEAD_M`, default 200 (no bound before): the in-lane targets, the lead and D-opt's S, only within this centre gap.
  - The metrics' lead (`lead_id`, `gap`) stays the nearest in-lane vehicle.
- run_info.json records `adj_ahead_m`, `adj_behind_m` and `lane_ahead_m`. Their absence means 30 / 5 / no bound.

**Who reads the window:**
- D-opt (STDAN, IAIMM-KF and CV predictors) and ISA-MPC take their targets from it.
- PC-ACC (its own perception), phantom_ref's reference roots and `pred_testset_20261116` do not.

**Gates** (lab server; the 5 reproduction cut-ins and the 5 cut-out smoke events; max |Δa_cmd| 0 each):

| condition | against |
|---|---|
| `NGSIM_ADJ_BEHIND_M=5` (the window code) | `noramp_20261119/nr_cutin`, `nr_cutout`, `cap_20261113/isa_cap55`, `cutout_20261117/isa`, `default_nr_20261119/iaimm_cutin` |
| `NGSIM_LANE_AHEAD_M` inf, and the default 200 (the range code) | the adjwin roots below, every arm and set |

**Roots with the new defaults:** `replay_out/adjwin_20261120/{design,isa,cv}_{cutin,cutout}` and `iaimm_cutin`.
- They were run with the window code.
- The 8 runs in which the 200 m range binds (`check_lane_ahead.py`, exact on every logged tick) were re-run with it and spliced in (`SPLICE_CAP200.md` in each root; run_info.json `lane_ahead_m` 200).
- Against the old roots: ISA-MPC and ablation (b) change in most runs (up to 3.9 m/s²). The design and Conventional ACC change by ≤ 1.3e-3 m/s², except one design cut-in run (0.60).
- The thesis tables' cells move by ≤ 0.3 %p in δ_max, ≤ 0.05 m/s³ in j_max fair, and ≤ 0.009 in phantom_ref. No contact or floor-breach count changes.

**To reproduce a root made before 2026-11-20:** set `NGSIM_ADJ_BEHIND_M=5 NGSIM_LANE_AHEAD_M=inf`.

### 2026-11-20, the token `refhold` (default unchanged)

- `build_controller` reads `refhold` (one line after noramp's): `cfg.reference_hold_speed`, the reference's ego speeds held at the measured speed in place of the previous solve's plan (`tokens.md`, `refhold`).
- The token is not in the default tail; `NGSIM_POLICY_TAIL` = the default tail + `_refhold` switches it on.
- md5 `ngsim_replay.py` 5cbdb6c2 → 3ddeab03 (backup `results/replay_backup_20261120_pre_refhold/`); `run_replay.py` unchanged (8f71814e). The lab server got it through `setup_acc_replay_env.sh push`, HEAD 1cf1c65.
- Gate (`replay_out/refhold_20261120/refhold_server.log`): without the token, REP and CO5 are bit-identical to `adjwin_20261120/design_cutin` and `design_cutout`.
- Declined by the student the same day (through smpc-mmpreds-82: "refhold: 채택하지마"). The line stays; the default does not use the token.

### Condition of older roots

- `NGSIM_ADJ_BEHIND_M=5 NGSIM_LANE_AHEAD_M=inf` gives the target window of every root made before 2026-11-20.

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
