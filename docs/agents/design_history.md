# Design history

Moved verbatim from AGENTS.md on 2026-11-12 (git `3d9bcf8`); the text is unchanged. Reformatted for readability the same day: the words are unchanged and long paragraphs became bullets (checks: `docs/agents/_reformat/check_*.py`). The current design is summarised in AGENTS.md §1.

## Design decisions since 2026-10-17 (AGENTS.md §1)

### 2026-10-17

- **The current design is D-opt, `dopt100lkr_trajcutin`** (the maintainer's decision on 2026-10-17).

### 2026-10-22

- Since 2026-10-22 the maintainer runs it with `rmove1000_rjerk40` in place of `rmove1000` (§10).

### 2026-10-27

- Since 2026-10-27 it also carries `entrypass0` (§4)
  - w stays 100, `pricep2` is dropped and the emergency slack (5000) stays (maintainer's decisions after the 2026-10-26 NGSIM probes).
- `entrypass9` was chosen first
  - after the re-calibration under the new replay defaults (`replay_out/cal_v20s0_20261027/result.md`: at λ 0 its phantom-braking risk is 0.330 m/s² against the 0.527 that λ 0.4 used to guarantee, and it led every held-out metric) the maintainer adopted `entrypass0`, attributing its one risk case (ds3_v1358, the real cut-in's hypothesis dropped 0.8 s before entry) to the replay environment, whose logged vehicles do not react to the simulated ego.
- From 2026-10-27 `entrypass0` is the default option for every new CARLA and NGSIM run (maintainer's decision)
  - a run may leave it out only when it is about entrypass itself or reproduces an older root, and must then say so (full string in §4's current-design table).

### 2026-10-28

- Since 2026-10-28 it also carries `nocutout` and `lksigma` (§4), the maintainer's adoption after the full evaluation on the final NGSIM paper set (`replay_out/ncls_eval_20261028/result.md`, λ 0, paired against the design without them):
  - critical cut-in δ_max −3.8 %p (SE 0.8), a_min +0.33 (0.04), j_max −0.46 (0.06)
  - mild cut-in δ_max −1.9 %p (0.8)
  - no-cut-in j_max −0.34 (0.04), runs ≤ −2 m/s² 9 → 6, contacts 1 → 0
  - costs: critical T_rec +0.06 s (0.04) and +15 ms per tick
  - the re-calibrated λ̂ stays 0.

### 2026-10-29

- On 2026-10-29 afternoon the paper's NGSIM table first ran it with `alag0.5_ra0` (§4: a first-order acceleration lag τ = 0.5 s in the prediction model, r_a 0) on a replay ego with the same lag for every controller (`NGSIM_EGO_LAG_TAU` 0.5; `replay_out/alag_eval_20261029/result.md`: against the design without it, critical cut-in δ_max −3.4 %p (SE 0.8), T_rec −0.10 s (0.03), λ̂ 0).
- That evening the maintainer dropped the lagged ego because it is unfair to the baselines, whose models do not know the lag
  - planning them from the measured acceleration made them worse (contacts PC-ACC 26 → 53, ISA-MPC 7 → 41).
- **Since 2026-10-29 evening the paper's NGSIM table uses the ideal replay ego (no lag, as every root before 2026-10-29) and runs the design with `jinput_ra0`** (§4: the jerk as the MPC input, r_a 0; maintainer's decision after `replay_out/jinput_20261029/result.md`), scoring acceleration and jerk on the ego's actual acceleration (§10).
  - Paired against the acceleration-input design (r_a 1):
    - one-tick j_max −0.98 / −0.87 / −0.50 m/s³ (critical / mild / no cut-in; SE 0.06 / 0.06 / 0.03), now below ISA-MPC's in every group (−1.00 / −0.43 / −0.41, SE 0.10 / 0.09 / 0.06), phantom_ep −0.022 m/s² (0.007)
    - cost: critical δ_max +2.4 %p (0.8), floor breaches 6 → 9
    - no contact.
  - `jinput` and `ra0` changed together, and λ is not re-calibrated for it yet.
  - Not decided for CARLA.

### 2026-10-30

- **Since 2026-10-30 the current design also carries `fullfoot`** (§4: a cut-in cell's floor keeps the whole L + d0, factor 1 in place of α
  - the σ term keeps β_j = min(p, β_ref)
  - maintainer's decision after `replay_out/fullfoot_20261029/result.md`, which against the design with α measured critical / mild δ_max −2.1 / −2.0 %p (SE 0.7 / 0.4) and no-cut-in phantom_ep +0.052 m/s² (SE 0.010)).
  - λ is not re-calibrated for it (maintainer: not yet), and it has not been run in CARLA.

### 2026-10-31

- **Since 2026-10-31 it also carries `l1slack1e6`** (§4: the emergency slack costs 1e6·Σξ in place of 5000·Σξ², an exact penalty, so ξ > 0 now means that the hard rows have no solution
  - maintainer's decision after `replay_out/l1slack_20261031/result.md`, which against the design with `fullfoot` measured critical δ_max −0.9 %p (SE 0.3), j_max +0.06 to +0.09, phantom_ep +0.000 (SE 0.004), contacts and fallbacks 0).
  - λ is not re-calibrated for it either, and it has not been run in CARLA.

### 2026-11-02

- **Since 2026-11-02 the paper's NGSIM tables run it with `safelayer`** (§4: PC-ACC's safety layer around the command
  - maintainer's decision after `replay_out/safelayer_20261102/result.md`, where the layer acted on 2 warm-up ticks in 889 runs and left every table column unchanged
  - tables `replay_out/safelayer_20261102/tables_paper_safelayer.md`, one with PC-ACC as ported and one with PC-ACC as published).
  - Not decided for CARLA, where it has never run.

### 2026-11-03

- **Since 2026-11-03 it also carries `leadmerge`** (§4: joint modes with the same effective lead at every step are one scenario, probability summed
  - maintainer's decision after `replay_out/leadmerge_20261103/result.md`, where every paper-table column was unchanged and the QP time per tick fell 121 → 40 ms on average).
  - Not run in CARLA.
- **Since 2026-11-03 the paper's NGSIM tables also run it with `safebranch`** (§4: the safety layer's fail-safe as one more branch of the scenario tree, so that the MPC plans the contingency itself
  - maintainer's decision, adopted as is with the branch weight 1e-3, after `replay_out/safebranch_20261103/result.md`: every printed table cell moves by at most one unit of its last digit, the pre-registered 2-SE rule failed on those third-decimal shifts, and the branch removed the design's one fail-safe warm-up tick
  - tables `replay_out/safebranch_20261103/tables_paper_safebranch.md`).
  - Like `safelayer`, which it requires, it is not decided for CARLA.
- **Since the evening of 2026-11-03 it also carries `mergefirst`** (§4: every joint mode merged before the `modes8` cap, the classes ranked by their summed probability
  - maintainer's decision after `replay_out/mergefirst_20261103/result.md`: `leadmerge`'s cap had cut a distinct lead sequence on 0.38 % of scored ticks and put a less likely class at scenario 0 on 0.04 %
  - printed table cells unchanged except critical j_max 2.06 → 2.05 and one threshold run in no-cut-in a_min ≤ −2 (8 → 7)
  - tables `replay_out/mergefirst_20261103/tables_paper_mergefirst.md`
  - with the replay's ghost fix of that evening (§10) `replay_out/ghostfix_20261103/tables_paper_ghostfix.md`).
  - Not run in CARLA.
- **Since the night of 2026-11-03 it also carries `lkrsigma`** (§4: lkr's release test also asks for the step-0 σ term that the released hard row requires
  - maintainer's decision, final-check item 6, after `replay_out/lkrsigma_20261103/result.md`: printed table cells unchanged except critical j_max 2.05 → 2.03, step-0-only slack ticks 38 → 35 (critical) and 183 → 181 (mild, with the ghost fix), no time cost at equal load
  - tables `replay_out/lkrsigma_20261103/tables_paper_lkrsigma.md`).
  - Not run in CARLA.
- **Since the night of 2026-11-03 it also carries `lkhold`, and the NGSIM replay tests its predicted trajectories by overlap with the ego's path** (`NGSIM_LANE_TEST=path`, §10: a vehicle is in at a predicted step when its body overlaps the ego's laterally, the safety layer's own test, in place of the midpoints between binned lane centres
  - maintainer's decision, final-check items 4 and 5, after `replay_out/lanefix_20261103/result_path.md`: critical δ_max −2.5 %p (SE 0.5), T_rec −0.10 s, no-cut-in phantom_ep +0.036 (SE 0.011), j_max +0.10 to +0.16, no contact, no safety-layer tick
  - tables `replay_out/lanefix_20261103/tables_paper_path.md`).
  - Not run in CARLA, whose lane test is the map's.

### 2026-11-04

- **Since 2026-11-04 the paper's NGSIM tables also run it with `entrant2`** (§4: the safety layer also checks its close entrants, vehicles that join the ego's path inside the layer's safe distance, against them braking at −2 m/s²
  - maintainer's decision, final-check item 12, after `replay_out/entrant_20261104/result.md`: close entrants in 3 of 889 runs, all avoidable at entry, the layer acting on no tick, every printed table cell unchanged except critical δ_avg 0.73 → 0.72 and fallback runs 1 → 0, both from a load-induced time-limit fallback tick of the earlier root
  - tables `replay_out/entrant_20261104/tables_paper_entrant2.md`).
  - Like `safelayer`, which it requires, not decided for CARLA.

### 2026-11-06

- **Since 2026-11-06 the NGSIM design also carries `pcal` and `occin0.1`** (§4: the calibrated lane-change probabilities, and cut-in cells switched on where p·P_in(k) > 0.1
  - maintainer's decision after `replay_out/pcal_tune_20261105/result.md`, where no arm passed the pre-registered rule: on the tuning set `occin0.1` with `pcal` gave no-cut-in phantom_ep 0.346 → 0.074 m/s² and runs a_min ≤ −2 27 → 15, at critical δ_max +5.6 %p (SE 2.0) and floor breaches critical 3 → 4, mild 0 → 1, which the maintainer accepted).
  - Final set (2026-11-06, `replay_out/pcal_cvar_20261106/result.md`, root `fin_pcal_occin0.1_20261106` + `lo_`; paper layout `replay_out/pcal_cvar_20261106/tables_paper_pcal.md`; `cvar` on top was tuned first and not selected: every `cvar` arm braked more on no-cut-ins than `occin0.1` alone, 0.083–0.117 against 0.074 m/s², and ε 0.05 and 0.1 give identical runs under `occin0.1`), paired against the paper root with `entrant2`:
    - no-cut-in phantom_ep 0.188 → 0.032 m/s² (SE 0.019), runs a_min ≤ −2 11 → 5, j_max −0.39
    - critical δ_max 9.1 → 16.4 % (+7.3 %p, SE 1.4), Δt_ant 0.97 s (140 runs) → 0.42 s (120 runs), j_max +0.92 (runs at the 5 m/s³ bound 18 → 50 critical, 24 → 57 mild, plus 3 fail-safe runs
      - corrected 2026-11-06 evening: the TV's labelled calibrated probability rises as gradually as the raw one, 0.9 s from 10 to 90 %, only about 1.1 s later
      - the step to p ≈ 1 is the summed cut-in probability, into which `trajcutin` puts the entering lane-keeping trajectory with the large calibrated q_LK, in 34 of 54 / 42 of 58 jump runs of the `bcts` / `pcal` arm with ε 0.05
      - `replay_out/bcts_20261106/`
      - the earlier "the calibrated p rises from 0.1 to 0.9 in 0.1–0.2 s" read that sum),
      - one contact (ds3_v1780, −2.67 m: the ego accelerated from 11.3 to 15.4 m/s beside the slower TV while its calibrated cut-in probability was ≤ 0.01, raw 0.05–0.13, and the non-reactive TV entered 0.2 m ahead; the paper root had held 11.3 m/s and the TV entered 11.6 m ahead)
    - mild δ_max 5.9 → 11.5 % and floor breaches 3 → 9.
  - Critical floor breaches read 9 → 8 only because the 5 removed ones are TVs that merged beside the ego and became ghosts (TV side entries critical / mild 4 / 16 → 13 / 20; recounted with side entries live once clear: critical 13 → 12, mild 7 → 11).
  - Every new breach has the ego 1–13 m further ahead at the TV's entry than in the paper root (no early braking).
  - The replay's default tail is unchanged until the maintainer has weighed the contact.
  - `occin0.05` (O5) on the final set (2026-11-06, maintainer's request, `replay_out/pcal_o5_20261106/result.md`, root `fin_pcal_occin0.05_20261106` + `lo_`, whose lead-only loop equals O10's on all 400 episodes):
    - no contact
    - against O10 critical / mild δ_max −3.0 / −2.4 %p (SE 0.5 / 0.4), floor breaches recounted 12 / 11 → 11 / 7, j_max 3.12 / 2.42 → 2.93 / 2.36 (runs at the jerk bound 50 → 48 critical, the same mechanism), at phantom_ep 0.032 → 0.070 m/s² (paper design 0.188)
    - ds3_v1780 entered 2.5 m ahead (O10 0.2 m) with a minimum gap of 1.95 m.
  - Decision pending between O10 and O5.
- **Since the evening of 2026-11-06 `bcts` replaces `pcal`** (§4: bias-corrected temperature scaling of STDAN's three probabilities, fitted to STDAN's own ±2.8 s label
  - maintainer's decision after the adversarial review of `pcal`'s map, `final_paper/appendix/calibration_review_r1.md`
  - method record `final_paper/appendix/calibration.md`)
  - `occin`'s ε is taken from the final set (maintainer: a controller threshold, not a predictor parameter; the paper discloses it), runs `replay_out/bcts_20261106/result.md` (ε 0.02 / 0.05 / 0.1):
    - at equal ε the closed loop equals `pcal`'s (ε 0.05: critical δ_max 12.8 vs 13.4 %, phantom_ep 0.062 vs 0.070; ε 0.1: the same ds3_v1780 contact)
    - ε 0.02 has the smallest cut-in cost (against the paper design critical / mild δ_max +1.0 / +0.9 %p, SE 1.2 / 0.9, floor breaches 9 / 3 → 6 / 3, recounted 13 / 7 → 10 / 6, no contact, phantom_ep 0.188 → 0.121, j_max +0.51 / +0.36)
    - the j_max rise is the calibrated labelled probability starting 1.1 s later plus `trajcutin`'s entering lane-keeping trajectory carrying q_LK (the step to p ≈ 1)
    - no arm passes the `pcal` tuning rule (ε 0.02 misses rule 2 by 0.04 %p)
    - ε not chosen (maintainer).
  - Not run in CARLA.
- **Since the night of 2026-11-06 the NGSIM design also carries `nolkentry`** (§4: an adjacent vehicle's lane-keeping and away-direction trajectories never lead
  - maintainer's decision: a lane-keeping hypothesis must not act as a cut-in, which the logic requires whatever the metrics, although on the final set it raised critical δ_max by +3.5 to +5.1 %p and left j_max unchanged, `replay_out/nolkentry_20261106/result.md`).
  - The paper root without calibration has not been run with it, and the replay's default tail is unchanged (ε is still open).

### 2026-11-10

- **Since 2026-11-10 the NGSIM design also carries `pingate`, `floorrisk0.05` and `floorhyp`, with `occin`'s ε fixed at 0.05** (§4: a hypothesis with p > ε has its cells at the steps where P_in(k) > ε, whatever p, and p sets the floor instead, gap ≥ L + d0 + Φ⁻¹(1 − ε/p)·σ, so that p·P_in(k)·P(gap < L + d0) ≤ ε per hypothesis and step
  - maintainer's decision after `replay_out/pingate_20261109/result.md`, against N5, the design with `_bcts_occin0.05_nolkentry`:
    - the TV's hypothesis on 0.33 s earlier (critical, SE 0.07),
    - critical δ_max +0.1 %p (SE 0.8), mild −1.1 %p (SE 0.5),
    - Δt_ant, T_rec and j_max unchanged,
    - no-cut-in phantom_ep 0.031 → 0.058 m/s² (SE 0.008),
    - critical floor breaches 10 → 13 (4 new, all pass → yield flips),
    - the ds3_v1780 contact stays (−3.50 → −2.11 m)).
  - Its string is the PG row of §4's current-design table.
  - Not run in CARLA.
- **Since the evening of 2026-11-10 the NGSIM design is NL**: PG with the cut-in ε at 10 % (`occin0.1`, `floorrisk0.1`), plus `nopricesigma` (every priced row asks for D on the mean gap; the floors keep σ in their own hard rows) and `lkonesided0.05` (the lead's lane-keeping row d_safe + 1.645σ, its own ε at 5 %)
  - (§4; maintainer's decision after `replay_out/nps_lk_20261110/result.md`, three changes run together and not decomposed, paired against PG:
    - critical / mild a_min +0.13 / +0.12 (SE 0.03), j_max −0.27 / −0.30, δ_max −0.6 / +0.6 %p (SE 0.8 / 0.3), T_rec +0.12 s, floor breaches 13 / 6 → 12 / 5
    - no-cut-in phantom_ep 0.058 → 0.029 (SE 0.009), runs a_min ≤ −2 7 → 5
    - the TV's hypothesis on 0.62 / 0.45 s later
    - the ds3_v1780 contact stays, −1.30 m, an entry the fail-safe could not avoid at the entry tick).
  - Its string is the third-to-last row of §4's current-design table, and it was the replay's default from that evening to the night of 2026-11-11 (§10).
  - Not run in CARLA.

### 2026-11-11

- **Since the night of 2026-11-11 the NGSIM design is NP with w 3**: NL without `entrypass0` and `pingate`, with `dopt3lkr` in place of `dopt100lkr`.
  - Maintainer's decision: NP after the start-margin re-measurement, and w 3 for ride comfort.
  - Measured on the final set at the logged set speed, with every controller starting at d_safe + 3 m (`NGSIM_V_SET=logged NGSIM_SPAWN_MARGIN=3`; `replay_out/startmargin_20261111/result.md`):
    - NP (w3) critical / mild δ_max 21.7 / 11.3 %, floor breaches 9 / 3, critical a_min −0.98, j_max 1.57, T_rec 1.12 s longer than NP (w100) with 17 runs not recovered
    - no-cut-in phantom_ep 0.108 (own lead-only), runs a_min ≤ −2 15, v_avg 6.35.
  - Against NP (w100): δ_max +10.1 / +5.6 %p, a_min +0.34, j_max −0.66.
  - Against ISA-MPC (slack 5) at the same condition: δ_max +4.0 / +1.5 %p (SE 1.8 / 1.0), a_min +1.10, phantom_ref −0.44.
  - `entrypass0` and `pingate` were removed after `replay_out/noep_nl_20261111/result.md`.
  - Its string is the second-to-last row of §4's current-design table, and it was the replay's default until later that night (§10)
    - the results above used the logged set speed and the 3 m start margin, which are the replay's default condition since the same night.
  - Not run in CARLA.
- **Later on the night of 2026-11-11 the NGSIM design dropped `safebranch`** (maintainer): the safety layer stays (`safelayer` + `entrant2`, with `failsafe_recovery`), but its fail-safe is no longer a branch of the scenario tree, so the MPC no longer plans the layer's contingency.
  - Its string is the last row of §4's current-design table, and the replay's default since then (§10)
    - its first runs, at today's limits and at the unified limits (§4 `amin<x>` to `fsjerk<x>`), are P0 and P1 of `replay_out/iso_limits_20261111/`.
  - Not run in CARLA.

### 2026-11-12

- **Since 2026-11-12 the NGSIM replay's defaults are the unified limits and live side entries** (maintainer):
  - D-opt's default string is P1's, the design above with `_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10` (the last row of §4's current-design table)
  - ISA-MPC and the PC-ACC port run at the same comfort limits
  - and a vehicle that changes lanes into the ego lane beside the ego stays live instead of becoming a ghost (`NGSIM_SIDE_ENTRY`, §10).
- First runs: `replay_out/side_live_20261112/`.
- Result on the final set (`replay_out/side_live_20261112/result.md`, against the 11-11 ISO roots):
  - Live side entries add only side contacts: D-opt / ISA-MPC 0.055 / PC-ACC as published / port +32 / +32 / +15 / +19, none rear-end. Of these, 28 / 29 / 13 / 19 have the bodies overlapping laterally, and 13 / 24 / 9 / 12 start at the vehicle's entry tick.
  - Critical δ_max rises by +10.9 / +7.5 / +2.5 / +3.7 %p and floor breaches by +12 / +12 / +5 / +7, with phantom braking unchanged.
  - D-opt still has the fewest floor breaches and the least phantom braking; its critical δ_max (32.1 %) is level with the port's (28.9 %).
- Not run in CARLA, whose agent is unchanged.

### D-opt definition and evidence

- Its policy string is `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_rmove1000`: Proposed plus three tokens (§4).
  - `dopt100`: on cut-in cells α_j scales only the footprint L + d0, and the optimizer chooses how much of the time gap τv to keep, at a price 100·P_j·shortfall².
  - `lkr`: the same price on the lane-keeping cells of a vehicle that has had a cut-in cell, until it leads and the gap has held the full standoff for 0.5 s. Every other in-lane lead keeps the hard standoff.
  - `trajcutin`: an adjacent vehicle's lane-keeping trajectory that enters the ego lane is a cut-in mode of its own.
  - Proposed (`cutin_chance0.6` alone) is kept unchanged so that the paper's sweeps reproduce. The paper and its tables (2026-09-10) predate D-opt.
  - Against Proposed on the same code (`cutin_postfix_cmp_20261016`, `cutout_postfix_cmp_20261016`, 66 runs each):
    - aggressive cut-in a_min_cmd −2.73 → −2.09 m/s² (+3.0 SE) and j_max_cmd_filt 4.07 → 3.14 (−2.4 SE)
    - normal cut-in a_min +1.4 SE, j_max −1.6 SE
    - no-cut-in and cut-out not distinguishable (|diff| ≤ 1.4 SE).
  - NGSIM ablation of `dopt100lkr` alone (2026-10-27, held-out 400 episodes, new replay defaults, λ 0 and 0.4, `replay_out/final_heldout_v20s0_ep0_nodopt_20261027/`; results for the paper in `paper/ablation.md`): without it,
    - braking and jerk rise in every group (critical cut-in j_max +0.8, a_avg +0.13; no-cut-in j_max +0.15–0.30; 3–8 SE),
    - mild-cut-in δ_max falls 1.8–2.0 %p and T_rec 0.08–0.12 s (about 2 SE),
    - critical δ_max is not distinguishable.
    - Without D-opt a cut-in cell demands α_j·d_safe, which below p = 0.6 is less than D-opt plans (α = 0.46 of d_safe at p = 0.3), so one critical cut-in (ds2_v2477, λ 0.4) kept accelerating, `entrypass0` dropped the real cut-in and the ego touched it (−0.93 m).
    - `entrypass0` relies on D-opt.
    - The ladder's bottom rung, the full d_safe on every cell (no `cutin_chance0.6`, `final_heldout_v20s0_ep0_nochance_20261027`), had the least intrusion (critical δ_max 6.2 / 8.5 % at λ 0 / 0.4 against 13.9 / 18.5 % for the adopted design, floor breaches 2 against 8–9, no contact: ds6_v2088's comes from the chance token's cut-out α floor) and the most phantom braking (0.44 / 0.41 against 0.27 / 0.18 m/s²).

## Every row of the current-design table (AGENTS.md §4)

The current design (§1), not in the paper:

| Label | Policy string |
|---|---|
| D-opt | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_rmove1000` |
| D-opt, 2026-10-27 | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_rmove1000_rjerk40` |
| D-opt, 2026-10-28 | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40` |
| D-opt, NGSIM lagged ego (2026-10-29 afternoon, superseded) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_alag0.5_ra0` |
| D-opt, NGSIM paper (2026-10-29 evening) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0` |
| D-opt, 2026-10-30 | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot` |
| D-opt, NGSIM paper (2026-10-30) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot` |
| D-opt, current (2026-10-31) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6` |
| D-opt, NGSIM paper (2026-10-31) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6` |
| D-opt, NGSIM paper (2026-11-02) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer` |
| D-opt, current (2026-11-03) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6_leadmerge` |
| D-opt, NGSIM paper (2026-11-03) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge` |
| D-opt, NGSIM paper (2026-11-03, with `safebranch`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch` |
| D-opt, current (2026-11-03 evening, with `mergefirst`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6_leadmerge_mergefirst` |
| D-opt, NGSIM paper (2026-11-03 evening, with `mergefirst`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst` |
| D-opt, current (2026-11-03 night, with `lkrsigma`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6_leadmerge_mergefirst_lkrsigma` |
| D-opt, NGSIM paper (2026-11-03 night, with `lkrsigma`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma` |
| D-opt, current (2026-11-03 night, with `lkhold`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_fullfoot_l1slack1e6_leadmerge_mergefirst_lkrsigma_lkhold` |
| D-opt, NGSIM paper (2026-11-03 night, with `lkhold` and `NGSIM_LANE_TEST=path`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_lkhold` |
| D-opt, NGSIM paper (2026-11-04, with `entrant2`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_lkhold_entrant2` |
| D-opt, NGSIM design (2026-11-10, PG) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.05_nolkentry_pingate_floorrisk0.05_floorhyp` |
| D-opt, NGSIM design (2026-11-10 evening, NL) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt100lkr_trajcutin_ctx<λ>_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_pingate_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05` |
| D-opt, NGSIM design (2026-11-11 night, NP w3) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx<λ>_latgate_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05` |
| D-opt, NGSIM design (2026-11-11 night, NP w3 without `safebranch`) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx<λ>_latgate_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05` |
| D-opt, NGSIM design (2026-11-12, unified limits; P1) | `acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx<λ>_latgate_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10` |

### Which row identifies which roots

- The first row identifies the older roots, such as the same-code roots below
- the second the runs of 2026-10-27/28 before the adoption of `nocutout` + `lksigma` (e.g. `full_v20s0_ep0_paper_20261028`)
- every new CARLA run uses the eighteenth and every new NGSIM run the twentieth with `NGSIM_LANE_TEST=path`
  - (the last row, NP with w 3 without `safebranch` at the unified limits, is the NGSIM design and the replay's default since 2026-11-12
  - the row before it, the same at the fork's limits, was both from later on the night of 2026-11-11
  - the row before that, NP with w 3, was both earlier that night
  - the row before that, NL, was the NGSIM design and the replay's default from the evening of 2026-11-10, its root ran it with `cutin_chance0.6`, which `chancecells` renames
  - the twentieth needs `NGSIM_POLICY_TAIL=_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_lkhold_entrant2`)
  - (since 2026-11-04, the adoption of `entrant2`
  - the nineteenth is the design before it, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_path_20261103`, used since the adoption of H on the night of 2026-11-03
  - the sixteenth and the seventeenth with `NGSIM_LANE_TEST=midpoint` are the design before that, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_lkrsigma_ghost_20261103`, used since the adoption of `lkrsigma` that night
  - the fourteenth and the fifteenth are the design before that, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst_ghost_20261103`, used since the adoption of `mergefirst` that evening
  - the eleventh and the thirteenth are the design before that, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_20261103`
  - the thirteenth since the adoption of `safebranch` that afternoon, the twelfth the design before that, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_20261103`
  - the eleventh and the twelfth since the adoption of `leadmerge` earlier that day
  - the eighth and the tenth are the design before that, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_safelayer_20261102`
  - the tenth since 2026-11-02, the paper's tables with `safelayer`
  - the ninth is the design before it, e.g. `fin_jinput_ra0_fullfoot_l1slack1e6_20261031`
  - the eighth and the ninth since 2026-10-31, the adoption of `l1slack1e6`
  - the sixth and the seventh are the design before that, e.g. `fin_jinput_ra0_fullfoot_20261029`, and the third and the fifth the design before `fullfoot`, e.g. `fin_jinput_ra0_clip_20261029`), on the ideal replay ego (§1, §10), with λ set through `ctx<λ>_latgate` as in the NGSIM replay
- the fourth, on the lagged ego (`NGSIM_EGO_LAG_TAU=0.5`), identifies the roots of 2026-10-29 afternoon (`replay_out/alag_eval_20261029/`).
