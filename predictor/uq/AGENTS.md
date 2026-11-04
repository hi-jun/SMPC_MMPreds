# predictor/uq — conformal novelty detection and deep ensemble for STDAN-vel

Audience: AI coding assistants and contributors who will use or extend the uncertainty-quantification (UQ) code in this directory.
State: 2026-10-13, the day it was ported. Where this file and the code disagree, the code is right.

---

## 1. What this directory is

- **Origin.** The code and artefacts come from the maintainer's thesis repository `/home/core-dev/HJ/graduate_project/beliefplanning`. That repository holds a risk-aware contingency planner (RACP, Frenet sampling planner in CommonRoad) and uses the same STDAN.
  - Source commits: `9c7fcb1` (UQ-in-risk integration, 2026-08-29) and `8f61949` (ensemble trained and accepted). The novelty pipeline dates from 2026-08-18, the intent CP gate from `665b5da` (2026-08-13).
- **Two UQ tools, both for the 9-mode STDAN-vel trunk:**
  1. **Conformal novelty detector.** Every forward pass gets a p-value for "has the encoder seen an input like this?". The score is Mahalanobis on encoder features; the p-value has Mondrian conformal calibration and a Bates calibration-conditional correction.
  2. **Deep ensemble.** Five STDAN-vel members, each trained with its own seed on a bootstrap resample (M = 5, the recipe of Shao et al., T-ITS 2025, "CUAP").
- **Also included: the intent split-CP gate library** (`cp_gate.py`). It ships without a calibration for this checkpoint (§7).
- **Nothing here is wired into the controller.** No policy token reads it, no adapter calls it, and nothing under `scripts/` imports it. Integration is open work (§7).

## 2. Which predictor these artefacts belong to — read before using anything

The detector and the members are tied to one set of trunk weights.

| Fact | Detail |
|---|---|
| **member0 = this repository's `stdan_vel` checkpoint** | `predictor/stdan_vel/checkpoints/0113_ratio211/best_model.pt`, md5 `d52c2339b0a7e60882c6d5b86d790d5b`. The `stdan_vel` policy token loads it through `STDANVelACCAdapter(..., mc_dropout=False)` in `scripts/carla/policies/acc_nair_smpc_agent.py`. It is byte-identical to the source repository's `stdan/ckpt/augmented/best_model.pt`, the primary model of every RACP UQ experiment |
| **The model code is the same** | `predictor/stdan_vel/model5f_mult_vel.py` and `loader2_hdf5_vel.py` are byte-identical to the source repository's training copies. The members load into this repository's classes (verified, §9) |
| **Output units** | Columns 0/1 of the decoder are velocity in ft/s and columns 2/3 are velocity precision 1/σ. The checkpoint was trained through `integrate_distribution`. `STDANVelACCAdapter` already reads it that way (integrates the mean, sums the variances) |
| **Not valid for `stdan_3int`** | The paper predictor (`predictor/stdan_3int_signed_tcross_velint/`) has a different head (3 intentions plus t_cross) and different training data. Its features mean nothing to this detector, and its outputs must not be mixed with these members |

**To give `stdan_3int` the same tools:**

- **Detector.**
  - Extract its encoder features and refit with `training/extract_features.py` + `training/fit_novelty.py` on its own Train and Val splits.
  - Its `GDEncoder` also has `lstm` (called for the history, then for the neighbours) and `first_glu`, so `StreamTap` attaches unchanged.
  - Its intent head is `Generator.op_intent`, not `op_lat`, so the logit hook in `StdanNovelty` and in `extract_features.PreLogitTap` must point there.
- **Ensemble.** Its trainer (`train/train_3int_signed_tcross.py`) has no `--seed` or `--bootstrap`. Port the pattern from `training/train5f_vel.py`:
  - re-seed `random`, `np.random` and `torch` from `--seed` before the models are built
  - replace the training rows with `rng.randint(0, N, N)` and save the indices
  - keep Val fixed
  - then train 5 members and run an acceptance check like `training/eval_ensemble.py`

## 3. File map

| Path | Role | Source (`beliefplanning/…`) |
|---|---|---|
| `conformal_ood.py` | Conformal p-values, Bates corrections (Simes, DKWM, asymptotic, Monte-Carlo), `MahalanobisNovelty`, `KnnNovelty`, `MondrianCalibration`, `load_novelty()`, `validity_report()`. Pure numpy | `predictors/conformal_ood.py` (only the default detector path, its comment and one docstring line changed) |
| `cp_gate.py` | Split-CP over the intent softmax: LAC/APS scores, `ConformalIntentGate` (credibility × confidence gate) | `predictors/cp_gate.py` (verbatim) |
| `stream_tap.py` | `StreamTap` (encoder feature hooks), `encoder_inputs`, and `StdanNovelty`, the runtime wrapper (§4.3) | `stdan/training/isap_head.py`, `train_isap_head.py`, and the inline novelty code of `stdan/stdan_predictor.py::_run_model` |
| `smoke_test_uq.py` | Loads the detector, member0 and the five members through `STDANVelACCAdapter`, and runs one synthetic scene | new |
| `ckpt/novelty_maha_concat_aug_g1.npz` | **The detector** (§4.2) | `stdan/ckpt/` |
| `ckpt/novelty_fit_report_aug_g1.json` | Its fit report (n_ref, n_cal, Bates b₁, simulated coverage, the numpy-vs-oodeel check) | `stdan/ckpt/` |
| `ckpt/ensemble/member{1..5}/best_model.pt` | **The ensemble members** (§5) | `stdan/ckpt/ensemble/` |
| `ckpt/ensemble/manifest.json`, `eval_report.json` | Training record and acceptance report. Paths inside `manifest.json` are relative to the source repository | `stdan/ckpt/ensemble/` |
| `training/extract_features.py` | Encoder features of an NGSIM split, sampled whole-track and thinned (§4.2) | `stdan/training/` |
| `training/fit_novelty.py` | Fits Mahalanobis and kNN detectors with oodeel, then calibrates them and saves `.npz` files | `stdan/training/` |
| `training/eval_novelty.py` | Validity, shift, Bates-fluctuation, control and runtime analyses (needs feature files, §8) | `stdan/training/` |
| `training/cp_calibrate.py`, `cp_mondrian.py` | Calibration and coverage checks for the intent CP gate (§7). Fixed here on 2026-11-04, still wrong in the source: the Mondrian-by-true-class report tested every candidate class against the true class's threshold (right coverage, wrong set size; synthetic check 1.605 against 2.083); it now uses `evaluate_class_conditional` | `stdan/training/` |
| `training/train5f_vel.py` | STDAN-vel trainer with `--seed --bootstrap --out --data --val-data --epochs --patience` | `stdan/training/` |
| `training/eval_ensemble.py` | Ensemble acceptance check | `stdan/training/` |
| `training/evaluate5f_vel.py`, `early_stopping.py`, `config.py`, `highd_raw.py` | Dependencies of the scripts above (`config.py` is the `config` alias the loader imports) | `stdan/training/` |

`training/` takes `config_vel`, `loader2_hdf5_vel` and `model5f_mult_vel` from `predictor/stdan_vel/` (appended to `sys.path`). `evaluate5f_vel`, `early_stopping` and `config` come from the copies next to the scripts. The copies of `train5f_vel.py` and `evaluate5f_vel.py` in `predictor/stdan_vel/` do not import in this repository: they need a `config` module and `loader2_highD_1107_final`, which it does not have. They were left untouched.

## 4. Conformal novelty detector

### 4.1 What it computes

- **Feature.** z = [agent | social], 128-d.
  - agent: the last-step output of `GDEncoder.lstm` on the target's history (the first of the two LSTM calls per forward).
  - social: the last-step output of `GDEncoder.first_glu`, the spatial-attention GLU before the residual.
- **Score.** −max_c log N(z; μ_c, Σ): Mahalanobis with per-class means over the three lateral classes and one tied covariance (Lee et al. 2018, as implemented by oodeel). A higher score means more novel.
- **p-value (Mondrian).** The calibration scores are split by the predictor's own argmax intent class g (0 LK, 1 LLC, 2 RLC), and p = (1 + #{cal_g ≥ s}) / (n_g + 1). **A small p means unfamiliar; p ≤ α is an alarm.**
  - Under exchangeability, P(p ≤ α | g) ≤ α holds for every group.
  - `p_cc` is the calibration-conditional version: Simes within each group, δ = 0.05.
- **Why Mondrian.** About 94 % of NGSIM is lane keeping, so marginal calibration reports rare as novel. In the source repository, 71.9 % of the samples the predictor called a right lane change raised an alarm at α = 0.05 under marginal calibration.

### 4.2 How the shipped detector was built

Run in the `hmg` env from `training/`. The calibration command was confirmed by re-extracting it in this repository: the result was bit-identical (§9).

```bash
python extract_features.py --data /home/core-dev/HJ/dataset/NGsim/0113_ratio211 --split Train --gap 3 \
    --out ../ckpt/features/ref_train_aug.npz             # reference (mu, Sigma): 19,508 rows
python extract_features.py --split Val --gap 1 --exclude-overlap-with Train Test \
    --out ../ckpt/features/cal_val_aug_g1.npz            # calibration: 76,602 rows, 932 tracks
python fit_novelty.py --ref ../ckpt/features/ref_train_aug.npz \
    --cal ../ckpt/features/cal_val_aug_g1.npz --tag aug_g1   # -> ../ckpt/novelty_*_aug_g1.npz
```

- **Calibration data.** The calibration split is the real, unbalanced NGSIM Val (`raw_data_ngsim/dataset_t_v_t`), thinned to one row per second per track. The 12 tracks shared with Train or Test are dropped.
  - Mondrian n = 71,568 / 3,357 / 1,677. The marginal correction uses the Monte-Carlo sequence (δ = 0.05).
- **Reference data.** The reference split is the class-balanced, mirrored 0113_ratio211 Train the trunk was trained on.
- **Check on the ID TestSet** (51,384 rows, 3 s thinning, Val-overlapping tracks dropped; recomputed at port time). The alarm rate at α = 0.05 is 0.041 overall and 0.041 / 0.044 / 0.037 per group (n 47,835 / 2,316 / 1,233).
- `fit_novelty.py` also writes kNN and output-score detectors (`novelty_knn_*`, `novelty_{mls,msp,energy}*`). Only `maha_concat` was selected and shipped. The selection evidence is in the source repository's `REPORT_conformal_novelty.md` §6, §10–11 (in Korean).

### 4.3 Using it at run time

```python
from predictor.stdan_vel.acc_adapter import STDANVelACCAdapter
from predictor.uq.stream_tap import StdanNovelty

adapter = STDANVelACCAdapter(mc_dropout=False)                 # member0
novelty = StdanNovelty(adapter.gdEncoder, adapter.generator)   # default detector
pred = adapter.predict_raw([tid], trackings)[tid]
r = novelty.result()          # arrays over the batch of the LAST forward pass
alarm = r["p"][0] <= 0.05     # also r["p_cc"], r["score"], r["group"]
```

- **The hooks keep only the last forward.** `predict_raw` loops over targets, so call it once per target and read `result()` after each call.
- **The model must run in eval mode** (`mc_dropout=False`): the calibration features were extracted with dropout off. The `stdan_vel` token already does this.
- **Cost.** One forward per target, as before, plus a 128-d Mahalanobis evaluation against three class means and a binary search in the calibration array.

### 4.4 Pitfalls measured in the source repository

1. **Padded history is out of distribution.**
   - In RACP, the first 3.3 s of every scenario used a history back-extrapolated at constant velocity. 43–83 % of those steps raised an alarm at α = 0.05, against 4–7 % once 33 real steps existed. RACP therefore trusted an alarm only with a full real history: u = 𝟙[p ≤ α ∧ n_real_hist ≥ 33].
   - **This repository pads differently.** `_history_to_ngsim` (in the `stdan_3int` adapter, inherited by `STDANVelACCAdapter`) repeats the first position, so the vehicle looks parked and then moving. This has not been measured here; expect at least as many alarms until 33 real samples exist.
2. **The input pipeline must match the loader.**
   - The detector was calibrated on `loader2_hdf5_vel` tensors. The CARLA adapter differs from the training grid: its edges are at ±29.9 m, while training kept |dy| < 27.43 m (root `AGENTS.md` §10). The padding above differs too.
   - Such a mismatch moves the score, and the alarms then flag the adapter rather than the scene. Measure the alarm rate on plain in-distribution CARLA traffic before trusting an alarm.
3. **What it flags.**
   - In RACP it caught 32 m/s traffic: a 70.5 % step alarm rate on that benchmark against 2.1–9.9 % on the three normal-speed sets (re-computed at port time with the source's `evaluation/uq_efficiency.py`). The source analysis also traced alarms to "approach without braking", i.e. closing at ≥ ~6 m/s inside 28 m, which is the p99 tail of NGSIM.
   - Root `AGENTS.md` §8.7 already lists ego 20 m/s as outside NGSIM.
   - It is a **competence-boundary detector**: it says when the input is unlike the training data. It does not say when a prediction or a belief update is wrong for other reasons.
4. **Statistics.**
   - With one fixed calibration set, the realised false-alarm rate fluctuates 1.6–2.3× more than the iid Beta(l, n+1−l) prediction. The cause is dependence between vehicles at the same time and place; n_eff/n ≈ 0.24–0.5.
   - Split by track, never by frame.
   - Test exchangeability with a track-permutation two-sample KS on scores. Do not KS-test the discrete p-values against Uniform(0, 1).

## 5. Deep ensemble

### 5.1 Recipe and members

- **Training.** `training/train5f_vel.py --seed S --bootstrap --epochs 60 --patience 15 --out <dir>`, run in the `hmg` env with `WANDB_MODE=offline`.
  - Data: 0113_ratio211 Train/Val (lateral classes LK:LLC:RLC undersampled to 2:1:1, mirrored). Each member's Train rows are a bootstrap resample; Val is fixed.
  - Trained 2026-08-29 in 3.5 h, two runs at a time on an RTX 3090.

| Member | Seed | Best val loss | Last epoch | md5 |
|---|---|---|---|---|
| member1 | 10 | 88.95 | 50 | `375602a1…` |
| member2 | 20 | 92.19 | 36 | `c0f40943…` |
| member3 | 30 | 92.71 | 29 | `be15c5eb…` |
| member4 | 60 | 90.18 | 38 | `2578c486…` |
| member5 | 90 | 90.24 | 32 | `add3aeb5…` |

- **member0 is not a bootstrap member.** It is the deployed `stdan_vel` checkpoint (seed 72, full Train). In RACP it drove the intent belief, the targets and the geometry in every arm, and the **risk ensemble was members 1–5 only**.

### 5.2 Acceptance (`training/eval_ensemble.py`)

The check runs on the first 2,048 rows of 0113_ratio211 Val. Each model gives single-mode output (`train_flag = True`, the model's own argmax manoeuvre), the velocity is integrated, and the result is compared with `fut_pos`.

- **Masked MSE [ft²]:** member0 49.34; members 53.82 / 51.33 / 51.64 / 49.80 / 53.21. All are ≤ 1.10× member0.
- **Pairwise ADE between models:** 0.57–0.66 m (mean 0.623 m).
- All md5s are distinct. Result: `accept: true`.

### 5.3 Using the members

```python
from predictor.stdan_vel.acc_adapter import STDANVelACCAdapter

members = [STDANVelACCAdapter(ckpt_path="predictor/uq/ckpt/ensemble/member%d/best_model.pt" % i,
                              mc_dropout=False) for i in range(1, 6)]
preds = [m.predict_raw([tid], trackings)[tid] for m in members]
```

- **Modes per member.** Each prediction has three lateral modes (LK, LLC, RLC) of the "Constant" longitudinal mode, because `STDANVelACCAdapter` keeps `longitudinal_mode_idx = 0`. All nine modes remain in `raw_pred_vel_distribution_9mode`. RACP used all nine per member.
- **Cost.** Every member is one more forward pass per target and call.

### 5.4 How RACP combined them (design reference, not code here)

- **Risk view.** The members' modes were stacked member-major (9 × 5 rows) into a separate "risk view". Only the risk computation read it; the belief, the targets and the plots kept reading member0.
- **Aggregation.** Per member, risk = the belief-weighted mode risk. Per obstacle, risk = the **maximum over members** (`risk_assessment/risk_costs.py::_blended_member_risk`).
- **Disagreement diagnostics.** Two quantities were logged: the mean pairwise distance between the members' probability-weighted end points, and the mean pairwise Jensen–Shannon divergence of their intent distributions (`stdan/racp_adapter.py::_ens_disagreement`).
- **The CP arm used the same slot.** An alarmed obstacle (u = 1) got its risk aggregated by the worst mode (maxR) instead of the belief-weighted sum (wR), and its risk view added:
  - a constant-velocity row with weight 0
  - covariance inflation σ = 3·h m at lead time h
  - a widened collision-probability cull margin (6 m)

  (`RISK_NOVELTY` in `stdan/racp_adapter.py`; `build_risk_view` and `blended_mode_risk` in `risk_assessment/collision_probability.py`.)

## 6. What happened when RACP used them

Benchmark S1: out-of-distribution cut-in at 32 m/s with a 5–8 m gap, 240 scenarios. The counts below are collisions out of 240, re-counted from the source repository's result files at port time.

| How risk entered the planner | base (member0) | CP-gated risk | ensemble (members 1–5) |
|---|---|---|---|
| Only as a hard feasibility gate (δ = 0.05, `risk_cost = 0`), the original planner | **28** | 60 | 42 |
| Gate plus cost term `risk_cost = 0.1` | 43 | — | 42 |
| Gate plus cost term `risk_cost = 1.0` (`weights_riskonly.json`, nothing else changed) | 56 | 52 | **36** |

- **Paired against base:**
  - ensemble under the hard gate: 0 fixed, 14 broken
  - ensemble with the cost term at 1.0: 20 fixed, 0 broken
  - CP under the hard gate: 21 fixed, 53 broken
- **In-distribution sets.** Every arm had 0 collisions on the normal-speed causal cut-in set (S2), and both UQ arms reproduced the base trajectories exactly there (240/240 byte-identical).
- **Ensemble disagreement.** Mean pairwise end-point distance / JSD were 3.14 m / 0.142 on S1, against 1.81–2.43 m / 0.036–0.052 on the three normal-speed sets.
- **Reading.**
  - The epistemic signal was usable when the planner priced risk as a cost, and harmful when risk could only remove candidates.
  - Worst-case aggregation under a hard constraint removes evasive manoeuvres first, because evasion enters uncertain regions; what remains is braking in lane.
  - **No configuration beat the untouched gate baseline (28).** Adding the cost term hurt the baseline faster than the ensemble recovered it.
- **Relevance here.** This repository's safety constraint is soft, but its slack costs 5000·slack² (root `AGENTS.md` §3), which puts it near the hard-gate end of that spectrum. Measure before assuming either direction.

## 7. Open items

- **Not integrated.** There is no policy token, no `policy_log` field for p or for member spread, and no ensemble input to `MultimodalLeadPrediction` or the chance constraint.
- **Padding and grid mismatch** (§4.4, items 1–2) are unmeasured in this repository. Do this first if alarms will drive anything.
- **The intent CP gate is uncalibrated for this checkpoint.**
  - The source calibration (`cp_calibration.npz`) belongs to an older, position-output trunk and was not copied.
  - `training/cp_calibrate.py` uses `balanced_loader` (class-balanced sampling with replacement). Its calibration set is therefore not exchangeable with deployment inputs, a weakness the novelty pipeline's own sampler (`extract_features.py`) was written to avoid. Treat its coverage figures as indicative.
- **`stdan_3int` port** (§2).

## 8. Environments and how to run

| Env | Runs | Notes |
|---|---|---|
| `mmpreds-c0913` (py3.7, torch 1.13 CPU) | `conformal_ood.py`, `cp_gate.py`, `stream_tap.py`, `smoke_test_uq.py` | Verified with numpy 1.21.6 (from `~/.local`, the default) and with numpy 1.19.5 (`PYTHONNOUSERSITE=1`). The results were identical |
| `hmg` (py3.9, torch 2.7 CUDA) | everything in `training/` | Needs h5py and, for the trainer, wandb, pandas, seaborn and scikit-learn. `fit_novelty.py` and `eval_novelty.py` put `/home/core-dev/HJ/graduate_project/oodeel` on `sys.path`; it is not pip-installed. kNN needs faiss-cpu |

```bash
python predictor/uq/smoke_test_uq.py          # from the repository root, mmpreds-c0913

cd predictor/uq/training                      # hmg; the relative defaults assume this directory
python eval_ensemble.py                       # rewrites ../ckpt/ensemble/eval_report.json
WANDB_MODE=offline python train5f_vel.py --seed <S> --bootstrap --epochs 60 --patience 15 --out <dir>
python cp_calibrate.py --data /home/core-dev/HJ/dataset/raw_data_ngsim/dataset_t_v_t --out-dir <dir>
```

- **Dataset paths are absolute.** They point to `/home/core-dev/HJ/dataset/…` on the maintainer's machine.
- **Always pass `--out` to the trainer.** Without it, the output goes to `predictor/stdan_vel/config_vel.py`'s `args['path']`, relative to the working directory.
- **Importing `evaluate5f_vel` writes an empty `A.xlsx`** into the working directory (`pd.ExcelWriter` at import). This is old behaviour, kept as is.
- **`eval_novelty.py` needs feature files** in `../ckpt/features/`, which are not shipped (1.1 GB in the source repository, `stdan/ckpt/features/`). Note that the source's aug files carry different suffixes (`cal_val_aug_g1`, `test_id_aug`), so `--feat-suffix` cannot pick both; rename or link first. Figures go to `results/uq_novelty/`.
- **Markdown is gitignored** in this repository: add this file with `git add -f`.

## 9. Verification done at port time (2026-10-13)

| Check | Result |
|---|---|
| md5 of the five members and of the detector against `manifest.json` and the source | identical |
| `StdanNovelty` on member0 in `mmpreds-c0913`, 1,024 calibration rows, against the features the source stored (GPU, torch 2.7) | features max abs diff 1.7e-6; Mondrian groups identical; p differs by at most one calibration rank (1/1678 in the RLC group) |
| Members 0–5 loaded through `STDANVelACCAdapter` in `mmpreds-c0913`, `eval_ensemble.py` protocol | MSE within 1.5e-5 (relative) of `eval_report.json` |
| `training/eval_ensemble.py` run from this repository (`hmg`) | reproduces `eval_report.json` exactly |
| `training/extract_features.py`, calibration command of §4.2 | 76,602 rows; row indices and all feature arrays bit-identical to the source's `cal_val_aug_g1.npz` |
| `training/extract_features.py`, reference command of §4.2 | 19,508 rows; row indices and all feature arrays bit-identical to the source's `ref_train_aug.npz` |
| `training/fit_novelty.py` on the source's stored features (`--features concat`) | every array of the refitted `novelty_maha_concat` (μ, precision, calibration scores, Bates sequence, Mondrian arrays) identical to the shipped file |
| `training/train5f_vel.py --seed 10 --bootstrap --epochs 1` | bootstrap indices identical to member1's `boot_idx.npz` (291,963 rows); the epoch trained and saved |
| `training/cp_calibrate.py`, `cp_mondrian.py` (5 batches each), `eval_novelty.py --help` | ran |
| `smoke_test_uq.py` | passes in 1.8 s |

## 10. Where to dig deeper (source repository, local paths)

All paths are under `/home/core-dev/HJ/graduate_project/beliefplanning/`:

- `REPORT_conformal_novelty.md`: method, validity, the Mondrian rationale, highD external validation, runtime findings (in Korean).
- `REPORT_prediction_units.md`: why the older position-output trunk is read differently (it does not apply to this checkpoint).
- `stdan/racp_adapter.py`: `RISK_NOVELTY`, `STDAN_ENSEMBLE`, the risk view, `_ens_disagreement`.
- `risk_assessment/collision_probability.py`, `risk_assessment/risk_costs.py`: how the risk view is consumed.
- `run_uq_matrix.sh`, `evaluation/uq_metrics.py`, `uq_mcnemar.py`, `uq_efficiency.py`: the experiment runner and the paired analyses. Results are in `planner/Frenet/results/eval/*_uq{A,B,C}_*`, `*_ro{A,B,C}_oodmarg`, `*_r01{A,C}_oodmarg`.
- `stdan/ckpt/features/`: extracted features. `stdan/ckpt/ensemble/member*/`: per-epoch checkpoints and `boot_idx.npz`.

## 11. Can the ensemble itself flag OOD? (SODA-MPC test, 2026-10-18)

**Full reference:** `/home/core-dev/HJ/graduate_project/beliefplanning/planner/Frenet/results/ensemble_ood/AGENTS.md`. It holds the code, data, every table and the pitfalls. Read it before building an ensemble-based OOD switch here.

**What was tested.** SODA-MPC (Contreras et al., L4DC 2025, arXiv:2406.02436):
- score = largest eigenvalue of the covariance of the members' predictions
- a split-conformal threshold calibrated with one point per NGSIM track
- applied to members 1–5 of `ckpt/ensemble/`, the same files as here
- member0, the reference model, is this repository's `stdan_vel` checkpoint

**Results** (δ = 0.05; cut-in vehicle in the RACP scenarios):

| | ensemble covariance ρ (0.1 s) | ensemble intent MI | Mahalanobis (`maha_concat`, §4) |
|---|---|---|---|
| held-out NGSIM false alarms | 0.047 | 0.051 | 0.050 |
| 32 m/s scenarios | 1.00 | 0.53–0.87 | 0.92–1.00 |
| normal-speed scenarios (9–14 m/s) | **0.22–0.30** (0.31–0.51 at 1–3 s) | 0.11–0.22 | **0.04–0.18** |
| AUROC, 32 m/s vs normal-speed | 0.971 | 0.852 | 0.974 |
| NGSIM input scaled ×0.2 (slower) | 0.04 | 0.68 | 0.73 |

**What ρ measures in this model.**
- **Prediction scale, not disagreement.** Relative to the predicted travel distance, the members agree *more* on fast OOD inputs than on NGSIM (spread/travel 0.020 vs 0.033). Normalised that way, highD detection is 0 %.
- **Rare situations.** ρ alarms 32–40 % in the rarest 1 % of NGSIM situations and 1.5–2 % in the most common half. That includes 64 % of real NGSIM unbraked approaches (closing ≥ 6 m/s within 28 m), which are training data.
- **Intent disagreement.** The members disagree about the intent (MI rises 17–20× on every OOD set). They do not disagree about the trajectory shape.

**Consequences for this repository.**
- **Too many alarms for a switch.** A SODA-MPC-style switch on ρ would put a large share of ordinary cut-in approaches into the fallback mode, since the cut-in scenarios are built around the approach.
- **The quieter detector.** Mahalanobis separates OOD equally well with far fewer alarms in normal traffic.
- **MI is a different signal.** It reads as "the outcome is contested", not "the input is new".
- **The trajectory-level rule saturates.** "Any step above the threshold" flagged 100 % of cut-in trajectories, including normal-speed ones.
- **None of the scores marks high-error predictions well.** AUROC for the top-10 % error is 0.60–0.69 on NGSIM.

**Grid-cell duplicates (applies here too).**
- **The bug in RACP.** RACP's `make_tensor` sometimes placed two neighbours in one social-grid cell (38 of 50,883 steps). With more neighbours than mask cells, `masked_scatter_` fills the social tensor misaligned across time, so those predictions use scrambled neighbour features.
- **Same pattern here.** `STDAN3IntACCAdapter.make_tensor` in `predictor/stdan_3int_signed_tcross_velint/acc_adapter.py`, which `STDANVelACCAdapter` inherits, appends every neighbour inside the grid without checking for an occupied cell. It has not been measured in the CARLA scenes.
