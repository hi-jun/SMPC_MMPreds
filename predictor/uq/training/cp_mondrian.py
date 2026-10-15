"""Mondrian (group-conditional) split-conformal calibration of STDAN's intent head.

Why: the marginal CP calibrated by cp_calibrate.py hits its coverage target on
average (0.912 at alpha=0.1) while covering only 0.516 of the maneuver phase of
the cut-in benchmark. That is not a bug - marginal CP is *allowed* to pay for a
hard subpopulation with an easy one. Mondrian CP fixes a taxonomy up front and
takes a separate quantile inside each category, which buys

    P(Y in C(X) | category(X, Y) = k) >= 1 - alpha   for every k

instead of one average. Two taxonomies are measured here, and they answer
different questions:

  by TRUE CLASS   - stops the dominant lane-keeping class from eating the
                    coverage budget of the two rare lane-change classes.
                    Caveat: this guarantees "when the truth is RLC the set
                    contains RLC", NOT "when I say RLC I am right" - the
                    planner needs the latter and Mondrian does not give it.

  by COVARIATE    - buckets on quantities observable at run time (how much the
                    target has drifted laterally, and how fast). This is the
                    one that can expose the pre-onset regime: if CP is honest
                    there, the "no lateral motion yet" bucket must demand
                    larger sets to hit its target. The ideal bucketing would be
                    time-to-onset, but that is the label - hence the proxies.

Run (hmg env, from this directory):
    python cp_mondrian.py --data /home/core-dev/HJ/dataset/raw_data_ngsim/dataset_t_v_t
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
# config_vel / loader2_hdf5_vel / model5f_mult_vel: this repository's
# predictor/stdan_vel copies (same model and loader as the source repository)
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "stdan_vel")))

from config_vel import args                        # noqa: E402
import loader2_hdf5_vel as lo                      # noqa: E402
from extract_features import STDAN_VEL_CKPT, load_trunk  # noqa: E402
from predictor.uq.stream_tap import encoder_inputs  # noqa: E402
from predictor.uq.cp_gate import lac_scores   # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
CLASS_NAMES = ("직진", "좌변경", "우변경")


def collect(gd, gen, loader, device, max_batches):
    """Softmax probs, labels and run-time-observable covariates over a split.

    Covariates come from `hist` (the target's own observed track) only - nothing
    that needs the future, so every one of them is available to the planner at
    decision time.
    """
    probs, labels, drift, vlat = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            hist = batch[0]                      # (T, B, 2) observed track
            enc_in, lat_enc = encoder_inputs(batch, device)
            values = gd(*enc_in)
            if isinstance(values, tuple):
                values = values[0]
            dummy_lat = torch.zeros(values.shape[0], args["lat_length"],
                                    device=device)
            dummy_lon = torch.zeros(values.shape[0], args["lon_length"],
                                    device=device)
            lat_pred = gen(values, dummy_lat, dummy_lon)[1]
            probs.append(lat_pred.detach().cpu().numpy())
            labels.append(lat_enc.argmax(-1).cpu().numpy())
            h = hist.cpu().numpy()               # (T, B, 2)
            drift.append(h[-1] - h[0])           # total displacement, both axes
            vlat.append((h[-1] - h[-6]) / 0.5)   # recent velocity, both axes
    return (np.concatenate(probs, 0), np.concatenate(labels, 0),
            np.concatenate(drift, 0), np.concatenate(vlat, 0))


def lateral_axis(drift, labels):
    """Which column of `hist` is lateral, decided from the data.

    The lateral axis is the one on which the two lane-change classes drift in
    OPPOSITE directions while lane keeping stays near zero; the longitudinal
    axis has the same sign for every class.
    """
    best, best_sep = 0, -np.inf
    for col in (0, 1):
        m = [drift[labels == c, col].mean() for c in range(3)]
        sep = abs(m[1] - m[2]) - abs(m[0])
        if sep > best_sep:
            best, best_sep = col, sep
    return best


def quantile_threshold(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score.

    A class is kept in the set iff its score is at or below this. Equivalent to
    the p-value form used at run time, but far cheaper to apply per group.
    """
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:                       # too few points for this alpha
        return np.inf
    return float(np.sort(scores)[k - 1])


def evaluate(p_te, y_te, group_te, thresholds, default_thr):
    """Coverage / set size per group under a per-group threshold."""
    s = lac_scores(p_te)                                  # (N, 3)
    thr = np.array([thresholds.get(int(g), default_thr) for g in group_te])
    keep = s <= thr[:, None]
    covered = keep[np.arange(len(y_te)), y_te]
    return covered, keep.sum(1)


def report_by(name, groups_te, covered, sizes, y_te, p_te, labels_map):
    rows = []
    for g in sorted(set(groups_te.tolist())):
        m = groups_te == g
        rows.append({
            "group": labels_map(g), "n": int(m.sum()),
            "coverage": float(covered[m].mean()),
            "set_size": float(sizes[m].mean()),
            "accuracy": float((p_te[m].argmax(-1) == y_te[m]).mean()),
        })
    print(f"\n  {name}")
    print(f"    {'그룹':<14}{'n':>7}{'커버리지':>10}{'집합크기':>10}{'정확도':>9}")
    for r in rows:
        print(f"    {r['group']:<14}{r['n']:>7}{r['coverage']:>10.3f}"
              f"{r['set_size']:>10.2f}{r['accuracy']:>9.3f}")
    return rows


def balanced_loader(mat, batches, seed=0):
    ds = lo.NgsimDataset(mat, t_f=50, d_s=1)
    lat = np.asarray(ds.D[:, 9], dtype=int) - 1
    w = (1.0 / np.maximum(np.bincount(lat, minlength=3), 1))[lat]
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=batches * 128,
                                    replacement=True, generator=g)
    return DataLoader(ds, batch_size=128, sampler=sampler, num_workers=2,
                      collate_fn=ds.collate_fn, drop_last=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default=STDAN_VEL_CKPT,
                    help="trunk checkpoint: pair dir or best_model.pt")
    ap.add_argument("--out-dir", default=CKPT_DIR)
    ap.add_argument("--cal-batches", type=int, default=200)
    ap.add_argument("--test-batches", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    cli = ap.parse_args()

    device = args["device"]
    torch.manual_seed(72)
    gd, gen = load_trunk(device, ckpt=cli.ckpt)

    p_cal, y_cal, d_cal, v_cal = collect(
        gd, gen, balanced_loader(os.path.join(cli.data, "ValSet.mat"),
                                 cli.cal_batches, seed=1), device, cli.cal_batches)
    p_te, y_te, d_te, v_te = collect(
        gd, gen, balanced_loader(os.path.join(cli.data, "TestSet.mat"),
                                 cli.test_batches, seed=2), device, cli.test_batches)

    col = lateral_axis(d_cal, y_cal)
    print(f"횡방향 축 = hist[:, :, {col}]  (클래스별 평균 이동: " +
          ", ".join(f"{CLASS_NAMES[c]} {d_cal[y_cal==c, col].mean():+.2f}"
                    for c in range(3)) + ")")
    print(f"보정 n={len(y_cal)}  검증 n={len(y_te)}  alpha={cli.alpha}")

    s_cal = np.array([lac_scores(p)[int(y)] for p, y in zip(p_cal, y_cal)])
    alpha = cli.alpha
    out = {"alpha": alpha, "lateral_col": int(col),
           "n_cal": int(len(y_cal)), "n_test": int(len(y_te))}

    # --- covariate buckets: how much lateral motion has ALREADY happened -----
    # bucket 0 is the pre-onset regime the planner needs and cannot resolve.
    EDGES = [0.15, 0.5]      # m/s of recent lateral speed
    def bucket(vl):
        a = np.abs(vl[:, col])
        return np.digitize(a, EDGES)
    b_cal, b_te = bucket(v_cal), bucket(v_te)
    BNAMES = {0: "횡움직임 없음", 1: "약한 횡움직임", 2: "뚜렷한 횡움직임"}

    # ---------------- 1. marginal CP, split apart afterwards ----------------
    thr_marg = quantile_threshold(s_cal, alpha)
    cov, sz = evaluate(p_te, y_te, np.zeros(len(y_te), int),
                       {0: thr_marg}, thr_marg)
    print(f"\n[1] 기존 방식 (전체 하나의 분위수, 임계 {thr_marg:.3f})")
    print(f"    전체        커버리지 {cov.mean():.3f}  집합크기 {sz.mean():.2f}")
    out["marginal"] = {"threshold": thr_marg, "coverage": float(cov.mean()),
                       "set_size": float(sz.mean())}
    out["marginal_by_class"] = report_by(
        "정답 클래스별로 쪼개보면", y_te, cov, sz, y_te, p_te,
        lambda g: CLASS_NAMES[g])
    out["marginal_by_bucket"] = report_by(
        "상황 버킷별로 쪼개보면", b_te, cov, sz, y_te, p_te, BNAMES.get)

    # ---------------- 2. Mondrian by true class ----------------------------
    thr_cls = {c: quantile_threshold(s_cal[y_cal == c], alpha) for c in range(3)}
    cov_c, sz_c = evaluate(p_te, y_te, y_te, thr_cls, thr_marg)
    print(f"\n[2] Mondrian - 정답 클래스별 분위수 "
          f"(임계 {', '.join(f'{CLASS_NAMES[c]} {thr_cls[c]:.3f}' for c in range(3))})")
    out["mondrian_class"] = report_by(
        "클래스별", y_te, cov_c, sz_c, y_te, p_te, lambda g: CLASS_NAMES[g])
    print(f"    {'전체':<14}{len(y_te):>7}{cov_c.mean():>10.3f}{sz_c.mean():>10.2f}")
    out["mondrian_class_overall"] = {"coverage": float(cov_c.mean()),
                                     "set_size": float(sz_c.mean())}

    # ---------------- 3. Mondrian by covariate bucket ----------------------
    thr_b = {b: quantile_threshold(s_cal[b_cal == b], alpha)
             for b in np.unique(b_cal)}
    cov_b, sz_b = evaluate(p_te, y_te, b_te, thr_b, thr_marg)
    print(f"\n[3] Mondrian - 상황 버킷별 분위수 "
          f"(임계 {', '.join(f'{BNAMES[b]} {thr_b[b]:.3f}' for b in sorted(thr_b))})")
    out["mondrian_bucket"] = report_by(
        "버킷별", b_te, cov_b, sz_b, y_te, p_te, BNAMES.get)
    print(f"    {'전체':<14}{len(y_te):>7}{cov_b.mean():>10.3f}{sz_b.mean():>10.2f}")
    out["mondrian_bucket_overall"] = {"coverage": float(cov_b.mean()),
                                      "set_size": float(sz_b.mean())}
    out["bucket_cal_counts"] = {int(b): int((b_cal == b).sum())
                                for b in np.unique(b_cal)}
    out["bucket_thresholds"] = {int(b): float(t) for b, t in thr_b.items()}

    path = os.path.join(cli.out_dir, "cp_mondrian_report.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("\nsaved", path)
