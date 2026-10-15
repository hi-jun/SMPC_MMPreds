"""Fit novelty scores on the reference features and calibrate conformal p-values.

Follows Novello et al. (ICML'24): the OOD score is one of oodeel's detectors,
fitted on in-distribution REFERENCE features (TrainSet, track-thinned) and
scored on the CALIBRATION set (ValSet, other tracks) to obtain the calibration
scores that turn a new score into a p-value. Two feature-based scores and three
output-based baselines:

    mahalanobis  Lee et al. 2018 (per-class mean, tied covariance)   [feature]
    knn          Sun et al. 2022 (L2-normalised, k-th neighbour)      [feature]
    mls / msp / energy    from the intent-head logits                 [output]

oodeel wants a model, not arrays; per its own feature_extractor tutorial we
hand it an nn.Sequential(Identity) surrogate so that "layer 'feat'" IS the
precomputed feature. `eps=0.0` disables oodeel's input-gradient perturbation,
which cannot run on precomputed features anyway.

The stored artefact is plain arrays (mu, precision / bank, calibration scores,
Bates sequence) so the planner env - older numpy/sklearn, no faiss - can load
it with np.load. Before saving, the numpy re-implementation in
predictor/uq/conformal_ood.py is checked against oodeel's own scores on the same
batch (|diff| < 1e-4), so what runs online is provably what was calibrated.

Run (hmg env, from this directory):
    python fit_novelty.py --ref ../ckpt/features/ref_train.npz --cal ../ckpt/features/cal_val.npz
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/core-dev/HJ/graduate_project/oodeel")
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))

from oodeel.methods import Mahalanobis, DKNN            # noqa: E402
from oodeel.datasets import load_data_handler           # noqa: E402
from predictor.uq import conformal_ood as C  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")
FEATURES = ("agent", "social", "concat", "values_last", "pre_logit")


def feature_matrix(npz, name):
    if name == "concat":
        return np.concatenate([npz["agent"], npz["social"]], axis=1).astype(np.float32)
    return np.asarray(npz[name], dtype=np.float32)


class _Passthrough(nn.Module):
    """Identity with one dummy parameter: oodeel's TorchOperator locates the
    device via next(model.parameters()), which a bare nn.Identity cannot give."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return x


def surrogate_and_loader(X, y, batch=512):
    """oodeel Route A: passthrough model + array dataset -> the feature IS layer 'feat'."""
    model = nn.Sequential(OrderedDict([("feat", _Passthrough()), ("head", _Passthrough())]))
    handler = load_data_handler("torch")
    ds = handler.load_dataset((X.astype(np.float32), y.astype(np.int64)))
    ds = handler.prepare(ds, batch_size=batch)
    return model, ds


def fit_mahalanobis(Xr, yr):
    det = Mahalanobis(eps=0.0)
    model, ds = surrogate_and_loader(Xr, yr)
    det.fit(model, feature_layers_id=["feat"], fit_dataset=ds)
    mus, pinv = det._layer_stats[0]
    classes = list(det._classes)
    mu = np.stack([det.op.convert_to_numpy(mus[c]) for c in classes], 0)
    prec = det.op.convert_to_numpy(pinv)
    return det, mu, prec, classes


def fit_knn(Xr, yr, k):
    det = DKNN(nearest=k, use_gpu=False)
    model, ds = surrogate_and_loader(Xr, yr)
    det.fit(model, feature_layers_id=["feat"], fit_dataset=ds)
    return det


def oodeel_scores(det, X, y=None):
    """Score precomputed features through oodeel's own pipeline."""
    handler = load_data_handler("torch")
    y = np.zeros(len(X), dtype=np.int64) if y is None else y
    ds = handler.prepare(handler.load_dataset((X.astype(np.float32), y)), batch_size=1024)
    scores, _ = det.score(ds)
    return np.asarray(scores, dtype=float)


def output_scores(logits, kind):
    z = np.asarray(logits, dtype=float)
    if kind == "mls":
        return -z.max(1)
    if kind == "msp":
        p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        return -p.max(1)
    if kind == "energy":
        m = z.max(1, keepdims=True)
        return -(m[:, 0] + np.log(np.exp(z - m).sum(1)))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(CKPT_DIR, "features", "ref_train.npz"))
    ap.add_argument("--cal", default=os.path.join(CKPT_DIR, "features", "cal_val.npz"))
    ap.add_argument("--features", nargs="*", default=list(FEATURES))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--method", default="monte_carlo", choices=list(C.SEQUENCES))
    ap.add_argument("--out-dir", default=CKPT_DIR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-mondrian", action="store_true")
    cli = ap.parse_args()

    ref = np.load(cli.ref); cal = np.load(cli.cal)
    yr = np.asarray(ref["labels"], dtype=np.int64)
    n_cal = len(cal["labels"])
    # Mondrian group = the predictor's own argmax class. It is observable at run
    # time (unlike the true label) and it is exactly what drives the imbalance:
    # 94% of the calibration set is lane keeping, so a marginal calibration
    # reports "rare" as "novel".
    g_cal = np.asarray(cal["logits"]).argmax(1)
    print(f"ref {len(yr)} (classes {np.bincount(yr, minlength=3)}), cal {n_cal} "
          f"(predicted-class groups {np.bincount(g_cal, minlength=3)})")

    # one Bates sequence per n_cal (expensive part of the MC method); reused
    b_seq = C.SEQUENCES[cli.method](n_cal, cli.delta) if cli.method != "monte_carlo" \
        else C.monte_carlo_sequence(n_cal, cli.delta, n_sim=20000)
    cov = C.coverage_probability(b_seq, n_sim=20000, rng=np.random.RandomState(11))
    print(f"Bates {cli.method} sequence: n={n_cal} delta={cli.delta} "
          f"b1={b_seq[0]:.5f} independent coverage {cov:.4f}")

    report = {"n_ref": int(len(yr)), "n_cal": int(n_cal), "delta": cli.delta,
              "method": cli.method, "b1": float(b_seq[0]), "seq_coverage": float(cov),
              "scores": {}}
    tag = f"_{cli.tag}" if cli.tag else ""

    for feat in cli.features:
        Xr, Xc = feature_matrix(ref, feat), feature_matrix(cal, feat)

        # ---- Mahalanobis
        det, mu, prec, classes = fit_mahalanobis(Xr, yr)
        s_ood = oodeel_scores(det, Xc)
        mond = None if cli.no_mondrian else C.MondrianCalibration(
            {int(g): s_ood[g_cal == g] for g in np.unique(g_cal)},
            delta=cli.delta, method="simes")
        ours = C.MahalanobisNovelty(mu, prec, cal_scores=s_ood, delta=cli.delta,
                                    method=cli.method, b_seq=b_seq,
                                    feature=feat, classes=classes, mondrian=mond)
        s_ours = ours.score(Xc)
        # oodeel scores in float32, we in float64: compare relatively
        diff = float(np.max(np.abs(s_ours - s_ood) / np.maximum(1.0, np.abs(s_ood))))
        assert diff < 1e-4, f"mahalanobis {feat}: numpy vs oodeel differ (rel) by {diff}"
        ours.save(os.path.join(cli.out_dir, f"novelty_maha_{feat}{tag}.npz"))
        report["scores"][f"maha_{feat}"] = {
            "cal_score_median": float(np.median(s_ood)), "cal_score_p90": float(np.percentile(s_ood, 90)),
            "max_abs_diff_vs_oodeel": diff, "d": int(Xr.shape[1])}
        print(f"  maha  {feat:<12} d={Xr.shape[1]:>3}  cal score med {np.median(s_ood):8.2f}  |numpy-oodeel| {diff:.2e}")

        # ---- kNN
        det_k = fit_knn(Xr, yr, cli.k)
        s_k = oodeel_scores(det_k, Xc)
        mond_k = None if cli.no_mondrian else C.MondrianCalibration(
            {int(g): s_k[g_cal == g] for g in np.unique(g_cal)},
            delta=cli.delta, method="simes")
        ours_k = C.KnnNovelty(Xr, cli.k, cal_scores=s_k, delta=cli.delta,
                              method=cli.method, b_seq=b_seq, feature=feat,
                              mondrian=mond_k)
        s_ko = ours_k.score(Xc)
        diff_k = float(np.max(np.abs(s_ko - s_k) / np.maximum(1e-3, np.abs(s_k))))
        assert diff_k < 1e-3, f"knn {feat}: numpy vs oodeel differ (rel) by {diff_k}"
        ours_k.save(os.path.join(cli.out_dir, f"novelty_knn_{feat}{tag}.npz"))
        report["scores"][f"knn_{feat}"] = {
            "cal_score_median": float(np.median(s_k)), "cal_score_p90": float(np.percentile(s_k, 90)),
            "max_abs_diff_vs_oodeel": diff_k, "k": cli.k}
        print(f"  knn   {feat:<12} d={Xr.shape[1]:>3}  cal score med {np.median(s_k):8.4f}  |numpy-oodeel| {diff_k:.2e}")

    # ---- output-based baselines: calibration scores only (no fit needed)
    for kind in ("mls", "msp", "energy"):
        s = output_scores(cal["logits"], kind)
        extra = ({} if cli.no_mondrian else C.MondrianCalibration(
            {int(g): s[g_cal == g] for g in np.unique(g_cal)},
            delta=cli.delta, method="simes").to_dict())
        np.savez(os.path.join(cli.out_dir, f"novelty_{kind}{tag}.npz"), kind=kind,
                 cal_scores=np.sort(s), delta=cli.delta, method=cli.method,
                 b_seq=b_seq, **extra)
        report["scores"][kind] = {"cal_score_median": float(np.median(s))}
        print(f"  {kind:<5} cal score med {np.median(s):8.4f}")

    with open(os.path.join(cli.out_dir, f"novelty_fit_report{tag}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved", os.path.join(cli.out_dir, f"novelty_fit_report{tag}.json"))


if __name__ == "__main__":
    main()
