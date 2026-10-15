"""Split-conformal calibration of STDAN's lateral-intent softmax.

Phase 2 of the spec asks for a plain 3-class softmax intent classifier over the
interaction context. STDAN already HAS one - its lateral maneuver head, trained
on exactly that context - so no new model is trained here; we calibrate it.

Splits: TrainSet trained the model, ValSet is the calibration split and TestSet
validates coverage. The NGSIM preprocessing splits by trajectory, so no track
is shared between them (the requirement that makes the CP guarantee valid).

Run (hmg env, from this directory):
    python cp_calibrate.py --data /home/core-dev/HJ/dataset/raw_data_ngsim/dataset_t_v_t

Writes ../ckpt/cp_calibration.npz (cal_scores) and prints the realized coverage
and mean set size on the test split.
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

from config_vel import args                       # noqa: E402
import loader2_hdf5_vel as lo                     # noqa: E402
from extract_features import STDAN_VEL_CKPT, load_trunk  # noqa: E402
from predictor.uq.stream_tap import encoder_inputs  # noqa: E402
from predictor.uq.cp_gate import (   # noqa: E402
    ConformalIntentGate, true_class_scores,
)

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ckpt")


def collect_probs(gd, gen, loader, device, max_batches):
    """Lateral softmax probabilities and labels over a split."""
    probs, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            enc_in, lat_enc = encoder_inputs(batch, device)
            values = gd(*enc_in)
            if isinstance(values, tuple):
                values = values[0]
            n = values.shape[0] if values.dim() > 2 else 1
            dummy_lat = torch.zeros(values.shape[0], args["lat_length"],
                                    device=device)
            dummy_lon = torch.zeros(values.shape[0], args["lon_length"],
                                    device=device)
            out = gen(values, dummy_lat, dummy_lon)
            lat_pred = out[1]
            probs.append(lat_pred.detach().cpu().numpy())
            labels.append(lat_enc.argmax(-1).cpu().numpy())
    return np.concatenate(probs, 0), np.concatenate(labels, 0)


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
    ap.add_argument("--score", choices=["lac", "aps"], default="lac",
                    help="lac gives the credibility semantics the gate needs")
    cli = ap.parse_args()
    device = args["device"]
    torch.manual_seed(72)
    gd, gen = load_trunk(device, ckpt=cli.ckpt)

    cal_loader = balanced_loader(os.path.join(cli.data, "ValSet.mat"),
                                 cli.cal_batches, seed=1)
    p_cal, y_cal = collect_probs(gd, gen, cal_loader, device, cli.cal_batches)
    cal_scores = true_class_scores(p_cal, y_cal, score=cli.score)
    np.savez(os.path.join(cli.out_dir, "cp_calibration.npz"),
             cal_scores=cal_scores, score=cli.score)
    acc_cal = float((p_cal.argmax(-1) == y_cal).mean())
    print(f"calibration: n={len(cal_scores)}  분류 정확도 {acc_cal:.3f}  "
          f"score 중앙값 {np.median(cal_scores):.3f}")

    # coverage validation on the untouched test split (spec fig5: coverage and
    # set size must ALWAYS be reported together)
    gate = ConformalIntentGate(cal_scores, score=cli.score)
    test_loader = balanced_loader(os.path.join(cli.data, "TestSet.mat"),
                                  cli.test_batches, seed=2)
    p_te, y_te = collect_probs(gd, gen, test_loader, device, cli.test_batches)
    report = {"cal_n": int(len(cal_scores)), "cal_acc": acc_cal}
    print(f'\n{"alpha":>6} {"목표 커버리지":>12} {"실현":>8} {"평균 집합크기":>12}')
    for alpha in (0.05, 0.1, 0.2):
        covered, sizes = [], []
        for p, y in zip(p_te, y_te):
            pv = gate.p_values(p)
            keep = pv > alpha
            covered.append(bool(keep[int(y)]))
            sizes.append(int(keep.sum()))
        cov, sz = float(np.mean(covered)), float(np.mean(sizes))
        report[f"alpha_{alpha}"] = {"coverage": cov, "set_size": sz}
        print(f'{alpha:>6} {1-alpha:>12.2f} {cov:>8.3f} {sz:>12.2f}')

    # what the gate will look like at runtime
    gs, creds, confs = [], [], []
    for p in p_te:
        g_, _, cred, conf = gate.gate(p)
        gs.append(g_); creds.append(cred); confs.append(conf)
    report["gate"] = {"median": float(np.median(gs)),
                      "p10": float(np.percentile(gs, 10)),
                      "p90": float(np.percentile(gs, 90)),
                      "credibility_median": float(np.median(creds)),
                      "confidence_median": float(np.median(confs))}
    print(f'\n게이트 g: 중앙 {np.median(gs):.3f} '
          f'(10% {np.percentile(gs,10):.3f} / 90% {np.percentile(gs,90):.3f})'
          f' | credibility 중앙 {np.median(creds):.3f}'
          f' | confidence 중앙 {np.median(confs):.3f}')
    with open(os.path.join(cli.out_dir, "cp_calibration_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved", os.path.join(cli.out_dir, "cp_calibration.npz"))
