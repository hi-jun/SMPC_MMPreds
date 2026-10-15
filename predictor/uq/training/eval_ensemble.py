"""Deep-ensemble member acceptance check (run in the hmg env).

For member0 (the deployed augmented best_model.pt) and every trained member:
teacher-forced velocity output integrated to positions on a fixed ValSet
subset -> masked MSE (the training val metric) and the pairwise ADE between
members (ft -> m). Acceptance: member MSE <= 1.10 x member0, all checkpoint
md5s distinct, every pairwise ADE > 1e-3 m (near-zero = seed not applied).

Usage:
  cd predictor/uq/training && python eval_ensemble.py \
      [--members ../ckpt/ensemble] [--n 2048] \
      [--val /home/core-dev/HJ/dataset/NGsim/0113_ratio211/ValSet.mat]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch as t
from torch.utils.data import DataLoader, Subset

# config_vel / loader2_hdf5_vel / model5f_mult_vel: this repository's
# predictor/stdan_vel copies (same model and loader as the source repository)
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "stdan_vel")))
import loader2_hdf5_vel as lo  # noqa: E402
from config_vel import *  # noqa: F401,F403,E402 - args, model, device

DT = 0.1


def load_member(ckpt_path):
    ck = t.load(ckpt_path, map_location=device, weights_only=False)
    gd = model.GDEncoder(args).to(device)
    g = model.Generator(args).to(device)
    gd.load_state_dict(ck["gdEncoder"])
    g.load_state_dict(ck["generator"])
    gd.eval(); g.eval()
    return gd, g


def forward_member(gd, g, loader):
    """Teacher-forced mean positions (T,B,2) per batch + masked MSE."""
    outs, se, cnt = [], 0.0, 0.0
    with t.no_grad():
        for data in loader:
            (hist, nbrs, mask, lat_enc, lon_enc, fut_vel, fut_pos, op_mask,
             va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls,
             map_positions) = data
            hist, nbrs, mask = hist.to(device), nbrs.to(device), mask.to(device)
            lat_enc, lon_enc = lat_enc.to(device), lon_enc.to(device)
            va, nbrsva = va.to(device), nbrsva.to(device)
            lane, nbrslane = lane.to(device), nbrslane.to(device)
            cls, nbrscls = cls.to(device), nbrscls.to(device)
            fut_pos = fut_pos[:args['out_length']].to(device)
            op_mask = op_mask[:args['out_length']].to(device)
            values = gd(hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls)
            g_out, _, _ = g(values, lat_enc, lon_enc)
            mu_p = t.cumsum(g_out[:, :, 0:2] * DT, dim=0)   # integrate velocity
            d2 = ((mu_p - fut_pos[:, :, 0:2]) ** 2).sum(-1)
            m = op_mask[:, :, 0]
            se += float((d2 * m).sum()); cnt += float(m.sum())
            outs.append((mu_p.cpu().numpy(), m.cpu().numpy()))
    return outs, se / max(cnt, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="../ckpt/ensemble")
    ap.add_argument("--member0",
                    default="../../stdan_vel/checkpoints/0113_ratio211/best_model.pt")
    ap.add_argument("--val",
                    default="/home/core-dev/HJ/dataset/NGsim/0113_ratio211/ValSet.mat")
    ap.add_argument("--n", type=int, default=2048)
    args_cli = ap.parse_args()

    args['train_flag'] = True          # teacher-forced single-mode output
    ds = lo.NgsimDataset(args_cli.val, t_f=50, d_s=1)
    sub = Subset(ds, range(min(args_cli.n, len(ds))))
    loader = DataLoader(sub, batch_size=args['batch_size'], shuffle=False,
                        num_workers=4, collate_fn=ds.collate_fn)

    ckpts = {"member0": args_cli.member0}
    for i in range(1, 20):
        p = os.path.join(args_cli.members, f"member{i}", "best_model.pt")
        if os.path.exists(p):
            ckpts[f"member{i}"] = p

    outs, mses, md5s = {}, {}, {}
    for name, path in ckpts.items():
        gd, g = load_member(path)
        outs[name], mses[name] = forward_member(gd, g, loader)
        md5s[name] = hashlib.md5(open(path, "rb").read()).hexdigest()
        print(f"{name}: val MSE {mses[name]:.4f}  "
              f"(x{mses[name]/mses['member0']:.3f} of member0)  md5 {md5s[name][:8]}")

    names = list(ckpts)
    ade = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            num = den = 0.0
            for (pa, m), (pb, _) in zip(outs[a], outs[b]):
                d = np.sqrt(((pa - pb) ** 2).sum(-1)) * m
                num += d.sum(); den += m.sum()
            ade[i, j] = ade[j, i] = num / max(den, 1.0) * 0.3048   # ft -> m
    print("\npairwise ADE [m] (masked mean):")
    for i, a in enumerate(names):
        print(f"  {a:<9}" + "".join(f"{ade[i, j]:>8.3f}" for j in range(len(names))))

    member_names = [n for n in names if n != "member0"]
    ok_mse = all(mses[n] <= 1.10 * mses["member0"] for n in member_names)
    ok_md5 = len(set(md5s.values())) == len(md5s)
    off = [ade[i, j] for i in range(len(names)) for j in range(i + 1, len(names))]
    ok_div = all(d > 1e-3 for d in off)
    print(f"\nacceptance: mse<=1.10x {ok_mse}, md5 distinct {ok_md5}, "
          f"diversity>1e-3m {ok_div}; mean pairwise ADE {np.mean(off):.3f} m")

    report = {"mse": mses, "md5": md5s,
              "pairwise_ade_m": {f"{a}|{names[j]}": float(ade[i, j])
                                 for i, a in enumerate(names)
                                 for j in range(i + 1, len(names))},
              "accept": bool(ok_mse and ok_md5 and ok_div)}
    out_path = os.path.join(args_cli.members, "eval_report.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("report ->", out_path)


if __name__ == "__main__":
    main()
