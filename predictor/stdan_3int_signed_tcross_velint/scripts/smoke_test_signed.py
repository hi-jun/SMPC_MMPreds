from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from experiments.stdan_3int_signed_tcross_velint.configs.config_3int_signed_tcross import DATA_ROOT, args, device
from experiments.stdan_3int_signed_tcross_velint.loaders.loader_3int_signed_tcross import Ngsim3IntTcrossDataset
from experiments.stdan_3int_signed_tcross_velint.models import model5f_3int_signed_tcross as model
from experiments.stdan_3int_signed_tcross_velint.train.losses_3int_signed_tcross import (
    classification_loss,
    endpoint_loss,
    integrate_velocity_distribution,
    masked_mse,
    t_cross_loss,
    velocity_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def pick_mat(cli_mat):
    if cli_mat:
        return Path(cli_mat)
    full = DATA_ROOT / "TrainSet.mat"
    smoke = DATA_ROOT / "smoke" / "TrainSet.mat"
    return full if full.exists() else smoke


def main():
    cli = parse_args()
    mat_path = pick_mat(cli.mat)
    if not mat_path.exists():
        raise FileNotFoundError(f"No processed dataset found for smoke test: {mat_path}")

    dataset = Ngsim3IntTcrossDataset(mat_path, t_f=args["out_length"], d_s=1)
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    data = next(iter(loader))
    (
        hist, nbrs, mask, intent_enc, intent_label, t_cross_raw, t_cross_norm, t_cross_mask,
        fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions,
    ) = data

    if not set(torch.unique(intent_label).tolist()).issubset({0, 1, 2}):
        raise AssertionError("Intention labels must be 0, 1, 2")
    if not set(torch.unique(t_cross_mask.long()).tolist()).issubset({0, 1}):
        raise AssertionError("t_cross_mask must be binary")
    valid = t_cross_mask.view(-1) > 0.5
    if torch.any(valid):
        max_signed_step = args["t_cross_norm_denominator"]
        if torch.any((t_cross_raw.view(-1)[valid] < -max_signed_step) | (t_cross_raw.view(-1)[valid] > max_signed_step)):
            raise AssertionError("t_cross_raw must be signed relative steps within the LC labeling window")

    args["train_flag"] = True
    gd_encoder = model.GDEncoder(args).to(device)
    generator = model.Generator(args).to(device)
    gd_encoder.train()
    generator.train()

    batch = {
        "hist": hist.to(device),
        "nbrs": nbrs.to(device),
        "mask": mask.to(device),
        "intent_enc": intent_enc.to(device),
        "intent_label": intent_label.to(device),
        "t_cross_norm": t_cross_norm.to(device),
        "t_cross_mask": t_cross_mask.to(device),
        "fut": fut[:args["out_length"], :, :].to(device),
        "op_mask": op_mask[:args["out_length"], :, :].to(device),
        "va": va.to(device),
        "nbrsva": nbrsva.to(device),
        "lane": lane.to(device),
        "nbrslane": nbrslane.to(device),
        "cls": cls.to(device),
        "nbrscls": nbrscls.to(device),
    }

    values = gd_encoder(
        batch["hist"], batch["nbrs"], batch["mask"], batch["va"], batch["nbrsva"],
        batch["lane"], batch["nbrslane"], batch["cls"], batch["nbrscls"],
    )
    g_vel, intent_pred, t_cross_pred = generator(values, batch["intent_enc"])
    g_pos = integrate_velocity_distribution(g_vel, dt=args["dt"])
    if intent_pred.shape != (cli.batch_size, 3):
        raise AssertionError(f"Expected 3 intention outputs, got {intent_pred.shape}")
    if t_cross_pred.shape != (cli.batch_size, 1):
        raise AssertionError(f"Expected scalar t_cross output, got {t_cross_pred.shape}")

    pred_loss = masked_mse(g_pos, batch["fut"], batch["op_mask"])
    cls_loss = classification_loss(intent_pred, batch["intent_enc"])
    end_loss = endpoint_loss(g_pos, batch["fut"], batch["op_mask"])
    cross_loss = t_cross_loss(t_cross_pred, batch["t_cross_norm"], batch["t_cross_mask"])
    vel_mae, vel_mse = velocity_metrics(g_vel, batch["fut"], batch["op_mask"], args["dt"])
    total = pred_loss + cls_loss + end_loss + cross_loss
    if not torch.isfinite(total):
        raise AssertionError("Smoke-test loss is not finite")
    total.backward()

    summary = {
        "mat_path": str(mat_path),
        "batch_size": cli.batch_size,
        "intent_labels": sorted(set(intent_label.tolist())),
        "valid_t_cross_in_batch": int(torch.sum(t_cross_mask).item()),
        "pred_loss": float(pred_loss.item()),
        "cls_loss": float(cls_loss.item()),
        "endpoint_loss": float(end_loss.item()),
        "velocity_mae": float(vel_mae.item()),
        "velocity_mse": float(vel_mse.item()),
        "t_cross_loss": float(cross_loss.item()),
        "total_loss": float(total.item()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
