from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.stdan_3int_signed_tcross_velint.configs.config_3int_signed_tcross import (
    CHECKPOINT_ROOT,
    FIGURE_ROOT,
    TEST_MAT,
    VAL_MAT,
    args,
    device,
)
from experiments.stdan_3int_signed_tcross_velint.loaders.loader_3int_signed_tcross import Ngsim3IntTcrossDataset
from experiments.stdan_3int_signed_tcross_velint.models import model5f_3int_signed_tcross as model
from experiments.stdan_3int_signed_tcross_velint.train.losses_3int_signed_tcross import (
    classification_loss,
    endpoint_loss,
    integrate_velocity_modes,
    masked_mse,
    masked_nll,
    t_cross_loss,
    t_cross_metrics,
    trajectory_metrics,
    velocity_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=args["batch_size"])
    parser.add_argument("--num-workers", type=int, default=args["num_worker"])
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def unpack_batch(data):
    (
        hist, nbrs, mask, intent_enc, intent_label, t_cross_raw, t_cross_norm, t_cross_mask,
        fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions,
    ) = data
    return {
        "hist": hist.to(device),
        "nbrs": nbrs.to(device),
        "mask": mask.to(device),
        "intent_enc": intent_enc.to(device),
        "intent_label": intent_label.to(device),
        "t_cross_raw": t_cross_raw.to(device),
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


def select_predicted_mode(fut_modes, intent_pred):
    pred_label = torch.argmax(intent_pred, dim=-1)
    selected = torch.zeros_like(fut_modes[0])
    for cls_id, fut_pred in enumerate(fut_modes):
        cls_mask = pred_label == cls_id
        if torch.any(cls_mask):
            selected[:, cls_mask, :] = fut_pred[:, cls_mask, :]
    return selected, pred_label


def safe_load(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    cli = parse_args()
    args["batch_size"] = cli.batch_size
    args["num_worker"] = cli.num_workers
    args["train_flag"] = False
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_ROOT / "best_model.pt" if cli.checkpoint == "best" else Path(cli.checkpoint)
    checkpoint = safe_load(checkpoint_path)

    gd_encoder = model.GDEncoder(args).to(device)
    generator = model.Generator(args).to(device)
    gd_encoder.load_state_dict(checkpoint["gdEncoder"])
    generator.load_state_dict(checkpoint["generator"])
    gd_encoder.eval()
    generator.eval()
    generator.train_flag = False

    mat_path = TEST_MAT if cli.split == "test" else VAL_MAT
    dataset = Ngsim3IntTcrossDataset(mat_path, t_f=args["out_length"], d_s=1)
    loader = DataLoader(
        dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_worker"],
        collate_fn=dataset.collate_fn,
    )

    sums = {
        "total_loss": 0.0,
        "pred_loss": 0.0,
        "cls_loss": 0.0,
        "endpoint_loss": 0.0,
        "t_cross_loss": 0.0,
        "ade": 0.0,
        "fde": 0.0,
        "endpoint_error": 0.0,
        "velocity_mae": 0.0,
        "velocity_mse": 0.0,
        "t_cross_mae": 0.0,
        "t_cross_mse": 0.0,
        "t_cross_raw_mae": 0.0,
        "t_cross_raw_mse": 0.0,
        "t_cross_raw_bias": 0.0,
        "valid_t_cross_ratio": 0.0,
        "invalid_lc_no_cross": 0.0,
        "batches": 0,
    }
    all_pred, all_gt = [], []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc=f"eval {cli.split}")):
            if cli.max_batches is not None and batch_idx >= cli.max_batches:
                break
            batch = unpack_batch(data)
            values = gd_encoder(
                batch["hist"], batch["nbrs"], batch["mask"], batch["va"], batch["nbrsva"],
                batch["lane"], batch["nbrslane"], batch["cls"], batch["nbrscls"],
            )
            fut_vel_modes, intent_pred, t_cross_pred = generator(values, batch["intent_enc"])
            fut_modes = integrate_velocity_modes(fut_vel_modes, dt=args["dt"])
            fut_pred, pred_label = select_predicted_mode(fut_modes, intent_pred)
            fut_vel_pred, _ = select_predicted_mode(fut_vel_modes, intent_pred)

            if args["val_use_mse"]:
                pred_loss = masked_mse(fut_pred, batch["fut"], batch["op_mask"])
            else:
                pred_loss = masked_nll(fut_pred, batch["fut"], batch["op_mask"])
            cls_loss = args["intention_weight"] * classification_loss(intent_pred, batch["intent_enc"])
            end_loss = endpoint_loss(fut_pred, batch["fut"], batch["op_mask"])
            cross_loss = t_cross_loss(t_cross_pred, batch["t_cross_norm"], batch["t_cross_mask"])
            total = pred_loss + cls_loss + args["lambda_endpoint"] * end_loss + args["lambda_tcross"] * cross_loss
            ade, fde = trajectory_metrics(fut_pred, batch["fut"], batch["op_mask"])
            vel_mae, vel_mse = velocity_metrics(fut_vel_pred, batch["fut"], batch["op_mask"], args["dt"])
            tc_mae, tc_mse, tc_bias = t_cross_metrics(
                t_cross_pred, batch["t_cross_norm"], batch["t_cross_mask"], args["t_cross_norm_denominator"]
            )
            invalid_lc = ((batch["intent_label"] != 0) & (batch["t_cross_mask"].view(-1) < 0.5)).float().mean()

            sums["total_loss"] += total.item()
            sums["pred_loss"] += pred_loss.item()
            sums["cls_loss"] += cls_loss.item()
            sums["endpoint_loss"] += end_loss.item()
            sums["t_cross_loss"] += cross_loss.item()
            sums["ade"] += ade.item()
            sums["fde"] += fde.item()
            sums["endpoint_error"] += end_loss.sqrt().item()
            sums["velocity_mae"] += vel_mae.item()
            sums["velocity_mse"] += vel_mse.item()
            sums["t_cross_mae"] += tc_mae.item()
            sums["t_cross_mse"] += tc_mse.item()
            sums["t_cross_raw_mae"] += tc_mae.item()
            sums["t_cross_raw_mse"] += tc_mse.item()
            sums["t_cross_raw_bias"] += tc_bias.item()
            sums["valid_t_cross_ratio"] += batch["t_cross_mask"].float().mean().item()
            sums["invalid_lc_no_cross"] += invalid_lc.item()
            sums["batches"] += 1
            all_pred.extend(pred_label.detach().cpu().tolist())
            all_gt.extend(batch["intent_label"].detach().cpu().tolist())

    batches = max(sums.pop("batches"), 1)
    metrics = {k: v / batches for k, v in sums.items()}
    metrics["accuracy"] = accuracy_score(all_gt, all_pred) if all_gt else 0.0
    class_names = ["LK", "LLC", "RLC"]
    cm = confusion_matrix(all_gt, all_pred, labels=[0, 1, 2])
    for cls_id, cls_name in enumerate(class_names):
        denom = cm[cls_id].sum()
        metrics[f"acc_{cls_name}"] = float(cm[cls_id, cls_id] / denom) if denom else 0.0

    metrics_path = FIGURE_ROOT / f"{cli.split}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("3-Intention Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Prediction")
    plt.tight_layout()
    fig_path = FIGURE_ROOT / f"{cli.split}_confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {fig_path}")


if __name__ == "__main__":
    main()
