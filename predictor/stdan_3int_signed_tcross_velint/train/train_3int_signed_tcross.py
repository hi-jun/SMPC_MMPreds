from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXP_ROOT = REPO_ROOT / "experiments" / "stdan_3int_signed_tcross_velint"
os.environ.setdefault("WANDB_DIR", str(EXP_ROOT / "wandb"))
os.environ.setdefault("WANDB_CACHE_DIR", str(EXP_ROOT / "wandb" / ".cache"))
os.environ.setdefault("WANDB_CONFIG_DIR", str(EXP_ROOT / "wandb" / ".config"))
os.environ.setdefault("WANDB_DATA_DIR", str(EXP_ROOT / "wandb" / ".data"))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.stdan_3int_signed_tcross_velint.configs.config_3int_signed_tcross import (
    CHECKPOINT_ROOT,
    TRAIN_MAT,
    VAL_MAT,
    WANDB_DIR,
    args,
    device,
    learning_rate,
)
from experiments.stdan_3int_signed_tcross_velint.loaders.loader_3int_signed_tcross import Ngsim3IntTcrossDataset
from experiments.stdan_3int_signed_tcross_velint.models import model5f_3int_signed_tcross as model
from experiments.stdan_3int_signed_tcross_velint.train.losses_3int_signed_tcross import (
    classification_loss,
    endpoint_loss,
    integrate_velocity_distribution,
    masked_mse,
    masked_nll,
    t_cross_loss,
    t_cross_metrics,
    trajectory_metrics,
    velocity_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=args["epoch"])
    parser.add_argument("--batch-size", type=int, default=args["batch_size"])
    parser.add_argument("--num-workers", type=int, default=args["num_worker"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def unpack_batch(data):
    (
        hist, nbrs, mask, intent_enc, intent_label, t_cross_raw, t_cross_norm, t_cross_mask,
        fut, op_mask, va, nbrsva, lane, nbrslane, dis, nbrsdis, cls, nbrscls, map_positions,
    ) = data
    tensors = {
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
        "dis": dis.to(device),
        "nbrsdis": nbrsdis.to(device),
        "cls": cls.to(device),
        "nbrscls": nbrscls.to(device),
        "map_positions": map_positions.to(device),
    }
    return tensors


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total += param_norm.item() ** 2
    return total ** 0.5


def compute_losses(g_vel, g_pos, intent_pred, t_cross_pred, batch, epoch):
    if args["use_mse"] or epoch < args["pre_epoch"]:
        pred_loss = masked_mse(g_pos, batch["fut"], batch["op_mask"])
    else:
        pred_loss = masked_nll(g_pos, batch["fut"], batch["op_mask"])
    raw_cls = classification_loss(intent_pred, batch["intent_enc"])
    cls_loss = args["intention_weight"] * raw_cls
    end_loss = endpoint_loss(g_pos, batch["fut"], batch["op_mask"])
    cross_loss = t_cross_loss(t_cross_pred, batch["t_cross_norm"], batch["t_cross_mask"])
    total = (
        pred_loss
        + cls_loss
        + args["lambda_endpoint"] * end_loss
        + args["lambda_tcross"] * cross_loss
    )
    return total, pred_loss, cls_loss, raw_cls, end_loss, cross_loss


def init_metrics():
    return {
        "total_loss": 0.0,
        "pred_loss": 0.0,
        "cls_loss": 0.0,
        "raw_cls_loss": 0.0,
        "endpoint_loss": 0.0,
        "t_cross_loss": 0.0,
        "accuracy": 0.0,
        "acc_LK": 0.0,
        "acc_LLC": 0.0,
        "acc_RLC": 0.0,
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


def update_metrics(metrics, losses, g_vel, g_pos, intent_pred, t_cross_pred, batch):
    total, pred_loss, cls_loss, raw_cls, end_loss, cross_loss = losses
    pred_label = torch.argmax(intent_pred, dim=-1)
    label = batch["intent_label"]
    correct = pred_label == label
    ade, fde = trajectory_metrics(g_pos, batch["fut"], batch["op_mask"])
    vel_mae, vel_mse = velocity_metrics(g_vel, batch["fut"], batch["op_mask"], args["dt"])
    tc_mae, tc_mse, tc_bias = t_cross_metrics(
        t_cross_pred, batch["t_cross_norm"], batch["t_cross_mask"], args["t_cross_norm_denominator"]
    )
    lc_invalid = ((label != 0) & (batch["t_cross_mask"].view(-1) < 0.5)).float().mean()
    per_class = []
    for cls_id in range(args["num_intentions"]):
        cls_mask = label == cls_id
        if torch.any(cls_mask):
            per_class.append(correct[cls_mask].float().mean().item())
        else:
            per_class.append(0.0)

    metrics["total_loss"] += total.item()
    metrics["pred_loss"] += pred_loss.item()
    metrics["cls_loss"] += cls_loss.item()
    metrics["raw_cls_loss"] += raw_cls.item()
    metrics["endpoint_loss"] += end_loss.item()
    metrics["t_cross_loss"] += cross_loss.item()
    metrics["accuracy"] += correct.float().mean().item()
    metrics["acc_LK"] += per_class[0]
    metrics["acc_LLC"] += per_class[1]
    metrics["acc_RLC"] += per_class[2]
    metrics["ade"] += ade.item()
    metrics["fde"] += fde.item()
    metrics["endpoint_error"] += end_loss.detach().sqrt().item()
    metrics["velocity_mae"] += vel_mae.item()
    metrics["velocity_mse"] += vel_mse.item()
    metrics["t_cross_mae"] += tc_mae.item()
    metrics["t_cross_mse"] += tc_mse.item()
    metrics["t_cross_raw_mae"] += tc_mae.item()
    metrics["t_cross_raw_mse"] += tc_mse.item()
    metrics["t_cross_raw_bias"] += tc_bias.item()
    metrics["valid_t_cross_ratio"] += batch["t_cross_mask"].float().mean().item()
    metrics["invalid_lc_no_cross"] += lc_invalid.item()
    metrics["batches"] += 1


def finalize_metrics(metrics):
    batches = max(metrics.pop("batches"), 1)
    return {k: v / batches for k, v in metrics.items()}


def run_epoch(gd_encoder, generator, dataloader, optimizers, epoch, train=True, max_batches=None):
    if train:
        gd_encoder.train()
        generator.train()
        generator.train_flag = True
    else:
        gd_encoder.eval()
        generator.eval()
        generator.train_flag = True # 이건 왜 True?

    metrics = init_metrics()
    iterator = tqdm(dataloader, desc=f"{'train' if train else 'val'} epoch {epoch + 1}")
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, data in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = unpack_batch(data)
            values = gd_encoder(
                batch["hist"], batch["nbrs"], batch["mask"], batch["va"], batch["nbrsva"],
                batch["lane"], batch["nbrslane"], batch["cls"], batch["nbrscls"],
            )
            g_vel, intent_pred, t_cross_pred = generator(values, batch["intent_enc"])
            g_pos = integrate_velocity_distribution(g_vel, dt=args["dt"])
            losses = compute_losses(g_vel, g_pos, intent_pred, t_cross_pred, batch, epoch)

            if train:
                for opt in optimizers:
                    opt.zero_grad()
                losses[0].backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args["grad_clip"])
                torch.nn.utils.clip_grad_norm_(gd_encoder.parameters(), args["grad_clip"])
                for opt in optimizers:
                    opt.step()

            update_metrics(metrics, losses, g_vel, g_pos, intent_pred, t_cross_pred, batch)
            iterator.set_postfix(loss=losses[0].item())

    final = finalize_metrics(metrics)
    if train:
        final["grad_norm_generator"] = grad_norm(generator.parameters())
        final["grad_norm_encoder"] = grad_norm(gd_encoder.parameters())
    return final


def save_checkpoint(path, gd_encoder, generator, epoch, val_metrics, train_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "gdEncoder": gd_encoder.state_dict(),
            "generator": generator.state_dict(),
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "args": {k: str(v) if k == "device" else v for k, v in args.items()},
        },
        path,
    )


def main():
    cli = parse_args()
    args["epoch"] = cli.epochs
    args["batch_size"] = cli.batch_size
    args["num_worker"] = cli.num_workers

    if not torch.cuda.is_available():
        print("CUDA is not available through PyTorch; running on CPU.")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    WANDB_DIR.mkdir(parents=True, exist_ok=True)

    run_name = cli.name or f"3int_signed_tcross_{time.strftime('%Y%m%d_%H%M%S')}"
    wb = None
    if not cli.no_wandb:
        wb = wandb.init(
            entity="sjun0803-seoul-national-university",  # Replace with your WandB entity if needed
            project=args["wandb_project"],
            name=run_name,
            dir=str(WANDB_DIR),
            config={k: str(v) if isinstance(v, Path) else v for k, v in args.items()},
        )

    train_dataset = Ngsim3IntTcrossDataset(TRAIN_MAT, t_f=args["out_length"], d_s=1)
    val_dataset = Ngsim3IntTcrossDataset(VAL_MAT, t_f=args["out_length"], d_s=1)
    train_loader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=args["num_worker"],
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_worker"],
        collate_fn=val_dataset.collate_fn,
    )

    args["train_flag"] = True
    gd_encoder = model.GDEncoder(args).to(device)
    generator = model.Generator(args).to(device)
    if wb is not None:
        wandb.watch(gd_encoder, log="gradients", log_freq=1000)
        wandb.watch(generator, log="gradients", log_freq=1000)

    optimizer_gd = optim.Adam(gd_encoder.parameters(), lr=learning_rate)
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    scheduler_gd = ReduceLROnPlateau(optimizer_gd, mode="min", factor=0.5, patience=5)
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    patience = 0
    history = []

    for epoch in range(cli.epochs):
        train_metrics = run_epoch(
            gd_encoder, generator, train_loader, [optimizer_gd, optimizer_g],
            epoch, train=True, max_batches=cli.max_batches,
        )
        val_metrics = run_epoch(
            gd_encoder, generator, val_loader, [], epoch, train=False, max_batches=cli.max_batches,
        )
        val_loss = val_metrics["total_loss"]
        scheduler_gd.step(val_loss)
        scheduler_g.step(val_loss)

        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer_g.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if wb is not None:
            wandb.log(row, step=epoch)

        save_checkpoint(CHECKPOINT_ROOT / "last_model.pt", gd_encoder, generator, epoch + 1, val_metrics, train_metrics)
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            save_checkpoint(CHECKPOINT_ROOT / "best_model.pt", gd_encoder, generator, epoch + 1, val_metrics, train_metrics)
        else:
            patience += 1

        with (CHECKPOINT_ROOT / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(f"epoch {epoch + 1}: train={train_metrics['total_loss']:.4f}, val={val_loss:.4f}, best={best_val:.4f}")
        if patience >= args["early_stop_patience"]:
            print("Early stopping triggered.")
            break

    if wb is not None:
        wandb.finish()


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    main()
