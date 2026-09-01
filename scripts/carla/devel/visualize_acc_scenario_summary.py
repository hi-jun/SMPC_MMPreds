from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INTENTION_COLORS = {
    "LK": "#7b2cbf",
    "LLC": "#1f77b4",
    "RLC": "#ff7f0e",
}


def _load_summary(path: Path) -> dict:
    if path.is_dir():
        path = path / "summary.json"
    with path.open("r") as f:
        return json.load(f)


def _ego_steps(summary: dict) -> list[dict]:
    for key, log in summary.get("policy_logs", {}).items():
        if key.startswith("ego_") and isinstance(log, dict):
            return log.get("steps", [])
    return []


def _target_log(summary: dict) -> dict:
    for key, log in summary.get("policy_logs", {}).items():
        if key.startswith("target_cutin") and isinstance(log, dict):
            return log
    return {}


def _processed_target(step: dict) -> dict | None:
    targets = step.get("stdan_debug", {}).get("processed_targets", [])
    if not targets:
        return None
    return targets[0]


def _series(steps: list[dict]) -> dict:
    rows = []
    for step in steps:
        target = _processed_target(step)
        raw = target.get("raw_mode_prob", {}) if target else {}
        acc = target.get("acc_mode_prob", {}) if target else {}
        mask = target.get("ego_lane_occupancy_mask", []) if target else []
        rows.append({
            "time_s": float(step.get("time_s", np.nan)),
            "ego_s": float(step.get("ego_s", np.nan)),
            "ego_v": float(step.get("ego_v", np.nan)),
            "target_s": float(step.get("target_s", np.nan)),
            "target_d": float(step.get("target_d", np.nan)),
            "accel_cmd": float(step.get("accel_cmd", np.nan)),
            "relation": target.get("relation_to_ego_lane") if target else None,
            "LK": float(raw.get("LK", np.nan)),
            "LLC": float(raw.get("LLC", np.nan)),
            "RLC": float(raw.get("RLC", np.nan)),
            "acc_cutin": float(acc.get("cutin", np.nan)),
            "acc_lk": float(acc.get("lk", np.nan)),
            "occupancy": bool(any(mask)),
            "source": target.get("lane_membership_source") if target else None,
        })
    return {key: np.asarray([row[key] for row in rows], dtype=object) for key in rows[0].keys()} if rows else {}


def _numeric(values):
    return np.asarray(values, dtype=float)


def _shade_relations(ax, time, relation):
    if len(time) == 0:
        return
    relation = np.asarray(relation, dtype=object)
    time = _numeric(time)
    spans = [
        ("left_adjacent", "#ffedd5"),
        ("right_adjacent", "#dbeafe"),
        ("ego_lane", "#dcfce7"),
    ]
    for name, color in spans:
        mask = relation == name
        if not np.any(mask):
            continue
        indices = np.where(mask)[0]
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx != prev + 1:
                ax.axvspan(time[start], time[prev], color=color, alpha=0.35, linewidth=0)
                start = idx
            prev = idx
        ax.axvspan(time[start], time[prev], color=color, alpha=0.35, linewidth=0)


def _plot_run(fig, axes, data, title, target_log):
    time = _numeric(data["time_s"])
    target_s = _numeric(data["target_s"])
    target_d = _numeric(data["target_d"])
    ego_s = _numeric(data["ego_s"])
    ego_v = _numeric(data["ego_v"])
    accel = _numeric(data["accel_cmd"])
    relation = data["relation"]

    ax = axes[0]
    ax.plot(target_s, target_d, color="#d62728", linewidth=2.0, label="target")
    ax.plot(ego_s, np.zeros_like(ego_s), color="#2ca02c", linewidth=2.0, label="ego")
    ax.axhline(0.0, color="0.25", linewidth=1.0, label="ego lane center")
    for edge in (-1.75, 1.75, 5.25, 8.75, 12.25):
        ax.axhline(edge, color="0.65", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Frenet d [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    _shade_relations(ax, time, relation)
    ax.plot(time, target_d, color="#d62728", linewidth=1.8, label="target d")
    occ = np.asarray(data["occupancy"], dtype=bool)
    if np.any(occ):
        ax.scatter(time[occ], target_d[occ], s=16, color="black", label="pred occupies ego lane")
    trigger_time = target_log.get("trigger_time_s")
    if trigger_time is not None:
        ax.axvline(float(trigger_time), color="black", linestyle="--", linewidth=1.2, label="trigger")
    ax.set_ylabel("target d [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    _shade_relations(ax, time, relation)
    for name, color in INTENTION_COLORS.items():
        ax.plot(time, _numeric(data[name]), color=color, linewidth=1.6, label=name)
    ax.plot(time, _numeric(data["acc_cutin"]), color="black", linestyle="--", linewidth=1.4, label="ACC cutin")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("probability")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[3]
    ax.plot(time, ego_v, color="#2ca02c", linewidth=1.6, label="ego v")
    ax2 = ax.twinx()
    ax2.plot(time, accel, color="#9467bd", linewidth=1.2, label="accel")
    ax.set_ylabel("ego v [m/s]")
    ax2.set_ylabel("accel cmd")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.2)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    status = (
        f"started={target_log.get('lane_change_started')}, "
        f"completed={target_log.get('lane_change_completed')}, "
        f"direction={target_log.get('lane_change_direction')}"
    )
    axes[0].text(
        0.02,
        0.92,
        status,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
    )


def make_figure(before_dir: Path, after_dir: Path, output_path: Path):
    before = _load_summary(before_dir)
    after = _load_summary(after_dir)
    before_data = _series(_ego_steps(before))
    after_data = _series(_ego_steps(after))
    if not before_data or not after_data:
        raise RuntimeError("Both summaries must contain ego policy steps.")

    fig, axes = plt.subplots(4, 2, figsize=(15, 11), constrained_layout=True)
    _plot_run(fig, axes[:, 0], before_data, "Before: full-history gated STDAN", _target_log(before))
    _plot_run(fig, axes[:, 1], after_data, "After: partial-history + CARLA waypoint occupancy", _target_log(after))
    fig.suptitle("STDAN ACC scenario comparison", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(make_figure(Path(args.before), Path(args.after), Path(args.output)))


if __name__ == "__main__":
    main()
