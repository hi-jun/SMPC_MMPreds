import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Rectangle
import numpy as np


def _load_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in ("feasible", "status", "solver_message"):
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def _interpolate_rows(rows, fps):
    if len(rows) < 2:
        return rows

    source_times = np.array([row["time_s"] for row in rows], dtype=float)
    frame_times = np.arange(source_times[0], source_times[-1] + 0.5 / fps, 1.0 / fps)
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, float)
    ]
    nonnumeric_keys = [
        key for key in rows[0].keys()
        if key not in numeric_keys
    ]
    interpolated = []
    for frame_time in frame_times:
        item = {}
        for key in numeric_keys:
            item[key] = float(np.interp(frame_time, source_times, [row[key] for row in rows]))
        nearest_idx = int(np.argmin(np.abs(source_times - frame_time)))
        for key in nonnumeric_keys:
            item[key] = rows[nearest_idx].get(key)
        item["time_s"] = float(frame_time)
        interpolated.append(item)
    return interpolated


def _is_true(value):
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _vehicle_patch(s, d, color, label):
    length = 4.5
    width = 1.8
    return Rectangle(
        (s - 0.5 * length, d - 0.5 * width),
        length,
        width,
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
        label=label,
    )


def _json_array(row, key):
    value = row.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def _plot_p_refs(ax, row, mode_names):
    colors = {
        "lane_keeping": "#7b2cbf",
        "cutout": "#ff7f0e",
    }
    offsets = {
        "lane_keeping": -0.45,
        "cutout": -0.15,
    }
    for mode_name in mode_names:
        p_ref = _json_array(row, "p_ref_%s" % mode_name)
        if p_ref is None:
            continue
        ref_d = np.full_like(p_ref, offsets.get(mode_name, -0.45), dtype=float)
        color = colors.get(mode_name, "black")
        ax.plot(
            p_ref,
            ref_d,
            "-",
            color=color,
            linewidth=2.0,
            alpha=0.9,
            label="p_ref %s" % mode_name,
        )
        ax.scatter(
            p_ref,
            ref_d,
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )


def _predict_mode_paths(row, horizon, dt, cutout_duration, cutout_start_delay, target_d_final):
    target_s = row["target_s"]
    target_d = row["target_d"]
    target_v = row["target_v"]
    lane_keeping = []
    cutout = []
    for step in range(horizon + 1):
        t = step * dt
        s = target_s + target_v * t
        lane_keeping.append((s, target_d))
        alpha = min(1.0, max(0.0, (t - cutout_start_delay) / max(cutout_duration, dt)))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        d_cut = target_d + alpha * (target_d_final - target_d)
        cutout.append((s, d_cut))
    return np.asarray(lane_keeping), np.asarray(cutout)


def make_animation(
        input_dir,
        output_name,
        fps,
        horizon,
        dt,
        cutout_duration,
        cutout_start_delay,
        target_d_final,
        interpolate):
    csv_path = os.path.join(input_dir, "trajectory.csv")
    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError("trajectory.csv is empty")

    if interpolate:
        rows = _interpolate_rows(rows, fps)

    output_path = os.path.join(input_dir, output_name)
    fig, ax = plt.subplots(figsize=(11, 5))
    speed_ax = fig.add_axes([0.58, 0.14, 0.28, 0.22])

    lane_centers = [0.0, target_d_final]
    lane_edges = [-1.75, 1.75, target_d_final + 1.75]
    full_times = [row["time_s"] for row in rows]
    full_speeds = [row["ego_v"] for row in rows]
    desired_speed = rows[0].get("desired_speed", max(full_speeds))

    def update(frame_idx):
        row = rows[frame_idx]
        ax.clear()
        speed_ax.clear()
        ego_s = row["ego_s"]
        desired = row.get("desired_speed", desired_speed)
        lead_active = _is_true(row.get("lead_active", "True"))
        gap_text = "inactive"
        if lead_active and np.isfinite(row.get("gap", np.nan)):
            gap_text = f"{row['gap']:.2f}m"

        ax.set_xlim(ego_s - 15.0, ego_s + 65.0)
        ax.set_ylim(-2.75, target_d_final + 2.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Frenet s [m]")
        ax.set_ylabel("Frenet d [m]")
        ax.set_title(
            "ACC Nair SMPC synthetic cut-out smoke | "
            f"t={row['time_s']:.1f}s, a={row['accel_cmd']:.2f}m/s², "
            f"v={row['ego_v']:.2f}/{desired:.2f}m/s, "
            f"gap={gap_text}, feasible={row['feasible']}"
        )

        for edge in lane_edges:
            style = "--" if abs(edge - 1.75) < 1e-6 else "-"
            ax.axhline(edge, color="0.45", linestyle=style, linewidth=1.0)
        for center in lane_centers:
            ax.axhline(center, color="0.85", linestyle=":", linewidth=0.8)

        if lead_active:
            unsafe_start = row["ego_s"] + 4.5 + 3.0 + 1.3 * row["ego_v"]
            ax.axvspan(row["ego_s"] - 2.25, unsafe_start, ymin=0.05, ymax=0.45, color="red", alpha=0.08)
            ax.text(ego_s - 13.5, -2.35, "red zone: desired ACC safety distance", fontsize=8)
        else:
            ax.text(ego_s - 13.5, -2.35, "lead inactive: target is outside ego lane", fontsize=8)

        ego = _vehicle_patch(row["ego_s"], 0.0, "#2ca02c", "ego")
        target = _vehicle_patch(row["target_s"], row["target_d"], "#d62728", "target")
        ax.add_patch(ego)
        ax.add_patch(target)

        lane_keeping, cutout = _predict_mode_paths(
            row, horizon, dt, cutout_duration, cutout_start_delay, target_d_final
        )
        ax.plot(lane_keeping[:, 0], lane_keeping[:, 1], "m.-", linewidth=1.5, markersize=3,
                label=f"lane_keeping p={row['p_lane_keeping']:.2f}")
        ax.plot(cutout[:, 0], cutout[:, 1], "c.-", linewidth=1.5, markersize=3,
                label=f"cutout p={row['p_cutout']:.2f}")
        _plot_p_refs(ax, row, ["lane_keeping", "cutout"])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)

        shown_times = full_times[:frame_idx + 1]
        shown_speeds = full_speeds[:frame_idx + 1]
        speed_ax.plot(shown_times, shown_speeds, color="#2ca02c", linewidth=1.5, label="ego v")
        speed_ax.axhline(desired, color="#1f77b4", linestyle="--", linewidth=1.2, label="desired v")
        speed_ax.set_xlim(full_times[0], full_times[-1])
        speed_ax.set_ylim(0.0, max(desired + 1.0, max(full_speeds) + 0.5))
        speed_ax.set_xlabel("t [s]", fontsize=7)
        speed_ax.set_ylabel("v [m/s]", fontsize=7)
        speed_ax.tick_params(labelsize=7)
        speed_ax.grid(True, alpha=0.25)
        speed_ax.legend(loc="lower right", fontsize=6)

    animation = FuncAnimation(fig, update, frames=len(rows), interval=1000.0 / fps)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(output_path, writer=writer)
    plt.close(fig)
    print(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output-name", default="cutout_scenario_10s_20fps.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--cutout-duration", type=float, default=1.5)
    parser.add_argument("--cutout-start-delay", type=float, default=0.5)
    parser.add_argument("--target-d-final", type=float, default=3.5)
    parser.add_argument("--no-interpolate", action="store_true")
    args = parser.parse_args()
    make_animation(
        args.input_dir,
        args.output_name,
        args.fps,
        args.horizon,
        args.dt,
        args.cutout_duration,
        args.cutout_start_delay,
        args.target_d_final,
        not args.no_interpolate,
    )


if __name__ == "__main__":
    main()
