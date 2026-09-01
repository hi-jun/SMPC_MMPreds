import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
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
        "cutin": "#ff7f0e",
    }
    offsets = {
        "lane_keeping": -0.45,
        "cutin": -0.15,
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


def _predict_mode_paths(row, horizon, dt, cutin_duration):
    target_s = row["target_s"]
    target_d = row["target_d"]
    target_v = row["target_v"]
    lane_keeping = []
    cutin = []
    for step in range(horizon + 1):
        t = step * dt
        s_lk = target_s + target_v * t
        d_lk = target_d
        v_cut = max(0.0, target_v - 0.5 * t)
        s_cut = target_s + target_v * t + 0.5 * (-0.5) * t * t
        alpha = min(1.0, max(0.0, t / max(cutin_duration, dt)))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        d_cut = target_d + alpha * (0.0 - target_d)
        lane_keeping.append((s_lk, d_lk))
        cutin.append((s_cut, d_cut))
    return np.asarray(lane_keeping), np.asarray(cutin)


def make_animation(input_dir, output_name, fps, horizon, dt, cutin_duration, interpolate):
    csv_path = os.path.join(input_dir, "trajectory.csv")
    rows = _load_rows(csv_path)
    if not rows:
        raise RuntimeError("trajectory.csv is empty")

    if interpolate:
        rows = _interpolate_rows(rows, fps)

    output_path = os.path.join(input_dir, output_name)
    fig, ax = plt.subplots(figsize=(11, 5))

    lane_centers = [0.0, 3.5]
    lane_edges = [-1.75, 1.75, 5.25]

    def update(frame_idx):
        row = rows[frame_idx]
        ax.clear()
        ego_s = row["ego_s"]
        actual_cutin = row.get("actual_cutin_enabled", 1.0)
        actual_label = "actual cut-in" if float(actual_cutin) >= 0.5 else "actual lane-keep"

        ax.set_xlim(ego_s - 15.0, ego_s + 55.0)
        ax.set_ylim(-2.75, 5.75)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Frenet s [m]")
        ax.set_ylabel("Frenet d [m]")
        ax.set_title(
            "ACC Nair SMPC synthetic cut-in smoke | "
            f"t={row['time_s']:.1f}s, a={row['accel_cmd']:.2f}m/s², "
            f"gap={row['gap']:.2f}m, feasible={row['feasible']}, {actual_label}"
        )

        for edge in lane_edges:
            style = "--" if abs(edge - 1.75) < 1e-6 else "-"
            ax.axhline(edge, color="0.45", linestyle=style, linewidth=1.0)
        for center in lane_centers:
            ax.axhline(center, color="0.85", linestyle=":", linewidth=0.8)
        ax.text(ego_s - 14.0, 0.2, "ego lane", color="0.35")
        ax.text(ego_s - 14.0, 3.7, "adjacent lane", color="0.35")

        lane_keeping, cutin = _predict_mode_paths(row, horizon, dt, cutin_duration)
        ax.plot(lane_keeping[:, 0], lane_keeping[:, 1], "m--", linewidth=2, label="pred lane_keeping")
        ax.plot(cutin[:, 0], cutin[:, 1], "c--", linewidth=2, label="pred cutin")
        _plot_p_refs(ax, row, ["lane_keeping", "cutin"])

        d_safe = 3.0 + 1.3 * row["ego_v"]
        safe_start = row["target_s"] - 4.5 - d_safe
        ax.add_patch(
            Rectangle(
                (safe_start, -1.0),
                row["target_s"] - safe_start,
                2.0,
                facecolor="red",
                alpha=0.12,
                edgecolor="red",
                linewidth=1.0,
                label="unsafe gap zone",
            )
        )

        ax.add_patch(_vehicle_patch(row["ego_s"], 0.0, "#2ca02c", "ego"))
        ax.add_patch(_vehicle_patch(row["target_s"], row["target_d"], "#d62728", "target/cutin TV"))
        ax.arrow(
            row["ego_s"] + 3.0,
            0.0,
            max(1.0, row["ego_v"] * 0.35),
            0.0,
            width=0.04,
            color="#2ca02c",
            length_includes_head=True,
        )
        ax.arrow(
            row["target_s"] + 3.0,
            row["target_d"],
            max(1.0, row["target_v"] * 0.35),
            0.0,
            width=0.04,
            color="#d62728",
            length_includes_head=True,
        )
        ax.text(row["ego_s"] - 2.0, -1.35, f"ego v={row['ego_v']:.1f}")
        ax.text(row["target_s"] - 3.0, row["target_d"] + 1.1, f"target v={row['target_v']:.1f}")
        ax.text(
            ego_s + 23.0,
            -2.25,
            f"P(lane_keep)={row['p_lane_keeping']:.1f}, P(cutin)={row['p_cutin']:.1f}",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        ax.legend(loc="upper right")

    animation = FuncAnimation(fig, update, frames=len(rows), interval=1000.0 / fps)
    if output_path.lower().endswith(".mp4"):
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
    else:
        writer = PillowWriter(fps=fps)
    animation.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output-name", default="scenario_animation.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--cutin-duration", type=float, default=2.0)
    parser.add_argument("--no-interpolate", action="store_true")
    args = parser.parse_args()
    print(make_animation(
        args.input_dir,
        args.output_name,
        args.fps,
        args.horizon,
        args.dt,
        args.cutin_duration,
        not args.no_interpolate,
    ))


if __name__ == "__main__":
    main()
