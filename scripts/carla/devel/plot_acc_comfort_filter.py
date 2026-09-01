from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_acc_overnight_sweep import (
    DEFAULT_COMFORT_LOWPASS_TAU_S,
    derivative_from_series,
    filtered_comfort_series,
    safe_float,
)


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def first_policy_log(result, prefix):
    for key, value in result.items():
        if str(key).startswith(prefix) and isinstance(value, dict):
            log = value.get("policy_log", {})
            if isinstance(log, dict):
                return log
    return {}


def event_policy_log(result):
    fallback = {}
    for key, value in result.items():
        if not str(key).startswith("target_") or not isinstance(value, dict):
            continue
        log = value.get("policy_log", {})
        if not isinstance(log, dict):
            continue
        if not fallback:
            fallback = log
        if safe_float(log.get("trigger_time_s")) is not None:
            return log
    return fallback


def step_series(steps, key):
    rows = []
    for step in steps:
        time_s = safe_float(step.get("time_s"))
        value = safe_float(step.get(key))
        if time_s is None or value is None:
            continue
        rows.append({"time_s": time_s, "value": value})
    return rows


def series_xy(rows):
    return [row["time_s"] for row in rows], [row["value"] for row in rows]


def abs_stats(rows):
    values = [abs(row["value"]) for row in rows if safe_float(row.get("value")) is not None]
    if not values:
        return "n/a"
    return f"mean |.|={sum(values) / len(values):.3f}, peak |.|={max(values):.3f}"


def tau_label(tau_s):
    return f"{float(tau_s):.2f}".replace(".", "p")


def add_event_marker(ax, event_time_s):
    if event_time_s is None:
        return
    ax.axvline(event_time_s, color="#444444", linestyle=":", linewidth=1.1, alpha=0.85)


def plot_lines(ax, rows, label, color, linewidth=1.8, alpha=1.0, linestyle="-"):
    if not rows:
        return
    x, y = series_xy(rows)
    ax.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def make_plot(run_dir, output_dir, tau_s, output_name=None):
    run_dir = Path(run_dir)
    result = load_pickle(run_dir / "scenario_result.pkl")
    ego_log = first_policy_log(result, "ego_")
    target_log = event_policy_log(result)
    steps = ego_log.get("steps", [])
    if not steps:
        raise RuntimeError(f"No ego policy steps found in {run_dir / 'scenario_result.pkl'}")

    cmd_accel = step_series(steps, "accel_cmd")
    cmd_jerk = derivative_from_series(cmd_accel)
    raw_actual_accel = step_series(steps, "actual_accel")
    raw_actual_jerk = step_series(steps, "actual_jerk")
    filt_actual_accel, filt_actual_jerk = filtered_comfort_series(steps, tau_s=tau_s)

    event_time_s = safe_float(target_log.get("trigger_time_s"))

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), sharex=True, constrained_layout=True)

    ax = axes[0]
    plot_lines(ax, cmd_accel, "cmd accel", "#1f77b4", linewidth=1.5, alpha=0.9)
    plot_lines(ax, raw_actual_accel, "raw CARLA actual accel", "#8a8a8a", linewidth=1.0, alpha=0.42, linestyle="--")
    plot_lines(
        ax,
        filt_actual_accel,
        f"filtered actual accel, tau={tau_s:.2f}s ({abs_stats(filt_actual_accel)})",
        "#d62728",
        linewidth=2.1,
        alpha=0.95,
    )
    add_event_marker(ax, event_time_s)
    ax.set_ylabel("accel [m/s^2]")
    ax.set_title("Commanded vs. Actual Acceleration")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    plot_lines(ax, cmd_jerk, "cmd jerk = d(cmd accel)/dt", "#1f77b4", linewidth=1.5, alpha=0.9)
    plot_lines(ax, raw_actual_jerk, "raw CARLA actual jerk", "#8a8a8a", linewidth=1.0, alpha=0.42, linestyle="--")
    plot_lines(
        ax,
        filt_actual_jerk,
        f"filtered actual jerk, tau={tau_s:.2f}s ({abs_stats(filt_actual_jerk)})",
        "#d62728",
        linewidth=2.1,
        alpha=0.95,
    )
    add_event_marker(ax, event_time_s)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("jerk [m/s^3]")
    ax.set_title("Commanded vs. Actual Jerk")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="best", fontsize=9)

    if event_time_s is not None:
        fig.text(0.995, 0.01, "dotted line: target trigger", ha="right", va="bottom", fontsize=8, color="#444444")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"comfort_filter_tau_{tau_label(tau_s)}.png"
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ACC command and filtered comfort signals.")
    parser.add_argument("run_dir", help="Run directory containing scenario_result.pkl")
    parser.add_argument("--output-dir", default=None, help="Output directory for the plot PNG")
    parser.add_argument("--tau", type=float, default=DEFAULT_COMFORT_LOWPASS_TAU_S, help="Low-pass filter time constant [s]")
    parser.add_argument("--output-name", default=None, help="Output PNG filename")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "comfort_filter_plots"
    output_path = make_plot(run_dir, output_dir, args.tau, args.output_name)
    print(output_path)


if __name__ == "__main__":
    main()
