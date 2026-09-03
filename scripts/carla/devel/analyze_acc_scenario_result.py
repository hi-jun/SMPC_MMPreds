from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _load_summary(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    with path.open("r") as f:
        return json.load(f)


def _iter_ego_steps(summary):
    for policy_log in summary.get("policy_logs", {}).values():
        if isinstance(policy_log, dict) and policy_log.get("controller") == "acc_nair_smpc":
            for step in policy_log.get("steps", []):
                yield step


def _cutin_mode_for_relation(relation):
    if relation == "left_adjacent":
        return "RLC"
    if relation == "right_adjacent":
        return "LLC"
    return None


def _cutin_mode_from_debug(target):
    raw_indices = target.get("acc_mode_raw_indices", {}).get("cutin")
    if raw_indices:
        names = ["LK", "LLC", "RLC"]
        idx = int(raw_indices[0])
        if 0 <= idx < len(names):
            return names[idx]
    return _cutin_mode_for_relation(target.get("relation_to_ego_lane"))


def _mode_name_from_index(index):
    names = ["LK", "LLC", "RLC"]
    if isinstance(index, int) and 0 <= index < len(names):
        return names[index]
    return None


def _toward_ego(mode_d, relation):
    mode = _cutin_mode_for_relation(relation)
    if mode is None or mode not in mode_d:
        return None
    trend = mode_d[mode]
    start_abs = abs(float(trend["start"]))
    end_abs = abs(float(trend["end"]))
    return end_abs < start_abs - 0.05


def _extract_predictor_target_rows(summary):
    rows = []
    for step_idx, step in enumerate(_iter_ego_steps(summary)):
        debug = step.get("stdan_debug", {})
        time_s = step.get("time_s")
        for target in debug.get("processed_targets", []):
            relation = target.get("relation_to_ego_lane")
            raw_prob = target.get("raw_mode_prob", {})
            acc_prob = target.get("acc_mode_prob", {})
            mode_d = target.get("mode_d", {})
            cutin_mode = _cutin_mode_from_debug(target)
            row = {
                "step": step_idx,
                "time_s": time_s,
                "vehicle_id": target.get("vehicle_id"),
                "relation_to_ego_lane": relation,
                "raw_LK": raw_prob.get("LK"),
                "raw_LLC": raw_prob.get("LLC"),
                "raw_RLC": raw_prob.get("RLC"),
                "acc_lk": acc_prob.get("lk"),
                "acc_cutin": acc_prob.get("cutin"),
                "acc_cutout": acc_prob.get("cutout"),
                "signed_t_cross": target.get("signed_t_cross"),
                "branch_step": target.get("branch_step"),
                "ego_lane_start_idx": target.get("ego_lane_start_idx"),
                "ego_lane_end_idx": target.get("ego_lane_end_idx"),
                "cutin_mode": cutin_mode,
                "trajectory_suggested_cutin_mode": _mode_name_from_index(
                    target.get("trajectory_suggested_cutin_raw_idx")),
                "used_trajectory_aware_cutin_mapping": bool(
                    target.get("used_trajectory_aware_cutin_mapping", False)),
                "cutin_mode_toward_ego": (
                    cutin_mode in mode_d
                    and abs(float(mode_d[cutin_mode]["end"])) < abs(float(mode_d[cutin_mode]["start"])) - 0.05
                ),
            }
            if cutin_mode is not None and cutin_mode in mode_d:
                trend = mode_d[cutin_mode]
                row.update({
                    "cutin_d_start": trend.get("start"),
                    "cutin_d_end": trend.get("end"),
                    "cutin_d_min": trend.get("min"),
                    "cutin_d_max": trend.get("max"),
                    "cutin_d_delta": trend.get("delta"),
                })
            rows.append(row)
    return rows


def analyze_summary(summary):
    steps = list(_iter_ego_steps(summary))
    target_rows = _extract_predictor_target_rows(summary)
    cutin_rows = [
        row for row in target_rows
        if row["relation_to_ego_lane"] in ("left_adjacent", "right_adjacent")
    ]
    finite = lambda values: [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    ego_v = finite(step.get("ego_v") for step in steps)
    cmd_accel = finite(step.get("accel_cmd") for step in steps)
    min_gap = finite(step.get("min_gap") for step in steps)
    target_d = finite(step.get("target_d") for step in steps)
    cmd_accel_abs = [abs(value) for value in cmd_accel]
    cmd_jerk = _derivative_from_step_key(steps, "accel_cmd")
    cmd_jerk_abs = [abs(value) for value in cmd_jerk]
    actual_accel = finite(step.get("actual_accel") for step in steps)
    actual_jerk = finite(step.get("actual_jerk") for step in steps)
    if not actual_accel:
        actual_accel = _actual_accel_from_speed_steps(steps)
    if not actual_jerk:
        actual_jerk = _jerk_from_accel_steps(steps, actual_accel)
    # The first finite difference straddles the spawn transient, so its
    # acceleration and jerk are physically meaningless. Every run in the
    # 2026-09-02 sweep peaked on that sample (one hit 5324 m/s^3 against a
    # 78 m/s^3 99th percentile), so drop it before taking any statistic.
    actual_accel = actual_accel[1:]
    actual_jerk = actual_jerk[1:]
    actual_accel_abs = [abs(value) for value in actual_accel]
    actual_jerk_abs = [abs(value) for value in actual_jerk]

    toward = [row["cutin_mode_toward_ego"] for row in cutin_rows
              if row["cutin_mode_toward_ego"] is not None]
    acc_cutin = finite(row.get("acc_cutin") for row in cutin_rows)
    ego_lane_predicted = [
        row for row in target_rows
        if isinstance(row.get("ego_lane_start_idx"), int) and row["ego_lane_start_idx"] < 999
    ]
    return {
        "ran_successfully": bool(summary.get("ran_successfully")),
        "error": summary.get("error"),
        "num_controller_steps": len(steps),
        "num_predictor_target_rows": len(target_rows),
        "num_adjacent_target_rows": len(cutin_rows),
        "cutin_toward_ego_fraction": (
            sum(1 for value in toward if value) / len(toward) if toward else None
        ),
        "max_acc_cutin_prob": max(acc_cutin) if acc_cutin else None,
        "mean_acc_cutin_prob": sum(acc_cutin) / len(acc_cutin) if acc_cutin else None,
        "ego_lane_prediction_rows": len(ego_lane_predicted),
        "ego_v_min": min(ego_v) if ego_v else None,
        "ego_v_max": max(ego_v) if ego_v else None,
        "accel_min": min(cmd_accel) if cmd_accel else None,
        "accel_max": max(cmd_accel) if cmd_accel else None,
        "cmd_accel_mean": sum(cmd_accel) / len(cmd_accel) if cmd_accel else None,
        "cmd_accel_min": min(cmd_accel) if cmd_accel else None,
        "cmd_accel_max": max(cmd_accel) if cmd_accel else None,
        "cmd_accel_abs_mean": sum(cmd_accel_abs) / len(cmd_accel_abs) if cmd_accel_abs else None,
        "cmd_accel_abs_max": max(cmd_accel_abs) if cmd_accel_abs else None,
        "cmd_jerk_mean": sum(cmd_jerk) / len(cmd_jerk) if cmd_jerk else None,
        "cmd_jerk_min": min(cmd_jerk) if cmd_jerk else None,
        "cmd_jerk_max": max(cmd_jerk) if cmd_jerk else None,
        "cmd_jerk_abs_mean": sum(cmd_jerk_abs) / len(cmd_jerk_abs) if cmd_jerk_abs else None,
        "cmd_jerk_abs_max": max(cmd_jerk_abs) if cmd_jerk_abs else None,
        "actual_accel_mean": sum(actual_accel) / len(actual_accel) if actual_accel else None,
        "actual_accel_min": min(actual_accel) if actual_accel else None,
        "actual_accel_max": max(actual_accel) if actual_accel else None,
        "actual_accel_abs_mean": sum(actual_accel_abs) / len(actual_accel_abs) if actual_accel_abs else None,
        "actual_accel_abs_max": max(actual_accel_abs) if actual_accel_abs else None,
        "actual_jerk_mean": sum(actual_jerk) / len(actual_jerk) if actual_jerk else None,
        "actual_jerk_min": min(actual_jerk) if actual_jerk else None,
        "actual_jerk_max": max(actual_jerk) if actual_jerk else None,
        "actual_jerk_abs_mean": sum(actual_jerk_abs) / len(actual_jerk_abs) if actual_jerk_abs else None,
        "actual_jerk_abs_max": max(actual_jerk_abs) if actual_jerk_abs else None,
        "actual_jerk_abs_p99": _percentile(actual_jerk_abs, 99.0),
        "min_gap": min(min_gap) if min_gap else None,
        "target_d_min": min(target_d) if target_d else None,
        "target_d_max": max(target_d) if target_d else None,
    }


def _percentile(values, pct):
    # Differentiating CARLA's 20 Hz speed signal twice leaves the peak dominated
    # by numerical noise, so report a robust upper quantile alongside the max.
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _derivative_from_step_key(steps, key):
    values = []
    prev_time = None
    prev_value = None
    for step in steps:
        time_s = step.get("time_s")
        value = step.get(key)
        if not (
                isinstance(time_s, (int, float))
                and isinstance(value, (int, float))
                and math.isfinite(float(time_s))
                and math.isfinite(float(value))):
            continue
        time_s = float(time_s)
        value = float(value)
        if prev_time is not None:
            dt = time_s - prev_time
            if dt > 1.0e-6:
                values.append((value - prev_value) / dt)
        prev_time = time_s
        prev_value = value
    return values


def _actual_accel_from_speed_steps(steps):
    return _derivative_from_step_key(
        [
            {"time_s": step.get("time_s"), "ego_speed_for_actual_accel": step.get("ego_v")}
            for step in steps
        ],
        "ego_speed_for_actual_accel",
    )


def _jerk_from_accel_steps(steps, actual_accel):
    accel_steps = []
    accel_iter = iter(actual_accel)
    prev_valid_time = None
    for step in steps:
        time_s = step.get("time_s")
        speed = step.get("ego_v")
        if not (
                isinstance(time_s, (int, float))
                and isinstance(speed, (int, float))
                and math.isfinite(float(time_s))
                and math.isfinite(float(speed))):
            continue
        time_s = float(time_s)
        if prev_valid_time is None:
            prev_valid_time = time_s
            continue
        try:
            accel = next(accel_iter)
        except StopIteration:
            break
        accel_steps.append({"time_s": time_s, "actual_accel": accel})
        prev_valid_time = time_s
    return _derivative_from_step_key(accel_steps, "actual_accel")


def analyze_path(path):
    summary = _load_summary(path)
    metrics = analyze_summary(summary)
    metrics["path"] = str(path)
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    rows = [analyze_path(path) for path in args.paths]
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
