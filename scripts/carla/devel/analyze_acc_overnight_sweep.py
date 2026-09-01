from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SWEEP_ROOT = "results/acc_scenario_sweep/persistent_reuse_world_6run_validation_0701_1628"
DEFAULT_WARMUP_EXCLUDE_S = 0.5
DEFAULT_EVENT_WINDOW_S = 5.0
DEFAULT_DECEL_THRESHOLD = -0.4
DEFAULT_ACCEL_THRESHOLD = 0.4
DEFAULT_SAFE_D0 = 3.0
DEFAULT_TIME_HEADWAY = 1.3
DEFAULT_VEHICLE_LENGTH = 4.5
DEFAULT_COMFORT_LOWPASS_TAU_S = 0.25
DEFAULT_INFERRED_CONTACT_ACCEL_THRESHOLD = 40.0
SUSTAINED_SAMPLES = 5
MD_MAX_PARAM_ROWS = 40
MD_MAX_RUN_ROWS = 80


def is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite(values):
    return [float(value) for value in values if is_number(value)]


def mean(values):
    vals = finite(values)
    return sum(vals) / len(vals) if vals else None


def std(values):
    vals = finite(values)
    if len(vals) < 2:
        return 0.0 if vals else None
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((value - avg) ** 2 for value in vals) / (len(vals) - 1))


def min_or_none(values):
    vals = finite(values)
    return min(vals) if vals else None


def max_or_none(values):
    vals = finite(values)
    return max(vals) if vals else None


def mae(values):
    vals = finite(values)
    return sum(abs(value) for value in vals) / len(vals) if vals else None


def rmse(values):
    vals = finite(values)
    return math.sqrt(sum(value * value for value in vals) / len(vals)) if vals else None


def safe_float(value):
    return float(value) if is_number(value) else None


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def find_run_dirs(sweep_root):
    sweep_root = Path(sweep_root)
    return sorted({path.parent for path in sweep_root.rglob("summary.json")})


def first_actor_key(result, prefix):
    for key, value in result.items():
        if key.startswith(prefix) and isinstance(value, dict):
            return key
    return None


def actor_keys(result):
    return [
        key for key, value in result.items()
        if isinstance(value, dict) and "state_trajectory" in value
    ]


def ego_policy_log(summary, result):
    for log in summary.get("policy_logs", {}).values():
        if isinstance(log, dict) and log.get("controller") == "acc_nair_smpc":
            return log
    for key, value in result.items():
        if key.startswith("ego_") and isinstance(value, dict):
            log = value.get("policy_log", {})
            if isinstance(log, dict) and log.get("controller") == "acc_nair_smpc":
                return log
    return {}


def target_policy_log(summary, result):
    for key, log in summary.get("policy_logs", {}).items():
        if key.startswith("target_") and isinstance(log, dict):
            return log
    for key, value in result.items():
        if key.startswith("target_") and isinstance(value, dict):
            log = value.get("policy_log", {})
            if isinstance(log, dict):
                return log
    return {}


def scenario_kind(resolved, run_dir):
    if resolved.get("kind"):
        return resolved.get("kind")
    for part in reversed(run_dir.parts):
        if "cutout" in part:
            return "cutout"
        if "cutin" in part:
            return "cutin"
    return "unknown"


def clean_scenario_label(text):
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\d+[_-]+", "", text)
    text = re.sub(r"[_-]\d+$", "", text)
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def scenario_label(resolved, run_dir, sweep_root=None):
    for key in ("scenario_label", "scenario_name", "scenario_set"):
        label = clean_scenario_label(resolved.get(key))
        if label:
            return label
    scenario = resolved.get("scenario", {})
    if isinstance(scenario, dict):
        for key in ("label", "name", "scenario_label", "scenario_name"):
            label = clean_scenario_label(scenario.get(key))
            if label:
                return label

    parts = list(Path(run_dir).parts)
    if sweep_root is not None:
        try:
            parts = list(Path(run_dir).relative_to(sweep_root).parts)
        except ValueError:
            pass
    kind = scenario_kind(resolved, Path(run_dir))

    if len(parts) >= 3:
        label = clean_scenario_label(parts[-2])
        if label and not label.startswith("acc "):
            return label

    if parts:
        run_name_label = clean_scenario_label(parts[-1])
        if kind != "unknown" and run_name_label and run_name_label != kind:
            return run_name_label

    return kind


def ego_policy(resolved, run_dir):
    if resolved.get("ego_policy"):
        return resolved.get("ego_policy")
    parts = list(run_dir.parts)
    for part in reversed(parts):
        if part.startswith("acc_"):
            return part
    return "unknown"


def sweep_params(resolved):
    params = resolved.get("sweep_params", {})
    return params if isinstance(params, dict) else {}


def ego_nominal_speed(resolved):
    for vehicle in resolved.get("scenario", {}).get("vehicle_params", []):
        if vehicle.get("role") == "ego":
            return safe_float(vehicle.get("nominal_speed"))
    params = sweep_params(resolved)
    return safe_float(params.get("ego_speed")) or 12.0


def actor_length(result, actor_key):
    if actor_key is None or actor_key not in result:
        return DEFAULT_VEHICLE_LENGTH
    actor = result.get(actor_key, {})
    length = safe_float(actor.get("l_f"))
    rear = safe_float(actor.get("l_r"))
    if length is not None and rear is not None:
        return length + rear
    return DEFAULT_VEHICLE_LENGTH


def filtered_steps(steps, warmup_s):
    if not steps:
        return []
    first_time = safe_float(steps[0].get("time_s"))
    if first_time is None:
        return steps
    return [
        step for step in steps
        if safe_float(step.get("time_s")) is not None
        and float(step["time_s"]) >= first_time + warmup_s
    ]


def derivative_from_steps(steps, key):
    values = []
    prev_time = None
    prev_value = None
    for step in steps:
        time_s = safe_float(step.get("time_s"))
        value = safe_float(step.get(key))
        if time_s is None or value is None:
            continue
        if prev_time is not None:
            dt = time_s - prev_time
            if dt > 1.0e-6:
                values.append({"time_s": time_s, "value": (value - prev_value) / dt})
        prev_time = time_s
        prev_value = value
    return values


def series_from_step_key(steps, key, fallback_key=None):
    values = [
        {"time_s": safe_float(step.get("time_s")), "value": safe_float(step.get(key))}
        for step in steps
    ]
    values = [row for row in values if row["time_s"] is not None and row["value"] is not None]
    if values or fallback_key is None:
        return values
    return derivative_from_steps(steps, fallback_key)


def event_window_steps(steps, event_time, window_s):
    if event_time is None:
        return []
    return [
        step for step in steps
        if safe_float(step.get("time_s")) is not None
        and event_time - window_s <= float(step["time_s"]) <= event_time + window_s
    ]


def lane_key(record):
    if not isinstance(record, dict):
        return None
    road_id = record.get("road_id")
    lane_id = record.get("lane_id")
    if road_id is None or lane_id is None:
        return None
    return int(road_id), int(lane_id)


def lane_keys(actor_result):
    records = actor_result.get("lane_trajectory", [])
    return [lane_key(record) for record in records]


def most_common_lane_key(records, limit=None):
    keys = [lane_key(record) for record in (records[:limit] if limit else records)]
    keys = [key for key in keys if key is not None]
    if not keys:
        return None
    return Counter(keys).most_common(1)[0][0]


def lane_event_time(kind, result):
    ego_key = first_actor_key(result, "ego_")
    target_key = first_actor_key(result, "target_")
    if ego_key is None or target_key is None:
        return None
    ego_records = result[ego_key].get("lane_trajectory", [])
    target_records = result[target_key].get("lane_trajectory", [])
    if not ego_records or not target_records:
        return None
    ego_lane = most_common_lane_key(ego_records, limit=min(20, len(ego_records)))
    target_initial_lane = most_common_lane_key(target_records, limit=min(20, len(target_records)))
    if ego_lane is None or target_initial_lane is None:
        return None

    for record in target_records:
        key = lane_key(record)
        time_s = safe_float(record.get("time_s"))
        if key is None or time_s is None:
            continue
        if "cutout" in kind and key != target_initial_lane:
            return time_s
        if "cutin" in kind and key == ego_lane:
            return time_s
    return None


def lateral_event_time(kind, resolved, result):
    ego_key = first_actor_key(result, "ego_")
    target_key = first_actor_key(result, "target_")
    if ego_key is None or target_key is None:
        return None
    ego_state = result[ego_key].get("state_trajectory")
    target_state = result[target_key].get("state_trajectory")
    if ego_state is None or target_state is None or len(ego_state) == 0 or len(target_state) == 0:
        return None
    params = sweep_params(resolved)
    lane_width = safe_float(params.get("lane_width")) or 3.5
    ego_x = float(ego_state[0][1])
    left = ego_x - lane_width / 2.0
    right = ego_x + lane_width / 2.0
    for row in target_state:
        time_s = safe_float(row[0])
        x = safe_float(row[1])
        if time_s is None or x is None:
            continue
        in_ego_lane = left <= x <= right
        if "cutin" in kind and in_ego_lane:
            return time_s
        if "cutout" in kind and not in_ego_lane:
            return time_s
    return None


def event_anchor(kind, resolved, result, target_log):
    time_s = lane_event_time(kind, result)
    if time_s is not None:
        return time_s, "lane_id"
    trigger_time = safe_float(target_log.get("trigger_time_s"))
    if trigger_time is not None:
        return trigger_time, "trigger_fallback"
    time_s = lateral_event_time(kind, resolved, result)
    if time_s is not None:
        return time_s, "lateral_fallback"
    return None, "none"


def lowpass_filter_time_series(samples, tau_s):
    filtered = []
    prev_time = None
    prev_value = None
    for row in samples:
        time_s = safe_float(row.get("time_s"))
        value = safe_float(row.get("value"))
        if time_s is None or value is None:
            continue
        if prev_value is None or prev_time is None:
            filt = value
        else:
            dt = max(0.0, time_s - prev_time)
            alpha = dt / (float(tau_s) + dt) if dt > 0.0 else 0.0
            filt = prev_value + alpha * (value - prev_value)
        filtered.append({"time_s": time_s, "value": filt})
        prev_time = time_s
        prev_value = filt
    return filtered


def derivative_from_series(rows):
    values = []
    prev_time = None
    prev_value = None
    for row in rows:
        time_s = safe_float(row.get("time_s"))
        value = safe_float(row.get("value"))
        if time_s is None or value is None:
            continue
        if prev_time is not None:
            dt = time_s - prev_time
            if dt > 1.0e-6:
                values.append({"time_s": time_s, "value": (value - prev_value) / dt})
        prev_time = time_s
        prev_value = value
    return values


def sequence_value(row, index):
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


def command_accel_rows_from_steps(steps):
    rows = []
    for step in steps:
        time_s = safe_float(step.get("time_s"))
        accel = safe_float(step.get("accel_cmd"))
        if time_s is None or accel is None:
            continue
        rows.append({"time_s": time_s, "value": accel})
    return rows


def command_accel_rows_from_result(result, fallback_steps=None):
    ego_key = first_actor_key(result, "ego_")
    if ego_key is not None:
        ego = result.get(ego_key, {})
        states = ego.get("state_trajectory", [])
        inputs = ego.get("input_trajectory", [])
        if states is None:
            states = []
        if inputs is None:
            inputs = []
        rows = []
        for state, control in zip(states, inputs):
            time_s = safe_float(sequence_value(state, 0))
            accel = safe_float(sequence_value(control, 0))
            if time_s is None or accel is None:
                continue
            rows.append({"time_s": time_s, "value": accel})
        if rows:
            return rows
    return command_accel_rows_from_steps(fallback_steps or [])


def contact_free_rows(rows, contact_time_s=None):
    if contact_time_s is None:
        return list(rows)
    return [
        row for row in rows
        if safe_float(row.get("time_s")) is not None
        and float(row["time_s"]) < contact_time_s
    ]


def filtered_comfort_series(steps, contact_time_s=None, tau_s=DEFAULT_COMFORT_LOWPASS_TAU_S):
    speed_samples = []
    for step in steps:
        time_s = safe_float(step.get("time_s"))
        speed = safe_float(step.get("ego_v"))
        if time_s is None or speed is None:
            continue
        if contact_time_s is not None and time_s >= contact_time_s:
            continue
        speed_samples.append({"time_s": time_s, "value": speed})
    filtered_speed = lowpass_filter_time_series(speed_samples, tau_s)
    accel = derivative_from_series(filtered_speed)
    jerk = derivative_from_series(accel)
    return accel, jerk


def command_comfort_series(accel_rows, contact_time_s=None):
    accel = contact_free_rows(accel_rows, contact_time_s=contact_time_s)
    jerk = derivative_from_series(accel)
    return accel, jerk


def ego_collision_events(result):
    ego_key = first_actor_key(result, "ego_")
    collision_log = result.get("_collision_log", [])
    if not isinstance(collision_log, list):
        collision_log = []
    if ego_key is None:
        return []
    return [
        event for event in collision_log
        if isinstance(event, dict)
        and (
            event.get("actor_key") == ego_key
            or event.get("other_actor_role") == "ego"
        )
    ]


def first_event_time(events):
    times = [safe_float(event.get("time_s")) for event in events if isinstance(event, dict)]
    times = [time_s for time_s in times if time_s is not None]
    return min(times) if times else None


def inferred_contact_events_from_steps(steps, threshold=DEFAULT_INFERRED_CONTACT_ACCEL_THRESHOLD):
    events = []
    prev_time = None
    prev_speed = None
    for step in steps:
        time_s = safe_float(step.get("time_s"))
        speed = safe_float(step.get("ego_v"))
        if time_s is None or speed is None:
            continue
        if prev_time is not None:
            dt = time_s - prev_time
            if dt > 1.0e-6:
                accel = (speed - prev_speed) / dt
                if abs(accel) >= threshold:
                    events.append({
                        "time_s": time_s,
                        "source": "ego_speed_jump",
                        "estimated_accel": accel,
                    })
        prev_time = time_s
        prev_speed = speed
    return events


def contact_summary(result, steps):
    collision_log = result.get("_collision_log", [])
    if not isinstance(collision_log, list):
        collision_log = []
    collision_sensor_errors = result.get("_collision_sensor_errors", [])
    if not isinstance(collision_sensor_errors, list):
        collision_sensor_errors = []
    ego_events = ego_collision_events(result)
    inferred_events = inferred_contact_events_from_steps(steps)
    sensor_first = first_event_time(ego_events)
    inferred_first = first_event_time(inferred_events)
    contact_times = [time_s for time_s in (sensor_first, inferred_first) if time_s is not None]
    first_contact_time_s = min(contact_times) if contact_times else None
    collision_exclusion_used = (
        sensor_first is not None
        and first_contact_time_s is not None
        and abs(sensor_first - first_contact_time_s) < 1.0e-6
    )
    inferred_exclusion_used = (
        inferred_first is not None
        and first_contact_time_s is not None
        and abs(inferred_first - first_contact_time_s) < 1.0e-6
    )
    exclusion_sources = []
    if collision_exclusion_used:
        exclusion_sources.append("collision_sensor")
    if inferred_exclusion_used:
        exclusion_sources.append("inferred_speed_jump")
    return {
        "collision_count": len(collision_log),
        "ego_collision_count": len(ego_events),
        "inferred_contact_count": len(inferred_events),
        "collision_sensor_error_count": len(collision_sensor_errors),
        "first_collision_time_s": sensor_first,
        "first_inferred_contact_time_s": inferred_first,
        "first_contact_time_s": first_contact_time_s,
        "collision_log_available": bool(collision_log),
        "comfort_exclusion_source": "+".join(exclusion_sources),
        "comfort_exclusion_by_collision": collision_exclusion_used,
        "comfort_exclusion_by_inferred_contact": inferred_exclusion_used,
        "inferred_contact_detected": bool(inferred_events),
    }


def step_time_series_metric(steps, desired_speed, contact_time_s=None, comfort_accel_rows=None):
    ego_v = finite(step.get("ego_v") for step in steps)
    speed_errors = [value - desired_speed for value in ego_v]
    if comfort_accel_rows is None:
        comfort_accel_rows = command_accel_rows_from_steps(steps)
    actual_accel, actual_jerk = command_comfort_series(
        comfort_accel_rows,
        contact_time_s=contact_time_s,
    )
    accel_vals = [row["value"] for row in actual_accel]
    jerk_vals = [row["value"] for row in actual_jerk]
    accel_cmd = finite(step.get("accel_cmd") for step in steps)
    solve_times = finite(step.get("solve_time") for step in steps)
    predictor_times = finite(step.get("predictor_time") for step in steps)
    prediction_modes = finite(step.get("prediction_num_modes") for step in steps)
    min_gaps = finite(step.get("min_gap") for step in steps)
    min_gap_violations = [max(0.0, -value) for value in min_gaps]
    return {
        "num_steps": len(steps),
        "mean_velocity": mean(ego_v),
        "speed_mae": mae(speed_errors),
        "speed_rmse": rmse(speed_errors),
        "actual_accel_abs_mean": mean(abs(value) for value in accel_vals),
        "actual_accel_abs_peak": max_or_none(abs(value) for value in accel_vals),
        "actual_accel_min": min_or_none(accel_vals),
        "actual_accel_max": max_or_none(accel_vals),
        "actual_jerk_abs_mean": mean(abs(value) for value in jerk_vals),
        "actual_jerk_abs_peak": max_or_none(abs(value) for value in jerk_vals),
        "actual_jerk_min": min_or_none(jerk_vals),
        "actual_jerk_max": max_or_none(jerk_vals),
        "accel_cmd_min": min_or_none(accel_cmd),
        "accel_cmd_max": max_or_none(accel_cmd),
        "accel_cmd_mean": mean(accel_cmd),
        "solve_time_mean": mean(solve_times),
        "solve_time_max": max_or_none(solve_times),
        "predictor_time_mean": mean(predictor_times),
        "predictor_time_max": max_or_none(predictor_times),
        "prediction_num_modes_mean": mean(prediction_modes),
        "controller_min_gap_min": min_or_none(min_gaps),
        "controller_min_gap_mean": mean(min_gaps),
        "controller_min_gap_violation_mean": mean(min_gap_violations),
        "controller_min_gap_violation_peak": max_or_none(min_gap_violations),
    }


def nearest_state_index(times, time_s, start_idx=0):
    best_idx = None
    best_dt = None
    for idx in range(start_idx, len(times)):
        dt = abs(times[idx] - time_s)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_idx = idx
        if best_dt is not None and times[idx] > time_s and dt > best_dt:
            break
    return best_idx


def filtered_time_rows(rows, warmup_s):
    if not rows:
        return []
    first_time = safe_float(rows[0].get("time_s"))
    if first_time is None:
        return rows
    return [
        row for row in rows
        if safe_float(row.get("time_s")) is not None
        and float(row["time_s"]) >= first_time + warmup_s
    ]


def event_window_rows(rows, event_time, window_s):
    if event_time is None:
        return []
    return [
        row for row in rows
        if safe_float(row.get("time_s")) is not None
        and event_time - window_s <= float(row["time_s"]) <= event_time + window_s
    ]


def free_speed_reference_rows_from_ego_states(ego_states, free_speed):
    rows = []
    if ego_states is None:
        return rows
    for ego_state in ego_states:
        time_s = safe_float(ego_state[0])
        ego_v = safe_float(ego_state[4])
        if time_s is None or ego_v is None:
            continue
        rows.append({
            "time_s": time_s,
            "ego_speed": ego_v,
            "desired_speed": free_speed,
            "speed_error": ego_v - free_speed,
            "lead_actor": None,
            "lead_speed": None,
        })
    return rows


def lead_aware_speed_reference_rows(result, free_speed):
    ego_key = first_actor_key(result, "ego_")
    if ego_key is None:
        return [], "missing_state_trajectory"
    ego_result = result.get(ego_key, {})
    ego_states = ego_result.get("state_trajectory")
    ego_lanes = ego_result.get("lane_trajectory", [])
    if ego_states is None or len(ego_states) == 0:
        return [], "missing_state_trajectory"

    other_keys = [key for key in actor_keys(result) if key != ego_key]
    fallback_rows = free_speed_reference_rows_from_ego_states(ego_states, free_speed)
    if not ego_lanes or any(not result[key].get("lane_trajectory") for key in other_keys):
        return fallback_rows, "free_speed_fallback_missing_lane_trajectory"
    if not other_keys:
        return fallback_rows, "free_speed_no_other_actors"

    actor_times = {}
    for key in other_keys:
        states = result[key].get("state_trajectory", [])
        if len(states) == 0:
            return fallback_rows, "free_speed_fallback_missing_actor_state"
        actor_times[key] = [float(row[0]) for row in states]

    rows = []
    last_indices = {key: 0 for key in other_keys}
    for idx, ego_state in enumerate(ego_states):
        if idx >= len(ego_lanes):
            break
        time_s = safe_float(ego_state[0])
        ego_lane = lane_key(ego_lanes[idx])
        ego_v = safe_float(ego_state[4])
        if time_s is None or ego_lane is None or ego_v is None:
            continue
        ego_x = float(ego_state[1])
        ego_y = float(ego_state[2])
        ego_yaw = float(ego_state[3])
        best_gap = None
        best_actor = None
        best_speed = None
        for key in other_keys:
            states = result[key].get("state_trajectory", [])
            lanes = result[key].get("lane_trajectory", [])
            if len(states) == 0 or not lanes:
                continue
            state_idx = nearest_state_index(actor_times[key], time_s, start_idx=last_indices[key])
            if state_idx is None or state_idx >= len(lanes):
                continue
            last_indices[key] = state_idx
            if lane_key(lanes[state_idx]) != ego_lane:
                continue
            actor_state = states[state_idx]
            dx = float(actor_state[1]) - ego_x
            dy = float(actor_state[2]) - ego_y
            s_rel = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
            if s_rel <= 0.0:
                continue
            gap = s_rel - actor_length(result, key)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_actor = key
                best_speed = safe_float(actor_state[4])
        desired_speed = min(free_speed, best_speed) if best_speed is not None else free_speed
        rows.append({
            "time_s": time_s,
            "ego_speed": ego_v,
            "desired_speed": desired_speed,
            "speed_error": ego_v - desired_speed,
            "lead_actor": best_actor,
            "lead_speed": best_speed,
        })
    return rows, "lane_trajectory"


def lead_aware_speed_metrics(rows):
    errors = [row.get("speed_error") for row in rows]
    lead_rows = [row for row in rows if row.get("lead_actor") is not None]
    return {
        "lead_aware_speed_mae": mae(errors),
        "lead_aware_speed_rmse": rmse(errors),
        "lead_aware_speed_num_samples": len(finite(errors)),
        "lead_aware_speed_lead_fraction": (
            len(lead_rows) / len(rows) if rows else None
        ),
    }


def actual_gap_series(result):
    ego_key = first_actor_key(result, "ego_")
    if ego_key is None:
        return [], "missing_lane_trajectory"
    ego_result = result.get(ego_key, {})
    ego_states = ego_result.get("state_trajectory")
    ego_lanes = ego_result.get("lane_trajectory", [])
    if ego_states is None or len(ego_states) == 0 or not ego_lanes:
        return [], "missing_lane_trajectory"

    other_keys = [key for key in actor_keys(result) if key != ego_key]
    if not other_keys:
        return [], "missing_lane_trajectory"
    for key in other_keys:
        if not result[key].get("lane_trajectory"):
            return [], "missing_lane_trajectory"

    actor_times = {}
    for key in other_keys:
        actor_times[key] = [float(row[0]) for row in result[key].get("state_trajectory", [])]

    ego_length = actor_length(result, ego_key)
    rows = []
    last_indices = {key: 0 for key in other_keys}
    for idx, ego_state in enumerate(ego_states):
        if idx >= len(ego_lanes):
            break
        time_s = safe_float(ego_state[0])
        ego_lane = lane_key(ego_lanes[idx])
        ego_v = safe_float(ego_state[4])
        if time_s is None or ego_lane is None or ego_v is None:
            continue
        ego_x = float(ego_state[1])
        ego_y = float(ego_state[2])
        ego_yaw = float(ego_state[3])
        best_gap = None
        best_actor = None
        for key in other_keys:
            states = result[key].get("state_trajectory", [])
            lanes = result[key].get("lane_trajectory", [])
            if len(states) == 0 or not lanes:
                continue
            state_idx = nearest_state_index(actor_times[key], time_s, start_idx=last_indices[key])
            if state_idx is None or state_idx >= len(lanes):
                continue
            last_indices[key] = state_idx
            if lane_key(lanes[state_idx]) != ego_lane:
                continue
            actor_state = states[state_idx]
            dx = float(actor_state[1]) - ego_x
            dy = float(actor_state[2]) - ego_y
            s_rel = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
            if s_rel <= 0.0:
                continue
            gap = s_rel - actor_length(result, key)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_actor = key
        if best_gap is None:
            continue
        safe_gap = ego_length + DEFAULT_SAFE_D0 + DEFAULT_TIME_HEADWAY * ego_v
        rows.append({
            "time_s": time_s,
            "gap_error": best_gap - safe_gap,
            "actual_gap": best_gap,
            "safe_gap": safe_gap,
            "lead_actor": best_actor,
        })
    return rows, "lane_trajectory"


def gap_metrics(gap_rows):
    errors = finite(row.get("gap_error") for row in gap_rows)
    violations = [max(0.0, -value) for value in errors]
    integral = 0.0
    prev_time = None
    for row in gap_rows:
        time_s = safe_float(row.get("time_s"))
        error = safe_float(row.get("gap_error"))
        if time_s is None or error is None:
            continue
        if prev_time is not None:
            integral += max(0.0, -error) * max(0.0, time_s - prev_time)
        prev_time = time_s
    return {
        "gap_mae": mae(errors),
        "gap_rmse": rmse(errors),
        "gap_violation_mean": mean(violations),
        "gap_violation_peak": max_or_none(violations),
        "gap_violation_integral": integral if errors else None,
    }


def sustained_time(steps, key, predicate, samples=SUSTAINED_SAMPLES):
    count = 0
    first_time = None
    for step in steps:
        value = safe_float(step.get(key))
        time_s = safe_float(step.get("time_s"))
        if value is not None and time_s is not None and predicate(value):
            if count == 0:
                first_time = time_s
            count += 1
            if count >= samples:
                return first_time
        else:
            count = 0
            first_time = None
    return None


def gap_recovery_time(gap_rows, event_time):
    if event_time is None:
        return None, None
    rows = [
        row for row in gap_rows
        if safe_float(row.get("time_s")) is not None
        and safe_float(row.get("gap_error")) is not None
        and float(row["time_s"]) >= event_time
    ]
    violation_idx = None
    for idx, row in enumerate(rows):
        if float(row["gap_error"]) < 0.0:
            violation_idx = idx
            break
    if violation_idx is None:
        return True, 0.0
    violation_time = float(rows[violation_idx]["time_s"])
    for idx in range(violation_idx + 1, len(rows)):
        start_time = float(rows[idx]["time_s"])
        if float(rows[idx]["gap_error"]) < 0.0:
            continue
        window = [
            row for row in rows[idx:]
            if float(row["time_s"]) <= start_time + 1.0 + 1.0e-9
        ]
        if window and window[-1]["time_s"] >= start_time + 1.0 - 1.0e-9:
            if all(float(row["gap_error"]) >= 0.0 for row in window):
                return True, start_time - violation_time
    return False, None


def prefix_dict(prefix, values):
    return {f"{prefix}_{key}": value for key, value in values.items()}


def analyze_run(run_dir, sweep_root, event_window_s, warmup_exclude_s):
    run_dir = Path(run_dir)
    summary = read_json(run_dir / "summary.json")
    resolved = read_json(run_dir / "resolved_config.json")
    result = load_pickle(run_dir / "scenario_result.pkl")
    kind = scenario_kind(resolved, run_dir)
    label = scenario_label(resolved, run_dir, sweep_root)
    policy = ego_policy(resolved, run_dir)
    target_log = target_policy_log(summary, result)
    ego_log = ego_policy_log(summary, result)
    steps = ego_log.get("steps", [])
    steps = steps if isinstance(steps, list) else []
    trimmed_steps = filtered_steps(steps, warmup_exclude_s)
    contact = contact_summary(result, trimmed_steps)
    event_time, event_source = event_anchor(kind, resolved, result, target_log)
    event_steps = event_window_steps(trimmed_steps, event_time, event_window_s)
    desired_speed = ego_nominal_speed(resolved)
    gap_rows, gap_source = actual_gap_series(result)
    event_gap_rows = event_window_rows(gap_rows, event_time, event_window_s)
    speed_reference_rows, speed_reference_source = lead_aware_speed_reference_rows(result, desired_speed)
    trimmed_speed_reference_rows = filtered_time_rows(speed_reference_rows, warmup_exclude_s)
    event_speed_reference_rows = event_window_rows(trimmed_speed_reference_rows, event_time, event_window_s)
    comfort_accel_rows = command_accel_rows_from_result(result, steps)
    trimmed_comfort_accel_rows = filtered_time_rows(comfort_accel_rows, warmup_exclude_s)
    event_comfort_accel_rows = event_window_rows(trimmed_comfort_accel_rows, event_time, event_window_s)

    full_step_metrics = step_time_series_metric(
        trimmed_steps,
        desired_speed,
        contact_time_s=contact["first_contact_time_s"],
        comfort_accel_rows=trimmed_comfort_accel_rows,
    )
    event_step_metrics = step_time_series_metric(
        event_steps,
        desired_speed,
        contact_time_s=contact["first_contact_time_s"],
        comfort_accel_rows=event_comfort_accel_rows,
    )
    full_gap_metrics = gap_metrics(gap_rows)
    event_gap_metrics = gap_metrics(event_gap_rows)
    full_speed_reference_metrics = lead_aware_speed_metrics(trimmed_speed_reference_rows)
    event_speed_reference_metrics = lead_aware_speed_metrics(event_speed_reference_rows)
    event_available = event_time is not None
    cutin_decel_time = None
    cutout_accel_time = None
    if event_available:
        if "cutin" in kind:
            cutin_decel_time = sustained_time(
                event_steps,
                "accel_cmd",
                lambda value: value <= DEFAULT_DECEL_THRESHOLD,
            )
        if "cutout" in kind:
            cutout_accel_time = sustained_time(
                event_steps,
                "accel_cmd",
                lambda value: value >= DEFAULT_ACCEL_THRESHOLD,
            )
    gap_recovered, recovery_s = gap_recovery_time(gap_rows, event_time)

    params = sweep_params(resolved)
    row = {
        "run_path": str(run_dir),
        "run_name": run_dir.name,
        "relative_run_path": str(run_dir.relative_to(sweep_root)),
        "kind": kind,
        "scenario_label": label,
        "ego_policy": policy,
        "ran_successfully": bool(summary.get("ran_successfully")),
        "error": summary.get("error"),
        "event_available": bool(event_available),
        "event_time_s": event_time,
        "event_anchor_source": event_source,
        "event_window_s": event_window_s,
        "desired_speed": desired_speed,
        "speed_reference_source": speed_reference_source,
        "gap_source": gap_source,
        "collision_log_available": contact["collision_log_available"],
        "collision_count": contact["collision_count"],
        "ego_collision_count": contact["ego_collision_count"],
        "inferred_contact_count": contact["inferred_contact_count"],
        "collision_sensor_error_count": contact["collision_sensor_error_count"],
        "first_collision_time_s": contact["first_collision_time_s"],
        "first_inferred_contact_time_s": contact["first_inferred_contact_time_s"],
        "first_contact_time_s": contact["first_contact_time_s"],
        "comfort_exclusion_source": contact["comfort_exclusion_source"],
        "comfort_exclusion_by_collision": contact["comfort_exclusion_by_collision"],
        "comfort_exclusion_by_inferred_contact": contact["comfort_exclusion_by_inferred_contact"],
        "inferred_contact_detected": contact["inferred_contact_detected"],
        "lane_change_started": target_log.get("lane_change_started", target_log.get("cutout_started")),
        "lane_change_completed": target_log.get("lane_change_completed", target_log.get("cutout_completed")),
        "target_initial_lane_id": target_log.get("target_initial_lane_id", target_log.get("source_lane_id")),
        "target_final_lane_id": target_log.get("target_final_lane_id", target_log.get("target_lane_id")),
        "cutin_decel_start_rel_s": (
            cutin_decel_time - event_time
            if cutin_decel_time is not None and event_time is not None
            else None
        ),
        "cutin_decel_lead_s": (
            event_time - cutin_decel_time
            if cutin_decel_time is not None and event_time is not None
            else None
        ),
        "cutout_accel_start_rel_s": (
            cutout_accel_time - event_time
            if cutout_accel_time is not None and event_time is not None
            else None
        ),
        "cutout_accel_lead_s": (
            event_time - cutout_accel_time
            if cutout_accel_time is not None and event_time is not None
            else None
        ),
        "gap_recovered": gap_recovered,
        "gap_recovery_time_s": recovery_s,
    }
    row.update({f"sweep_{key}": value for key, value in params.items()})
    row.update(prefix_dict("full", full_step_metrics))
    row.update(prefix_dict("event", event_step_metrics))
    row.update(prefix_dict("full", full_speed_reference_metrics))
    row.update(prefix_dict("event", event_speed_reference_metrics))
    row.update(prefix_dict("full", full_gap_metrics))
    row.update(prefix_dict("event", event_gap_metrics))
    return row


def completion_report(run_dirs, sweep_root):
    rows = []
    reason_counts = Counter()
    grouped = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0, "failure_reasons": Counter()})
    for run_dir in run_dirs:
        summary = read_json(run_dir / "summary.json")
        resolved = read_json(run_dir / "resolved_config.json")
        kind = scenario_kind(resolved, run_dir)
        label = scenario_label(resolved, run_dir, sweep_root)
        policy = ego_policy(resolved, run_dir)
        ok = bool(summary.get("ran_successfully"))
        error = summary.get("error")
        reason = "success" if ok else str(error or "unknown_failure").split("\n")[0]
        if not ok:
            reason_counts[reason] += 1
        key = (policy, label, kind)
        grouped[key]["total"] += 1
        grouped[key]["success"] += int(ok)
        grouped[key]["fail"] += int(not ok)
        if not ok:
            grouped[key]["failure_reasons"][reason] += 1
        rows.append({
            "run_path": str(run_dir),
            "relative_run_path": str(run_dir.relative_to(sweep_root)),
            "ego_policy": policy,
            "kind": kind,
            "scenario_label": label,
            "ran_successfully": ok,
            "error": error,
        })
    group_rows = []
    for (policy, label, kind), data in sorted(grouped.items()):
        group_rows.append({
            "ego_policy": policy,
            "kind": kind,
            "scenario_label": label,
            "total": data["total"],
            "success": data["success"],
            "fail": data["fail"],
            "success_rate": data["success"] / data["total"] if data["total"] else None,
            "failure_reasons": dict(data["failure_reasons"]),
        })
    total = len(rows)
    success = sum(1 for row in rows if row["ran_successfully"])
    return {
        "sweep_root": str(sweep_root),
        "total": total,
        "success": success,
        "fail": total - success,
        "success_rate": success / total if total else None,
        "valid_for_performance": bool(total and success == total),
        "failure_reasons": dict(reason_counts),
        "groups": group_rows,
        "runs": rows,
    }


SUMMARY_METRICS = [
    "full_mean_velocity",
    "full_speed_mae",
    "full_speed_rmse",
    "full_lead_aware_speed_mae",
    "full_lead_aware_speed_rmse",
    "full_lead_aware_speed_num_samples",
    "full_lead_aware_speed_lead_fraction",
    "full_actual_accel_abs_peak",
    "full_actual_jerk_abs_mean",
    "full_actual_jerk_abs_peak",
    "full_gap_mae",
    "full_gap_rmse",
    "full_gap_violation_mean",
    "full_gap_violation_peak",
    "full_controller_min_gap_min",
    "full_controller_min_gap_violation_mean",
    "event_mean_velocity",
    "event_speed_mae",
    "event_speed_rmse",
    "event_lead_aware_speed_mae",
    "event_lead_aware_speed_rmse",
    "event_lead_aware_speed_num_samples",
    "event_lead_aware_speed_lead_fraction",
    "event_actual_accel_abs_peak",
    "event_actual_jerk_abs_mean",
    "event_actual_jerk_abs_peak",
    "event_gap_violation_mean",
    "event_gap_violation_peak",
    "cutin_decel_start_rel_s",
    "cutin_decel_lead_s",
    "cutout_accel_start_rel_s",
    "cutout_accel_lead_s",
    "gap_recovery_time_s",
    "full_solve_time_mean",
    "full_predictor_time_mean",
]


def aggregate_rows(rows, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out = []
    for key_values, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        row = {key: value for key, value in zip(group_keys, key_values)}
        row["num_runs"] = len(group)
        row["num_success"] = sum(1 for item in group if item.get("ran_successfully"))
        row["num_event_available"] = sum(1 for item in group if item.get("event_available"))
        row["num_lane_change_started"] = sum(1 for item in group if item.get("lane_change_started") is True)
        row["num_lane_change_completed"] = sum(1 for item in group if item.get("lane_change_completed") is True)
        row["num_runs_with_collision"] = sum(
            1 for item in group
            if int(item.get("collision_count") or 0) > 0
        )
        row["num_runs_with_comfort_exclusion"] = sum(
            1 for item in group
            if item.get("first_contact_time_s") is not None
        )
        row["num_runs_with_collision_comfort_exclusion"] = sum(
            1 for item in group
            if item.get("comfort_exclusion_by_collision") is True
        )
        row["num_runs_with_inferred_contact_exclusion"] = sum(
            1 for item in group
            if item.get("comfort_exclusion_by_inferred_contact") is True
        )
        row["num_runs_with_inferred_contact"] = sum(
            1 for item in group
            if int(item.get("inferred_contact_count") or 0) > 0
        )
        row["num_collision_events"] = sum(int(item.get("collision_count") or 0) for item in group)
        row["num_ego_collision_events"] = sum(int(item.get("ego_collision_count") or 0) for item in group)
        row["num_inferred_contact_events"] = sum(
            int(item.get("inferred_contact_count") or 0) for item in group
        )
        row["num_collision_sensor_errors"] = sum(
            int(item.get("collision_sensor_error_count") or 0) for item in group
        )
        for metric in SUMMARY_METRICS:
            values = [item.get(metric) for item in group]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = std(values)
        out.append(row)
    return out


def sweep_values(rows):
    values = defaultdict(set)
    for row in rows:
        for key, value in row.items():
            if key.startswith("sweep_"):
                values[key].add(value)
    return [
        {
            "parameter": key.replace("sweep_", "", 1),
            "values": ",".join(str(value) for value in sorted(vals, key=lambda item: str(item))),
            "num_values": len(vals),
        }
        for key, vals in sorted(values.items())
    ]


def parameter_group_keys(rows):
    keys = ["scenario_label", "kind", "ego_policy"]
    sweep_keys = sorted({key for row in rows for key in row if key.startswith("sweep_")})
    return keys + sweep_keys


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def markdown_cell(value):
    text = format_value(value)
    return text.replace("\n", " ").replace("|", "\\|")


def column_key(column):
    return column[0] if isinstance(column, tuple) else column


def column_label(column):
    return column[1] if isinstance(column, tuple) else column


def markdown_table(rows, columns):
    if not rows:
        return "_No rows._"
    lines = []
    lines.append("| " + " | ".join(markdown_cell(column_label(column)) for column in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(row.get(column_key(column))) for column in columns) + " |")
    return "\n".join(lines)


def report_group_rows(report):
    rows = []
    for group in report.get("groups", []):
        rows.append({
            "kind": group.get("kind"),
            "scenario_label": group.get("scenario_label", group.get("kind")),
            "ego_policy": group.get("ego_policy"),
            "total": group.get("total"),
            "success": group.get("success"),
            "fail": group.get("fail"),
            "success_rate": group.get("success_rate"),
            "top_failure": top_failure(group.get("failure_reasons", {})),
        })
    return rows


def top_failure(failure_reasons):
    if not failure_reasons:
        return ""
    reason, count = sorted(failure_reasons.items(), key=lambda item: (-item[1], item[0]))[0]
    return f"{count}x {reason}"


def short_policy_name(name):
    text = str(name)
    prefixes = [
        "acc_nair_smpc_",
        "acc_nair_",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def figure_markdown(figure_rows, figure_note):
    if not figure_rows:
        note = figure_note or "No figures generated."
        return ["_No figures generated._", "", f"_{note}_"]
    lines = []
    for row in figure_rows:
        lines.extend([
            f"### {row['title']}",
            "",
            f"![{row['title']}]({row['path']})",
            "",
        ])
        if row.get("description"):
            lines.extend([f"_{row['description']}_", ""])
    if figure_note:
        lines.extend([f"_{figure_note}_", ""])
    return lines


def load_matplotlib():
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return None, str(exc)
    return plt, None


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_completion_figure(plt, output_dir, report):
    rows = report_group_rows(report)
    if not rows:
        return None
    labels = [f"{row['scenario_label']} / {short_policy_name(row['ego_policy'])}" for row in rows]
    success = [float(row.get("success") or 0.0) for row in rows]
    fail = [float(row.get("fail") or 0.0) for row in rows]
    y_pos = list(range(len(rows)))
    height = max(4.0, min(12.0, 1.5 + 0.38 * len(rows)))
    fig, ax = plt.subplots(figsize=(10.5, height))
    ax.barh(y_pos, success, label="success", color="#2e7d32")
    ax.barh(y_pos, fail, left=success, label="fail", color="#b23b3b")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("runs")
    ax.set_title("Completion by Scenario and Controller")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    path = output_dir / "figures" / "completion_by_controller.png"
    save_figure(fig, path)
    plt.close(fig)
    return {
        "title": "Completion By Controller",
        "path": path.relative_to(output_dir).as_posix(),
        "description": "Stacked success/fail counts for each scenario-controller group.",
    }


def plot_metric_grid(plt, output_dir, rows, metrics, filename, title):
    if not rows:
        return None
    labels = [f"{row.get('kind')} / {short_policy_name(row.get('ego_policy'))}" for row in rows]
    valid_metrics = []
    for key, label in metrics:
        values = [safe_float(row.get(key)) for row in rows]
        if any(value is not None for value in values):
            valid_metrics.append((key, label, values))
    if not valid_metrics:
        return None

    n_axes = len(valid_metrics)
    height = max(3.2 * n_axes, min(18.0, 1.8 * n_axes + 0.28 * len(labels) * n_axes))
    fig, axes = plt.subplots(n_axes, 1, figsize=(10.5, height), squeeze=False)
    palette = ["#3266a8", "#9a5b1f", "#7f3c8d", "#2f7f72", "#b23b3b"]
    y_pos = list(range(len(labels)))
    for idx, (key, label, values) in enumerate(valid_metrics):
        ax = axes[idx][0]
        plot_values = [0.0 if value is None else value for value in values]
        colors = [palette[idx % len(palette)] if value is not None else "#cccccc" for value in values]
        ax.barh(y_pos, plot_values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.25)
        for y_idx, value in enumerate(values):
            if value is not None:
                ax.text(value, y_idx, f" {format_value(value)}", va="center", fontsize=8)
    fig.suptitle(title, fontsize=12)
    path = output_dir / "figures" / filename
    save_figure(fig, path)
    plt.close(fig)
    return {
        "title": title,
        "path": path.relative_to(output_dir).as_posix(),
        "description": "Gray bars indicate missing metrics.",
    }


def make_analysis_figures(output_dir, report, scenario_summary):
    output_dir = Path(output_dir)
    plt, error = load_matplotlib()
    if plt is None:
        return [], f"Figure generation skipped: matplotlib import failed: {error}"
    figures = []
    completion = plot_completion_figure(plt, output_dir, report)
    if completion:
        figures.append(completion)
    full_metrics = [
        ("full_speed_mae_mean", "Full Run Speed MAE"),
        ("full_actual_accel_abs_peak_mean", "Full Run Peak Absolute Acceleration"),
        ("full_actual_jerk_abs_peak_mean", "Full Run Peak Absolute Jerk"),
        ("full_solve_time_mean_mean", "Full Run Mean MPC Solve Time"),
        ("full_predictor_time_mean_mean", "Full Run Mean Predictor Inference Time"),
    ]
    event_metrics = [
        ("event_speed_mae_mean", "Event Window Speed MAE"),
        ("event_actual_accel_abs_peak_mean", "Event Window Peak Absolute Acceleration"),
        ("event_actual_jerk_abs_mean_mean", "Event Window Mean Absolute Jerk"),
        ("cutin_decel_start_rel_s_mean", "Cut-in Deceleration Start Time"),
        ("cutout_accel_start_rel_s_mean", "Cut-out Acceleration Start Time"),
        ("gap_recovery_time_s_mean", "Gap Recovery Time"),
    ]
    full = plot_metric_grid(
        plt,
        output_dir,
        scenario_summary,
        full_metrics,
        "controller_full_run_metrics.png",
        "Controller Full-Run Metrics",
    )
    if full:
        figures.append(full)
    event = plot_metric_grid(
        plt,
        output_dir,
        scenario_summary,
        event_metrics,
        "controller_event_window_metrics.png",
        "Controller Event-Window Metrics",
    )
    if event:
        figures.append(event)
    return figures, None


FULL_RUN_PUBLICATION_GROUPS = [
    (
        "Tracking Performance",
        [
            {
                "key": "full_gap_mae",
                "label": r"$e_d$ MAE [m]",
                "latex": r"$e_d$ MAE [m]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "full_lead_aware_speed_mae",
                "label": r"$e_v$ MAE [m/s]",
                "latex": r"$e_v$ MAE [m/s]",
                "direction": "min",
                "precision": 2,
            },
        ],
    ),
    (
        "Comfort",
        [
            {
                "key": "full_actual_jerk_abs_mean",
                "label": r"Avg $|j|$ [m/s$^3$]",
                "latex": r"Avg $|j|$ [m/s$^3$]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "full_actual_jerk_abs_peak",
                "label": r"Peak $|j|$ [m/s$^3$]",
                "latex": r"Peak $|j|$ [m/s$^3$]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "full_actual_accel_abs_peak",
                "label": r"Peak $|a|$ [m/s$^2$]",
                "latex": r"Peak $|a|$ [m/s$^2$]",
                "direction": "min",
                "precision": 2,
            },
        ],
    ),
    (
        "Safety",
        [
            {
                "key": "full_gap_violation_mean",
                "label": r"Mean Viol. [m]",
                "latex": r"Mean Viol. [m]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "full_gap_violation_peak",
                "label": r"Peak Viol. [m]",
                "latex": r"Peak Viol. [m]",
                "direction": "min",
                "precision": 2,
            },
        ],
    ),
    (
        "Solve Time",
        [
            {
                "key": "full_solve_time_mean",
                "label": r"Avg [s]",
                "latex": r"Avg [s]",
                "direction": "min",
                "precision": 3,
            },
        ],
    ),
]


EVENT_WINDOW_PUBLICATION_GROUPS = [
    (
        "Response Timing",
        [
            {
                "key": "cutin_decel_lead_s",
                "label": r"Decel Lead [s]",
                "latex": r"Decel Lead [s]",
                "direction": "max",
                "precision": 2,
            },
            {
                "key": "cutout_accel_lead_s",
                "label": r"Accel Lead [s]",
                "latex": r"Accel Lead [s]",
                "direction": "max",
                "precision": 2,
            },
        ],
    ),
    (
        "Comfort",
        [
            {
                "key": "event_actual_jerk_abs_mean",
                "label": r"Avg $|j|$ [m/s$^3$]",
                "latex": r"Avg $|j|$ [m/s$^3$]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "event_actual_jerk_abs_peak",
                "label": r"Peak $|j|$ [m/s$^3$]",
                "latex": r"Peak $|j|$ [m/s$^3$]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "event_actual_accel_abs_peak",
                "label": r"Peak $|a|$ [m/s$^2$]",
                "latex": r"Peak $|a|$ [m/s$^2$]",
                "direction": "min",
                "precision": 2,
            },
        ],
    ),
    (
        "Safety",
        [
            {
                "key": "event_gap_violation_mean",
                "label": r"Mean Viol. [m]",
                "latex": r"Mean Viol. [m]",
                "direction": "min",
                "precision": 2,
            },
            {
                "key": "event_gap_violation_peak",
                "label": r"Peak Viol. [m]",
                "latex": r"Peak Viol. [m]",
                "direction": "min",
                "precision": 2,
            },
        ],
    ),
]


PUBLICATION_TABLE_SPECS = [
    {
        "name": "full_run_controller_comparison",
        "title": "Full-Run Controller Comparison",
        "caption": "Controller comparison over the full scenario after the initial warm-up interval.",
        "label": "tab:full-run-controller-comparison",
        "groups": FULL_RUN_PUBLICATION_GROUPS,
    },
    {
        "name": "event_window_controller_comparison",
        "title": "Event-Window Controller Comparison",
        "caption": "Controller comparison in the event window around the cut-in or cut-out event.",
        "label": "tab:event-window-controller-comparison",
        "groups": EVENT_WINDOW_PUBLICATION_GROUPS,
    },
]


def publication_metrics(groups):
    return [metric for _, metrics in groups for metric in metrics]


def latex_escape(value):
    text = "" if value is None else str(value)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for src, dst in replacements:
        text = text.replace(src, dst)
    return text


def publication_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("scenario_label") or row.get("kind") or ""),
            str(row.get("ego_policy") or ""),
        ),
    )


def grouped_publication_rows(rows):
    grouped = []
    current_label = None
    current_rows = []
    for row in publication_rows(rows):
        label = row.get("scenario_label") or row.get("kind") or "unknown"
        if current_label is not None and label != current_label:
            grouped.append((current_label, current_rows))
            current_rows = []
        current_label = label
        current_rows.append(row)
    if current_label is not None:
        grouped.append((current_label, current_rows))
    return grouped


def publication_best_cells(rows, groups):
    rows = publication_rows(rows)
    best = set()
    scenario_to_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        scenario_to_indices[row.get("scenario_label") or row.get("kind") or "unknown"].append(idx)

    for indices in scenario_to_indices.values():
        for metric in publication_metrics(groups):
            values = []
            for idx in indices:
                value = safe_float(rows[idx].get(f"{metric['key']}_mean"))
                if value is not None:
                    values.append((idx, value))
            if not values:
                continue
            if metric.get("direction") == "max":
                best_value = max(value for _, value in values)
            else:
                best_value = min(value for _, value in values)
            for idx, value in values:
                if abs(value - best_value) <= 1.0e-12:
                    best.add((idx, metric["key"]))
    return best


def format_publication_cell(row, metric, for_latex=False, bold=False):
    avg = safe_float(row.get(f"{metric['key']}_mean"))
    if avg is None:
        return "--"
    sd = safe_float(row.get(f"{metric['key']}_std"))
    precision = int(metric.get("precision", 2))
    if sd is None:
        text = f"{avg:.{precision}f}"
    elif for_latex:
        text = f"{avg:.{precision}f} $\\pm$ {sd:.{precision}f}"
    else:
        text = f"{avg:.{precision}f} $\\pm$ {sd:.{precision}f}"
    if for_latex and bold:
        return rf"\textbf{{{text}}}"
    return text


def write_latex_publication_table(path, rows, spec):
    rows = publication_rows(rows)
    groups = spec["groups"]
    metrics = publication_metrics(groups)
    best_cells = publication_best_cells(rows, groups)
    align = "ll" + "c" * len(metrics)
    lines = [
        r"% Requires \usepackage{booktabs,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{latex_escape(spec['caption'])}}}",
        rf"\label{{{spec['label']}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
    ]

    header = ["Scenario", "Controller"]
    cmidrules = []
    col_idx = 3
    for group, group_metrics in groups:
        header.append(rf"\multicolumn{{{len(group_metrics)}}}{{c}}{{{latex_escape(group)}}}")
        cmidrules.append(rf"\cmidrule(lr){{{col_idx}-{col_idx + len(group_metrics) - 1}}}")
        col_idx += len(group_metrics)
    lines.append(" & ".join(header) + r" \\")
    lines.append(" ".join(cmidrules))
    lines.append(" & ".join(["", ""] + [metric["latex"] for metric in metrics]) + r" \\")
    lines.append(r"\midrule")

    global_idx = 0
    grouped_rows = grouped_publication_rows(rows)
    for group_idx, (scenario, group_rows) in enumerate(grouped_rows):
        for row_idx, row in enumerate(group_rows):
            if row_idx == 0 and len(group_rows) > 1:
                scenario_cell = rf"\multirow{{{len(group_rows)}}}{{*}}{{{latex_escape(scenario)}}}"
            elif row_idx == 0:
                scenario_cell = latex_escape(scenario)
            else:
                scenario_cell = ""
            cells = [
                scenario_cell,
                latex_escape(short_policy_name(row.get("ego_policy"))),
            ]
            for metric in metrics:
                cells.append(
                    format_publication_cell(
                        row,
                        metric,
                        for_latex=True,
                        bold=(global_idx, metric["key"]) in best_cells,
                    )
                )
            lines.append(" & ".join(cells) + r" \\")
            global_idx += 1
        if group_idx != len(grouped_rows) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def draw_publication_table(plt, rows, spec, output_base):
    import matplotlib.patches as patches

    rows = publication_rows(rows)
    groups = spec["groups"]
    metrics = publication_metrics(groups)
    best_cells = publication_best_cells(rows, groups)
    if not rows:
        return

    scenario_w = 1.9
    controller_w = 2.35
    metric_w = 1.33
    col_widths = [scenario_w, controller_w] + [metric_w for _ in metrics]
    x_edges = [0.0]
    for width in col_widths:
        x_edges.append(x_edges[-1] + width)
    total_w = x_edges[-1]

    group_header_h = 0.48
    metric_header_h = 0.68
    row_h = 0.54
    margin = 0.18
    total_h = margin * 2 + group_header_h + metric_header_h + row_h * len(rows)
    fig, ax = plt.subplots(figsize=(total_w, total_h))
    ax.set_xlim(0.0, total_w)
    ax.set_ylim(0.0, total_h)
    ax.axis("off")

    header_color = "#e9edf2"
    subheader_color = "#f5f7fa"
    scenario_color = "#f7f7f7"
    edge_color = "#3b3b3b"
    light_edge = "#9a9a9a"
    y_top = total_h - margin
    header_total_h = group_header_h + metric_header_h
    header_bottom = y_top - header_total_h

    for col_idx, title in enumerate(["Scenario", "Controller"]):
        x0 = x_edges[col_idx]
        width = col_widths[col_idx]
        ax.add_patch(patches.Rectangle(
            (x0, header_bottom),
            width,
            header_total_h,
            facecolor=header_color,
            edgecolor=edge_color,
            linewidth=0.8,
        ))
        ax.text(
            x0 + width / 2.0,
            header_bottom + header_total_h / 2.0,
            title,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    metric_start_col = 2
    for group, group_metrics in groups:
        start_col = metric_start_col
        end_col = metric_start_col + len(group_metrics)
        x0 = x_edges[start_col]
        width = x_edges[end_col] - x0
        ax.add_patch(patches.Rectangle(
            (x0, y_top - group_header_h),
            width,
            group_header_h,
            facecolor=header_color,
            edgecolor=edge_color,
            linewidth=0.8,
        ))
        ax.text(
            x0 + width / 2.0,
            y_top - group_header_h / 2.0,
            group,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        metric_start_col = end_col

    metric_col = 2
    for metric in metrics:
        x0 = x_edges[metric_col]
        width = col_widths[metric_col]
        ax.add_patch(patches.Rectangle(
            (x0, header_bottom),
            width,
            metric_header_h,
            facecolor=subheader_color,
            edgecolor=edge_color,
            linewidth=0.6,
        ))
        ax.text(
            x0 + width / 2.0,
            header_bottom + metric_header_h / 2.0,
            metric["label"],
            ha="center",
            va="center",
            fontsize=7,
        )
        metric_col += 1

    y = header_bottom
    global_idx = 0
    for scenario, group_rows in grouped_publication_rows(rows):
        group_h = row_h * len(group_rows)
        ax.add_patch(patches.Rectangle(
            (x_edges[0], y - group_h),
            scenario_w,
            group_h,
            facecolor=scenario_color,
            edgecolor=edge_color,
            linewidth=0.8,
        ))
        ax.text(
            x_edges[0] + scenario_w / 2.0,
            y - group_h / 2.0,
            scenario,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
        )
        for row in group_rows:
            row_bottom = y - row_h
            ax.add_patch(patches.Rectangle(
                (x_edges[1], row_bottom),
                controller_w,
                row_h,
                facecolor="white",
                edgecolor=light_edge,
                linewidth=0.5,
            ))
            ax.text(
                x_edges[1] + controller_w / 2.0,
                row_bottom + row_h / 2.0,
                short_policy_name(row.get("ego_policy")).replace("_", " "),
                ha="center",
                va="center",
                fontsize=7,
            )
            for metric_idx, metric in enumerate(metrics):
                col_idx = metric_idx + 2
                x0 = x_edges[col_idx]
                width = col_widths[col_idx]
                ax.add_patch(patches.Rectangle(
                    (x0, row_bottom),
                    width,
                    row_h,
                    facecolor="white",
                    edgecolor=light_edge,
                    linewidth=0.5,
                ))
                is_best = (global_idx, metric["key"]) in best_cells
                ax.text(
                    x0 + width / 2.0,
                    row_bottom + row_h / 2.0,
                    format_publication_cell(row, metric),
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    fontweight="bold" if is_best else "normal",
                )
            y = row_bottom
            global_idx += 1
        ax.plot([0.0, total_w], [y, y], color=edge_color, linewidth=0.8)

    ax.plot([0.0, total_w], [y_top, y_top], color=edge_color, linewidth=1.0)
    ax.plot([0.0, total_w], [header_bottom, header_bottom], color=edge_color, linewidth=0.9)
    ax.plot([0.0, total_w], [margin, margin], color=edge_color, linewidth=1.0)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_publication_tables(output_dir, publication_summary):
    output_dir = Path(output_dir)
    tables = []
    if not publication_summary:
        return [], "Publication tables skipped: no aggregated performance rows."

    table_dir = output_dir / "tables"
    for spec in PUBLICATION_TABLE_SPECS:
        tex_path = table_dir / f"{spec['name']}.tex"
        write_latex_publication_table(tex_path, publication_summary, spec)
        tables.append({
            "title": spec["title"],
            "tex": tex_path.relative_to(output_dir).as_posix(),
            "pdf": (table_dir / f"{spec['name']}.pdf").relative_to(output_dir).as_posix(),
            "png": (table_dir / f"{spec['name']}.png").relative_to(output_dir).as_posix(),
        })

    plt, error = load_matplotlib()
    if plt is None:
        return tables, f"Publication table PDF/PNG rendering skipped: matplotlib import failed: {error}"
    for spec in PUBLICATION_TABLE_SPECS:
        draw_publication_table(plt, publication_summary, spec, table_dir / spec["name"])
    return tables, None


def publication_table_markdown(table_rows, note):
    if not table_rows:
        return ["_No publication tables generated._", "", f"_{note}_" if note else ""]
    rows = []
    for row in table_rows:
        rows.append({
            "table": row["title"],
            "tex": row["tex"],
            "pdf": row["pdf"],
            "png": row["png"],
        })
    lines = [
        markdown_table(rows, ["table", "tex", "pdf", "png"]),
        "",
        "_LaTeX source tables require `booktabs` and `multirow` in the parent document._",
    ]
    if note:
        lines.extend(["", f"_{note}_"])
    return lines


def metric_definition_markdown(warmup_s, event_window_s):
    return [
        "- Full-run metrics discard the initial "
        f"{format_value(warmup_s)} s warm-up interval before aggregation.",
        "- Free-flow target speed `v_free` is the ego vehicle `nominal_speed`; "
        "if absent, the sweep `ego_speed`; if absent, 12.0 m/s.",
        "- Lead-aware desired speed is `v_des(t) = min(v_free, v_lead(t))` when "
        "a same-lane leading vehicle is found, and `v_des(t) = v_free` otherwise.",
        "- Speed tracking error is `e_v(t) = v_ego(t) - v_des(t)`; the publication "
        "table reports the MAE of this lead-aware error.",
        "- Comfort metrics follow the Nair-style closed-loop evaluation: "
        "acceleration is the controller command `input_trajectory[:, 0]` "
        "(`accel_cmd` fallback for legacy logs), and jerk is "
        "`diff(accel_cmd) / diff(time_s)`.",
        "- The jerk `time_s` is the ego `state_trajectory[:, 0]` timestamp recorded "
        "with each closed-loop input, not wall-clock time or a hard-coded controller "
        "period.",
        "- Columns with the historical `actual_accel`/`actual_jerk` names therefore "
        "contain command-based comfort values, not raw CARLA `get_acceleration()` "
        "or low-pass-filtered speed derivatives.",
        "- Samples at or after the first collision/contact time are excluded from "
        "comfort metrics. Collision sensor events are preferred when they provide "
        "the earliest ego contact; otherwise contact can be inferred from ego speed "
        "jumps above "
        f"{format_value(DEFAULT_INFERRED_CONTACT_ACCEL_THRESHOLD)} m/s^2.",
        "- Collision/contact counts are stored in `run_metrics.csv` and "
        "`publication_scenario_policy_summary.csv`, but omitted from publication tables.",
        "- The lead vehicle for gap metrics is the closest actor ahead of ego in "
        "the same lane, using `lane_trajectory` and actor `state_trajectory`.",
        "- Actual gap is the ego-frame longitudinal distance to that lead vehicle "
        "minus the lead vehicle length.",
        "- Safe distance is `d_safe(t) = L_ego + d0 + T_h v_ego(t)`, with "
        f"`d0 = {format_value(DEFAULT_SAFE_D0)} m` and "
        f"`T_h = {format_value(DEFAULT_TIME_HEADWAY)} s`.",
        "- Gap error is `e_d(t) = d_actual(t) - d_safe(t)`; gap violation is "
        "`max(0, -e_d(t))`.",
        "- Event-window metrics use samples within "
        f"`event_time +/- {format_value(event_window_s)} s`.",
        "- Event time is the lane-trajectory cut-in/cut-out time when available, "
        "then target trigger time, then lateral-position fallback.",
        "- Cut-in deceleration start is the first time in the event window where "
        f"`accel_cmd <= {format_value(DEFAULT_DECEL_THRESHOLD)}` persists for "
        f"{SUSTAINED_SAMPLES} consecutive samples.",
        "- Cut-out acceleration start is the first time in the event window where "
        f"`accel_cmd >= {format_value(DEFAULT_ACCEL_THRESHOLD)}` persists for "
        f"{SUSTAINED_SAMPLES} consecutive samples.",
        "- Response timing is reported as positive lead time: "
        "`lead = event_time - action_start_time`; positive values mean the controller "
        "acted before the cut-in/cut-out event.",
    ]


def write_summary_md(
    path,
    report,
    run_rows,
    scenario_summary,
    publication_summary,
    param_summary,
    sweep_value_rows,
    figure_rows,
    figure_note,
    publication_table_rows,
    publication_table_note,
    allow_partial,
    warmup_s,
    event_window_s,
):
    path = Path(path)
    key_run_cols = [
        ("relative_run_path", "run"),
        ("scenario_label", "scenario"),
        ("kind", "kind"),
        ("ego_policy", "controller"),
        ("ran_successfully", "ok"),
        ("lane_change_started", "LC start"),
        ("lane_change_completed", "LC done"),
        ("event_anchor_source", "anchor"),
        ("event_time_s", "event t"),
        ("comfort_exclusion_source", "comfort exclusion"),
        ("collision_count", "collision events"),
        ("inferred_contact_count", "inferred contacts"),
        ("full_mean_velocity", "mean v"),
        ("full_speed_mae", "speed MAE"),
        ("full_lead_aware_speed_mae", "lead-aware speed MAE"),
        ("full_actual_accel_abs_peak", "peak |acc|"),
        ("full_actual_jerk_abs_peak", "peak |jerk|"),
        ("full_solve_time_mean", "solve s"),
        ("full_predictor_time_mean", "pred s"),
    ]
    completion_cols = [
        ("scenario_label", "scenario"),
        ("kind", "kind"),
        ("ego_policy", "controller"),
        ("total", "runs"),
        ("success", "success"),
        ("fail", "fail"),
        ("success_rate", "success rate"),
        ("top_failure", "top failure"),
    ]
    full_summary_cols = [
        ("kind", "scenario"),
        ("ego_policy", "controller"),
        ("num_runs", "runs"),
        ("num_success", "success"),
        ("num_event_available", "events"),
        ("num_lane_change_started", "LC start"),
        ("num_lane_change_completed", "LC done"),
        ("full_mean_velocity_mean", "mean v"),
        ("full_speed_mae_mean", "speed MAE"),
        ("full_lead_aware_speed_mae_mean", "lead-aware speed MAE"),
        ("full_speed_rmse_mean", "speed RMSE"),
        ("full_actual_accel_abs_peak_mean", "peak |acc|"),
        ("full_actual_jerk_abs_mean_mean", "mean |jerk|"),
        ("full_actual_jerk_abs_peak_mean", "peak |jerk|"),
        ("full_solve_time_mean_mean", "solve s"),
        ("full_predictor_time_mean_mean", "pred s"),
    ]
    event_summary_cols = [
        ("kind", "scenario"),
        ("ego_policy", "controller"),
        ("num_event_available", "events"),
        ("event_mean_velocity_mean", "event mean v"),
        ("event_speed_mae_mean", "event speed MAE"),
        ("event_speed_rmse_mean", "event speed RMSE"),
        ("event_actual_accel_abs_peak_mean", "event peak |acc|"),
        ("event_actual_jerk_abs_mean_mean", "event mean |jerk|"),
        ("event_actual_jerk_abs_peak_mean", "event peak |jerk|"),
        ("event_gap_violation_mean_mean", "event gap viol"),
        ("event_gap_violation_peak_mean", "event peak gap viol"),
        ("cutin_decel_lead_s_mean", "cutin decel lead s"),
        ("cutout_accel_lead_s_mean", "cutout accel lead s"),
        ("gap_recovery_time_s_mean", "gap recovery s"),
    ]
    actual_gap_cols = [
        ("kind", "scenario"),
        ("ego_policy", "controller"),
        ("full_gap_mae_mean", "gap MAE"),
        ("full_gap_rmse_mean", "gap RMSE"),
        ("full_gap_violation_mean_mean", "gap viol mean"),
        ("full_gap_violation_peak_mean", "gap viol peak"),
        ("full_controller_min_gap_min_mean", "debug min_gap min"),
        ("full_controller_min_gap_violation_mean_mean", "debug min_gap viol"),
    ]
    comfort_exclusion_cols = [
        ("scenario_label", "scenario"),
        ("ego_policy", "controller"),
        ("num_runs", "runs"),
        ("num_runs_with_comfort_exclusion", "excluded runs"),
        ("num_runs_with_collision_comfort_exclusion", "collision-excluded"),
        ("num_runs_with_inferred_contact_exclusion", "inferred-excluded"),
        ("num_runs_with_inferred_contact", "inferred detected"),
        ("num_collision_events", "collision events"),
        ("num_ego_collision_events", "ego collision events"),
        ("num_inferred_contact_events", "inferred events"),
        ("num_collision_sensor_errors", "sensor setup errors"),
    ]
    param_preview_cols = [
        ("scenario_label", "scenario"),
        ("kind", "kind"),
        ("ego_policy", "controller"),
        ("sweep_trigger_distance", "trigger"),
        ("sweep_lane_change_distance", "LC dist"),
        ("sweep_target_lead_gap", "lead gap"),
        ("sweep_target_lead_speed_delta", "lead dv"),
        ("sweep_target_start_gap", "start gap"),
        ("num_runs", "runs"),
        ("num_success", "success"),
        ("num_event_available", "events"),
        ("full_mean_velocity_mean", "mean v"),
        ("full_lead_aware_speed_mae_mean", "lead-aware speed MAE"),
        ("cutin_decel_lead_s_mean", "cutin decel lead s"),
    ]
    reason_rows = [
        {"reason": reason, "count": count}
        for reason, count in sorted(report["failure_reasons"].items(), key=lambda item: (-item[1], item[0]))
    ]
    param_preview_rows = param_summary[:MD_MAX_PARAM_ROWS]
    omitted_param_rows = max(0, len(param_summary) - len(param_preview_rows))
    run_preview_rows = run_rows[:MD_MAX_RUN_ROWS]
    omitted_run_rows = max(0, len(run_rows) - len(run_preview_rows))
    performance_status = "enabled"
    if not report.get("valid_for_performance") and not allow_partial:
        performance_status = "skipped because not all runs completed; use `--allow-partial` for partial aggregation"
    content = [
        "# ACC Sweep Analysis",
        "",
        "## Completion",
        "",
        f"- sweep_root: `{report['sweep_root']}`",
        f"- total: {report['total']}",
        f"- success: {report['success']}",
        f"- fail: {report['fail']}",
        f"- valid_for_performance: {report['valid_for_performance']}",
        f"- allow_partial: {allow_partial}",
        f"- performance aggregation: {performance_status}",
        "",
        "## Failure Reasons",
        "",
        markdown_table(reason_rows, ["reason", "count"]),
        "",
        "## Completion By Controller",
        "",
        markdown_table(report_group_rows(report), completion_cols),
        "",
        "## Publication Tables",
        "",
        *publication_table_markdown(publication_table_rows, publication_table_note),
        "",
        "## Metric Definitions",
        "",
        *metric_definition_markdown(warmup_s, event_window_s),
        "",
        "## Comfort Exclusion Diagnostics",
        "",
        markdown_table(publication_summary, comfort_exclusion_cols),
        "",
        "## Figures",
        "",
        *figure_markdown(figure_rows, figure_note),
        "",
        "## Controller Comparison - Full Run",
        "",
        markdown_table(scenario_summary, full_summary_cols),
        "",
        "## Controller Comparison - Event Window",
        "",
        markdown_table(scenario_summary, event_summary_cols),
        "",
        "## Actual Gap And Debug Safety",
        "",
        markdown_table(scenario_summary, actual_gap_cols),
        "",
        "## Sweep Values",
        "",
        markdown_table(sweep_value_rows, [("parameter", "parameter"), ("values", "values"), ("num_values", "n")]),
        "",
        "## Parameter Combination Preview",
        "",
        markdown_table(param_preview_rows, param_preview_cols),
        "",
        f"_Showing {len(param_preview_rows)} of {len(param_summary)} parameter-summary rows._",
        "",
        f"_Omitted rows: {omitted_param_rows}. See `scenario_param_policy_summary.csv` for the full table._",
        "",
        "## Run Metrics Preview",
        "",
        markdown_table(run_preview_rows, key_run_cols),
        "",
        f"_Showing {len(run_preview_rows)} of {len(run_rows)} run rows._",
        "",
        f"_Omitted rows: {omitted_run_rows}. See `run_metrics.csv` for the full table._",
        "",
        "## Notes",
        "",
        "- Official actual gap metrics require `lane_trajectory`; legacy runs without it leave official gap fields empty.",
        "- Controller `min_gap` is reported only as debug columns with the `controller_min_gap` prefix.",
        "- Detailed raw tables: `run_metrics.csv`, `scenario_policy_summary.csv`, `scenario_param_policy_summary.csv`, `sweep_values.csv`, `completion_report.json`.",
    ]
    path.write_text("\n".join(content) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--event-window-s", type=float, default=DEFAULT_EVENT_WINDOW_S)
    parser.add_argument("--warmup-exclude-s", type=float, default=DEFAULT_WARMUP_EXCLUDE_S)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = find_run_dirs(sweep_root)
    report = completion_report(run_dirs, sweep_root)
    write_json(output_dir / "completion_report.json", report)

    run_rows = [
        analyze_run(run_dir, sweep_root, args.event_window_s, args.warmup_exclude_s)
        for run_dir in run_dirs
    ]
    write_csv(output_dir / "run_metrics.csv", run_rows)

    aggregate_source_rows = [
        row for row in run_rows
        if row.get("ran_successfully") and (report["valid_for_performance"] or args.allow_partial)
    ]
    scenario_summary = aggregate_rows(aggregate_source_rows, ["kind", "ego_policy"])
    publication_summary = aggregate_rows(aggregate_source_rows, ["scenario_label", "kind", "ego_policy"])
    param_summary = aggregate_rows(aggregate_source_rows, parameter_group_keys(aggregate_source_rows))
    sweep_value_rows = sweep_values(run_rows)
    write_csv(output_dir / "scenario_policy_summary.csv", scenario_summary)
    write_csv(output_dir / "publication_scenario_policy_summary.csv", publication_summary)
    write_csv(output_dir / "scenario_param_policy_summary.csv", param_summary)
    write_csv(output_dir / "sweep_values.csv", sweep_value_rows)
    figure_rows, figure_note = make_analysis_figures(output_dir, report, scenario_summary)
    publication_table_rows, publication_table_note = make_publication_tables(output_dir, publication_summary)
    write_summary_md(
        output_dir / "summary.md",
        report,
        run_rows,
        scenario_summary,
        publication_summary,
        param_summary,
        sweep_value_rows,
        figure_rows,
        figure_note,
        publication_table_rows,
        publication_table_note,
        args.allow_partial,
        args.warmup_exclude_s,
        args.event_window_s,
    )

    print(f"wrote analysis to {output_dir}")
    print(
        "completion: "
        f"{report['success']}/{report['total']} success, "
        f"{report['fail']} fail, "
        f"valid_for_performance={report['valid_for_performance']}"
    )
    if scenario_summary:
        for row in scenario_summary:
            print(
                f"{row.get('kind')} / {row.get('ego_policy')}: "
                f"runs={row.get('num_runs')} "
                f"events={row.get('num_event_available')} "
                f"mean_v={format_value(row.get('full_mean_velocity_mean'))} "
                f"event_mean_v={format_value(row.get('event_mean_velocity_mean'))}"
            )
    elif not report["valid_for_performance"]:
        print("performance aggregation skipped; rerun with --allow-partial for partial summaries")


if __name__ == "__main__":
    main()
