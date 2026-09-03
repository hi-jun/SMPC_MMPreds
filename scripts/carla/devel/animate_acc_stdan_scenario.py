from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np


MODE_COLORS = {
    "LK": "#7b2cbf",
    "LLC": "#1f77b4",
    "RLC": "#ff7f0e",
}
EGO_REF_COLORS = ["#006d77", "#83c5be", "#264653", "#2a9d8f"]
NO_LEAD_GAP_SENTINEL_M = 1000.0
NO_LEAD_GAP_ATOL_M = 1.0e-3
ACC_VEHICLE_LENGTH_M = 4.5
ACC_MIN_CLEARANCE_M = 3.0
ACC_TIME_HEADWAY_S = 1.3


def _load_summary(path: Path) -> dict:
    if path.is_dir():
        path = path / "summary.json"
    with path.open("r") as f:
        return json.load(f)


def _ego_steps(summary: dict) -> list[dict]:
    for log in summary.get("policy_logs", {}).values():
        if isinstance(log, dict) and log.get("controller") == "acc_nair_smpc":
            return log.get("steps", [])
    return []


def _target_log(summary: dict) -> dict:
    for key, log in summary.get("policy_logs", {}).items():
        if key.startswith("target_") and isinstance(log, dict):
            return log
    return {}


def _first_target(step: dict) -> dict | None:
    targets = step.get("stdan_debug", {}).get("processed_targets", [])
    return targets[0] if targets else None


def _targets(step: dict) -> list[dict]:
    targets = step.get("stdan_debug", {}).get("processed_targets", [])
    return targets if isinstance(targets, list) else []


def _scenario_config(input_path: Path) -> dict:
    base = input_path if input_path.is_dir() else input_path.parent
    path = base / "resolved_config.json"
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f).get("scenario", {})


def _actor_logs(input_path: Path) -> dict[str, dict]:
    base = input_path if input_path.is_dir() else input_path.parent
    path = base / "scenario_result.pkl"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        result = pickle.load(f)
    actors = {}
    for key, value in result.items():
        if not isinstance(value, dict) or "state_trajectory" not in value:
            continue
        states = np.asarray(value.get("state_trajectory", []), dtype=float)
        if states.ndim != 2 or states.shape[1] < 3:
            continue
        role = "_".join(key.split("_")[:-1]) if "_" in key else key
        actors[role] = {
            "key": key,
            "states": states,
            "lane_trajectory": value.get("lane_trajectory", []),
            "color": _role_color(role),
        }
    return actors


def _role_color(role: str) -> str:
    if role == "ego":
        return "#2ca02c"
    if role.startswith("target_"):
        return "#d62728"
    if "front_slow" in role or "cutin_lane_lead" in role:
        return "#ffb000"
    if "outer_blocker" in role:
        return "#1f77b4"
    return "#777777"


def _route_geometry(scenario: dict) -> dict:
    vehicles = scenario.get("vehicle_params", [])
    ego = next((v for v in vehicles if v.get("role") == "ego"), {})
    ego_left_offset = float(ego.get("start_left_offset", 0.0))
    lane_width = 3.5
    offsets = {
        str(v.get("role")): float(v.get("start_left_offset", 0.0)) - ego_left_offset
        for v in vehicles
    }
    params = {str(v.get("role")): v for v in vehicles}
    return {"lane_width": lane_width, "role_offsets": offsets, "role_params": params}


def _actor_xy(actor: dict, time_s: float) -> tuple[float, float] | None:
    states = actor["states"]
    times = states[:, 0]
    if times.size == 0 or time_s < times[0] - 0.2 or time_s > times[-1] + 0.2:
        return None
    idx = int(np.argmin(np.abs(times - time_s)))
    x = float(states[idx, 1])
    y_rhs = float(states[idx, 2])
    return x, y_rhs


def _target_prediction_origin(target: dict) -> tuple[float, float] | None:
    for mode in MODE_COLORS:
        path = _mode_xy_path(target, mode, None)
        if path is not None and path.size:
            return float(path[0, 0]), float(path[0, 1])
    return None


def _target_role(
        target: dict,
        actors: dict[str, dict],
        time_s: float) -> str | None:
    origin = _target_prediction_origin(target)
    if origin is None:
        return None
    candidates = []
    origin_arr = np.asarray(origin, dtype=float)
    for role, actor in actors.items():
        if not role.startswith("target_"):
            continue
        xy = _actor_xy(actor, time_s)
        if xy is None:
            continue
        distance = float(np.linalg.norm(origin_arr - np.asarray(xy, dtype=float)))
        candidates.append((distance, role))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _target_for_role(
        step: dict,
        actors: dict[str, dict],
        time_s: float,
        role_name: str) -> dict | None:
    targets = _targets(step)
    if not targets:
        return None
    role_actor = actors.get(role_name)
    role_xy = _actor_xy(role_actor, time_s) if role_actor is not None else None
    if role_xy is None:
        return _first_target(step)
    role_arr = np.asarray(role_xy, dtype=float)
    candidates = []
    for target in targets:
        origin = _target_prediction_origin(target)
        if origin is None:
            continue
        distance = float(np.linalg.norm(np.asarray(origin, dtype=float) - role_arr))
        candidates.append((distance, target))
    if not candidates:
        return _first_target(step)
    return min(candidates, key=lambda item: item[0])[1]


def _vehicle_id_role_map(
        steps: list[dict],
        actors: dict[str, dict]) -> dict[int, str]:
    vehicle_roles = {}
    for step in steps:
        time_s = _float(step.get("time_s"))
        for target in _targets(step):
            vehicle_id = target.get("vehicle_id")
            if vehicle_id is None:
                continue
            role = _target_role(target, actors, time_s)
            if role is not None:
                vehicle_roles[int(vehicle_id)] = role
    return vehicle_roles


def _dominant_current_effective_lead(step: dict) -> tuple[int, str | None] | None:
    metadata = step.get("stdan_debug", {}).get("scenario_metadata", [])
    if not isinstance(metadata, list):
        return None

    scores = {}
    mode_scores = {}
    for scenario in metadata:
        if not isinstance(scenario, dict):
            continue
        vehicle_ids = scenario.get("selected_vehicle_ids", [])
        if not vehicle_ids:
            keys = scenario.get("effective_lead_keys", [])
            if keys and isinstance(keys[0], list) and keys[0] and keys[0][0] != "inactive":
                vehicle_ids = [keys[0][0]]
        if not vehicle_ids or vehicle_ids[0] is None:
            continue
        try:
            vehicle_id = int(vehicle_ids[0])
        except (TypeError, ValueError):
            continue
        probability = _float(scenario.get("probability"), default=0.0)
        scores[vehicle_id] = scores.get(vehicle_id, 0.0) + max(0.0, probability)

        mode_names = scenario.get("selected_mode_names", [])
        mode_name = mode_names[0] if mode_names else None
        mode_scores[(vehicle_id, mode_name)] = (
            mode_scores.get((vehicle_id, mode_name), 0.0) + max(0.0, probability)
        )

    if not scores:
        return None
    vehicle_id = max(scores.items(), key=lambda item: item[1])[0]
    mode_candidates = [
        (score, mode_name)
        for (candidate_id, mode_name), score in mode_scores.items()
        if candidate_id == vehicle_id
    ]
    mode_name = max(mode_candidates, key=lambda item: item[0])[1] if mode_candidates else None
    return vehicle_id, mode_name


def _effective_lead_state(
        step: dict,
        actors: dict[str, dict],
        vehicle_id_to_role: dict[int, str],
        time_s: float) -> dict | None:
    lead = _dominant_current_effective_lead(step)
    if lead is None:
        return None
    vehicle_id, mode_name = lead
    role = vehicle_id_to_role.get(vehicle_id)
    ego_actor = actors.get("ego")
    lead_actor = actors.get(role) if role is not None else None
    if ego_actor is None or lead_actor is None:
        return None
    ego_pose = _actor_pose(ego_actor, time_s)
    lead_pose = _actor_pose(lead_actor, time_s)
    if ego_pose is None or lead_pose is None:
        return None
    # Scenario animations use CARLA RHS y as the road longitudinal axis:
    # ahead of ego is smaller y_rhs, so ego_y - lead_y is the center gap.
    gap = float(ego_pose[1]) - float(lead_pose[1])
    return {
        "vehicle_id": vehicle_id,
        "mode_name": mode_name,
        "role": role,
        "gap": gap if np.isfinite(gap) and gap < NO_LEAD_GAP_SENTINEL_M else np.nan,
        "speed": float(lead_pose[3]),
    }


def _actor_trace_xy(actor: dict, time_s: float) -> np.ndarray:
    states = actor["states"]
    if states.ndim != 2 or states.shape[1] < 3:
        return np.empty((0, 2), dtype=float)
    mask = states[:, 0] <= time_s + 1e-6
    if not np.any(mask):
        return np.empty((0, 2), dtype=float)
    return states[mask, 1:3]


def _actor_full_trace_xy(actor: dict) -> np.ndarray:
    states = actor["states"]
    if states.ndim != 2 or states.shape[1] < 3:
        return np.empty((0, 2), dtype=float)
    points = states[:, 1:3]
    finite = np.all(np.isfinite(points), axis=1)
    return points[finite]


def _actor_pose(actor: dict, time_s: float) -> tuple[float, float, float, float] | None:
    states = actor["states"]
    if states.ndim != 2 or states.shape[1] < 3:
        return None
    times = states[:, 0]
    if times.size == 0 or time_s < times[0] - 0.2 or time_s > times[-1] + 0.2:
        return None
    idx = int(np.argmin(np.abs(times - time_s)))
    x = float(states[idx, 1])
    y_rhs = float(states[idx, 2])
    yaw = float(states[idx, 3]) if states.shape[1] > 3 else -0.5 * np.pi
    speed = float(states[idx, 4]) if states.shape[1] > 4 else np.nan
    return x, y_rhs, yaw, speed


def _float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _gap_for_plot(ego_s: float, target_s: float) -> float:
    if not np.isfinite(ego_s) or not np.isfinite(target_s):
        return np.nan
    gap = float(target_s) - float(ego_s)
    if abs(gap - NO_LEAD_GAP_SENTINEL_M) <= NO_LEAD_GAP_ATOL_M:
        return np.nan
    return gap


def _gap_text(gap: float) -> str:
    return "none" if not np.isfinite(gap) else f"{gap:.1f}m"


def _safe_center_distance(ego_v: float) -> float:
    if not np.isfinite(ego_v):
        return np.nan
    return ACC_VEHICLE_LENGTH_M + ACC_MIN_CLEARANCE_M + ACC_TIME_HEADWAY_S * float(ego_v)


def _occupancy_path_indices(mode_mask: np.ndarray, path: np.ndarray) -> np.ndarray:
    mask = np.asarray(mode_mask, dtype=bool).reshape(-1)
    path_len = int(path.shape[0]) if path.ndim >= 2 else 0
    if mask.size == 0 or path_len == 0:
        return np.empty(0, dtype=int)

    occ_steps = np.flatnonzero(mask)
    if path_len == mask.size:
        path_indices = occ_steps
    elif mask.size > 1 and path_len >= mask.size - 1:
        future_steps = occ_steps[occ_steps > 0]
        if future_steps.size == 0:
            return np.empty(0, dtype=int)
        sample_ratio = float(path_len) / float(mask.size - 1)
        path_indices = np.rint(future_steps * sample_ratio).astype(int) - 1
    else:
        path_indices = occ_steps

    path_indices = path_indices[(0 <= path_indices) & (path_indices < path_len)]
    return np.unique(path_indices.astype(int))


def _series(
        steps: list[dict],
        actors: Optional[dict[str, dict]] = None,
        target_role: str = "target_cutin") -> dict:
    actors = actors or {}
    vehicle_id_to_role = _vehicle_id_role_map(steps, actors) if actors else {}
    values = {
        "time_s": [],
        "ego_s": [],
        "ego_v": [],
        "target_s": [],
        "target_d": [],
        "target_v": [],
        "accel_cmd": [],
        "gap": [],
        "safe_dist": [],
        "safe_dist_relaxed": [],
        "feasible": [],
        "chance_margin": [],
        "LK": [],
        "LLC": [],
        "RLC": [],
        "acc_lk": [],
        "acc_cutin": [],
        "acc_cutout": [],
        "cutin_predicted": [],
    }
    for step in steps:
        time_s = _float(step.get("time_s"))
        target = _target_for_role(step, actors, time_s, target_role)
        raw = target.get("raw_mode_prob", {}) if target else {}
        acc = target.get("acc_mode_prob", {}) if target else {}
        ego_s = _float(step.get("ego_s"))
        ego_v = _float(step.get("ego_v"))
        target_s = _float(step.get("target_s"))
        effective_lead = _effective_lead_state(
            step,
            actors,
            vehicle_id_to_role,
            time_s,
        )
        values["time_s"].append(time_s)
        values["ego_s"].append(ego_s)
        values["ego_v"].append(ego_v)
        values["target_s"].append(target_s)
        values["target_d"].append(_float(step.get("target_d")))
        # Only a vehicle actually in the ego lane counts as the lead; before the
        # cut-in vehicle crosses, the logged target is the next-lane car and its
        # speed must not be drawn as a lead's.
        values["target_v"].append(
            _float(effective_lead.get("speed")) if effective_lead else float("nan")
        )
        # The predictor has the cut-in vehicle entering the ego lane within the
        # horizon (its predicted trajectory, whose length is the horizon), while
        # it is still in the next lane.
        occupancy = target.get("ego_lane_occupancy_mask") or [] if target else []
        values["cutin_predicted"].append(bool(
            target
            and target.get("relation_to_ego_lane") != "ego_lane"
            and _float(target.get("ego_lane_start_idx"), default=np.inf) < max(len(occupancy), 1)
        ))
        values["accel_cmd"].append(_float(step.get("accel_cmd")))
        values["gap"].append(
            _gap_for_plot(0.0, effective_lead.get("gap")) if effective_lead
            else _gap_for_plot(ego_s, target_s)
        )
        values["safe_dist"].append(_safe_center_distance(ego_v))
        # Requirement the constraint holds the ego to right now, once the
        # gap-recovery relaxation is active; NaN while the standoff is unrelaxed.
        scale = step.get("clearance_scale_now")
        values["safe_dist_relaxed"].append(
            _safe_center_distance(ego_v) * float(scale)
            if isinstance(scale, (int, float)) and math.isfinite(float(scale)) and float(scale) < 0.999
            else float("nan")
        )
        values["feasible"].append(bool(step.get("feasible")))
        margin = step.get("chance_margin_min")
        values["chance_margin"].append(
            float(margin) if isinstance(margin, (int, float)) and math.isfinite(float(margin))
            else float("nan")
        )
        for name in ("LK", "LLC", "RLC"):
            values[name].append(_float(raw.get(name)))
        for name in ("lk", "cutin", "cutout"):
            values[f"acc_{name}"].append(_float(acc.get(name)))
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def _vehicle(ax, x: float, y_rhs: float, color: str, label: str):
    length = 4.6
    width = 1.9
    patch = Rectangle(
        (x - width / 2.0, y_rhs - length / 2.0),
        width,
        length,
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.92,
        label=label,
        zorder=5,
    )
    ax.add_patch(patch)


def _oriented_vehicle(ax, x: float, y_rhs: float, yaw: float, color: str, label: str):
    length = 4.6
    width = 1.9
    transform = Affine2D().rotate(yaw).translate(x, y_rhs) + ax.transData
    patch = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.94,
        label=label,
        transform=transform,
        zorder=5,
    )
    ax.add_patch(patch)
    nose_x = x + 0.55 * length * np.cos(yaw)
    nose_y = y_rhs + 0.55 * length * np.sin(yaw)
    ax.plot([x, nose_x], [y_rhs, nose_y], color="black", linewidth=1.0, zorder=6)


def _mode_xy_path(target: dict, mode: str, current_xy: tuple[float, float] | None) -> np.ndarray | None:
    traj = None
    if target:
        traj = target.get("full_pred_traj_xy", {}).get(mode)
        if not traj:
            traj = target.get("pred_traj_xy", {}).get(mode)
    if traj:
        arr = np.asarray(traj, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            if current_xy is None:
                return arr[:, :2]
            return np.vstack((np.asarray(current_xy, dtype=float).reshape(1, 2), arr[:, :2]))
    return None


def _ego_reference_paths(
        step: dict,
        ego_s: float,
        ego_xy: tuple[float, float] | None) -> list[tuple[str, np.ndarray]]:
    # ACC currently logs 1D Frenet references only.  For this straight-road
    # visualization, project those s references onto the current ego lane.
    if ego_xy is None:
        return []
    ref = step.get("acc_reference", {})
    s_ref = np.asarray(ref.get("s_ref", []), dtype=float)
    if s_ref.ndim == 1 and s_ref.size:
        s_ref = s_ref.reshape(1, -1)
    if s_ref.ndim != 2 or not s_ref.size:
        return []
    paths = []
    ego_x, ego_y_rhs = float(ego_xy[0]), float(ego_xy[1])
    for mode_idx in range(s_ref.shape[0]):
        delta_s = s_ref[mode_idx] - float(ego_s)
        display_x = np.full_like(delta_s, ego_x)
        display_y_rhs = ego_y_rhs - delta_s
        paths.append((f"ego ref m{mode_idx}", np.stack((display_x, display_y_rhs), axis=1)))
    return paths


def _draw_lanes(ax, ego_lane_center_x: float, lane_width: float = 3.5):
    for edge in (-0.5, 0.5, 1.5, 2.5):
        x = ego_lane_center_x + edge * lane_width
        style = "--" if edge > 0.0 else "-"
        ax.axvline(x, color="0.55", linestyle=style, linewidth=0.8, zorder=0)
    ax.axvline(ego_lane_center_x, color="0.25", linewidth=1.0, zorder=0)


def _draw_lanes_for_fixed_view(
        ax,
        ego_lane_center_x: float,
        lane_width: float,
        x_min: float,
        x_max: float):
    lane_width = float(lane_width)
    if lane_width <= 0.0:
        return
    edge_start = int(np.floor((x_min - ego_lane_center_x) / lane_width - 0.5)) - 1
    edge_end = int(np.ceil((x_max - ego_lane_center_x) / lane_width - 0.5)) + 1
    for edge_idx in range(edge_start, edge_end + 1):
        x = ego_lane_center_x + (edge_idx + 0.5) * lane_width
        ax.axvline(x, color="0.58", linestyle="--", linewidth=0.9, zorder=0)

    center_start = int(np.floor((x_min - ego_lane_center_x) / lane_width)) - 1
    center_end = int(np.ceil((x_max - ego_lane_center_x) / lane_width)) + 1
    for center_idx in range(center_start, center_end + 1):
        x = ego_lane_center_x + center_idx * lane_width
        if not (x_min <= x <= x_max):
            continue
        if center_idx == 0:
            ax.axvline(x, color="0.22", linewidth=1.2, zorder=0)
        else:
            ax.axvline(x, color="0.82", linestyle=":", linewidth=0.8, zorder=0)


def _fixed_actual_view_limits(
        actors: dict[str, dict],
        route: dict,
        extra_points: Optional[np.ndarray] = None,
        x_padding: float = 4.0,
        y_padding: float = 12.0) -> tuple[float, float, float, float]:
    point_sets = [_actor_full_trace_xy(actor) for actor in actors.values()]
    point_sets = [points for points in point_sets if points.size]
    if extra_points is not None and extra_points.size:
        point_sets.append(extra_points)
    if point_sets:
        points = np.vstack(point_sets)
        x_min = float(np.min(points[:, 0]))
        x_max = float(np.max(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        y_max = float(np.max(points[:, 1]))
    else:
        lane_center = float(route.get("ego_lane_center_x", 0.0))
        x_min = lane_center - 8.0
        x_max = lane_center + 8.0
        y_min = -20.0
        y_max = 120.0

    lane_width = float(route.get("lane_width", 3.5))
    ego_lane_center_x = float(route.get("ego_lane_center_x", 0.0))
    x_min = min(x_min, ego_lane_center_x - 1.5 * lane_width)
    x_max = max(x_max, ego_lane_center_x + 3.5 * lane_width)
    return (
        x_min - float(x_padding),
        x_max + float(x_padding),
        y_max + float(y_padding),
        y_min - float(y_padding),
    )


def _actual_cutin_time(actors: dict[str, dict], target_log: dict) -> float | None:
    target_actor = actors.get("target_cutin")
    ego_lane_id = target_log.get("ego_lane_id")
    lane_records = target_actor.get("lane_trajectory", []) if target_actor is not None else []
    if ego_lane_id is not None and lane_records:
        for record in lane_records:
            if record.get("lane_id") == ego_lane_id:
                time_s = _float(record.get("time_s"))
                if np.isfinite(time_s):
                    return time_s

    trigger_time = _float(target_log.get("trigger_time_s"))
    return trigger_time if np.isfinite(trigger_time) else None


def _shade_spans(ax, times, mask, color, label, alpha=0.10):
    """Shade every contiguous span where ``mask`` is true, labelled once."""
    start = None
    labelled = False
    for idx, on in enumerate(list(mask) + [False]):
        if on and start is None:
            start = idx
        elif not on and start is not None:
            ax.axvspan(times[start], times[min(idx, len(times) - 1)],
                       color=color, alpha=alpha, zorder=0,
                       label=None if labelled else label)
            labelled = True
            start = None

def _draw_cutin_time(ax, cutin_time_s: float | None, label: str | None = None):
    if cutin_time_s is None or not np.isfinite(float(cutin_time_s)):
        return
    ax.axvline(
        float(cutin_time_s),
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label=label,
    )


def _infer_actual_fps(actors: dict[str, dict], default: int = 20) -> int:
    for actor in actors.values():
        states = actor["states"]
        if states.ndim != 2 or states.shape[0] < 2:
            continue
        dt = np.diff(states[:, 0])
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        if dt.size:
            return max(1, int(round(1.0 / float(np.median(dt)))))
    return int(default)


def _step_times(steps: list[dict]) -> np.ndarray:
    return np.asarray([_float(step.get("time_s")) for step in steps], dtype=float)


def _nearest_step(
        steps: list[dict],
        step_times: np.ndarray,
        time_s: float) -> dict | None:
    if not steps or step_times.size == 0:
        return None
    finite = np.isfinite(step_times)
    if not np.any(finite):
        return None
    finite_indices = np.flatnonzero(finite)
    nearest = finite_indices[int(np.argmin(np.abs(step_times[finite] - float(time_s))))]
    return steps[int(nearest)]


def _prediction_points(steps: list[dict]) -> np.ndarray:
    points = []
    for step in steps:
        for target in step.get("stdan_debug", {}).get("processed_targets", []):
            for mode in MODE_COLORS:
                path = _mode_xy_path(target, mode, None)
                if path is not None and path.size:
                    points.append(path[:, :2])
    if not points:
        return np.empty((0, 2), dtype=float)
    all_points = np.vstack(points)
    finite = np.all(np.isfinite(all_points), axis=1)
    return all_points[finite]


def _draw_actual_predictions(ax, step: dict | None, used_labels: set[str]):
    if step is None:
        return
    targets = step.get("stdan_debug", {}).get("processed_targets", [])
    for target in targets:
        raw_prob = target.get("raw_mode_prob", {})
        for mode, color in MODE_COLORS.items():
            path = _mode_xy_path(target, mode, None)
            if path is None or not path.size:
                continue
            prob = _float(raw_prob.get(mode))
            prob_weight = 0.0 if not np.isfinite(prob) else float(np.clip(prob, 0.0, 1.0))
            label = f"{mode} pred" if mode not in used_labels else None
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                linestyle="--",
                linewidth=1.8,
                alpha=0.25 + 0.55 * prob_weight,
                label=label,
                zorder=3,
            )
            ax.scatter(
                path[:, 0],
                path[:, 1],
                color=color,
                s=12,
                alpha=0.12 + 0.35 * prob_weight,
                zorder=3,
            )
            used_labels.add(mode)


def _draw_barh(ax, labels, values, colors, title):
    ax.clear()
    y = np.arange(len(labels))
    clean = [0.0 if not np.isfinite(v) else float(v) for v in values]
    ax.barh(y, clean, color=colors, alpha=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    for idx, value in enumerate(clean):
        ax.text(min(value + 0.03, 0.92), idx, f"{value:.2f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)


def _plot_probability_series(
        ax,
        data: dict,
        label_prefix: str,
        linestyle: str = "-",
        alpha: float = 1.0):
    def label(name: str) -> str:
        return f"{label_prefix} {name}" if label_prefix else name

    ax.plot(
        data["time_s"],
        data["LK"],
        color=MODE_COLORS["LK"],
        linestyle=linestyle,
        alpha=alpha,
        label=label("LK"),
    )
    ax.plot(
        data["time_s"],
        data["LLC"],
        color=MODE_COLORS["LLC"],
        linestyle=linestyle,
        alpha=alpha,
        label=label("LLC"),
    )
    ax.plot(
        data["time_s"],
        data["RLC"],
        color=MODE_COLORS["RLC"],
        linestyle=linestyle,
        alpha=alpha,
        label=label("RLC"),
    )
    ax.plot(
        data["time_s"],
        data["acc_cutin"],
        color="black",
        linestyle=linestyle,
        alpha=alpha,
        label=label("ACC cutin"),
    )


def make_animation(
        input_path: Path,
        output_path: Path,
        fps: int,
        stride: int,
        scene_x_window: Optional[float] = None,
        scene_equal_aspect: bool = False,
        scene_y_ahead: float = 90.0,
        scene_y_behind: float = 20.0,
        scene_expand_to_points: bool = False):
    summary = _load_summary(input_path)
    steps = _ego_steps(summary)
    if not steps:
        raise RuntimeError("No ACC ego policy steps found in summary.")
    frames = list(range(0, len(steps), max(1, int(stride))))
    target_log = _target_log(summary)
    scenario = _scenario_config(input_path)
    route = _route_geometry(scenario)
    route["target_trigger_time"] = target_log.get("trigger_time_s")
    actors = _actor_logs(input_path)
    if "ego" in actors:
        ego_states = actors["ego"]["states"]
        if ego_states.size:
            route["ego_lane_center_x"] = float(ego_states[0, 1])
    vehicle_id_to_role = _vehicle_id_role_map(steps, actors) if actors else {}
    data = _series(steps, actors, target_role="target_cutin")
    data_right = _series(steps, actors, target_role="target_right_lane_straight")
    cutin_time_s = _actual_cutin_time(actors, target_log)

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[2.2, 1.0, 1.0],
        height_ratios=[1.45, 0.9, 0.9],
    )
    ax_scene = fig.add_subplot(gs[0, 0])
    ax_raw = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[0, 2])
    ax_prob = fig.add_subplot(gs[1, 0])
    ax_prob_right = fig.add_subplot(gs[2, 0])
    ax_ctrl = fig.add_subplot(gs[1:, 1:])
    ax_ctrl_2 = ax_ctrl.twinx()
    ax_ctrl_3 = ax_ctrl.twinx()
    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        top=0.94,
        bottom=0.07,
        wspace=0.32,
        hspace=0.48,
    )

    def update(frame_number):
        idx = frames[frame_number]
        step = steps[idx]
        time_s = _float(step.get("time_s"))
        target = _target_for_role(step, actors, time_s, "target_cutin")
        prediction_targets = _targets(step)
        ego_s = _float(step.get("ego_s"))
        ego_v = _float(step.get("ego_v"))
        target_s = _float(step.get("target_s"))
        target_d = _float(step.get("target_d"))
        target_v = _float(step.get("target_v"))
        accel = _float(step.get("accel_cmd"))
        effective_lead = _effective_lead_state(
            step,
            actors,
            vehicle_id_to_role,
            time_s,
        )
        gap = (
            _gap_for_plot(0.0, effective_lead.get("gap")) if effective_lead
            else _gap_for_plot(ego_s, target_s)
        )
        ego_actor_xy = _actor_xy(actors["ego"], time_s) if "ego" in actors else None
        ego_lane_center_x = float(route.get(
            "ego_lane_center_x",
            ego_actor_xy[0] if ego_actor_xy is not None else 0.0,
        ))

        ax_scene.clear()
        _draw_lanes(ax_scene, ego_lane_center_x, route.get("lane_width", 3.5))
        relation = target.get("relation_to_ego_lane") if target else "no_stdan"
        ax_scene.set_title(
            f"t={time_s:.2f}s | gap={_gap_text(gap)} | a={accel:.2f}m/s^2 | rel={relation}",
            fontsize=10,
        )
        ax_scene.set_xlabel("CARLA x [m]")
        ax_scene.set_ylabel("CARLA y_RHS=-y [m]")

        scene_points = []
        limit_points = []
        if actors:
            for role, actor in actors.items():
                xy = _actor_xy(actor, time_s)
                if xy is None:
                    continue
                x_actor, y_actor = xy
                scene_points.append([x_actor, y_actor])
                limit_points.append([x_actor, y_actor])
                trace = _actor_trace_xy(actor, time_s)
                if trace.size:
                    ax_scene.plot(
                        trace[:, 0],
                        trace[:, 1],
                        color=actor["color"],
                        linewidth=1.2,
                        alpha=0.35,
                        zorder=2,
                    )
                _vehicle(ax_scene, x_actor, y_actor, actor["color"], "_nolegend_")
                ax_scene.text(
                    x_actor + 0.6,
                    y_actor + 2.7,
                    role.replace("traffic_", ""),
                    fontsize=8,
                    color="black",
                    zorder=7,
                )

        for ref_idx, (_label, ref_path) in enumerate(
                _ego_reference_paths(step, ego_s, ego_actor_xy)):
            color = EGO_REF_COLORS[ref_idx % len(EGO_REF_COLORS)]
            scene_points.extend(ref_path.tolist())
            limit_points.extend(ref_path.tolist())
            ax_scene.plot(
                ref_path[:, 0],
                ref_path[:, 1],
                color=color,
                linestyle="--",
                linewidth=2.0,
                marker=".",
                markersize=5,
                alpha=0.95,
                label="_nolegend_",
                zorder=3,
            )

        used_prediction_labels = set()
        for pred_target in prediction_targets:
            raw_prob = pred_target.get("raw_mode_prob", {})
            for mode, color in MODE_COLORS.items():
                path = _mode_xy_path(pred_target, mode, None)
                prob = raw_prob.get(mode, np.nan)
                if path is None:
                    continue
                scene_points.extend(path.tolist())
                limit_points.extend(path.tolist())
                alpha = 0.25 + 0.65 * (0.0 if not np.isfinite(_float(prob)) else _float(prob))
                label = mode if mode not in used_prediction_labels else "_nolegend_"
                ax_scene.plot(
                    path[:, 0],
                    path[:, 1],
                    color=color,
                    linewidth=2.0,
                    alpha=alpha,
                    label=label,
                    zorder=3,
                )
                ax_scene.scatter(path[:, 0], path[:, 1], color=color, s=18, alpha=alpha, zorder=4)
                used_prediction_labels.add(mode)

                mode_mask = np.asarray(
                    pred_target.get("mode_lane_membership", {}).get(mode, []),
                    dtype=bool,
                )
                if mode_mask.size:
                    occ_idx = _occupancy_path_indices(mode_mask, path)
                    if occ_idx.size:
                        ax_scene.scatter(
                            path[occ_idx, 0],
                            path[occ_idx, 1],
                            s=55,
                            facecolor="none",
                            edgecolor=color,
                            linewidth=1.5,
                            label="_nolegend_",
                            zorder=6,
                        )

        if not actors:
            ax_scene.text(
                0.5,
                0.5,
                "scenario_result.pkl is required for CARLA x/y visualization",
                transform=ax_scene.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        if scene_points:
            points = np.asarray(scene_points, dtype=float)
            finite = np.all(np.isfinite(points), axis=1)
            points = points[finite]
        else:
            points = np.empty((0, 2), dtype=float)
        if limit_points:
            axis_points = np.asarray(limit_points, dtype=float)
            finite = np.all(np.isfinite(axis_points), axis=1)
            axis_points = axis_points[finite]
        else:
            axis_points = points
        if ego_actor_xy is not None:
            ego_x, ego_y = ego_actor_xy
            if scene_x_window is not None and scene_x_window > 0.0:
                x_center = float(np.median(axis_points[:, 0])) if axis_points.size else float(ego_x)
                x_min = x_center - 0.5 * float(scene_x_window)
                x_max = x_center + 0.5 * float(scene_x_window)
            else:
                x_min = min(ego_lane_center_x - 5.0, np.min(axis_points[:, 0]) - 3.0 if axis_points.size else ego_x - 5.0)
                x_max = max(ego_lane_center_x + 12.5, np.max(axis_points[:, 0]) + 3.0 if axis_points.size else ego_x + 12.5)
            y_bottom = ego_y + float(scene_y_behind)
            y_top = ego_y - float(scene_y_ahead)
            if scene_expand_to_points and axis_points.size:
                y_bottom = max(y_bottom, np.max(axis_points[:, 1]) + 8.0)
                y_top = min(y_top, np.min(axis_points[:, 1]) - 8.0)
            ax_scene.set_xlim(x_min, x_max)
            ax_scene.set_ylim(y_bottom, y_top)
        elif axis_points.size:
            if scene_x_window is not None and scene_x_window > 0.0:
                x_center = float(np.median(axis_points[:, 0]))
                ax_scene.set_xlim(
                    x_center - 0.5 * float(scene_x_window),
                    x_center + 0.5 * float(scene_x_window),
                )
            else:
                ax_scene.set_xlim(np.min(axis_points[:, 0]) - 3.0, np.max(axis_points[:, 0]) + 3.0)
            ax_scene.set_ylim(np.max(axis_points[:, 1]) + 8.0, np.min(axis_points[:, 1]) - 8.0)
        else:
            if scene_x_window is not None and scene_x_window > 0.0:
                ax_scene.set_xlim(
                    ego_lane_center_x - 0.5 * float(scene_x_window),
                    ego_lane_center_x + 0.5 * float(scene_x_window),
                )
            else:
                ax_scene.set_xlim(ego_lane_center_x - 5.0, ego_lane_center_x + 12.5)
            ax_scene.set_ylim(float(scene_y_behind), -float(scene_y_ahead))
        if scene_equal_aspect:
            ax_scene.set_aspect("equal", adjustable="datalim")
        else:
            ax_scene.set_aspect("auto")
        handles, labels = ax_scene.get_legend_handles_labels()
        if labels:
            ax_scene.legend(handles, labels, loc="upper right", fontsize=8)
        ax_scene.grid(True, alpha=0.2)

        raw = target.get("raw_mode_prob", {}) if target else {}
        acc = target.get("acc_mode_prob", {}) if target else {}
        _draw_barh(
            ax_raw,
            ["LK", "LLC", "RLC"],
            [_float(raw.get("LK")), _float(raw.get("LLC")), _float(raw.get("RLC"))],
            [MODE_COLORS["LK"], MODE_COLORS["LLC"], MODE_COLORS["RLC"]],
            "STDAN prob (target_cutin)",
        )
        _draw_barh(
            ax_acc,
            ["lk", "cutin", "cutout"],
            [_float(acc.get("lk")), _float(acc.get("cutin")), _float(acc.get("cutout"))],
            ["#7b2cbf", "#111111", "#2ca02c"],
            "ACC prob (target_cutin)",
        )

        ax_prob.clear()
        _plot_probability_series(ax_prob, data, "")
        _draw_cutin_time(ax_prob, cutin_time_s, label="actual cut-in")
        ax_prob.axvline(time_s, color="0.2", linewidth=1.0)
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.set_title("probability (target_cutin)", fontsize=10)
        ax_prob.set_ylabel("probability")
        ax_prob.legend(loc="upper right", fontsize=7, ncol=2)
        ax_prob.grid(True, alpha=0.2)

        ax_prob_right.clear()
        _plot_probability_series(ax_prob_right, data_right, "")
        _draw_cutin_time(ax_prob_right, cutin_time_s, label="actual cut-in")
        ax_prob_right.axvline(time_s, color="0.2", linewidth=1.0)
        ax_prob_right.set_ylim(-0.05, 1.05)
        ax_prob_right.set_title("probability (target_right_lane_straight)", fontsize=10)
        ax_prob_right.set_xlabel("time [s]")
        ax_prob_right.set_ylabel("probability")
        ax_prob_right.legend(loc="upper right", fontsize=7, ncol=2)
        ax_prob_right.grid(True, alpha=0.2)

        ax_ctrl.clear()
        ax_ctrl_2.clear()
        ax_ctrl_3.clear()
        # clear() resets spine positions, so the third axis is re-offset every frame.
        ax_ctrl_3.spines["right"].set_position(("axes", 1.09))
        ax_ctrl.plot(data["time_s"], data["ego_v"], color="#2ca02c", label="ego v")
        ax_ctrl.plot(data["time_s"], data["target_v"], color="#d62728", label="lead v (in lane)")
        ax_ctrl.plot(data["time_s"], data["gap"], color="#1f77b4", label="gap")
        ax_ctrl.plot(
            data["time_s"],
            data["safe_dist"],
            color="#111111",
            linestyle=":",
            linewidth=1.8,
            label="safe dist",
        )
        ax_ctrl.plot(
            data["time_s"],
            data["safe_dist_relaxed"],
            color="#111111",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="safe dist (relaxed)",
        )
        # The margin is a distance, so it shares the metre axis with the gap
        # and the requirement it is measured against.
        ax_ctrl.plot(
            data["time_s"],
            data["chance_margin"],
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label="chance margin",
        )
        ax_ctrl.axhline(0.0, color="#d62728", linewidth=0.6, alpha=0.4)
        # Shade the spans the solver had to slack the safety constraint on, so
        # an infeasible stretch is visible next to the speeds that caused it.
        _shade_spans(ax_ctrl, data["time_s"], ~np.asarray(data["feasible"], dtype=bool),
                     "#d62728", "infeasible")
        # Green while the predictor has the cut-in vehicle entering the ego
        # lane within the horizon: the controller is already planning against
        # it although it has not crossed yet.
        _shade_spans(ax_ctrl, data["time_s"], np.asarray(data["cutin_predicted"], dtype=bool),
                     "#2ca02c", "cut-in predicted")
        _draw_cutin_time(ax_ctrl, cutin_time_s, label="actual cut-in")
        ax_ctrl.axvline(time_s, color="0.2", linewidth=1.0)
        ax_ctrl.set_xlabel("time [s]")
        ax_ctrl.set_ylabel("speed [m/s] / gap [m]")
        ax_ctrl.grid(True, alpha=0.2)
        ax_ctrl_2.plot(data["time_s"], data["accel_cmd"], color="#9467bd", label="accel cmd", alpha=0.9)
        ax_ctrl_2.set_ylabel("accel cmd [m/s^2]")
        ax_ctrl_3.plot(data["time_s"], data["acc_cutin"], color="#111111", linewidth=1.0,
                       alpha=0.75, label="cut-in prob")
        ax_ctrl_3.set_ylim(-0.02, 1.02)
        ax_ctrl_3.set_ylabel("cut-in prob")
        lines, labels = ax_ctrl.get_legend_handles_labels()
        lines2, labels2 = ax_ctrl_2.get_legend_handles_labels()
        lines3, labels3 = ax_ctrl_3.get_legend_handles_labels()
        ax_ctrl.legend(lines + lines2 + lines3, labels + labels2 + labels3,
                       loc="upper right", fontsize=8)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp4":
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


def make_actual_trajectory_animation(
        input_path: Path,
        output_path: Path,
        fps: Optional[int],
        stride: int):
    summary = _load_summary(input_path)
    steps = _ego_steps(summary)
    step_times = _step_times(steps)
    scenario = _scenario_config(input_path)
    route = _route_geometry(scenario)
    actors = _actor_logs(input_path)
    if not actors:
        raise RuntimeError("scenario_result.pkl with actor state_trajectory entries is required.")
    if "ego" in actors:
        ego_states = actors["ego"]["states"]
        if ego_states.size:
            route["ego_lane_center_x"] = float(ego_states[0, 1])
    if fps is None:
        fps = _infer_actual_fps(actors)

    base_actor = actors.get("ego", next(iter(actors.values())))
    base_states = base_actor["states"]
    if base_states.ndim != 2 or base_states.shape[0] == 0:
        raise RuntimeError("No actor trajectory samples found.")
    frame_indices = list(range(0, base_states.shape[0], max(1, int(stride))))
    if frame_indices[-1] != base_states.shape[0] - 1:
        frame_indices.append(base_states.shape[0] - 1)
    frame_times = base_states[frame_indices, 0]

    prediction_points = _prediction_points(steps)
    x_min, x_max, y_bottom, y_top = _fixed_actual_view_limits(
        actors,
        route,
        extra_points=prediction_points,
    )
    lane_width = float(route.get("lane_width", 3.5))
    ego_lane_center_x = float(route.get("ego_lane_center_x", 0.0))

    fig, ax = plt.subplots(figsize=(11, 15), dpi=120)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07)

    def update(frame_number):
        time_s = float(frame_times[frame_number])
        ax.clear()
        ax.set_title(f"Actual CARLA trajectories | t={time_s:.2f}s", fontsize=14)
        ax.set_xlabel("CARLA x [m]", fontsize=12)
        ax.set_ylabel("CARLA y_RHS=-y [m]", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_bottom, y_top)
        ax.set_aspect("auto")
        _draw_lanes_for_fixed_view(ax, ego_lane_center_x, lane_width, x_min, x_max)

        used_prediction_labels = set()
        _draw_actual_predictions(
            ax,
            _nearest_step(steps, step_times, time_s),
            used_prediction_labels,
        )

        for role, actor in actors.items():
            full_trace = _actor_full_trace_xy(actor)
            if full_trace.size:
                ax.plot(
                    full_trace[:, 0],
                    full_trace[:, 1],
                    color=actor["color"],
                    linewidth=1.0,
                    alpha=0.18,
                    zorder=1,
                )
            trace = _actor_trace_xy(actor, time_s)
            if trace.size:
                ax.plot(
                    trace[:, 0],
                    trace[:, 1],
                    color=actor["color"],
                    linewidth=2.2,
                    alpha=0.65,
                    zorder=2,
                )
            pose = _actor_pose(actor, time_s)
            if pose is None:
                continue
            x_actor, y_actor, yaw_actor, speed_actor = pose
            _oriented_vehicle(ax, x_actor, y_actor, yaw_actor, actor["color"], role)
            label = role.replace("traffic_", "")
            if np.isfinite(speed_actor):
                label = f"{label} {speed_actor:.1f}m/s"
            ax.text(
                x_actor + 0.55,
                y_actor + 2.7,
                label,
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
                zorder=7,
            )

        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    animation = FuncAnimation(fig, update, frames=len(frame_times), interval=1000.0 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp4":
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
    parser.add_argument("input", help="Scenario output directory or summary.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--actual-output",
        default=None,
        help="Output path for the fixed-view actual CARLA trajectory animation.",
    )
    parser.add_argument(
        "--actual-fps",
        type=int,
        default=None,
        help="FPS for the actual CARLA trajectory animation. Defaults to --fps.",
    )
    parser.add_argument(
        "--actual-stride",
        type=int,
        default=None,
        help="Frame stride for the actual CARLA trajectory animation. Defaults to --stride.",
    )
    parser.add_argument(
        "--skip-actual-trajectory",
        action="store_true",
        help="Only write the existing STDAN scenario animation.",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument(
        "--scene-x-window",
        type=float,
        default=None,
        help="Fixed CARLA x-axis width in meters for the scenario animation panel.",
    )
    parser.add_argument(
        "--scene-equal-aspect",
        action="store_true",
        help="Render the scenario panel with equal CARLA x/y meter scale.",
    )
    parser.add_argument(
        "--scene-y-ahead",
        type=float,
        default=90.0,
        help="Meters ahead of ego to keep visible in the scenario panel.",
    )
    parser.add_argument(
        "--scene-y-behind",
        type=float,
        default=20.0,
        help="Meters behind ego to keep visible in the scenario panel.",
    )
    parser.add_argument(
        "--scene-expand-to-points",
        action="store_true",
        help="Expand scenario y-axis to include all current prediction/reference points.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path if input_path.is_dir() else input_path.parent
        output_path = output_path / "stdan_scenario_animation.mp4"
    else:
        output_path = Path(args.output)
    print(make_animation(
        input_path,
        output_path,
        args.fps,
        args.stride,
        scene_x_window=args.scene_x_window,
        scene_equal_aspect=args.scene_equal_aspect,
        scene_y_ahead=args.scene_y_ahead,
        scene_y_behind=args.scene_y_behind,
        scene_expand_to_points=args.scene_expand_to_points,
    ))
    if not args.skip_actual_trajectory:
        if args.actual_output is None:
            actual_output_path = input_path if input_path.is_dir() else input_path.parent
            actual_output_path = actual_output_path / "actual_carla_trajectory.mp4"
        else:
            actual_output_path = Path(args.actual_output)
        print(make_actual_trajectory_animation(
            input_path,
            actual_output_path,
            args.actual_fps if args.actual_fps is not None else args.fps,
            args.actual_stride if args.actual_stride is not None else args.stride,
        ))


if __name__ == "__main__":
    main()
