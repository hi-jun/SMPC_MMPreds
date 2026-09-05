from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CARLA_SCRIPT_DIR = SCRIPT_DIR.parent
SMPC_ROOT = CARLA_SCRIPT_DIR.parent.parent
SCENARIO_DIR = CARLA_SCRIPT_DIR / "scenarios"

LANE_WIDTH_M = 3.5
DEFAULT_LANE_SHIFT_RIGHT = 1
DEFAULT_SPAWN_SETTLE_S = 1.5
DEFAULT_CRUISE_WARMUP_S = 0.0


RECOMMENDED_SWEEP_RANGES = {
    "common": {
        "ego_speed": [10.0, 12.0, 14.0],
        "target_speed": [8.0, 10.0, 12.0],
        "target_start_gap": [35.0, 40.0, 45.0],
        "trigger_distance": [20.0, 25.0, 30.0],
    },
    "cutin": {
        "lane_change_distance_same_lane": [4.0, 6.0, 8.0],
        "lane_change_distance": [18.0, 24.0, 30.0],
        "target_lead_gap": [18.0, 24.0, 30.0],
        "target_lead_speed_delta": [-2.0, -3.0],
        "ego_lead_gap": [45.0, 55.0],
        "ego_lead_speed_delta": [-1.0, -2.0],
        "outer_blocker_count": [2, 3, 4],
        "outer_blocker_spacing": [10.0, 14.0, 18.0],
        "outer_blocker_speed_delta": [-1.0, -2.0],
    },
    "aggressive_cutin": {
        "lane_change_distance_same_lane": [0.0, 2.0, 4.0],
        "lane_change_distance": [8.0, 12.0, 16.0],
        "target_lead_gap": [12.0, 16.0, 20.0],
        "target_lead_speed_delta": [-3.0, -4.0],
        "outer_blocker_count": [3, 4],
        "outer_blocker_spacing": [8.0, 10.0, 12.0],
        "outer_blocker_speed_delta": [-1.0, -2.0],
    },
    "cutout": {
        "lane_change_distance_same_lane": [2.0, 5.0, 8.0],
        "lane_change_distance": [12.0, 18.0, 24.0],
        "cutout_direction": ["left"],
        "lead_gap": [24.0, 32.0, 45.0],
        "lead_speed": [7.0, 9.0, 11.0],
    },
}


def _base_carla_params(
        max_sim_time_s=12.0,
        spawn_settle_s=DEFAULT_SPAWN_SETTLE_S,
        spawn_settle_realtime=False,
        timeout_period=30.0,
        cruise_warmup_s=DEFAULT_CRUISE_WARMUP_S):
    return {
        "map_str": "Town04",
        "weather_str": "ClearNoon",
        "fps": 20,
        "intersection_csv_loc": "cutin_route.csv",
        "timeout_period": float(timeout_period),
        "max_sim_time_s": float(max_sim_time_s),
        "spawn_settle_s": float(spawn_settle_s),
        "spawn_settle_realtime": bool(spawn_settle_realtime),
        "cruise_warmup_s": float(cruise_warmup_s),
    }


def _base_drone_params(lane_shift_right=DEFAULT_LANE_SHIFT_RIGHT, lane_width=LANE_WIDTH_M):
    return {
        "x": -9.4 - float(lane_shift_right) * float(lane_width),
        "z": 50.0,
        "visualize_opencv": False,
        "save_avi": False,
        "overlay_gmm": False,
        "overlay_ego_info": True,
        "overlay_mode_probs": False,
        "overlay_traj_hist": True,
    }


def _right_shifted_left_offset(left_offset, params):
    lane_width = float(params.get("lane_width", LANE_WIDTH_M))
    lane_shift_right = float(params.get("lane_shift_right", DEFAULT_LANE_SHIFT_RIGHT))
    return float(left_offset) - lane_shift_right * lane_width


def _vehicle(
        role,
        policy_type,
        left_offset,
        start_s,
        goal_s,
        speed,
        color,
        vehicle_type="vehicle.audi.tt",
        **extra):
    vehicle = {
        "role": role,
        "vehicle_type": vehicle_type,
        "vehicle_color": color,
        "policy_type": policy_type,
        "intersection_start_node_idx": 0,
        "intersection_goal_node_idx": 0,
        "start_left_offset": float(left_offset),
        "goal_left_offset": float(left_offset),
        "start_longitudinal_offset": float(start_s),
        "goal_longitudinal_offset": float(goal_s),
        "nominal_speed": float(speed),
        "init_speed": float(speed),
    }
    vehicle.update(extra)
    return vehicle


def _apply_cruise_warmup_offsets(vehicles, ego_speed, cruise_warmup_s):
    # Every vehicle cruises at its own nominal speed during the warmup, so the
    # configured gaps would be partly consumed before control begins. Push each
    # vehicle forward by the distance the ego gains on it during the warmup:
    # the ego shifts by zero and every gap is back at its configured value the
    # moment the warmup ends. Goals are left alone -- they sit far beyond what
    # anything reaches inside max_sim_time_s.
    if cruise_warmup_s <= 0.0:
        return vehicles
    for vehicle in vehicles:
        shift = (float(ego_speed) - float(vehicle["nominal_speed"])) * float(cruise_warmup_s)
        vehicle["start_longitudinal_offset"] = float(vehicle["start_longitudinal_offset"]) + shift
    return vehicles


def _common_params(params):
    merged = {
        "ego_speed": 12.0,
        "target_speed": 10.0,
        "target_start_gap": 40.0,
        "trigger_distance": 25.0,
        "max_sim_time_s": 12.0,
        "route_goal_s": 220.0,
        "traffic_goal_margin_s": 180.0,
        "ego_horizon": 15,
        "lane_width": LANE_WIDTH_M,
        "lane_shift_right": DEFAULT_LANE_SHIFT_RIGHT,
        "spawn_settle_s": DEFAULT_SPAWN_SETTLE_S,
        "spawn_settle_realtime": False,
        "cruise_warmup_s": DEFAULT_CRUISE_WARMUP_S,
        "approach_time_s": 0.0,
        "ego_gap_at_trigger": 0.0,
        # Cut-in kinds: lane change as a duration (0 = use lane_change_distance).
        # A fixed distance takes longer at lower cut-in speeds, and the cut-in
        # vehicle, closing on its own lead, hit that lead while still
        # overlapping its lane (TV 11/13 m/s with 45-55 m); a duration keeps the
        # manoeuvre identical in the cut-in vehicle's frame at every speed.
        "lane_change_time_s": 0.0,
        "carla_timeout_period": 30.0,
    }
    merged.update(params or {})
    # Expressing the approach phase as a duration instead of a distance keeps it
    # equal across ego speeds. The cut-in vehicle closes on its own lead the
    # whole time, so a distance-based gap would leave that lead gap -- and the
    # merge urgency it represents -- different for every ego speed.
    approach_time_s = float(merged.get("approach_time_s") or 0.0)
    if approach_time_s > 0.0:
        closing = float(merged["ego_speed"]) - float(merged["target_speed"])
        if closing <= 0.0:
            raise ValueError(
                "approach_time_s needs ego_speed > target_speed so the ego closes "
                f"on the cut-in vehicle (got {merged['ego_speed']} vs {merged['target_speed']})")
        merged["target_start_gap"] = float(merged["trigger_distance"]) + approach_time_s * closing
    return merged


def _lead_gap_trigger_time(p):
    """Seconds until the vehicle at ``target_speed`` closes on the lead in its
    own lane (``target_lead_gap`` ahead, ``target_lead_speed_delta`` slower)
    to ``trigger_distance``; raises if it never gets there."""
    target_speed = float(p["target_speed"])
    lead_speed = max(1.0, target_speed + float(p["target_lead_speed_delta"]))
    closing_on_lead = target_speed - lead_speed
    if closing_on_lead <= 0.0:
        raise ValueError(
            "lead-gap trigger needs target_speed > target lead speed "
            f"(got {target_speed} vs {lead_speed})")
    t_star = (float(p["target_lead_gap"]) - float(p["trigger_distance"])) / closing_on_lead
    if t_star <= 0.0:
        raise ValueError(
            "target_lead_gap must exceed trigger_distance for the cut-in vehicle to "
            f"reach the trigger (got {p['target_lead_gap']} vs {p['trigger_distance']})")
    return t_star


def _apply_lead_gap_trigger(p):
    """Cut-in kinds: the cut-in vehicle starts its lane change when it is
    ``trigger_distance`` behind the lead in its own lane (NGSIM's
    lead-gap-at-manoeuvre-start), not when the ego gets close.  Tying the
    trigger to the ego made a cautious ego delay the manoeuvre until the cut-in
    vehicle ran into its own lead.  The moment of the manoeuvre then follows
    from the lead geometry alone, and ``ego_gap_at_trigger`` places the ego so
    that its gap to the cut-in vehicle at that moment is the calibrated value.
    """
    t_star = _lead_gap_trigger_time(p)
    p["lane_change_trigger_mode"] = "lead_gap"
    p["approach_time_s"] = t_star
    ego_gap = float(p.get("ego_gap_at_trigger") or 0.0)
    if ego_gap > 0.0:
        p["target_start_gap"] = ego_gap + t_star * (float(p["ego_speed"]) - float(p["target_speed"]))
    return p


# no_cutin_decel: the cut-in vehicle never leaves its lane.  Its speed cap
# toward the lead in its own lane (distance_triggered_lane_change_agent) is
# re-tuned so that v_cap reaches the nominal speed exactly at
# trigger_distance -- the vehicle starts decelerating there instead of
# changing lane -- and it settles 4.5 + min_gap behind the lead at the
# lead's speed.  The ego placement (ego_gap_at_trigger) is unchanged, so the
# ego meets the same approach as in the cut-in kinds.
NO_CUTIN_SPEED_CAP_GAIN = 0.5


def _no_cutin_speed_cap(p, lead_speed):
    # 4.5 is SPEED_CAP_VEHICLE_LENGTH_M of the lane-change agent.
    min_gap = (float(p["trigger_distance"]) - 4.5
               - (float(p["target_speed"]) - float(lead_speed)) / NO_CUTIN_SPEED_CAP_GAIN)
    if min_gap < 1.0:
        raise ValueError(
            "no_cutin_decel needs a larger trigger_distance or a smaller target-to-lead "
            f"speed difference (speed cap min gap {min_gap:.2f} m < 1.0 m)")
    return {
        "lane_change_trigger_distance": 0.0,
        "lane_change_trigger_mode": "never",
        "speed_cap_min_gap_m": min_gap,
        "speed_cap_gain": NO_CUTIN_SPEED_CAP_GAIN,
    }


def _traffic_goal_s(p, start_s, speed):
    sim_travel = float(speed) * float(p["max_sim_time_s"])
    return max(
        float(p["route_goal_s"]),
        float(start_s) + sim_travel + float(p.get("traffic_goal_margin_s", 180.0)),
    )


def make_cutin_scenario(params=None, aggressive=False, ego_lead=False, no_cutin=False):
    p = _common_params(params)
    if aggressive:
        p.setdefault("lane_change_distance_same_lane", 1.0)
        p.setdefault("lane_change_distance", 10.0)
        p.setdefault("outer_blocker_count", 3)
        p.setdefault("outer_blocker_spacing", 9.0)
    else:
        p.setdefault("lane_change_distance_same_lane", 5.0)
        p.setdefault("lane_change_distance", 24.0)
        p.setdefault("outer_blocker_count", 3)
        p.setdefault("outer_blocker_spacing", 14.0)
    lane_change_time_s = float(p.get("lane_change_time_s") or 0.0)
    if lane_change_time_s > 0.0:
        p["lane_change_distance"] = lane_change_time_s * float(p["target_speed"])
    p.setdefault("outer_blocker_speed_delta", -1.5)
    p.setdefault("target_left_offset", 3.5)
    p.setdefault("outer_blocker_left_offset", 7.0)
    p.setdefault("target_lead_gap", 16.0 if aggressive else 24.0)
    p.setdefault("target_lead_speed_delta", -3.0 if aggressive else -2.0)
    p.setdefault("right_lane_target_left_offset", -3.5)
    p.setdefault("right_lane_target_gap", p["target_start_gap"] + 8.0)
    p.setdefault("right_lane_target_speed_delta", -1.5)
    p.setdefault(
        "right_lane_target_speed",
        max(1.0, float(p["target_speed"]) + float(p["right_lane_target_speed_delta"])),
    )
    if ego_lead:
        p.setdefault("ego_lead_gap", 65.0)
        p.setdefault("ego_lead_speed_delta", -1.0)
        p.setdefault(
            "ego_lead_speed",
            max(1.0, float(p["ego_speed"]) + float(p["ego_lead_speed_delta"])),
        )
    ego_left_offset = _right_shifted_left_offset(0.0, p)
    target_left_offset = _right_shifted_left_offset(p["target_left_offset"], p)
    outer_blocker_left_offset = _right_shifted_left_offset(p["outer_blocker_left_offset"], p)
    right_lane_target_left_offset = _right_shifted_left_offset(
        p.get("right_lane_target_left_offset", -3.5), p)

    _apply_lead_gap_trigger(p)
    target_lead_speed = max(1.0, float(p["target_speed"]) + float(p["target_lead_speed_delta"]))
    if no_cutin:
        target_trigger = _no_cutin_speed_cap(p, target_lead_speed)
    else:
        target_trigger = {
            "lane_change_trigger_distance": p["trigger_distance"],
            "lane_change_trigger_mode": "lead_gap",
        }
    vehicles = [
        _vehicle(
            "target_cutin",
            "distance_triggered_lane_change",
            target_left_offset,
            p["target_start_gap"],
            p["route_goal_s"],
            p["target_speed"],
            "186, 0, 0",
            lane_change_distance_same_lane=p["lane_change_distance_same_lane"],
            lane_change_distance_other_lane=120.0,
            lane_change_distance=p["lane_change_distance"],
            **target_trigger,
        )
    ]
    target_lead_start_s = p["target_start_gap"] + p["target_lead_gap"]
    vehicles.append(
        _vehicle(
            "traffic_cutin_lane_lead",
            "fixed_lane_speed",
            target_left_offset,
            target_lead_start_s,
            _traffic_goal_s(p, target_lead_start_s, target_lead_speed),
            target_lead_speed,
            "255, 180, 0",
        )
    )
    if ego_lead:
        ego_lead_start_s = float(p["ego_lead_gap"])
        vehicles.append(
            _vehicle(
                "target_ego_lane_lead",
                "fixed_lane_speed",
                ego_left_offset,
                ego_lead_start_s,
                _traffic_goal_s(p, ego_lead_start_s, p["ego_lead_speed"]),
                p["ego_lead_speed"],
                "255, 180, 0",
            )
        )
    right_lane_target_start_s = float(p["right_lane_target_gap"])
    vehicles.append(
        _vehicle(
            "target_right_lane_straight",
            "fixed_lane_speed",
            right_lane_target_left_offset,
            right_lane_target_start_s,
            _traffic_goal_s(p, right_lane_target_start_s, p["right_lane_target_speed"]),
            p["right_lane_target_speed"],
            "65, 63, 197",
        )
    )
    blocker_speed = max(1.0, float(p["target_speed"]) + float(p["outer_blocker_speed_delta"]))
    for idx in range(int(p["outer_blocker_count"])):
        blocker_start_s = p["target_start_gap"] + (idx + 1) * p["outer_blocker_spacing"]
        vehicles.append(
            _vehicle(
                f"traffic_outer_blocker_{idx}",
                "fixed_lane_speed",
                outer_blocker_left_offset,
                blocker_start_s,
                _traffic_goal_s(p, blocker_start_s, blocker_speed),
                blocker_speed,
                "255, 180, 0",
            )
        )
    vehicles.append(
        _vehicle(
            "ego",
            "acc_nair_smpc",
            ego_left_offset,
            0.0,
            p["route_goal_s"],
            p["ego_speed"],
            "0, 128, 0",
            N=int(p["ego_horizon"]),
        )
    )
    return {
        "carla_params": _base_carla_params(
            p["max_sim_time_s"],
            p["spawn_settle_s"],
            p["spawn_settle_realtime"],
            p["carla_timeout_period"],
            cruise_warmup_s=p["cruise_warmup_s"],
        ),
        "drone_viz_params": _base_drone_params(p["lane_shift_right"], p["lane_width"]),
        "vehicle_params": _apply_cruise_warmup_offsets(
            vehicles, p["ego_speed"], p["cruise_warmup_s"]),
    }


def make_cutout_scenario(params=None, with_lead=False, trigger_mode="ego_gap"):
    # "ego_gap" (cutout_with_lead / cutout_no_lead): the vehicle ahead leaves
    # the ego lane once the ego is within trigger_distance.  The paper kinds
    # start the ego target_start_gap behind a slower lead vehicle (LV) that it
    # settles behind in ACC, and the LV leaves on its own: "time"
    # (cutout_no_sublv) trigger_time_s after control starts, or "lead_gap"
    # (cutout_sublv) when its gap to a slow lead ahead of it (subLV, placed
    # like the cut-in kinds' lead) closes to trigger_distance -- the ego then
    # has to slow down for the subLV.
    params = dict(params or {})
    if trigger_mode != "ego_gap":
        params.setdefault("target_start_gap", 30.0)
    p = _common_params(params)
    p.setdefault("lane_change_distance_same_lane", 5.0 if trigger_mode == "ego_gap" else 0.0)
    p.setdefault("lane_change_distance", 18.0)
    lane_change_time_s = float(p.get("lane_change_time_s") or 0.0)
    if trigger_mode != "ego_gap" and lane_change_time_s > 0.0:
        p["lane_change_distance"] = lane_change_time_s * float(p["target_speed"])
    p.setdefault("cutout_direction", "left")
    target_trigger = {"lane_change_trigger_distance": p["trigger_distance"]}
    if trigger_mode == "time":
        # 8 s / 65 m (an 8 s approach): the ego ACC undershoots to ~11 m/s
        # while closing 17 -> 13 m/s and is still recovering at 5-6 s.
        p.setdefault("trigger_time_s", 8.0)
        target_trigger.update({
            "lane_change_trigger_mode": "time",
            "lane_change_trigger_time_s": p["trigger_time_s"],
        })
    elif trigger_mode == "lead_gap":
        p.setdefault("target_lead_gap", 65.0)
        p.setdefault("target_lead_speed_delta", -6.0)
        _lead_gap_trigger_time(p)
        p["lead_gap"] = float(p["target_lead_gap"])
        p["lead_speed"] = max(1.0, float(p["target_speed"]) + float(p["target_lead_speed_delta"]))
        target_trigger["lane_change_trigger_mode"] = "lead_gap"
    else:
        p.setdefault("lead_gap", 32.0)
        p.setdefault("lead_speed", max(1.0, p["target_speed"] - 1.0))
    ego_lane_left_offset = _right_shifted_left_offset(0.0, p)

    vehicles = [
        _vehicle(
            "target_cutout",
            "distance_triggered_cutout",
            ego_lane_left_offset,
            p["target_start_gap"],
            p["route_goal_s"],
            p["target_speed"],
            "186, 0, 0",
            lane_change_distance_same_lane=p["lane_change_distance_same_lane"],
            lane_change_distance_other_lane=120.0,
            lane_change_distance=p["lane_change_distance"],
            cutout_direction=p["cutout_direction"],
            **target_trigger,
        )
    ]
    if with_lead:
        vehicles.append(
            _vehicle(
                "target_lead_after_cutout",
                "fixed_lane_speed",
                ego_lane_left_offset,
                p["target_start_gap"] + p["lead_gap"],
                p["route_goal_s"],
                p["lead_speed"],
                "255, 180, 0",
            )
        )
    vehicles.append(
        _vehicle(
            "ego",
            "acc_nair_smpc",
            ego_lane_left_offset,
            0.0,
            p["route_goal_s"],
            p["ego_speed"],
            "0, 128, 0",
            N=int(p["ego_horizon"]),
        )
    )
    return {
        "carla_params": _base_carla_params(
            p["max_sim_time_s"],
            p["spawn_settle_s"],
            p["spawn_settle_realtime"],
            p["carla_timeout_period"],
            cruise_warmup_s=p["cruise_warmup_s"],
        ),
        "drone_viz_params": _base_drone_params(p["lane_shift_right"], p["lane_width"]),
        "vehicle_params": _apply_cruise_warmup_offsets(
            vehicles, p["ego_speed"], p["cruise_warmup_s"]),
    }


def make_scenario(kind, params=None):
    if kind == "cutin":
        return make_cutin_scenario(params, aggressive=False)
    if kind == "cutin_with_ego_lead":
        return make_cutin_scenario(params, aggressive=False, ego_lead=True)
    if kind == "aggressive_cutin":
        return make_cutin_scenario(params, aggressive=True)
    if kind == "aggressive_cutin_with_ego_lead":
        return make_cutin_scenario(params, aggressive=True, ego_lead=True)
    if kind == "no_cutin_decel":
        return make_cutin_scenario(params, aggressive=False, no_cutin=True)
    if kind == "cutout_with_lead":
        return make_cutout_scenario(params, with_lead=True)
    if kind == "cutout_no_lead":
        return make_cutout_scenario(params, with_lead=False)
    if kind == "cutout_no_sublv":
        return make_cutout_scenario(params, with_lead=False, trigger_mode="time")
    if kind == "cutout_sublv":
        return make_cutout_scenario(params, with_lead=True, trigger_mode="lead_gap")
    raise ValueError(f"Unsupported scenario kind: {kind}")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_default_scenarios(output_dir=SCENARIO_DIR):
    output_dir = Path(output_dir)
    files = {
        "scenario_lk_acc_cutin_blocked.json": make_scenario("cutin"),
        "scenario_lk_acc_cutin_with_ego_lead.json": make_scenario("cutin_with_ego_lead"),
        "scenario_lk_acc_cutin_aggressive_blocked.json": make_scenario("aggressive_cutin"),
        "scenario_lk_acc_cutin_aggressive_with_ego_lead.json": make_scenario("aggressive_cutin_with_ego_lead"),
        "scenario_lk_acc_cutout_with_lead.json": make_scenario("cutout_with_lead"),
        "scenario_lk_acc_cutout_no_lead.json": make_scenario("cutout_no_lead"),
    }
    for name, scenario in files.items():
        write_json(output_dir / name, scenario)
    return sorted(str(output_dir / name) for name in files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(SCENARIO_DIR))
    parser.add_argument("--write-recommended-sweep", action="store_true")
    args = parser.parse_args()

    written = write_default_scenarios(args.output_dir)
    for path in written:
        print(path)
    if args.write_recommended_sweep:
        sweep_path = Path(args.output_dir).parent / "devel" / "configs" / "acc_scenario_sweep_recommended.json"
        write_json(sweep_path, RECOMMENDED_SWEEP_RANGES)
        print(sweep_path)


if __name__ == "__main__":
    main()
