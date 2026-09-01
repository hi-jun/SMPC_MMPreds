import argparse
import copy
import itertools
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, CARLA_SCRIPT_DIR)

from run_all_scenarios import run_with_tvs
from run_single_cutin import (
    DEFAULT_SCENARIO,
    SMPC_ROOT,
    _load_json,
    _scene_from_scenario_path,
    _write_json,
    _write_summary,
)


def _parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _find_vehicle(scenario_dict, predicate):
    for vehicle_params in scenario_dict["vehicle_params"]:
        if predicate(vehicle_params):
            return vehicle_params
    raise ValueError("Could not find requested vehicle params in scenario.")


def _make_run_scenario(base_scenario, params):
    scenario = copy.deepcopy(base_scenario)
    target = _find_vehicle(scenario, lambda vp: "target" in vp["role"])
    ego = _find_vehicle(scenario, lambda vp: vp["role"] == "ego")

    target["policy_type"] = "distance_triggered_lane_change"
    target["lane_change_trigger_distance"] = params["lane_change_trigger_distance"]
    target["start_longitudinal_offset"] = params["target_start_longitudinal_offset"]
    target["nominal_speed"] = params["target_speed"]
    target["init_speed"] = params["target_speed"]
    target["goal_left_offset"] = target["start_left_offset"]
    target.pop("lane_change_direction", None)

    ego["start_left_offset"] = 0.0
    ego["goal_left_offset"] = 0.0
    ego["start_longitudinal_offset"] = params["ego_start_longitudinal_offset"]
    ego["init_speed"] = params["ego_speed"]

    return scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--ego-policy", default="smpc_var_risk")
    parser.add_argument("--output-root", default=os.path.join(SMPC_ROOT, "results", "cutin_sweep"))
    parser.add_argument("--trigger-distances", default="15,20,30,40")
    parser.add_argument("--target-speeds", default="8,10,12")
    parser.add_argument("--target-start-offsets", default="20,30,40")
    parser.add_argument("--ego-speeds", default="10,12")
    parser.add_argument("--ego-start-offsets", default="0")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_scenario = _load_json(args.scenario)
    os.makedirs(args.output_root, exist_ok=True)

    combinations = itertools.product(
        _parse_float_list(args.trigger_distances),
        _parse_float_list(args.target_speeds),
        _parse_float_list(args.target_start_offsets),
        _parse_float_list(args.ego_speeds),
        _parse_float_list(args.ego_start_offsets),
    )

    scene = _scene_from_scenario_path(args.scenario)
    for idx, combo in enumerate(combinations, start=1):
        if args.max_runs > 0 and idx > args.max_runs:
            break

        params = {
            "lane_change_trigger_distance": combo[0],
            "target_speed": combo[1],
            "target_start_longitudinal_offset": combo[2],
            "ego_speed": combo[3],
            "ego_start_longitudinal_offset": combo[4],
        }
        run_scenario = _make_run_scenario(base_scenario, params)
        savedir = os.path.join(args.output_root, f"run_{idx:04d}")
        os.makedirs(savedir, exist_ok=True)
        _write_json(os.path.join(savedir, "resolved_config.json"), {
            "scenario": run_scenario,
            "ego_policy": args.ego_policy,
            "sweep_params": params,
        })

        if args.dry_run:
            _write_summary(savedir, False, error="dry_run")
            print(f"run_{idx:04d}: dry_run=True")
            continue

        ran_successfully = False
        error = None
        try:
            ran_successfully = run_with_tvs(
                scene, copy.deepcopy(run_scenario), {}, args.ego_policy, savedir)
        except Exception as exc:
            error = repr(exc)

        _write_summary(savedir, ran_successfully, error=error)
        print(f"run_{idx:04d}: ran_successfully={ran_successfully} error={error}")


if __name__ == "__main__":
    main()
