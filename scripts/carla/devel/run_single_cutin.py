import argparse
import copy
import json
import os
import pickle
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SMPC_ROOT = os.path.abspath(os.path.join(CARLA_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, CARLA_SCRIPT_DIR)

from run_all_scenarios import run_with_tvs


DEFAULT_SCENARIO = os.path.join(
    CARLA_SCRIPT_DIR, "scenarios", "scenario_lk_cutin_triggered.json")


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _scene_from_scenario_path(scenario_path):
    scenario_name = os.path.basename(scenario_path)
    if "lk" in scenario_name:
        return "highway"
    return "intersection"


def _default_output_dir(scenario_path, ego_policy):
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
    return os.path.join(
        SMPC_ROOT, "results", "cutin_single", f"{scenario_name}_{ego_policy}")


def _policy_logs_from_result(savedir):
    pkl_path = os.path.join(savedir, "scenario_result.pkl")
    if not os.path.exists(pkl_path):
        return {}

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    policy_logs = {}
    for actor_key, actor_result in result.items():
        if isinstance(actor_result, dict) and "policy_log" in actor_result:
            policy_logs[actor_key] = actor_result["policy_log"]
    return policy_logs


def _write_summary(savedir, ran_successfully, error=None):
    summary = {
        "ran_successfully": ran_successfully,
        "error": error,
        "policy_logs": _policy_logs_from_result(savedir),
    }
    _write_json(os.path.join(savedir, "summary.json"), summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--ego-policy", default="smpc_var_risk")
    parser.add_argument("--ego-init", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    scenario_dict = _load_json(args.scenario)
    ego_init_dict = _load_json(args.ego_init) if args.ego_init else {}
    savedir = args.output_dir or _default_output_dir(args.scenario, args.ego_policy)
    os.makedirs(savedir, exist_ok=True)

    _write_json(os.path.join(savedir, "resolved_config.json"), {
        "scenario": scenario_dict,
        "ego_init": ego_init_dict,
        "ego_policy": args.ego_policy,
    })

    ran_successfully = False
    error = None
    try:
        ran_successfully = run_with_tvs(
            _scene_from_scenario_path(args.scenario),
            copy.deepcopy(scenario_dict),
            ego_init_dict,
            args.ego_policy,
            savedir)
    except Exception as exc:
        error = repr(exc)

    _write_summary(savedir, ran_successfully, error=error)

    if error is not None:
        raise RuntimeError(error)

    print(f"savedir={savedir}")
    print(f"ran_successfully={ran_successfully}")


if __name__ == "__main__":
    main()
