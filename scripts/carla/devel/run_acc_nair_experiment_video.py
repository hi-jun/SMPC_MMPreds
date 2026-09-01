import argparse
import json
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SMPC_ROOT = os.path.abspath(os.path.join(CARLA_SCRIPT_DIR, "..", ".."))


SCENARIOS = {
    "cutin": {
        "runner": "run_acc_nair_cutin_smoke.py",
        "visualizer": "visualize_acc_nair_cutin_smoke.py",
        "result_subdir": "acc_nair_cutin_validation",
        "probability_arg": "cutin_probability",
        "video_defaults_from_runner": {
            "horizon": "horizon",
            "dt": "dt",
            "cutin_duration": "cutin_duration",
        },
    },
    "cutout": {
        "runner": "run_acc_nair_cutout_smoke.py",
        "visualizer": "visualize_acc_nair_cutout_smoke.py",
        "result_subdir": "acc_nair_cutout_validation",
        "probability_arg": "cutout_probability",
        "video_defaults_from_runner": {
            "horizon": "horizon",
            "dt": "dt",
            "cutout_duration": "cutout_duration",
            "cutout_start_delay": "cutout_start_delay",
            "target_d_final": "target_d_final",
        },
    },
}


AGGRESSIVE_RUNNER_ARGS = {
    "cutin": {
        "cutin_start_delay": 0.0,
        "cutin_duration": 0.8,
        "ego_v0": 13.0,
        "target_s0": 22.0,
        "target_v0": 8.0,
        "actual_target_accel": -1.0,
    },
    "cutout": {
        "cutout_start_delay": 0.0,
        "cutout_duration": 0.8,
        "ego_v0": 10.0,
        "target_v0": 7.0,
        "initial_gap_margin": 0.0,
    },
}


def _strip_json_comments(text):
    result = []
    idx = 0
    in_string = False
    escape = False
    while idx < len(text):
        char = text[idx]
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            idx += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            idx += 1
            continue
        if char == "/" and next_char == "/":
            idx += 2
            while idx < len(text) and text[idx] not in "\r\n":
                idx += 1
            continue
        if char == "/" and next_char == "*":
            idx += 2
            while idx + 1 < len(text) and not (text[idx] == "*" and text[idx + 1] == "/"):
                idx += 1
            idx += 2
            continue
        result.append(char)
        idx += 1
    return "".join(result)


def _load_config(path):
    with open(path, "r") as f:
        config = json.loads(_strip_json_comments(f.read()))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    return config


def _cli_name(name):
    return "--" + name.replace("_", "-")


def _append_cli_args(command, values):
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(_cli_name(key))
            continue
        command.extend([_cli_name(key), str(value)])


def _scenario_config(config):
    scenario = config.get("scenario")
    if scenario not in SCENARIOS:
        raise ValueError("scenario must be one of: %s" % ", ".join(sorted(SCENARIOS)))
    name = config.get("name")
    if not name:
        raise ValueError("config must include a non-empty name")
    return scenario, str(name), SCENARIOS[scenario]


def _output_dir(config, scenario_info, name):
    output_dir = config.get("output_dir")
    if output_dir:
        return os.path.abspath(output_dir)
    output_root = config.get("output_root")
    if output_root:
        output_root = os.path.abspath(output_root)
    else:
        output_root = os.path.join(SMPC_ROOT, "results", scenario_info["result_subdir"])
    return os.path.join(output_root, name)


def _runner_args(config, scenario, scenario_info, output_dir):
    runner_args = {}
    if config.get("aggressive", False):
        runner_args.update(AGGRESSIVE_RUNNER_ARGS[scenario])
    runner_args.update(config.get("runner_args", {}))
    if "probability" in config and scenario_info["probability_arg"] not in runner_args:
        runner_args[scenario_info["probability_arg"]] = config["probability"]
    runner_args["output_dir"] = output_dir
    return runner_args


def _video_args(config, scenario_info, runner_args, output_name):
    video_args = {}
    for video_key, runner_key in scenario_info["video_defaults_from_runner"].items():
        if runner_key in runner_args:
            video_args[video_key] = runner_args[runner_key]
    video_args.update(config.get("video_args", {}))
    video_args["output_name"] = output_name
    return video_args


def _write_resolved_config(path, config):
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def run(config_path):
    config = _load_config(config_path)
    scenario, name, scenario_info = _scenario_config(config)
    output_dir = _output_dir(config, scenario_info, name)
    os.makedirs(output_dir, exist_ok=True)

    runner_args = _runner_args(config, scenario, scenario_info, output_dir)
    output_name = config.get("video_name") or ("%s.mp4" % name)
    video_args = _video_args(config, scenario_info, runner_args, output_name)

    env = os.environ.copy()
    env_updates = config.get("env", {})
    if env_updates:
        env.update({str(key): str(value) for key, value in env_updates.items()})

    runner_command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, scenario_info["runner"]),
    ]
    _append_cli_args(runner_command, runner_args)
    runner_command.extend(config.get("runner_extra_args", []))

    visualizer_command = [
        sys.executable,
        os.path.join(SCRIPT_DIR, scenario_info["visualizer"]),
        output_dir,
    ]
    _append_cli_args(visualizer_command, video_args)
    visualizer_command.extend(config.get("video_extra_args", []))

    resolved = {
        "config_path": os.path.abspath(config_path),
        "scenario": scenario,
        "name": name,
        "aggressive": bool(config.get("aggressive", False)),
        "output_dir": output_dir,
        "runner_args": runner_args,
        "video_args": video_args,
        "runner_command": runner_command,
        "visualizer_command": visualizer_command,
    }
    _write_resolved_config(os.path.join(output_dir, "resolved_experiment_config.json"), resolved)

    subprocess.run(runner_command, check=True, cwd=SMPC_ROOT, env=env)
    subprocess.run(visualizer_command, check=True, cwd=SMPC_ROOT, env=env)

    video_path = os.path.join(output_dir, output_name)
    result = {
        "output_dir": output_dir,
        "summary_path": os.path.join(output_dir, "summary.json"),
        "trajectory_path": os.path.join(output_dir, "trajectory.csv"),
        "video_path": video_path,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="JSON/JSONC config with scenario and name")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
