from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CARLA_SCRIPT_DIR = SCRIPT_DIR.parent
SMPC_ROOT = CARLA_SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CARLA_SCRIPT_DIR))

from acc_scenario_factory import make_scenario, write_json  # noqa: E402
from analyze_acc_scenario_result import analyze_summary  # noqa: E402
from run_all_scenarios import run_with_tvs  # noqa: E402


def _parse_csv(value, cast=float):
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _bool_kind(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"Invalid bool value: {value}")


def _default_output_root():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SMPC_ROOT / "results" / "acc_scenario_sweep" / timestamp


def _param_grid(args):
    aggressive = args.kind in ("aggressive_cutin", "aggressive_cutin_with_ego_lead")
    same_lane_distances = args.same_lane_distances or ("0,2,4" if aggressive else "5")
    lane_change_distances = args.lane_change_distances or ("8,12,16" if aggressive else "18")
    outer_blocker_counts = args.outer_blocker_counts or ("3,4" if aggressive else "3")
    outer_blocker_spacings = args.outer_blocker_spacings or ("8,10,12" if aggressive else "12")
    outer_blocker_speed_deltas = args.outer_blocker_speed_deltas or "-1.5"
    common = {
        "ego_speed": _parse_csv(args.ego_speeds),
        "target_speed": _parse_csv(args.target_speeds),
        "target_start_gap": _parse_csv(args.target_start_gaps),
        "trigger_distance": _parse_csv(args.trigger_distances),
        "lane_change_distance_same_lane": _parse_csv(same_lane_distances),
        "lane_change_distance": _parse_csv(lane_change_distances),
        "max_sim_time_s": [args.max_sim_time],
        "ego_horizon": _parse_csv(args.ego_horizons, int),
        "spawn_settle_s": [args.spawn_settle_s],
        "spawn_settle_realtime": [args.spawn_settle_realtime],
        "carla_timeout_period": [args.carla_timeout_period],
        "lane_shift_right": [args.lane_shift_right],
    }
    if args.kind in (
            "cutin",
            "aggressive_cutin",
            "cutin_with_ego_lead",
            "aggressive_cutin_with_ego_lead"):
        common.update({
            "target_lead_gap": _parse_csv(args.target_lead_gaps),
            "target_lead_speed_delta": _parse_csv(args.target_lead_speed_deltas),
            "outer_blocker_count": _parse_csv(outer_blocker_counts, int),
            "outer_blocker_spacing": _parse_csv(outer_blocker_spacings),
            "outer_blocker_speed_delta": _parse_csv(outer_blocker_speed_deltas),
        })
        if args.kind in ("cutin_with_ego_lead", "aggressive_cutin_with_ego_lead"):
            common.update({
                "ego_lead_gap": _parse_csv(args.ego_lead_gaps),
                "ego_lead_speed_delta": _parse_csv(args.ego_lead_speed_deltas),
            })
    if args.kind in ("cutout_with_lead", "cutout_no_lead"):
        common.update({
            "cutout_direction": _parse_csv(args.cutout_directions, str),
        })
        if args.kind == "cutout_with_lead":
            common.update({
                "lead_gap": _parse_csv(args.lead_gaps),
                "lead_speed": _parse_csv(args.lead_speeds),
            })
    keys = list(common.keys())
    for values in itertools.product(*(common[key] for key in keys)):
        yield dict(zip(keys, values))


def _write_summary(savedir, ran_successfully, error=None):
    summary = {
        "ran_successfully": ran_successfully,
        "error": error,
        "policy_logs": _policy_logs_from_result(savedir),
    }
    write_json(Path(savedir) / "summary.json", summary)
    return summary


def _successful_manifest_rows(output_root):
    manifest_path = Path(output_root) / "sweep_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r") as f:
            rows = json.load(f)
    except Exception:
        return {}
    successful = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("ran_successfully"):
            continue
        try:
            run_idx = int(row.get("run_idx"))
        except (TypeError, ValueError):
            continue
        successful[run_idx] = row
    return successful


def _values_match(expected, actual):
    if isinstance(expected, bool):
        return bool(actual) == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= 1e-9
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _params_match(expected, actual):
    if not isinstance(actual, dict):
        return False
    for key, value in expected.items():
        if key not in actual or not _values_match(value, actual[key]):
            return False
    return True


def _successful_run_dir_row(output_root, kind, run_idx, params):
    run_dir = Path(output_root) / f"{kind}_{run_idx:04d}"
    summary_path = run_dir / "summary.json"
    resolved_path = run_dir / "resolved_config.json"
    metrics_path = run_dir / "metrics.json"
    pkl_path = run_dir / "scenario_result.pkl"
    if not (
            summary_path.exists()
            and resolved_path.exists()
            and metrics_path.exists()
            and pkl_path.exists()):
        return None
    try:
        with summary_path.open("r") as f:
            summary = json.load(f)
        with resolved_path.open("r") as f:
            resolved = json.load(f)
    except Exception:
        return None
    if not summary.get("ran_successfully"):
        return None
    if not _params_match(params, resolved.get("sweep_params", {})):
        return None
    row = {
        "run_idx": int(run_idx),
        "ran_successfully": True,
        "error": summary.get("error"),
        "port": int(resolved.get("port", 0)),
    }
    row.update(params)
    return row


def _successful_resume_rows(output_root, args, param_rows):
    successful = _successful_manifest_rows(output_root)
    for run_idx, params in enumerate(param_rows, start=1):
        if run_idx in successful and _params_match(params, successful[run_idx]):
            continue
        row = _successful_run_dir_row(output_root, args.kind, run_idx, params)
        if row is not None:
            successful[run_idx] = row
    return successful


def _policy_logs_from_result(savedir):
    import pickle

    pkl_path = Path(savedir) / "scenario_result.pkl"
    if not pkl_path.exists():
        return {}
    with pkl_path.open("rb") as f:
        result = pickle.load(f)
    policy_logs = {}
    for actor_key, actor_result in result.items():
        if isinstance(actor_result, dict) and "policy_log" in actor_result:
            policy_logs[actor_key] = actor_result["policy_log"]
    return policy_logs


def _start_carla_server(carla_root, port, log_path):
    carla_root = Path(carla_root)
    cmd = [
        str(carla_root / "CarlaUE4.sh"),
        "-RenderOffScreen",
        "-nosound",
        "-quality-level=Low",
        f"-carla-rpc-port={int(port)}",
    ]
    log_file = Path(log_path).open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(carla_root),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, log_file


def _wait_for_carla(port, timeout_s=60.0):
    import carla

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            client = carla.Client("localhost", int(port))
            client.set_timeout(2.0)
            client.get_server_version()
            return True
        except Exception:
            time.sleep(1.0)
    return False


def _stop_carla_server(proc, log_file):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            proc.wait(timeout=10.0)
    if log_file is not None:
        log_file.close()


def _run_one(args, run_idx, params, output_root):
    run_dir = output_root / f"{args.kind}_{run_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scenario = make_scenario(args.kind, params)
    run_port = int(args.port)
    if args.fresh_server_per_run:
        run_port += (int(run_idx) - 1) * int(args.port_step)
    scenario["carla_params"]["port"] = run_port
    write_json(run_dir / "resolved_config.json", {
        "kind": args.kind,
        "ego_policy": args.ego_policy,
        "port": run_port,
        "sweep_params": params,
        "scenario": scenario,
    })

    if args.dry_run:
        summary = {"ran_successfully": False, "error": "dry_run", "policy_logs": {}}
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "metrics.json", analyze_summary(summary))
        return False, "dry_run"

    proc = None
    log_file = None
    try:
        if args.fresh_server_per_run:
            proc, log_file = _start_carla_server(
                args.carla_root,
                run_port,
                run_dir / "carla_server.log",
            )
            if not _wait_for_carla(run_port, timeout_s=args.server_timeout):
                raise RuntimeError("CARLA server did not become ready before timeout")

        ran_successfully = run_with_tvs(
            "highway",
            copy.deepcopy(scenario),
            {},
            args.ego_policy,
            str(run_dir),
        )
        summary = _write_summary(run_dir, ran_successfully, error=None)
        metrics = analyze_summary(summary)
        write_json(run_dir / "metrics.json", metrics)
        return ran_successfully, None
    except Exception as exc:
        summary = _write_summary(run_dir, False, error=repr(exc))
        metrics = analyze_summary(summary)
        write_json(run_dir / "metrics.json", metrics)
        return False, repr(exc)
    finally:
        if args.fresh_server_per_run:
            _stop_carla_server(proc, log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=[
                            "cutin",
                            "cutin_with_ego_lead",
                            "aggressive_cutin",
                            "aggressive_cutin_with_ego_lead",
                            "cutout_with_lead",
                            "cutout_no_lead",
                        ],
                        default="cutin")
    parser.add_argument("--ego-policy", default="acc_nair_smpc_stdan_3int")
    parser.add_argument("--output-root", default=str(_default_output_root()))
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip successful rows already present in sweep_manifest.json.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--port-step", type=int, default=2,
                        help="Port increment when --fresh-server-per-run is used.")
    parser.add_argument("--fresh-server-per-run", action="store_true")
    parser.add_argument("--carla-root", default=os.getenv("CARLA_ROOT", "/home/core-dev/CARLA/carla_0_9_13"))
    parser.add_argument("--server-timeout", type=float, default=60.0)
    parser.add_argument("--carla-timeout-period", type=float, default=30.0)
    parser.add_argument("--max-sim-time", type=float, default=12.0)
    parser.add_argument("--ego-horizons", default="15")
    parser.add_argument("--spawn-settle-s", type=float, default=1.5)
    parser.add_argument("--spawn-settle-realtime", dest="spawn_settle_realtime", action="store_true", default=False)
    parser.add_argument("--no-spawn-settle-realtime", dest="spawn_settle_realtime", action="store_false")
    parser.add_argument("--lane-shift-right", type=int, default=1)

    parser.add_argument("--ego-speeds", default="12")
    parser.add_argument("--target-speeds", default="10")
    parser.add_argument("--target-start-gaps", default="35,40")
    parser.add_argument("--trigger-distances", default="25")
    parser.add_argument("--same-lane-distances", default=None)
    parser.add_argument("--lane-change-distances", default=None)
    parser.add_argument("--target-lead-gaps", default="24")
    parser.add_argument("--target-lead-speed-deltas", default="-2")
    parser.add_argument("--ego-lead-gaps", default="65")
    parser.add_argument("--ego-lead-speed-deltas", default="-1")
    parser.add_argument("--outer-blocker-counts", default=None)
    parser.add_argument("--outer-blocker-spacings", default=None)
    parser.add_argument("--outer-blocker-speed-deltas", default=None)
    parser.add_argument("--cutout-directions", default="left")
    parser.add_argument("--lead-gaps", default="32")
    parser.add_argument("--lead-speeds", default="9")
    parser.add_argument("--stop-on-run-failure", dest="stop_on_run_failure", action="store_true", default=False)
    parser.add_argument("--no-stop-on-run-failure", dest="stop_on_run_failure", action="store_false")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "sweep_config.json", vars(args))
    param_rows = list(_param_grid(args))
    resume_rows = _successful_resume_rows(output_root, args, param_rows) if args.resume else {}

    rows = []
    for run_idx, params in enumerate(param_rows, start=1):
        if args.max_runs > 0 and run_idx > args.max_runs:
            break
        if run_idx in resume_rows:
            row = resume_rows[run_idx]
            rows.append(row)
            print(json.dumps({
                "run_idx": run_idx,
                "resume_skipped": True,
                "ran_successfully": row.get("ran_successfully"),
            }, sort_keys=True))
            continue
        ran, error = _run_one(args, run_idx, params, output_root)
        row = {
            "run_idx": run_idx,
            "ran_successfully": ran,
            "error": error,
            "port": int(args.port) + (
                (int(run_idx) - 1) * int(args.port_step)
                if args.fresh_server_per_run else 0
            ),
        }
        row.update(params)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
        if args.stop_on_run_failure and not ran and error != "dry_run":
            break

    write_json(output_root / "sweep_manifest.json", rows)
    failed = any(
        not row.get("ran_successfully") and row.get("error") != "dry_run"
        for row in rows
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
