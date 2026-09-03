from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SMPC_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_PYTHON = "/home/core-dev/anaconda3/envs/mmpreds-c0913/bin/python"
DEFAULT_CARLA_ROOT = "/home/core-dev/CARLA/carla_0_9_13"
DEFAULT_COMPARE_POLICIES = (
    "acc_nair_smpc_const_lead,"
    "acc_nair_smpc_stdan_3int_best_mode_scalar_chance_optimized_eta,"
    "acc_nair_smpc_stdan_3int_scalar_chance_optimized_eta_no_k,"
    "acc_nair_smpc_stdan_3int_scalar_chance_fixed_risk_no_k"
)


# Default overnight grid:
# - cut-in normal/aggressive use the ranges agreed during scenario design.
# - cut-in groups include a right-lane target vehicle so STDAN sees a third
#   prediction target, matching the mode8 validation setup.
# - outer blocker spacing is fixed at 12 m per the latest request.
# - lane_change_distance_same_lane is fixed to avoid doubling the sweep size.
# - cutout grid is intentionally moderate so a full pass finishes in one night.
SWEEP_GROUPS = [
    {
        "name": "cutin_normal",
        "kind": "cutin",
        "args": {
            "--target-start-gaps": "40",
            "--ego-speeds": "14,15,16",
            "--target-speeds": "12.0",
            "--trigger-distances": "25,30",
            "--same-lane-distances": "0",
            "--lane-change-distances": "20,25",
            "--target-lead-gaps": "18,24,30",
            "--target-lead-speed-deltas": "-2",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12.0",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
    {
        "name": "cutin_aggressive",
        "kind": "aggressive_cutin",
        "args": {
            "--target-start-gaps": "40",
            "--ego-speeds": "14,15,16",
            "--target-speeds": "12.0",
            "--trigger-distances": "15,20",
            "--same-lane-distances": "0",
            "--lane-change-distances": "12,15,18",
            "--target-lead-gaps": "18,24,30",
            "--target-lead-speed-deltas": "-2",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12.0",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
    {
        "name": "cutin_ego_lead",
        "kind": "cutin_with_ego_lead",
        "args": {
            "--target-start-gaps": "40",
            "--trigger-distances": "25,30",
            "--same-lane-distances": "5",
            "--lane-change-distances": "24,30",
            "--target-lead-gaps": "18,24",
            "--target-lead-speed-deltas": "-1,-3",
            "--ego-lead-gaps": "45,55",
            "--ego-lead-speed-deltas": "-1,-2",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
    {
        "name": "cutin_ego_lead_aggressive",
        "kind": "aggressive_cutin_with_ego_lead",
        "args": {
            "--target-start-gaps": "40",
            "--trigger-distances": "15,20",
            "--same-lane-distances": "0",
            "--lane-change-distances": "10,18",
            "--target-lead-gaps": "12,18",
            "--target-lead-speed-deltas": "-3,-5",
            "--ego-lead-gaps": "45,55",
            "--ego-lead-speed-deltas": "-1,-2",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
    {
        "name": "cutout_no_lead",
        "kind": "cutout_no_lead",
        "args": {
            "--target-start-gaps": "40",
            "--target-speeds": "8,10",
            "--trigger-distances": "15,20",
            "--same-lane-distances": "0",
            "--lane-change-distances": "12,18,24",
        },
    },
    {
        "name": "cutout_with_lead",
        "kind": "cutout_with_lead",
        "args": {
            "--target-start-gaps": "40",
            "--target-speeds": "8,10",
            "--trigger-distances": "15,20",
            "--same-lane-distances": "0",
            "--lane-change-distances": "12,18,24",
            "--lead-gaps": "25,35",
            "--lead-speeds": "8",
        },
    },
]


OPTIONAL_SWEEP_GROUPS = [
    {
        "name": "cutin_ego_lead_brake_validation",
        "kind": "cutin_with_ego_lead",
        "args": {
            "--target-start-gaps": "40",
            "--ego-speeds": "14",
            "--target-speeds": "12",
            "--trigger-distances": "25",
            "--same-lane-distances": "0",
            "--lane-change-distances": "50",
            "--target-lead-gaps": "30",
            "--target-lead-speed-deltas": "-3",
            "--ego-lead-gaps": "65",
            "--ego-lead-speed-deltas": "-1",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
]


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SMPC_ROOT / "results" / "acc_scenario_sweep" / f"overnight_{stamp}"


def _append_arg(command: list[str], key: str, value: str | int | float) -> None:
    value_text = str(value)
    if value_text.startswith("-"):
        command.append(f"{key}={value_text}")
    else:
        command.extend([key, value_text])


def _carla_pids_for_port(port: int) -> list[int]:
    pattern = f"carla-rpc-port={int(port)}"
    proc = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, cmd = line.partition(" ")
        if pattern in cmd and ("CarlaUE4" in cmd or "CarlaUE4.sh" in cmd):
            try:
                pids.append(int(pid_text))
            except ValueError:
                pass
    return pids


def _cleanup_carla_port(port: int, grace_s: float) -> list[int]:
    pids = _carla_pids_for_port(port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids and grace_s > 0:
        time.sleep(grace_s)
    for pid in _carla_pids_for_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def _start_carla_server(carla_root: str, port: int, log_path: Path) -> tuple[subprocess.Popen, object]:
    carla_root_path = Path(carla_root)
    cmd = [
        str(carla_root_path / "CarlaUE4.sh"),
        "-RenderOffScreen",
        "-nosound",
        "-quality-level=Low",
        f"-carla-rpc-port={int(port)}",
    ]
    log_file = Path(log_path).open("w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(carla_root_path),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, log_file


def _wait_for_carla(carla_root: str, port: int, timeout_s: float) -> bool:
    python_api = str(Path(carla_root) / "PythonAPI")
    if python_api not in sys.path:
        sys.path.insert(0, python_api)
    import carla

    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            client = carla.Client("localhost", int(port))
            client.set_timeout(2.0)
            client.get_server_version()
            return True
        except Exception:
            time.sleep(1.0)
    return False


def _stop_carla_server(proc: subprocess.Popen | None, log_file: object | None) -> None:
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


def _build_command(
        args: argparse.Namespace,
        ego_policy: str,
        group: dict,
        output_root: Path,
        group_idx: int,
        run_idx: int) -> tuple[list[str], int, Path]:
    policy_dir = _safe_path_component(ego_policy)
    group_output = output_root / policy_dir / f"{group_idx:02d}_{group['name']}"
    group_port = (
        args.port
        if args.same_port_per_group
        else args.port + (run_idx - 1) * args.port_step
    )
    command = [
        args.python,
        "scripts/carla/devel/sweep_acc_scenarios.py",
        "--kind",
        group["kind"],
        "--ego-policy",
        ego_policy,
        "--max-runs",
        str(args.max_runs_per_group),
        "--spawn-settle-s",
        str(args.spawn_settle_s),
        "--cruise-warmup-s",
        str(args.cruise_warmup_s),
        "--approach-time-s",
        str(args.approach_time_s),
        "--ego-gap-at-trigger",
        str(args.ego_gap_at_trigger),
        "--max-sim-time",
        str(args.max_sim_time),
        "--ego-horizons",
        str(args.ego_horizon),
        "--port",
        str(group_port),
        "--port-step",
        str(args.port_step),
        "--server-timeout",
        str(args.server_timeout),
        "--carla-timeout-period",
        str(args.carla_timeout_period),
        "--carla-root",
        args.carla_root,
        "--output-root",
        str(group_output),
    ]
    if args.fresh_server_per_run:
        command.append("--fresh-server-per-run")
    if args.dry_run:
        command.append("--dry-run")
    if args.resume:
        command.append("--resume")
    if args.stop_on_group_failure:
        command.append("--stop-on-run-failure")
    else:
        command.append("--no-stop-on-run-failure")
    for key, value in group["args"].items():
        _append_arg(command, key, value)
    return command, int(group_port), group_output


def _run_group(
        args: argparse.Namespace,
        ego_policy: str,
        group: dict,
        output_root: Path,
        group_idx: int,
        run_idx: int) -> dict:
    command, group_port, group_output = _build_command(
        args, ego_policy, group, output_root, group_idx, run_idx)
    group_output.mkdir(parents=True, exist_ok=True)
    log_path = group_output / "runner_stdout.log"
    carla_log_path = group_output / "carla_server.log"

    env = os.environ.copy()
    env.setdefault("CARLA_ROOT", args.carla_root)
    env.setdefault("GRB_LICENSE_FILE", "/home/core-dev/gurobi.lic")
    env.setdefault("GUROBI_HOME", "/opt/gurobi1302/linux64")
    env.setdefault("GUROBI_VERSION", "130")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("ACC_NAIR_WANDB", "0")
    env.setdefault("ACC_REUSE_CARLA_WORLD", "1")

    pre_cleanup_pids = []
    if args.cleanup_carla and not args.dry_run:
        pre_cleanup_pids = _cleanup_carla_port(group_port, args.server_cleanup_s)

    start = time.monotonic()
    group_carla_proc = None
    group_carla_log_file = None
    completed = None
    group_server_error = None
    try:
        if args.start_group_carla and not args.fresh_server_per_run and not args.dry_run:
            group_carla_proc, group_carla_log_file = _start_carla_server(
                args.carla_root,
                group_port,
                carla_log_path,
            )
            if not _wait_for_carla(args.carla_root, group_port, args.server_timeout):
                group_server_error = "CARLA group server did not become ready before timeout"

        if group_server_error is None:
            if args.stream_child_output:
                completed = subprocess.run(command, cwd=SMPC_ROOT, env=env, text=True)
            else:
                with log_path.open("w") as log_file:
                    completed = subprocess.run(
                        command,
                        cwd=SMPC_ROOT,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
    finally:
        if args.start_group_carla and not args.fresh_server_per_run and not args.dry_run:
            _stop_carla_server(group_carla_proc, group_carla_log_file)
    duration = time.monotonic() - start

    post_cleanup_pids = []
    if args.cleanup_carla and not args.dry_run:
        post_cleanup_pids = _cleanup_carla_port(group_port, args.server_cleanup_s)

    row = {
        "ego_policy": ego_policy,
        "name": group["name"],
        "kind": group["kind"],
        "returncode": int(completed.returncode) if completed is not None else 1,
        "wall_elapsed_s": float(duration),
        "port": int(group_port),
        "pre_cleanup_pids": pre_cleanup_pids,
        "post_cleanup_pids": post_cleanup_pids,
        "output_root": str(group_output),
        "log_path": str(log_path),
        "carla_log_path": str(carla_log_path),
        "command": command,
    }
    if group_server_error is not None:
        row.update({
            "sweep_manifest": [],
            "run_count": 0,
            "failure_count": 1,
            "ran_successfully": False,
            "error": group_server_error,
        })
        return row

    manifest_path = group_output / "sweep_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            sweep_rows = json.load(f)
        failures = [
            item for item in sweep_rows
            if not item.get("ran_successfully") and item.get("error") != "dry_run"
        ]
        row.update({
            "sweep_manifest": sweep_rows,
            "run_count": len(sweep_rows),
            "failure_count": len(failures),
            "ran_successfully": completed.returncode == 0 and not failures,
        })
    else:
        row.update({
            "sweep_manifest": [],
            "run_count": 0,
            "failure_count": 1,
            "ran_successfully": False,
            "error": "sweep_manifest.json was not created",
        })
    return row


def _safe_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _selected_policies(args: argparse.Namespace) -> list[str]:
    policy_text = args.ego_policies if args.ego_policies else args.ego_policy
    policies = [item.strip() for item in str(policy_text).split(",") if item.strip()]
    if not policies:
        raise ValueError("At least one ego policy is required")
    return policies


def _selected_groups(include: str) -> list[dict]:
    if include == "all":
        return SWEEP_GROUPS
    available_groups = SWEEP_GROUPS + OPTIONAL_SWEEP_GROUPS
    selected = {item.strip() for item in include.split(",") if item.strip()}
    groups = [group for group in available_groups if group["name"] in selected or group["kind"] in selected]
    missing = selected - {group["name"] for group in groups} - {group["kind"] for group in groups}
    if missing:
        raise ValueError(f"Unknown sweep group(s): {sorted(missing)}")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--carla-root", default=os.getenv("CARLA_ROOT", DEFAULT_CARLA_ROOT))
    parser.add_argument("--ego-policy", default="acc_nair_smpc_stdan_3int")
    parser.add_argument("--ego-policies", default=DEFAULT_COMPARE_POLICIES,
                        help=(
                            "Comma-separated ego policies. Example: "
                            f"{DEFAULT_COMPARE_POLICIES}"
                        ))
    parser.add_argument("--include", default="all",
                        help="Comma-separated group names/kinds, or 'all'.")
    parser.add_argument("--port", type=int, default=2015)
    parser.add_argument("--port-step", type=int, default=2,
                        help="Port increment between CARLA instances; CARLA also uses rpc_port+1.")
    parser.add_argument("--same-port-per-group", action="store_true")
    parser.add_argument("--server-timeout", type=float, default=150.0)
    parser.add_argument("--carla-timeout-period", type=float, default=30.0)
    parser.add_argument("--server-cleanup-s", type=float, default=2.0)
    parser.add_argument("--spawn-settle-s", type=float, default=6.0)
    parser.add_argument("--cruise-warmup-s", type=float, default=0.0)
    parser.add_argument("--approach-time-s", type=float, default=0.0)
    parser.add_argument("--ego-gap-at-trigger", type=float, default=22.0)
    parser.add_argument("--max-sim-time", type=float, default=25.0)
    parser.add_argument("--ego-horizon", type=int, default=15)
    parser.add_argument("--max-runs-per-group", type=int, default=0,
                        help="0 means run the full grid for each group.")
    parser.add_argument("--fresh-server-per-run", action="store_true", default=False)
    parser.add_argument("--no-fresh-server-per-run", dest="fresh_server_per_run", action="store_false")
    parser.add_argument("--cleanup-carla", action="store_true", default=True)
    parser.add_argument("--no-cleanup-carla", dest="cleanup_carla", action="store_false")
    parser.add_argument("--start-group-carla", dest="start_group_carla", action="store_true", default=True)
    parser.add_argument("--no-start-group-carla", dest="start_group_carla", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from successful child sweep_manifest.json rows.")
    parser.add_argument("--stream-child-output", action="store_true")
    parser.add_argument("--stop-on-group-failure", dest="stop_on_group_failure", action="store_true", default=True)
    parser.add_argument("--no-stop-on-group-failure", dest="stop_on_group_failure", action="store_false")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    groups = _selected_groups(args.include)
    policies = _selected_policies(args)
    with (output_root / "overnight_sweep_config.json").open("w") as f:
        json.dump(
            {"args": vars(args), "groups": groups, "ego_policies": policies},
            f,
            indent=2,
            sort_keys=True,
        )

    rows = []
    total = len(policies) * len(groups)
    run_idx = 0
    stop_requested = False
    for ego_policy in policies:
        for group_idx, group in enumerate(groups, start=1):
            run_idx += 1
            print(
                f"[{run_idx}/{total}] running {ego_policy} / {group['name']} ({group['kind']})",
                flush=True,
            )
            row = _run_group(args, ego_policy, group, output_root, group_idx, run_idx)
            rows.append(row)
            print(json.dumps({
                "ego_policy": row["ego_policy"],
                "name": row["name"],
                "kind": row["kind"],
                "port": row["port"],
                "returncode": row["returncode"],
                "run_count": row["run_count"],
                "failure_count": row["failure_count"],
                "ran_successfully": row["ran_successfully"],
                "wall_elapsed_s": row["wall_elapsed_s"],
            }, sort_keys=True), flush=True)
            if row["returncode"] != 0 or (args.stop_on_group_failure and not row["ran_successfully"]):
                stop_requested = True
                break
        if stop_requested:
            break

    with (output_root / "overnight_sweep_manifest.json").open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    return 0 if rows and all(row.get("ran_successfully") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
