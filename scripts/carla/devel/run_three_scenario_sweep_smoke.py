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


SCENARIOS = [
    {
        "name": "cutin_normal",
        "kind": "cutin",
        "args": {
            "--target-start-gaps": "40",
            "--trigger-distances": "25",
            "--same-lane-distances": "5",
            "--lane-change-distances": "24",
            "--target-lead-gaps": "18",
            "--target-lead-speed-deltas": "-1",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
    {
        "name": "cutin_aggressive",
        "kind": "aggressive_cutin",
        "args": {
            "--target-start-gaps": "40",
            "--trigger-distances": "15",
            "--same-lane-distances": "0",
            "--lane-change-distances": "10",
            "--target-lead-gaps": "12",
            "--target-lead-speed-deltas": "-5",
            "--outer-blocker-counts": "3",
            "--outer-blocker-spacings": "12",
            "--outer-blocker-speed-deltas": "-1.5",
        },
    },
    {
        "name": "cutout_with_lead",
        "kind": "cutout_with_lead",
        "args": {
            "--target-start-gaps": "40",
            "--trigger-distances": "20",
            "--same-lane-distances": "0",
            "--lane-change-distances": "15",
            "--target-speeds": "8",
            "--lead-gaps": "25",
            "--lead-speeds": "8",
        },
    },
]


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SMPC_ROOT / "results" / "acc_scenario_sweep" / f"three_scenario_smoke_{stamp}"


def _append_arg(command: list[str], key: str, value: str | int | float) -> None:
    command.extend([key, str(value)])


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


def _run_case(args: argparse.Namespace, case: dict, output_root: Path, case_idx: int) -> dict:
    case_output = output_root / f"{case_idx:02d}_{case['name']}"
    case_port = args.port if args.same_port_per_case else args.port + case_idx - 1
    command = [
        args.python,
        "scripts/carla/devel/sweep_acc_scenarios.py",
        "--kind",
        case["kind"],
        "--ego-policy",
        args.ego_policy,
        "--max-runs",
        "1",
        "--spawn-settle-s",
        str(args.spawn_settle_s),
        "--max-sim-time",
        str(args.max_sim_time),
        "--ego-horizons",
        str(args.ego_horizon),
        "--port",
        str(case_port),
        "--server-timeout",
        str(args.server_timeout),
        "--carla-root",
        args.carla_root,
        "--output-root",
        str(case_output),
    ]
    if args.fresh_server_per_run:
        command.append("--fresh-server-per-run")
    for key, value in case["args"].items():
        _append_arg(command, key, value)

    env = os.environ.copy()
    env.setdefault("CARLA_ROOT", args.carla_root)
    env.setdefault("GRB_LICENSE_FILE", "/home/core-dev/gurobi.lic")
    env.setdefault("GUROBI_HOME", "/opt/gurobi1302/linux64")
    env.setdefault("GUROBI_VERSION", "130")
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("ACC_NAIR_WANDB", "0")

    pre_cleanup_pids = []
    if args.cleanup_carla:
        pre_cleanup_pids = _cleanup_carla_port(case_port, args.server_cleanup_s)

    start = time.monotonic()
    completed = subprocess.run(command, cwd=SMPC_ROOT, env=env, text=True)
    duration = time.monotonic() - start
    post_cleanup_pids = []
    if args.cleanup_carla:
        post_cleanup_pids = _cleanup_carla_port(case_port, args.server_cleanup_s)
    row = {
        "name": case["name"],
        "kind": case["kind"],
        "returncode": int(completed.returncode),
        "wall_elapsed_s": float(duration),
        "port": int(case_port),
        "pre_cleanup_pids": pre_cleanup_pids,
        "post_cleanup_pids": post_cleanup_pids,
        "output_root": str(case_output),
        "command": command,
    }
    manifest_path = case_output / "sweep_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r") as f:
            sweep_rows = json.load(f)
        row["sweep_manifest"] = sweep_rows
        row["ran_successfully"] = bool(sweep_rows and sweep_rows[0].get("ran_successfully"))
        row["error"] = None if not sweep_rows else sweep_rows[0].get("error")
    else:
        row["ran_successfully"] = False
        row["error"] = "sweep_manifest.json was not created"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--carla-root", default=os.getenv("CARLA_ROOT", DEFAULT_CARLA_ROOT))
    parser.add_argument("--ego-policy", default="acc_nair_smpc_stdan_3int")
    parser.add_argument("--port", type=int, default=2015)
    parser.add_argument("--same-port-per-case", action="store_true")
    parser.add_argument("--server-timeout", type=float, default=150.0)
    parser.add_argument("--server-cleanup-s", type=float, default=2.0)
    parser.add_argument("--spawn-settle-s", type=float, default=6.0)
    parser.add_argument("--max-sim-time", type=float, default=12.0)
    parser.add_argument("--ego-horizon", type=int, default=15)
    parser.add_argument("--fresh-server-per-run", action="store_true", default=True)
    parser.add_argument("--no-fresh-server-per-run", dest="fresh_server_per_run", action="store_false")
    parser.add_argument("--cleanup-carla", action="store_true", default=True)
    parser.add_argument("--no-cleanup-carla", dest="cleanup_carla", action="store_false")
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else _default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "three_scenario_smoke_config.json").open("w") as f:
        json.dump({"args": vars(args), "scenarios": SCENARIOS}, f, indent=2, sort_keys=True)

    rows = []
    for idx, case in enumerate(SCENARIOS, start=1):
        print(f"[{idx}/{len(SCENARIOS)}] running {case['name']} ({case['kind']})", flush=True)
        row = _run_case(args, case, output_root, idx)
        rows.append(row)
        print(json.dumps({
            "name": row["name"],
            "port": row["port"],
            "returncode": row["returncode"],
            "ran_successfully": row["ran_successfully"],
            "wall_elapsed_s": row["wall_elapsed_s"],
            "error": row["error"],
        }, sort_keys=True), flush=True)
        if row["returncode"] != 0 or not row["ran_successfully"]:
            break

    with (output_root / "three_scenario_smoke_manifest.json").open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
    return 0 if len(rows) == len(SCENARIOS) and all(row["ran_successfully"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
