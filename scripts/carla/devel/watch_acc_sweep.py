#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SMPC_ROOT = SCRIPT_DIR.parents[2]
CRASH_PATTERNS = (
    "Segmentation fault",
    "Signal 11",
    "CommonUnixCrashHandler",
    "Engine crash handling finished",
)
TIMEOUT_PATTERNS = (
    "time-out of 30000ms while waiting for the simulator",
    "CARLA server did not become ready before timeout",
    "failed to destroy actor",
)
ACTIVITY_NAMES = (
    "runner_stdout.log",
    "sweep_manifest.json",
    "overnight_sweep_manifest.json",
    "summary.json",
    "metrics.json",
    "resolved_config.json",
)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str, log_file) -> None:
    line = f"[{_now_text()}] {message}"
    print(line, flush=True)
    print(line, file=log_file, flush=True)


def _infer_output_root(command: list[str]) -> Path | None:
    for idx, item in enumerate(command):
        if item == "--output-root" and idx + 1 < len(command):
            return Path(command[idx + 1])
        if item.startswith("--output-root="):
            return Path(item.split("=", 1)[1])
    return None


def _tail_text(path: Path, max_bytes: int = 32768) -> str:
    try:
        with path.open("rb") as f:
            size = path.stat().st_size
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _latest_matching_file(root: Path, name: str) -> Path | None:
    if not root.exists():
        return None
    matches = list(root.rglob(name))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _latest_activity_time(root: Path) -> float | None:
    if not root.exists():
        return None
    latest = None
    for name in ACTIVITY_NAMES:
        for path in root.rglob(name):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if latest is None or mtime > latest:
                latest = mtime
    return latest


def _crash_log_since(root: Path, since_epoch_s: float) -> Path | None:
    if not root.exists():
        return None
    candidates = []
    for path in root.rglob("carla_server.log"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime < since_epoch_s:
            continue
        text = _tail_text(path)
        if any(pattern in text for pattern in CRASH_PATTERNS):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _timeout_log_since(root: Path, since_epoch_s: float) -> Path | None:
    latest = _latest_matching_file(root, "runner_stdout.log")
    if latest is None:
        return None
    try:
        if latest.stat().st_mtime < since_epoch_s:
            return None
    except OSError:
        return None
    text = _tail_text(latest)
    if any(pattern in text for pattern in TIMEOUT_PATTERNS):
        return latest
    return None


def _carla_pids() -> list[int]:
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
        if "CarlaUE4" not in cmd:
            continue
        try:
            pids.append(int(pid_text))
        except ValueError:
            continue
    return pids


def _cleanup_carla(grace_s: float, log_file) -> list[int]:
    pids = _carla_pids()
    if pids:
        _log(f"cleaning CARLA processes: {pids}", log_file)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if pids and grace_s > 0:
        time.sleep(grace_s)
    remaining = _carla_pids()
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return pids


def _stop_child(proc: subprocess.Popen, log_file, term_timeout_s: float) -> None:
    if proc.poll() is not None:
        return
    _log(f"stopping sweep process pid={proc.pid}", log_file)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.terminate()
    try:
        proc.wait(timeout=term_timeout_s)
    except subprocess.TimeoutExpired:
        _log(f"force killing sweep process pid={proc.pid}", log_file)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
        proc.wait(timeout=term_timeout_s)


def _start_attempt(command: list[str], output_root: Path, attempt: int, log_file) -> tuple[subprocess.Popen, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    attempt_log_path = output_root / f"watchdog_attempt_{attempt:03d}.log"
    attempt_log = attempt_log_path.open("a")
    _log(f"starting attempt={attempt} command={' '.join(command)}", log_file)
    _log(f"attempt stdout/stderr -> {attempt_log_path}", log_file)
    proc = subprocess.Popen(
        command,
        cwd=SMPC_ROOT,
        stdout=attempt_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _log(f"started sweep pid={proc.pid}", log_file)
    return proc, attempt_log


def run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing sweep command after --")

    output_root = Path(args.output_root) if args.output_root else _infer_output_root(command)
    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = SMPC_ROOT / "results" / "acc_scenario_sweep" / f"watchdog_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "watchdog.log"

    stop_requested = False

    def _handle_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    restarts = 0
    attempt = 0
    with log_path.open("a") as log_file:
        _log("watchdog started", log_file)
        _log(f"output_root={output_root}", log_file)
        while not stop_requested:
            attempt += 1
            if args.cleanup_carla_before_start:
                _cleanup_carla(args.carla_cleanup_grace_s, log_file)

            attempt_start = time.time()
            no_carla_since = None
            proc, attempt_log = _start_attempt(command, output_root, attempt, log_file)
            restart_reason = None
            try:
                while not stop_requested:
                    returncode = proc.poll()
                    if returncode is not None:
                        attempt_log.close()
                        if returncode == 0:
                            _log(f"sweep finished successfully pid={proc.pid}", log_file)
                            return 0
                        restart_reason = f"sweep exited with returncode={returncode}"
                        break

                    now = time.time()
                    crash_log = _crash_log_since(output_root, attempt_start)
                    if crash_log is not None:
                        crash_age = now - crash_log.stat().st_mtime
                        if crash_age >= args.crash_grace_s:
                            restart_reason = f"CARLA crash detected in {crash_log}"
                            break

                    timeout_log = _timeout_log_since(output_root, attempt_start)
                    if timeout_log is not None and now - attempt_start >= args.startup_grace_s:
                        restart_reason = f"simulator timeout detected in {timeout_log}"
                        break

                    pids = _carla_pids()
                    if pids:
                        no_carla_since = None
                    elif now - attempt_start >= args.startup_grace_s:
                        if no_carla_since is None:
                            no_carla_since = now
                        elif now - no_carla_since >= args.no_carla_grace_s:
                            restart_reason = "no CARLA process found while sweep is still running"
                            break

                    if args.stall_s > 0 and now - attempt_start >= args.startup_grace_s:
                        activity_time = _latest_activity_time(output_root)
                        if activity_time is not None and now - activity_time >= args.stall_s:
                            restart_reason = (
                                f"no output activity for {now - activity_time:.1f}s "
                                f"(threshold={args.stall_s:.1f}s)"
                            )
                            break

                    time.sleep(args.poll_s)
            finally:
                if not attempt_log.closed:
                    attempt_log.close()

            if stop_requested:
                _stop_child(proc, log_file, args.term_timeout_s)
                _cleanup_carla(args.carla_cleanup_grace_s, log_file)
                _log("watchdog stopped by signal", log_file)
                return 130

            restarts += 1
            _log(f"restart requested: {restart_reason}", log_file)
            _stop_child(proc, log_file, args.term_timeout_s)
            _cleanup_carla(args.carla_cleanup_grace_s, log_file)

            if args.max_restarts > 0 and restarts > args.max_restarts:
                _log(f"max restarts exceeded: {args.max_restarts}", log_file)
                return 1
            if args.restart_delay_s > 0:
                time.sleep(args.restart_delay_s)

        return 130


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ACC sweep under an external watchdog. The watchdog restarts "
            "the command when the sweep exits nonzero, CARLA crashes, CARLA "
            "disappears while the sweep is active, or output stops advancing."
        )
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--startup-grace-s", type=float, default=180.0)
    parser.add_argument("--crash-grace-s", type=float, default=60.0)
    parser.add_argument("--no-carla-grace-s", type=float, default=120.0)
    parser.add_argument("--stall-s", type=float, default=1800.0,
                        help="Restart if output files do not change for this long. Use 0 to disable.")
    parser.add_argument("--restart-delay-s", type=float, default=10.0)
    parser.add_argument("--term-timeout-s", type=float, default=20.0)
    parser.add_argument("--carla-cleanup-grace-s", type=float, default=2.0)
    parser.add_argument("--max-restarts", type=int, default=0,
                        help="0 means unlimited restarts.")
    parser.add_argument("--cleanup-carla-before-start", action="store_true", default=True)
    parser.add_argument("--no-cleanup-carla-before-start",
                        dest="cleanup_carla_before_start",
                        action="store_false")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
