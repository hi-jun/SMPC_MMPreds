from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _ego_steps(summary):
    for policy_log in summary.get("policy_logs", {}).values():
        if isinstance(policy_log, dict) and policy_log.get("controller") == "acc_nair_smpc":
            return policy_log.get("steps", [])
    return []


def _bad_reasons(step):
    status = str(step.get("status", ""))
    solve_path = str(step.get("solve_path", ""))
    solver_message = str(step.get("solver_message", ""))
    reasons = []
    if step.get("feasible") is not True:
        reasons.append("feasible_false")
    if status != "optimal":
        reasons.append("status_not_optimal")
    combined = " ".join((status, solve_path, solver_message)).lower()
    if "fallback" in combined:
        reasons.append("fallback")
    if "infeasible" in combined:
        reasons.append("infeasible")
    return reasons


def _solver_failure_reasons(step):
    status = str(step.get("status", ""))
    solve_path = str(step.get("solve_path", ""))
    solver_message = str(step.get("solver_message", ""))
    reasons = []
    if status != "optimal":
        reasons.append("status_not_optimal")
    combined = " ".join((status, solve_path, solver_message)).lower()
    if "fallback" in combined:
        reasons.append("fallback")
    if "infeasible" in combined or "inf_or_unbd" in combined:
        reasons.append("infeasible")
    return reasons


def _run_row(summary_path):
    with summary_path.open("r") as file_obj:
        summary = json.load(file_obj)
    resolved_path = summary_path.parent / "resolved_config.json"
    resolved = {}
    if resolved_path.exists():
        with resolved_path.open("r") as file_obj:
            resolved = json.load(file_obj)

    bad_steps = []
    safety_violations = []
    solver_failures = []
    for step_idx, step in enumerate(_ego_steps(summary)):
        reasons = _bad_reasons(step)
        if reasons:
            bad_steps.append((step_idx, step, reasons))
        if step.get("feasible") is not True:
            safety_violations.append((step_idx, step))
        solver_reasons = _solver_failure_reasons(step)
        if solver_reasons:
            solver_failures.append((step_idx, step, solver_reasons))

    row = {
        "run_dir": str(summary_path.parent),
        "ego_policy": resolved.get("ego_policy"),
        "ran_successfully": bool(summary.get("ran_successfully")),
        "num_controller_steps": len(_ego_steps(summary)),
        "bad_step_count": len(bad_steps),
        "safety_violation_step_count": len(safety_violations),
        "solver_failure_step_count": len(solver_failures),
        "first_bad_step_idx": None,
        "time_s": None,
        "bad_reason": None,
        "status": None,
        "feasible": None,
        "solve_path": None,
        "solver_message": None,
        "ego_s": None,
        "ego_v": None,
        "min_gap": None,
        "chance_margin_min": None,
        "first_safety_violation_step_idx": None,
        "first_safety_violation_time_s": None,
        "first_solver_failure_step_idx": None,
        "first_solver_failure_time_s": None,
        "first_solver_failure_reason": None,
        "first_solver_failure_status": None,
        "first_solver_failure_message": None,
    }
    if bad_steps:
        step_idx, step, reasons = bad_steps[0]
        row.update({
            "first_bad_step_idx": step_idx,
            "time_s": step.get("time_s"),
            "bad_reason": ",".join(reasons),
            "status": step.get("status"),
            "feasible": step.get("feasible"),
            "solve_path": step.get("solve_path"),
            "solver_message": step.get("solver_message"),
            "ego_s": step.get("ego_s"),
            "ego_v": step.get("ego_v"),
            "min_gap": step.get("min_gap"),
            "chance_margin_min": step.get("chance_margin_min"),
        })
    if safety_violations:
        step_idx, step = safety_violations[0]
        row["first_safety_violation_step_idx"] = step_idx
        row["first_safety_violation_time_s"] = step.get("time_s")
    if solver_failures:
        step_idx, step, reasons = solver_failures[0]
        row.update({
            "first_solver_failure_step_idx": step_idx,
            "first_solver_failure_time_s": step.get("time_s"),
            "first_solver_failure_reason": ",".join(reasons),
            "first_solver_failure_status": step.get("status"),
            "first_solver_failure_message": step.get("solver_message"),
        })
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Validation output root or one run directory")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if (input_path / "summary.json").exists():
        summary_paths = [input_path / "summary.json"]
    else:
        summary_paths = sorted(input_path.rglob("summary.json"))
    if not summary_paths:
        raise RuntimeError(f"No summary.json found below {input_path}")

    rows = [_run_row(path) for path in summary_paths]
    output_dir = Path(args.output_dir) if args.output_dir else input_path
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "carla_brake_distance_validation.json"
    csv_path = output_dir / "carla_brake_distance_validation.csv"
    with json_path.open("w") as file_obj:
        json.dump(rows, file_obj, indent=2, sort_keys=True)
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
