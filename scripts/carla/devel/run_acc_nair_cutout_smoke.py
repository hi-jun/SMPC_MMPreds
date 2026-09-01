import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARLA_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SMPC_ROOT = os.path.abspath(os.path.join(CARLA_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, CARLA_SCRIPT_DIR)

from utils.acc_nair_smpc import (  # noqa: E402
    NairACCSMPC,
    SAFETY_BRAKE_DISTANCE,
    SAFETY_NOMINAL_SAFE_DISTANCE,
    SAFETY_SCALAR_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    VARIANT_PROPOSED,
    original_nair_acc_config,
    rollout_dynamics,
    safety_function,
)
from utils.synthetic_lk_cutout_prediction import (  # noqa: E402
    SyntheticCutOutConfig,
    SyntheticLaneKeepingCutOutPredictor,
)


def _variant_from_name(name):
    if name in ("open_loop", "multimodal_ol", "acc_nair_smpc_ol"):
        return VARIANT_MULTIMODAL_OL
    if name in ("fixed_risk", "acc_nair_smpc_fixed_risk"):
        return VARIANT_FIXED_RISK
    return VARIANT_PROPOSED


def _default_output_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SMPC_ROOT, "results", "acc_nair_cutout_validation", timestamp)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _reference_columns(solution, mode_names):
    columns = {}
    for mode_idx, mode_name in enumerate(mode_names):
        columns["p_ref_%s" % mode_name] = json.dumps(
            [float(value) for value in solution.reference.s_ref[mode_idx]]
        )
        columns["v_ref_%s" % mode_name] = json.dumps(
            [float(value) for value in solution.reference.v_ref[mode_idx]]
        )
        columns["a_ref_%s" % mode_name] = json.dumps(
            [float(value) for value in solution.reference.a_ref[mode_idx]]
        )
    return columns


def _maybe_write_plot(path, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return "plot_skipped: %r" % (exc,)

    times = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(times, [row["ego_v"] for row in rows], label="ego_v")
    axes[0].plot(times, [row["target_v"] for row in rows], label="target_v")
    axes[0].set_ylabel("speed [m/s]")
    axes[0].legend()
    axes[1].plot(times, [row["accel_cmd"] for row in rows])
    axes[1].set_ylabel("accel [m/s^2]")
    axes[2].plot(times, [row["gap"] for row in rows])
    axes[2].set_ylabel("ACC gap [m]")
    axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return "plot_saved"


def _is_lead_active(target_d, args):
    return abs(target_d - args.target_d0) < args.nonblocking_d_threshold


def _mode_probabilities(args, target_d=None):
    if (
            args.collapse_probability_after_clearance
            and target_d is not None
            and not _is_lead_active(target_d, args)):
        return [0.0, 1.0]
    cutout_probability = float(np.clip(args.cutout_probability, 0.0, 1.0))
    return [1.0 - cutout_probability, cutout_probability]


def _initial_target_s(args, config):
    if args.target_s0 is not None:
        return args.target_s0
    desired_gap = (
        config.vehicle_length
        + config.d0
        + config.time_headway * args.ego_v0
        + args.initial_gap_margin
    )
    return args.ego_s0 + desired_gap


def run(args):
    output_dir = args.output_dir or _default_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    config = original_nair_acc_config(
        horizon=args.horizon,
        dt=args.dt,
        desired_speed=args.desired_speed,
        num_modes=2,
        controller_variant=_variant_from_name(args.variant),
        gurobi_time_limit=args.gurobi_time_limit,
    )
    config.safety_constraint_mode = args.safety_constraint_mode
    config.__post_init__()
    controller = NairACCSMPC(config)
    probabilities = _mode_probabilities(args, args.target_d0)
    predictor = SyntheticLaneKeepingCutOutPredictor(
        SyntheticCutOutConfig(
            horizon=args.horizon,
            dt=args.dt,
            probabilities=probabilities,
            cutout_start_delay=args.cutout_start_delay,
            cutout_duration=args.cutout_duration,
            nonblocking_d_threshold=args.nonblocking_d_threshold,
            adjacent_lane_d=args.target_d_final,
        )
    )

    ego = np.array([args.ego_s0, args.ego_v0], dtype=float)
    target_s0 = _initial_target_s(args, config)
    target = np.array([target_s0, args.target_d0, args.target_v0], dtype=float)
    rows = []

    for step in range(args.steps):
        time_s = step * args.dt
        start = time.time()
        probabilities = _mode_probabilities(args, target[1])
        synthetic = predictor.predict(
            ego_state=ego,
            target_state=target,
            probabilities=probabilities,
        )
        solution = controller.solve(ego, synthetic.prediction)
        solve_time = time.time() - start
        accel = float(np.clip(solution.action, config.a_min, config.a_max))

        raw_gap = safety_function(ego, np.array([target[0], target[2]]), config)
        lead_active = _is_lead_active(target[1], args)
        gap = raw_gap if lead_active else np.nan
        row = {
            "step": step,
            "time_s": round(time_s, 6),
            "ego_s": float(ego[0]),
            "ego_v": float(ego[1]),
            "desired_speed": float(config.desired_speed),
            "accel_cmd": accel,
            "target_s": float(target[0]),
            "target_d": float(target[1]),
            "target_v": float(target[2]),
            "p_lane_keeping": probabilities[0],
            "p_cutout": probabilities[1],
            "lead_active": bool(lead_active),
            "feasible": bool(solution.feasible),
            "status": solution.status,
            "solver_message": solution.solver_message,
            "safety_constraint_mode": solution.safety_constraint_mode,
            "gap": float(gap),
            "raw_target_gap": float(raw_gap),
            "solve_time_s": solve_time,
        }
        row.update(_reference_columns(solution, synthetic.mode_names))
        rows.append(row)

        ego = rollout_dynamics(ego, np.array([accel]), args.dt)[-1]

        target[2] = max(0.0, target[2] + args.dt * args.actual_target_accel)
        target[0] = target[0] + args.dt * target[2]
        next_time_s = (step + 1) * args.dt
        alpha = min(1.0, max(0.0, (next_time_s - args.cutout_start_delay) / max(args.cutout_duration, args.dt)))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        target[1] = args.target_d0 + alpha * (args.target_d_final - args.target_d0)

    feasible_rate = float(np.mean([row["feasible"] for row in rows])) if rows else 0.0
    accel_values = [row["accel_cmd"] for row in rows]
    solve_times = [row["solve_time_s"] for row in rows]
    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    active_gaps = [row["gap"] for row in rows if np.isfinite(row["gap"])]
    summary = {
        "output_dir": output_dir,
        "steps": args.steps,
        "variant": args.variant,
        "mode_names": ["lane_keeping", "cutout"],
        "mode_probabilities": probabilities,
        "collapse_probability_after_clearance": args.collapse_probability_after_clearance,
        "initial_target_s": float(target_s0),
        "initial_gap_margin": args.initial_gap_margin,
        "cutout_complete_time": args.cutout_start_delay + args.cutout_duration,
        "lead_active_steps": int(sum(row["lead_active"] for row in rows)),
        "safety_constraint_mode": args.safety_constraint_mode,
        "status_counts": status_counts,
        "first_solver_message": rows[0]["solver_message"] if rows else "",
        "feasibility_rate": feasible_rate,
        "min_active_gap": min(active_gaps) if active_gaps else None,
        "min_raw_target_gap": min(row["raw_target_gap"] for row in rows),
        "accel_min": min(accel_values),
        "accel_max": max(accel_values),
        "solve_time_mean": float(np.mean(solve_times)),
        "solve_time_max": float(np.max(solve_times)),
        "accel_bounds": [config.a_min, config.a_max],
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    _write_csv(os.path.join(output_dir, "trajectory.csv"), rows)
    summary["plot_status"] = _maybe_write_plot(os.path.join(output_dir, "speed_accel_gap.png"), rows)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--variant", default="proposed")
    parser.add_argument("--desired-speed", type=float, default=15.0)
    parser.add_argument(
        "--safety-constraint-mode",
        choices=[SAFETY_NOMINAL_SAFE_DISTANCE, SAFETY_SCALAR_CHANCE, SAFETY_BRAKE_DISTANCE],
        default=SAFETY_NOMINAL_SAFE_DISTANCE,
    )
    parser.add_argument("--cutout-probability", type=float, default=0.7)
    parser.add_argument("--collapse-probability-after-clearance", action="store_true", default=True)
    parser.add_argument("--keep-probability-after-clearance", dest="collapse_probability_after_clearance",
                        action="store_false")
    parser.add_argument("--cutout-start-delay", type=float, default=0.5)
    parser.add_argument("--cutout-duration", type=float, default=1.5)
    parser.add_argument("--nonblocking-d-threshold", type=float, default=1.75)
    parser.add_argument("--ego-s0", type=float, default=0.0)
    parser.add_argument("--ego-v0", type=float, default=8.0)
    parser.add_argument("--target-s0", type=float, default=None)
    parser.add_argument("--initial-gap-margin", type=float, default=1.0)
    parser.add_argument("--target-d0", type=float, default=0.0)
    parser.add_argument("--target-d-final", type=float, default=3.5)
    parser.add_argument("--target-v0", type=float, default=8.0)
    parser.add_argument("--actual-target-accel", type=float, default=0.0)
    parser.add_argument("--gurobi-time-limit", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
