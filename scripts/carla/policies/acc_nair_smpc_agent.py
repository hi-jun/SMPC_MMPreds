import os
import re
import sys
import time

import carla
import numpy as np

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
rootdir = os.path.abspath(__file__).split('scripts')[0]
sys.path.append(rootdir)
from rasterizer.agent_history import AgentHistory
from utils import frenet_trajectory_handler as fth
from utils.acc_nair_smpc import (
    BRAKE_DISTANCE_BOUND_HARD_BAND,
    BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL,
    MultimodalLeadPrediction,
    NairACCSMPC,
    PolicySharingTree,
    RISK_FIXED,
    RISK_OPTIMIZED_ETA,
    RISK_PROBABILITY_WEIGHTED,
    SAFETY_BRAKE_DISTANCE,
    SAFETY_CONFIDENCE_CHANCE,
    SAFETY_NOMINAL_SAFE_DISTANCE,
    SAFETY_SCALAR_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    VARIANT_PROPOSED,
    make_constant_accel_lead_prediction,
    original_nair_acc_config,
    required_standoff,
    safety_function,
)
from utils.low_level_control import LowLevelControl
from utils.synthetic_lk_cutin_prediction import (
    SyntheticCutInConfig,
    SyntheticLaneKeepingCutInPredictor,
)
from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import relation_from_waypoints
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (
    INTENTION_NAMES,
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    REL_OTHER,
    REL_RIGHT_ADJACENT,
)
from predictor.stdan_vel.acc_adapter import STDANVelACCAdapter
from predictor.lstm.acc_adapter import LSTMACCAdapter


class ACCNairSMPCAgent(object):
    """CARLA adapter for the 1D ACC Nair-style multimodal SMPC controller."""

    # Predicted seconds-to-ego-lane at which the gap re-establishment clock starts.
    # Waiting for the lane assignment to flip is too late: the braking is over by
    # then. One second ahead lands the relaxation on the deceleration itself.
    GAP_RECOVERY_ONSET_TLC_S = 1.0
    # Below this probability a predicted mode is dropped outright, for every
    # STDAN policy (an explicit ``cutin_gate<value>`` overrides it).  For the
    # chance constraint it is Benciolini's vanishing: the ellipse collapses to
    # a point as beta -> 0 and a point can be driven around, whereas the 1-D
    # counterpart still orders the ego behind the hypothetical cut-in even at
    # zero standoff.  For the plain multimodal policy a 1-2 % hypothesis held
    # at full standoff kept the ego behind a lead that had already left the
    # lane (cut-out, 2026-09-05).  Lane-keeping neighbours score <= 0.05 in the
    # 2026-09-04 sweep; 0.1 keeps a margin above that.
    CUTIN_CHANCE_VANISH_BELOW = 0.1

    def __init__(
            self,
            vehicle,
            goal_location,
            nominal_speed_mps=8.0,
            dt=0.2,
            N=8,
            N_modes=2,
            smpc_config="proposed"):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.DT = dt
        self.N = N
        self.nominal_speed = nominal_speed_mps
        self.goal_reached = False
        self.smpc_config = str(smpc_config)
        self.mode_probabilities = self._parse_mode_probabilities(smpc_config)
        self.predictor_type = self._parse_predictor_type(smpc_config)
        self.best_mode_only = self._parse_best_mode_only(smpc_config)
        threshold = self._parse_cutin_probability_threshold(smpc_config)
        if threshold is None:
            threshold = (self.CUTIN_CHANCE_VANISH_BELOW
                         if self.predictor_type in ("stdan_3int", "stdan_vel") else 0.0)
        self.cutin_probability_threshold = threshold
        self.cutin_clearance_ramp_ref = self._parse_cutin_clearance_ramp_ref(smpc_config)
        self.cutin_chance_ref = self._parse_cutin_chance_ref(smpc_config)
        if self.cutin_chance_ref > 0.0 and self.cutin_clearance_ramp_ref > 0.0:
            raise ValueError(
                "cutin_ramp and cutin_chance both set the cut-in standoff scale; use one")
        self.cutin_clearance_tlc_ref = self._parse_cutin_clearance_tlc_ref(smpc_config)
        self.gap_recovery_s = self._parse_gap_recovery_s(smpc_config)
        # Drive the ego as the exact double integrator the MPC models.  The
        # throttle/brake maps in LowLevelControl realise about 1.5x the
        # commanded acceleration plus 0.9 m/s^2 at zero command on this
        # vehicle, which turns the following loop into a limit cycle; the
        # ``carla_actuator`` token keeps that path for comparison.
        self.ideal_actuator = "carla_actuator" not in self.smpc_config
        self._ideal_v_set_prev = None
        # vehicle id -> time the gap re-establishment clock started for it
        self._ego_lane_entry_time = {}
        # vehicle id -> gap / full standoff when that clock started; the
        # relaxation accepts the gap the vehicle actually arrived with
        self._gap_recovery_start_scale = {}
        # vehicle id -> predicted seconds until it reaches the ego lane, from the
        # previous cycle.  Read one step late because the recovery scale has to be
        # decided before the predictor that produces it runs.
        self._predicted_lane_entry_s = {}
        self.enable_wandb_logging = os.getenv("ACC_NAIR_WANDB", "0") == "1"
        self._wandb = None
        self.goal_location = goal_location
        self._frenet_traj = None
        self._stdan_model_yaw = 0.0
        self._ego_route_debug = {"source": "uninitialized"}

        variant = self._parse_variant(smpc_config)
        self.controller_config = original_nair_acc_config(
            horizon=N,
            dt=dt,
            desired_speed=nominal_speed_mps,
            num_modes=self._parse_controller_num_modes(smpc_config, N_modes),
            controller_variant=variant,
            gurobi_time_limit=0.15,
        )
        self.controller_config.safety_constraint_mode = self._parse_safety_constraint_mode(smpc_config)
        self.controller_config.brake_distance_bound_mode = (
            self._parse_brake_distance_bound_mode(smpc_config)
        )
        self.controller_config.risk_allocation_mode = self._parse_risk_allocation_mode(smpc_config)
        self.controller_config.optimize_k = self._parse_optimize_k(smpc_config)
        self.controller_config.__post_init__()
        self.controller = NairACCSMPC(self.controller_config)
        self.synthetic_predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=N,
                dt=dt,
                probabilities=self.mode_probabilities,
                ego_lane_d=0.0,
            )
        )
        self.stdan_predictor = None
        self.stdan_history_secs = None
        self.stdan_history_closeness_eps = None
        self.agent_history = None
        if self.predictor_type in ("stdan_3int", "stdan_vel", "lstm"):
            if self.predictor_type == "lstm":
                self.stdan_predictor = LSTMACCAdapter(
                    ckpt_path=os.getenv("ACC_NAIR_LSTM_CKPT", "predictor/lstm/ckpt/best_model.pt"),
                    history=3.0,
                    future=5.0,
                    dt=0.1,
                )
            elif self.predictor_type == "stdan_vel":
                self.stdan_predictor = STDANVelACCAdapter(
                    ckpt_path="predictor/stdan_vel/checkpoints/0113_ratio211/best_model.pt",
                    history=3.0,
                    future=5.0,
                    dt=0.1,
                    mc_dropout=False,
                )
            else:
                stdan_ckpt = os.getenv(
                    "ACC_NAIR_STDAN_CKPT",
                    os.getenv(
                        "ACC_NAIR_STDAN_3INT_CKPT",
                        "predictor/stdan_3int_signed_tcross_velint/checkpoints_out30/best_model.pt",
                    ),
                )
                self.stdan_predictor = STDAN3IntACCAdapter(
                    ckpt_path=stdan_ckpt,
                    history=3.0,
                    future=max(3.0, float(N) * float(dt)),
                    dt=0.1,
                    mc_dropout=False,
                    # 0.2 s acceleration feature: cancels the target vehicles'
                    # 0.2 s speed ripple that dithered the prediction each tick.
                    # The LSTM adapter builds its own STDAN3IntACCAdapter with
                    # the trained 0.1 s difference and is not affected.
                    accel_baseline_s=0.2,
                )
            self.stdan_history_secs = [
                round(i * self.stdan_predictor.dt, 2)
                for i in range(self.stdan_predictor.full_length - 1, -1, -1)
            ]
            self.stdan_history_closeness_eps = self.stdan_predictor.dt / 2.0 + 1e-6
            history_max_length = max(
                100,
                int(np.ceil((max(self.stdan_history_secs) + self.stdan_predictor.dt) / max(self.DT, 1.0e-6))) + 5,
            )
            self.agent_history = AgentHistory(self.world.get_actors(), history_max_length=history_max_length)
        self._low_level_control = LowLevelControl(vehicle)
        self.policy_log = []
        self._prev_actual_accel_sample = None
        self._prev_accel_cmd = 0.0
        self._prev_accel_cmd_time_s = None

    def done(self):
        return self.goal_reached

    @staticmethod
    def _build_current_lane_reference(
            init_waypoint,
            goal_location,
            ego_location,
            vehicle_yaw_deg,
            step_distance=1.0):
        if init_waypoint is None:
            raise RuntimeError("Could not find a driving waypoint for ACC ego route.")

        try:
            goal_distance = ego_location.distance(goal_location)
        except Exception:
            goal_distance = 0.0
        target_distance = max(80.0, float(goal_distance) + 30.0)

        direction, direction_score = ACCNairSMPCAgent._select_lane_chain_direction(
            init_waypoint,
            goal_location,
            vehicle_yaw_deg,
            step_distance,
        )
        if direction_score < 0.5:
            way_s, way_xy, way_yaw = ACCNairSMPCAgent._build_straight_heading_reference(
                ego_location,
                vehicle_yaw_deg,
                target_distance,
                step_distance,
            )
            debug = {
                "source": "straight_heading_fallback",
                "direction": direction,
                "direction_score": float(direction_score),
            }
            return way_s, way_xy, way_yaw, debug

        waypoints = [init_waypoint]
        distance = 0.0
        current = init_waypoint
        while distance < target_distance:
            candidates = ACCNairSMPCAgent._lane_chain_candidates(
                current, direction, step_distance)
            if not candidates:
                break
            next_waypoint = ACCNairSMPCAgent._select_current_lane_candidate(
                current, candidates)
            distance += next_waypoint.transform.location.distance(
                current.transform.location)
            waypoints.append(next_waypoint)
            current = next_waypoint

        if len(waypoints) < 2:
            raise RuntimeError("Could not extend ACC ego current-lane route.")
        way_s, way_xy, way_yaw = ACCNairSMPCAgent._waypoints_to_directional_frenet_reference(
            waypoints)
        ego_start_xy = np.array([ego_location.x, -ego_location.y], dtype=float)
        way_xy = way_xy + (ego_start_xy - way_xy[0])
        if not ACCNairSMPCAgent._reference_aligns_with_vehicle_heading(
                way_xy, vehicle_yaw_deg):
            way_s, way_xy, way_yaw = ACCNairSMPCAgent._build_straight_heading_reference(
                ego_location,
                vehicle_yaw_deg,
                target_distance,
                step_distance,
            )
            debug = {
                "source": "straight_heading_alignment_fallback",
                "direction": direction,
                "direction_score": float(direction_score),
            }
            return way_s, way_xy, way_yaw, debug

        debug = {
            "source": "current_lane_waypoints",
            "direction": direction,
            "direction_score": float(direction_score),
        }
        return way_s, way_xy, way_yaw, debug

    @staticmethod
    def _select_lane_chain_direction(
            init_waypoint,
            goal_location,
            vehicle_yaw_deg,
            step_distance):
        start = init_waypoint.transform.location
        del goal_location
        yaw = np.radians(float(vehicle_yaw_deg))
        heading_vec = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)

        scores = []
        for direction in ("next", "previous"):
            candidates = ACCNairSMPCAgent._lane_chain_candidates(
                init_waypoint, direction, step_distance)
            if not candidates:
                continue
            waypoint = ACCNairSMPCAgent._select_current_lane_candidate(
                init_waypoint, candidates)
            loc = waypoint.transform.location
            step_vec = np.array([
                float(loc.x - start.x),
                float(loc.y - start.y),
            ], dtype=float)
            denom = max(
                np.linalg.norm(heading_vec) * np.linalg.norm(step_vec),
                1.0e-6,
            )
            scores.append((float(np.dot(heading_vec, step_vec) / denom), direction))

        if not scores:
            raise RuntimeError("Could not find next/previous ACC ego lane waypoints.")
        score, direction = max(scores, key=lambda item: item[0])
        return direction, score

    @staticmethod
    def _lane_chain_candidates(waypoint, direction, step_distance):
        if direction == "next":
            return waypoint.next(step_distance)
        if direction == "previous" and hasattr(waypoint, "previous"):
            return waypoint.previous(step_distance)
        return []

    @staticmethod
    def _select_current_lane_candidate(current_waypoint, next_waypoints):
        same_lane = [
            waypoint for waypoint in next_waypoints
            if (waypoint.road_id == current_waypoint.road_id
                and waypoint.lane_id == current_waypoint.lane_id)
        ]
        if same_lane:
            return same_lane[0]

        same_lane_id = [
            waypoint for waypoint in next_waypoints
            if waypoint.lane_id == current_waypoint.lane_id
        ]
        if same_lane_id:
            return same_lane_id[0]

        return next_waypoints[0]

    @staticmethod
    def _waypoints_to_directional_frenet_reference(waypoints):
        way_xy = np.array([
            [
                waypoint.transform.location.x,
                -waypoint.transform.location.y,
            ]
            for waypoint in waypoints
        ], dtype=float)
        diff_xy = np.diff(way_xy, axis=0)
        segment_lengths = np.sqrt(np.sum(np.square(diff_xy), axis=1))
        valid = segment_lengths > 1.0e-6
        if not np.any(valid):
            raise RuntimeError("ACC ego current-lane route has no valid segments.")
        way_s = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        segment_yaw = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])
        way_yaw = np.empty(way_xy.shape[0], dtype=float)
        way_yaw[:-1] = segment_yaw
        way_yaw[-1] = segment_yaw[-1]
        return way_s, way_xy, way_yaw

    @staticmethod
    def _build_straight_heading_reference(
            ego_location,
            vehicle_yaw_deg,
            target_distance,
            step_distance):
        way_s = np.arange(0.0, target_distance + step_distance, step_distance)
        psi = -fth.fix_angle(np.radians(float(vehicle_yaw_deg)))
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        start_xy = np.array([ego_location.x, -ego_location.y], dtype=float)
        way_xy = start_xy[None, :] + way_s[:, None] * heading_rhs[None, :]
        way_yaw = np.full(way_s.shape, psi, dtype=float)
        return way_s, way_xy, way_yaw

    @staticmethod
    def _reference_aligns_with_vehicle_heading(way_xy, vehicle_yaw_deg):
        if len(way_xy) < 2:
            return False
        psi = -fth.fix_angle(np.radians(float(vehicle_yaw_deg)))
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        step_rhs = way_xy[1] - way_xy[0]
        denom = max(np.linalg.norm(heading_rhs) * np.linalg.norm(step_rhs), 1.0e-6)
        return float(np.dot(heading_rhs, step_rhs) / denom) >= 0.5

    def _ensure_frenet_reference_initialized(self):
        if self._frenet_traj is not None:
            return

        carla_map = self.world.get_map()
        vehicle_transform = self.vehicle.get_transform()
        init_waypoint = carla_map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        way_s, way_xy, way_yaw, self._ego_route_debug = self._build_current_lane_reference(
            init_waypoint,
            self.goal_location,
            vehicle_transform.location,
            vehicle_transform.rotation.yaw,
        )
        self._frenet_traj = fth.FrenetTrajectoryHandler(
            way_s, way_xy, way_yaw, s_resolution=0.5)
        self._stdan_model_yaw = float(self._frenet_traj.trajectory[0, 3])

    def run_step(self, pred_dict):
        del pred_dict

        x, y, psi, speed = self._ego_state_rhs()
        self._ensure_frenet_reference_initialized()
        s_ego, ey, epsi = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
        z0 = np.array([x, y, psi, speed])
        u0 = np.array([self.controller_config.a_min, 0.0])
        solve_time = np.nan
        is_feasible = False
        time_s = self._elapsed_seconds()
        previous_accel_cmd = self._prev_accel_cmd
        command_dt = self._command_dt(time_s)
        raw_action = self.controller_config.a_min
        command_jerk = None

        if self.goal_reached or self._frenet_traj.reached_trajectory_end(s_ego, resolution=5.0):
            self.goal_reached = True
            raw_action = self.controller_config.a_min
            action = self._limit_accel_command(raw_action, previous_accel_cmd, command_dt)
            v_des = max(0.0, speed + action * self.DT)
            df_des = 0.0
            u0 = np.array([action, df_des])
        else:
            prediction_bundle = self._prediction_bundle(s_ego, speed)
            start = time.time()
            solution = self.controller.solve(
                np.array([s_ego, speed]),
                prediction_bundle["prediction"],
                previous_accel_cmd=previous_accel_cmd,
                command_dt=command_dt,
            )
            solve_time = time.time() - start
            raw_action = float(solution.action)
            action = self._limit_accel_command(raw_action, previous_accel_cmd, command_dt)
            df_des = self._lane_keeping_steer(ey, epsi)
            v_des = np.clip(speed + action * self.DT,
                            self.controller_config.v_min,
                            self.controller_config.v_max)
            u0 = np.array([action, df_des])
            is_feasible = bool(solution.feasible)
            min_gap = self._min_gap(solution, prediction_bundle["prediction"])
            actual_accel = self._actual_longitudinal_accel()
            actual_jerk = self._actual_longitudinal_jerk(time_s, actual_accel)
            command_jerk = self._command_jerk(action, command_dt)
            self.policy_log.append({
                "time_s": time_s,
                "status": solution.status,
                "feasible": is_feasible,
                "solve_time": solve_time,
                "ego_s": float(s_ego),
                "ego_v": float(speed),
                "accel_cmd_raw": raw_action,
                "accel_cmd": action,
                "command_dt": command_dt,
                "command_jerk": command_jerk,
                "command_jerk_limit": self.controller_config.jerk_limit,
                "actual_accel": actual_accel,
                "actual_jerk": actual_jerk,
                "steer_cmd": float(df_des),
                "target_s": float(prediction_bundle["selected_target_state"][0]),
                "target_d": float(prediction_bundle["selected_target_state"][1]),
                "target_v": float(prediction_bundle["selected_target_state"][2]),
                "mode_names": prediction_bundle["prediction"].mode_names,
                "mode_probabilities": prediction_bundle["prediction"].probabilities.tolist(),
                "prediction_num_modes": int(prediction_bundle["prediction"].num_modes),
                "controller_config_num_modes": int(self.controller_config.num_modes),
                "safety_constraint_mode": solution.safety_constraint_mode,
                "brake_distance_bound_mode": self.controller_config.brake_distance_bound_mode,
                "brake_distance_hard_band_width": (
                    self.controller_config.brake_distance_hard_band_width
                ),
                "risk_allocation_mode": solution.risk_allocation_mode,
                "optimize_k": bool(self.controller_config.optimize_k),
                "solver_message": solution.solver_message,
                "solve_path": solution.solve_path,
                "solve_timing": dict(solution.timing_breakdown or {}),
                "num_decision_variables": int(solution.num_decision_variables),
                "baseline_decision_variables": int(solution.baseline_decision_variables),
                "policy_group_count": int(solution.policy_group_count),
                "policy_split_steps": list(solution.policy_split_steps),
                "first_policy_split_step": (
                    -1 if solution.first_policy_split_step is None
                    else int(solution.first_policy_split_step)
                ),
                "chance_margin_min": float(solution.chance_margin_min),
                "tightening_max": float(solution.tightening_max),
                "clearance_scale_now": self._clearance_scale_now(prediction_bundle["prediction"]),
                "predictor_time": prediction_bundle.get("debug", {}).get("predictor_time"),
                "predictor_type": self.predictor_type,
                "ego_route_debug": self._ego_route_debug,
                "stdan_debug": prediction_bundle.get("debug", {}),
                "min_gap": min_gap,
                "prediction_covariance": self._prediction_covariance_debug(
                    prediction_bundle["prediction"]
                ),
                "acc_reference": self._acc_reference_debug(solution),
            })
            self._log_wandb(self.policy_log[-1], solution, prediction_bundle)

        self._remember_accel_cmd(time_s, action)
        control = self._low_level_control.update(speed, action, v_des, df_des)
        if self.ideal_actuator:
            # The throttle/brake stay applied so the wheels keep pace with the
            # chassis; the velocity override just pins the speed each tick.
            self._apply_ideal_longitudinal(speed, action, command_dt)
        return control, z0, u0, is_feasible, solve_time

    def _apply_ideal_longitudinal(self, speed, action, command_dt):
        """Integrate the accel command and pin the speed along the heading.

        ``set_target_velocity`` is applied before the physics step, which then
        moves the speed by whatever the drivetrain and drag do within the tick.
        The difference between the speed set last tick and the speed measured
        now is that per-tick physics effect; it is added back in advance so
        the measured speed follows the integrated command.
        """
        eaten = 0.0
        if self._ideal_v_set_prev is not None:
            eaten = float(self._ideal_v_set_prev) - float(speed)
        v_next = max(0.0, float(speed) + float(action) * float(command_dt) + eaten)
        yaw = np.radians(self.vehicle.get_transform().rotation.yaw)
        self.vehicle.set_target_velocity(carla.Vector3D(
            x=v_next * np.cos(yaw), y=v_next * np.sin(yaw), z=0.0))
        self._ideal_v_set_prev = v_next

    def get_cut_in_log(self):
        return {
            "controller": "acc_nair_smpc",
            "mode_names": ["lane_keeping", "cutin"],
            "mode_probabilities": list(self.mode_probabilities),
            "cutin_chance_ref": float(self.cutin_chance_ref),
            "cutin_probability_threshold": float(self.cutin_probability_threshold),
            "steps": self.policy_log,
        }

    def _prediction_bundle(self, s_ego, speed):
        if self.predictor_type == "constant_velocity_lead":
            return self._constant_velocity_lead_prediction_bundle(s_ego, speed)
        if self.predictor_type in ("stdan_3int", "stdan_vel", "lstm"):
            bundle = self._stdan_prediction_bundle(s_ego, speed)
            if bundle is not None:
                return self._maybe_collapse_bundle_to_best_mode(bundle)
        target_state = self._selected_target_state(s_ego)
        synthetic = self.synthetic_predictor.predict(
            ego_state=np.array([s_ego, speed]),
            target_state=target_state,
            probabilities=self.mode_probabilities,
        )
        return self._maybe_collapse_bundle_to_best_mode({
            "prediction": synthetic.prediction,
            "selected_target_state": target_state,
            "debug": {"fallback": self.predictor_type != "synthetic"},
        })

    def _constant_velocity_lead_prediction_bundle(self, s_ego, speed):
        lead_state, lead_debug = self._selected_ego_lane_lead_state(s_ego, speed)
        horizon = int(self.controller_config.horizon)
        if lead_debug["selected_vehicle_id"] is None:
            means = np.zeros((1, horizon + 1, 2), dtype=float)
            means[:, :, 0] = float(s_ego)
            means[:, :, 1] = float(self.nominal_speed)
            prediction = MultimodalLeadPrediction(
                means=means,
                probabilities=np.array([1.0]),
                covariances=np.zeros((1, horizon + 1, 2, 2), dtype=float),
                mode_names=["default_no_ego_lane_lead"],
                active_mask=np.zeros((1, horizon + 1), dtype=bool),
            )
        else:
            prediction = make_constant_accel_lead_prediction(
                s0=float(lead_state[0]),
                v0=float(lead_state[2]),
                accelerations=np.array([0.0]),
                probabilities=np.array([1.0]),
                horizon=horizon,
                dt=float(self.controller_config.dt),
                covariance=np.diag([0.25, 0.09]),
                mode_names=["constant_velocity_lead"],
            )
        return {
            "prediction": prediction,
            "selected_target_state": lead_state,
            "debug": {
                "fallback": False,
                "predictor_type": self.predictor_type,
                "predictor_time": 0.0,
                "prediction_num_modes": int(prediction.num_modes),
                "controller_config_num_modes": int(self.controller_config.num_modes),
                "constant_velocity_lead": lead_debug,
            },
        }

    def _maybe_collapse_bundle_to_best_mode(self, bundle):
        if not self.best_mode_only:
            return bundle
        prediction = bundle["prediction"]
        if prediction.num_modes <= 1:
            bundle.setdefault("debug", {})["best_mode_only"] = True
            return bundle
        best_idx = int(np.argmax(prediction.probabilities))
        mode_names = prediction.mode_names or [f"mode_{idx}" for idx in range(prediction.num_modes)]
        policy_tree = None
        if prediction.policy_tree is not None:
            policy_tree = PolicySharingTree(
                policy_groups_by_step=prediction.policy_tree.policy_groups_by_step[
                    best_idx: best_idx + 1
                ].copy(),
                policy_split_steps=(),
                mode_names=[mode_names[best_idx]],
            )
        collapsed = MultimodalLeadPrediction(
            means=prediction.means[best_idx: best_idx + 1].copy(),
            probabilities=np.array([1.0]),
            covariances=prediction.covariances[best_idx: best_idx + 1].copy(),
            T=None if prediction.T is None else prediction.T[best_idx: best_idx + 1].copy(),
            c=None if prediction.c is None else prediction.c[best_idx: best_idx + 1].copy(),
            mode_names=[mode_names[best_idx]],
            policy_tree=policy_tree,
            active_mask=prediction.active_mask[best_idx: best_idx + 1].copy(),
            k_group_map=(
                None if prediction.k_group_map is None
                else prediction.k_group_map[best_idx: best_idx + 1].copy()
            ),
            k_group_names=prediction.k_group_names,
        )
        collapsed_bundle = dict(bundle)
        collapsed_bundle["prediction"] = collapsed
        debug = dict(collapsed_bundle.get("debug", {}))
        debug.update({
            "best_mode_only": True,
            "best_mode_source_num_modes": int(prediction.num_modes),
            "best_mode_index": best_idx,
            "best_mode_probability": float(prediction.probabilities[best_idx]),
            "best_mode_name": mode_names[best_idx],
            "prediction_num_modes": int(collapsed.num_modes),
            "controller_config_num_modes": int(self.controller_config.num_modes),
        })
        collapsed_bundle["debug"] = debug
        return collapsed_bundle

    def _stdan_prediction_bundle(self, s_ego, speed):
        self._update_agent_history()
        trackings = self._stdan_trackings()
        target_states = {}
        target_relations = {}
        target_current_rhs_xy = {}
        nearest_target = None
        carla_map = self.world.get_map()
        ego_wp = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            role = actor.attributes.get("role_name", "")
            if "target" not in role:
                continue
            x, y, psi, actor_speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv < s_ego - 5.0:
                continue
            target_states[actor.id] = np.array([s_tv, d_tv, actor_speed], dtype=float)
            target_current_rhs_xy[actor.id] = np.array([x, y], dtype=float)
            target_wp = carla_map.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            # Only the same-lane test is taken from the waypoints: it compares
            # road_id/lane_id directly and is exact.  Which *side* an adjacent lane
            # sits on comes from the Frenet offset instead, because CARLA's
            # get_left_lane()/get_right_lane() are posed in the simulator's
            # left-handed frame and disagreed with the sign of d in 8 of 10 runs of
            # the 2026-09-02 sweep -- and disagreed with each other across runs of
            # identical geometry.  Everything downstream (lane_occupancy_from_d, the
            # LLC/RLC to cut-in mapping, the controller's own Frenet state) reads d
            # with +d to the left, so the relation has to be posed the same way.
            relation = relation_from_waypoints(ego_wp, target_wp)
            if relation != REL_EGO_LANE:
                relation = self._relation_from_frenet_offset(d_tv)
            target_relations[actor.id] = relation
            if nearest_target is None or s_tv - s_ego < nearest_target[0]:
                nearest_target = (s_tv - s_ego, target_states[actor.id])

        if not target_states:
            return None

        def lane_membership_fn(target_id, sampled_raw_traj_xy):
            return self._stdan_ego_lane_memberships_from_carla_waypoints(
                ego_wp,
                target_current_rhs_xy.get(target_id),
                sampled_raw_traj_xy,
            )

        gap_recovery_elapsed, gap_recovery_start_scale = self._gap_recovery_elapsed(
            target_relations, target_states, np.array([s_ego, speed]))
        predictor_start = time.time()
        try:
            result = self.stdan_predictor.predict_acc(
                ego_state=np.array([s_ego, speed]),
                target_states_frenet=target_states,
                target_relations=target_relations,
                trackings=trackings,
                frenet_handler=self._frenet_traj,
                horizon=self.N,
                desired_speed=self.nominal_speed,
                num_modes=self.controller_config.num_modes,
                controller_dt=self.DT,
                lane_membership_fn=lane_membership_fn,
                model_yaw=self._stdan_model_yaw,
                cutin_probability_threshold=self.cutin_probability_threshold,
                cutin_clearance_ramp_ref=self.cutin_clearance_ramp_ref,
                cutin_chance_ref=self.cutin_chance_ref,
                cutin_clearance_tlc_ref=self.cutin_clearance_tlc_ref,
                gap_recovery_elapsed=gap_recovery_elapsed,
                gap_recovery_s=self.gap_recovery_s,
                gap_recovery_start_scale=gap_recovery_start_scale,
            )
            predictor_time = time.time() - predictor_start
        except Exception as exc:
            return {
                "prediction": self.synthetic_predictor.predict(
                    np.array([s_ego, speed]), nearest_target[1], self.mode_probabilities
                ).prediction,
                "selected_target_state": nearest_target[1],
                "debug": {
                    "fallback": True,
                    "reason": repr(exc),
                    "predictor_time": time.time() - predictor_start,
                    "controller_config_num_modes": int(self.controller_config.num_modes),
                },
            }
        selected_target_state = nearest_target[1] if nearest_target is not None else np.array([s_ego, 0.0, speed])
        result["selected_target_state"] = selected_target_state
        self._cache_predicted_lane_entry(result.get("processed_targets", []))
        result["debug"] = {
            "num_targets": len(target_states),
            "relations": target_relations,
            "stdan_model_yaw": self._stdan_model_yaw,
            # Which weights produced these predictions.  The checkpoint comes
            # from an environment variable, so runs made in different shells
            # can silently use different predictors; the log has to say which.
            "stdan_ckpt": str(getattr(self.stdan_predictor, "ckpt_path", "")),
            "gap_recovery_start_scale": dict(self._gap_recovery_start_scale),
            "predictor_time": predictor_time,
            "prediction_num_modes": int(result["prediction"].num_modes),
            "controller_config_num_modes": int(self.controller_config.num_modes),
            "scenario_metadata": result.get("scenario_metadata", []),
            "processed_targets": self._stdan_processed_target_debug(
                result.get("processed_targets", [])),
        }
        return result

    def _acc_reference_debug(self, solution):
        reference = solution.reference
        return {
            "s_ref": np.asarray(reference.s_ref, dtype=float).round(4).tolist(),
            "v_ref": np.asarray(reference.v_ref, dtype=float).round(4).tolist(),
            "a_ref": np.asarray(reference.a_ref, dtype=float).round(4).tolist(),
            "prev_u": np.asarray(reference.prev_u, dtype=float).round(4).tolist(),
            "x_nominal": np.asarray(solution.x_nominal, dtype=float).round(4).tolist(),
            "u_nominal": np.asarray(solution.u_nominal, dtype=float).round(4).tolist(),
        }

    @staticmethod
    def _clearance_scale_now(prediction):
        """Standoff scale applied to the lead the ego has right now, for plotting.

        Only modes active at step 0 count, so a low-probability cut-in
        hypothesis about a next-lane vehicle does not show up as a relaxation
        of the gap to the vehicle actually ahead.
        """
        active_now = np.asarray(prediction.active_mask, dtype=bool)[:, 0]
        if not active_now.any():
            return float("nan")
        return float(np.min(np.asarray(prediction.clearance_scale, dtype=float)[active_now, 0]))

    def _prediction_covariance_debug(self, prediction):
        covariances = np.asarray(prediction.covariances, dtype=float)
        if covariances.ndim != 4 or covariances.shape[-2:] != (2, 2):
            return {}
        diag = np.maximum(np.diagonal(covariances, axis1=2, axis2=3), 0.0)
        var_s = diag[:, :, 0]
        var_v = diag[:, :, 1]
        std_s = np.sqrt(var_s)
        std_v = np.sqrt(var_v)
        mode_names = prediction.mode_names or [
            f"mode_{idx}" for idx in range(prediction.num_modes)
        ]
        return {
            "mode_names": [str(name) for name in mode_names],
            "active_mask": np.asarray(prediction.active_mask, dtype=bool).tolist(),
            "var_s": np.asarray(var_s, dtype=float).round(6).tolist(),
            "var_v": np.asarray(var_v, dtype=float).round(6).tolist(),
            "std_s": np.asarray(std_s, dtype=float).round(6).tolist(),
            "std_v": np.asarray(std_v, dtype=float).round(6).tolist(),
        }

    def _stdan_processed_target_debug(self, processed_targets):
        summaries = []
        for target in processed_targets:
            raw = target.raw_prediction or {}
            traj_frenet = np.asarray(raw.get("pred_traj_frenet", []), dtype=float)
            full_traj_frenet = np.asarray(raw.get("full_pred_traj_frenet", []), dtype=float)
            sampled_xy = np.asarray(raw.get("sampled_raw_traj_xy", []), dtype=float)
            full_xy = np.asarray(raw.get("raw_traj_xy", []), dtype=float)
            lead_covariance = np.asarray(raw.get("pred_lead_covariance", []), dtype=float)
            mode_d = {}
            for mode_idx, mode_name in enumerate(INTENTION_NAMES):
                if traj_frenet.ndim < 3 or mode_idx >= traj_frenet.shape[0] or traj_frenet.shape[1] == 0:
                    continue
                d_values = traj_frenet[mode_idx, :, 1]
                mode_d[mode_name] = {
                    "start": float(d_values[0]),
                    "end": float(d_values[-1]),
                    "min": float(np.min(d_values)),
                    "max": float(np.max(d_values)),
                    "delta": float(d_values[-1] - d_values[0]),
                }
            occupancy = target.branch_info.get("ego_lane_occupancy_mask", [])
            mode_membership_arr = np.asarray(
                target.branch_info.get("mode_lane_membership", []),
                dtype=bool,
            )
            pred_traj_frenet = {}
            full_pred_traj_frenet = {}
            pred_traj_xy = {}
            full_pred_traj_xy = {}
            mode_lane_membership = {}
            predictor_covariance = {}
            for mode_idx, mode_name in enumerate(INTENTION_NAMES):
                if traj_frenet.ndim == 3 and mode_idx < traj_frenet.shape[0]:
                    pred_traj_frenet[mode_name] = np.asarray(
                        traj_frenet[mode_idx], dtype=float
                    ).round(4).tolist()
                if full_traj_frenet.ndim == 3 and mode_idx < full_traj_frenet.shape[0]:
                    full_pred_traj_frenet[mode_name] = np.asarray(
                        full_traj_frenet[mode_idx], dtype=float
                    ).round(4).tolist()
                if sampled_xy.ndim == 3 and mode_idx < sampled_xy.shape[0]:
                    pred_traj_xy[mode_name] = np.asarray(
                        sampled_xy[mode_idx], dtype=float
                    ).round(4).tolist()
                if full_xy.ndim == 3 and mode_idx < full_xy.shape[0]:
                    full_pred_traj_xy[mode_name] = np.asarray(
                        full_xy[mode_idx], dtype=float
                    ).round(4).tolist()
                if mode_membership_arr.ndim == 2 and mode_idx < mode_membership_arr.shape[0]:
                    mode_lane_membership[mode_name] = [
                        bool(value)
                        for value in mode_membership_arr[mode_idx].reshape(-1)
                    ]
                if (
                        lead_covariance.ndim == 4
                        and mode_idx < lead_covariance.shape[0]
                        and lead_covariance.shape[-2:] == (2, 2)):
                    diag = np.maximum(np.diagonal(
                        lead_covariance[mode_idx], axis1=1, axis2=2), 0.0)
                    predictor_covariance[mode_name] = {
                        "var_s": np.asarray(diag[:, 0], dtype=float).round(6).tolist(),
                        "var_v": np.asarray(diag[:, 1], dtype=float).round(6).tolist(),
                        "std_s": np.sqrt(diag[:, 0]).round(6).tolist(),
                        "std_v": np.sqrt(diag[:, 1]).round(6).tolist(),
                    }
            summaries.append({
                "vehicle_id": int(target.vehicle_id),
                "relation_to_ego_lane": target.relation_to_ego_lane,
                "acc_mode_raw_indices": {
                    mode.mode_name: [int(idx) for idx in mode.raw_mode_indices]
                    for mode in target.mode_predictions
                },
                "raw_mode_prob": {
                    str(key): float(value)
                    for key, value in target.raw_mode_prob.items()
                },
                "acc_mode_prob": {
                    str(key): float(value)
                    for key, value in target.acc_mode_prob.items()
                },
                "signed_t_cross": float(raw.get("signed_t_cross", 0.0)),
                "raw_longitudinal_prob": {
                    str(idx): float(value)
                    for idx, value in enumerate(np.asarray(
                        raw.get("raw_longitudinal_prob", []), dtype=float).reshape(-1))
                },
                "selected_longitudinal_idx": raw.get("selected_longitudinal_idx"),
                "selected_longitudinal_mode": raw.get("selected_longitudinal_mode"),
                "ego_lane_membership_split_step": int(target.branch_info.get(
                    "ego_lane_membership_split_step",
                    target.branch_info.get("branch_step", -1),
                )),
                "branch_step": int(target.branch_info.get("branch_step", -1)),
                "ego_lane_start_idx": int(target.branch_info.get("ego_lane_start_idx", 999)),
                "ego_lane_end_idx": int(target.branch_info.get("ego_lane_end_idx", -1)),
                "lane_membership_source": str(target.branch_info.get(
                    "lane_membership_source", "")),
                "trajectory_suggested_cutin_raw_idx": target.branch_info.get(
                    "trajectory_suggested_cutin_raw_idx"),
                "used_trajectory_aware_cutin_mapping": bool(target.branch_info.get(
                    "used_trajectory_aware_cutin_mapping", False)),
                "cutin_chance_confidences": target.branch_info.get(
                    "cutin_chance_confidences", {}),
                "cutin_probability_threshold": float(target.branch_info.get(
                    "cutin_probability_threshold", 0.0)),
                "gated_cutin_modes": list(target.branch_info.get("gated_cutin_modes", [])),
                "ego_lane_occupancy_mask": [
                    bool(value) for value in np.asarray(occupancy, dtype=bool).reshape(-1)
                ],
                "mode_lane_membership": mode_lane_membership,
                "mode_d": mode_d,
                "pred_traj_frenet": pred_traj_frenet,
                "full_pred_traj_frenet": full_pred_traj_frenet,
                "pred_traj_xy": pred_traj_xy,
                "full_pred_traj_xy": full_pred_traj_xy,
                "predictor_covariance": predictor_covariance,
            })
        return summaries

    def _stdan_ego_lane_memberships_from_carla_waypoints(
            self,
            ego_wp,
            current_rhs_xy,
            sampled_raw_traj_xy):
        sampled = np.asarray(sampled_raw_traj_xy, dtype=float)
        if sampled.ndim != 3:
            raise ValueError("sampled_raw_traj_xy must have shape (num_modes, horizon, 2)")
        memberships = np.zeros((sampled.shape[0], sampled.shape[1] + 1), dtype=bool)
        current_on_ego_lane = self._rhs_xy_on_ego_lane(ego_wp, current_rhs_xy)
        memberships[:, 0] = current_on_ego_lane
        for mode_idx in range(sampled.shape[0]):
            for step_idx, point in enumerate(sampled[mode_idx], start=1):
                memberships[mode_idx, step_idx] = self._rhs_xy_on_ego_lane(ego_wp, point)
        return memberships

    def _rhs_xy_on_ego_lane(self, ego_wp, rhs_xy):
        if ego_wp is None or rhs_xy is None:
            return False
        point = np.asarray(rhs_xy, dtype=float).reshape(-1)
        if point.size < 2:
            return False
        location = carla.Location(x=float(point[0]), y=-float(point[1]), z=ego_wp.transform.location.z)
        waypoint = self.world.get_map().get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return False
        return waypoint.lane_id == ego_wp.lane_id

    def _cache_predicted_lane_entry(self, processed_targets):
        """Remember how soon each target is predicted to reach the ego lane."""
        if self.gap_recovery_s <= 0.0:
            return
        entries = {}
        for target in processed_targets:
            start_idx = getattr(target, "ego_lane_start_idx", None)
            if start_idx is None:
                branch = getattr(target, "branch_info", None) or {}
                start_idx = branch.get("ego_lane_start_idx")
            if not isinstance(start_idx, (int, float)):
                continue
            entries[getattr(target, "vehicle_id", None)] = float(start_idx) * self.DT
        self._predicted_lane_entry_s = entries

    def _gap_recovery_elapsed(self, target_relations, target_states=None, ego_state=None):
        """Seconds each vehicle has been treated as a lead, for the gap re-establishment.

        Also returns, per vehicle, the gap it had as a fraction of the full
        standoff when its clock started -- the relaxation starts from there,
        so a vehicle arriving outside the standoff is not relaxed at all.

        The clock starts when the vehicle is *about* to be in the way, not when it
        finishes merging: by the time the lane assignment flips, the ego has
        already done its braking and relaxing the standoff then changes nothing.
        Production ACC switches target on recognition for the same reason.

        Entries are dropped once a vehicle is neither in lane nor imminent, so one
        that merges, leaves and merges again starts its recovery over.
        """
        if self.gap_recovery_s <= 0.0:
            return None, None
        now = self._elapsed_seconds()
        tracked = {
            vehicle_id for vehicle_id, relation in target_relations.items()
            if relation == REL_EGO_LANE
        }
        for vehicle_id, entry_s in self._predicted_lane_entry_s.items():
            if entry_s <= self.GAP_RECOVERY_ONSET_TLC_S:
                tracked.add(vehicle_id)
        for vehicle_id in list(self._ego_lane_entry_time):
            if vehicle_id not in tracked:
                del self._ego_lane_entry_time[vehicle_id]
                self._gap_recovery_start_scale.pop(vehicle_id, None)
        elapsed = {}
        for vehicle_id in tracked:
            if vehicle_id not in self._ego_lane_entry_time:
                self._ego_lane_entry_time[vehicle_id] = now
                if target_states is not None and ego_state is not None and vehicle_id in target_states:
                    gap = float(target_states[vehicle_id][0]) - float(ego_state[0])
                    full = required_standoff(ego_state, self.controller_config)
                    self._gap_recovery_start_scale[vehicle_id] = float(
                        np.clip(gap / max(full, 1.0e-6), 0.0, 1.0))
            elapsed[vehicle_id] = max(0.0, now - self._ego_lane_entry_time[vehicle_id])
        return elapsed, dict(self._gap_recovery_start_scale)

    def _update_agent_history(self):
        if self.agent_history is None:
            return
        snapshot = self.world.get_snapshot()
        if snapshot is not None:
            self.agent_history.update(snapshot, self.world)

    def _stdan_trackings(self):
        if self.agent_history is None:
            return {}
        trackings = {}
        for actor in self.world.get_actors().filter("vehicle*"):
            actor_history = self.agent_history.vehicles.get(actor.id)
            if actor_history is None or not actor_history.time_history:
                continue
            times = np.asarray(actor_history.time_history, dtype=float)
            poses = np.asarray(actor_history.pose_history, dtype=np.float64)
            current_time = float(times[-1])
            history = []
            for hsec in self.stdan_history_secs:
                query_time = current_time - float(hsec)
                if query_time < times[0] - self.stdan_history_closeness_eps:
                    continue
                idx = int(np.argmin(np.abs(times - query_time)))
                if abs(float(times[idx]) - query_time) <= self.stdan_history_closeness_eps:
                    history.append(poses[idx])
            if history:
                trackings[actor.id] = np.asarray(history, dtype=np.float64)
        return trackings

    def _ego_state_rhs(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        x = transform.location.x
        y = -transform.location.y
        psi = -fth.fix_angle(np.radians(transform.rotation.yaw))
        speed = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2))
        return x, y, psi, speed

    def _selected_target_state(self, s_ego):
        candidates = []
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            role = actor.attributes.get("role_name", "")
            if "target" not in role:
                continue
            x, y, psi, speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv >= s_ego - 5.0:
                candidates.append((s_tv - s_ego, s_tv, d_tv, speed))

        if not candidates:
            return np.array([s_ego + 1000.0, 3.5, self.nominal_speed])

        _, s_tv, d_tv, speed = sorted(candidates, key=lambda item: item[0])[0]
        return np.array([s_tv, d_tv, speed], dtype=float)

    def _selected_ego_lane_lead_state(self, s_ego, speed):
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        candidates = []
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            x, y, psi, actor_speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv < s_ego - 1.0:
                continue
            target_wp = carla_map.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            relation = relation_from_waypoints(ego_wp, target_wp)
            if relation != REL_EGO_LANE and abs(d_tv) > self.EGO_LANE_HALF_WIDTH_M:
                continue
            role = actor.attributes.get("role_name", "")
            candidates.append((s_tv - s_ego, actor.id, role, s_tv, d_tv, actor_speed))

        if not candidates:
            return (
                np.array([s_ego + 1000.0, 0.0, speed], dtype=float),
                {
                    "selected_vehicle_id": None,
                    "selected_role": None,
                    "candidate_count": 0,
                    "source": "carla_waypoint_ego_lane",
                },
            )

        gap, actor_id, role, s_tv, d_tv, actor_speed = sorted(candidates, key=lambda item: item[0])[0]
        return (
            np.array([s_tv, d_tv, actor_speed], dtype=float),
            {
                "selected_vehicle_id": int(actor_id),
                "selected_role": role,
                "candidate_count": len(candidates),
                "gap": float(gap),
                "source": "carla_waypoint_ego_lane",
            },
        )

    @staticmethod
    def _parse_predictor_type(smpc_config):
        config = str(smpc_config)
        if "const_lead" in config or "constant_lead" in config:
            return "constant_velocity_lead"
        if "lstm" in config:
            return "lstm"
        if "stdan_vel" in config:
            return "stdan_vel"
        if "stdan_3int" in config:
            return "stdan_3int"
        return "synthetic"

    @staticmethod
    def _parse_best_mode_only(smpc_config):
        return "best_mode" in str(smpc_config)

    @staticmethod
    def _parse_controller_num_modes(smpc_config, requested_num_modes):
        config = str(smpc_config)
        if "best_mode" in config or "const_lead" in config or "constant_lead" in config or "lstm" in config:
            return 1
        match = re.search(r"(?:num_modes|modes|mode)[_-]?(\d+)", config)
        if match:
            return max(1, int(match.group(1)))
        return max(1, int(requested_num_modes))

    @staticmethod
    def _parse_variant(smpc_config):
        config = str(smpc_config).lower()
        if config in ("open_loop", "multimodal_ol", "acc_nair_smpc_ol") or "open_loop" in config or "multimodal_ol" in config or "_ol" in config:
            return VARIANT_MULTIMODAL_OL
        if "fixed_risk" in config:
            return VARIANT_FIXED_RISK
        return VARIANT_PROPOSED

    @staticmethod
    def _parse_safety_constraint_mode(smpc_config):
        config = str(smpc_config).lower()
        if re.search(r"cutin_chance[0-9]", config):
            return SAFETY_CONFIDENCE_CHANCE
        if (
                "brake_distance" in config
                or "braking_distance" in config
                or "d_brake" in config
                or "dbrake" in config
                or "collision_distance" in config):
            return SAFETY_BRAKE_DISTANCE
        if "scalar_chance" in config or "scalar_probability" in config or "optimized_eta" in config:
            return SAFETY_SCALAR_CHANCE
        return SAFETY_NOMINAL_SAFE_DISTANCE

    @staticmethod
    def _parse_brake_distance_bound_mode(smpc_config):
        config = str(smpc_config).lower()
        if (
                "brake_hard_band" in config
                or "brake_distance_hard_band" in config
                or "dbrake_hard_band" in config):
            return BRAKE_DISTANCE_BOUND_HARD_BAND
        return BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL

    @staticmethod
    def _parse_risk_allocation_mode(smpc_config):
        config = str(smpc_config)
        if "optimized_eta" in config:
            return RISK_OPTIMIZED_ETA
        if "probability_weighted" in config:
            return RISK_PROBABILITY_WEIGHTED
        return RISK_FIXED

    @staticmethod
    def _parse_optimize_k(smpc_config):
        config = str(smpc_config).lower()
        if (
                "no_k" in config
                or "zero_k" in config
                or "fixed_k" in config
                or "disable_k" in config):
            return False
        return (
            "with_k" in config
            or "optimize_k" in config
            or "enable_k" in config
            or "k_opt" in config
            or "opt_k" in config
            or "kopt" in config
        )

    @staticmethod
    def _parse_cutin_probability_threshold(smpc_config):
        """Read ``cutin_gate<value>`` from the config string (None when absent).

        Modes below this probability generate no collision-avoidance
        constraint, so the ego keeps speed instead of yielding to an unlikely
        merge.  Absent, STDAN policies use ``CUTIN_CHANCE_VANISH_BELOW``.  See
        ``gate_unlikely_cutin_modes`` and ``chance_cutout_clearance``.
        """
        match = re.search(r"cutin_gate([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_cutin_clearance_ramp_ref(smpc_config):
        """Read ``cutin_ramp<value>`` from the config string (default: disabled).

        The value is the reference confidence at which a cut-in mode keeps its
        present standoff; less likely modes shrink theirs continuously.  See
        ``cutin_clearance_scale``.
        """
        match = re.search(r"cutin_ramp([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_cutin_chance_ref(smpc_config):
        """Read ``cutin_chance<value>`` from the config string (default: disabled).

        The value is the reference confidence ``beta_ref`` of the Benciolini
        confidence chance constraint on cut-in modes; the token also switches
        the controller to ``confidence_chance`` safety constraints.  Modes below
        ``CUTIN_CHANCE_VANISH_BELOW`` vanish as for every STDAN policy.  See
        ``chance_cutin_clearance``.
        """
        match = re.search(r"cutin_chance([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_cutin_clearance_tlc_ref(smpc_config):
        """Read ``cutin_tlc<value>`` from the config string (default: disabled).

        The value is the time-to-lane-crossing at which a cut-in mode stops
        reserving standoff; it keeps the full standoff at zero.  See
        ``tlc_cutin_clearance``.
        """
        match = re.search(r"cutin_tlc([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_gap_recovery_s(smpc_config):
        """Read ``gap_recover<value>`` from the config string (default: disabled).

        The value is how long the standoff takes to return to its full time gap
        after a vehicle enters the ego lane.  See ``gap_reestablish_clearance``.
        """
        match = re.search(r"gap_recover([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_mode_probabilities(smpc_config):
        if "cutin_high" in str(smpc_config):
            return [0.3, 0.7]
        return [0.6, 0.4]

    # Half a lane: a vehicle whose centre is within it is in the ego lane even
    # when the waypoint test fails because the two vehicles sit on different
    # road segments (Town04 road 47 -> 1073), which otherwise drops an in-lane
    # lead for a second or two.
    EGO_LANE_HALF_WIDTH_M = 1.75

    @classmethod
    def _relation_from_frenet_offset(cls, d_tv):
        if d_tv > cls.EGO_LANE_HALF_WIDTH_M:
            return REL_LEFT_ADJACENT
        if d_tv < -cls.EGO_LANE_HALF_WIDTH_M:
            return REL_RIGHT_ADJACENT
        return REL_EGO_LANE

    @staticmethod
    def _lane_keeping_steer(ey, epsi):
        return float(np.clip(-0.15 * ey - 0.8 * epsi, -0.45, 0.45))

    def _min_gap(self, solution, prediction):
        min_gap = np.inf
        linearization_speed = float(solution.x_nominal[0, 0, 1])
        for mode in range(prediction.num_modes):
            for step in range(prediction.horizon + 1):
                if not prediction.active_mask[mode, step]:
                    continue
                min_gap = min(
                    min_gap,
                    safety_function(
                        solution.x_nominal[mode, step],
                        prediction.means[mode, step],
                        self.controller_config,
                        linearization_speed=linearization_speed,
                    ),
                )
        return float(min_gap) if np.isfinite(min_gap) else np.nan

    def _actor_state_rhs(self, actor):
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        x = transform.location.x
        y = -transform.location.y
        psi = -fth.fix_angle(np.radians(transform.rotation.yaw))
        speed = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2))
        return x, y, psi, speed

    def _command_dt(self, time_s):
        fallback_dt = max(float(self.DT), 1.0e-6)
        if time_s is None or self._prev_accel_cmd_time_s is None:
            return fallback_dt
        dt = float(time_s) - float(self._prev_accel_cmd_time_s)
        if not np.isfinite(dt) or dt <= 1.0e-6:
            return fallback_dt
        return dt

    def _limit_accel_command(self, action, previous_accel_cmd, command_dt):
        action = float(np.clip(
            action,
            self.controller_config.a_min,
            self.controller_config.a_max,
        ))
        jerk_limit = self.controller_config.jerk_limit
        if jerk_limit is None:
            return action
        max_delta = float(jerk_limit) * max(float(command_dt), 1.0e-6)
        limited = np.clip(
            action,
            float(previous_accel_cmd) - max_delta,
            float(previous_accel_cmd) + max_delta,
        )
        return float(np.clip(
            limited,
            self.controller_config.a_min,
            self.controller_config.a_max,
        ))

    def _command_jerk(self, action, command_dt):
        if self._prev_accel_cmd_time_s is None:
            return None
        dt = max(float(command_dt), 1.0e-6)
        return float((float(action) - float(self._prev_accel_cmd)) / dt)

    def _remember_accel_cmd(self, time_s, action):
        self._prev_accel_cmd = float(action)
        self._prev_accel_cmd_time_s = None if time_s is None else float(time_s)

    def _actual_longitudinal_accel(self):
        _, _, psi, _ = self._ego_state_rhs()
        accel = self.vehicle.get_acceleration()
        accel_rhs = np.array([float(accel.x), -float(accel.y)], dtype=float)
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        return float(np.dot(accel_rhs, heading_rhs))

    def _actual_longitudinal_jerk(self, time_s, actual_accel):
        if time_s is None or actual_accel is None:
            return None
        time_s = float(time_s)
        actual_accel = float(actual_accel)
        previous = self._prev_actual_accel_sample
        self._prev_actual_accel_sample = (time_s, actual_accel)
        if previous is None:
            return None
        prev_time, prev_accel = previous
        dt = time_s - prev_time
        if dt <= 1.0e-6:
            return None
        return float((actual_accel - prev_accel) / dt)

    def _elapsed_seconds(self):
        snapshot = self.world.get_snapshot()
        if snapshot is None or snapshot.timestamp is None:
            return None
        return snapshot.timestamp.elapsed_seconds

    def _log_wandb(self, row, solution, prediction_bundle):
        if not self.enable_wandb_logging:
            return
        try:
            import wandb
            if wandb.run is None:
                wandb.init(project="carla", name=f"acc_nair_{self.predictor_type}", reinit=True)
            payload = {
                "acc/time_s": row["time_s"],
                "acc/ego_s": row["ego_s"],
                "acc/ego_v": row["ego_v"],
                "acc/accel_cmd": row["accel_cmd"],
                "acc/accel_cmd_raw": row.get("accel_cmd_raw"),
                "acc/command_jerk": row.get("command_jerk"),
                "acc/command_jerk_limit": row.get("command_jerk_limit"),
                "acc/min_gap": row["min_gap"],
                "acc/feasible": float(row["feasible"]),
                "acc/solve_time": row["solve_time"],
                "acc/chance_margin_min": solution.chance_margin_min,
                "acc/first_policy_split_step": (
                    -1 if solution.first_policy_split_step is None
                    else int(solution.first_policy_split_step)
                ),
                "acc/policy_branch_step": (
                    -1 if solution.policy_branch_step is None
                    else int(solution.policy_branch_step)
                ),
            }
            for target in prediction_bundle.get("processed_targets", []):
                prefix = f"stdan/{target.vehicle_id}"
                for key, value in target.raw_mode_prob.items():
                    payload[f"{prefix}/raw_{key}"] = value
                for key, value in target.acc_mode_prob.items():
                    payload[f"{prefix}/acc_{key}"] = value
                payload[f"{prefix}/ego_lane_membership_split_step"] = target.branch_info.get(
                    "ego_lane_membership_split_step",
                    target.branch_info["branch_step"],
                )
                payload[f"{prefix}/branch_step"] = target.branch_info["branch_step"]
                payload[f"{prefix}/ego_lane_start_idx"] = target.branch_info["ego_lane_start_idx"]
                payload[f"{prefix}/ego_lane_end_idx"] = target.branch_info["ego_lane_end_idx"]
                if target.raw_prediction is not None:
                    payload[f"{prefix}/signed_t_cross"] = target.raw_prediction.get("signed_t_cross", 0.0)
            wandb.log(payload)
        except Exception:
            return
