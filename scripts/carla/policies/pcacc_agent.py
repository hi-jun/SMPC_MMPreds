"""CARLA ego policy ``acc_pcacc``: the provably-correct and comfortable ACC of
Althoff, Maierhofer & Pek (IEEE T-IV 2021), a rule-based baseline without a
predictor.  The controller itself is ``utils/pcacc.py``; this file is the
simulator side: perception from measured states, lane membership, steering,
the ideal actuator and the ``policy_log`` the aggregator reads.

Perception (Sec. II-C, III-C of the paper), per tick and per other vehicle:

* ``s_lat`` and ``theta`` are the vehicle's lateral and heading deviation from
  the centre of *its own* lane, from ``carla_map.get_waypoint(location,
  project_to_road=True, lane_type=Driving)``; the same-lane and adjacent-lane
  relations come from the waypoint lane ids (``get_left_lane`` /
  ``get_right_lane`` of the ego waypoint), never from a fixed +/-3.5 m offset.
* The occupancy of the vehicle intersects the ego lane when its waypoint is the
  ego lane, or when it is in an adjacent lane and its body reaches over the
  lane line (lateral deviation toward the ego lane > lane_width / 2 - half the
  vehicle width).
* The cut-in rule (``s_lat`` > 0.4 m and ``theta`` > 0.035 rad toward the ego
  lane for 0.5 s) and the 1.0 s leave lag are in ``LaneMembershipTracker``.
* The gap is centre to centre minus L = 4.5 m, the bumper gap the aggregator
  scores delta with (so delta = 0 means the fork's standoff is held exactly);
  with ``paperdsafe`` it is rear bumper of the vehicle minus front bumper of
  the ego (``bounding_box.extent.x``), the paper's s_p - s.

Safe distance and limits: this fork's by default (``utils/pcacc.py``): the
standoff d0 + tau v = 3 + 1.3 v on that gap, a in [-3, 2] m/s^2, |j| <= 5
m/s^3; the lead's a_p_min = -10.5 of Table I stays.  ``paperdsafe`` restores
the paper's braking-based d_safe and ``amin10_amax3_jlim10`` Table I's
limits; ``amin<x>``, ``amax<x>``, ``jlim<x>``, ``apmin<x>``, ``acutin<x>``,
``tc<x>`` in the policy string override the numbers.
The issued command is clipped to the acceleration bounds and, except on the
acceleration-bounded path, to the jerk bound.  The ego's acceleration state of
the paper's model is the last issued command (the ideal actuator realises it).
``N`` / ``N_modes`` of the SMPC interface are ignored: the horizons are the
paper's h_k = 6 s and t_c.
"""
import os
import sys
import time

import carla
import numpy as np

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent
from utils import frenet_trajectory_handler as fth
from utils.low_level_control import (
    IdealLongitudinalActuator, LowLevelControl, lane_keeping_steer)
from utils.pcacc import (
    LaneMembershipTracker,
    PCACCConfig,
    PCACCController,
    VehicleObs,
)


class PCACCAgent(object):
    """Same interface as ``ACCNairSMPCAgent``: ``run_step(pred_dict)`` returns
    ``(control, z0, u0, is_feasible, solve_time)``; ``get_cut_in_log()`` is the
    ``policy_log``."""

    # Same-lane fallback across a road-segment boundary (the lane ids match but
    # the road ids do not): the aggregator's LANE_HALF_WIDTH.
    LANE_HALF_WIDTH_M = 1.75

    def __init__(
            self,
            vehicle,
            goal_location,
            nominal_speed_mps=8.0,
            dt=0.2,
            N=8,
            N_modes=2,
            smpc_config="acc_pcacc"):
        del N, N_modes
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.DT = float(dt)
        self.nominal_speed = float(nominal_speed_mps)
        self.policy_string = str(smpc_config)
        self.config = PCACCConfig.from_policy_string(self.policy_string)
        self.controller = PCACCController(self.config)
        self.tracker = LaneMembershipTracker(self.config)
        self.predictor_type = "none"
        self.goal_location = goal_location
        self.goal_reached = False
        self._frenet_traj = None
        self._ego_route_debug = {"source": "uninitialized"}
        self._low_level_control = LowLevelControl(vehicle)
        # The scenario runner points ``command_batch`` at its per-tick list.
        self._ideal_actuator = IdealLongitudinalActuator(vehicle)
        self._prev_accel_cmd = 0.0
        self._prev_accel_cmd_time_s = None
        self._prev_actual_accel_sample = None
        self._clock = 0.0
        self.policy_log = []

    def done(self):
        return self.goal_reached

    # ------------------------------------------------------------------ frames
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
        way_s, way_xy, way_yaw, self._ego_route_debug = ACCNairSMPCAgent._build_current_lane_reference(
            init_waypoint,
            self.goal_location,
            vehicle_transform.location,
            vehicle_transform.rotation.yaw,
        )
        self._frenet_traj = fth.FrenetTrajectoryHandler(way_s, way_xy, way_yaw, s_resolution=0.5)

    @staticmethod
    def _actor_state_rhs(actor):
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        x = transform.location.x
        y = -transform.location.y
        psi = -fth.fix_angle(np.radians(transform.rotation.yaw))
        speed = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2))
        return x, y, psi, speed

    @staticmethod
    def _own_lane_deviation(waypoint, x, y, psi):
        """Lateral (+left) and heading deviation of an RHS pose from its lane centre."""
        loc = waypoint.transform.location
        psi_wp = -fth.fix_angle(np.radians(waypoint.transform.rotation.yaw))
        dx = x - loc.x
        dy = y + loc.y
        s_lat = -np.sin(psi_wp) * dx + np.cos(psi_wp) * dy
        theta = fth.fix_angle(psi - psi_wp)
        return float(s_lat), float(theta)

    def _observe(self, time_s, s_ego, ego_waypoint):
        """Preceding vehicles as ``VehicleObs`` plus a per-vehicle log row."""
        carla_map = self.world.get_map()
        cfg = self.config
        ego_half_len = float(self.vehicle.bounding_box.extent.x)
        neighbours = {}
        for side, lane in (("left", ego_waypoint.get_left_lane()), ("right", ego_waypoint.get_right_lane())):
            if lane is not None and lane.lane_id * ego_waypoint.lane_id > 0:
                neighbours[lane.lane_id] = side
        vehicles, rows = [], []
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            x, y, psi, speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv <= s_ego:
                continue
            if cfg.paper_d_safe:
                gap = (s_tv - float(actor.bounding_box.extent.x)) - (s_ego + ego_half_len)
            else:
                gap = s_tv - s_ego - cfg.vehicle_length
            if gap > cfg.sensor_range_m:
                continue
            waypoint = carla_map.get_waypoint(
                actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            if waypoint is None:
                continue
            s_lat, theta = self._own_lane_deviation(waypoint, x, y, psi)
            same_lane = waypoint.lane_id == ego_waypoint.lane_id and (
                waypoint.road_id == ego_waypoint.road_id or abs(d_tv) <= self.LANE_HALF_WIDTH_M)
            side = neighbours.get(waypoint.lane_id) if not same_lane else None
            adjacent = side is not None
            # toward the ego lane: a vehicle on the ego's left reaches it by moving right
            sign = -1.0 if side == "left" else 1.0
            s_lat_toward = sign * s_lat if adjacent else None
            theta_toward = sign * theta if adjacent else None
            occupies = same_lane or (
                adjacent and s_lat_toward > 0.5 * float(waypoint.lane_width) - float(actor.bounding_box.extent.y))
            state = self.tracker.update(actor.id, time_s, occupies, adjacent, s_lat_toward, theta_toward)
            role = actor.attributes.get("role_name", "")
            vehicles.append(VehicleObs(int(actor.id), float(gap), float(speed), bool(state.member), role))
            rows.append({
                "vid": int(actor.id),
                "role": role,
                "s": float(s_tv),
                "d": float(d_tv),
                "gap": float(gap),
                "v": float(speed),
                "lane_id": int(waypoint.lane_id),
                "road_id": int(waypoint.road_id),
                "s_lat": s_lat,
                "theta": theta,
                "side": side,
                "same_lane": bool(same_lane),
                "occupies": bool(occupies),
                "predicted": bool(state.predicted),
                "member": bool(state.member),
                "rule_ok_since": state.rule_ok_since,
            })
        return vehicles, rows

    # ------------------------------------------------------------------ step
    def run_step(self, pred_dict):
        del pred_dict
        cfg = self.config
        x, y, psi, speed = self._actor_state_rhs(self.vehicle)
        self._ensure_frenet_reference_initialized()
        s_ego, ey, epsi = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
        z0 = np.array([x, y, psi, speed])
        time_s = self._elapsed_seconds()
        previous_accel_cmd = self._prev_accel_cmd
        command_dt = self._command_dt(time_s)
        # The controller's clocks (rule window, leave lag, clearing time) need a
        # time even when the snapshot has none.
        self._clock = float(time_s) if time_s is not None else self._clock + command_dt
        solve_time = np.nan
        is_feasible = False

        if self.goal_reached or self._frenet_traj.reached_trajectory_end(s_ego, resolution=5.0):
            self.goal_reached = True
            action = self._limit_accel_command(cfg.a_min, previous_accel_cmd, command_dt, True)
            df_des = 0.0
        else:
            ego_waypoint = self.world.get_map().get_waypoint(
                self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            start = time.time()
            vehicles, rows = self._observe(self._clock, s_ego, ego_waypoint)
            decision = self.controller.step(
                self._clock, speed, previous_accel_cmd, self.nominal_speed, vehicles, command_dt)
            solve_time = time.time() - start
            action = self._limit_accel_command(
                decision.accel, previous_accel_cmd, command_dt, decision.jerk_limited)
            df_des = lane_keeping_steer(ey, epsi)
            is_feasible = bool(decision.feasible)
            for row in rows:
                dec = decision.per_vehicle.get(row["vid"])
                row["path"] = None if dec is None else dec.path
                row["accel"] = None if dec is None else float(dec.accel)
                row["d_safe"] = None if dec is None else float(dec.d_safe)
                row["extra"] = {} if dec is None else {k: float(v) for k, v in dec.extra.items()}
            lead = decision.per_vehicle.get(decision.lead_vid)
            lead_row = next((row for row in rows if row["vid"] == decision.lead_vid), None)
            margins = [dec.gap - dec.d_safe for dec in decision.per_vehicle.values()]
            actual_accel = self._actual_longitudinal_accel(psi)
            self.policy_log.append({
                "time_s": time_s,
                "status": decision.path,
                "feasible": is_feasible,
                "solve_time": float(solve_time),
                "ego_s": float(s_ego),
                "ego_v": float(speed),
                "ego_a_state": float(previous_accel_cmd),
                "accel_cmd_raw": float(decision.accel),
                "accel_cmd": float(action),
                "command_dt": float(command_dt),
                "command_jerk": self._command_jerk(action, command_dt),
                "command_jerk_limit": float(cfg.jerk_max),
                "actual_accel": actual_accel,
                "actual_jerk": self._actual_longitudinal_jerk(time_s, actual_accel),
                "steer_cmd": float(df_des),
                "solve_path": decision.path,
                "predictor_type": self.predictor_type,
                "ego_route_debug": self._ego_route_debug,
                "min_gap": float(min(margins)) if margins else np.nan,
                "target_s": None if lead_row is None else float(lead_row["s"]),
                "target_d": None if lead_row is None else float(lead_row["d"]),
                "target_v": None if lead is None else float(lead.speed),
                "pcacc": {
                    "lead_vid": decision.lead_vid,
                    "lead_role": None if lead_row is None else lead_row["role"],
                    "lead_gap": None if lead is None else float(lead.gap),
                    "lead_d_safe": None if lead is None else float(lead.d_safe),
                    "a_cruise": float(decision.a_cruise),
                    "keep_vids": list(decision.keep_vids),
                    "recap_vids": list(decision.recap_vids),
                    "vehicles": rows,
                },
            })

        self._remember_accel_cmd(time_s, action)
        v_des = float(np.clip(speed + action * command_dt, cfg.v_min, cfg.v_max))
        control = self._low_level_control.update(speed, action, v_des, df_des)
        # Steering only: the ideal actuator pins the speed to the command.
        control.throttle = 0.0
        control.brake = 0.0
        self._ideal_actuator.update(action)
        u0 = np.array([action, df_des])
        return control, z0, u0, is_feasible, solve_time

    def get_cut_in_log(self):
        return {
            "controller": "acc_pcacc",
            "policy": self.policy_string,
            "config": self.config.as_dict(),
            "steps": self.policy_log,
        }

    # ------------------------------------------------------------------ helpers
    def _command_dt(self, time_s):
        settings_dt = None
        try:
            settings_dt = self.world.get_settings().fixed_delta_seconds
        except Exception:
            pass
        fallback_dt = float(settings_dt) if settings_dt else 0.05
        if time_s is None or self._prev_accel_cmd_time_s is None:
            return fallback_dt
        dt = float(time_s) - float(self._prev_accel_cmd_time_s)
        if not np.isfinite(dt) or dt <= 1.0e-6:
            return fallback_dt
        return dt

    def _limit_accel_command(self, action, previous_accel_cmd, command_dt, jerk_limited):
        cfg = self.config
        action = float(np.clip(action, cfg.a_min, cfg.a_max))
        if not jerk_limited:
            return action
        max_delta = float(cfg.jerk_max) * max(float(command_dt), 1.0e-6)
        limited = np.clip(action, float(previous_accel_cmd) - max_delta, float(previous_accel_cmd) + max_delta)
        return float(np.clip(limited, cfg.a_min, cfg.a_max))

    def _command_jerk(self, action, command_dt):
        if self._prev_accel_cmd_time_s is None:
            return None
        return float((float(action) - float(self._prev_accel_cmd)) / max(float(command_dt), 1.0e-6))

    def _remember_accel_cmd(self, time_s, action):
        self._prev_accel_cmd = float(action)
        self._prev_accel_cmd_time_s = None if time_s is None else float(time_s)

    def _actual_longitudinal_accel(self, psi):
        accel = self.vehicle.get_acceleration()
        accel_rhs = np.array([float(accel.x), -float(accel.y)], dtype=float)
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        return float(np.dot(accel_rhs, heading_rhs))

    def _actual_longitudinal_jerk(self, time_s, actual_accel):
        if time_s is None or actual_accel is None:
            return None
        previous = self._prev_actual_accel_sample
        self._prev_actual_accel_sample = (float(time_s), float(actual_accel))
        if previous is None:
            return None
        prev_time, prev_accel = previous
        dt = float(time_s) - prev_time
        if dt <= 1.0e-6:
            return None
        return float((float(actual_accel) - prev_accel) / dt)

    def _elapsed_seconds(self):
        snapshot = self.world.get_snapshot()
        if snapshot is None or snapshot.timestamp is None:
            return None
        return snapshot.timestamp.elapsed_seconds
