import math

import carla
import numpy as np

from policies.mpc_agent import MPCAgent
from utils import frenet_trajectory_handler as fth


MIN_CURRENT_LANE_ROUTE_DISTANCE = 500.0
MIN_POST_LANE_CHANGE_ROUTE_DISTANCE = 500.0
SPEED_CAP_VEHICLE_LENGTH_M = 4.5
# The cap starts biting at the centre-to-centre gap where v_cap reaches the
# nominal speed:  4.5 + MIN_GAP + (nominal - lead_speed) / GAIN.  For the cut-in
# scenarios (15 m/s over a 9 m/s lead) that is 12.0 m, below both lead-gap
# triggers (17 m normal, 13 m aggressive), so the cap no longer decelerates the
# vehicle before it starts its lane change.  Keep this engagement gap below the
# smallest trigger distance in use; a larger nominal-to-lead speed difference
# raises it.
SPEED_CAP_MIN_LONGITUDINAL_GAP_M = 2.5
SPEED_CAP_GAIN = 1.2
SPEED_CAP_LATERAL_THRESHOLD_M = 2.5
SPEED_CAP_YAW_THRESHOLD_DEG = 45.0


class DistanceTriggeredLaneChangeAgent(MPCAgent):
    """MPC target policy that starts a lane change when ego is close enough."""

    def __init__(
            self,
            vehicle,
            goal_location,
            nominal_speed_mps=8.0,
            dt=0.2,
            N=8,
            N_modes=3,
            trigger_distance_m=30.0,
            distance_same_lane=5.0,
            distance_other_lane=100.0,
            distance_lane_change=25.0,
            trigger_mode="ego_gap"):
        super().__init__(
            vehicle,
            goal_location,
            nominal_speed_mps=nominal_speed_mps,
            dt=dt,
            N=N,
            N_modes=N_modes)

        self.trigger_distance_m = trigger_distance_m
        # "ego_gap": start when the target-lane follower (ego) is within
        # trigger_distance_m behind.  "lead_gap": start when this vehicle is
        # within trigger_distance_m of the lead in its own lane -- the natural
        # cut-in motive, and independent of what the ego does, so every ego
        # policy meets the same manoeuvre.
        if trigger_mode not in ("ego_gap", "lead_gap"):
            raise ValueError(f"unknown trigger_mode {trigger_mode!r}")
        self.trigger_mode = trigger_mode
        self.ego_gap_at_start = None
        self.distance_same_lane = distance_same_lane
        self.distance_other_lane = distance_other_lane
        self.distance_lane_change = distance_lane_change

        self.lane_change_started = False
        self.lane_change_completed = False
        # No lane change while the ego is alongside: whenever the longitudinal
        # bumper-to-bumper distance to the ego is within one vehicle width the
        # change must not start, and one already under way turns back unless
        # the vehicle has reached the ego lane.
        self.ego_proximity_guard_m = None
        self.ego_proximity_guard_events = []
        self._guard_hold_logged = False
        self.lane_change_direction = None
        self.trigger_time_s = None
        self.trigger_distance_at_start = None
        self.ego_lane_id = None
        self._ego_lane_key = None
        self._ego_lane_yaw = None
        self.target_initial_lane_id = self._get_actor_lane_id(self.vehicle)
        self.target_final_lane_id = None
        self._switch_to_current_lane_route()

    def run_step(self, pred_dict):
        # Refresh the lane assignment first so the guard below never turns
        # back a vehicle that already reached the ego lane.
        self._update_lane_change_completion()
        ego_actor = self._find_ego_actor()
        ego_alongside, guard_detail = self._ego_within_guard(ego_actor)
        if not self.lane_change_started:
            ego_gap = self._longitudinal_trigger_distance(self.vehicle, ego_actor)
            if self.trigger_mode == "lead_gap":
                lead = self._find_same_lane_front_lead()
                trigger_distance = None if lead is None else float(lead[1])
            else:
                trigger_distance = ego_gap
            if trigger_distance is not None and trigger_distance <= self.trigger_distance_m:
                if ego_alongside:
                    self._record_guard_event("hold", guard_detail)
                else:
                    self._guard_hold_logged = False
                    self.trigger_distance_at_start = trigger_distance
                    self.ego_gap_at_start = ego_gap
                    self.trigger_time_s = self._get_elapsed_seconds()
                    self._switch_to_lane_change_route(ego_actor)
                    self.lane_change_started = True
        elif not self.lane_change_completed and ego_alongside:
            self._switch_to_current_lane_route()
            self.lane_change_started = False
            self.lane_change_direction = None
            self._guard_hold_logged = False
            self._record_guard_event("abort", guard_detail)

        result = super().run_step(pred_dict)
        self._update_lane_change_completion()
        return result

    def get_cut_in_log(self):
        return {
            "trigger_time_s": self.trigger_time_s,
            "trigger_mode": self.trigger_mode,
            "trigger_distance_at_start": self.trigger_distance_at_start,
            "ego_gap_at_start": self.ego_gap_at_start,
            "ego_lane_id": self.ego_lane_id,
            "target_initial_lane_id": self.target_initial_lane_id,
            "target_final_lane_id": self.target_final_lane_id,
            "lane_change_started": self.lane_change_started,
            "lane_change_completed": self.lane_change_completed,
            "lane_change_direction": self.lane_change_direction,
            "ego_proximity_guard_m": self.ego_proximity_guard_m,
            "ego_proximity_guard_events": self.ego_proximity_guard_events,
        }

    def _get_reference_traj(self, x0, y0, psi0, v0):
        ref_dict = super()._get_reference_traj(x0, y0, psi0, v0)
        lead = self._find_same_lane_front_lead()
        if lead is None:
            return ref_dict

        _, actual_gap, lead_speed = lead
        desired_gap = SPEED_CAP_VEHICLE_LENGTH_M + SPEED_CAP_MIN_LONGITUDINAL_GAP_M
        v_cap = lead_speed + SPEED_CAP_GAIN * (actual_gap - desired_gap)
        v_cap = min(self.nominal_speed, max(self.V_MIN, v_cap))
        v_ref = np.minimum(np.asarray(ref_dict["v_ref"], dtype=float), v_cap)
        if np.allclose(v_ref, ref_dict["v_ref"]):
            return ref_dict

        s0, _, _ = self._frenet_traj.convert_global_to_frenet_frame(x0, y0, psi0)
        s_ref = s0 + np.cumsum(v_ref) * self.DT
        traj = self._frenet_traj.trajectory
        s_ref = np.clip(s_ref, traj[0, 0], traj[-1, 0])

        ref_dict["x_ref"] = np.interp(s_ref, traj[:, 0], traj[:, 1])
        ref_dict["y_ref"] = np.interp(s_ref, traj[:, 0], traj[:, 2])
        psi_ref = np.interp(s_ref, traj[:, 0], np.unwrap(traj[:, 3]))
        ref_dict["psi_ref"] = fth.fix_angle(psi_ref - psi0) + psi0
        ref_dict["v_ref"] = v_ref
        return ref_dict

    def _ego_within_guard(self, ego_actor):
        """Is the ego longitudinally within one vehicle width, bumper to bumper?

        The gap is centre-to-centre along the ego heading minus both half
        lengths, taken as a magnitude so a just-passed ego still blocks; the
        guard width is the wider of the two vehicles.
        """
        if ego_actor is None:
            return False, None
        ego_transform = ego_actor.get_transform()
        ego_location = ego_transform.location
        location = self.vehicle.get_location()
        yaw = math.radians(ego_transform.rotation.yaw)
        centre_gap = ((location.x - ego_location.x) * math.cos(yaw)
                      + (location.y - ego_location.y) * math.sin(yaw))
        half_lengths = float(self.vehicle.bounding_box.extent.x
                             + ego_actor.bounding_box.extent.x)
        guard = 2.0 * float(max(self.vehicle.bounding_box.extent.y,
                                ego_actor.bounding_box.extent.y))
        self.ego_proximity_guard_m = guard
        bumper_gap = abs(centre_gap) - half_lengths
        detail = {
            "ego_centre_gap_m": float(centre_gap),
            "ego_bumper_gap_m": float(bumper_gap),
        }
        return bumper_gap <= guard, detail

    def _record_guard_event(self, kind, detail):
        if kind == "hold":
            if self._guard_hold_logged:
                return
            self._guard_hold_logged = True
        event = {"event": kind, "time_s": self._get_elapsed_seconds()}
        if detail:
            event.update(detail)
        self.ego_proximity_guard_events.append(event)

    @staticmethod
    def _longitudinal_trigger_distance(actor, reference_actor):
        if actor is None or reference_actor is None:
            return None

        actor_location = actor.get_location()
        reference_location = reference_actor.get_location()
        if actor_location is None or reference_location is None:
            return None

        reference_yaw = math.radians(reference_actor.get_transform().rotation.yaw)
        dx = actor_location.x - reference_location.x
        dy = actor_location.y - reference_location.y
        longitudinal_gap = dx * math.cos(reference_yaw) + dy * math.sin(reference_yaw)
        if longitudinal_gap < 0.0:
            return None
        return float(longitudinal_gap)

    def _find_ego_actor(self):
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            if actor.attributes.get("role_name") == "ego":
                return actor
        return None

    def _find_same_lane_front_lead(self):
        carla_map = self.world.get_map()
        target_location = self.vehicle.get_location()
        target_transform = self.vehicle.get_transform()
        target_waypoint = carla_map.get_waypoint(
            target_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if target_waypoint is None:
            return None

        target_yaw = math.radians(target_transform.rotation.yaw)
        forward_x = math.cos(target_yaw)
        forward_y = math.sin(target_yaw)
        best = None
        # While the lane change is under way the vehicle is committed to the
        # target lane: it must not keep pacing the lead it is leaving behind
        # (it would brake to that lead's speed mid-manoeuvre), and it may still
        # be lateral of both lane centres, so the lead search moves to the
        # target lane and drops the lateral-offset test.
        # Once started, the search stays on the target lane for good: the
        # waypoint lookup flickers between lanes around the boundary, and a
        # single tick back on the old lane would re-cap on the lead being passed.
        target_lane_id = getattr(self, "ego_lane_id", None) if self.lane_change_started else None
        lane_id_ref = target_waypoint.lane_id if target_lane_id is None else target_lane_id
        changing = target_lane_id is not None and not self.lane_change_completed

        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue

            actor_location = actor.get_location()
            actor_waypoint = carla_map.get_waypoint(
                actor_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if not self._is_same_lane_front_candidate(
                    target_waypoint, actor_waypoint,
                    lane_id=lane_id_ref, check_lateral=not changing):
                continue

            dx = actor_location.x - target_location.x
            dy = actor_location.y - target_location.y
            longitudinal_gap = dx * forward_x + dy * forward_y
            if longitudinal_gap <= 0.0:
                continue

            actor_velocity = actor.get_velocity()
            lead_speed = actor_velocity.x * forward_x + actor_velocity.y * forward_y
            lead_speed = max(0.0, lead_speed)

            if best is None or longitudinal_gap < best[1]:
                best = (actor, float(longitudinal_gap), float(lead_speed))

        return best

    @staticmethod
    def _is_same_lane_front_candidate(target_waypoint, actor_waypoint,
                                      lane_id=None, check_lateral=True):
        if target_waypoint is None or actor_waypoint is None:
            return False
        reference_lane_id = target_waypoint.lane_id if lane_id is None else lane_id
        if reference_lane_id != actor_waypoint.lane_id:
            return False

        yaw_diff = DistanceTriggeredLaneChangeAgent._angle_diff_deg(
            target_waypoint.transform.rotation.yaw,
            actor_waypoint.transform.rotation.yaw)
        if abs(yaw_diff) > SPEED_CAP_YAW_THRESHOLD_DEG:
            return False

        if not check_lateral:
            return True
        lateral_offset = DistanceTriggeredLaneChangeAgent._lateral_offset_to_waypoint(
            actor_waypoint, target_waypoint)
        return abs(lateral_offset) <= SPEED_CAP_LATERAL_THRESHOLD_M

    def _switch_to_lane_change_route(self, ego_actor):
        plan = self._generate_lane_change_plan(ego_actor)
        way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(plan)
        self._frenet_traj = fth.FrenetTrajectoryHandler(
            way_s, way_xy, way_yaw, s_resolution=0.5)
        self._fit_velocity_profile()
        self.goal_reached = False
        self.warm_start = None

    def _switch_to_current_lane_route(self):
        plan = self._generate_current_lane_plan()
        way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(plan)
        self._frenet_traj = fth.FrenetTrajectoryHandler(
            way_s, way_xy, way_yaw, s_resolution=0.5)
        self._fit_velocity_profile()
        self.goal_reached = False
        self.warm_start = None

    def _generate_current_lane_plan(self):
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Could not find a driving waypoint for pre-trigger target.")

        plan = [(waypoint, None)]
        step_distance = 2.0
        keep_distance = max(
            self.distance_same_lane + self.distance_lane_change + self.distance_other_lane,
            MIN_CURRENT_LANE_ROUTE_DISTANCE,
        )
        self._append_forward_waypoints(plan, keep_distance, step_distance)
        return plan

    def _generate_lane_change_plan(self, ego_actor):
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        ego_waypoint = carla_map.get_waypoint(
            ego_actor.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Could not find a driving waypoint for lane-change actor.")
        if ego_waypoint is None:
            raise RuntimeError("Could not find a driving waypoint for ego actor.")

        plan = [(waypoint, None)]
        step_distance = 2.0

        self._append_forward_waypoints(plan, self.distance_same_lane, step_distance)

        self.ego_lane_id = ego_waypoint.lane_id
        self._ego_lane_key = self._lane_key(ego_waypoint)
        self._ego_lane_yaw = ego_waypoint.transform.rotation.yaw
        self.lane_change_direction, target_lane_waypoint = self._choose_ego_lane_change(
            plan[-1][0], ego_waypoint)

        if target_lane_waypoint is None or target_lane_waypoint.lane_type != carla.LaneType.Driving:
            raise RuntimeError("Ego lane is not an adjacent driving lane for cut-in target.")

        target_lane_next = target_lane_waypoint.next(self.distance_lane_change)
        if not target_lane_next:
            raise RuntimeError("Could not extend lane-change route on the target lane.")
        plan.append((target_lane_next[0], None))

        self._append_forward_waypoints(
            plan,
            max(self.distance_other_lane, MIN_POST_LANE_CHANGE_ROUTE_DISTANCE),
            step_distance,
        )
        return plan

    @staticmethod
    def _append_forward_waypoints(plan, target_distance, step_distance):
        distance = 0.0
        while distance < target_distance:
            next_waypoints = plan[-1][0].next(step_distance)
            if not next_waypoints:
                break
            next_waypoint = next_waypoints[0]
            distance += next_waypoint.transform.location.distance(
                plan[-1][0].transform.location)
            plan.append((next_waypoint, None))

    @staticmethod
    def _choose_ego_lane_change(target_waypoint, ego_waypoint):
        left_waypoint = target_waypoint.get_left_lane()
        if DistanceTriggeredLaneChangeAgent._same_logical_lane(left_waypoint, ego_waypoint):
            return "left", left_waypoint

        right_waypoint = target_waypoint.get_right_lane()
        if DistanceTriggeredLaneChangeAgent._same_logical_lane(right_waypoint, ego_waypoint):
            return "right", right_waypoint

        raise RuntimeError(
            "Ego lane is not adjacent to the target vehicle lane. "
            f"target={DistanceTriggeredLaneChangeAgent._waypoint_debug(target_waypoint)}, "
            f"ego={DistanceTriggeredLaneChangeAgent._waypoint_debug(ego_waypoint)}, "
            f"left={DistanceTriggeredLaneChangeAgent._waypoint_debug(left_waypoint)}, "
            f"right={DistanceTriggeredLaneChangeAgent._waypoint_debug(right_waypoint)}")

    @staticmethod
    def _same_logical_lane(waypoint, other_waypoint):
        lane_key = DistanceTriggeredLaneChangeAgent._lane_key(waypoint)
        other_lane_key = DistanceTriggeredLaneChangeAgent._lane_key(other_waypoint)
        if lane_key is None or other_lane_key is None:
            return False
        if lane_key == other_lane_key:
            return True

        yaw_diff = DistanceTriggeredLaneChangeAgent._angle_diff_deg(
            waypoint.transform.rotation.yaw,
            other_waypoint.transform.rotation.yaw)
        distance = waypoint.transform.location.distance(
            other_waypoint.transform.location)
        lateral_offset = DistanceTriggeredLaneChangeAgent._lateral_offset_to_waypoint(
            other_waypoint, waypoint)
        return (waypoint.lane_id == other_waypoint.lane_id
                and abs(yaw_diff) <= 45.0
                and distance <= 75.0
                and abs(lateral_offset) <= 2.5)

    @staticmethod
    def _lane_key(waypoint):
        if waypoint is None:
            return None
        return waypoint.lane_id

    def _update_lane_change_completion(self):
        if not self.lane_change_started or self.lane_change_completed:
            return

        carla_map = self.world.get_map()
        target_waypoint = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._lane_key(target_waypoint) == self._ego_lane_key
                or self._same_lane_id_and_heading(target_waypoint)):
            self.lane_change_completed = True
            self.target_final_lane_id = target_waypoint.lane_id

    def _get_actor_lane_id(self, actor):
        waypoint = self.world.get_map().get_waypoint(
            actor.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return None
        return waypoint.lane_id

    def _get_elapsed_seconds(self):
        snapshot = self.world.get_snapshot()
        if snapshot is None or snapshot.timestamp is None:
            return None
        return snapshot.timestamp.elapsed_seconds

    @staticmethod
    def _angle_diff_deg(a, b):
        return (a - b + 180.0) % 360.0 - 180.0

    @staticmethod
    def _lateral_offset_to_waypoint(point_waypoint, reference_waypoint):
        point_loc = point_waypoint.transform.location
        ref_loc = reference_waypoint.transform.location
        yaw_rad = math.radians(reference_waypoint.transform.rotation.yaw)
        dx = point_loc.x - ref_loc.x
        dy = point_loc.y - ref_loc.y
        return -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

    def _same_lane_id_and_heading(self, waypoint):
        if waypoint is None or self.ego_lane_id is None or self._ego_lane_yaw is None:
            return False
        yaw_diff = self._angle_diff_deg(
            waypoint.transform.rotation.yaw,
            self._ego_lane_yaw)
        return waypoint.lane_id == self.ego_lane_id and abs(yaw_diff) <= 45.0

    @staticmethod
    def _waypoint_debug(waypoint):
        if waypoint is None:
            return None
        loc = waypoint.transform.location
        return {
            "road_id": waypoint.road_id,
            "lane_id": waypoint.lane_id,
            "lane_type": str(waypoint.lane_type),
            "x": round(loc.x, 3),
            "y": round(loc.y, 3),
            "yaw": round(waypoint.transform.rotation.yaw, 3),
        }
