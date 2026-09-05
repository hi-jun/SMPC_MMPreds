import carla

from policies.lead_speed_cap import (
    LeadSpeedCapMixin,
    SPEED_CAP_GAIN,
    SPEED_CAP_MIN_LONGITUDINAL_GAP_M,
)
from policies.mpc_agent import MPCAgent
from utils import frenet_trajectory_handler as fth


MIN_POST_CUTOUT_ROUTE_DISTANCE = 500.0


class DistanceTriggeredCutOutAgent(LeadSpeedCapMixin, MPCAgent):
    """MPC target policy that leaves the ego lane on a distance or time trigger."""

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
            cutout_direction="left",
            trigger_mode="ego_gap",
            trigger_time_s=0.0,
            speed_cap_min_gap_m=SPEED_CAP_MIN_LONGITUDINAL_GAP_M,
            speed_cap_gain=SPEED_CAP_GAIN):
        super().__init__(
            vehicle,
            goal_location,
            nominal_speed_mps=nominal_speed_mps,
            dt=dt,
            N=N,
            N_modes=N_modes)

        self.trigger_distance_m = trigger_distance_m
        # "ego_gap": leave once the ego is within trigger_distance_m (straight
        # line).  "lead_gap": leave once the lead in this lane is within
        # trigger_distance_m (centre to centre), as the cut-in kinds do.
        # "time": leave trigger_time_s after control starts (nothing ahead);
        # kept as trigger_delay_s since self.trigger_time_s logs when it fired.
        if trigger_mode not in ("ego_gap", "lead_gap", "time"):
            raise ValueError(f"unknown trigger_mode {trigger_mode!r}")
        self.trigger_mode = trigger_mode
        self.trigger_delay_s = float(trigger_time_s)
        self.speed_cap_min_gap_m = float(speed_cap_min_gap_m)
        self.speed_cap_gain = float(speed_cap_gain)
        self.distance_same_lane = distance_same_lane
        self.distance_other_lane = distance_other_lane
        self.distance_lane_change = distance_lane_change
        self.cutout_direction = str(cutout_direction).lower()

        self.cutout_started = False
        self.cutout_completed = False
        self.control_start_time_s = None
        self.trigger_time_s = None
        self.trigger_distance_at_start = None
        self.source_lane_id = self._get_actor_lane_id(self.vehicle)
        self.target_lane_id = None
        self._target_lane_key = None
        self._target_lane_yaw = None
        # Follow the spawn lane until the trigger: the planner route from
        # MPCAgent changes lane on its own (into the right lane at the Town04
        # merge), which put the vehicle beside the ego instead of ahead of it.
        self._switch_to_current_lane_route()

    def run_step(self, pred_dict):
        if self.control_start_time_s is None:
            self.control_start_time_s = self._get_elapsed_seconds()
        if not self.cutout_started:
            trigger_distance = None
            if self.trigger_mode == "time":
                triggered = (self._get_elapsed_seconds() - self.control_start_time_s
                             >= self.trigger_delay_s)
            else:
                if self.trigger_mode == "lead_gap":
                    lead = self._find_same_lane_front_lead()
                    trigger_distance = None if lead is None else float(lead[1])
                else:
                    trigger_distance = self._ego_distance()
                triggered = (trigger_distance is not None
                             and trigger_distance <= self.trigger_distance_m)
            if triggered:
                self.trigger_distance_at_start = trigger_distance
                self.trigger_time_s = self._get_elapsed_seconds()
                self._switch_to_cutout_route()
                self.cutout_started = True

        result = super().run_step(pred_dict)
        self._update_cutout_completion()
        return result

    def get_cut_in_log(self):
        return {
            "trigger_time_s": self.trigger_time_s,
            "trigger_mode": self.trigger_mode,
            "trigger_delay_s": self.trigger_delay_s,
            "control_start_time_s": self.control_start_time_s,
            "trigger_distance_at_start": self.trigger_distance_at_start,
            "source_lane_id": self.source_lane_id,
            "target_lane_id": self.target_lane_id,
            "cutout_started": self.cutout_started,
            "cutout_completed": self.cutout_completed,
            "cutout_direction": self.cutout_direction,
            "speed_cap_min_gap_m": self.speed_cap_min_gap_m,
            "speed_cap_gain": self.speed_cap_gain,
        }

    def _speed_cap_search_lane(self):
        # Once the cut-out started, pace the lane being entered rather than
        # the lead being left (and stay there: the waypoint lookup flickers
        # between lanes around the boundary).
        target_lane_id = self.target_lane_id if self.cutout_started else None
        return target_lane_id, target_lane_id is not None and not self.cutout_completed

    def _ego_distance(self):
        ego_actor = self._find_ego_actor()
        if ego_actor is None:
            return None
        return self.vehicle.get_location().distance(ego_actor.get_location())

    def _find_ego_actor(self):
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            if actor.attributes.get("role_name") == "ego":
                return actor
        return None

    def _switch_to_cutout_route(self):
        self._switch_to_route(self._generate_cutout_plan())

    def _switch_to_current_lane_route(self):
        self._switch_to_route(self._generate_current_lane_plan())

    def _switch_to_route(self, plan):
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
            raise RuntimeError("Could not find a driving waypoint for pre-trigger cutout actor.")

        plan = [(waypoint, None)]
        keep_distance = max(
            self.distance_same_lane + self.distance_lane_change + self.distance_other_lane,
            MIN_POST_CUTOUT_ROUTE_DISTANCE,
        )
        self._append_forward_waypoints(plan, keep_distance, 2.0)
        return plan

    def _generate_cutout_plan(self):
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:
            raise RuntimeError("Could not find a driving waypoint for cutout actor.")

        plan = [(waypoint, None)]
        step_distance = 2.0
        self._append_forward_waypoints(plan, self.distance_same_lane, step_distance)

        target_lane_waypoint = self._select_adjacent_lane(plan[-1][0])
        if target_lane_waypoint is None or target_lane_waypoint.lane_type != carla.LaneType.Driving:
            raise RuntimeError(
                f"Could not find {self.cutout_direction} driving lane for cutout actor.")

        self.target_lane_id = target_lane_waypoint.lane_id
        self._target_lane_key = self._lane_key(target_lane_waypoint)
        self._target_lane_yaw = target_lane_waypoint.transform.rotation.yaw
        target_lane_next = target_lane_waypoint.next(self.distance_lane_change)
        if not target_lane_next:
            raise RuntimeError("Could not extend cutout route on the target lane.")
        plan.append((target_lane_next[0], None))

        self._append_forward_waypoints(
            plan,
            max(self.distance_other_lane, MIN_POST_CUTOUT_ROUTE_DISTANCE),
            step_distance,
        )
        return plan

    def _select_adjacent_lane(self, waypoint):
        if self.cutout_direction == "left":
            return waypoint.get_left_lane()
        if self.cutout_direction == "right":
            return waypoint.get_right_lane()
        raise ValueError("cutout_direction must be 'left' or 'right'")

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

    def _update_cutout_completion(self):
        if not self.cutout_started or self.cutout_completed:
            return
        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if (self._lane_key(waypoint) == self._target_lane_key
                or self._same_lane_id_and_heading(waypoint)):
            self.cutout_completed = True

    def _get_actor_lane_id(self, actor):
        waypoint = self.world.get_map().get_waypoint(
            actor.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return None
        return waypoint.lane_id

    @staticmethod
    def _lane_key(waypoint):
        if waypoint is None:
            return None
        return waypoint.road_id, waypoint.lane_id

    def _same_lane_id_and_heading(self, waypoint):
        if waypoint is None or self.target_lane_id is None or self._target_lane_yaw is None:
            return False
        yaw_diff = self._angle_diff_deg(
            waypoint.transform.rotation.yaw,
            self._target_lane_yaw)
        return waypoint.lane_id == self.target_lane_id and abs(yaw_diff) <= 45.0

    def _get_elapsed_seconds(self):
        snapshot = self.world.get_snapshot()
        if snapshot is None or snapshot.timestamp is None:
            return None
        return snapshot.timestamp.elapsed_seconds
