import math

import carla

from policies.mpc_agent import MPCAgent
from utils import frenet_trajectory_handler as fth


MIN_POST_CUTOUT_ROUTE_DISTANCE = 500.0


class DistanceTriggeredCutOutAgent(MPCAgent):
    """MPC target policy that leaves the ego lane after ego is close enough."""

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
            cutout_direction="left"):
        super().__init__(
            vehicle,
            goal_location,
            nominal_speed_mps=nominal_speed_mps,
            dt=dt,
            N=N,
            N_modes=N_modes)

        self.trigger_distance_m = trigger_distance_m
        self.distance_same_lane = distance_same_lane
        self.distance_other_lane = distance_other_lane
        self.distance_lane_change = distance_lane_change
        self.cutout_direction = str(cutout_direction).lower()

        self.cutout_started = False
        self.cutout_completed = False
        self.trigger_time_s = None
        self.trigger_distance_at_start = None
        self.source_lane_id = self._get_actor_lane_id(self.vehicle)
        self.target_lane_id = None
        self._target_lane_key = None
        self._target_lane_yaw = None

    def run_step(self, pred_dict):
        if not self.cutout_started:
            ego_actor = self._find_ego_actor()
            if ego_actor is not None and self._within_trigger_distance(
                    self.vehicle, ego_actor, self.trigger_distance_m):
                self.trigger_distance_at_start = self.vehicle.get_location().distance(
                    ego_actor.get_location())
                self.trigger_time_s = self._get_elapsed_seconds()
                self._switch_to_cutout_route()
                self.cutout_started = True

        result = super().run_step(pred_dict)
        self._update_cutout_completion()
        return result

    def get_cut_in_log(self):
        return {
            "trigger_time_s": self.trigger_time_s,
            "trigger_distance_at_start": self.trigger_distance_at_start,
            "source_lane_id": self.source_lane_id,
            "target_lane_id": self.target_lane_id,
            "cutout_started": self.cutout_started,
            "cutout_completed": self.cutout_completed,
            "cutout_direction": self.cutout_direction,
        }

    @staticmethod
    def _within_trigger_distance(actor, reference_actor, trigger_distance_m):
        if actor is None or reference_actor is None:
            return False
        actor_location = actor.get_location()
        reference_location = reference_actor.get_location()
        if actor_location is None or reference_location is None:
            return False
        return actor_location.distance(reference_location) <= trigger_distance_m

    def _find_ego_actor(self):
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            if actor.attributes.get("role_name") == "ego":
                return actor
        return None

    def _switch_to_cutout_route(self):
        plan = self._generate_cutout_plan()
        way_s, way_xy, way_yaw = fth.extract_path_from_waypoints(plan)
        self._frenet_traj = fth.FrenetTrajectoryHandler(
            way_s, way_xy, way_yaw, s_resolution=0.5)
        self._fit_velocity_profile()
        self.goal_reached = False
        self.warm_start = None

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
