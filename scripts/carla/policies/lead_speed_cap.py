import math

import carla
import numpy as np

from utils import frenet_trajectory_handler as fth


SPEED_CAP_VEHICLE_LENGTH_M = 4.5
# The cap starts biting at the centre-to-centre gap where v_cap reaches the
# nominal speed:  4.5 + MIN_GAP + (nominal - lead_speed) / GAIN.  For the cut-in
# scenarios (15 m/s over a 9 m/s lead) that is 12.0 m, below both lead-gap
# triggers (17 m normal, 13 m aggressive), so the cap no longer decelerates the
# vehicle before it starts its lane change.  Keep this engagement gap below the
# smallest trigger distance in use; a larger nominal-to-lead speed difference
# raises it.  Both values can be overridden per vehicle through the
# constructor (the no-cut-in kind paces its lead with them).
SPEED_CAP_MIN_LONGITUDINAL_GAP_M = 2.5
SPEED_CAP_GAIN = 1.2
SPEED_CAP_LATERAL_THRESHOLD_M = 2.5
SPEED_CAP_YAW_THRESHOLD_DEG = 45.0


class LeadSpeedCapMixin(object):
    """Caps the MPC reference speed toward the nearest vehicle ahead in the
    lane being followed, so a vehicle closing on its lead does not run into it
    before or during its lane change.

    Mix in ahead of MPCAgent.  The agent sets ``speed_cap_min_gap_m`` and
    ``speed_cap_gain`` and implements ``_speed_cap_search_lane`` returning
    ``(lane_id, changing)``: the lane to search (None: the lane under the
    vehicle) and whether a lane change into it is under way.
    """

    def _get_reference_traj(self, x0, y0, psi0, v0):
        ref_dict = super()._get_reference_traj(x0, y0, psi0, v0)
        lead = self._find_same_lane_front_lead()
        if lead is None:
            return ref_dict

        _, actual_gap, lead_speed = lead
        desired_gap = SPEED_CAP_VEHICLE_LENGTH_M + self.speed_cap_min_gap_m
        v_cap = lead_speed + self.speed_cap_gain * (actual_gap - desired_gap)
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
        target_lane_id, changing = self._speed_cap_search_lane()
        lane_id_ref = target_waypoint.lane_id if target_lane_id is None else target_lane_id

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

        yaw_diff = LeadSpeedCapMixin._angle_diff_deg(
            target_waypoint.transform.rotation.yaw,
            actor_waypoint.transform.rotation.yaw)
        if abs(yaw_diff) > SPEED_CAP_YAW_THRESHOLD_DEG:
            return False

        if not check_lateral:
            return True
        lateral_offset = LeadSpeedCapMixin._lateral_offset_to_waypoint(
            actor_waypoint, target_waypoint)
        return abs(lateral_offset) <= SPEED_CAP_LATERAL_THRESHOLD_M

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
