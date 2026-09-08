import os
import sys

import carla
import numpy as np

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
from utils import frenet_trajectory_handler as fth
from utils.low_level_control import (
    IdealLongitudinalActuator, LowLevelControl, lane_keeping_steer)


class FixedLaneSpeedAgent(object):
    """Speed-tracking policy that keeps its spawn lane for fixed-lane validation.

    The lane-keeping steer is the ego's: without it the heading picked up in
    the spawn hand-over (0.1-0.9 deg, run to run) stays for good, the pinned
    speed follows that heading, and at 7 m/s the vehicle reached the lane
    boundary in 14 s, where the map waypoint put it in the next lane and the
    ACC dropped it as a lead (2026-09-08).
    """

    def __init__(self, vehicle, goal_location, nominal_speed_mps=8.0):
        self.vehicle = vehicle
        self.nominal_speed = nominal_speed_mps
        self._low_level_control = LowLevelControl(vehicle)
        self._ideal_actuator = IdealLongitudinalActuator(vehicle)
        self._map = vehicle.get_world().get_map()
        self.goal_reached = False

    def done(self):
        return self.goal_reached

    def _lane_errors(self, vehicle_tf, psi):
        """Lateral offset (+left) and heading error to the lane centre, in the
        right-handed frame (y and yaw negated from CARLA)."""
        waypoint_tf = self._map.get_waypoint(
            vehicle_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        ).transform
        psi_wp = -fth.fix_angle(np.radians(waypoint_tf.rotation.yaw))
        dx = vehicle_tf.location.x - waypoint_tf.location.x
        dy = -vehicle_tf.location.y + waypoint_tf.location.y
        ey = -np.sin(psi_wp) * dx + np.cos(psi_wp) * dy
        epsi = fth.fix_angle(psi - psi_wp)
        return float(ey), float(epsi)

    def run_step(self, pred_dict):
        vehicle_tf = self.vehicle.get_transform()
        vehicle_vel = self.vehicle.get_velocity()

        x, y = vehicle_tf.location.x, -vehicle_tf.location.y
        psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))
        speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)

        z0 = np.array([x, y, psi, speed])
        ey, epsi = self._lane_errors(vehicle_tf, psi)
        df_des = lane_keeping_steer(ey, epsi)
        u0 = np.array([0.0, df_des])
        # No throttle: the ideal actuator holds the speed exactly, and any
        # throttle on a pinned vehicle sustains a two-tick speed cycle once
        # excited (+/-0.06 m/s at 7 m/s; pin experiment 2026-09-07).
        steer = float(np.clip(-df_des / self._low_level_control.max_steer_angle, -1.0, 1.0))
        control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=steer)
        self._ideal_actuator.track_speed(self.nominal_speed)
        return control, z0, u0, True, np.nan
