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
from utils.low_level_control import LowLevelControl


class FixedLaneSpeedAgent(object):
    """Speed-tracking ego policy with zero steering for fixed-lane validation."""

    def __init__(self, vehicle, goal_location, nominal_speed_mps=8.0):
        self.vehicle = vehicle
        self.nominal_speed = nominal_speed_mps
        self._low_level_control = LowLevelControl(vehicle)
        self.goal_reached = False

    def done(self):
        return self.goal_reached

    def run_step(self, pred_dict):
        vehicle_tf = self.vehicle.get_transform()
        vehicle_vel = self.vehicle.get_velocity()

        x, y = vehicle_tf.location.x, -vehicle_tf.location.y
        psi = -fth.fix_angle(np.radians(vehicle_tf.rotation.yaw))
        speed = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)

        z0 = np.array([x, y, psi, speed])
        u0 = np.array([0.0, 0.0])
        control = self._low_level_control.update(
            speed,
            0.0,
            self.nominal_speed,
            0.0)
        return control, z0, u0, True, np.nan
