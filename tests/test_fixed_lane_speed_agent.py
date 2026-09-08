import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

import carla  # noqa: E402

from policies.fixed_lane_speed_agent import FixedLaneSpeedAgent  # noqa: E402

WHEELBASE = 2.6
MAX_STEER_DEG = 70.0


class _FakeLaneVehicle:
    """Kinematic bicycle in CARLA's frame (y to the right, yaw clockwise) on a
    straight lane along +x at y = 0.  ``set_target_velocity`` is applied at
    the next tick, as the simulator does."""

    def __init__(self, x, y, yaw_deg, speed, dt=0.05):
        self.x, self.y, self.yaw_deg, self.dt = float(x), float(y), float(yaw_deg), float(dt)
        self._vel = speed * self._heading()
        self._set = None
        self.steer = 0.0

    def _heading(self):
        yaw = np.radians(self.yaw_deg)
        return np.array([np.cos(yaw), np.sin(yaw)])

    def get_world(self):
        vehicle = self

        class _Settings:
            fixed_delta_seconds = vehicle.dt

        class _Waypoint:
            def __init__(self, x):
                self.transform = carla.Transform(carla.Location(x=x, y=0.0), carla.Rotation(yaw=0.0))

        class _Map:
            def get_waypoint(self, location, project_to_road=True, lane_type=None):
                return _Waypoint(location.x)

        class _World:
            def get_settings(self):
                return _Settings()

            def get_map(self):
                return _Map()

        return _World()

    def get_physics_control(self):
        class _Wheel:
            max_steer_angle = MAX_STEER_DEG

        class _Physics:
            wheels = [_Wheel()]

        return _Physics()

    def get_transform(self):
        return carla.Transform(carla.Location(x=self.x, y=self.y), carla.Rotation(yaw=self.yaw_deg))

    def get_velocity(self):
        return carla.Vector3D(x=float(self._vel[0]), y=float(self._vel[1]), z=0.0)

    def set_target_velocity(self, v):
        self._set = np.array([v.x, v.y])

    def apply(self, control):
        self.steer = float(control.steer)

    def tick(self):
        if self._set is not None:
            self._vel = self._set
        speed = float(np.linalg.norm(self._vel))
        delta = np.radians(self.steer * MAX_STEER_DEG)
        # Positive steer turns right, which is a positive (clockwise) yaw rate in CARLA.
        self.yaw_deg += np.degrees(speed / WHEELBASE * np.tan(delta) * self.dt)
        h = self._heading()
        self.x += speed * h[0] * self.dt
        self.y += speed * h[1] * self.dt
        self._vel = speed * h


def _steer_for(x, y, yaw_deg):
    vehicle = _FakeLaneVehicle(x, y, yaw_deg, speed=7.0)
    control, _, _, _, _ = FixedLaneSpeedAgent(vehicle, None, nominal_speed_mps=7.0).run_step({})
    return control.steer


class TestFixedLaneSpeedAgent(unittest.TestCase):
    def test_on_the_centre_and_aligned_drives_straight(self):
        self.assertAlmostEqual(_steer_for(0.0, 0.0, 0.0), 0.0, places=9)

    def test_offset_to_the_right_steers_left(self):
        self.assertLess(_steer_for(0.0, 1.0, 0.0), 0.0)

    def test_heading_to_the_right_steers_left(self):
        self.assertLess(_steer_for(0.0, 0.0, 2.0), 0.0)

    def test_offset_and_heading_error_are_closed_out(self):
        # The spawn hand-over left the subLV 0.4 m off centre with up to 0.9 deg
        # of heading error; it must be back on the centre line well inside the
        # 14 s that the drift used to take to reach the lane boundary.
        vehicle = _FakeLaneVehicle(0.0, -0.4, -0.9, speed=7.0)
        agent = FixedLaneSpeedAgent(vehicle, None, nominal_speed_mps=7.0)
        offsets = []
        for _ in range(int(10.0 / vehicle.dt)):
            control, _, _, _, _ = agent.run_step({})
            vehicle.apply(control)
            vehicle.tick()
            offsets.append(abs(vehicle.y))
        self.assertLess(max(offsets), 0.6)
        self.assertLess(abs(vehicle.y), 0.02)
        self.assertLess(abs(vehicle.yaw_deg), 0.05)
        self.assertAlmostEqual(float(np.linalg.norm(vehicle._vel)), 7.0, places=6)


if __name__ == "__main__":
    unittest.main()
