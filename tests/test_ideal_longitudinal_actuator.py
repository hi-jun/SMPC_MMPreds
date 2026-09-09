import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

import carla  # noqa: E402

from utils.low_level_control import IdealLongitudinalActuator  # noqa: E402


class _FakeVehicle:
    """Straight-driving vehicle whose physics step eats ``drag`` m/s per tick
    and adds ``slip`` m/s of lateral velocity, along a yaw in degrees.

    ``alternating`` adds a two-tick component to the loss, the shape the
    reference-snap limit cycle leaves on a target vehicle."""

    def __init__(self, yaw_deg, speed, drag=0.1, slip=0.0, dt=0.05,
                 alternating=0.0):
        self.yaw_deg = float(yaw_deg)
        self.drag = float(drag)
        self.slip = float(slip)
        self.dt = float(dt)
        self.alternating = float(alternating)
        self._ticks = 0
        self._set = None
        h = self._heading()
        self._vel = np.array([speed * h[0], speed * h[1]])

    def _heading(self):
        yaw = np.radians(self.yaw_deg)
        return np.array([np.cos(yaw), np.sin(yaw)])

    def get_world(self):
        vehicle = self

        class _Settings:
            fixed_delta_seconds = vehicle.dt

        class _World:
            def get_settings(self):
                return _Settings()

        return _World()

    def get_transform(self):
        rotation = carla.Rotation(yaw=self.yaw_deg)
        return carla.Transform(carla.Location(), rotation)

    def get_velocity(self):
        return carla.Vector3D(x=float(self._vel[0]), y=float(self._vel[1]), z=0.0)

    def set_target_velocity(self, v):
        self._set = np.array([v.x, v.y])

    def tick(self):
        h = self._heading()
        n = np.array([-h[1], h[0]])
        v = self._set if self._set is not None else self._vel
        v_lon = float(np.dot(v, h)) - self.drag - (-1) ** self._ticks * self.alternating
        self._ticks += 1
        v_lat = float(np.dot(v, n)) + self.slip
        self._vel = v_lon * h + v_lat * n


class TestIdealLongitudinalActuator(unittest.TestCase):
    def test_measured_speed_follows_the_integrated_command(self):
        vehicle = _FakeVehicle(yaw_deg=-90.0, speed=13.0, drag=0.1)
        actuator = IdealLongitudinalActuator(vehicle)
        speeds = []
        for _ in range(20):
            actuator.update(0.0)
            vehicle.tick()
            speeds.append(float(np.linalg.norm(vehicle._vel)))
        # The drag is lost until the same-phase compensation has seen it (one
        # tick later than the two-tick mean it replaced, and half a tick of
        # drag lower); from then on the speed holds exactly.
        self.assertAlmostEqual(speeds[0], 12.9, places=5)
        for speed in speeds[1:]:
            self.assertAlmostEqual(speed, 12.8, places=5)

        for _ in range(10):
            actuator.update(-2.0)
            vehicle.tick()
        self.assertAlmostEqual(float(np.linalg.norm(vehicle._vel)), 12.8 - 2.0 * 0.05 * 10, places=5)

    def test_a_two_tick_disturbance_is_cancelled_not_sustained(self):
        """The compensation predicts the next tick's loss, so it must match it.

        The two-tick mean used until 2026-09-10 is flat against an alternating
        loss while the loss it predicts is not, so it passed the alternation
        through every tick and a cut-in target at 11 m/s kept a +/-0.04 m/s
        saw-tooth for the whole run -- which a 0.1 s finite-difference
        acceleration feature reads as an acceleration flipping sign every
        sample.  The same-phase value cancels it.
        """
        vehicle = _FakeVehicle(yaw_deg=-90.0, speed=11.0, drag=0.02,
                               alternating=0.02)
        actuator = IdealLongitudinalActuator(vehicle)
        speeds = []
        for _ in range(40):
            actuator.update(0.0)
            vehicle.tick()
            speeds.append(float(np.linalg.norm(vehicle._vel)))
        settled = np.asarray(speeds[20:])
        steps = np.abs(np.diff(settled))
        self.assertLess(float(np.max(steps)), 1e-6, "no residual ripple")
        # Two-tick mean, same disturbance: 0.02 m/s of ripple on every tick,
        # sign alternating, forever.
        self.assertAlmostEqual(float(settled.mean()), settled[0], places=6)

    def test_track_speed_ramps_at_the_limit_then_holds_exactly(self):
        vehicle = _FakeVehicle(yaw_deg=0.0, speed=5.0, drag=0.0)
        actuator = IdealLongitudinalActuator(vehicle)
        actuator.track_speed(7.0, a_max=2.0)
        vehicle.tick()
        self.assertAlmostEqual(float(vehicle._vel[0]), 5.0 + 2.0 * 0.05, places=5)
        for _ in range(40):
            actuator.track_speed(7.0, a_max=2.0)
            vehicle.tick()
        self.assertAlmostEqual(float(vehicle._vel[0]), 7.0, places=5)
        self.assertAlmostEqual(float(vehicle._vel[1]), 0.0, places=5)

    def test_lateral_velocity_from_the_tyres_is_left_alone(self):
        vehicle = _FakeVehicle(yaw_deg=30.0, speed=10.0, drag=0.0, slip=0.2)
        actuator = IdealLongitudinalActuator(vehicle)
        h = vehicle._heading()
        n = np.array([-h[1], h[0]])
        for _ in range(3):
            actuator.update(0.0)
            vehicle.tick()
        self.assertAlmostEqual(float(np.dot(vehicle._vel, h)), 10.0, places=5)
        self.assertAlmostEqual(float(np.dot(vehicle._vel, n)), 0.6, places=5)


if __name__ == "__main__":
    unittest.main()
