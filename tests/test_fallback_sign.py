import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    VARIANT_FIXED_RISK,
    original_nair_acc_config,
)


class TestFallbackSign(unittest.TestCase):
    """``_fallback_policy`` brakes behind a lead closer than the standoff (the sign was flipped until 2026-11-03)."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    EGO = np.array([0.0, 14.0])

    def _config(self, jerk_input):
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=1,
            controller_variant=VARIANT_FIXED_RISK)
        config.jerk_input = jerk_input
        config.allow_solver_fallback = True
        config.__post_init__()
        return config

    def _prediction(self, gap0):
        """One lead at constant speed 14 m/s, ``gap0`` m ahead (centre gap), at every step."""
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = gap0 + 14.0 * self.DT * np.arange(steps)
        means[0, :, 1] = 14.0
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0]),
            covariances=np.tile(np.diag([0.25, 0.09]), (1, steps, 1, 1)),
            active_mask=np.ones((1, steps), dtype=bool),
            mode_names=["10:lk"],
        )

    def _first_accel(self, jerk_input, gap0, accel0=0.0):
        controller = NairACCSMPC(self._config(jerk_input))
        prediction = self._prediction(gap0)
        reference = controller.reference_adapter.generate(self.EGO, prediction)
        reference = controller._with_linearization_input(reference, prediction)
        controller._set_command_context(accel0, self.COMMAND_DT, ego_accel=accel0, ego_speed=self.EGO[1])
        policy, _ = controller._fallback_policy(self.EGO, prediction, reference)
        first = reference.prev_u[0, 0] + policy.h[0, 0, 0]
        # jinput: the first input is a jerk, the planned acceleration after one step is accel0 + dt * j0
        return accel0 + self.DT * first if jerk_input else first

    def test_brakes_when_the_lead_is_too_close(self):
        # standoff L + d0 + tau v = 4.5 + 3 + 1.3 * 14 = 25.7 m against a 10 m centre gap
        for jerk_input in (True, False):
            with self.subTest(jerk_input=jerk_input):
                self.assertLess(self._first_accel(jerk_input, gap0=10.0), -0.1)

    def test_does_not_accelerate_toward_a_close_lead_while_braking(self):
        for jerk_input in (True, False):
            with self.subTest(jerk_input=jerk_input):
                self.assertLessEqual(self._first_accel(jerk_input, gap0=10.0, accel0=-2.0), -2.0 + 1e-9)

    def test_holds_a_ref_when_the_gap_is_ample(self):
        for jerk_input in (True, False):
            with self.subTest(jerk_input=jerk_input):
                self.assertAlmostEqual(self._first_accel(jerk_input, gap0=60.0), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
