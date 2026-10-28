import os
import sys
import unittest

import numpy as np
from scipy.linalg import expm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    acc_lag_coefficients,
    acc_lag_step,
    acc_step,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestAccelLagModel(unittest.TestCase):
    """``alag<tau>``: tau * da/dt + a = u in the prediction model."""

    HORIZON = 15
    EGO = np.array([0.0, 14.0])

    def test_coefficients_are_the_exact_zero_order_hold(self):
        for dt, tau in ((0.2, 1.0), (0.1, 0.45), (0.2, 0.3)):
            # [s, v, a, u] with u held: the matrix exponential of the augmented system
            continuous = np.zeros((4, 4))
            continuous[0, 1] = continuous[1, 2] = 1.0
            continuous[2, 2], continuous[2, 3] = -1.0 / tau, 1.0 / tau
            phi = expm(continuous * dt)
            c_sa, c_va, decay = acc_lag_coefficients(dt, tau)
            np.testing.assert_allclose(phi[0, :], [1.0, dt, c_sa, 0.5 * dt ** 2 - c_sa], atol=1e-12)
            np.testing.assert_allclose(phi[1, :], [0.0, 1.0, c_va, dt - c_va], atol=1e-12)
            np.testing.assert_allclose(phi[2, :], [0.0, 0.0, decay, 1.0 - decay], atol=1e-12)

    def test_small_tau_is_the_double_integrator(self):
        state, accel = acc_lag_step([3.0, 14.0], 0.7, -2.0, 0.2, 1.0e-6)
        np.testing.assert_allclose(state, acc_step([3.0, 14.0], -2.0, 0.2), atol=1e-5)
        self.assertAlmostEqual(accel, -2.0, places=6)

    def _config(self, **overrides):
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=0.2, desired_speed=20.0, num_modes=2,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.standoff_shortfall_weight = 100.0
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap0=30.0, lead_v=12.0):
        """Lane keeping without a lead, and a cut-in that leads at every step."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[:, :, 0] = gap0 + lead_v * 0.2 * np.arange(steps)
        means[:, :, 1] = lead_v
        active = np.zeros((2, steps), dtype=bool)
        active[1] = True
        confidence = np.full((2, steps), np.nan)
        confidence[1] = 0.4
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([0.6, 0.4]),
            covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)),
            active_mask=active,
            chance_confidence=confidence,
            lead_mode_lk=np.zeros((2, steps), dtype=bool),
            lead_vehicle_ids=np.where(active, 10, -1),
            clearance_scale=np.ones((2, steps)),
            mode_names=["10:lk", "10:cutin"],
        )

    def _solve(self, ego_accel, previous_accel_cmd=0.0, **overrides):
        controller = NairACCSMPC(self._config(**overrides))
        return controller.solve(self.EGO, self._prediction(), previous_accel_cmd=previous_accel_cmd,
                                command_dt=0.1, ego_accel=ego_accel)

    def test_plan_follows_the_lagged_model_from_the_actual_acceleration(self):
        for ego_accel in (None, 0.8, -1.5):
            solution = self._solve(ego_accel, previous_accel_cmd=0.3, accel_lag_tau=1.0)
            self.assertNotEqual(solution.solve_path, "fallback")
            start = 0.3 if ego_accel is None else ego_accel  # without a measurement: the previous command
            for mode in range(2):
                state, accel = self.EGO.copy(), start
                for step in range(self.HORIZON):
                    state, accel = acc_lag_step(state, accel, solution.u_nominal[mode, step], 0.2, 1.0)
                    np.testing.assert_allclose(solution.x_nominal[mode, step + 1], state, atol=1e-6)

    def test_actual_acceleration_enters_the_plan(self):
        # Same command history, different actual acceleration: the lag carries it into the speed.
        rising = self._solve(1.0, accel_lag_tau=1.0)
        falling = self._solve(-1.0, accel_lag_tau=1.0)
        self.assertGreater(rising.x_nominal[0, 1, 1], falling.x_nominal[0, 1, 1] + 0.2)
        # The double integrator ignores it.
        plain_up = self._solve(1.0)
        plain_down = self._solve(-1.0)
        np.testing.assert_allclose(plain_up.x_nominal, plain_down.x_nominal, atol=1e-9)

    def test_invalid_lag_raises(self):
        with self.assertRaises(ValueError):
            self._config(accel_lag_tau=0.0)
        with self.assertRaises(ValueError):
            self._config(accel_lag_tau=1.0, controller_variant=VARIANT_MULTIMODAL_OL,
                         standoff_shortfall_weight=None)

    def test_policy_tokens(self):
        base = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr"
                "_trajcutin_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40")
        self.assertIsNone(ACCNairSMPCAgent._parse_accel_lag_tau(base))
        self.assertIsNone(ACCNairSMPCAgent._parse_r_a(base))
        self.assertIsNone(ACCNairSMPCAgent._parse_r_a(base.replace("_ctx0", "_cutin_ramp0.6_ctx0")))
        policy = base + "_alag1_ra0"
        self.assertEqual(ACCNairSMPCAgent._parse_accel_lag_tau(policy), 1.0)
        self.assertEqual(ACCNairSMPCAgent._parse_r_a(policy), 0.0)
        self.assertEqual(ACCNairSMPCAgent._parse_accel_lag_tau(base + "_alag0.45"), 0.45)
        self.assertEqual(ACCNairSMPCAgent._parse_r_a(base.replace("_rmove", "_ra0.5_rmove")), 0.5)


if __name__ == "__main__":
    unittest.main()
