import os
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.stats import norm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestNoCap(unittest.TestCase):
    """``nocap`` with ``onerow``: no upper bound on an active cell's shortfall, so the row
    ``gap >= L + d0 + tau v + z sigma - short - xi`` has no hard floor and the emergency slack stays 0; without the
    token the cap ``short <= tau v`` holds and the slack takes what lies below ``L + d0 + z sigma``."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    TAU = 1.3
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])
    AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
             "_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold"
             "_entrant2_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_nocap")

    def _config(self, no_cap=True, **overrides):
        """Arm C: onerow w 10 + zconf0.95 (+ nocap) with the NGSIM arm's cost and limits; one scenario."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=1, controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 10.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
        config.failsafe_recovery = True
        config.failsafe_decel, config.failsafe_jerk = -8.0, 10.0
        config.one_row = True
        config.one_row_z_conf = 0.95
        config.one_row_no_cap = no_cap
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _lead_only(self, gap0, lead_v):
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[0, :, 1] = lead_v
        return MultimodalLeadPrediction(
            means=means, probabilities=np.array([1.0]), covariances=np.tile(self.COV, (1, steps, 1, 1)),
            active_mask=np.ones((1, steps), dtype=bool),
            chance_confidence=np.full((1, steps), np.nan),
            sigma_confidence=np.full((1, steps), np.nan),
            lead_mode_lk=np.ones((1, steps), dtype=bool),
            lead_vehicle_ids=np.full((1, steps), 10, dtype=int),
            clearance_scale=np.ones((1, steps)),
            lead_mode_probability=np.ones((1, steps)),
            mode_names=["10:lk"])

    def _solve(self, config, prediction):
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, 10.0]), prediction,
                                    previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return controller, solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_one_row_no_cap
        self.assertTrue(parse(self.AGENT))
        self.assertTrue(parse("onerow_nocap_ra0"))
        self.assertFalse(parse(self.AGENT.replace("_nocap", "")))
        self.assertFalse(parse("acc_x_passmiqp_passnocap"))       # passnocap is another token
        self.assertFalse(parse("acc_x_nocapx"))

    def test_validation(self):
        self.assertTrue(self._config().one_row_no_cap)
        with self.assertRaises(ValueError):
            self._config(one_row=False, one_row_z_conf=None)
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=self.AGENT.replace("_onerow_zconf0.95", ""))
        self.assertIn("nocap", str(caught.exception))

    def test_cap_removed_with_the_token_kept_without(self):
        # the lead 0.5 m inside the floor L + d0 + z sigma at step 0 (both at 10 m/s)
        z_sigma = norm.ppf(0.95) * safety_std(self.COV, self._config())
        prediction = self._lead_only(self.L_D0 + z_sigma - 0.5, 10.0)
        # nocap: the shortfall goes beyond tau v by the 0.5 m, the slack stays 0
        controller, solution = self._solve(self._config(), prediction)
        self.assertLess(solution.slack_max, 1.0e-6)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], self.TAU * 10.0 + 0.5, places=3)
        # without the token: the shortfall at its cap tau v, the slack takes the 0.5 m
        controller, solution = self._solve(self._config(no_cap=False), prediction)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], self.TAU * 10.0, places=3)
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.5, places=4)

    def test_inactive_cells_keep_no_shortfall(self):
        prediction = self._lead_only(40.0, 10.0)
        prediction.active_mask[0, 8:] = False
        controller, _ = self._solve(self._config(), prediction)
        np.testing.assert_allclose(controller._standoff_shortfall[0, 8:], 0.0, atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
