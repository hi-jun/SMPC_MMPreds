import os
import sys
import unittest

import numpy as np
from scipy.stats import norm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    one_row_sigma_term,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402

ZERO = ((0.0, 0.0), (0.0, 0.0))


class TestNoEgoNoise(unittest.TestCase):
    """``noegonoise`` (2026-11-18): the margin's sigma drops the ego's process-noise term ``h_ego Q h_ego^T``
    (0.164 m with the default ``process_noise_cov``), so it is the predicted lead position's standard deviation alone;
    the agent and the NGSIM replay set ``process_noise_cov`` to zero, which only ``safety_std`` reads."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    COV = np.diag([0.25, 0.09])
    AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
             "_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_leadmerge_mergefirst_lkhold_entrant2_bcts_nolkentry"
             "_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_snap160_aheadonly_noegonoise")

    def _config(self, no_ego_noise=True):
        """onerow w 10 + zconf0.95 with the NGSIM arm's cost and limits; ``no_ego_noise`` as the token sets it."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=1,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 10.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
        config.one_row = True
        config.one_row_z_conf = 0.95
        if no_ego_noise:
            config.process_noise_cov = ZERO
        config.__post_init__()
        return config

    def _lead_only(self, gap0, lead_v, cov):
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[0, :, 1] = lead_v
        return MultimodalLeadPrediction(
            means=means, probabilities=np.array([1.0]), covariances=np.tile(cov, (1, steps, 1, 1)),
            active_mask=np.ones((1, steps), dtype=bool),
            chance_confidence=np.full((1, steps), np.nan),
            sigma_confidence=np.full((1, steps), np.nan),
            lead_mode_lk=np.ones((1, steps), dtype=bool),
            lead_vehicle_ids=np.full((1, steps), 10, dtype=int),
            clearance_scale=np.ones((1, steps)),
            lead_mode_probability=np.ones((1, steps)),
            mode_names=["10:lk"])

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_no_ego_noise
        self.assertTrue(parse(self.AGENT))
        self.assertTrue(parse("noegonoise_onerow"))
        self.assertFalse(parse(self.AGENT.replace("_noegonoise", "")))
        self.assertFalse(parse("acc_x_xnoegonoise"))
        self.assertFalse(parse("acc_x_noegonoisex"))

    def test_safety_std(self):
        # default: sqrt(0.1^2 + 1.3^2 * 0.1^2) = 0.16401 m even with no predicted variance
        self.assertAlmostEqual(safety_std(np.zeros((2, 2)), self._config(no_ego_noise=False)),
                               np.sqrt(0.01 + 1.69 * 0.01), places=12)
        self.assertAlmostEqual(safety_std(self.COV, self._config(no_ego_noise=False)),
                               np.sqrt(0.25 + 0.01 + 1.69 * 0.01), places=12)
        # noegonoise: the lead's own standard deviation, 0 without predicted variance
        self.assertEqual(safety_std(np.zeros((2, 2)), self._config()), 0.0)
        self.assertAlmostEqual(safety_std(self.COV, self._config()), 0.5, places=15)

    def test_sigma_term(self):
        config = self._config()
        self.assertAlmostEqual(one_row_sigma_term(self._lead_only(30.0, 10.0, self.COV), 0, 5, config),
                               norm.ppf(0.95) * 0.5, places=12)
        self.assertEqual(one_row_sigma_term(self._lead_only(30.0, 10.0, np.zeros((2, 2))), 0, 5, config), 0.0)

    def test_solve_without_predicted_variance(self):
        # a constant-velocity lead with zero covariance (the CV-ACC arm): sigma 0, the row is plain d_safe - short
        config = self._config()
        prediction = self._lead_only(30.0, 10.0, np.zeros((2, 2)))
        controller = NairACCSMPC(config)
        self.assertEqual(controller._cell_tightening(prediction, None, 0, 5), 0.0)
        solution = controller.solve(np.array([0.0, 10.0]), prediction,
                                    previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")


if __name__ == "__main__":
    unittest.main()
