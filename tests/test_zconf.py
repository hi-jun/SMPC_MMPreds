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
    OldACCReferenceAdapter,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    one_row_sigma_term,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestZConf(unittest.TestCase):
    """``zconf<b>`` with ``onerow``: each active cell's row gets back the sigma term ``z sigma``,
    ``z = Phi^-1(min(p, b))`` with ``p`` the cell's own lead-mode probability, so the row reads
    ``gap >= L + d0 + tau v - short - xi + z sigma`` with ``0 <= short <= tau v`` and the floor through the cap is
    ``L + d0 + z sigma``; the reference keeps the plain ``d_safe`` at the planned speed."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    TAU = 1.3
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])
    AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
             "_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold"
             "_entrant2_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95")

    def _config(self, z_conf=0.95, num_modes=2, **overrides):
        """onerow with w 10 and the NGSIM arm's cost and limits."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=num_modes,
            controller_variant=VARIANT_FIXED_RISK)
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
        config.one_row_z_conf = z_conf
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap_lead, gap_cutin, lead_v, p_hyp=0.3):
        """Scenario 0: the ego-lane lead's lane-keeping cells (p 1) ``gap_lead`` ahead; scenario 1: a cut-in hypothesis
        with probability ``p_hyp`` ``gap_cutin`` ahead.  Both at ``lead_v``."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[0, :, 0] = gap_lead + lead_v * self.DT * np.arange(steps)
        means[1, :, 0] = gap_cutin + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        confidence = np.full((2, steps), np.nan)
        confidence[1] = min(p_hyp, 0.6)
        lead_p = np.ones((2, steps))
        lead_p[1] = p_hyp
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0 - p_hyp, p_hyp]),
            covariances=np.tile(self.COV, (2, steps, 1, 1)),
            active_mask=np.ones((2, steps), dtype=bool),
            chance_confidence=confidence,
            sigma_confidence=np.full((2, steps), np.nan),
            lead_mode_lk=np.array([[True] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
            clearance_scale=np.ones((2, steps)),
            lead_mode_probability=lead_p,
            mode_names=["10:lk", "11:cutin"],
        )

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

    def _solve(self, config, prediction, v_ego=10.0):
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, v_ego]), prediction,
                                    previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return controller, solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_one_row_z_conf
        self.assertEqual(parse(self.AGENT), 0.95)
        self.assertEqual(parse("onerow_zconf0.9_ra0"), 0.9)
        self.assertIsNone(parse(self.AGENT.replace("_zconf0.95", "")))
        self.assertIsNone(parse("acc_x_xzconf0.95"))
        self.assertIsNone(parse("acc_x_zconf0.95x"))

    def test_config_and_agent_validation(self):
        self.assertEqual(self._config().one_row_z_conf, 0.95)
        with self.assertRaises(ValueError):                    # needs onerow
            self._config(one_row=False)
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                self._config(z_conf=bad)
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=self.AGENT.replace("_onerow", ""))
        self.assertIn("zconf", str(caught.exception))

    def test_z_values(self):
        prediction = self._prediction(80.0, 30.0, 10.0, p_hyp=0.3)
        config = self._config()
        sigma = safety_std(self.COV, config)
        self.assertAlmostEqual(norm.ppf(0.95), 1.6448536, places=6)
        self.assertAlmostEqual(norm.ppf(0.3), -0.5244005, places=6)
        # the lane-keeping lead (p 1): z = Phi^-1(0.95); the cut-in hypothesis (p 0.3): z = Phi^-1(0.3) < 0, not clamped
        self.assertAlmostEqual(one_row_sigma_term(prediction, 0, 5, config), norm.ppf(0.95) * sigma, places=12)
        self.assertAlmostEqual(one_row_sigma_term(prediction, 1, 5, config), norm.ppf(0.3) * sigma, places=12)
        controller = NairACCSMPC(config)
        self.assertAlmostEqual(controller._cell_tightening(prediction, None, 0, 5), norm.ppf(0.95) * sigma, places=12)
        self.assertAlmostEqual(controller._cell_tightening(prediction, None, 1, 5), norm.ppf(0.3) * sigma, places=12)
        # without the token onerow has no sigma term
        self.assertEqual(NairACCSMPC(self._config(z_conf=None))._cell_tightening(prediction, None, 0, 5), 0.0)
        self.assertEqual(one_row_sigma_term(prediction, 0, 5, self._config(z_conf=None)), 0.0)

    def test_prediction_without_cell_probabilities(self):
        # the replay's no-target placeholder (no lead_mode_probability, no cut-in chance cell): lane-keeping cells, p 1
        config = self._config()
        sigma = safety_std(self.COV, config)
        placeholder = self._lead_only(1000.0, 20.0)
        placeholder.lead_mode_probability = None
        self.assertAlmostEqual(one_row_sigma_term(placeholder, 0, 3, config), norm.ppf(0.95) * sigma, places=12)
        self._solve(self._config(num_modes=1), placeholder)
        # with a cut-in chance cell but no cell probabilities it raises
        prediction = self._prediction(80.0, 30.0, 10.0)
        prediction.lead_mode_probability = None
        with self.assertRaises(ValueError):
            one_row_sigma_term(prediction, 1, 3, config)

    def test_lane_keeping_row_and_floor(self):
        sigma = safety_std(self.COV, self._config(num_modes=1))
        z_sigma = norm.ppf(0.95) * sigma
        d_safe = self.L_D0 + self.TAU * 10.0
        # 2 m inside d_safe + z sigma: the shortfall takes it, no slack
        controller, solution = self._solve(self._config(num_modes=1), self._lead_only(d_safe + z_sigma - 2.0, 10.0))
        self.assertLess(solution.slack_max, 1.0e-6)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], 2.0, places=3)
        # 0.5 m inside the floor L + d0 + z sigma: the shortfall at its cap tau v, the slack takes the rest
        controller, solution = self._solve(self._config(num_modes=1), self._lead_only(self.L_D0 + z_sigma - 0.5, 10.0))
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], self.TAU * 10.0, places=3)
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.5, places=4)

    def test_cutin_row_with_negative_z(self):
        # p 0.3: z sigma < 0, so the floor through the cap lies below L + d0; a cut-in 0.1 m beyond that floor needs
        # no slack, its shortfall is tau v - 0.1
        sigma = safety_std(self.COV, self._config())
        floor = self.L_D0 + norm.ppf(0.3) * sigma
        prediction = self._prediction(200.0, floor + 0.1, 10.0, p_hyp=0.3)
        controller, solution = self._solve(self._config(), prediction)
        self.assertLess(solution.slack_max, 1.0e-6)
        self.assertAlmostEqual(controller._standoff_shortfall[1, 0], self.TAU * 10.0 - 0.1, places=3)

    def test_reference_keeps_plain_dsafe(self):
        ego = np.array([0.0, 10.0])
        prediction = self._prediction(25.0, 20.0, 8.0, p_hyp=0.3)
        with_z = OldACCReferenceAdapter(self._config()).generate(ego, prediction)
        without = OldACCReferenceAdapter(self._config(z_conf=None)).generate(ego, prediction)
        np.testing.assert_array_equal(with_z.v_ref, without.v_ref)
        np.testing.assert_array_equal(with_z.s_ref, without.s_ref)


if __name__ == "__main__":
    unittest.main()
