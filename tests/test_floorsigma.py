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
    one_row_floor_terms,
    one_row_sigma_term,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestFloorSigma(unittest.TestCase):
    """``floorsigma`` with ``onerow`` and ``zconf<b>``: the priced row has no sigma term,
    ``gap >= L + d0 + tau v - short - xi`` with ``short >= 0`` uncapped and priced ``w P_j short^2``, and every active
    cell has a hard floor row ``gap >= L + d0 + z sigma - xi`` that shares its emergency slack; the reference keeps
    the plain ``d_safe``."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    TAU = 1.3
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])
    AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
             "_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_leadmerge_mergefirst_lkhold_nolkentry_amin3.5_amax2"
             "_jlim2.5_jlimup5_onerow_zconf0.95_snap160_aheadonly_noegonoise_floorsigma")

    def _config(self, floor_sigma=True, num_modes=1, **overrides):
        """The final design's controller (onerow w 10 + zconf0.95 + snap160, ISO limits, no ego noise)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=num_modes,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 10.0
        config.jerk_input = True
        config.snap_weight = 160.0
        config.slack_l1_weight = 1.0e6
        config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
        config.process_noise_cov = ((0.0, 0.0), (0.0, 0.0))
        config.one_row = True
        config.one_row_z_conf = 0.95
        config.one_row_floor_sigma = floor_sigma
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _lead_only(self, gap0, lead_v, cov=None):
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[0, :, 1] = lead_v
        return MultimodalLeadPrediction(
            means=means, probabilities=np.array([1.0]),
            covariances=np.tile(self.COV if cov is None else cov, (1, steps, 1, 1)),
            active_mask=np.ones((1, steps), dtype=bool),
            chance_confidence=np.full((1, steps), np.nan),
            sigma_confidence=np.full((1, steps), np.nan),
            lead_mode_lk=np.ones((1, steps), dtype=bool),
            lead_vehicle_ids=np.full((1, steps), 10, dtype=int),
            clearance_scale=np.ones((1, steps)),
            lead_mode_probability=np.ones((1, steps)),
            mode_names=["10:lk"])

    def _two_scenarios(self, gap_lead, gap_cutin, lead_v, p_hyp=0.3):
        """Scenario 0: the lead's lane keeping (p 1); scenario 1: a cut-in hypothesis with probability ``p_hyp``."""
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
            means=means, probabilities=np.array([1.0 - p_hyp, p_hyp]),
            covariances=np.tile(self.COV, (2, steps, 1, 1)),
            active_mask=np.ones((2, steps), dtype=bool),
            chance_confidence=confidence,
            sigma_confidence=np.full((2, steps), np.nan),
            lead_mode_lk=np.array([[True] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
            clearance_scale=np.ones((2, steps)),
            lead_mode_probability=lead_p,
            mode_names=["10:lk", "11:cutin"])

    def _solve(self, config, prediction, v_ego=10.0):
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, v_ego]), prediction,
                                    previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return controller, solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_floor_sigma
        self.assertTrue(parse(self.AGENT))
        self.assertTrue(parse("onerow_floorsigma_flooreps0.05"))
        self.assertFalse(parse(self.AGENT.replace("_floorsigma", "")))
        self.assertFalse(parse("acc_x_xfloorsigma"))
        self.assertFalse(parse("acc_x_floorsigmax"))

    def test_validation(self):
        self.assertTrue(self._config().one_row_floor_sigma)
        with self.assertRaises(ValueError):                     # needs onerow
            self._config(one_row=False, one_row_z_conf=None)
        with self.assertRaises(ValueError) as caught:           # no sigma term to move
            self._config(one_row_z_conf=None)
        self.assertIn("zconf", str(caught.exception))
        with self.assertRaises(ValueError) as caught:           # nocap drops every hard row
            self._config(one_row_no_cap=True)
        self.assertIn("nocap", str(caught.exception))
        agent = ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                                 smpc_config=self.AGENT)
        self.assertTrue(agent.controller_config.one_row_floor_sigma)
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=self.AGENT.replace("_onerow_zconf0.95", ""))
        self.assertIn("floorsigma", str(caught.exception))

    def test_the_sigma_term_moves_to_the_floor(self):
        prediction = self._two_scenarios(80.0, 30.0, 10.0, p_hyp=0.3)
        prediction.active_mask[1, 9:] = False
        config = self._config(num_modes=2)
        controller = NairACCSMPC(config)
        terms, clipped = one_row_floor_terms(prediction, config)
        self.assertFalse(clipped.any())
        for mode in range(2):
            for step in range(self.HORIZON + 1):
                # the priced row has none; the floor row has zconf's z sigma on the active cells, 0 elsewhere
                self.assertEqual(controller._cell_tightening(prediction, None, mode, step), 0.0)
                expected = (one_row_sigma_term(prediction, mode, step, config)
                            if prediction.active_mask[mode, step] else 0.0)
                self.assertEqual(terms[mode, step], expected)
        sigma = safety_std(self.COV, config)
        self.assertAlmostEqual(terms[0, 3], norm.ppf(0.95) * sigma, places=12)
        self.assertAlmostEqual(terms[1, 3], norm.ppf(0.3) * sigma, places=12)

    def test_priced_target_is_the_plain_dsafe(self):
        # 2 m inside d_safe and above the floor: the shortfall is the 2 m (with zconf in the row it was 2 m + z sigma)
        z_sigma = norm.ppf(0.95) * safety_std(self.COV, self._config())
        d_safe = self.L_D0 + self.TAU * 10.0
        prediction = self._lead_only(d_safe - 2.0, 10.0)
        controller, solution = self._solve(self._config(), prediction)
        self.assertLess(solution.slack_max, 1.0e-6)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], 2.0, places=3)
        controller, _ = self._solve(self._config(floor_sigma=False), prediction)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], 2.0 + z_sigma, places=3)

    def test_floor_row(self):
        # 0.5 m inside the floor L + d0 + z sigma: the slack takes the 0.5 m and the shortfall is tau v - z sigma, the
        # single row's values under the cap short <= tau v - z sigma
        z_sigma = norm.ppf(0.95) * safety_std(self.COV, self._config())
        controller, solution = self._solve(self._config(), self._lead_only(self.L_D0 + z_sigma - 0.5, 10.0))
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.5, places=4)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], self.TAU * 10.0 - z_sigma, places=3)
        self.assertAlmostEqual(solution.chance_margin_min, -0.5, places=3)
        self.assertFalse(solution.feasible is None)

    def test_low_speed_floor_above_the_time_gap(self):
        # v 0.5 m/s and sigma 2 m: T = 1.645 * 2 = 3.29 m > tau v = 0.65 m, where the cap tau v - T has no feasible
        # shortfall; the floor row holds: 0.2 m beyond it no slack and no shortfall, 0.3 m inside it a 0.3 m slack
        cov = np.diag([4.0, 0.09])
        config = self._config()
        floor = self.L_D0 + norm.ppf(0.95) * safety_std(cov, config)
        self.assertGreater(floor - self.L_D0, self.TAU * 0.5)
        controller, solution = self._solve(config, self._lead_only(floor + 0.2, 0.5, cov=cov), v_ego=0.5)
        self.assertLess(float(solution.slack[0, 0]), 1.0e-6)
        self.assertLess(controller._standoff_shortfall[0, 0], 1.0e-4)
        controller, solution = self._solve(self._config(), self._lead_only(floor - 0.3, 0.5, cov=cov), v_ego=0.5)
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.3, places=4)
        self.assertLess(controller._standoff_shortfall[0, 0], 1.0e-4)

    def test_rows_at_the_solution(self):
        # at every active cell of the solved plan: the slack is what the floor row needs, max(0, L + d0 + T - gap),
        # and the shortfall what the priced row needs on top of it, max(0, L + d0 + tau v - gap - xi).  First a plan
        # that keeps every floor (a cut-in 3 m beyond its floor, closing at 2 m/s), then one that cannot (0.2 m beyond
        # it, closing at 4 m/s), where only the slack is compared: with slacks of metres priced at 1e6 per metre the
        # barrier's tolerance leaves the shortfall of the other scenario's last steps up to 0.2 m above its optimum
        config = self._config(num_modes=2)
        floor_cutin = self.L_D0 + norm.ppf(0.3) * safety_std(self.COV, config)
        for gap_lead, gap_cutin, lead_v, keeps_floor in ((20.0, floor_cutin + 3.0, 8.0, True),
                                                         (14.0, floor_cutin + 0.2, 6.0, False)):
            prediction = self._two_scenarios(gap_lead, gap_cutin, lead_v, p_hyp=0.3)
            controller, solution = self._solve(config, prediction)
            terms, _ = one_row_floor_terms(prediction, config)
            gap = prediction.means[:, :, 0] - solution.x_nominal[:, :, 0]
            xi = np.maximum(0.0, self.L_D0 + terms - gap)
            np.testing.assert_allclose(solution.slack, xi, atol=1.0e-4)
            self.assertEqual(solution.slack_max < 1.0e-6, keeps_floor)
            if keeps_floor:
                short = np.maximum(0.0, self.L_D0 + self.TAU * solution.x_nominal[:, :, 1] - gap - xi)
                np.testing.assert_allclose(controller._standoff_shortfall, short, atol=1.0e-3)
                self.assertGreater(float(np.max(controller._standoff_shortfall)), 5.0)

    def test_inactive_cells_keep_no_shortfall(self):
        prediction = self._lead_only(30.0, 10.0)
        prediction.active_mask[0, 8:] = False
        controller, _ = self._solve(self._config(), prediction)
        np.testing.assert_allclose(controller._standoff_shortfall[0, 8:], 0.0, atol=1.0e-9)

    def test_without_the_token_no_floor_row(self):
        controller = NairACCSMPC(self._config(floor_sigma=False))
        problem = controller._get_feedback_scalar_chance_qp(1)
        self.assertIsNone(problem.floor_mask_param)
        self.assertIsNone(problem.floor_sigma_param)
        problem = NairACCSMPC(self._config())._get_feedback_scalar_chance_qp(1)
        self.assertIsNotNone(problem.floor_mask_param)
        self.assertIsNone(problem.floor_cell_param)

    def test_reference_keeps_plain_dsafe(self):
        ego = np.array([0.0, 10.0])
        prediction = self._two_scenarios(25.0, 20.0, 8.0, p_hyp=0.3)
        with_token = OldACCReferenceAdapter(self._config(num_modes=2)).generate(ego, prediction)
        without = OldACCReferenceAdapter(self._config(floor_sigma=False, num_modes=2)).generate(ego, prediction)
        np.testing.assert_array_equal(with_token.v_ref, without.v_ref)
        np.testing.assert_array_equal(with_token.s_ref, without.s_ref)


if __name__ == "__main__":
    unittest.main()
