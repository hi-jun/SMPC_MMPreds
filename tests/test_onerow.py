import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    OldACCReferenceAdapter,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    confidence_quantile,
    original_nair_acc_config,
    safety_std,
    standoff_shortfall_mask,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestOneRow(unittest.TestCase):
    """``onerow``: one row on every active lead cell, ``gap >= L + d0 + tau v - short - xi`` with
    ``0 <= short <= tau v`` priced at ``w P_j short^2``; no sigma term, no floor row, no hard lane-keeping row; the
    reference asks every cell for the plain ``d_safe`` at the planned speed."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    TAU = 1.3
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])
    # the NGSIM arm's string (onerow_20261112) without the NGSIM-only tokens bcts and occin0.1, which the agent refuses
    AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3_trajcutin_ctx0_latgate"
             "_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold"
             "_entrant2_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow")

    def _config(self, one_row=True, num_modes=2, **overrides):
        """The NGSIM arm's cost and limits (dopt3, jinput, r_a 0, rjerk40, l1slack1e6, ISO limits, safelayer)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=num_modes,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 3.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
        config.failsafe_recovery = True
        config.failsafe_decel, config.failsafe_jerk = -8.0, 10.0
        config.one_row = one_row
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap_lead, gap_cutin, lead_v, p_hyp=0.3, lead_scale=1.0):
        """Scenario 0: the ego-lane lead's lane-keeping cells (lksigma's confidence 0.6, p 1) ``gap_lead`` ahead;
        scenario 1: a cut-in hypothesis with probability ``p_hyp`` ``gap_cutin`` ahead.  Both at ``lead_v``.
        ``lead_scale`` is the lead cells' factor (1 under nocutout)."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[0, :, 0] = gap_lead + lead_v * self.DT * np.arange(steps)
        means[1, :, 0] = gap_cutin + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        confidence = np.full((2, steps), np.nan)
        confidence[1] = min(p_hyp, 0.6)
        sigma_confidence = np.full((2, steps), np.nan)
        sigma_confidence[0] = 0.6
        lead_p = np.ones((2, steps))
        lead_p[1] = p_hyp
        scale = np.ones((2, steps))
        scale[0] = lead_scale
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0 - p_hyp, p_hyp]),
            covariances=np.tile(self.COV, (2, steps, 1, 1)),
            active_mask=np.ones((2, steps), dtype=bool),
            chance_confidence=confidence,
            sigma_confidence=sigma_confidence,
            lead_mode_lk=np.array([[True] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
            clearance_scale=scale,
            lead_mode_probability=lead_p,
            mode_names=["10:lk", "11:cutin"],
        )

    def _lead_only(self, gap0, lead_v):
        """The ego-lane lead's lane-keeping cells alone (vehicle 10, p 1, lksigma's confidence 0.6)."""
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[0, :, 1] = lead_v
        return MultimodalLeadPrediction(
            means=means, probabilities=np.array([1.0]), covariances=np.tile(self.COV, (1, steps, 1, 1)),
            active_mask=np.ones((1, steps), dtype=bool),
            chance_confidence=np.full((1, steps), np.nan),
            sigma_confidence=np.full((1, steps), 0.6),
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

    def _agent(self, policy):
        return ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                                smpc_config=policy)

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_one_row
        self.assertTrue(parse(self.AGENT))
        self.assertTrue(parse("onerow_ra0"))
        self.assertFalse(parse(self.AGENT.replace("_onerow", "")))
        self.assertFalse(parse("acc_x_xonerow"))
        self.assertFalse(parse("acc_x_onerowx"))
        # the arm's dopt token is plain dopt<w>
        self.assertEqual(ACCNairSMPCAgent._parse_standoff_shortfall_weight(self.AGENT), 3.0)
        self.assertFalse(ACCNairSMPCAgent._parse_standoff_shortfall_on_lk(self.AGENT))

    def test_config_validation(self):
        self.assertTrue(self._config().one_row)
        bad = [
            dict(standoff_shortfall_weight=None),                           # needs dopt<w>
            dict(standoff_shortfall_on_lk=True),                            # not dopt<w>lk
            dict(standoff_shortfall_on_lk=True, standoff_shortfall_lk_cutin_only=True),   # not dopt<w>lkr
            dict(standoff_shortfall_on_lk=True, standoff_shortfall_lk_cutin_only=True,
                 standoff_shortfall_lk_release_sigma=True),                 # not lkrsigma
            dict(floor_risk=0.1),
            dict(floor_risk=0.1, floor_risk_hypothesis=True),
            dict(floor_risk=0.1, floor_risk_hypothesis=True, floor_risk_top=True),
            dict(floor_risk=0.1, price_row_no_sigma=True),
            dict(lane_keep_risk=0.05),
            dict(cutin_cvar_risk=0.5, cutin_cvar_bound=2.0),
            dict(pass_miqp_margin=0.0),
            dict(relative_brake_decel=3.0),
            dict(failsafe_branch=True),
        ]
        for overrides in bad:
            with self.assertRaises(ValueError, msg=str(overrides)) as caught:
                self._config(**overrides)
            self.assertIn("one_row", str(caught.exception), msg=str(overrides))

    def test_agent_combinations(self):
        # every refused combination raises in the agent before a predictor is built
        refused = {
            "_dopt3_": "_dopt3lk_", "_fullfoot_": "_",
        }
        for old, new in refused.items():
            with self.assertRaises(ValueError, msg=new) as caught:
                self._agent(self.AGENT.replace(old, new))
            self.assertIn("onerow", str(caught.exception))
        with self.assertRaises(ValueError):          # dopt<w>lkr (also refused through lk)
            self._agent(self.AGENT.replace("_dopt3_", "_dopt3lkr_"))
        with self.assertRaises(ValueError):          # no dopt<w> at all (fullfoot refuses first)
            self._agent(self.AGENT.replace("_dopt3_", "_"))
        added = ["_floorrisk0.1", "_floorrisk0.1_floorhyp", "_floorrisk0.1_floorhyp_nopricesigma", "_lksigma",
                 "_lksigma_lkonesided0.05", "_lksigma_lkrsigma", "_cvar0.5_2", "_passmiqp", "_relbrake3",
                 "_safebranch"]
        for tokens in added:
            with self.assertRaises(ValueError, msg=tokens) as caught:
                self._agent(self.AGENT + tokens)
            self.assertIn("onerow", str(caught.exception), msg=tokens)

    def test_rows_have_no_sigma_and_every_active_cell_is_priced(self):
        prediction = self._prediction(80.0, 30.0, 10.0, lead_scale=0.7)
        prediction.active_mask[1, :3] = False          # the cut-in enters at step 3
        controller = NairACCSMPC(self._config())
        for mode in range(2):
            for step in range(self.HORIZON + 1):
                self.assertEqual(controller._cell_tightening(prediction, None, mode, step), 0.0)
        np.testing.assert_array_equal(controller._priced_cells(prediction), prediction.active_mask)
        # the factor on L + d0 is 1 on every cell, the caller's prediction untouched
        one_row = controller._one_row_prediction(prediction)
        np.testing.assert_array_equal(one_row.clearance_scale, np.ones((2, self.HORIZON + 1)))
        self.assertEqual(float(prediction.clearance_scale[0, 0]), 0.7)
        # no floor row in the QP
        self.assertIsNone(controller._get_feedback_scalar_chance_qp(2).floor_cell_param)
        # without the token the priced cells are dopt's (the cut-in chance cells, with dopt<w>lk the lead's too)
        plain = NairACCSMPC(self._config(one_row=False))
        np.testing.assert_array_equal(plain._priced_cells(prediction), standoff_shortfall_mask(prediction, False))
        lk = NairACCSMPC(self._config(one_row=False, standoff_shortfall_on_lk=True))
        np.testing.assert_array_equal(lk._priced_cells(prediction), standoff_shortfall_mask(prediction, True))
        sigma = safety_std(self.COV, self._config())
        self.assertAlmostEqual(plain._cell_tightening(prediction, None, 1, 5), confidence_quantile(0.3) * sigma,
                               places=12)

    def test_lane_keeping_cell_is_priced_with_the_cap(self):
        # the lead 2 m inside d_safe at step 0 (both at 10 m/s): the shortfall is 2 m and no slack; without the
        # token (plain dopt3) the lane-keeping row is hard, with lksigma's q(0.6) sigma, and both go to the slack
        d_safe = self.L_D0 + self.TAU * 10.0
        sigma = safety_std(self.COV, self._config(num_modes=1))
        controller, solution = self._solve(self._config(num_modes=1), self._lead_only(d_safe - 2.0, 10.0))
        self.assertLess(solution.slack_max, 1.0e-6)
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], 2.0, places=3)
        self.assertTrue(np.all(np.isfinite(solution.planned_standoff)))
        self.assertAlmostEqual(float(solution.planned_standoff[0, 0]), d_safe - 2.0, places=3)
        _, plain = self._solve(self._config(one_row=False, num_modes=1), self._lead_only(d_safe - 2.0, 10.0))
        self.assertAlmostEqual(float(plain.slack[0, 0]), 2.0 + confidence_quantile(0.6) * sigma, places=4)
        # 0.5 m inside L + d0: the shortfall sits at its cap tau v = 13 m and the slack takes the rest
        controller, solution = self._solve(self._config(num_modes=1), self._lead_only(self.L_D0 - 0.5, 10.0))
        self.assertAlmostEqual(controller._standoff_shortfall[0, 0], self.TAU * 10.0, places=3)
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.5, places=4)

    def test_cutin_cell_has_no_sigma_and_no_floor_row(self):
        sigma = safety_std(self.COV, self._config())
        # the cut-in 2 m inside d_safe at step 0: the shortfall is 2 m (with the sigma term in the row it was
        # 2 m + q(0.3) sigma)
        prediction = self._prediction(200.0, self.L_D0 + 13.0 - 2.0, 10.0, p_hyp=0.3)
        for one_row, expected in ((True, 2.0), (False, 2.0 + confidence_quantile(0.3) * sigma)):
            controller, solution = self._solve(self._config(one_row=one_row), prediction)
            self.assertLess(solution.slack_max, 1.0e-6)
            self.assertAlmostEqual(controller._standoff_shortfall[1, 0], expected, places=3)
        # 0.1 m outside L + d0: no slack (the floorrisk floor L + d0 + z sigma, z = Phi^-1(1 - 0.1 / 0.3), would
        # need 0.43 sigma = 0.24 m of slack here)
        prediction = self._prediction(200.0, self.L_D0 + 0.1, 10.0, p_hyp=0.3)
        controller, solution = self._solve(self._config(), prediction)
        self.assertLess(solution.slack_max, 1.0e-6)
        self.assertAlmostEqual(controller._standoff_shortfall[1, 0], self.TAU * 10.0 - 0.1, places=3)

    def test_reference_is_plain_dsafe_at_the_planned_speed(self):
        ego = np.array([0.0, 10.0])
        prediction = self._prediction(25.0, 20.0, 8.0, p_hyp=0.3)
        config = self._config()
        adapter = OldACCReferenceAdapter(config)
        planned_v = np.linspace(10.0, 9.0, self.HORIZON + 1)
        adapter.previous_planned_speed = planned_v
        # a remembered standoff is ignored
        adapter.remember_planned_standoff(prediction, np.full((2, self.HORIZON + 1), 3.0))
        reference = adapter.generate(ego, prediction)
        base_v = adapter._old_speed_ramp(10.0)
        planned_s = np.concatenate(([0.0], np.cumsum(planned_v[:-1] * self.DT)))
        for mode in range(2):
            for step in range(self.HORIZON + 1):
                lead_s = float(prediction.means[mode, step, 0])
                safe_gap = self.L_D0 + self.TAU * planned_v[step]
                expected = min(base_v[step], adapter._lead_limited_speed(
                    lead_s, 8.0, safe_gap, planned_s[step], base_v[step]))
                self.assertAlmostEqual(reference.v_ref[mode, step], expected, places=12)
        # without the token the same prediction asks for more (the sigma terms, and on the cut-in the remembered
        # standoff)
        old = OldACCReferenceAdapter(self._config(one_row=False))
        old.previous_planned_speed = planned_v
        old_reference = old.generate(ego, prediction)
        self.assertTrue(np.all(old_reference.v_ref[0] <= reference.v_ref[0] + 1.0e-12))
        self.assertTrue(np.any(old_reference.v_ref[0] < reference.v_ref[0] - 1.0e-6))


if __name__ == "__main__":
    unittest.main()
