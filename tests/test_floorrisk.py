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
    SAFETY_NOMINAL_SAFE_DISTANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    confidence_quantile,
    floor_risk_cells,
    floor_risk_quantile,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestFloorRisk(unittest.TestCase):
    """``floorrisk<eps>``: a cut-in chance cell's hard floor ``L + d0 + q(beta_j) sigma`` replaced by the
    chance constraint ``P_j P(gap < L + d0) <= eps``, i.e. ``L + d0 + z_j sigma`` with
    ``z_j = Phi^-1(1 - eps / P_j)`` (no floor at ``P_j <= eps``), the priced row uncapped there."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    L_D0 = 4.5 + 3.0

    def _config(self, risk=0.2, **overrides):
        """The current design (dopt100, jinput, r_a 0, l1slack1e6; fullfoot is the prediction's factor 1)
        with or without the token."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=2,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.floor_risk = risk
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap_lk, gap_cutin, lead_v, p_cutin=0.4, beta=0.4):
        """Mode 0: an in-lane lane-keeping lead (hard standoff) ``gap_lk`` ahead; mode 1: a cut-in chance
        cell (fullfoot: factor 1, ``beta_j = beta``) ``gap_cutin`` ahead, probability ``p_cutin``.  Both lead
        at every step, at ``lead_v``."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[0, :, 0] = gap_lk + lead_v * self.DT * np.arange(steps)
        means[1, :, 0] = gap_cutin + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        confidence = np.full((2, steps), np.nan)
        confidence[1] = beta
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0 - p_cutin, p_cutin]),
            covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)),
            active_mask=np.ones((2, steps), dtype=bool),
            chance_confidence=confidence,
            lead_mode_lk=np.array([[True] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
            clearance_scale=np.ones((2, steps)),
            mode_names=["10:lk", "11:cutin"],
        )

    def _solve(self, config, ego_v, prediction):
        controller = NairACCSMPC(config)
        solution = controller.solve(
            np.array([0.0, ego_v]), prediction,
            previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return controller, solution

    def _sigma(self, config):
        return safety_std(np.diag([0.25, 0.09]), config)

    @staticmethod
    def _gaps(solution, prediction, mode):
        return prediction.means[mode, :, 0] - solution.x_nominal[mode, :, 0]

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_floor_risk
        self.assertEqual(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0_fullfoot_l1slack1e6_safelayer_floorrisk0.2"),
                         0.2)
        self.assertEqual(parse("acc_x_floorrisk0.05_safelayer"), 0.05)
        self.assertEqual(parse("acc_x_floorrisk.5"), 0.5)
        self.assertIsNone(parse("acc_x_xfloorrisk0.2"))
        self.assertIsNone(parse("acc_x_floorrisk0.2x_ra0"))
        self.assertIsNone(parse("acc_x_floorrisk_ra0"))
        self.assertIsNone(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0_fullfoot_l1slack1e6_safelayer"))

    def test_config_validation(self):
        for risk in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                self._config(risk=risk)
        with self.assertRaises(ValueError):        # needs dopt<w>
            self._config(standoff_shortfall_weight=None)
        with self.assertRaises(ValueError):        # needs cutin_chance (the confidence chance rows)
            self._config(standoff_shortfall_weight=None, safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE)
        with self.assertRaises(ValueError):        # not with relbrake<b>
            self._config(relative_brake_decel=3.0)
        with self.assertRaises(ValueError):        # cached feedback QP only
            self._config(controller_variant=VARIANT_MULTIMODAL_OL, standoff_shortfall_weight=None,
                         jerk_input=False, slack_l1_weight=None)

    def test_quantile_and_removal(self):
        # z_j = Phi^-1(1 - eps / P_j); the prereg's eps 0.2 column: no floor at P <= 0.2, -0.43 / 0.25 / 0.43 / 0.84
        z = floor_risk_quantile([0.1, 0.2, 0.3, 0.5, 0.6, 1.0], 0.2)
        self.assertTrue(np.all(np.isneginf(z[:2])))
        np.testing.assert_allclose(z[2:], [-0.4307, 0.2533, 0.4307, 0.8416], atol=1e-4)
        # the chance constraint holds with equality at the floor: P_j P(gap < L + d0) = P_j (1 - Phi(z)) = eps
        np.testing.assert_allclose(np.array([0.3, 0.5, 0.6, 1.0]) * (1.0 - norm.cdf(z[2:])), 0.2, rtol=1e-12)
        # per cell: z_j times the sigma of the row's sigma term, on the cut-in chance cells only
        config = self._config(risk=0.2)
        prediction = self._prediction(30.0, 30.0, 10.0, p_cutin=0.3, beta=0.3)
        prediction.active_mask[1, :4] = False
        cells, floor_sigma = floor_risk_cells(prediction, config)
        self.assertTrue(np.all(cells[1, 4:]) and not np.any(cells[1, :4]) and not np.any(cells[0]))
        self.assertTrue(np.all(np.isnan(floor_sigma[~cells])))
        np.testing.assert_allclose(floor_sigma[1, 4:], z[2] * self._sigma(config), rtol=1e-12)
        # a scenario at P_j <= eps keeps no floor
        prediction = self._prediction(30.0, 30.0, 10.0, p_cutin=0.2, beta=0.2)
        cells, floor_sigma = floor_risk_cells(prediction, config)
        self.assertTrue(np.all(cells[1]) and np.all(np.isneginf(floor_sigma[1])))

    def test_without_the_token_nothing_is_added(self):
        config = self._config(risk=None)
        self.assertIsNone(config.floor_risk)
        prediction = self._prediction(80.0, 10.0, 10.0)
        controller, solution = self._solve(config, 13.0, prediction)
        problem = next(iter(controller._feedback_scalar_chance_qp_cache.values()))
        self.assertIsNone(problem.floor_cell_param)
        self.assertIsNone(problem.floor_mask_param)
        self.assertIsNone(problem.floor_sigma_param)
        self.assertEqual((solution.n_floor_free, solution.n_floor_lowered, solution.n_floor_raised), (0, 0, 0))
        # the margin to the (old) floor is reported either way: here the floor binds through the slack
        self.assertAlmostEqual(solution.floor_old_margin_min, -solution.slack_max, places=6)
        self.assertGreater(solution.slack_max, 0.1)

    def test_qp_key_and_parameters_carry_the_risk(self):
        prediction = self._prediction(80.0, 10.0, 10.0)
        controller_on, _ = self._solve(self._config(risk=0.3), 13.0, prediction)
        controller_off, _ = self._solve(self._config(risk=None), 13.0, prediction)
        key_on = next(iter(controller_on._feedback_scalar_chance_qp_cache))
        key_off = next(iter(controller_off._feedback_scalar_chance_qp_cache))
        differ = [i for i, (a, b) in enumerate(zip(key_on, key_off)) if a != b]
        self.assertEqual(len(key_on), len(key_off))
        self.assertEqual(len(differ), 1)
        self.assertEqual(key_on[differ[0]], 0.3)
        self.assertIsNone(key_off[differ[0]])
        problem = next(iter(controller_on._feedback_scalar_chance_qp_cache.values()))
        self.assertIsNotNone(problem.floor_cell_param)
        self.assertIsNotNone(problem.floor_mask_param)
        self.assertIsNotNone(problem.floor_sigma_param)

    def test_matching_quantile_reproduces_the_design(self):
        # eps = P_j (1 - beta_j) / 2 makes z_j = q(beta_j): P 0.5, beta 0.5, eps 0.125 (exact in binary).  The
        # two-row form then has the design's feasible set and optimum; with Gurobi's tolerances tightened
        # from the config's 1e-4 the two solves agree to the solver's precision, floor binding or not.
        for ego_v, gap in ((13.0, 10.0), (11.0, 8.5)):
            prediction = self._prediction(80.0, gap, 10.0, p_cutin=0.5, beta=0.5)
            _, design = self._solve(self._config(risk=None, optimizer_ftol=1e-9), ego_v, prediction)
            _, token = self._solve(self._config(risk=0.125, optimizer_ftol=1e-9), ego_v, prediction)
            self.assertEqual((token.n_floor_free, token.n_floor_lowered, token.n_floor_raised), (0, 0, 0))
            np.testing.assert_allclose(token.u_nominal, design.u_nominal, atol=1e-5)
            np.testing.assert_allclose(token.x_nominal, design.x_nominal, atol=1e-6)
            np.testing.assert_allclose(token.slack, design.slack, atol=1e-8)
            self.assertAlmostEqual(token.action, design.action, places=7)
            self.assertAlmostEqual(token.cost / design.cost, 1.0, places=8)
            # the cut-in cells' chance margin is now the margin to that (same) floor
            self.assertAlmostEqual(token.chance_margin_min, token.floor_old_margin_min, places=6)
        # at the config's own tolerance the slack (the floor binds in this case) and the command still agree
        _, design = self._solve(self._config(risk=None), 13.0, self._prediction(80.0, 10.0, 10.0, p_cutin=0.5, beta=0.5))
        _, token = self._solve(self._config(risk=0.125), 13.0, self._prediction(80.0, 10.0, 10.0, p_cutin=0.5, beta=0.5))
        self.assertGreater(design.slack_max, 0.1)
        np.testing.assert_allclose(token.slack, design.slack, atol=1e-4)
        self.assertAlmostEqual(token.action, design.action, places=5)

    def test_no_floor_below_the_risk(self):
        # P 0.4 <= eps 0.5: ego and the cut-in lead at 10 m/s, 7.0 m apart, below the old floor
        # L + d0 + q(0.4) sigma = 7.78 m at step 0, where the gap is fixed
        config = self._config(risk=0.5)
        prediction = self._prediction(80.0, 7.0, 10.0)
        old_floor = self.L_D0 + confidence_quantile(0.4) * self._sigma(config)
        _, design = self._solve(self._config(risk=None), 10.0, prediction)
        _, token = self._solve(config, 10.0, prediction)
        self.assertEqual(token.n_floor_free, self.HORIZON + 1)
        # the design pays the violation with the emergency slack; the token needs none
        self.assertAlmostEqual(design.slack_max, old_floor - 7.0, places=4)
        self.assertLess(token.slack_max, 1e-6)
        self.assertTrue(token.feasible)
        # the plan sits below the old floor (reported by the margin), with the price covering the whole
        # deficit: the shortfall exceeds its old cap tau v, so the planned standoff is below L + d0
        self.assertAlmostEqual(token.floor_old_margin_min, 7.0 - old_floor, places=4)
        self.assertLess(token.floor_old_margin_min, 0.0)
        tau_v0 = config.time_headway * 10.0
        short0 = self.L_D0 + tau_v0 - token.planned_standoff[1, 0]
        self.assertAlmostEqual(short0, old_floor + tau_v0 - 7.0, places=3)
        self.assertGreater(short0, tau_v0)
        self.assertLess(np.nanmin(token.planned_standoff[1]), self.L_D0)
        # the lane-keeping cell is untouched and is now the only one in the chance margin (the free cells are inf)
        self.assertTrue(np.all(np.isnan(token.planned_standoff[0])))
        self.assertGreater(token.chance_margin_min, 50.0)

    def test_lowered_floor(self):
        # eps 0.2, P 0.3, beta 0.3: z = -0.43 < q(0.3) = 0.39, the floor 7.27 m against 7.70 m
        config = self._config(risk=0.2)
        prediction = self._prediction(80.0, 7.4, 10.0, p_cutin=0.3, beta=0.3)
        z = floor_risk_quantile([0.3], 0.2)[0]
        self.assertLess(z, confidence_quantile(0.3))
        _, design = self._solve(self._config(risk=None), 10.0, prediction)
        _, token = self._solve(config, 10.0, prediction)
        self.assertEqual(token.n_floor_lowered, self.HORIZON + 1)
        self.assertGreater(design.slack_max, 0.25)        # 7.70 - 7.4
        self.assertLess(token.slack_max, 1e-6)
        # the margin to the imposed floor at step 0 is 7.4 - 7.27, and the plan brakes less at once
        self.assertAlmostEqual(token.chance_margin_min, 7.4 - (self.L_D0 + z * self._sigma(config)), places=4)
        self.assertGreater(token.action, design.action + 0.05)

    def test_raised_floor(self):
        # eps 0.01, P 0.4: z = 1.96 > q(0.4) = 0.52, the floor 8.53 m against 7.78 m
        config = self._config(risk=0.01)
        sigma = self._sigma(config)
        z = floor_risk_quantile([0.4], 0.01)[0]
        new_floor = self.L_D0 + z * sigma
        self.assertAlmostEqual(z, 1.96, places=2)
        # ego 12 m/s closing on the cut-in lead at 10 m/s, 9.5 m ahead: both solves brake at the jerk limit
        # and the gap bottoms out at 8.28 m, above the old floor and below the raised one, which the
        # exact penalty then reports as the slack
        prediction = self._prediction(80.0, 9.5, 10.0)
        _, design = self._solve(self._config(risk=None), 12.0, prediction)
        _, token = self._solve(config, 12.0, prediction)
        self.assertEqual(token.n_floor_raised, self.HORIZON + 1)
        self.assertLess(design.slack_max, 1e-6)
        self.assertGreater(design.floor_old_margin_min, 0.4)
        min_gap = float(np.min(self._gaps(token, prediction, 1)))
        self.assertGreater(token.slack_max, 0.2)
        self.assertAlmostEqual(token.slack_max, new_floor - min_gap, places=3)
        self.assertAlmostEqual(token.chance_margin_min, -token.slack_max, places=3)
        np.testing.assert_allclose(self._gaps(token, prediction, 1), self._gaps(design, prediction, 1), atol=1e-2)
        # closing at 1 m/s from 9.0 m the raised floor holds without slack, and the cut-in cells' margin
        # is the old one less (z - q) sigma
        prediction = self._prediction(80.0, 9.0, 10.0)
        _, token = self._solve(config, 11.0, prediction)
        self.assertLess(token.slack_max, 1e-6)
        self.assertGreaterEqual(float(np.min(self._gaps(token, prediction, 1))), new_floor - 1e-6)
        self.assertAlmostEqual(token.chance_margin_min,
                               token.floor_old_margin_min - (z - confidence_quantile(0.4)) * sigma, places=4)

    def test_lead_only_is_unaffected(self):
        # two lane-keeping leads, no cut-in chance cell: the token adds rows that read slack >= 0 and leaves
        # the cap rows as they are, so the solve is the same
        prediction = self._prediction(30.0, 30.0, 10.0)
        prediction.chance_confidence[:] = np.nan
        prediction.lead_mode_lk[:] = True
        _, design = self._solve(self._config(risk=None), 15.0, prediction)
        _, token = self._solve(self._config(risk=0.2), 15.0, prediction)
        np.testing.assert_array_equal(token.u_nominal, design.u_nominal)
        self.assertEqual(token.action, design.action)
        self.assertEqual((token.n_floor_free, token.n_floor_lowered, token.n_floor_raised), (0, 0, 0))
        self.assertTrue(np.isnan(token.floor_old_margin_min))

    def test_other_priced_cells_keep_their_cap(self):
        # dopt<w>lk prices the lane-keeping cell too; with the token its shortfall still stops at tau v
        # (planned standoff >= L + d0, the 0.5 m it lacks at step 0 paid by the slack), while the free
        # cut-in cell at the same 7.0 m goes below without slack
        config = self._config(risk=0.5, standoff_shortfall_on_lk=True)
        prediction = self._prediction(7.0, 7.0, 10.0)
        _, token = self._solve(config, 10.0, prediction)
        self.assertEqual(token.n_floor_free, self.HORIZON + 1)
        self.assertGreaterEqual(float(np.nanmin(token.planned_standoff[0])), self.L_D0 - 1e-6)
        self.assertAlmostEqual(float(token.slack[0, 0]), self.L_D0 - 7.0, places=4)
        self.assertLess(float(np.nanmin(token.planned_standoff[1])), self.L_D0 - 0.1)
        self.assertLess(float(np.max(token.slack[1])), 1e-6)


if __name__ == "__main__":
    unittest.main()
