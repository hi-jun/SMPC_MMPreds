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
    OldACCReferenceAdapter,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    confidence_quantile,
    floor_risk_cells,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestNoPriceSigma(unittest.TestCase):
    """``nopricesigma``: a priced cell asks for its standoff D on the mean gap (no sigma term) in its priced row and
    in the reference; its floor keeps the sigma term in its own hard row, the floorrisk floor ``L + d0 + z sigma``
    on a cut-in cell and ``L + d0`` plus the cell's own sigma term on a priced lane-keeping cell."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])

    def _config(self, no_sigma=True, num_modes=2, **overrides):
        """The current design's cost and floor (dopt100, jinput, r_a 0, l1slack1e6, floorrisk0.1 + floorhyp)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=num_modes,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.floor_risk = 0.1
        config.floor_risk_hypothesis = True
        config.price_row_no_sigma = no_sigma
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap_lead, gap_cutin, lead_v, p_hyp=0.3):
        """Scenario 0: the ego-lane lead's lane-keeping cells (lksigma's confidence 0.6, p 1) ``gap_lead`` ahead;
        scenario 1: a cut-in hypothesis with probability ``p_hyp`` ``gap_cutin`` ahead.  Both at ``lead_v``."""
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
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0 - p_hyp, p_hyp]),
            covariances=np.tile(self.COV, (2, steps, 1, 1)),
            active_mask=np.ones((2, steps), dtype=bool),
            chance_confidence=confidence,
            sigma_confidence=sigma_confidence,
            lead_mode_lk=np.array([[True] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
            clearance_scale=np.ones((2, steps)),
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

    def _solve(self, config, prediction):
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, 10.0]), prediction,
                                    previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return controller, solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_price_row_no_sigma
        self.assertTrue(parse("acc_x_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05"))
        self.assertTrue(parse("nopricesigma_ra0"))
        self.assertFalse(parse("acc_x_floorrisk0.1_floorhyp"))
        self.assertFalse(parse("acc_x_xnopricesigma"))
        self.assertFalse(parse("acc_x_nopricesigmax"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self._config(floor_risk=None, floor_risk_hypothesis=False)
        self.assertTrue(self._config().price_row_no_sigma)

    def test_row_sigma_term(self):
        # priced cells drop the sigma term in their row; the hard lane-keeping cell keeps it
        prediction = self._prediction(80.0, 30.0, 10.0)
        sigma = safety_std(self.COV, self._config())
        controller = NairACCSMPC(self._config())
        self.assertEqual(controller._cell_tightening(prediction, None, 1, 5), 0.0)
        self.assertAlmostEqual(controller._cell_tightening(prediction, None, 0, 5),
                               confidence_quantile(0.6) * sigma, places=12)
        old = NairACCSMPC(self._config(no_sigma=False))
        self.assertAlmostEqual(old._cell_tightening(prediction, None, 1, 5), confidence_quantile(0.3) * sigma,
                               places=12)
        # with dopt<w>lk the lane-keeping cells are priced too: no sigma term in the row, a floor row with it
        priced_lk = NairACCSMPC(self._config(standoff_shortfall_on_lk=True))
        self.assertEqual(priced_lk._cell_tightening(prediction, None, 0, 5), 0.0)
        cells, floor_sigma = floor_risk_cells(prediction, self._config(standoff_shortfall_on_lk=True))
        self.assertTrue(cells[0, 5] and cells[1, 5])
        self.assertAlmostEqual(floor_sigma[0, 5], confidence_quantile(0.6) * sigma, places=12)
        self.assertAlmostEqual(floor_sigma[1, 5], norm.ppf(1.0 - 0.1 / 0.3) * sigma, places=12)
        cells_off, _ = floor_risk_cells(prediction, self._config(no_sigma=False, standoff_shortfall_on_lk=True))
        self.assertFalse(cells_off[0, 5])

    def test_reference_asks_for_the_mean_gap(self):
        # the cut-in 20 m ahead at the ego's 10 m/s binds the reference (d_safe 20.5 m); the lead is far
        ego = np.array([0.0, 10.0])
        prediction = self._prediction(200.0, 20.0, 10.0, p_hyp=0.3)
        new = OldACCReferenceAdapter(self._config()).generate(ego, prediction)
        old = OldACCReferenceAdapter(self._config(no_sigma=False)).generate(ego, prediction)
        sigma = safety_std(self.COV, self._config())
        np.testing.assert_allclose(new.s_ref[1, 1:] - old.s_ref[1, 1:], confidence_quantile(0.3) * sigma,
                                   rtol=1.0e-9)
        np.testing.assert_allclose(new.s_ref[0], old.s_ref[0])

    def test_the_cutin_shortfall_is_measured_on_the_mean_gap(self):
        # the cut-in 2 m inside d_safe at step 0 (both at 10 m/s, the floor holds): the shortfall is 2 m, against
        # 2 m + q(0.3) sigma with the sigma term in the row
        sigma = safety_std(self.COV, self._config())
        prediction = self._prediction(200.0, self.L_D0 + 13.0 - 2.0, 10.0, p_hyp=0.3)
        for no_sigma, expected in ((True, 2.0), (False, 2.0 + confidence_quantile(0.3) * sigma)):
            controller, solution = self._solve(self._config(no_sigma=no_sigma), prediction)
            self.assertLess(solution.slack_max, 1e-6)
            self.assertAlmostEqual(controller._standoff_shortfall[1, 0], expected, places=3)

    def test_priced_lane_keeping_floor_keeps_its_sigma(self):
        # dopt<w>lk prices the lead's lane-keeping cells; the floor stays L + d0 + q(0.6) sigma (with nopricesigma
        # in its own row, without it through the time-gap cap), the top drops the sigma term
        sigma = safety_std(self.COV, self._config(num_modes=1))
        sigma_term = confidence_quantile(0.6) * sigma
        top_gap = self.L_D0 + 13.0 - 1.0
        floor_gap = self.L_D0 + 0.5 * sigma_term
        for no_sigma in (True, False):
            config = self._config(no_sigma=no_sigma, num_modes=1, standoff_shortfall_on_lk=True)
            controller, solution = self._solve(config, self._lead_only(top_gap, 10.0))
            self.assertLess(solution.slack_max, 1e-6)
            self.assertAlmostEqual(controller._standoff_shortfall[0, 0], 1.0 + (0.0 if no_sigma else sigma_term),
                                   places=3)
            _, solution = self._solve(config, self._lead_only(floor_gap, 10.0))
            self.assertAlmostEqual(float(solution.slack[0, 0]), 0.5 * sigma_term, places=4)


if __name__ == "__main__":
    unittest.main()
