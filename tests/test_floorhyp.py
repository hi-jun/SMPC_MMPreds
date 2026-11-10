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
    floor_risk_cell_quantile,
    floor_risk_cells,
    floor_risk_quantile,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestFloorRiskHypothesis(unittest.TestCase):
    """``floorhyp``: ``floorrisk<eps>``'s ``z = Phi^-1(1 - eps / p)`` from the probability ``p`` of each cell's
    own hypothesis (``lead_mode_probability``) instead of the scenario probability ``P_j``."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])

    def _config(self, risk=0.05, hypothesis=True, **overrides):
        """The current design (dopt100, jinput, r_a 0, l1slack1e6; fullfoot is the prediction's factor 1)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=3,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.floor_risk = risk
        config.floor_risk_hypothesis = hypothesis
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap_cutin, lead_v, p_hyp=0.3, split=(0.15, 0.15), beta=0.3):
        """Scenario 0: an in-lane lane-keeping lead 80 m ahead (probability 1 - sum(split)); scenarios 1 and 2:
        the same cut-in hypothesis (probability ``p_hyp`` for its vehicle, fullfoot factor 1, ``beta_j = beta``)
        ``gap_cutin`` ahead, the scenario probabilities ``split`` (it shares them with another vehicle's two
        modes).  All lead at every step, at ``lead_v``."""
        steps = self.HORIZON + 1
        means = np.zeros((3, steps, 2))
        means[0, :, 0] = 80.0 + lead_v * self.DT * np.arange(steps)
        means[1:, :, 0] = gap_cutin + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        confidence = np.full((3, steps), np.nan)
        confidence[1:] = beta
        lead_p = np.ones((3, steps))
        lead_p[1:] = p_hyp
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0 - sum(split), split[0], split[1]]),
            covariances=np.tile(self.COV, (3, steps, 1, 1)),
            active_mask=np.ones((3, steps), dtype=bool),
            chance_confidence=confidence,
            lead_mode_lk=np.array([[True] * steps, [False] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps, [11] * steps]),
            clearance_scale=np.ones((3, steps)),
            lead_mode_probability=lead_p,
            mode_names=["10:lk", "11:cutin", "11:cutin+12:cutin"],
        )

    def _solve(self, config, ego_v, prediction):
        solution = NairACCSMPC(config).solve(
            np.array([0.0, ego_v]), prediction,
            previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_floor_risk_hypothesis
        self.assertTrue(parse("acc_x_occin0.05_pingate_nolkentry_floorrisk0.05_floorhyp"))
        self.assertTrue(parse("floorhyp_ra0"))
        self.assertFalse(parse("acc_x_floorrisk0.05"))
        self.assertFalse(parse("acc_x_xfloorhyp"))
        self.assertFalse(parse("acc_x_floorhypx"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self._config(risk=None)
        self.assertFalse(self._config(hypothesis=False).floor_risk_hypothesis)

    def test_quantile_per_cell(self):
        prediction = self._prediction(30.0, 10.0)
        prediction.lead_mode_probability[2, 8:] = 0.04       # a cell whose hypothesis is at or below eps
        sigma = safety_std(self.COV, self._config())
        # the scenario quantile splits the hypothesis: P_j 0.15 -> z 0.43; its own p 0.3 -> z 0.97
        z_scen = floor_risk_quantile([0.15], 0.05)[0]
        z_hyp = floor_risk_quantile([0.3], 0.05)[0]
        self.assertAlmostEqual(z_scen, norm.ppf(1.0 - 0.05 / 0.15), places=12)
        self.assertAlmostEqual(z_hyp, norm.ppf(1.0 - 0.05 / 0.3), places=12)
        cells, hyp = floor_risk_cells(prediction, self._config())
        _, scen = floor_risk_cells(prediction, self._config(hypothesis=False))
        self.assertFalse(cells[0].any() or not cells[1:].all())
        np.testing.assert_allclose(hyp[1], z_hyp * sigma, rtol=1e-12)
        np.testing.assert_allclose(hyp[2, :8], z_hyp * sigma, rtol=1e-12)
        self.assertTrue(np.all(np.isneginf(hyp[2, 8:])))
        np.testing.assert_allclose(scen[1:], z_scen * sigma, rtol=1e-12)
        self.assertTrue(np.all(np.isnan(hyp[0])) and np.all(np.isnan(scen[0])))
        # without the token, the per-scenario value on every step, as before
        z = floor_risk_cell_quantile(prediction, self._config(hypothesis=False))
        np.testing.assert_array_equal(z, np.repeat(floor_risk_quantile(prediction.probabilities, 0.05)[:, None],
                                                   self.HORIZON + 1, axis=1))
        prediction.lead_mode_probability = None
        with self.assertRaises(ValueError):
            floor_risk_cells(prediction, self._config())
        # without a cut-in chance cell (the replay's placeholder for a tick with no target) none is needed
        prediction.chance_confidence[:] = np.nan
        cells, floor_sigma = floor_risk_cells(prediction, self._config())
        self.assertFalse(cells.any())
        self.assertTrue(np.all(np.isnan(floor_sigma)))

    def test_the_solve_keeps_the_hypothesis_floor(self):
        # ego and the cut-in lead at 10 m/s, 7.4 m apart at step 0, where the gap is fixed: the scenario floor
        # L + d0 + 0.43 sigma is met there, the hypothesis floor L + d0 + 0.97 sigma is not (the l1 slack pays)
        config = self._config()
        sigma = safety_std(self.COV, config)
        z_hyp = floor_risk_quantile([0.3], 0.05)[0]
        z_scen = floor_risk_quantile([0.15], 0.05)[0]
        gap0 = self.L_D0 + 0.7 * sigma
        prediction = self._prediction(gap0, 10.0)
        scen = self._solve(self._config(hypothesis=False), 10.0, prediction)
        hyp = self._solve(config, 10.0, prediction)
        self.assertAlmostEqual(scen.chance_margin_min, gap0 - (self.L_D0 + z_scen * sigma), places=4)
        self.assertAlmostEqual(hyp.chance_margin_min, gap0 - (self.L_D0 + z_hyp * sigma), places=4)
        self.assertLess(scen.slack_max, 1e-6)
        self.assertAlmostEqual(hyp.slack_max, (z_hyp - 0.7) * sigma, places=4)
        # q(0.3) = 0.39: the scenario quantile raises it a little, the hypothesis quantile more, on all 32 cells
        self.assertEqual((hyp.n_floor_free, hyp.n_floor_lowered, hyp.n_floor_raised), (0, 0, 32))
        self.assertLess(hyp.action, scen.action)


if __name__ == "__main__":
    unittest.main()
