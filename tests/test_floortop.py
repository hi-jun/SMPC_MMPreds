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
    cell_sigma_term,
    confidence_quantile,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestFloorTop(unittest.TestCase):
    """``floortop``: with ``floorhyp``, a cut-in chance cell's sigma term in its priced row and in the reference is
    the floor's one-sided ``z sigma``, ``z = Phi^-1(1 - eps / p)``, in place of ``q(beta) sigma``, so the top of the
    priced range is the floor plus the time gap; the lead's lane-keeping cells keep ``q(beta) sigma``."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])

    def _config(self, top=True, **overrides):
        """The current design (dopt100, jinput, r_a 0, l1slack1e6, floorrisk0.05 + floorhyp; fullfoot is the
        prediction's factor 1)."""
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
        config.floor_risk = 0.05
        config.floor_risk_hypothesis = True
        config.floor_risk_top = top
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap_cutin, lead_v, p_hyp=0.3):
        """Scenario 0: the ego-lane lead's lane-keeping cells 80 m ahead (lksigma's confidence 0.6, hard);
        scenario 1: a cut-in hypothesis with probability ``p_hyp`` (fullfoot factor 1, beta = min(p, 0.6))
        ``gap_cutin`` ahead.  Both lead at every step, at ``lead_v``."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[0, :, 0] = 80.0 + lead_v * self.DT * np.arange(steps)
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

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_floor_risk_top
        self.assertTrue(parse("acc_x_pingate_floorrisk0.05_floorhyp_floortop"))
        self.assertTrue(parse("floortop_ra0"))
        self.assertFalse(parse("acc_x_floorrisk0.05_floorhyp"))
        self.assertFalse(parse("acc_x_xfloortop"))
        self.assertFalse(parse("acc_x_floortopx"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self._config(floor_risk_hypothesis=False)
        self.assertTrue(self._config().floor_risk_top)
        self.assertFalse(self._config(top=False).floor_risk_top)

    def test_sigma_term(self):
        sigma = safety_std(self.COV, self._config())
        for p in (0.06, 0.1, 0.3, 0.9):
            prediction = self._prediction(30.0, 10.0, p_hyp=p)
            self.assertAlmostEqual(cell_sigma_term(prediction, 1, 5, self._config()),
                                   norm.ppf(1.0 - 0.05 / p) * sigma, places=12)
            self.assertAlmostEqual(cell_sigma_term(prediction, 1, 5, self._config(top=False)),
                                   confidence_quantile(min(p, 0.6)) * sigma, places=12)
            for config in (self._config(), self._config(top=False)):
                self.assertAlmostEqual(cell_sigma_term(prediction, 0, 5, config),
                                       confidence_quantile(0.6) * sigma, places=12)
        # below 2 eps the one-sided quantile is negative: the top falls below d_safe as the floor below L + d0
        self.assertLess(cell_sigma_term(self._prediction(30.0, 10.0, p_hyp=0.06), 1, 5, self._config()), 0.0)
        prediction = self._prediction(30.0, 10.0)
        prediction.lead_mode_probability[1, 5] = 0.04      # p <= eps: no one-sided quantile
        with self.assertRaises(ValueError):
            cell_sigma_term(prediction, 1, 5, self._config())

    def test_reference_asks_for_the_same_top(self):
        # the cut-in 20 m ahead at the ego's 10 m/s binds the reference (d_safe 20.5 m)
        ego = np.array([0.0, 10.0])
        prediction = self._prediction(20.0, 10.0, p_hyp=0.3)
        top = OldACCReferenceAdapter(self._config()).generate(ego, prediction)
        old = OldACCReferenceAdapter(self._config(top=False)).generate(ego, prediction)
        sigma = safety_std(self.COV, self._config())
        shift = (norm.ppf(1.0 - 0.05 / 0.3) - confidence_quantile(0.3)) * sigma
        np.testing.assert_allclose(old.s_ref[1, 1:] - top.s_ref[1, 1:], shift, rtol=1.0e-9)
        np.testing.assert_allclose(top.s_ref[0], old.s_ref[0])

    def test_the_solve_prices_from_the_floor_plus_tau_v(self):
        # ego and the cut-in at 10 m/s, 2 m inside d_safe at step 0, where the gap is fixed and both floors hold:
        # the step-0 shortfall is the top minus the gap, 2 m plus the sigma term of each setting
        sigma = safety_std(self.COV, self._config())
        gap0 = self.L_D0 + 1.3 * 10.0 - 2.0
        prediction = self._prediction(gap0, 10.0, p_hyp=0.3)
        expected = {True: norm.ppf(1.0 - 0.05 / 0.3), False: confidence_quantile(0.3)}
        for top, z in expected.items():
            controller = NairACCSMPC(self._config(top=top))
            solution = controller.solve(
                np.array([0.0, 10.0]), prediction,
                previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
            self.assertNotEqual(solution.solve_path, "fallback")
            self.assertLess(solution.slack_max, 1e-6)
            self.assertAlmostEqual(controller._standoff_shortfall[1, 0], 2.0 + z * sigma, places=3)


if __name__ == "__main__":
    unittest.main()
