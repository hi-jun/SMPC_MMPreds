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


class TestLkOneSided(unittest.TestCase):
    """``lkonesided<eps>``: a lane-keeping cell's sigma term (``lksigma``) is the one-sided ``z sigma``,
    ``z = Phi^-1(1 - eps / p)`` with the lead mode's ``p`` (1 under ``nocutout``: 1.645 at eps 0.05) and its own
    eps, in place of ``q(min(p, beta_ref)) sigma`` (0.84 sigma), in its row, the reference and lkrsigma's release
    test; cut-in cells keep theirs unless ``floortop`` is on."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    L_D0 = 4.5 + 3.0
    COV = np.diag([0.25, 0.09])
    Z = norm.ppf(1.0 - 0.05)
    Q = confidence_quantile(0.6)

    def _config(self, lane_keep=0.05, num_modes=2, **overrides):
        """The current design (dopt100lkr with lkrsigma, jinput, r_a 0, l1slack1e6, floorrisk0.05 + floorhyp)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=num_modes,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.standoff_shortfall_on_lk = True
        config.standoff_shortfall_lk_cutin_only = True
        config.standoff_shortfall_lk_release_sigma = True
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.floor_risk = 0.05
        config.floor_risk_hypothesis = True
        config.lane_keep_risk = lane_keep
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

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_lane_keep_risk
        self.assertEqual(parse("acc_x_nocutout_lksigma_floorrisk0.1_floorhyp_lkonesided0.05"), 0.05)
        self.assertEqual(parse("lkonesided0.1_ra0"), 0.1)
        self.assertIsNone(parse("acc_x_floorrisk0.05_floorhyp_floortop"))
        self.assertIsNone(parse("acc_x_lkonesided"))
        self.assertIsNone(parse("acc_x_xlkonesided0.05"))
        self.assertIsNone(parse("acc_x_lkonesided0.05x"))

    def test_config_validation(self):
        for bad in (0.0, 1.0, 1.5):
            with self.assertRaises(ValueError):
                self._config(lane_keep=bad)
        self.assertEqual(self._config().lane_keep_risk, 0.05)
        self.assertIsNone(self._config(lane_keep=None).lane_keep_risk)

    def test_own_eps(self):
        # the cut-in floor's eps (floorrisk) and the lane-keeping eps are apart
        sigma = safety_std(self.COV, self._config())
        prediction = self._prediction(80.0, 30.0, 10.0, p_hyp=0.3)
        config = self._config(lane_keep=0.05, floor_risk=0.1, floor_risk_top=True)
        self.assertAlmostEqual(cell_sigma_term(prediction, 0, 5, config), self.Z * sigma, places=12)
        self.assertAlmostEqual(cell_sigma_term(prediction, 1, 5, config), norm.ppf(1.0 - 0.1 / 0.3) * sigma,
                               places=12)

    def test_sigma_term(self):
        sigma = safety_std(self.COV, self._config())
        prediction = self._prediction(80.0, 30.0, 10.0, p_hyp=0.3)
        # the lead's lane-keeping cell: z sigma with the token, q(0.6) sigma without
        self.assertAlmostEqual(cell_sigma_term(prediction, 0, 5, self._config()), self.Z * sigma, places=12)
        self.assertAlmostEqual(cell_sigma_term(prediction, 0, 5, self._config(lane_keep=None)),
                               self.Q * sigma, places=12)
        # the cut-in cell keeps q(min(p, 0.6)) sigma unless floortop is on
        self.assertAlmostEqual(cell_sigma_term(prediction, 1, 5, self._config()),
                               confidence_quantile(0.3) * sigma, places=12)
        both = self._config(floor_risk_top=True)
        self.assertAlmostEqual(cell_sigma_term(prediction, 1, 5, both), norm.ppf(1.0 - 0.05 / 0.3) * sigma,
                               places=12)
        self.assertAlmostEqual(cell_sigma_term(prediction, 0, 5, both), self.Z * sigma, places=12)
        # a lane-keeping cell without lksigma's sigma term keeps none
        prediction.sigma_confidence[0, 5] = np.nan
        self.assertEqual(cell_sigma_term(prediction, 0, 5, self._config()), 0.0)
        prediction.sigma_confidence[0, 5] = 0.6
        prediction.lead_mode_probability[0, 5] = 0.04      # p <= eps: no one-sided quantile
        with self.assertRaises(ValueError):
            cell_sigma_term(prediction, 0, 5, self._config())

    def test_reference_asks_for_the_same_row(self):
        # the lead 20 m ahead at the ego's 10 m/s binds the reference (d_safe 20.5 m); the cut-in is far
        ego = np.array([0.0, 10.0])
        prediction = self._prediction(20.0, 200.0, 10.0, p_hyp=0.3)
        new = OldACCReferenceAdapter(self._config()).generate(ego, prediction)
        old = OldACCReferenceAdapter(self._config(lane_keep=None)).generate(ego, prediction)
        sigma = safety_std(self.COV, self._config())
        np.testing.assert_allclose(old.s_ref[0, 1:] - new.s_ref[0, 1:], (self.Z - self.Q) * sigma, rtol=1.0e-9)
        np.testing.assert_allclose(new.s_ref[1], old.s_ref[1])

    def test_the_hard_row_asks_for_z_sigma(self):
        # the lead at 10 m/s with the ego, d_safe + 1.2 sigma ahead: the step-0 row (no decision variable) holds
        # with q(0.6) = 0.84 and needs the l1 slack (z - 1.2) sigma with z = 1.645
        sigma = safety_std(self.COV, self._config(num_modes=1))
        gap0 = self.L_D0 + 1.3 * 10.0 + 1.2 * sigma
        for lane_keep, expected in ((None, 0.0), (0.05, (self.Z - 1.2) * sigma)):
            controller = NairACCSMPC(self._config(lane_keep=lane_keep, num_modes=1))
            solution = controller.solve(
                np.array([0.0, 10.0]), self._lead_only(gap0, 10.0),
                previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
            self.assertNotEqual(solution.solve_path, "fallback")
            self.assertAlmostEqual(float(solution.slack[0, 0]), expected, places=4)

    def test_lkr_release_asks_for_z_sigma(self):
        # vehicle 10 armed as a cut-in, then the lane-keeping lead at step 0 between d_safe + q sigma and
        # d_safe + z sigma (no lead variance at step 0, so sigma is the ego's process term 0.164 m)
        ego = np.array([0.0, 14.0])
        steps = self.HORIZON + 1
        sigma0 = safety_std(np.zeros((2, 2)), self._config(num_modes=1))
        gap0 = self.L_D0 + 1.3 * 14.0 + 0.5 * (self.Q + self.Z) * sigma0

        def single(gap, cutin):
            means = np.zeros((1, steps, 2))
            means[0, :, 0] = gap + 14.0 * self.DT * np.arange(steps)
            means[0, :, 1] = 14.0
            covariances = np.zeros((1, steps, 2, 2))
            covariances[0, 1:] = self.COV
            return MultimodalLeadPrediction(
                means=means, probabilities=np.array([1.0]), covariances=covariances,
                active_mask=np.ones((1, steps), dtype=bool),
                chance_confidence=np.full((1, steps), 0.6 if cutin else np.nan),
                sigma_confidence=np.full((1, steps), np.nan if cutin else 0.6),
                lead_mode_lk=np.full((1, steps), not cutin),
                lead_vehicle_ids=np.full((1, steps), 10, dtype=int),
                lead_mode_probability=np.full((1, steps), 0.6 if cutin else 1.0),
                mode_names=["10:cutin" if cutin else "10:lk"])

        released = {}
        for lane_keep in (None, 0.05):
            controller = NairACCSMPC(self._config(lane_keep=lane_keep, num_modes=1))
            controller._set_command_context(0.0, self.COMMAND_DT, ego_accel=0.0, ego_speed=ego[1])
            controller._restrict_lk_pricing(ego, single(15.0, cutin=True))
            for _ in range(int(round(NairACCSMPC.LK_RECOVERY_HOLD_S / self.COMMAND_DT))):
                controller._restrict_lk_pricing(ego, single(gap0, cutin=False))
            released[lane_keep] = 10 not in controller._priced_cutin_vehicles
        self.assertTrue(released[None])
        self.assertFalse(released[0.05])


if __name__ == "__main__":
    unittest.main()
