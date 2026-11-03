import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    confidence_quantile,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestLkrReleaseSigma(unittest.TestCase):
    """``lkrsigma``: lkr's release test also asks for the step-0 cell's sigma term (lksigma's)."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    EGO = np.array([0.0, 14.0])
    STANDOFF = 4.5 + 3.0 + 1.3 * 14.0                       # L + d0 + tau v, centre to centre
    SIGMA_TERM = confidence_quantile(0.6) * np.sqrt(0.01 + 1.3 ** 2 * 0.01)   # step 0: no lead variance

    def _controller(self, release_sigma):
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=1, controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.standoff_shortfall_weight = 100.0
        config.standoff_shortfall_on_lk = True
        config.standoff_shortfall_lk_cutin_only = True
        config.standoff_shortfall_lk_release_sigma = release_sigma
        config.__post_init__()
        controller = NairACCSMPC(config)
        controller._set_command_context(0.0, self.COMMAND_DT, ego_accel=0.0, ego_speed=self.EGO[1])
        return controller

    def _prediction(self, gap0, cutin):
        """Vehicle 10 at ``gap0`` (centre gap) and 14 m/s: a cut-in cell (finite chance confidence), or the
        ego-lane lead's lane-keeping cell with lksigma's confidence 0.6; no lead variance at step 0."""
        steps = self.HORIZON + 1
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = gap0 + 14.0 * self.DT * np.arange(steps)
        means[0, :, 1] = 14.0
        covariances = np.zeros((1, steps, 2, 2))
        covariances[0, 1:] = np.diag([0.25, 0.09])
        return MultimodalLeadPrediction(
            means=means, probabilities=np.array([1.0]), covariances=covariances,
            active_mask=np.ones((1, steps), dtype=bool),
            chance_confidence=np.full((1, steps), 0.6 if cutin else np.nan),
            sigma_confidence=np.full((1, steps), np.nan if cutin else 0.6),
            lead_mode_lk=np.full((1, steps), not cutin),
            lead_vehicle_ids=np.full((1, steps), 10, dtype=int),
            mode_names=["10:cutin" if cutin else "10:lk"])

    def _released_after_hold(self, release_sigma, gap0):
        controller = self._controller(release_sigma)
        controller._restrict_lk_pricing(self.EGO, self._prediction(15.0, cutin=True))     # armed as a cut-in
        lead = self._prediction(gap0, cutin=False)
        for _ in range(int(round(NairACCSMPC.LK_RECOVERY_HOLD_S / self.COMMAND_DT))):
            controller._restrict_lk_pricing(self.EGO, lead)
        return 10 not in controller._priced_cutin_vehicles and 10 not in controller._armed_cutin_vehicles

    def test_the_sigma_term_is_the_hard_rows(self):
        self.assertAlmostEqual(self.SIGMA_TERM, 0.8416 * 0.1640, places=3)

    def test_release_needs_the_sigma_term_too(self):
        between = self.STANDOFF + 0.5 * self.SIGMA_TERM       # above d_safe, below d_safe + q sigma
        self.assertTrue(self._released_after_hold(False, between))
        self.assertFalse(self._released_after_hold(True, between))
        above = self.STANDOFF + self.SIGMA_TERM + 0.01
        self.assertTrue(self._released_after_hold(True, above))

    def test_needs_lkr(self):
        config = original_nair_acc_config(horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=1)
        config.standoff_shortfall_lk_release_sigma = True
        with self.assertRaises(ValueError):
            config.__post_init__()

    def test_policy_token(self):
        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        self.assertFalse(ACCNairSMPCAgent._parse_lk_release_sigma(base + "_lksigma"))
        self.assertTrue(ACCNairSMPCAgent._parse_lk_release_sigma(base + "_lksigma_lkrsigma"))
        self.assertFalse(ACCNairSMPCAgent._parse_lk_release_sigma(base + "_lksigma_lkrsigmax"))


if __name__ == "__main__":
    unittest.main()
