import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    RISK_PROBABILITY_WEIGHTED,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestL1SlackPenalty(unittest.TestCase):
    """``l1slack<w>``: the emergency slack costs ``w * sum(slack)`` instead of
    ``slack_weight * sum(slack^2)``, an exact penalty."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    W = 1.0e6

    def _config(self, weight=W, **overrides):
        """The design (dopt100, jinput, r_a 0) with or without the token."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=2,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        config.slack_l1_weight = weight
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap0, lead_v):
        """Mode 0: an in-lane lane-keeping lead (hard standoff); mode 1: a cut-in chance cell (dopt
        cell, factor 1).  Both lead at every step, at ``lead_v``."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[:, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        confidence = np.full((2, steps), np.nan)
        confidence[1] = 0.4
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([0.6, 0.4]),
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
        return controller, solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_slack_l1_weight
        self.assertEqual(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0_fullfoot_l1slack1e6"),
                         1.0e6)
        self.assertEqual(parse("acc_x_l1slack100000_jinput"), 1.0e5)
        self.assertEqual(parse("acc_x_l1slack2.5e5_ra0"), 2.5e5)
        self.assertIsNone(parse("acc_x_xl1slack1e6"))
        self.assertIsNone(parse("acc_x_l1slack1e6x_ra0"))
        self.assertIsNone(parse("acc_x_l1slack_ra0"))
        self.assertIsNone(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0_fullfoot"))

    def test_without_the_token_nothing_changes(self):
        config = self._config(weight=None)
        self.assertIsNone(config.slack_l1_weight)
        controller, solution = self._solve(config, 15.0, self._prediction(gap0=30.0, lead_v=10.0))
        self.assertNotEqual(solution.solve_path, "fallback")
        key = next(iter(controller._feedback_scalar_chance_qp_cache))
        self.assertIsNone(key[-1])
        # the quadratic cost leaves a little slack on the binding rows of a problem the hard rows can hold
        self.assertGreater(solution.slack_max, 0.03)

    def test_slack_is_reported_per_cell(self):
        prediction = self._prediction(gap0=36.0, lead_v=10.0)
        prediction.active_mask[1, :6] = False
        _, solution = self._solve(self._config(), 18.0, prediction)
        self.assertEqual(solution.slack.shape, (2, self.HORIZON + 1))
        self.assertTrue(np.all(solution.slack[1, :6] == 0.0))
        self.assertAlmostEqual(solution.slack_max, float(np.max(solution.slack[prediction.active_mask])), places=12)

    def test_no_slack_while_the_hard_rows_have_a_solution(self):
        # ego 15 m/s behind leads at 10 m/s, 30 m ahead: braking to about -2.3 m/s^2 keeps every row
        prediction = self._prediction(gap0=30.0, lead_v=10.0)
        _, quadratic = self._solve(self._config(weight=None), 15.0, prediction)
        _, exact = self._solve(self._config(), 15.0, prediction)
        self.assertNotEqual(exact.solve_path, "fallback")
        self.assertGreater(quadratic.slack_max, 0.03)
        self.assertLess(exact.slack_max, 1e-4)      # the solver tolerance
        self.assertTrue(exact.feasible)
        # what the quadratic cost bought with its slack, the exact one pays in braking
        self.assertLess(np.min(exact.u_nominal), np.min(quadratic.u_nominal) - 0.01)
        # and the solution does not depend on the weight above the multipliers
        _, larger = self._solve(self._config(weight=10.0 * self.W), 15.0, prediction)
        np.testing.assert_allclose(larger.u_nominal, exact.u_nominal, atol=1e-4)

    def test_without_a_solution_the_least_total_violation(self):
        # ego 18 m/s, leads at 10 m/s, 36 m ahead: no input keeps every row, so slack is unavoidable
        prediction = self._prediction(gap0=36.0, lead_v=10.0)
        _, quadratic = self._solve(self._config(weight=None), 18.0, prediction)
        _, exact = self._solve(self._config(), 18.0, prediction)
        self.assertNotEqual(exact.solve_path, "fallback")
        self.assertGreater(exact.slack_max, 1e-3)
        self.assertLessEqual(float(np.sum(exact.slack)), float(np.sum(quadratic.slack)) + 1e-2)
        # the first jerk sits on the limit: a_0 + (-5) * 0.1
        self.assertAlmostEqual(exact.action, -0.5, places=4)

    def test_other_solve_paths_raise(self):
        with self.assertRaises(ValueError):
            self._config(weight=0.0)
        with self.assertRaises(ValueError):
            self._config(controller_variant=VARIANT_MULTIMODAL_OL, standoff_shortfall_weight=None,
                         jerk_input=False)
        # a configuration that would solve outside the cached feedback QP
        config = self._config(standoff_shortfall_weight=None, jerk_input=False,
                              risk_allocation_mode=RISK_PROBABILITY_WEIGHTED)
        with self.assertRaises(ValueError):
            NairACCSMPC(config).solve(np.array([0.0, 15.0]), self._prediction(gap0=40.0, lead_v=10.0),
                                      previous_accel_cmd=0.0, command_dt=self.COMMAND_DT)


if __name__ == "__main__":
    unittest.main()
