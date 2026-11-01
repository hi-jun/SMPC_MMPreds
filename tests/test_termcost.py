import os
import sys
import unittest

import numpy as np
from scipy.linalg import solve_discrete_are


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    original_nair_acc_config,
    terminal_cost_matrix,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestTerminalCost(unittest.TestCase):
    """``termcost<c>``: the terminal cost-to-go ``c * 3 * P_j * [e_N, a_N] (P - Q) [e_N, a_N]^T``
    on the cached feedback QP with the jerk input."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1

    def _config(self, scale=1.0, **overrides):
        """The current design (dopt100, jinput, r_a 0, l1slack1e6) with or without the token."""
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
        config.terminal_cost_scale = scale
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap0, lead_v):
        """Mode 0: an in-lane lane-keeping lead (hard standoff); mode 1: a cut-in chance cell.
        Both lead at every step, at ``lead_v``."""
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

    def _solve(self, config, ego_v, prediction, ego_accel=0.0):
        controller = NairACCSMPC(config)
        solution = controller.solve(
            np.array([0.0, ego_v]), prediction,
            previous_accel_cmd=ego_accel, command_dt=self.COMMAND_DT, ego_accel=ego_accel)
        return controller, solution

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_terminal_cost_scale
        self.assertEqual(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0_fullfoot_l1slack1e6_termcost0.5"),
                         0.5)
        self.assertEqual(parse("acc_x_termcost1_ra0"), 1.0)
        self.assertEqual(parse("acc_x_termcost4"), 4.0)
        self.assertIsNone(parse("acc_x_xtermcost1"))
        self.assertIsNone(parse("acc_x_termcost1x_ra0"))
        self.assertIsNone(parse("acc_x_termcost_ra0"))
        self.assertIsNone(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0_fullfoot_l1slack1e6"))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self._config(scale=0.0)
        with self.assertRaises(ValueError):
            self._config(scale=-1.0)
        with self.assertRaises(ValueError):
            self._config(scale=1.0, jerk_input=False)
        with self.assertRaises(ValueError):
            self._config(scale=1.0, optimize_k=True)
        with self.assertRaises(ValueError):
            self._config(scale=1.0, controller_variant=VARIANT_MULTIMODAL_OL)

    def test_terminal_matrix_is_the_riccati_cost_to_go(self):
        config = self._config()
        pq = terminal_cost_matrix(config)
        a_mat = np.array([[1.0, self.DT], [0.0, 1.0]])
        b_mat = np.array([[0.5 * self.DT ** 2], [self.DT]])
        q_mat = np.diag([config.q_v, 0.0])
        p_mat = solve_discrete_are(a_mat, b_mat, q_mat, np.array([[config.r_jerk]]))
        np.testing.assert_allclose(pq, p_mat - q_mat, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(pq, [[17.2898, 31.6228], [31.6228, 112.5127]], atol=1e-3)
        self.assertTrue(np.all(np.linalg.eigvalsh(pq) > 0.0))      # convex term

    def test_without_the_token_the_qp_key_and_solution_are_unchanged(self):
        config = self._config(scale=None)
        self.assertIsNone(config.terminal_cost_scale)
        prediction = self._prediction(gap0=80.0, lead_v=10.0)
        controller, solution = self._solve(config, 8.0, prediction, ego_accel=-1.5)
        self.assertNotEqual(solution.solve_path, "fallback")
        key = next(iter(controller._feedback_scalar_chance_qp_cache))
        self.assertIsNone(key[-1])
        _, again = self._solve(self._config(scale=None), 8.0, prediction, ego_accel=-1.5)
        np.testing.assert_array_equal(again.j_nominal, solution.j_nominal)

    def test_cost_to_go_releases_the_brake_faster(self):
        # ego 8 m/s decelerating at -1.5 m/s^2, the leads 80 m ahead at 10 m/s: no row binds, the plan
        # only tracks the speed reference.  Without the terminal cost the plan does not account for the
        # speed deficit beyond the horizon and releases the brake slowly; with it the first jerk is larger
        # and the plan ends with a smaller acceleration deficit.
        prediction = self._prediction(gap0=80.0, lead_v=10.0)
        _, without = self._solve(self._config(scale=None), 8.0, prediction, ego_accel=-1.5)
        _, with_term = self._solve(self._config(scale=1.0), 8.0, prediction, ego_accel=-1.5)
        self.assertNotEqual(with_term.solve_path, "fallback")
        self.assertTrue(with_term.feasible)
        self.assertGreater(with_term.j_nominal[0, 0], without.j_nominal[0, 0] + 0.1)
        self.assertGreater(with_term.u_nominal[0, -1], without.u_nominal[0, -1])
        # the planned speed at the horizon end is closer to the reference
        v_ref_end = without.reference.v_ref[0, -1]
        self.assertLess(abs(with_term.x_nominal[0, -1, 1] - v_ref_end), abs(without.x_nominal[0, -1, 1] - v_ref_end))
        # and the scale is monotone: a larger c pulls the end of the plan further toward the reference
        _, larger = self._solve(self._config(scale=4.0), 8.0, prediction, ego_accel=-1.5)
        self.assertLess(abs(larger.x_nominal[0, -1, 1] - v_ref_end), abs(with_term.x_nominal[0, -1, 1] - v_ref_end))

    def test_safety_rows_still_hold_with_the_term(self):
        # ego 15 m/s behind leads at 10 m/s, 30 m ahead: braking keeps every row; the term never relaxes them
        prediction = self._prediction(gap0=30.0, lead_v=10.0)
        _, solution = self._solve(self._config(scale=4.0), 15.0, prediction)
        self.assertNotEqual(solution.solve_path, "fallback")
        self.assertTrue(solution.feasible)
        self.assertLess(solution.slack_max, 1e-4)
        self.assertLess(np.min(solution.u_nominal), -1.0)

    def test_qp_key_carries_the_scale(self):
        prediction = self._prediction(gap0=80.0, lead_v=10.0)
        controller, _ = self._solve(self._config(scale=0.5), 8.0, prediction)
        key = next(iter(controller._feedback_scalar_chance_qp_cache))
        self.assertEqual(key[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
