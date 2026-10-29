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
    SAFETY_BRAKE_DISTANCE,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    original_nair_acc_config,
    required_standoff,
    speed_gap,
    speed_gap_tangent,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestRelativeBrakeStandoff(unittest.TestCase):
    """``relbrake<b>``: the relative-speed safe distance
    ``L + d0 + max(0, tau v + (v^2 - v_l^2) / (2 b))`` on every lead cell."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    B = 3.0
    L_D0 = 4.5 + 3.0

    def _config(self, relbrake=True, **overrides):
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
        config.relative_brake_decel = self.B if relbrake else None
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap0, lead_v, cutin_scale=0.8):
        """Mode 0: an in-lane lane-keeping lead (deterministic cell, factor 1); mode 1: a cut-in
        chance cell (dopt cell, factor ``cutin_scale``).  Both lead at every step, at ``lead_v``."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[:, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        confidence = np.full((2, steps), np.nan)
        confidence[1] = 0.4
        scale = np.ones((2, steps))
        scale[1] = cutin_scale
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([0.6, 0.4]),
            covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)),
            active_mask=np.ones((2, steps), dtype=bool),
            chance_confidence=confidence,
            lead_mode_lk=np.array([[True] * steps, [False] * steps]),
            lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
            clearance_scale=scale,
            mode_names=["10:lk", "11:cutin"],
        )

    def _solve(self, config, ego_v, prediction, controller=None):
        controller = controller if controller is not None else NairACCSMPC(config)
        solution = controller.solve(
            np.array([0.0, ego_v]), prediction,
            previous_accel_cmd=0.0, command_dt=self.COMMAND_DT, ego_accel=0.0)
        return controller, solution

    def test_bumper_requirement_examples(self):
        # (ego, lead) m/s -> bumper requirement d - L [m]: the maintainer's four examples
        config = self._config()
        for ego_v, lead_v, expected in ((9.1, 11.1, 8.1), (15.0, 15.0, 22.5), (15.0, 10.0, 43.3), (20.0, 23.0, 7.5)):
            bumper = required_standoff([0.0, ego_v], config, lead_speed=lead_v) - config.vehicle_length
            self.assertAlmostEqual(bumper, expected, places=1)
        # the floor L + d0: a lead pulling away cannot lower it further
        self.assertEqual(required_standoff([0.0, 5.0], config, lead_speed=40.0), self.L_D0)
        # a negative predicted lead speed is clipped at 0
        self.assertEqual(speed_gap(10.0, -3.0, config), speed_gap(10.0, 0.0, config))

    def test_without_the_token_nothing_changes(self):
        config = self._config(relbrake=False)
        self.assertIsNone(config.relative_brake_decel)
        for ego_v, lead_v in ((9.1, 11.1), (15.0, 10.0)):
            self.assertEqual(required_standoff([0.0, ego_v], config, lead_speed=lead_v),
                             required_standoff([0.0, ego_v], config))
            self.assertEqual(speed_gap(ego_v, lead_v, config), config.time_headway * ego_v)
        controller, solution = self._solve(config, 15.0, self._prediction(gap0=40.0, lead_v=10.0))
        self.assertNotEqual(solution.solve_path, "fallback")
        problem = next(iter(controller._feedback_scalar_chance_qp_cache.values()))
        self.assertIsNone(problem.rel_slope_param)
        self.assertEqual(solution.speed_gap_tangent_error_max, 0.0)

    def test_equal_speeds_give_the_plain_standoff(self):
        config = self._config()
        for v in (5.0, 14.0, 20.0):
            self.assertEqual(required_standoff([0.0, v], config, lead_speed=v),
                             self.L_D0 + config.time_headway * v)
            # the tangent at the planned speed, with the lead at that speed, is tau * v as well
            slope, intercept = speed_gap_tangent(v, v, config)
            self.assertAlmostEqual(float(slope * v + intercept), config.time_headway * v, places=9)
        # a first solve plans from the measured speed, so with the lead at that speed the rows ask
        # for the plain standoff: the plan equals the plain controller's (up to the solver tolerance)
        prediction = self._prediction(gap0=60.0, lead_v=14.0)
        _, plain = self._solve(self._config(relbrake=False), 14.0, prediction)
        _, relative = self._solve(config, 14.0, prediction)
        self.assertNotEqual(relative.solve_path, "fallback")
        np.testing.assert_allclose(relative.reference.s_ref, plain.reference.s_ref, atol=1e-9)
        np.testing.assert_allclose(relative.u_nominal, plain.u_nominal, atol=1e-5)

    def test_faster_lead_sits_at_the_floor_and_the_qp_stays_feasible(self):
        # ego 10 m/s, both leads 20 m/s 0.5 m beyond the footprint L + d0: tau v = 13 m would be
        # demanded without the token (emergency slack); h = 13 + (100 - 400) / 6 < 0 on every cell,
        # the dopt cell included, so the requirement is the floor and nothing is relaxed
        config = self._config()
        prediction = self._prediction(gap0=self.L_D0 + 0.5, lead_v=20.0)
        _, solution = self._solve(config, 10.0, prediction)
        self.assertNotEqual(solution.solve_path, "fallback")
        self.assertEqual(solution.status, "optimal")
        self.assertLess(solution.slack_max, 1e-3)      # solver tolerance; the plain rule needs > 1 m below
        self.assertTrue(solution.feasible)
        # the dopt cell plans its floor alpha (L + d0), with no shortfall to give up: with h < 0 the
        # shortfall relaxes nothing, so the solver leaves it at its tolerance (a few mm) rather than at 0
        np.testing.assert_allclose(solution.planned_standoff[1], 0.8 * self.L_D0, atol=1e-2)
        self.assertTrue(np.all(np.isnan(solution.planned_standoff[0])))
        # the floor holds along the plan (safety_function is the margin to c (L + d0) here)
        self.assertGreaterEqual(np.min(solution.safety_values), -1e-6)
        # and the ego is not braked for a lead that is pulling away
        self.assertGreater(np.min(solution.u_nominal), -0.1)
        # the same without the token: the time gap is demanded and the slack has to carry it
        _, plain = self._solve(self._config(relbrake=False), 10.0, prediction)
        self.assertGreater(plain.slack_max, 1.0)

    def test_closing_lead_asks_more_and_the_reference_carries_it(self):
        # ego 15 m/s, leads at 10 m/s, 40 m ahead: the plain rule needs 27 m, the relative one 47.8 m
        prediction = self._prediction(gap0=40.0, lead_v=10.0)
        _, plain = self._solve(self._config(relbrake=False), 15.0, prediction)
        _, relative = self._solve(self._config(), 15.0, prediction)
        self.assertNotEqual(relative.solve_path, "fallback")
        self.assertLess(np.min(relative.u_nominal), np.min(plain.u_nominal) - 0.3)
        # the reference asks for the larger standoff on every lead cell (s_ref is the min of the free
        # ramp and lead_s - safe_gap: at step 0 the plain rule's 13 m of room leaves the ramp, the
        # relative rule's -7.8 m pulls the reference back)
        self.assertTrue(np.all(relative.reference.s_ref <= plain.reference.s_ref + 1e-9))
        self.assertLess(relative.reference.s_ref[0, 0], plain.reference.s_ref[0, 0] - 5.0)
        # and for a smaller one behind a faster lead, down to the floor: 20 m ahead at 20 m/s, the plain
        # rule wants 27 m (reference pulled back 7 m), the relative one 7.5 m (the ramp stands)
        prediction = self._prediction(gap0=20.0, lead_v=20.0)
        _, plain = self._solve(self._config(relbrake=False), 15.0, prediction)
        _, relative = self._solve(self._config(), 15.0, prediction)
        self.assertGreater(relative.reference.s_ref[0, 0], plain.reference.s_ref[0, 0] + 5.0)

    def test_tangent_is_exact_at_the_planned_speed_and_below_elsewhere(self):
        config = self._config()
        for v_bar, lead_v in ((12.0, 8.0), (18.0, 20.0), (6.0, 0.0)):
            slope, intercept = speed_gap_tangent(v_bar, lead_v, config)
            exact = config.time_headway * v_bar + (v_bar ** 2 - lead_v ** 2) / (2.0 * self.B)
            self.assertAlmostEqual(float(slope * v_bar + intercept), exact, places=9)
            for v in (v_bar - 1.0, v_bar + 2.5):
                tangent = float(slope * v + intercept)
                true_h = config.time_headway * v + (v ** 2 - lead_v ** 2) / (2.0 * self.B)
                self.assertAlmostEqual(true_h - tangent, (v - v_bar) ** 2 / (2.0 * self.B), places=9)
        # broadcasting: one planned speed per step against a lead speed per cell
        slope, intercept = speed_gap_tangent(np.full((1, 4), 12.0), np.array([[8.0] * 4, [20.0] * 4]), config)
        self.assertEqual(slope.shape, (2, 4))
        self.assertEqual(intercept.shape, (2, 4))
        # the solution reports the largest tangent error at the solved plan
        _, solution = self._solve(config, 15.0, self._prediction(gap0=40.0, lead_v=10.0))
        planned = solution.x_nominal[:, :, 1] - 15.0      # the first solve plans from the measured speed
        self.assertAlmostEqual(solution.speed_gap_tangent_error_max,
                               float(np.max(planned ** 2)) / (2.0 * self.B), places=9)

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_relative_brake_decel
        self.assertEqual(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_relbrake3"), 3.0)
        self.assertEqual(parse("acc_x_relbrake3_jinput_ra0"), 3.0)
        self.assertEqual(parse("acc_x_relbrake2.5_ra0"), 2.5)
        self.assertIsNone(parse("acc_x_xrelbrake3"))
        self.assertIsNone(parse("acc_x_relbrake3x_ra0"))
        self.assertIsNone(parse("acc_x_relbrake_ra0"))
        self.assertIsNone(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_jinput_ra0"))

    def test_other_solve_paths_raise(self):
        with self.assertRaises(ValueError):
            self._config(relative_brake_decel=0.0)
        with self.assertRaises(ValueError):
            self._config(controller_variant=VARIANT_MULTIMODAL_OL, standoff_shortfall_weight=None,
                         jerk_input=False)
        with self.assertRaises(ValueError):
            self._config(safety_constraint_mode=SAFETY_BRAKE_DISTANCE, standoff_shortfall_weight=None,
                         jerk_input=False)
        # a configuration that would solve outside the cached feedback QP
        config = self._config(standoff_shortfall_weight=None, jerk_input=False,
                              risk_allocation_mode=RISK_PROBABILITY_WEIGHTED)
        with self.assertRaises(ValueError):
            NairACCSMPC(config).solve(np.array([0.0, 15.0]), self._prediction(gap0=40.0, lead_v=10.0),
                                      previous_accel_cmd=0.0, command_dt=self.COMMAND_DT)


if __name__ == "__main__":
    unittest.main()
