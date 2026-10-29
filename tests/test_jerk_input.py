import os
import sys
import unittest

import numpy as np
from scipy.linalg import expm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    acc_jerk_step,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestJerkInput(unittest.TestCase):
    """``jinput``: the jerk as the MPC input, state [s, v, a]."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    EGO = np.array([0.0, 14.0])

    def test_step_is_the_exact_triple_integrator(self):
        for dt in (0.2, 0.1):
            # [s, v, a, j] with j held: the matrix exponential of the augmented chain
            continuous = np.zeros((4, 4))
            continuous[0, 1] = continuous[1, 2] = continuous[2, 3] = 1.0
            phi = expm(continuous * dt)
            state, accel = acc_jerk_step([3.0, 14.0], 0.7, -2.0, dt)
            expected = phi @ np.array([3.0, 14.0, 0.7, -2.0])
            np.testing.assert_allclose([state[0], state[1], accel], expected[:3], atol=1e-12)

    def _config(self, **overrides):
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=2,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, gap0=30.0, lead_v=12.0):
        """Lane keeping without a lead, and a cut-in that leads at every step."""
        steps = self.HORIZON + 1
        means = np.zeros((2, steps, 2))
        means[:, :, 0] = gap0 + lead_v * self.DT * np.arange(steps)
        means[:, :, 1] = lead_v
        active = np.zeros((2, steps), dtype=bool)
        active[1] = True
        confidence = np.full((2, steps), np.nan)
        confidence[1] = 0.4
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([0.6, 0.4]),
            covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)),
            active_mask=active,
            chance_confidence=confidence,
            lead_mode_lk=np.zeros((2, steps), dtype=bool),
            lead_vehicle_ids=np.where(active, 10, -1),
            clearance_scale=np.ones((2, steps)),
            mode_names=["10:lk", "10:cutin"],
        )

    def _solve(self, ego_accel=None, ego=None, previous_accel_cmd=0.0, prediction=None, controller=None,
               **overrides):
        controller = controller if controller is not None else NairACCSMPC(self._config(**overrides))
        solution = controller.solve(
            self.EGO if ego is None else ego,
            self._prediction() if prediction is None else prediction,
            previous_accel_cmd=previous_accel_cmd, command_dt=self.COMMAND_DT, ego_accel=ego_accel)
        return controller, solution

    def _check_plan(self, solution, a0):
        """x_nominal and u_nominal follow the triple integrator from (EGO, a0) under j_nominal."""
        self.assertIsNotNone(solution.j_nominal)
        for mode in range(2):
            state, accel = self.EGO.copy(), a0
            for step in range(self.HORIZON):
                state, accel = acc_jerk_step(state, accel, solution.j_nominal[mode, step], self.DT)
                np.testing.assert_allclose(solution.x_nominal[mode, step + 1], state, atol=1e-6)
                self.assertAlmostEqual(solution.u_nominal[mode, step], accel, places=6)

    def test_plan_follows_the_triple_integrator_from_the_actual_acceleration(self):
        for ego_accel in (None, 0.8, -1.5):
            _, solution = self._solve(ego_accel, previous_accel_cmd=0.3)
            self.assertNotEqual(solution.solve_path, "fallback")
            self._check_plan(solution, 0.3 if ego_accel is None else ego_accel)

    def test_bounds_hold(self):
        config = self._config()
        # far below the set speed: a_max binds; a close, slower cut-in: braking
        far_lead, near_lead = dict(gap0=60.0, lead_v=12.0), dict(gap0=18.0, lead_v=6.0)
        for lead in (far_lead, near_lead):
            _, solution = self._solve(0.0, ego=np.array([0.0, 10.0]), prediction=self._prediction(**lead))
            self.assertNotEqual(solution.solve_path, "fallback")
            self.assertLessEqual(np.max(np.abs(solution.j_nominal)), config.jerk_limit + 1e-6)
            self.assertGreaterEqual(np.min(solution.u_nominal), config.a_min - 1e-6)
            self.assertLessEqual(np.max(solution.u_nominal), config.a_max + 1e-6)
            self.assertGreaterEqual(np.min(solution.x_nominal[:, :, 1]), config.v_min - 1e-6)
            self.assertLessEqual(np.max(solution.x_nominal[:, :, 1]), config.v_max + 1e-6)
        _, far = self._solve(0.0, ego=np.array([0.0, 10.0]), prediction=self._prediction(**far_lead))
        self.assertGreater(np.max(far.u_nominal), 0.5)        # it does accelerate toward the set speed
        _, near = self._solve(0.0, ego=np.array([0.0, 10.0]), prediction=self._prediction(**near_lead))
        self.assertLess(np.min(near.u_nominal[1]), -0.5)      # and brakes for the close, slower cut-in

    def test_issued_command_is_the_planned_acceleration_one_command_period_ahead(self):
        for ego_accel in (0.8, -1.5):
            _, solution = self._solve(ego_accel)
            self.assertAlmostEqual(solution.action, ego_accel + solution.j_nominal[0, 0] * self.COMMAND_DT, places=9)
            # between a_0 and the planned a_1, so within the acceleration bounds
            lo, hi = sorted((ego_accel, solution.u_nominal[0, 0]))
            self.assertGreaterEqual(solution.action, lo - 1e-9)
            self.assertLessEqual(solution.action, hi + 1e-9)

    def test_first_jerk_is_shared_across_scenarios(self):
        _, solution = self._solve(0.0, prediction=self._prediction(gap0=18.0))
        self.assertNotEqual(solution.solve_path, "fallback")
        np.testing.assert_allclose(solution.j_nominal[:, 0], solution.j_nominal[0, 0], atol=1e-7)
        # the scenarios differ later: lane keeping accelerates, the cut-in brakes
        self.assertGreater(np.max(np.abs(solution.j_nominal[0] - solution.j_nominal[1])), 1e-3)

    def test_warm_start_is_the_shifted_jerk_plan(self):
        controller, first = self._solve(0.0)
        _, second = self._solve(first.action, controller=controller)
        self.assertNotEqual(second.solve_path, "fallback")
        np.testing.assert_allclose(second.reference.prev_u[0, :-1], first.j_nominal[0, 1:], atol=1e-9)

    def test_standstill_clamp(self):
        # stopped with a negative acceleration: without the clamp v_1 >= 0 needs j_0 > 5
        _, stopped = self._solve(-1.5, ego=np.array([0.0, 0.0]), previous_accel_cmd=-1.5)
        self.assertNotEqual(stopped.solve_path, "fallback")
        self.assertAlmostEqual(stopped.action, stopped.j_nominal[0, 0] * self.COMMAND_DT, places=9)
        self.assertAlmostEqual(stopped.u_nominal[0, 0], self.DT * stopped.j_nominal[0, 0], places=9)
        self.assertGreaterEqual(np.min(stopped.x_nominal[:, :, 1]), -1e-9)
        # at v_max with a positive acceleration the clamp starts the plan from a_0 = 0
        config = self._config()
        _, capped = self._solve(1.0, ego=np.array([0.0, config.v_max]), previous_accel_cmd=1.0)
        self.assertNotEqual(capped.solve_path, "fallback")
        self.assertAlmostEqual(capped.u_nominal[0, 0], self.DT * capped.j_nominal[0, 0], places=9)
        self.assertLessEqual(capped.action, 1e-9)
        # the clamp is one-sided: a positive acceleration at standstill is kept
        _, pulling = self._solve(0.8, ego=np.array([0.0, 0.0]), previous_accel_cmd=0.8)
        self.assertAlmostEqual(pulling.u_nominal[0, 0], 0.8 + self.DT * pulling.j_nominal[0, 0], places=9)

    def test_fallback_plans_in_jerk_units(self):
        controller = NairACCSMPC(self._config())
        controller.config.allow_solver_fallback = True
        reference = controller.reference_adapter.generate(self.EGO, self._prediction(gap0=18.0))
        reference = controller._with_linearization_input(reference, self._prediction(gap0=18.0))
        controller._set_command_context(-1.0, self.COMMAND_DT, ego_accel=-1.0, ego_speed=self.EGO[1])
        policy, _ = controller._fallback_policy(self.EGO, self._prediction(gap0=18.0), reference)
        jerk = reference.prev_u + policy.h[:, :, 0]
        self.assertLessEqual(np.max(np.abs(jerk)), controller.config.jerk_limit + 1e-9)
        accel = -1.0 + self.DT * np.cumsum(jerk, axis=1)
        self.assertGreaterEqual(np.min(accel), controller.config.a_min - 1e-9)
        self.assertLessEqual(np.max(accel), controller.config.a_max + 1e-9)

    def test_invalid_combinations_raise(self):
        with self.assertRaises(ValueError):
            self._config(accel_lag_tau=0.5)
        with self.assertRaises(ValueError):
            self._config(controller_variant=VARIANT_MULTIMODAL_OL, standoff_shortfall_weight=None)

    def test_without_the_token_nothing_changes(self):
        plain = NairACCSMPC(self._config(jerk_input=False))
        solution = plain.solve(self.EGO, self._prediction(), previous_accel_cmd=0.3, command_dt=self.COMMAND_DT)
        self.assertIsNone(solution.j_nominal)
        self.assertAlmostEqual(solution.action, solution.reference.prev_u[0, 0] + solution.policy.h[0, 0, 0], places=9)

    def test_policy_token(self):
        base = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr"
                "_trajcutin_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40")
        self.assertFalse(ACCNairSMPCAgent._parse_jerk_input(base))
        self.assertTrue(ACCNairSMPCAgent._parse_jerk_input(base + "_jinput_ra0"))
        self.assertTrue(ACCNairSMPCAgent._parse_jerk_input(base + "_jinput"))
        self.assertFalse(ACCNairSMPCAgent._parse_jerk_input(base + "_jinputx"))
        self.assertFalse(ACCNairSMPCAgent._parse_jerk_input(base + "_nojinput"))


if __name__ == "__main__":
    unittest.main()
