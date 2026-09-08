import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CARLA_SCRIPT_DIR = os.path.join(ROOT, "scripts", "carla")
sys.path.insert(0, CARLA_SCRIPT_DIR)

from utils.acc_nair_smpc import (  # noqa: E402
    ACCReference,
    BRAKE_DISTANCE_BOUND_HARD_BAND,
    BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL,
    NairACCConfig,
    NairACCPolicy,
    NairACCSolution,
    NairACCSMPC,
    MultimodalLeadPrediction,
    OldACCReferenceAdapter,
    PolicySharingTree,
    RISK_FIXED,
    RISK_OPTIMIZED_ETA,
    RISK_PROBABILITY_WEIGHTED,
    SAFETY_BRAKE_DISTANCE,
    SAFETY_CONFIDENCE_CHANCE,
    SAFETY_NOMINAL_SAFE_DISTANCE,
    SAFETY_SCALAR_CHANCE,
    VARIANT_MULTIMODAL_OL,
    allocate_risk,
    cdf_lower_bound_lines,
    chance_tightening,
    confidence_quantile,
    confidence_tightening,
    exact_brake_distance,
    optimized_eta_target_probability,
    safety_function,
    safety_std,
    scalar_chance_margin,
    upper_bound_brake_distance,
)
from utils.synthetic_lk_cutin_prediction import (  # noqa: E402
    SyntheticCutInConfig,
    SyntheticLaneKeepingCutInPredictor,
)


class TestACCNairSMPCPort(unittest.TestCase):
    def test_import_and_config(self):
        config = NairACCConfig(horizon=3, dt=0.2, desired_speed=14.0, num_modes=2)
        controller = NairACCSMPC(config)
        self.assertEqual(controller.config.horizon, 3)
        self.assertEqual(controller.config.num_modes, 2)
        self.assertEqual(controller.config.solver_name, "gurobi")
        self.assertFalse(controller.config.optimize_k)

        brake_config = NairACCConfig(
            horizon=3,
            dt=0.2,
            safety_constraint_mode=SAFETY_BRAKE_DISTANCE,
        )
        self.assertEqual(brake_config.safety_constraint_mode, SAFETY_BRAKE_DISTANCE)

    def test_synthetic_provider_shapes(self):
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(horizon=4, dt=0.2, probabilities=[0.6, 0.4])
        )
        result = predictor.predict(
            ego_state=np.array([0.0, 12.0]),
            target_state=np.array([35.0, 3.5, 10.0]),
        )
        self.assertEqual(result.mode_names, ["lane_keeping", "cutin"])
        self.assertEqual(result.prediction.means.shape, (2, 5, 2))
        self.assertEqual(result.prediction.covariances.shape, (2, 5, 2, 2))
        self.assertEqual(result.frenet_mean.shape, (2, 5, 3))
        self.assertEqual(result.covariance_2d.shape, (2, 5, 3, 3))
        np.testing.assert_allclose(result.probabilities, np.array([0.6, 0.4]))

    def test_cutin_mode_blocks_when_predicted_path_reaches_ego_lane(self):
        cfg = SyntheticCutInConfig(
            horizon=10,
            dt=0.2,
            cutin_start_delay=0.0,
            cutin_duration=1.0,
            blocking_alpha=1.1,
            ego_lane_blocking_d_threshold=0.5,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(cfg)
        result = predictor.predict(
            ego_state=np.array([0.0, 12.0]),
            target_state=np.array([25.0, 3.5, 8.0]),
        )

        cutin_frenet = result.frenet_mean[1]
        cutin_effective = result.prediction.means[1]
        predicted_in_ego_lane = np.abs(cutin_frenet[:, 1] - cfg.ego_lane_d) <= cfg.ego_lane_blocking_d_threshold

        self.assertTrue(np.any(predicted_in_ego_lane))
        np.testing.assert_allclose(
            cutin_effective[predicted_in_ego_lane, 0],
            cutin_frenet[predicted_in_ego_lane, 0],
        )
        self.assertTrue(np.all(cutin_effective[~predicted_in_ego_lane, 0] >= cfg.nonblocking_gap))

    def test_chance_tightening_decreases_with_risk(self):
        config = NairACCConfig(horizon=3, dt=0.2)
        covariance = np.diag([0.25, 0.09])

        conservative = chance_tightening(covariance, 0.01, config)
        relaxed = chance_tightening(covariance, 0.20, config)

        self.assertGreater(conservative, relaxed)

    def test_original_nair_cdf_lower_bound_lines(self):
        config = NairACCConfig(horizon=3, dt=0.2)
        lines = cdf_lower_bound_lines(config)

        self.assertEqual(len(lines), 2)
        self.assertAlmostEqual(lines[0][0], (0.91 - 0.508) / (1.35 - 0.02))
        self.assertAlmostEqual(lines[0][1], 0.508 - lines[0][0] * 0.02)
        self.assertAlmostEqual(lines[1][0], (0.978 - 0.91) / (2.0 - 1.35))
        self.assertAlmostEqual(lines[1][1], 0.91 - lines[1][0] * 1.35)

    def test_optimized_eta_requires_scalar_chance(self):
        with self.assertRaises(ValueError):
            NairACCConfig(
                horizon=3,
                dt=0.2,
                safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE,
                risk_allocation_mode=RISK_OPTIMIZED_ETA,
            )

        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            risk_allocation_mode=RISK_OPTIMIZED_ETA,
            fixed_risk_level=0.05,
        )
        self.assertAlmostEqual(optimized_eta_target_probability(config), 0.95)

    def test_eta_tightening_uses_scalar_safety_std(self):
        config = NairACCConfig(horizon=3, dt=0.2)
        covariance = np.diag([0.25, 0.09])
        eta = 1.64

        self.assertAlmostEqual(
            eta * safety_std(covariance, config),
            chance_tightening(covariance, 1.0 - 0.9494974165258963, config),
        )

    def test_probability_weighted_risk_allocates_more_to_low_probability_mode(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            fixed_risk_level=0.05,
            risk_allocation_mode=RISK_PROBABILITY_WEIGHTED,
        )
        risks = allocate_risk(np.array([0.8, 0.2]), config)

        self.assertEqual(risks.shape, (2, 4))
        self.assertGreater(risks[1, 0], risks[0, 0])
        self.assertAlmostEqual(float(np.dot([0.8, 0.2], risks[:, 0])), 0.05)

    def test_scalar_chance_margin_matches_safety_minus_tightening(self):
        config = NairACCConfig(horizon=3, dt=0.2)
        ego = np.array([10.0, 12.0])
        lead = np.array([35.0, 10.0])
        covariance = np.diag([0.25, 0.09])
        risk = 0.05

        expected = safety_function(ego, lead, config) - chance_tightening(covariance, risk, config)

        self.assertAlmostEqual(scalar_chance_margin(ego, lead, covariance, risk, config), expected)

    def test_brake_distance_secant_is_upper_bound_on_interval(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            a_min=-8.0,
            d0=3.0,
            vehicle_length=4.5,
            safety_constraint_mode=SAFETY_BRAKE_DISTANCE,
        )
        v_min = 10.0
        v_max = 14.0
        speeds = np.linspace(v_min, v_max, 21)
        upper = upper_bound_brake_distance(speeds, config, v_min, v_max)
        exact = exact_brake_distance(speeds, config)

        self.assertTrue(np.all(upper + 1.0e-12 >= exact))
        self.assertAlmostEqual(float(upper[0]), float(exact[0]))
        self.assertAlmostEqual(float(upper[-1]), float(exact[-1]))

    def test_brake_distance_diagnostic_uses_exact_distance(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            a_min=-8.0,
            d0=3.0,
            vehicle_length=4.5,
            safety_constraint_mode=SAFETY_BRAKE_DISTANCE,
        )
        ego = np.array([10.0, 14.0])
        lead = np.array([35.0, 10.0])
        expected = (
            lead[0]
            - ego[0]
            - config.vehicle_length
            - config.d0
            - 14.0 ** 2 / (2.0 * 8.0)
        )

        self.assertAlmostEqual(
            safety_function(ego, lead, config, linearization_speed=12.0),
            expected,
        )

    def test_reachable_brake_distance_intervals_use_acceleration_bounds(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            v_min=0.0,
            v_max=15.0,
            a_min=-3.0,
            a_max=2.0,
            brake_distance_bound_mode=BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL,
        )
        controller = NairACCSMPC(config)

        v_min, v_max = controller._brake_speed_intervals(np.array([0.0, 14.0]), 2)

        np.testing.assert_allclose(v_min[0], [14.0, 13.4, 12.8, 12.2])
        np.testing.assert_allclose(v_max[0], [14.0, 14.4, 14.8, 15.0])
        np.testing.assert_allclose(v_min[0], v_min[1])
        np.testing.assert_allclose(v_max[0], v_max[1])

    def test_brake_distance_interval_contains_current_speed_above_config_max(self):
        config = NairACCConfig(horizon=2, dt=0.2, v_max=15.0)
        controller = NairACCSMPC(config)

        v_min, v_max = controller._brake_speed_intervals(np.array([0.0, 15.1]), 1)

        self.assertAlmostEqual(v_min[0, 0], 15.1)
        self.assertAlmostEqual(v_max[0, 0], 15.1)

    def test_prev_u_shifts_previous_solution_input(self):
        config = NairACCConfig(horizon=4, dt=0.2, num_modes=2, a_min=-10.0, a_max=10.0)
        controller = NairACCSMPC(config)
        policy = NairACCPolicy(
            h=np.zeros((2, 4, 1)),
            M=np.zeros((2, 4, 4, 2)),
            K=np.zeros((2, 4, 2)),
        )
        controller.last_solution = NairACCSolution(
            action=0.0,
            status="optimal",
            feasible=True,
            cost=0.0,
            policy=policy,
            reference=None,
            x_nominal=np.zeros((2, 5, 2)),
            u_nominal=np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 1.1, 1.2, 1.3]]),
            safety_values=np.zeros((2, 5)),
            risk_levels=np.zeros((2, 5)),
            solver_message="",
        )
        reference = ACCReference(
            s_ref=np.zeros((2, 5)),
            v_ref=np.zeros((2, 5)),
            a_ref=np.zeros((2, 4)),
            prev_u=np.zeros((2, 4)),
        )

        shifted = controller._with_linearization_input(
            reference,
            type("Prediction", (), {"num_modes": 2})(),
        )

        expected = np.array([[0.2, 0.3, 0.4, 0.4], [0.2, 0.3, 0.4, 0.4]])
        np.testing.assert_allclose(shifted.prev_u, expected)

    def test_policy_sharing_tree_shares_policy_before_split(self):
        config = NairACCConfig(horizon=4, dt=0.2, num_modes=2)
        controller = NairACCSMPC(config)
        tree = PolicySharingTree.single_split(num_modes=2, horizon=4, split_step=2)
        prediction = MultimodalLeadPrediction(
            means=np.zeros((2, 5, 2)),
            probabilities=np.array([0.6, 0.4]),
            policy_tree=tree,
        )

        layout = controller._policy_variable_layout(prediction)

        np.testing.assert_array_equal(layout.group_map[:, 0], np.array([0, 0]))
        np.testing.assert_array_equal(layout.group_map[:, 1], np.array([1, 1]))
        self.assertEqual(len(set(int(group) for group in layout.group_map[:, 0])), 1)
        self.assertEqual(layout.policy_split_steps, (2,))

    def test_optimized_eta_variables_follow_joint_mode_sequence_count(self):
        config = NairACCConfig(
            horizon=4,
            dt=0.2,
            num_modes=3,
            safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            risk_allocation_mode=RISK_OPTIMIZED_ETA,
        )
        controller = NairACCSMPC(config)
        tree = PolicySharingTree.single_split(num_modes=3, horizon=4, split_step=3)
        prediction = MultimodalLeadPrediction(
            means=np.zeros((3, 5, 2)),
            probabilities=np.array([0.5, 0.3, 0.2]),
            policy_tree=tree,
        )

        layout = controller._policy_variable_layout(prediction)
        fixed_risk_layout = NairACCSMPC(
            NairACCConfig(
                horizon=4,
                dt=0.2,
                num_modes=3,
                safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            )
        )._policy_variable_layout(prediction)

        self.assertEqual(prediction.num_joint_modes, 3)
        self.assertEqual(layout.group_map[:, 0].tolist(), [0, 0, 0])
        self.assertEqual(layout.policy_group_count, 6)
        self.assertEqual(
            layout.num_decision_variables - fixed_risk_layout.num_decision_variables,
            2 * prediction.num_joint_modes,
        )

    def test_k_variables_use_target_local_group_map(self):
        config = NairACCConfig(horizon=4, dt=0.2, num_modes=4, optimize_k=True)
        controller = NairACCSMPC(config)
        tree = PolicySharingTree.single_split(num_modes=4, horizon=4, split_step=2)
        prediction = MultimodalLeadPrediction(
            means=np.zeros((4, 5, 2)),
            probabilities=np.ones(4) / 4.0,
            policy_tree=tree,
            k_group_map=np.array([
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
            ]),
            k_group_names=["shared", "lead:lk", "lead:cutin"],
        )

        layout = controller._policy_variable_layout(prediction)

        self.assertEqual(layout.k_group_count, 3)
        self.assertEqual(layout.k_decision_variables, 12)
        self.assertLess(layout.k_decision_variables, config.num_modes * config.horizon * 2)
        np.testing.assert_array_equal(layout.k_group_map[0], layout.k_group_map[1])
        np.testing.assert_array_equal(layout.k_group_map[2], layout.k_group_map[3])

    def test_open_loop_layout_optimizes_h_without_feedback_gains(self):
        config = NairACCConfig(
            horizon=4,
            dt=0.2,
            num_modes=3,
            controller_variant=VARIANT_MULTIMODAL_OL,
            safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            risk_allocation_mode=RISK_OPTIMIZED_ETA,
        )
        controller = NairACCSMPC(config)
        tree = PolicySharingTree.single_split(num_modes=3, horizon=4, split_step=2)
        prediction = MultimodalLeadPrediction(
            means=np.zeros((3, 5, 2)),
            probabilities=np.array([0.5, 0.3, 0.2]),
            policy_tree=tree,
        )

        feedback_layout = controller._policy_variable_layout(prediction)
        open_loop_layout = controller._policy_variable_layout(prediction, open_loop=True)

        self.assertGreater(feedback_layout.policy_group_count, open_loop_layout.policy_group_count)
        self.assertEqual(open_loop_layout.policy_group_count, config.horizon)
        np.testing.assert_array_equal(
            open_loop_layout.group_map,
            np.tile(np.arange(config.horizon), (prediction.num_modes, 1)),
        )
        self.assertEqual(
            open_loop_layout.num_decision_variables,
            config.horizon + 1 + 2 * prediction.num_joint_modes,
        )

    def test_controller_solves_one_synthetic_step(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([32.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)

        self.assertTrue(np.isfinite(solution.action))
        self.assertGreaterEqual(solution.action, config.a_min - 1.0e-9)
        self.assertLessEqual(solution.action, config.a_max + 1.0e-9)

    def test_nominal_safe_distance_uses_cached_safe_distance_qp(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE,
            risk_allocation_mode=RISK_FIXED,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)

        self.assertTrue(np.isfinite(solution.action))
        self.assertEqual(solution.solve_path, "cached_feedback_safe_distance_qp")
        self.assertIn("cached_feedback_safe_distance_qp=1", solution.solver_message)
        self.assertAlmostEqual(solution.tightening_max, 0.0)
        np.testing.assert_allclose(solution.policy.K, 0.0)
        np.testing.assert_allclose(solution.policy.M, 0.0)

    def test_brake_distance_uses_cached_feedback_qp(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            safety_constraint_mode=SAFETY_BRAKE_DISTANCE,
            risk_allocation_mode=RISK_FIXED,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)

        self.assertTrue(np.isfinite(solution.action))
        self.assertEqual(solution.solve_path, "cached_feedback_brake_distance_qp")
        self.assertIn("cached_feedback_brake_distance_qp=1", solution.solver_message)
        self.assertAlmostEqual(solution.tightening_max, 0.0)
        np.testing.assert_allclose(solution.policy.K, 0.0)
        np.testing.assert_allclose(solution.policy.M, 0.0)

    def test_brake_distance_hard_band_constrains_nominal_speed(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            safety_constraint_mode=SAFETY_BRAKE_DISTANCE,
            brake_distance_bound_mode=BRAKE_DISTANCE_BOUND_HARD_BAND,
            brake_distance_hard_band_width=4.0,
            risk_allocation_mode=RISK_FIXED,
            allow_solver_fallback=False,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)

        self.assertEqual(solution.status, "optimal")
        self.assertTrue(np.all(solution.x_nominal[:, :, 1] >= 8.0 - 1.0e-7))
        self.assertTrue(np.all(solution.x_nominal[:, :, 1] <= 16.0 + 1.0e-7))

    def test_open_loop_controller_solves_h_only_policy(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            fixed_risk_level=0.05,
            controller_variant=VARIANT_MULTIMODAL_OL,
            safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            risk_allocation_mode=RISK_OPTIMIZED_ETA,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)
        expected_action = np.clip(
            solution.reference.prev_u[0, 0] + solution.policy.h[0, 0, 0],
            config.a_min,
            config.a_max,
        )

        self.assertTrue(np.isfinite(solution.action))
        self.assertAlmostEqual(solution.action, float(expected_action))
        np.testing.assert_allclose(solution.policy.K, 0.0)
        np.testing.assert_allclose(solution.policy.M, 0.0)

    def test_open_loop_fixed_risk_uses_cached_h_only_qp(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            controller_variant=VARIANT_MULTIMODAL_OL,
            safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            risk_allocation_mode=RISK_FIXED,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, 12.0]), result.prediction)

        self.assertTrue(np.isfinite(solution.action))
        self.assertIn("cached_open_loop_fixed_qp=1", solution.solver_message)
        self.assertEqual(solution.num_decision_variables, config.horizon + 1)
        np.testing.assert_allclose(solution.policy.h[0], solution.policy.h[1])
        np.testing.assert_allclose(solution.policy.K, 0.0)
        np.testing.assert_allclose(solution.policy.M, 0.0)

    def test_open_loop_brake_distance_uses_cached_h_only_qp(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            controller_variant=VARIANT_MULTIMODAL_OL,
            safety_constraint_mode=SAFETY_BRAKE_DISTANCE,
            risk_allocation_mode=RISK_FIXED,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)

        self.assertTrue(np.isfinite(solution.action))
        self.assertEqual(solution.solve_path, "cached_open_loop_brake_distance_qp")
        self.assertIn("cached_open_loop_brake_distance_qp=1", solution.solver_message)
        self.assertEqual(solution.num_decision_variables, config.horizon + 1)
        np.testing.assert_allclose(solution.policy.h[0], solution.policy.h[1])
        np.testing.assert_allclose(solution.policy.K, 0.0)
        np.testing.assert_allclose(solution.policy.M, 0.0)

    def test_optimized_eta_controller_solves_one_synthetic_step(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            fixed_risk_level=0.05,
            safety_constraint_mode=SAFETY_SCALAR_CHANCE,
            risk_allocation_mode=RISK_OPTIMIZED_ETA,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                probabilities=[0.6, 0.4],
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        result = predictor.predict(np.array([0.0, 12.0]), np.array([38.0, 3.5, 10.0]))
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), result.prediction)

        self.assertTrue(np.isfinite(solution.action))
        self.assertIsNotNone(solution.eta_levels)
        self.assertIsNotNone(solution.safe_probability_levels)
        self.assertGreaterEqual(solution.eta_min, config.eta_std_min - 1.0e-9)
        self.assertLessEqual(solution.eta_max, config.eta_std_max + 1.0e-9)
        self.assertGreaterEqual(
            float(np.dot(result.prediction.probabilities, solution.safe_probability_levels)),
            solution.target_safe_probability - 1.0e-6,
        )

    def test_higher_cutin_probability_changes_response(self):
        config = NairACCConfig(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            max_slsqp_iter=40,
            optimizer_ftol=1.0e-3,
        )
        predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=3,
                dt=0.2,
                cutin_start_delay=0.0,
                blocking_alpha=0.0,
            )
        )
        low = predictor.predict(
            np.array([0.0, 12.0]),
            np.array([30.0, 3.5, 9.0]),
            probabilities=[0.8, 0.2],
        )
        high = predictor.predict(
            np.array([0.0, 12.0]),
            np.array([30.0, 3.5, 9.0]),
            probabilities=[0.2, 0.8],
        )
        low_action = NairACCSMPC(config).solve(np.array([0.0, 12.0]), low.prediction).action
        high_action = NairACCSMPC(config).solve(np.array([0.0, 12.0]), high.prediction).action

        self.assertLessEqual(high_action, low_action + 1.0e-6)


class TestConfidenceChanceConstraint(unittest.TestCase):
    """``confidence_chance`` safety mode: tightening = q(beta) * safety_std per cell.

    Mode 0 below is an ego-lane lead (NaN confidence, deterministic gap); mode 1
    is a cut-in hypothesis carrying the Benciolini confidence ``beta``.
    """

    @staticmethod
    def _config(**overrides):
        kwargs = dict(
            horizon=3,
            dt=0.2,
            desired_speed=14.0,
            num_modes=2,
            safety_constraint_mode=SAFETY_CONFIDENCE_CHANCE,
            risk_allocation_mode=RISK_FIXED,
            max_slsqp_iter=35,
            optimizer_ftol=1.0e-3,
        )
        kwargs.update(overrides)
        return NairACCConfig(**kwargs)

    @staticmethod
    def _prediction(horizon=3, confidence=0.5, lead_s0=60.0):
        means = np.zeros((2, horizon + 1, 2))
        means[:, :, 0] = lead_s0 + 10.0 * 0.2 * np.arange(horizon + 1)
        means[:, :, 1] = 10.0
        covariances = np.zeros((2, horizon + 1, 2, 2))
        covariances[:, :, 0, 0] = 4.0
        covariances[:, :, 1, 1] = 0.09
        chance_confidence = np.full((2, horizon + 1), np.nan)
        chance_confidence[1, :] = confidence
        return MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([0.6, 0.4]),
            covariances=covariances,
            chance_confidence=chance_confidence,
        )

    def test_confidence_tightening_scales_safety_std(self):
        config = self._config()
        covariance = np.diag([4.0, 0.09])
        self.assertAlmostEqual(
            confidence_tightening(covariance, 0.95, config),
            1.959964 * safety_std(covariance, config),
            places=5,
        )
        self.assertEqual(confidence_tightening(covariance, float("nan"), config), 0.0)
        self.assertAlmostEqual(confidence_tightening(covariance, 0.0, config), 0.0, places=9)
        self.assertAlmostEqual(confidence_quantile(0.5), 0.67449, places=4)

    def test_rejects_optimized_eta(self):
        with self.assertRaises(ValueError):
            self._config(risk_allocation_mode=RISK_OPTIMIZED_ETA)

    def test_prediction_validates_confidence(self):
        with self.assertRaises(ValueError):
            self._prediction(confidence=1.5)
        default = MultimodalLeadPrediction(
            means=np.zeros((2, 4, 2)), probabilities=np.array([0.5, 0.5])
        )
        self.assertTrue(np.all(np.isnan(default.chance_confidence)))

    def test_controller_uses_cached_qp_and_tightens_only_confidence_cells(self):
        config = self._config()
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), self._prediction(confidence=0.5))

        self.assertTrue(np.isfinite(solution.action))
        self.assertEqual(solution.solve_path, "cached_feedback_confidence_chance_qp")
        self.assertIn("cached_feedback_confidence_chance_qp", solution.solver_message)
        expected = confidence_quantile(0.5) * safety_std(np.diag([4.0, 0.09]), config)
        self.assertAlmostEqual(solution.tightening_max, expected, places=6)
        self.assertAlmostEqual(solution.tightening_min, 0.0)
        self.assertTrue(solution.feasible)
        self.assertGreaterEqual(solution.chance_margin_min, -1.0e-6)

    def test_open_loop_variant_uses_cached_qp(self):
        config = self._config(controller_variant=VARIANT_MULTIMODAL_OL)
        solution = NairACCSMPC(config).solve(np.array([0.0, 12.0]), self._prediction(confidence=0.5))
        self.assertTrue(np.isfinite(solution.action))
        self.assertEqual(solution.solve_path, "cached_open_loop_confidence_chance_qp")
        self.assertGreater(solution.tightening_max, 0.0)

    def test_trim_keeps_confidence_and_clearance(self):
        prediction = self._prediction(horizon=5, confidence=0.5)
        prediction.clearance_scale[1, :] = 0.4
        trimmed = NairACCSMPC(self._config())._trim_prediction(prediction)
        self.assertEqual(trimmed.horizon, 3)
        np.testing.assert_allclose(trimmed.clearance_scale[1], 0.4)
        np.testing.assert_allclose(trimmed.chance_confidence[1], 0.5)
        self.assertTrue(np.all(np.isnan(trimmed.chance_confidence[0])))

    def test_reference_asks_for_the_sigma_term(self):
        config = self._config()
        ego = np.array([0.0, 10.0])
        # A lead 20 m ahead binds the reference (standoff is 4.5 + 3 + 1.3 * 10 = 20.5 m).
        with_sigma = OldACCReferenceAdapter(config).generate(
            ego, self._prediction(confidence=0.5, lead_s0=20.0))
        without = OldACCReferenceAdapter(config).generate(
            ego, self._prediction(confidence=float("nan"), lead_s0=20.0))
        sigma_term = confidence_quantile(0.5) * safety_std(np.diag([4.0, 0.09]), config)
        # Mode 1 carries the confidence: its reference sits sigma_term further back.
        np.testing.assert_allclose(
            without.s_ref[1, 1:] - with_sigma.s_ref[1, 1:], sigma_term, rtol=1.0e-9
        )
        # Mode 0 is deterministic either way.
        np.testing.assert_allclose(with_sigma.s_ref[0], without.s_ref[0])

    @staticmethod
    def _two_lane_vehicles(horizon, far_s0):
        """Near lead 27 m ahead at 13 m/s, a slower one further ahead at 7 m/s."""
        steps = np.arange(horizon + 1) * 0.2
        near = np.column_stack((27.0 + 13.0 * steps, np.full(horizon + 1, 13.0)))
        far = np.column_stack((far_s0 + 7.0 * steps, np.full(horizon + 1, 7.0)))
        return near, np.stack((near, far), axis=1)[None]

    @staticmethod
    def _shadowed_prediction(horizon, candidates, near, far_active):
        mask = np.zeros((1, horizon + 1, 2), dtype=bool)
        mask[:, :, 0] = True
        mask[:, :, 1] = far_active
        return MultimodalLeadPrediction(
            means=near[None].copy(),
            probabilities=np.array([1.0]),
            covariances=np.zeros((1, horizon + 1, 2, 2)),
            active_mask=np.ones((1, horizon + 1), dtype=bool),
            lead_candidates=candidates,
            lead_candidate_mask=mask,
        )

    def _shadowed_reference(self, far_s0, far_active=True, horizon=15):
        config = NairACCConfig(
            horizon=horizon, dt=0.2, desired_speed=17.0, num_modes=1,
            a_min=-3.0, a_max=2.0,
            safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE)
        near, candidates = self._two_lane_vehicles(horizon, far_s0)
        return OldACCReferenceAdapter(config).generate(
            np.array([0.0, 13.0]),
            self._shadowed_prediction(horizon, candidates, near, far_active))

    def test_reference_speed_answers_to_a_slower_vehicle_behind_the_near_lead(self):
        """A slower vehicle further ahead has to ease the reference speed while a
        faster one is still between: the ego ends up behind it either way, and
        waiting for the near lead to vacate the step turns that into a step."""
        near_only = self._shadowed_reference(54.0, far_active=False)
        shadowed = self._shadowed_reference(54.0)
        # Seeing only the near lead, the reference just holds its 13 m/s.
        np.testing.assert_allclose(near_only.v_ref[0], 13.0, atol=0.2)
        # Seeing past it, the reference eases off from the first step and keeps
        # going down over the horizon, within the ego's own acceleration limits.
        self.assertLess(shadowed.v_ref[0, 0], 11.0)
        self.assertLess(shadowed.v_ref[0, -1], 9.0)
        # The position reference is left to the near lead: at a given step it is
        # the tighter bound, and it is what the collision constraint uses.
        np.testing.assert_allclose(shadowed.s_ref[0], near_only.s_ref[0])

    def test_a_single_vehicle_ahead_is_untouched_by_the_shadowed_leads(self):
        """The baselines see one lead; nothing about them may change."""
        config = NairACCConfig(
            horizon=15, dt=0.2, desired_speed=17.0, num_modes=1,
            a_min=-3.0, a_max=2.0,
            safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE)
        near, _ = self._two_lane_vehicles(15, 54.0)
        without = OldACCReferenceAdapter(config).generate(
            np.array([0.0, 13.0]),
            MultimodalLeadPrediction(
                means=near[None].copy(),
                probabilities=np.array([1.0]),
                covariances=np.zeros((1, 16, 2, 2)),
                active_mask=np.ones((1, 16), dtype=bool),
            ))
        with_masked_off = self._shadowed_reference(54.0, far_active=False)
        np.testing.assert_allclose(with_masked_off.s_ref, without.s_ref)
        np.testing.assert_allclose(with_masked_off.v_ref, without.v_ref)

    def test_the_shadowed_pull_falls_away_with_distance(self):
        """The blend is scaled by the distance to the vehicle, so this is
        anticipation and not a speed limit copied off every car in sight."""
        near_only = self._shadowed_reference(54.0, far_active=False)
        close = self._shadowed_reference(54.0)
        distant = self._shadowed_reference(300.0)
        self.assertGreater(near_only.v_ref[0, -1] - close.v_ref[0, -1], 4.0)
        # 300 m ahead of a 3 s horizon it leaves the reference alone entirely.
        np.testing.assert_allclose(distant.v_ref[0, -1], near_only.v_ref[0, -1])
        self.assertLess(near_only.v_ref[0, 0] - distant.v_ref[0, 0], 0.6)

    def test_a_faster_vehicle_further_ahead_never_raises_the_reference(self):
        near_only = self._shadowed_reference(54.0, far_active=False)
        config = NairACCConfig(
            horizon=15, dt=0.2, desired_speed=17.0, num_modes=1,
            a_min=-3.0, a_max=2.0,
            safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE)
        near, candidates = self._two_lane_vehicles(15, 54.0)
        candidates[0, :, 1, 1] = 16.0   # the far vehicle now outruns the lead
        faster = OldACCReferenceAdapter(config).generate(
            np.array([0.0, 13.0]),
            self._shadowed_prediction(15, candidates, near, True))
        np.testing.assert_allclose(faster.v_ref[0], near_only.v_ref[0])


if __name__ == "__main__":
    unittest.main()
