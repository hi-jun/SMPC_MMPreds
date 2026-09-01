import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    REL_RIGHT_ADJACENT,
    build_multitarget_lead_prediction,
    build_policy_sharing_tree_from_effective_leads,
    ego_lane_membership_split_step_from_mask,
    ego_pose_relative_frenet,
    process_vehicle_prediction,
    reconstruct_acc_probabilities,
)
from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCConfig,
    OldACCReferenceAdapter,
)


class StraightFrenetHandler:
    def __init__(self, yaw=0.0):
        s = np.linspace(0.0, 100.0, 201)
        x = s * np.cos(float(yaw))
        y = s * np.sin(float(yaw))
        self.trajectory = np.column_stack((
            s,
            x,
            y,
            np.full_like(s, float(yaw)),
            np.zeros_like(s),
        ))

    def convert_global_to_frenet_frame(self, x_query, y_query, psi_query):
        del psi_query
        xy = self.trajectory[:, 1:3]
        point = np.array([float(x_query), float(y_query)])
        closest = int(np.argmin(np.linalg.norm(xy - point.reshape(1, 2), axis=1)))
        yaw = float(self.trajectory[closest, 3])
        rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        error = rot @ (point - xy[closest])
        return float(self.trajectory[closest, 0]), float(error[1]), 0.0


class TestSTDAN3IntACCIntegration(unittest.TestCase):
    def test_mode_reconstruction_left_right_and_ego_lane(self):
        raw = np.array([0.2, 0.3, 0.5])

        left = reconstruct_acc_probabilities(raw, REL_LEFT_ADJACENT)["acc_mode_prob"]
        self.assertAlmostEqual(left["cutin"], 0.5)
        self.assertAlmostEqual(left["lk"], 0.5)

        right = reconstruct_acc_probabilities(raw, REL_RIGHT_ADJACENT)["acc_mode_prob"]
        self.assertAlmostEqual(right["cutin"], 0.3)
        self.assertAlmostEqual(right["lk"], 0.7)

        ego = reconstruct_acc_probabilities(raw, REL_EGO_LANE)["acc_mode_prob"]
        self.assertAlmostEqual(ego["cutout"], 0.8)
        self.assertAlmostEqual(ego["lk"], 0.2)

    def test_ego_pose_relative_frenet(self):
        frenet = ego_pose_relative_frenet(
            np.array([[10.0, 0.0], [0.0, 2.0], [0.0, -2.0]]),
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
        )
        self.assertGreater(frenet[0, 0], 0.0)
        self.assertAlmostEqual(frenet[0, 1], 0.0)
        self.assertGreater(frenet[1, 1], 0.0)
        self.assertLess(frenet[2, 1], 0.0)

    def test_ego_lane_membership_split_detection(self):
        same = np.array([[True, True, True], [True, True, True]])
        self.assertEqual(ego_lane_membership_split_step_from_mask(same), 2)
        split = np.array([[True, True, True], [True, False, False]])
        self.assertEqual(ego_lane_membership_split_step_from_mask(split), 1)

    def test_effective_lead_history_policy_tree_splits_in_stages(self):
        horizon = 30
        lead = (1, "lk")
        inactive = ("inactive",)
        cutin = (2, "cutin")
        histories = [
            [lead] * horizon,
            [lead] * 14 + [inactive] * (horizon - 14),
            [lead] * 25 + [cutin] * (horizon - 25),
            [lead] * 14 + [inactive] * 11 + [cutin] * (horizon - 25),
        ]

        tree = build_policy_sharing_tree_from_effective_leads(histories, horizon)

        self.assertEqual(tree.policy_split_steps, (14, 25))
        self.assertEqual(len(set(tree.policy_groups_by_step[:, 13])), 1)
        self.assertEqual(len(set(tree.policy_groups_by_step[:, 14])), 2)
        self.assertEqual(len(set(tree.policy_groups_by_step[:, 24])), 2)
        self.assertEqual(len(set(tree.policy_groups_by_step[:, 25])), 4)

    def test_vehicle_postprocess_active_mask(self):
        raw = {
            "vehicle_id": 10,
            "raw_intention_prob": np.array([0.2, 0.3, 0.5]),
            "pred_traj_frenet": np.array([
                [[20.0, 3.5], [22.0, 3.5], [24.0, 3.5]],
                [[20.0, 3.5], [22.0, 1.0], [24.0, 0.0]],
                [[20.0, 3.5], [22.0, 2.0], [24.0, 0.0]],
            ]),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 2.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = process_vehicle_prediction(
            raw,
            REL_RIGHT_ADJACENT,
            current_frenet_state=np.array([18.0, 3.5, 8.0]),
            horizon=3,
            dt=0.2,
            ego_lane_threshold=0.5,
        )
        cutin = [mode for mode in processed.mode_predictions if mode.mode_name == "cutin"][0]
        self.assertFalse(cutin.active_mask[0])
        self.assertTrue(cutin.active_mask[-1])

    def test_multitarget_prediction_uses_inactive_mask_instead_of_far_lead(self):
        raw = {
            "vehicle_id": 10,
            "raw_intention_prob": np.array([1.0, 0.0, 0.0]),
            "pred_traj_frenet": np.array([[[20.0, 3.5], [22.0, 3.5], [24.0, 3.5]]] * 3),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 0.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = process_vehicle_prediction(
            raw,
            REL_LEFT_ADJACENT,
            current_frenet_state=np.array([18.0, 3.5, 8.0]),
            horizon=3,
            dt=0.2,
            ego_lane_threshold=0.5,
        )
        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=3,
            desired_speed=15.0,
            num_modes=2,
        )
        self.assertFalse(np.any(prediction.active_mask))

    def test_multitarget_prediction_exports_k_group_map(self):
        raw = {
            "vehicle_id": 10,
            "raw_intention_prob": np.array([0.4, 0.6, 0.0]),
            "pred_traj_frenet": np.array([
                [[20.0, 3.5], [22.0, 3.5], [24.0, 3.5]],
                [[20.0, 3.5], [22.0, 0.0], [24.0, 0.0]],
                [[20.0, 3.5], [22.0, 3.5], [24.0, 3.5]],
            ]),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = process_vehicle_prediction(
            raw,
            REL_RIGHT_ADJACENT,
            current_frenet_state=np.array([18.0, 3.5, 8.0]),
            horizon=3,
            dt=0.2,
            ego_lane_threshold=0.5,
        )

        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=3,
            desired_speed=15.0,
            num_modes=2,
        )

        self.assertIsNotNone(prediction.k_group_map)
        self.assertEqual(prediction.k_group_map.shape, (prediction.num_modes, 3))
        self.assertIn("inactive", prediction.k_group_names)
        self.assertGreaterEqual(len(prediction.k_group_names), 2)

    def test_velocity_distribution_covariance_uses_inverse_std(self):
        adapter = STDAN3IntACCAdapter(load_model=False)
        raw_tight = np.array([[[0.0, 5.0, 10.0, 10.0, 0.0]]], dtype=float)
        raw_wide = np.array([[[0.0, 5.0, 10.0, 2.0, 0.0]]], dtype=float)
        _, pos_tight = adapter._distribution_covariances_global(raw_tight, [0.0, 0.0, 0.0])
        _, pos_wide = adapter._distribution_covariances_global(raw_wide, [0.0, 0.0, 0.0])
        frenet_tight = adapter._global_covariances_to_frenet(
            pos_tight,
            np.array([[[1.0, 0.0]]]),
            StraightFrenetHandler(),
        )
        frenet_wide = adapter._global_covariances_to_frenet(
            pos_wide,
            np.array([[[1.0, 0.0]]]),
            StraightFrenetHandler(),
        )
        self.assertGreater(frenet_wide[0, 0, 0, 0], frenet_tight[0, 0, 0, 0])

    def test_velocity_distribution_covariance_uses_correlation(self):
        adapter = STDAN3IntACCAdapter(load_model=False)
        raw_pos_corr = np.array([[[0.0, 5.0, 3.0, 3.0, 0.8]]], dtype=float)
        raw_neg_corr = np.array([[[0.0, 5.0, 3.0, 3.0, -0.8]]], dtype=float)
        _, pos_corr = adapter._distribution_covariances_global(raw_pos_corr, [0.0, 0.0, np.pi / 4.0])
        _, neg_corr = adapter._distribution_covariances_global(raw_neg_corr, [0.0, 0.0, np.pi / 4.0])
        frenet_pos = adapter._global_covariances_to_frenet(
            pos_corr,
            np.array([[[1.0, 0.0]]]),
            StraightFrenetHandler(),
        )
        frenet_neg = adapter._global_covariances_to_frenet(
            neg_corr,
            np.array([[[1.0, 0.0]]]),
            StraightFrenetHandler(),
        )
        self.assertNotAlmostEqual(
            float(frenet_pos[0, 0, 0, 0]),
            float(frenet_neg[0, 0, 0, 0]),
        )

    def test_processed_prediction_carries_predictor_covariance_to_effective_lead(self):
        raw = {
            "vehicle_id": 10,
            "raw_intention_prob": np.array([0.0, 1.0, 0.0]),
            "pred_traj_frenet": np.array([
                [[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]],
                [[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]],
                [[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]],
            ]),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "pred_lead_covariance": np.zeros((3, 3, 2, 2)),
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        raw["pred_lead_covariance"][1, :, 0, 0] = [1.2, 1.4, 1.6]
        raw["pred_lead_covariance"][1, :, 1, 1] = [0.2, 0.3, 0.4]
        processed = process_vehicle_prediction(
            raw,
            REL_RIGHT_ADJACENT,
            current_frenet_state=np.array([18.0, 0.0, 8.0]),
            horizon=3,
            dt=0.2,
            ego_lane_threshold=0.5,
        )
        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=3,
            desired_speed=15.0,
            num_modes=1,
        )
        self.assertAlmostEqual(prediction.covariances[0, 0, 0, 0], 0.0)
        self.assertAlmostEqual(prediction.covariances[0, 1, 0, 0], 1.2)
        self.assertAlmostEqual(prediction.covariances[0, 3, 1, 1], 0.4)

    def test_multitarget_prediction_keeps_fallback_covariance_without_predictor_covariance(self):
        raw = {
            "vehicle_id": 10,
            "raw_intention_prob": np.array([0.0, 1.0, 0.0]),
            "pred_traj_frenet": np.array([
                [[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]],
                [[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]],
                [[20.0, 0.0], [22.0, 0.0], [24.0, 0.0]],
            ]),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = process_vehicle_prediction(
            raw,
            REL_RIGHT_ADJACENT,
            current_frenet_state=np.array([18.0, 0.0, 8.0]),
            horizon=3,
            dt=0.2,
            ego_lane_threshold=0.5,
        )
        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=3,
            desired_speed=15.0,
            num_modes=1,
        )
        self.assertAlmostEqual(prediction.covariances[0, 1, 0, 0], 0.25)
        self.assertAlmostEqual(prediction.covariances[0, 1, 1, 1], 0.09)

    def test_multitarget_prediction_includes_ego_lead_and_cutin_joint_modes(self):
        ego_lead_raw = {
            "vehicle_id": 20,
            "raw_intention_prob": np.array([0.5, 0.5, 0.0]),
            "pred_traj_frenet": np.array([
                [[32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0]],
                [[32.0, 0.4], [34.0, 1.0], [36.0, 2.0], [38.0, 3.5]],
                [[32.0, 0.0], [34.0, 0.0], [36.0, 0.0], [38.0, 0.0]],
            ]),
            "raw_pred_vel": np.ones((3, 4, 2)) * 10.0,
            "valid_mask": np.ones((3, 4), dtype=bool),
        }
        cutin_raw = {
            "vehicle_id": 10,
            "raw_intention_prob": np.array([0.5, 0.5, 0.0]),
            "pred_traj_frenet": np.array([
                [[22.0, 3.5], [24.0, 3.5], [26.0, 3.5], [28.0, 3.5]],
                [[22.0, 2.0], [24.0, 1.0], [26.0, 0.0], [28.0, 0.0]],
                [[22.0, 3.5], [24.0, 3.5], [26.0, 3.5], [28.0, 3.5]],
            ]),
            "raw_pred_vel": np.ones((3, 4, 2)) * 10.0,
            "valid_mask": np.ones((3, 4), dtype=bool),
        }
        ego_lead = process_vehicle_prediction(
            ego_lead_raw,
            REL_EGO_LANE,
            current_frenet_state=np.array([30.0, 0.0, 10.0]),
            horizon=4,
            dt=0.2,
            ego_lane_threshold=0.5,
        )
        cutin = process_vehicle_prediction(
            cutin_raw,
            REL_RIGHT_ADJACENT,
            current_frenet_state=np.array([20.0, 3.5, 10.0]),
            horizon=4,
            dt=0.2,
            ego_lane_threshold=0.5,
        )

        prediction, metadata = build_multitarget_lead_prediction(
            [ego_lead, cutin],
            ego_state=np.array([0.0, 10.0]),
            horizon=4,
            desired_speed=15.0,
            num_modes=4,
        )

        self.assertEqual(prediction.num_modes, 4)
        self.assertEqual(len(metadata), 4)
        self.assertIsNotNone(prediction.policy_tree)
        self.assertEqual(prediction.k_group_map.shape, (4, 4))
        self.assertIn("20:lk", prediction.k_group_names)

    def test_inactive_mask_keeps_default_reference(self):
        config = NairACCConfig(horizon=3, dt=0.2, desired_speed=15.0, num_modes=1)
        adapter = OldACCReferenceAdapter(config)
        means = np.zeros((1, 4, 2))
        means[:, :, 0] = 5.0
        means[:, :, 1] = 0.0
        prediction = MultimodalLeadPrediction(
            means=means,
            probabilities=np.array([1.0]),
            active_mask=np.zeros((1, 4), dtype=bool),
        )
        reference = adapter.generate(np.array([0.0, 10.0]), prediction)
        self.assertTrue(np.all(reference.v_ref[0] >= 10.0))
        self.assertTrue(np.all(reference.s_ref[0] >= 0.0))

    def test_adapter_tensor_shapes_without_loading_model(self):
        adapter = STDAN3IntACCAdapter(load_model=False)
        steps = adapter.full_length
        t = np.arange(steps, dtype=float) * adapter.dt
        trackings = {
            1: np.column_stack((10.0 * t, np.zeros(steps), np.zeros(steps))),
            2: np.column_stack((10.0 * t + 8.0, np.ones(steps) * 3.0, np.zeros(steps))),
        }
        tensors = adapter.make_tensor(1, trackings[1][-1], trackings)
        hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls = tensors
        self.assertEqual(tuple(hist.shape), (31, 1, 2))
        self.assertEqual(tuple(va.shape), (31, 1, 2))
        self.assertEqual(tuple(mask.shape), (1, 3, 13, 64))
        self.assertEqual(str(mask.dtype), "torch.bool")
        self.assertEqual(tuple(nbrs.shape), (31, 1, 2))
        self.assertEqual(tuple(nbrsva.shape), (31, 1, 2))
        self.assertEqual(tuple(lane.shape), (31, 1, 1))
        self.assertEqual(tuple(nbrslane.shape), (31, 1, 1))
        self.assertEqual(tuple(cls.shape), (31, 1, 1))
        self.assertEqual(tuple(nbrscls.shape), (31, 1, 1))

    def test_adapter_uses_ngsim_right_positive_lateral_axis(self):
        adapter = STDAN3IntACCAdapter(load_model=False)
        history = np.array([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 3.0, 0.0],
        ])
        ngsim = adapter._history_to_ngsim(history, target_state=[0.0, 0.0, 0.0], model_yaw=0.0)
        self.assertAlmostEqual(ngsim[-2, 0], 0.0)
        self.assertAlmostEqual(ngsim[-2, 1], 5.0)
        self.assertAlmostEqual(ngsim[-1, 0], -3.0)
        self.assertAlmostEqual(ngsim[-1, 1], 5.0)


if __name__ == "__main__":
    unittest.main()
