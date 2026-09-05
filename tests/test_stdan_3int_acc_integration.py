import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    ACCModePrediction,
    REL_EGO_LANE,
    REL_RIGHT_ADJACENT as _REL_RIGHT_ADJACENT,
    chance_cutin_clearance,
    chance_tolerance,
    cutin_clearance_scale,
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
    NairACCSMPC,
    OldACCReferenceAdapter,
    confidence_quantile,
    required_standoff,
    safety_function,
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


class TestCutInProbabilityGate(unittest.TestCase):
    """Low-probability cut-in modes must stop generating longitudinal constraints.

    This is the 1D ACC projection of the vanishing chance constraint from
    Benciolini et al. (T-IV 2023): a candidate trajectory whose estimated
    probability is negligible should not restrict the ego vehicle at all.
    """

    HORIZON = 3

    @staticmethod
    def _raw(intention_prob, cutin_d_profile, other_d=3.5):
        """Right-adjacent target; raw mode 1 (LLC) is the cut-in candidate."""
        s = np.array([20.0, 22.0, 24.0])
        stay = np.column_stack((s, np.full_like(s, other_d)))
        cutin = np.column_stack((s, np.asarray(cutin_d_profile, dtype=float)))
        return {
            "vehicle_id": 10,
            "raw_intention_prob": np.asarray(intention_prob, dtype=float),
            "pred_traj_frenet": np.stack((stay, cutin, stay)),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }

    def _process(self, raw, current_d=3.5, relation=REL_RIGHT_ADJACENT, **kwargs):
        return process_vehicle_prediction(
            raw,
            relation,
            current_frenet_state=np.array([18.0, current_d, 8.0]),
            horizon=self.HORIZON,
            dt=0.2,
            ego_lane_threshold=0.5,
            **kwargs
        )

    @staticmethod
    def _mode(processed, name):
        for mode in processed.mode_predictions:
            if mode.mode_name == name:
                return mode
        return None

    def test_low_probability_cutin_is_active_without_gate(self):
        raw = self._raw([0.90, 0.05, 0.05], [3.5, 0.0, 0.0])
        cutin = self._mode(self._process(raw), "cutin")

        self.assertIsNotNone(cutin)
        self.assertAlmostEqual(cutin.probability, 0.05, places=6)
        self.assertTrue(cutin.active_mask.any(), "baseline: cut-in constrains the ego")

    def test_low_probability_cutin_is_gated_out(self):
        raw = self._raw([0.90, 0.05, 0.05], [3.5, 0.0, 0.0])
        processed = self._process(raw, cutin_probability_threshold=0.10)
        cutin = self._mode(processed, "cutin")

        self.assertIsNotNone(cutin)
        self.assertFalse(cutin.active_mask.any(), "gated cut-in must not constrain")

        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=self.HORIZON,
            desired_speed=15.0,
            num_modes=2,
        )
        self.assertFalse(prediction.active_mask.any())

    def test_high_probability_cutin_survives_gate(self):
        raw = self._raw([0.30, 0.60, 0.10], [3.5, 0.0, 0.0])
        cutin = self._mode(self._process(raw, cutin_probability_threshold=0.10), "cutin")

        self.assertIsNotNone(cutin)
        self.assertAlmostEqual(cutin.probability, 0.60, places=6)
        self.assertTrue(cutin.active_mask.any(), "likely cut-in must still constrain")

    def test_gate_drops_a_negligible_lane_keeping_hypothesis_but_keeps_the_cutout_mode(self):
        s = np.array([20.0, 22.0, 24.0])
        ahead = np.column_stack((s, np.zeros_like(s)))
        raw = {
            "vehicle_id": 11,
            "raw_intention_prob": np.array([0.02, 0.49, 0.49]),
            "pred_traj_frenet": np.stack((ahead, ahead, ahead)),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = self._process(
            raw, current_d=0.0, relation=REL_EGO_LANE, cutin_probability_threshold=0.9
        )

        self.assertIsNone(self._mode(processed, "cutin"))
        lead = self._mode(processed, "lk")
        self.assertIsNotNone(lead)
        self.assertAlmostEqual(lead.probability, 0.02, places=6)
        # A 2 % "stays in lane" hypothesis no longer holds full standoff for the
        # whole horizon; the vehicle itself is still constrained through the
        # cutout mode for every step it is predicted present.
        self.assertFalse(lead.active_mask.any(), "negligible lk hypothesis vanishes")
        cutout = self._mode(processed, "cutout")
        self.assertIsNotNone(cutout)
        self.assertTrue(cutout.active_mask.all(), "cutout mode is never gated")

    def test_gate_preserves_current_ego_lane_occupancy(self):
        """A target already straddling the ego lane stays constrained at step 0."""
        raw = self._raw([0.90, 0.05, 0.05], [3.5, 0.0, 0.0])
        processed = self._process(raw, current_d=0.3, cutin_probability_threshold=0.10)

        self.assertFalse(self._mode(processed, "cutin").active_mask.any())
        lk = self._mode(processed, "lk")
        self.assertTrue(lk.active_mask[0], "observed encroachment must survive the gate")

    def test_threshold_zero_is_a_no_op(self):
        raw = self._raw([0.90, 0.05, 0.05], [3.5, 0.0, 0.0])
        baseline = self._mode(self._process(raw), "cutin")
        gated = self._mode(self._process(raw, cutin_probability_threshold=0.0), "cutin")

        np.testing.assert_array_equal(baseline.active_mask, gated.active_mask)


class TestCutInClearanceRamp(unittest.TestCase):
    """Standoff shrinks continuously with the cut-in probability.

    Benciolini et al. (T-IV 2023) Sec. V-B: the forbidden region scales with
    sqrt(zeta(beta_j)), and Remark 5 requires the footprint to sit inside that
    scaling so an unlikely mode stops reserving space entirely.
    """

    REF = 0.95
    HORIZON = 3

    def test_chance_tolerance_matches_closed_form(self):
        for beta in (0.95, 0.50, 0.10, 0.02):
            self.assertAlmostEqual(chance_tolerance(beta), -2.0 * np.log(1.0 - beta), places=9)
        self.assertAlmostEqual(np.sqrt(chance_tolerance(0.95)), 2.4478, places=3)

    def test_scale_is_monotone_and_vanishes(self):
        scales = [cutin_clearance_scale(p, self.REF) for p in (0.95, 0.50, 0.20, 0.10, 0.02)]
        self.assertTrue(all(a > b for a, b in zip(scales, scales[1:])), scales)
        self.assertAlmostEqual(scales[0], 1.0, places=9)
        self.assertAlmostEqual(scales[1], 0.4810, places=3)
        self.assertAlmostEqual(scales[4], 0.0821, places=3)
        self.assertLess(cutin_clearance_scale(1e-6, self.REF), 0.01)

    def test_scale_is_capped_at_one_and_disabled_by_default(self):
        self.assertEqual(cutin_clearance_scale(0.99, self.REF), 1.0)
        self.assertEqual(cutin_clearance_scale(0.02, 0.0), 1.0)

    def test_ramp_only_touches_cutin_modes(self):
        gate = TestCutInProbabilityGate()
        processed = gate._process(
            gate._raw([0.60, 0.30, 0.10], [3.5, 0.0, 0.0]),
            cutin_clearance_ramp_ref=self.REF,
        )
        by_name = {m.mode_name: m for m in processed.mode_predictions}
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)
        self.assertLess(by_name["cutin"].clearance_scale, 1.0)

    def test_ego_lane_lead_is_never_ramped(self):
        s = np.array([20.0, 22.0, 24.0])
        ahead = np.column_stack((s, np.zeros_like(s)))
        raw = {
            "vehicle_id": 11,
            "raw_intention_prob": np.array([0.02, 0.49, 0.49]),
            "pred_traj_frenet": np.stack((ahead, ahead, ahead)),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = TestCutInProbabilityGate()._process(
            raw, current_d=0.0, relation=REL_EGO_LANE, cutin_clearance_ramp_ref=self.REF
        )
        for mode in processed.mode_predictions:
            self.assertEqual(mode.clearance_scale, 1.0, mode.mode_name)

    def test_ramp_reaches_the_controller_prediction(self):
        gate = TestCutInProbabilityGate()
        processed = gate._process(
            gate._raw([0.90, 0.05, 0.05], [3.5, 0.0, 0.0]),
            cutin_clearance_ramp_ref=self.REF,
        )
        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=self.HORIZON,
            desired_speed=15.0,
            num_modes=2,
        )
        active = prediction.active_mask
        self.assertTrue(active.any())
        scaled = prediction.clearance_scale[active]
        self.assertTrue(np.all(scaled < 1.0), scaled)
        self.assertTrue(np.all(prediction.clearance_scale[~active] == 1.0))

    def test_required_gap_shrinks_with_the_scale(self):
        config = NairACCConfig(horizon=self.HORIZON, dt=0.2, d0=3.0, time_headway=1.3,
                               vehicle_length=4.5)
        ego, lead = np.array([0.0, 8.0]), np.array([100.0, 8.0])
        standoff = required_standoff(ego, config)
        self.assertAlmostEqual(standoff, 4.5 + 3.0 + 1.3 * 8.0, places=9)

        full = safety_function(ego, lead, config)
        scale = cutin_clearance_scale(0.05, self.REF)
        ramped = safety_function(ego, lead, config, clearance_scale=scale)
        self.assertAlmostEqual(full, 100.0 - standoff, places=9)
        self.assertAlmostEqual(ramped, 100.0 - scale * standoff, places=9)
        self.assertGreater(ramped, full)

    def test_symbolic_margin_agrees_with_numeric(self):
        config = NairACCConfig(horizon=self.HORIZON, dt=0.2, d0=3.0, time_headway=1.3,
                               vehicle_length=4.5)
        controller = NairACCSMPC(config)
        ego, lead = np.array([0.0, 8.0]), np.array([100.0, 8.0])
        for scale in (1.0, 0.48, 0.08, 0.0):
            self.assertAlmostEqual(
                controller._symbolic_safety_margin(
                    float(lead[0]), float(ego[0]), float(ego[1]), clearance_scale=scale
                ),
                safety_function(ego, lead, config, clearance_scale=scale),
                places=9,
                msg=f"scale={scale}",
            )


class TestCutInChanceConstraint(unittest.TestCase):
    """Cut-in modes become Benciolini confidence chance constraints.

    Benciolini et al. (T-IV 2023) eq. (19a): the forbidden region is
    (sigma + l_o) * q(beta_j) with beta_j = mu_j.  In 1-D the quantile is
    Phi^-1((1 + beta) / 2), the tuned standoff plays l_o, and the controller
    adds the sigma term from the same beta (see acc_nair_smpc.confidence_tightening).
    """

    REF = 0.95
    HORIZON = 3

    @staticmethod
    def _fake_mode(name, probability):
        return ACCModePrediction(
            vehicle_id=7,
            mode_name=name,
            probability=probability,
            frenet=np.zeros((4, 3)),
            active_mask=np.ones(4, dtype=bool),
            raw_mode_indices=[1],
            ego_lane_membership_split_step=0,
        )

    def test_confidence_quantile_matches_closed_form(self):
        self.assertAlmostEqual(confidence_quantile(0.95), 1.95996, places=4)
        self.assertAlmostEqual(confidence_quantile(0.50), 0.67449, places=4)
        self.assertAlmostEqual(confidence_quantile(0.0), 0.0, places=9)
        values = [confidence_quantile(b) for b in (0.02, 0.1, 0.3, 0.5, 0.8, 0.95)]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])), values)

    def test_confidence_is_capped_and_scale_vanishes(self):
        modes = [self._fake_mode("cutin", p) for p in (0.99, 0.50, 0.02)]
        chance_cutin_clearance(modes, self.REF)
        self.assertEqual([m.chance_confidence for m in modes], [0.95, 0.50, 0.02])
        scales = [m.clearance_scale for m in modes]
        self.assertAlmostEqual(scales[0], 1.0, places=9)
        self.assertAlmostEqual(scales[1], 0.3441, places=3)
        self.assertAlmostEqual(scales[2], 0.0128, places=3)
        self.assertTrue(all(a > b for a, b in zip(scales, scales[1:])), scales)

    def test_reference_zero_is_a_no_op(self):
        mode = self._fake_mode("cutin", 0.3)
        self.assertEqual(chance_cutin_clearance([mode], 0.0), {})
        self.assertTrue(np.isnan(mode.chance_confidence))
        self.assertEqual(mode.clearance_scale, 1.0)

    def test_chance_only_touches_cutin_modes(self):
        gate = TestCutInProbabilityGate()
        processed = gate._process(
            gate._raw([0.60, 0.30, 0.10], [3.5, 0.0, 0.0]),
            cutin_chance_ref=self.REF,
        )
        by_name = {m.mode_name: m for m in processed.mode_predictions}
        self.assertTrue(np.isnan(by_name["lk"].chance_confidence))
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)
        cutin = by_name["cutin"]
        self.assertAlmostEqual(cutin.chance_confidence, min(cutin.probability, self.REF), places=9)
        self.assertAlmostEqual(
            cutin.clearance_scale,
            confidence_quantile(cutin.chance_confidence) / confidence_quantile(self.REF),
            places=9,
        )
        self.assertLess(cutin.clearance_scale, 1.0)
        self.assertIn("cutin", processed.branch_info["cutin_chance_confidences"])

    def test_ego_lane_cutout_mode_is_never_a_chance_cell(self):
        """Only the lane-keeping hypothesis of an ego-lane vehicle is relaxed.

        See ``chance_cutout_clearance`` and tests/test_chance_cutout_extension.py.
        """
        s = np.array([20.0, 22.0, 24.0])
        ahead = np.column_stack((s, np.zeros_like(s)))
        raw = {
            "vehicle_id": 11,
            "raw_intention_prob": np.array([0.02, 0.49, 0.49]),
            "pred_traj_frenet": np.stack((ahead, ahead, ahead)),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }
        processed = TestCutInProbabilityGate()._process(
            raw, current_d=0.0, relation=REL_EGO_LANE, cutin_chance_ref=self.REF
        )
        by_name = {m.mode_name: m for m in processed.mode_predictions}
        self.assertTrue(np.isnan(by_name["cutout"].chance_confidence))
        self.assertEqual(by_name["cutout"].clearance_scale, 1.0)
        self.assertTrue(np.isnan(by_name["lk"].chance_confidence), "standoff factor only")
        self.assertAlmostEqual(
            by_name["lk"].clearance_scale,
            confidence_quantile(0.02) / confidence_quantile(self.REF),
            places=9,
        )
        self.assertTrue(by_name["lk"].active_mask.all(), "no vanish threshold given")

    def test_chance_reaches_the_controller_prediction(self):
        gate = TestCutInProbabilityGate()
        processed = gate._process(
            gate._raw([0.90, 0.05, 0.05], [3.5, 0.0, 0.0]),
            cutin_chance_ref=self.REF,
        )
        beta = gate._mode(processed, "cutin").chance_confidence
        prediction, _ = build_multitarget_lead_prediction(
            [processed],
            ego_state=np.array([0.0, 10.0]),
            horizon=self.HORIZON,
            desired_speed=15.0,
            num_modes=2,
        )
        active = prediction.active_mask
        self.assertTrue(active.any())
        np.testing.assert_allclose(prediction.chance_confidence[active], beta)
        self.assertTrue(np.all(np.isnan(prediction.chance_confidence[~active])))
        np.testing.assert_allclose(
            prediction.clearance_scale[active],
            confidence_quantile(beta) / confidence_quantile(self.REF),
        )


class TestLaneSideConvention(unittest.TestCase):
    """+d is to the left, and the cut-in mapping has to be posed the same way.

    CARLA is left-handed and its get_left_lane()/get_right_lane() disagreed with the
    sign of the Frenet offset in 8 of 10 runs of the 2026-09-02 sweep, which routed
    a real rightward merge into the non-blocking lane-keeping mode.
    """

    def test_left_adjacent_target_cuts_in_by_turning_right(self):
        acc = reconstruct_acc_probabilities([0.10, 0.05, 0.85], REL_LEFT_ADJACENT)["acc_mode_prob"]
        self.assertAlmostEqual(acc["cutin"], 0.85, places=6)
        self.assertAlmostEqual(acc["lk"], 0.15, places=6)

    def test_right_adjacent_target_cuts_in_by_turning_left(self):
        acc = reconstruct_acc_probabilities([0.10, 0.85, 0.05], REL_RIGHT_ADJACENT)["acc_mode_prob"]
        self.assertAlmostEqual(acc["cutin"], 0.85, places=6)
        self.assertAlmostEqual(acc["lk"], 0.15, places=6)

    def test_mislabelling_the_side_hides_the_merge(self):
        """The sweep's failure mode: a leftward target labelled right-adjacent."""
        raw = [0.10, 0.05, 0.85]           # LK, LLC, RLC -- an RLC merge
        correct = reconstruct_acc_probabilities(raw, REL_LEFT_ADJACENT)["acc_mode_prob"]
        flipped = reconstruct_acc_probabilities(raw, REL_RIGHT_ADJACENT)["acc_mode_prob"]
        self.assertGreater(correct["cutin"], 0.8)
        self.assertLess(flipped["cutin"], 0.1)

    def test_frenet_offset_sign_maps_positive_d_to_the_left(self):
        from utils import frenet_trajectory_handler as fth  # noqa: F401
        # The agent's own helper, kept in sync with the Frenet transform it consumes.
        def relation_from_frenet_offset(d):
            if d > 0.3:
                return REL_LEFT_ADJACENT
            if d < -0.3:
                return _REL_RIGHT_ADJACENT
            return REL_EGO_LANE
        self.assertEqual(relation_from_frenet_offset(3.34), REL_LEFT_ADJACENT)
        self.assertEqual(relation_from_frenet_offset(-3.34), _REL_RIGHT_ADJACENT)
        self.assertEqual(relation_from_frenet_offset(0.0), REL_EGO_LANE)


if __name__ == "__main__":
    unittest.main()
