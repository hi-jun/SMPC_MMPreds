import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor.lstm.acc_adapter import LSTMACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
)
from test_stdan_3int_acc_integration import StraightFrenetHandler  # noqa: E402
from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    SAFETY_NOMINAL_SAFE_DISTANCE,
    VARIANT_FIXED_RISK,
)


ROAD_YAW = -np.pi / 2.0  # Town04 route direction: along -y
SPEED = 15.0
LANE_WIDTH = 3.5
POLICY = "acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance"


def straight_history(adapter, lateral_x, seconds=3.2):
    """Straight-line history along -y at SPEED, world RHS (x, y, yaw) rows."""
    steps = int(round(seconds / adapter.dt)) + 1
    t = np.arange(steps, dtype=float) * adapter.dt
    return np.column_stack((
        np.full(steps, float(lateral_x)),
        -SPEED * t,
        np.full(steps, ROAD_YAW),
    ))


class TestLSTMACCAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = LSTMACCAdapter()

    def test_straight_history_continues_straight_ahead(self):
        history = straight_history(self.adapter, 0.0)
        raw = self.adapter.predict_raw([7], {7: history}, model_yaw=ROAD_YAW)[7]
        traj = raw["raw_traj_xy"]
        self.assertEqual(traj.shape, (1, 50, 2))
        self.assertEqual(raw["raw_pred_vel"].shape, (1, 50, 2))
        np.testing.assert_allclose(raw["raw_intention_prob"], [1.0])
        # Lateral deviation over the first 3 s stays well inside the lane.
        self.assertLess(float(np.mean(np.abs(traj[0, :30, 0]))), 1.0)
        # Longitudinal progress (along -y) tracks 15 m/s within +-30 %.
        progress = -(traj[0, :, 1] - history[-1, 1])
        for step in (9, 19, 29, 39, 49):
            expected = SPEED * (step + 1) * self.adapter.dt
            self.assertLess(abs(progress[step] - expected), 0.3 * expected, msg=f"step {step}")
        speed = np.linalg.norm(raw["raw_pred_vel"][0, :30], axis=1)
        self.assertLess(abs(float(np.mean(speed)) - SPEED), 0.3 * SPEED)

    def _predict_acc(self, lateral_x, relation, lane_membership_fn=None):
        history = straight_history(self.adapter, lateral_x)
        handler = StraightFrenetHandler(yaw=ROAD_YAW)
        s_tv, d_tv, _ = handler.convert_global_to_frenet_frame(history[-1, 0], history[-1, 1], ROAD_YAW)
        return self.adapter.predict_acc(
            ego_state=np.array([s_tv - 20.0, SPEED]),
            target_states_frenet={7: np.array([s_tv, d_tv, SPEED])},
            target_relations={7: relation},
            trackings={7: history},
            frenet_handler=handler,
            horizon=15,
            desired_speed=SPEED,
            num_modes=1,
            controller_dt=0.2,
            lane_membership_fn=lane_membership_fn,
            model_yaw=ROAD_YAW,
            cutin_probability_threshold=0.1,
            gap_recovery_s=3.0,
        )

    def test_adjacent_lane_target_never_constrains(self):
        result = self._predict_acc(LANE_WIDTH, REL_LEFT_ADJACENT)
        prediction = result["prediction"]
        self.assertIsInstance(prediction, MultimodalLeadPrediction)
        self.assertEqual(prediction.num_modes, 1)
        self.assertFalse(np.asarray(prediction.active_mask, dtype=bool).any())
        target, = result["processed_targets"]
        mode, = target.mode_predictions
        self.assertEqual(mode.mode_name, "lstm")
        self.assertAlmostEqual(mode.probability, 1.0)
        self.assertEqual(mode.frenet.shape, (16, 3))
        self.assertEqual(target.branch_info["ego_lane_start_idx"], 999)
        self.assertEqual(target.branch_info["lane_membership_source"], "frenet_d_threshold")

    def test_ego_lane_target_is_active_from_step_zero(self):
        def lane_membership_fn(target_id, sampled_raw_traj_xy):
            # CARLA-waypoint stand-in: the ego lane is |x| < half a lane width,
            # step 0 is the current position.
            self.assertEqual(target_id, 7)
            self.assertEqual(sampled_raw_traj_xy.shape, (1, 15, 2))
            future = np.abs(sampled_raw_traj_xy[:, :, 0]) < LANE_WIDTH / 2.0
            return np.column_stack((np.ones((1, 1), dtype=bool), future))

        result = self._predict_acc(0.0, REL_EGO_LANE, lane_membership_fn)
        prediction = result["prediction"]
        self.assertEqual(prediction.num_modes, 1)
        self.assertTrue(np.asarray(prediction.active_mask, dtype=bool).all())
        np.testing.assert_allclose(prediction.probabilities, [1.0])
        target, = result["processed_targets"]
        self.assertEqual(target.branch_info["ego_lane_start_idx"], 0)
        self.assertEqual(target.branch_info["ego_lane_end_idx"], 15)
        self.assertEqual(target.branch_info["lane_membership_source"], "carla_waypoint")
        self.assertEqual(target.acc_mode_prob, {"lstm": 1.0})
        # The lead means follow the target: ahead of the ego and moving.
        self.assertGreater(float(prediction.means[0, 1, 0]), float(prediction.means[0, 0, 0]))
        self.assertGreater(float(prediction.means[0, -1, 1]), 0.5 * SPEED)


class TestLSTMPolicyString(unittest.TestCase):
    def test_policy_string_parses_to_lstm_single_mode(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent

        self.assertEqual(ACCNairSMPCAgent._parse_predictor_type(POLICY), "lstm")
        self.assertEqual(ACCNairSMPCAgent._parse_controller_num_modes(POLICY, 3), 1)
        self.assertEqual(ACCNairSMPCAgent._parse_variant(POLICY), VARIANT_FIXED_RISK)
        self.assertEqual(ACCNairSMPCAgent._parse_safety_constraint_mode(POLICY), SAFETY_NOMINAL_SAFE_DISTANCE)
        self.assertFalse(ACCNairSMPCAgent._parse_best_mode_only(POLICY))
        self.assertEqual(ACCNairSMPCAgent._parse_predictor_type("acc_nair_smpc_stdan_3int"), "stdan_3int")
        self.assertEqual(ACCNairSMPCAgent._parse_predictor_type("acc_nair_smpc_stdan_vel"), "stdan_vel")


if __name__ == "__main__":
    unittest.main()
