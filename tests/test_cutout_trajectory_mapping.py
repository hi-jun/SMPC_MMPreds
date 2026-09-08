import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    ACCModePrediction,
    ACCProcessedPrediction,
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    build_multitarget_lead_prediction,
    leaves_ego_lane,
    process_vehicle_prediction,
)
from utils.acc_nair_smpc import NairACCSMPC, original_nair_acc_config  # noqa: E402


DT = 0.2
HORIZON = 15  # the CARLA runs' horizon: 16 steps including the measured state
IN, OUT = 0.2, 3.5  # lateral offsets well inside / outside the 0.5 m test threshold


def ego_lane_raw(probs, d_profiles, s0=30.0, speed=13.0, vehicle_id=1):
    """Ego-lane target whose raw modes LK/LLC/RLC follow the given lateral profiles."""
    s = s0 + speed * DT * np.arange(1, HORIZON + 1)
    trajs = [np.column_stack((s, np.asarray(d, dtype=float))) for d in d_profiles]
    vel = np.zeros((3, HORIZON, 2))
    vel[:, :, 0] = speed
    return {
        "vehicle_id": vehicle_id,
        "raw_intention_prob": np.asarray(probs, dtype=float),
        "pred_traj_frenet": np.stack(trajs),
        "raw_pred_vel": vel,
        "signed_t_cross": 1.0,
        "valid_mask": np.ones((3, HORIZON), dtype=bool),
    }


def process(raw, current_d, s0=30.0, speed=13.0, **kwargs):
    return process_vehicle_prediction(
        raw,
        REL_EGO_LANE,
        current_frenet_state=np.array([s0, current_d, speed]),
        horizon=HORIZON,
        dt=DT,
        ego_lane_threshold=0.5,
        **kwargs
    )


def waypoint_kwargs(memberships):
    """Occupancy as the agent supplies it from CARLA waypoints (lane id, not |d|)."""
    return {"mode_lane_memberships": memberships, "lane_membership_source": "carla_waypoint"}


def modes_by_name(processed):
    return {mode.mode_name: mode for mode in processed.mode_predictions}


def profile(*segments):
    """Concatenate (value, count) segments into a HORIZON-long lateral profile."""
    values = np.concatenate([np.full(count, value, dtype=float) for value, count in segments])
    assert values.size == HORIZON
    return values


def fake_mode(name, d_values, scale=1.0, active_mask=None):
    frenet = np.zeros((len(d_values), 3))
    frenet[:, 1] = d_values
    if active_mask is None:
        active_mask = np.ones(len(d_values), dtype=bool)
    return ACCModePrediction(
        vehicle_id=1,
        mode_name=name,
        probability=0.5,
        frenet=frenet,
        active_mask=np.asarray(active_mask, dtype=bool),
        raw_mode_indices=[1],
        ego_lane_membership_split_step=0,
        clearance_scale=scale,
    )


class TestLeavesEgoLane(unittest.TestCase):
    """The horizon end decides, and it has to stay outside for three steps.

    A lane keeper's LLC/RLC trajectories poke out of the lane for the last
    step or two on a fifth of the ticks in steady following; a real cut-out
    is 4-5 steps outside by the time its probability crosses 0.5.
    """

    def test_predicate(self):
        T, F = True, False
        self.assertFalse(leaves_ego_lane([T] * 16))
        self.assertTrue(leaves_ego_lane([T] * 11 + [F] * 5))
        self.assertTrue(leaves_ego_lane([T] * 13 + [F] * 3), "three steps outside at the end")
        self.assertFalse(leaves_ego_lane([T] * 14 + [F] * 2), "two steps: a horizon-end wobble")
        self.assertFalse(leaves_ego_lane([T] * 15 + [F]))
        self.assertTrue(leaves_ego_lane([F, T, T, F, F, F]), "outside now, in and then out")
        self.assertFalse(leaves_ego_lane([T] * 6 + [F] * 3 + [T] * 7), "wobble about the edge")
        self.assertFalse(leaves_ego_lane([F] * 16), "never inside: nothing to leave")
        self.assertFalse(leaves_ego_lane([]))
        self.assertTrue(leaves_ego_lane([T] * 15 + [F], sustained_steps=1), "the old one-step rule")


class TestEgoLaneModesFollowTheLabel(unittest.TestCase):
    """An ego-lane vehicle's raw LLC and RLC merge into ``cutout`` whatever their trajectories do.

    Where a predicted trajectory actually is, in the lane or not, is carried
    by the mode's occupancy mask, which is what the constraints use; the
    label only groups the probability mass.  Splitting the ego-lane modes by
    trajectory instead (caf17f2) changed no command in an offline replay and
    is reverted.
    """

    def test_llc_and_rlc_merge_into_cutout_with_their_probabilities_summed(self):
        stays = profile((IN, HORIZON))
        leaves = profile((IN, 10), (OUT, HORIZON - 10))
        for llc_d, cutout_mask in ((stays, [True] * (HORIZON + 1)), (leaves, [True] * 11 + [False] * 5)):
            processed = process(ego_lane_raw([0.3, 0.6, 0.1], [stays, llc_d, stays]), current_d=IN)
            by_name = modes_by_name(processed)
            self.assertEqual(sorted(by_name), ["cutout", "lk"])
            self.assertEqual(list(by_name["lk"].raw_mode_indices), [0])
            self.assertAlmostEqual(by_name["lk"].probability, 0.3)
            self.assertEqual(list(by_name["cutout"].raw_mode_indices), [1, 2])
            self.assertAlmostEqual(by_name["cutout"].probability, 0.7)
            self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.7)
            self.assertAlmostEqual(processed.acc_mode_prob["lk"], 0.3)
            # The trajectory shows in the occupancy mask (of the most probable raw mode), not in the label.
            np.testing.assert_array_equal(by_name["cutout"].active_mask, cutout_mask)
            self.assertTrue(by_name["lk"].active_mask.all())

    def test_a_certain_rlc_is_a_certain_cutout_even_with_every_step_in_lane(self):
        # The completed cut-in: |d| 1.45 now, RLC +1.40 -> -0.49 over the
        # horizon, 16/16 in lane by the CARLA waypoint test.
        rlc_d = np.linspace(1.40, -0.49, HORIZON)
        processed = process(
            ego_lane_raw([0.0, 0.0, 1.0], [rlc_d, rlc_d, rlc_d]), current_d=1.45,
            cutin_probability_threshold=0.1, **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool)))
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["cutout"], "lk has no mass")
        self.assertEqual(list(by_name["cutout"].raw_mode_indices), [1, 2])
        self.assertAlmostEqual(by_name["cutout"].probability, 1.0)
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 1.0)
        self.assertTrue(by_name["cutout"].active_mask.all())


class TestCutOutModeKeepsItsFullStandoff(unittest.TestCase):
    """No geometric relaxation of a cut-out mode (2026-09-08).

    ``lateral_overlap_clearance`` used to shrink a cut-out mode's standoff
    with the predicted lateral overlap.  It was removed: it raised the
    reference speed on the steps before the predicted crossing, which reads
    as "accelerate into the vacated gap, then brake" and delayed the response
    to a slower vehicle beyond the vacating one.  The cut-out mode is now
    deterministic and unrelaxed wherever it is active, for every policy.
    """

    def test_a_cutout_leaving_the_lane_is_not_relaxed(self):
        rlc_d = np.linspace(0.25, 1.6, HORIZON)    # drifts right, out of the lane
        raw = ego_lane_raw(
            [0.79, 0.10, 0.11], [profile((IN, HORIZON)), profile((IN, HORIZON)), rlc_d])
        memberships = np.ones((3, HORIZON + 1), dtype=bool)
        memberships[2, -3:] = False
        processed = process(raw, current_d=IN, **waypoint_kwargs(memberships))
        cutout = modes_by_name(processed)["cutout"]
        self.assertEqual(list(cutout.raw_mode_indices), [1, 2])
        np.testing.assert_array_equal(
            np.broadcast_to(cutout.clearance_scale, HORIZON + 1), 1.0)
        self.assertNotIn("lateral_overlap_scales", processed.branch_info)
        prediction, _ = build_multitarget_lead_prediction(
            [processed], ego_state=np.array([0.0, 13.0]), horizon=HORIZON,
            desired_speed=17.0, num_modes=8)
        idx = prediction.mode_names.index("1:cutout")
        np.testing.assert_array_equal(prediction.clearance_scale[idx], 1.0)

    def test_a_completed_cutin_also_keeps_it(self):
        # Logged case: |d| 1.45 now, RLC +1.40 -> -0.49 over the horizon, 16/16
        # in lane by the CARLA waypoint test; the taper alone would have left
        # it 35 % of its standoff at the moment it is most in the way.
        rlc_d = np.linspace(1.40, -0.49, HORIZON)
        processed = process(
            ego_lane_raw([0.0, 0.0, 1.0], [rlc_d, rlc_d, rlc_d]), current_d=1.45,
            cutin_probability_threshold=0.1, **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool)))
        cutout = modes_by_name(processed)["cutout"]
        self.assertAlmostEqual(cutout.probability, 1.0)
        self.assertTrue(cutout.active_mask.all())
        np.testing.assert_array_equal(np.broadcast_to(cutout.clearance_scale, HORIZON + 1), 1.0)


class TestOrdinaryFollowingIsUntouched(unittest.TestCase):
    """A lead driving straight at |d| < 0.3 keeps the full standoff and the same command.

    An earlier relaxation with a sigma term moved ordinary following by
    |delta a| = 0.49 m/s^2; the phantom ``cutout`` mass (~0.2) the label
    mapping puts on a lane keeper may not.
    """

    @staticmethod
    def _controller():
        return NairACCSMPC(original_nair_acc_config(
            horizon=HORIZON, dt=DT, desired_speed=17.0, num_modes=8))

    @staticmethod
    def _straight_lead():
        wobble = 0.15 + 0.1 * np.sin(np.linspace(0.0, np.pi, HORIZON))   # |d| < 0.3
        return ego_lane_raw([0.79, 0.11, 0.10], [wobble, -wobble, wobble])

    def test_both_label_mapped_modes_at_full_standoff(self):
        processed = process(self._straight_lead(), current_d=0.2, cutin_probability_threshold=0.1)
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["cutout", "lk"])
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.21)
        for name, mode in by_name.items():
            self.assertTrue(mode.active_mask.all(), name)
            self.assertEqual(mode.clearance_scale, 1.0, name)

    def test_command_matches_a_single_lane_keeping_mode(self):
        """The phantom cut-out (same trajectory) moves the command by < 0.02: the mask decides."""
        processed = process(self._straight_lead(), current_d=0.2, cutin_probability_threshold=0.1)
        lk = modes_by_name(processed)["lk"]
        single_lk = ACCProcessedPrediction(
            vehicle_id=1, relation_to_ego_lane=REL_EGO_LANE,
            mode_predictions=[ACCModePrediction(
                vehicle_id=1, mode_name="lk", probability=1.0,
                frenet=lk.frenet.copy(), active_mask=lk.active_mask.copy(), raw_mode_indices=[0, 1, 2],
                ego_lane_membership_split_step=lk.ego_lane_membership_split_step)],
            raw_mode_prob=processed.raw_mode_prob,
            acc_mode_prob={"lk": 1.0, "cutin": 0.0, "cutout": 0.0}, branch_info={})
        # Ego on its desired gap: L + d0 + tau v = 4.5 + 3 + 1.3 * 13 = 24.4 m behind s0 = 30.
        ego_state = np.array([30.0 - 24.4, 13.0])

        def solve(target):
            prediction, _ = build_multitarget_lead_prediction(
                [target], ego_state=ego_state, horizon=HORIZON, desired_speed=17.0, num_modes=8)
            np.testing.assert_array_equal(prediction.clearance_scale, 1.0)
            return float(self._controller().solve(ego_state, prediction).action)

        a_label, a_single = solve(processed), solve(single_lk)
        self.assertLess(abs(a_label - a_single), 0.02, (a_label, a_single))

    def test_a_cutout_mode_predicted_out_only_at_the_horizon_end_is_unrelaxed(self):
        """A drifting lead predicted out only at the horizon end keeps its standoff."""
        rlc_d = np.linspace(0.25, 0.75, HORIZON)
        raw = ego_lane_raw(   # RLC is the more probable, hence the representative, cut-out trajectory
            [0.79, 0.10, 0.11], [profile((IN, HORIZON)), profile((IN, HORIZON)), rlc_d])
        memberships = np.ones((3, HORIZON + 1), dtype=bool)
        memberships[2, -3:] = False       # 13/16 in lane: out for the last 0.6 s
        by_name = modes_by_name(process(
            raw, current_d=0.2, cutin_probability_threshold=0.1, **waypoint_kwargs(memberships)))
        cutout = by_name["cutout"]
        self.assertAlmostEqual(cutout.probability, 0.21)
        np.testing.assert_array_equal(cutout.active_mask, [True] * (HORIZON - 2) + [False] * 3)
        np.testing.assert_array_equal(
            np.broadcast_to(cutout.clearance_scale, HORIZON + 1), 1.0)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
