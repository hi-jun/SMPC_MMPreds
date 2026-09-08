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
    ends_outside_ego_lane,
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


class TestEndsOutsideEgoLane(unittest.TestCase):
    """The horizon end decides, and it has to stay outside for three steps.

    A lane keeper's LLC/RLC trajectories poke out of the lane for the last
    step or two on a fifth of the ticks in steady following; a real cut-out
    is 4-5 steps outside by the time its probability crosses 0.5.
    """

    def test_predicate(self):
        T, F = True, False
        self.assertFalse(ends_outside_ego_lane([T] * 16))
        self.assertTrue(ends_outside_ego_lane([T] * 11 + [F] * 5))
        self.assertTrue(ends_outside_ego_lane([T] * 13 + [F] * 3), "three steps outside at the end")
        self.assertFalse(ends_outside_ego_lane([T] * 14 + [F] * 2), "two steps: a horizon-end wobble")
        self.assertFalse(ends_outside_ego_lane([T] * 15 + [F]))
        self.assertTrue(ends_outside_ego_lane([F, T, T, F, F, F]), "outside now, in and then out")
        self.assertFalse(ends_outside_ego_lane([T] * 6 + [F] * 3 + [T] * 7), "wobble about the edge")
        self.assertTrue(ends_outside_ego_lane([F] * 16), "already fully out: still a cut-out")
        self.assertFalse(ends_outside_ego_lane([]))
        self.assertTrue(
            ends_outside_ego_lane([T] * 15 + [F], sustained_steps=1), "the old one-step rule")


class TestEgoLaneModesFollowTheTrajectory(unittest.TestCase):
    """An ego-lane vehicle's raw LLC/RLC is ``cutout`` only when its trajectory leaves.

    The label alone puts phantom cut-out mass on every lane keeper: 0.07
    median and 0.33 peak on the straight-driving second lead of the
    2026-09-08 runs, and 0.18 on the lead itself before its lane change
    begins.  Nothing downstream can tell that mass from a real departure, so
    a probabilistic standoff relaxation reacts to a car that is going
    nowhere.  Splitting by trajectory removes it where it is created and
    leaves the real departure untouched: on those runs the second lead's
    cut-out mass falls to zero at all 500 ticks while the 79 ticks of the
    departure keep theirs, because the trajectory starts leaving on the same
    tick the intention flips.  (Tried once before as caf17f2, reverted then
    as a no-op; it stopped being one when the standoff started reading the
    probability.)
    """

    def test_a_raw_mode_that_leaves_the_lane_carries_its_own_mass_into_cutout(self):
        stays = profile((IN, HORIZON))
        leaves = profile((IN, 10), (OUT, HORIZON - 10))
        processed = process(ego_lane_raw([0.3, 0.6, 0.1], [stays, leaves, stays]), current_d=IN)
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["cutout", "lk"])
        self.assertEqual(list(by_name["cutout"].raw_mode_indices), [1], "LLC leaves, RLC does not")
        self.assertAlmostEqual(by_name["cutout"].probability, 0.6)
        self.assertEqual(list(by_name["lk"].raw_mode_indices), [0, 2])
        self.assertAlmostEqual(by_name["lk"].probability, 0.4)
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.6)
        self.assertAlmostEqual(processed.acc_mode_prob["lk"], 0.4)
        np.testing.assert_array_equal(by_name["cutout"].active_mask, [True] * 11 + [False] * 5)
        self.assertTrue(by_name["lk"].active_mask.all())

    def test_a_lane_keeper_has_no_cutout_mode_at_all(self):
        stays = profile((IN, HORIZON))
        processed = process(ego_lane_raw([0.3, 0.6, 0.1], [stays, stays, stays]), current_d=IN)
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["lk"], "no trajectory leaves: nothing to cut out")
        self.assertEqual(list(by_name["lk"].raw_mode_indices), [0, 1, 2])
        self.assertAlmostEqual(by_name["lk"].probability, 1.0)
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.0)
        self.assertTrue(by_name["lk"].active_mask.all())

    def test_a_certain_rlc_that_never_leaves_the_lane_is_lane_keeping(self):
        # The completed cut-in: |d| 1.45 now, RLC +1.40 -> -0.49 over the
        # horizon, 16/16 in lane by the CARLA waypoint test.  The label turned
        # it into a certain cut-out the moment the relation flipped to
        # ego-lane, at a vehicle squarely in the ego path.
        rlc_d = np.linspace(1.40, -0.49, HORIZON)
        processed = process(
            ego_lane_raw([0.0, 0.0, 1.0], [rlc_d, rlc_d, rlc_d]), current_d=1.45,
            cutin_probability_threshold=0.1, **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool)))
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["lk"])
        self.assertAlmostEqual(by_name["lk"].probability, 1.0)
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.0)
        self.assertTrue(by_name["lk"].active_mask.all())


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
        self.assertEqual(list(cutout.raw_mode_indices), [2], "RLC is the trajectory that leaves")
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
        # it 35 % of its standoff at the moment it is most in the way.  It is
        # lane keeping under the trajectory split, and a lane keeper at p = 1
        # is not relaxed either.
        rlc_d = np.linspace(1.40, -0.49, HORIZON)
        processed = process(
            ego_lane_raw([0.0, 0.0, 1.0], [rlc_d, rlc_d, rlc_d]), current_d=1.45,
            cutin_probability_threshold=0.1, **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool)))
        lk = modes_by_name(processed)["lk"]
        self.assertAlmostEqual(lk.probability, 1.0)
        self.assertTrue(lk.active_mask.all())
        np.testing.assert_array_equal(np.broadcast_to(lk.clearance_scale, HORIZON + 1), 1.0)


class TestOrdinaryFollowingIsUntouched(unittest.TestCase):
    """A lead driving straight at |d| < 0.3 keeps the full standoff and the same command.

    An earlier relaxation with a sigma term moved ordinary following by
    |delta a| = 0.49 m/s^2; the phantom ``cutout`` mass (~0.2) the label
    mapping used to put on a lane keeper may not, and no longer exists to.
    """

    @staticmethod
    def _controller():
        return NairACCSMPC(original_nair_acc_config(
            horizon=HORIZON, dt=DT, desired_speed=17.0, num_modes=8))

    @staticmethod
    def _straight_lead():
        wobble = 0.15 + 0.1 * np.sin(np.linspace(0.0, np.pi, HORIZON))   # |d| < 0.3
        return ego_lane_raw([0.79, 0.11, 0.10], [wobble, -wobble, wobble])

    def test_a_lane_keeper_carries_no_cutout_mass(self):
        processed = process(self._straight_lead(), current_d=0.2, cutin_probability_threshold=0.1)
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["lk"], "the 0.21 of phantom cut-out mass is gone")
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.0)
        self.assertAlmostEqual(by_name["lk"].probability, 1.0)
        self.assertTrue(by_name["lk"].active_mask.all())
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)

    def test_command_matches_a_single_lane_keeping_mode(self):
        """The merged lane keeper commands what a single lane-keeping mode does."""
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
        self.assertEqual(list(cutout.raw_mode_indices), [2], "only RLC ends outside")
        self.assertAlmostEqual(cutout.probability, 0.11)
        np.testing.assert_array_equal(cutout.active_mask, [True] * (HORIZON - 2) + [False] * 3)
        np.testing.assert_array_equal(
            np.broadcast_to(cutout.clearance_scale, HORIZON + 1), 1.0)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
