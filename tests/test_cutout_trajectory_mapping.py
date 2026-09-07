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
    lateral_overlap_clearance,
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


def expected_overlap(d_values):
    return np.clip((1.8 - np.abs(np.asarray(d_values, dtype=float))) / 1.0, 0.0, 1.0)


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
    is reverted; the vehicle that has just cut in, which motivated it, is
    handled by the guard on ``lateral_overlap_clearance`` instead.
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


class TestLateralOverlapClearance(unittest.TestCase):
    """Defect 2 (cutout_no_sublv, 2026-09-06).

    Binary occupancy held the full standoff until the lane was completely
    clear, 1.2 s after the predictor had called the cut-out; the ego sat on
    its desired gap and could not accelerate.
    """

    D = [0.0, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5]
    EXPECTED = [1.0, 1.0, 1.0, 0.8, 0.5, 0.0, 0.0]
    LEAVING = [True, True, False, False, False, False, False]   # D at the 0.5 m threshold

    def test_scale_is_one_within_0_8_zero_beyond_1_8_and_linear_between(self):
        for sign in (1.0, -1.0):
            mode = fake_mode("cutout", sign * np.array(self.D), active_mask=self.LEAVING)
            scales = lateral_overlap_clearance([mode], REL_EGO_LANE)
            np.testing.assert_allclose(mode.clearance_scale, self.EXPECTED, atol=1e-12)
            np.testing.assert_allclose(scales["cutout"], self.EXPECTED, atol=1e-12)

    def test_takes_the_minimum_with_an_existing_scale(self):
        mode = fake_mode("cutout", self.D, scale=np.full(len(self.D), 0.6), active_mask=self.LEAVING)
        lateral_overlap_clearance([mode], REL_EGO_LANE)
        np.testing.assert_allclose(mode.clearance_scale, np.minimum(self.EXPECTED, 0.6))

    def test_is_measured_from_the_ego_lane_centre(self):
        mode = fake_mode("cutout", np.array(self.D) + 3.5, active_mask=self.LEAVING)
        lateral_overlap_clearance([mode], REL_EGO_LANE, ego_lane_d=3.5)
        np.testing.assert_allclose(mode.clearance_scale, self.EXPECTED, atol=1e-12)

    def test_lk_and_cutin_modes_are_left_alone(self):
        modes = [fake_mode("lk", self.D, active_mask=self.LEAVING),
                 fake_mode("cutin", self.D, active_mask=self.LEAVING)]
        self.assertEqual(lateral_overlap_clearance(modes, REL_EGO_LANE), {})
        for mode in modes:
            self.assertEqual(mode.clearance_scale, 1.0, mode.mode_name)
        adjacent = fake_mode("cutout", self.D, active_mask=self.LEAVING)
        self.assertEqual(lateral_overlap_clearance([adjacent], REL_LEFT_ADJACENT), {})
        self.assertEqual(adjacent.clearance_scale, 1.0)

    def test_a_cutout_mode_whose_trajectory_stays_in_lane_is_left_alone(self):
        """The guard: a cut-out label on a trajectory that does not leave is not relaxed."""
        for mask, why in (
                ([True] * 7, "never leaves"),
                ([True] * 5 + [False] * 2, "two steps out at the end: a wobble"),
                ([False] * 7, "never inside: nothing to leave")):
            mode = fake_mode("cutout", self.D, active_mask=mask)
            self.assertEqual(lateral_overlap_clearance([mode], REL_EGO_LANE), {}, why)
            self.assertEqual(mode.clearance_scale, 1.0, why)
        mode = fake_mode("cutout", self.D, active_mask=[True] * 4 + [False] * 3)
        lateral_overlap_clearance([mode], REL_EGO_LANE)
        np.testing.assert_allclose(mode.clearance_scale, self.EXPECTED, err_msg="three steps out: a departure")

    def test_cutout_mode_from_the_processor_carries_the_overlap(self):
        llc_d = np.linspace(0.35, 2.45, HORIZON)   # slides out over the horizon
        raw = ego_lane_raw(
            [0.3, 0.7, 0.0], [profile((IN, HORIZON)), llc_d, profile((IN, HORIZON))])
        memberships = np.ones((3, HORIZON + 1), dtype=bool)
        memberships[1, 1:] = llc_d < 1.75           # the waypoint test: lane half-width
        processed = process(raw, current_d=IN, **waypoint_kwargs(memberships))
        by_name = modes_by_name(processed)
        cutout = by_name["cutout"]
        expected = expected_overlap(np.concatenate(([IN], llc_d)))
        np.testing.assert_allclose(cutout.clearance_scale, expected)
        np.testing.assert_allclose(processed.branch_info["lateral_overlap_scales"]["cutout"], expected)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)
        self.assertLess(cutout.clearance_scale[cutout.active_mask][-1], 1.0,
                        "the standoff relaxes before the vehicle has left the lane")
        prediction, _ = build_multitarget_lead_prediction(
            [processed], ego_state=np.array([0.0, 13.0]), horizon=HORIZON,
            desired_speed=17.0, num_modes=8)
        idx = prediction.mode_names.index("1:cutout")
        active = prediction.active_mask[idx]
        np.testing.assert_array_equal(active, cutout.active_mask)
        np.testing.assert_allclose(prediction.clearance_scale[idx, active], expected[active])
        self.assertTrue(np.all(prediction.clearance_scale[idx, ~active] == 1.0))


class TestCompletedCutInKeepsItsStandoff(unittest.TestCase):
    """The guard on ``lateral_overlap_clearance`` (aggressive cut-in, 2026-09-06).

    A vehicle that has just cut in from the left keeps RLC = 1.00 -- it is
    still moving right -- while every predicted step stays inside the ego
    lane, so under the label mapping it is a certain ``cutout`` the moment
    its relation flips to ego-lane.  It sits at |d| ~ 1.45 m, where the
    overlap taper alone would leave it 35 % of its standoff at the moment it
    is most in the way; the taper therefore needs the mode's trajectory to
    actually leave the lane.
    """

    def test_completed_cutin_keeps_the_full_standoff_at_every_step(self):
        # Logged case: |d| 1.45 now, RLC +1.40 -> -0.49 over the horizon, 16/16
        # in lane by the CARLA waypoint test (the lane is 3.5 m wide).
        rlc_d = np.linspace(1.40, -0.49, HORIZON)
        processed = process(
            ego_lane_raw([0.0, 0.0, 1.0], [rlc_d, rlc_d, rlc_d]), current_d=1.45,
            cutin_probability_threshold=0.1, **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool)))
        cutout = modes_by_name(processed)["cutout"]
        self.assertAlmostEqual(cutout.probability, 1.0)
        self.assertTrue(cutout.active_mask.all())
        np.testing.assert_array_equal(np.broadcast_to(cutout.clearance_scale, HORIZON + 1), 1.0)
        self.assertEqual(processed.branch_info["lateral_overlap_scales"], {})
        self.assertAlmostEqual(expected_overlap([1.45])[0], 0.35, msg="what the taper alone would apply now")
        prediction, _ = build_multitarget_lead_prediction(
            [processed], ego_state=np.array([0.0, 17.0]), horizon=HORIZON,
            desired_speed=17.0, num_modes=8)
        self.assertEqual(prediction.mode_names, ["1:cutout"])
        self.assertTrue(prediction.active_mask.all())
        np.testing.assert_array_equal(prediction.clearance_scale, 1.0)

    def test_a_real_cutout_is_tapered_once_three_steps_are_outside(self):
        rlc_d = np.linspace(0.25, 1.6, HORIZON)    # drifts right, through the taper band
        raw = ego_lane_raw(
            [0.79, 0.10, 0.11], [profile((IN, HORIZON)), profile((IN, HORIZON)), rlc_d])
        expected = expected_overlap(np.concatenate(([IN], rlc_d)))
        for steps_out in (3, 2):
            memberships = np.ones((3, HORIZON + 1), dtype=bool)
            memberships[2, -steps_out:] = False
            processed = process(raw, current_d=IN, **waypoint_kwargs(memberships))
            cutout = modes_by_name(processed)["cutout"]
            self.assertEqual(list(cutout.raw_mode_indices), [1, 2])
            if steps_out >= 3:
                np.testing.assert_allclose(cutout.clearance_scale, expected)
                np.testing.assert_allclose(processed.branch_info["lateral_overlap_scales"]["cutout"], expected)
                self.assertLess(cutout.clearance_scale[cutout.active_mask].min(), 0.5)
            else:
                self.assertEqual(cutout.clearance_scale, 1.0, "two steps out: a horizon-end wobble")
                self.assertEqual(processed.branch_info["lateral_overlap_scales"], {})


class TestOrdinaryFollowingIsUntouched(unittest.TestCase):
    """A lead driving straight at |d| < 0.3 keeps the full standoff and the same command.

    An earlier relaxation with a sigma term moved ordinary following by
    |delta a| = 0.49 m/s^2; the overlap taper must not, and neither may the
    phantom ``cutout`` mass (~0.2) the label mapping puts on a lane keeper.
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
        self.assertEqual(processed.branch_info["lateral_overlap_scales"], {})

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

    def test_a_cutout_mode_still_inside_0_8_m_keeps_the_full_standoff(self):
        """A drifting lead predicted out only at the horizon end is not relaxed inside 0.8 m."""
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
        np.testing.assert_allclose(cutout.clearance_scale[cutout.active_mask], 1.0)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
