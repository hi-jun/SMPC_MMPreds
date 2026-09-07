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


def fake_mode(name, d_values, scale=1.0):
    frenet = np.zeros((len(d_values), 3))
    frenet[:, 1] = d_values
    return ACCModePrediction(
        vehicle_id=1,
        mode_name=name,
        probability=0.5,
        frenet=frenet,
        active_mask=np.ones(len(d_values), dtype=bool),
        raw_mode_indices=[1],
        ego_lane_membership_split_step=0,
        clearance_scale=scale,
    )


def expected_overlap(d_values):
    return np.clip((1.8 - np.abs(np.asarray(d_values, dtype=float))) / 1.0, 0.0, 1.0)


class TestLeavesEgoLane(unittest.TestCase):
    """The horizon end decides; a transient excursion is not a departure."""

    def test_predicate(self):
        T, F = True, False
        self.assertFalse(leaves_ego_lane([T] * 16))
        self.assertTrue(leaves_ego_lane([T] * 11 + [F] * 5))
        self.assertTrue(leaves_ego_lane([T] * 15 + [F]))
        self.assertTrue(leaves_ego_lane([F, T, T, F, F]), "outside now, in and then out")
        self.assertFalse(leaves_ego_lane([T] * 6 + [F] * 3 + [T] * 7), "wobble about the edge")
        self.assertFalse(leaves_ego_lane([F] * 16), "never inside: nothing to leave")
        self.assertFalse(leaves_ego_lane([]))


class TestEgoLaneModesFollowTheTrajectory(unittest.TestCase):
    """Defect 1 (aggressive cut-in, 2026-09-06).

    A vehicle that had just cut in from the left kept RLC = 1.00 -- it was
    still moving right -- while every predicted step stayed inside the ego
    lane.  The label mapping called that a certain cut-out the moment |d|
    dropped below the lane half-width, and the command went from -1.8 to
    +1.5 m/s^2 in 0.6 s, towards the vehicle that had just merged.
    """

    def test_completed_cutin_with_a_certain_rlc_label_is_lane_keeping(self):
        # Logged case: |d| 1.45 now, RLC +1.40 -> -0.49 over the horizon, 16/16
        # in lane by the CARLA waypoint test (the lane is 3.5 m wide).
        rlc_d = np.linspace(1.40, -0.49, HORIZON)
        raw = ego_lane_raw([0.0, 0.0, 1.0], [rlc_d, rlc_d, rlc_d])
        processed = process(
            raw, current_d=1.45, cutin_probability_threshold=0.1,
            **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool)))
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["lk"])
        lk = by_name["lk"]
        self.assertEqual(list(lk.raw_mode_indices), [0, 1, 2])
        self.assertAlmostEqual(lk.probability, 1.0)
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.0)
        self.assertAlmostEqual(processed.acc_mode_prob["lk"], 1.0)
        self.assertTrue(lk.active_mask.all())
        self.assertEqual(lk.clearance_scale, 1.0, "no overlap relaxation on a lane keeper")
        prediction, _ = build_multitarget_lead_prediction(
            [processed], ego_state=np.array([0.0, 17.0]), horizon=HORIZON,
            desired_speed=17.0, num_modes=8)
        self.assertEqual(prediction.mode_names, ["1:lk"])
        self.assertTrue(prediction.active_mask.all())
        np.testing.assert_array_equal(prediction.clearance_scale, 1.0)

    def test_a_trajectory_that_leaves_is_the_cutout_and_the_rest_keep_the_lane(self):
        raw = ego_lane_raw(
            [0.3, 0.6, 0.1],
            [profile((IN, HORIZON)),                   # LK stays
             profile((IN, 10), (OUT, HORIZON - 10)),   # LLC leaves after 2 s
             profile((IN, HORIZON))])                  # RLC stays
        processed = process(raw, current_d=IN)
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["cutout", "lk"])
        self.assertEqual(list(by_name["cutout"].raw_mode_indices), [1])
        self.assertAlmostEqual(by_name["cutout"].probability, 0.6)
        np.testing.assert_array_equal(
            by_name["cutout"].active_mask, [True] * 11 + [False] * 5)
        self.assertEqual(list(by_name["lk"].raw_mode_indices), [0, 2])
        self.assertAlmostEqual(by_name["lk"].probability, 0.4)
        self.assertTrue(by_name["lk"].active_mask.all())
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.6)

    def test_a_transient_excursion_is_not_a_cutout(self):
        raw = ego_lane_raw(
            [0.3, 0.6, 0.1],
            [profile((IN, HORIZON)),
             profile((IN, 6), (0.9, 3), (IN, HORIZON - 9)),   # brushes the edge, comes back
             profile((IN, HORIZON))])
        processed = process(raw, current_d=IN)
        self.assertEqual(sorted(modes_by_name(processed)), ["lk"])
        self.assertAlmostEqual(modes_by_name(processed)["lk"].probability, 1.0)
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.0)

    def test_split_follows_the_occupancy_mask_not_the_lateral_offset(self):
        """The CARLA-waypoint mask decides, since it also decides the constraints."""
        raw = ego_lane_raw([0.3, 0.6, 0.1], [profile((IN, HORIZON))] * 3)
        memberships = np.ones((3, HORIZON + 1), dtype=bool)
        memberships[2, 12:] = False   # the waypoint test puts RLC's last steps in the next lane
        by_name = modes_by_name(process(raw, current_d=IN, **waypoint_kwargs(memberships)))
        self.assertEqual(list(by_name["cutout"].raw_mode_indices), [2])
        self.assertAlmostEqual(by_name["cutout"].probability, 0.1)
        self.assertEqual(list(by_name["lk"].raw_mode_indices), [0, 1])
        # The converse: d says out, the mask says in -> lane keeping.
        raw = ego_lane_raw(
            [0.3, 0.6, 0.1],
            [profile((IN, HORIZON)), profile((IN, 10), (OUT, 5)), profile((IN, HORIZON))])
        by_name = modes_by_name(process(
            raw, current_d=IN, **waypoint_kwargs(np.ones((3, HORIZON + 1), dtype=bool))))
        self.assertEqual(sorted(by_name), ["lk"])


class TestLateralOverlapClearance(unittest.TestCase):
    """Defect 2 (cutout_no_sublv, 2026-09-06).

    Binary occupancy held the full standoff until the lane was completely
    clear, 1.2 s after the predictor had called the cut-out; the ego sat on
    its desired gap and could not accelerate.
    """

    D = [0.0, 0.5, 0.8, 1.0, 1.3, 1.8, 2.5]
    EXPECTED = [1.0, 1.0, 1.0, 0.8, 0.5, 0.0, 0.0]

    def test_scale_is_one_within_0_8_zero_beyond_1_8_and_linear_between(self):
        for sign in (1.0, -1.0):
            mode = fake_mode("cutout", sign * np.array(self.D))
            scales = lateral_overlap_clearance([mode], REL_EGO_LANE)
            np.testing.assert_allclose(mode.clearance_scale, self.EXPECTED, atol=1e-12)
            np.testing.assert_allclose(scales["cutout"], self.EXPECTED, atol=1e-12)

    def test_takes_the_minimum_with_an_existing_scale(self):
        mode = fake_mode("cutout", self.D, scale=np.full(len(self.D), 0.6))
        lateral_overlap_clearance([mode], REL_EGO_LANE)
        np.testing.assert_allclose(mode.clearance_scale, np.minimum(self.EXPECTED, 0.6))

    def test_is_measured_from_the_ego_lane_centre(self):
        mode = fake_mode("cutout", np.array(self.D) + 3.5)
        lateral_overlap_clearance([mode], REL_EGO_LANE, ego_lane_d=3.5)
        np.testing.assert_allclose(mode.clearance_scale, self.EXPECTED, atol=1e-12)

    def test_lk_and_cutin_modes_are_left_alone(self):
        modes = [fake_mode("lk", self.D), fake_mode("cutin", self.D)]
        self.assertEqual(lateral_overlap_clearance(modes, REL_EGO_LANE), {})
        for mode in modes:
            self.assertEqual(mode.clearance_scale, 1.0, mode.mode_name)
        adjacent = fake_mode("cutout", self.D)
        self.assertEqual(lateral_overlap_clearance([adjacent], REL_LEFT_ADJACENT), {})
        self.assertEqual(adjacent.clearance_scale, 1.0)

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


class TestOrdinaryFollowingIsUntouched(unittest.TestCase):
    """A lead driving straight at |d| < 0.3 keeps the full standoff and the same command.

    An earlier relaxation with a sigma term moved ordinary following by
    |delta a| = 0.49 m/s^2; the overlap taper and the trajectory split must not.
    """

    @staticmethod
    def _controller():
        return NairACCSMPC(original_nair_acc_config(
            horizon=HORIZON, dt=DT, desired_speed=17.0, num_modes=8))

    @staticmethod
    def _straight_lead():
        wobble = 0.15 + 0.1 * np.sin(np.linspace(0.0, np.pi, HORIZON))   # |d| < 0.3
        return ego_lane_raw([0.79, 0.11, 0.10], [wobble, -wobble, wobble])

    def test_single_lane_keeping_mode_at_full_standoff(self):
        processed = process(self._straight_lead(), current_d=0.2, cutin_probability_threshold=0.1)
        by_name = modes_by_name(processed)
        self.assertEqual(sorted(by_name), ["lk"])
        self.assertAlmostEqual(processed.acc_mode_prob["cutout"], 0.0)
        self.assertTrue(by_name["lk"].active_mask.all())
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)
        self.assertEqual(processed.branch_info["lateral_overlap_scales"], {})

    def test_command_matches_the_label_mapped_two_mode_prediction(self):
        """Folding the phantom cut-out (same trajectory) into lk moves the command by < 0.02."""
        processed = process(self._straight_lead(), current_d=0.2, cutin_probability_threshold=0.1)
        lk = modes_by_name(processed)["lk"]

        def copy_as(name, probability):
            return ACCModePrediction(
                vehicle_id=1, mode_name=name, probability=probability,
                frenet=lk.frenet.copy(), active_mask=lk.active_mask.copy(),
                raw_mode_indices=[0] if name == "lk" else [1, 2],
                ego_lane_membership_split_step=lk.ego_lane_membership_split_step)

        label_mapped = ACCProcessedPrediction(
            vehicle_id=1, relation_to_ego_lane=REL_EGO_LANE,
            mode_predictions=[copy_as("lk", 0.79), copy_as("cutout", 0.21)],
            raw_mode_prob=processed.raw_mode_prob,
            acc_mode_prob={"lk": 0.79, "cutin": 0.0, "cutout": 0.21}, branch_info={})
        # Ego on its desired gap: L + d0 + tau v = 4.5 + 3 + 1.3 * 13 = 24.4 m behind s0 = 30.
        ego_state = np.array([30.0 - 24.4, 13.0])

        def solve(target):
            prediction, _ = build_multitarget_lead_prediction(
                [target], ego_state=ego_state, horizon=HORIZON, desired_speed=17.0, num_modes=8)
            np.testing.assert_array_equal(prediction.clearance_scale, 1.0)
            return float(self._controller().solve(ego_state, prediction).action)

        a_new, a_old = solve(processed), solve(label_mapped)
        self.assertLess(abs(a_new - a_old), 0.02, (a_new, a_old))

    def test_a_cutout_mode_still_inside_0_8_m_keeps_the_full_standoff(self):
        """A drifting lead predicted out only at the horizon end is not relaxed inside 0.8 m."""
        rlc_d = np.linspace(0.25, 0.75, HORIZON)
        raw = ego_lane_raw(
            [0.79, 0.11, 0.10], [profile((IN, HORIZON)), profile((IN, HORIZON)), rlc_d])
        memberships = np.ones((3, HORIZON + 1), dtype=bool)
        memberships[2, -1] = False        # 15/16 in lane, as in the logs
        by_name = modes_by_name(process(
            raw, current_d=0.2, cutin_probability_threshold=0.1, **waypoint_kwargs(memberships)))
        cutout = by_name["cutout"]
        self.assertAlmostEqual(cutout.probability, 0.10)
        np.testing.assert_array_equal(cutout.active_mask, [True] * HORIZON + [False])
        np.testing.assert_allclose(cutout.clearance_scale[cutout.active_mask], 1.0)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)


if __name__ == "__main__":
    unittest.main()
