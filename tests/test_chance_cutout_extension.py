import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported as a module so its test cases are not collected a second time here.
import test_stdan_3int_acc_integration as integration  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    REL_RIGHT_ADJACENT,
    build_multitarget_lead_prediction,
    process_vehicle_prediction,
)
from utils.acc_nair_smpc import confidence_quantile  # noqa: E402


REF = 0.6
VANISH = 0.1
DT = 0.2


def ego_lane_raw(vehicle_id, s0, speed, lk_prob, cutout_prob, horizon, leave_step=None):
    """Ego-lane target: raw mode 0 (LK) stays, raw mode 1 (LLC) is the cut-out.

    ``leave_step`` is the first horizon step (step 0 = measured state) at which
    the cut-out trajectory is outside the ego lane.
    """
    s = s0 + speed * DT * np.arange(1, horizon + 1)
    stay = np.column_stack((s, np.zeros(horizon)))
    d_out = np.zeros(horizon)
    if leave_step is not None:
        d_out[leave_step - 1:] = 3.5
    cutout = np.column_stack((s, d_out))
    vel = np.zeros((3, horizon, 2))
    vel[:, :, 0] = speed
    return {
        "vehicle_id": vehicle_id,
        "raw_intention_prob": np.array([lk_prob, cutout_prob, 0.0]),
        "pred_traj_frenet": np.stack((stay, cutout, stay)),
        "raw_pred_vel": vel,
        "signed_t_cross": 1.0,
        "valid_mask": np.ones((3, horizon), dtype=bool),
    }


def process(raw, relation, s0, d0, speed, horizon, **kwargs):
    return process_vehicle_prediction(
        raw,
        relation,
        current_frenet_state=np.array([s0, d0, speed]),
        horizon=horizon,
        dt=DT,
        ego_lane_threshold=0.5,
        **kwargs
    )


def modes_by_name(processed):
    return {mode.mode_name: mode for mode in processed.mode_predictions}


def assert_cutout_unscaled_in_lane(cutout):
    """The cut-out mode gets no probability factor: full standoff at every in-lane step.

    Its per-step scale follows the lateral overlap
    (``lateral_overlap_clearance``), which is 1.0 at the lane centre and zero
    once the trajectory is outside -- on steps that are inactive anyway.
    """
    np.testing.assert_array_equal(np.asarray(cutout.clearance_scale)[cutout.active_mask], 1.0)


def scenario_index(prediction, name):
    return prediction.mode_names.index(name)


class TestEgoLaneLaneKeepingChance(unittest.TestCase):
    """Rule (i): an ego-lane vehicle's ``lk`` mode becomes a confidence chance constraint.

    In 1-D the lane-keeping hypothesis of a vacating lead is a full lead at
    every step, so it would block the ego from ever accelerating into the
    cut-out; ``chance_cutout_clearance`` scales it by its probability and
    vanishes it below the threshold.  The ``cutout`` mode stays deterministic:
    no probability factor, only the geometric overlap taper on its own steps.
    """

    HORIZON = 6

    def _lead(self, lk_prob, cutout_prob, **kwargs):
        raw = ego_lane_raw(1, 20.0, 10.0, lk_prob, cutout_prob, self.HORIZON, leave_step=4)
        return process(raw, REL_EGO_LANE, 20.0, 0.0, 10.0, self.HORIZON, **kwargs)

    def test_lk_mode_is_scaled_by_its_probability(self):
        by_name = modes_by_name(self._lead(
            0.4, 0.6, cutin_chance_ref=REF, cutin_probability_threshold=VANISH))
        lk, cutout = by_name["lk"], by_name["cutout"]
        self.assertTrue(np.isnan(lk.chance_confidence))
        self.assertAlmostEqual(lk.clearance_scale, 0.623, places=3)
        self.assertAlmostEqual(
            lk.clearance_scale, confidence_quantile(0.4) / confidence_quantile(REF), places=9)
        self.assertTrue(lk.active_mask.all())
        self.assertTrue(np.isnan(cutout.chance_confidence))
        assert_cutout_unscaled_in_lane(cutout)
        np.testing.assert_array_equal(cutout.active_mask, [True, True, True, True, False, False, False])

    def test_unlikely_lk_mode_vanishes(self):
        by_name = modes_by_name(self._lead(
            0.05, 0.95, cutin_chance_ref=REF, cutin_probability_threshold=VANISH))
        self.assertFalse(by_name["lk"].active_mask.any())
        self.assertTrue(np.isnan(by_name["lk"].chance_confidence))
        cutout = by_name["cutout"]
        self.assertTrue(np.isnan(cutout.chance_confidence))
        assert_cutout_unscaled_in_lane(cutout)
        self.assertTrue(cutout.active_mask[:4].all(), "the vehicle itself keeps its standoff")

    def test_chance_off_leaves_ego_lane_modes_alone(self):
        by_name = modes_by_name(self._lead(0.05, 0.95))
        for name in ("lk", "cutout"):
            self.assertTrue(np.isnan(by_name[name].chance_confidence), name)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)
        assert_cutout_unscaled_in_lane(by_name["cutout"])
        self.assertTrue(by_name["lk"].active_mask.all())
        self.assertTrue(by_name["cutout"].active_mask[:4].all())

    def test_vanishing_alone_drops_a_negligible_lk_mode_without_scaling(self):
        # The plain multimodal policy prunes below the threshold but never scales.
        by_name = modes_by_name(self._lead(0.05, 0.95, cutin_probability_threshold=VANISH))
        for name in ("lk", "cutout"):
            self.assertTrue(np.isnan(by_name[name].chance_confidence), name)
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)
        assert_cutout_unscaled_in_lane(by_name["cutout"])
        self.assertFalse(by_name["lk"].active_mask.any())
        self.assertTrue(by_name["cutout"].active_mask[:4].all())
        by_name = modes_by_name(self._lead(0.4, 0.6, cutin_probability_threshold=VANISH))
        self.assertTrue(by_name["lk"].active_mask.all())
        self.assertEqual(by_name["lk"].clearance_scale, 1.0)


class TestVacatedLaneCells(unittest.TestCase):
    """Rule (ii): a lead seen past a vacating ego-lane vehicle inherits its cut-out probability.

    The second lead only matters if the first one leaves, so the cell exists
    with that mode's probability: confidence ``min(p, beta_ref)`` and nothing
    at all below the vanishing threshold.
    """

    HORIZON = 8
    LEAVE_STEP = 5

    def _build(self, lv_lk, lv_cutout, lv_s0=20.0, chance=True):
        kwargs = {"cutin_chance_ref": REF, "cutin_probability_threshold": VANISH} if chance else {}
        lv = process(
            ego_lane_raw(1, lv_s0, 10.0, lv_lk, lv_cutout, self.HORIZON, leave_step=self.LEAVE_STEP),
            REL_EGO_LANE, lv_s0, 0.0, 10.0, self.HORIZON, **kwargs)
        sub_lv = process(
            ego_lane_raw(2, 40.0, 8.0, 0.98, 0.02, self.HORIZON, leave_step=3),
            REL_EGO_LANE, 40.0, 0.0, 8.0, self.HORIZON, **kwargs)
        return build_multitarget_lead_prediction(
            [lv, sub_lv],
            ego_state=np.array([0.0, 10.0]),
            horizon=self.HORIZON,
            desired_speed=15.0,
            num_modes=4,
            reference_beta=REF if chance else 0.0,
            vanish_threshold=VANISH if chance else 0.0,
        )

    def test_vacated_cells_inherit_the_cutout_probability(self):
        prediction, metadata = self._build(0.3, 0.7)
        idx = scenario_index(prediction, "1:cutout+2:lk")
        k = self.LEAVE_STEP
        self.assertEqual(metadata[idx]["selected_vehicle_ids"], [1] * k + [2] * (self.HORIZON + 1 - k))
        self.assertTrue(prediction.active_mask[idx].all())
        self.assertTrue(np.all(np.isnan(prediction.chance_confidence[idx, :k])))
        np.testing.assert_allclose(prediction.clearance_scale[idx, :k], 1.0)
        self.assertTrue(np.all(np.isnan(prediction.chance_confidence[idx, k:])))
        np.testing.assert_allclose(prediction.clearance_scale[idx, k:], 1.0)
        self.assertEqual(metadata[idx]["vacated_lane_steps"], list(range(k, self.HORIZON + 1)))

    def test_vacated_cells_vanish_with_an_unlikely_cutout(self):
        prediction, metadata = self._build(0.95, 0.05)
        idx = scenario_index(prediction, "1:cutout+2:lk")
        k = self.LEAVE_STEP
        self.assertTrue(prediction.active_mask[idx, :k].all())
        self.assertFalse(prediction.active_mask[idx, k:].any())
        self.assertEqual(metadata[idx]["selected_vehicle_ids"][k:], [None] * (self.HORIZON + 1 - k))
        self.assertEqual(
            metadata[idx]["effective_lead_keys"][k:], [("inactive",)] * (self.HORIZON + 1 - k))
        inactive_group = prediction.k_group_names.index("inactive")
        self.assertTrue(np.all(prediction.k_group_map[idx, k:] == inactive_group))
        self.assertTrue(np.all(prediction.k_group_map[idx, :k] != inactive_group))

    def test_lane_keeping_scenario_keeps_the_scaled_lead(self):
        prediction, metadata = self._build(0.3, 0.7)
        idx = scenario_index(prediction, "1:lk+2:lk")
        self.assertEqual(metadata[idx]["selected_vehicle_ids"], [1] * (self.HORIZON + 1))
        self.assertTrue(np.all(np.isnan(prediction.chance_confidence[idx])))
        np.testing.assert_allclose(
            prediction.clearance_scale[idx], confidence_quantile(0.3) / confidence_quantile(REF))
        self.assertEqual(metadata[idx]["vacated_lane_steps"], [])

    def test_vehicle_behind_the_ego_does_not_condition_the_cell(self):
        prediction, metadata = self._build(0.3, 0.7, lv_s0=-30.0)
        idx = scenario_index(prediction, "1:cutout+2:lk")
        self.assertEqual(metadata[idx]["selected_vehicle_ids"], [2] * (self.HORIZON + 1))
        self.assertEqual(metadata[idx]["vacated_lane_steps"], [])
        self.assertTrue(np.all(np.isnan(prediction.chance_confidence[idx])))
        np.testing.assert_allclose(prediction.clearance_scale[idx], 1.0)

    def test_chance_off_keeps_deterministic_cells(self):
        prediction, metadata = self._build(0.95, 0.05, chance=False)
        idx = scenario_index(prediction, "1:cutout+2:lk")
        k = self.LEAVE_STEP
        self.assertEqual(metadata[idx]["selected_vehicle_ids"], [1] * k + [2] * (self.HORIZON + 1 - k))
        self.assertTrue(prediction.active_mask[idx].all())
        self.assertTrue(np.all(np.isnan(prediction.chance_confidence[idx])))
        np.testing.assert_allclose(prediction.clearance_scale[idx], 1.0)
        self.assertEqual(metadata[idx]["vacated_lane_steps"], [])


class TestCutInRegression(unittest.TestCase):
    """The extension must not touch adjacent-lane vehicles.

    Left-adjacent TV (lk 0.3 / cutin 0.7, in the ego lane from step 3) and a
    right-lane straight car (cutin 0.04, gated).  Expected cells follow from
    the existing rules alone: the TV's cut-in mode carries beta = min(0.7, 0.6)
    = 0.6 and scale 1.0 from its lane-entry step, everything else is inactive.
    """

    HORIZON = 6

    @staticmethod
    def _adjacent_raw(vehicle_id, s0, lateral, raw_probs, cutin_raw_idx, entry_step):
        s = s0 + 10.0 * DT * np.arange(1, 7)
        stay = np.column_stack((s, np.full(6, lateral)))
        d_in = np.full(6, lateral)
        d_in[entry_step - 1:] = 0.0
        modes = [stay, stay, stay]
        modes[cutin_raw_idx] = np.column_stack((s, d_in))
        return {
            "vehicle_id": vehicle_id,
            "raw_intention_prob": np.asarray(raw_probs, dtype=float),
            "pred_traj_frenet": np.stack(modes),
            "raw_pred_vel": np.ones((3, 6, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 6), dtype=bool),
        }

    def _processed(self):
        kwargs = {"cutin_chance_ref": REF, "cutin_probability_threshold": VANISH}
        tv = process(
            self._adjacent_raw(10, 20.0, 3.5, [0.3, 0.0, 0.7], 2, 3),
            REL_LEFT_ADJACENT, 20.0, 3.5, 10.0, self.HORIZON, **kwargs)
        straight = process(
            self._adjacent_raw(11, 30.0, -3.5, [0.96, 0.04, 0.0], 1, 2),
            REL_RIGHT_ADJACENT, 30.0, -3.5, 10.0, self.HORIZON, **kwargs)
        return tv, straight

    def test_adjacent_lane_modes_follow_the_existing_rules(self):
        tv, straight = self._processed()
        tv_modes, straight_modes = modes_by_name(tv), modes_by_name(straight)
        self.assertAlmostEqual(tv_modes["cutin"].chance_confidence, REF, places=9)
        self.assertEqual(tv_modes["cutin"].clearance_scale, 1.0)
        np.testing.assert_array_equal(
            tv_modes["cutin"].active_mask, [False, False, False, True, True, True, True])
        self.assertFalse(straight_modes["cutin"].active_mask.any(), "gated below 0.1")
        for mode in (tv_modes["lk"], straight_modes["lk"]):
            self.assertTrue(np.isnan(mode.chance_confidence))
            self.assertEqual(mode.clearance_scale, 1.0)
            self.assertFalse(mode.active_mask.any())

    def test_cells_match_hand_computed_values_and_the_builder_without_the_extension(self):
        tv, straight = self._processed()
        build = lambda **kwargs: build_multitarget_lead_prediction(  # noqa: E731
            [tv, straight],
            ego_state=np.array([0.0, 10.0]),
            horizon=self.HORIZON,
            desired_speed=15.0,
            num_modes=4,
            **kwargs
        )
        extended, extended_meta = build(reference_beta=REF, vanish_threshold=VANISH)
        plain, plain_meta = build()

        expected_active = np.zeros((4, self.HORIZON + 1), dtype=bool)
        expected_confidence = np.full((4, self.HORIZON + 1), np.nan)
        for name in ("10:cutin+11:lk", "10:cutin+11:cutin"):
            idx = scenario_index(extended, name)
            expected_active[idx, 3:] = True
            expected_confidence[idx, 3:] = REF
        np.testing.assert_array_equal(extended.active_mask, expected_active)
        np.testing.assert_array_equal(extended.chance_confidence, expected_confidence)
        np.testing.assert_array_equal(extended.clearance_scale, np.ones((4, self.HORIZON + 1)))

        np.testing.assert_array_equal(extended.active_mask, plain.active_mask)
        np.testing.assert_array_equal(extended.chance_confidence, plain.chance_confidence)
        np.testing.assert_array_equal(extended.clearance_scale, plain.clearance_scale)
        np.testing.assert_array_equal(extended.means, plain.means)
        np.testing.assert_array_equal(extended.k_group_map, plain.k_group_map)
        self.assertEqual(
            [meta["effective_lead_keys"] for meta in extended_meta],
            [meta["effective_lead_keys"] for meta in plain_meta])
        self.assertTrue(all(meta["vacated_lane_steps"] == [] for meta in extended_meta))


if __name__ == "__main__":
    unittest.main()
