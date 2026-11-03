import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    REL_RIGHT_ADJACENT,
    ACCModePrediction,
    ACCProcessedPrediction,
    build_multitarget_lead_prediction,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestMergeFirst(unittest.TestCase):
    """``mergefirst``: every joint mode merged before the ``modes<N>`` cap, the classes ordered by their summed
    probability."""

    HORIZON = 15
    DT = 0.2
    CELLS = ("means", "covariances", "active_mask", "clearance_scale", "chance_confidence", "sigma_confidence",
             "lead_mode_lk", "lead_vehicle_ids", "lead_mode_probability")

    @staticmethod
    def _mode(vehicle_id, name, probability, start_s, speed, d, active_from=0, chance=float("nan")):
        steps = TestMergeFirst.HORIZON + 1
        frenet = np.array([[start_s + speed * TestMergeFirst.DT * k, d, speed] for k in range(steps)])
        mask = np.zeros(steps, dtype=bool)
        if active_from is not None:
            mask[active_from:] = True
        return ACCModePrediction(
            vehicle_id=vehicle_id, mode_name=name, probability=probability, frenet=frenet, active_mask=mask,
            raw_mode_indices=[1], ego_lane_membership_split_step=0, chance_confidence=chance)

    def _shadow_targets(self):
        """Vehicle 1 leads 60 m ahead.  Vehicles 2 (25 m ahead) and 3 (40 m ahead) each keep their lane (0.55,
        never a lead) or cut in from step 4 (0.45).  When both cut in, vehicle 2 is nearer and vehicle 3's
        cut-in is shadowed, so the joint modes (2 cut-in, 3 lk) and (2 cut-in, 3 cut-in) are one class
        "1:lk+2:cutin" (0.45) although each is less likely than (2 lk, 3 lk) alone (0.3025)."""
        lead = self._mode(1, "lk", 1.0, 60.0, 15.0, 0.0)
        lk2 = self._mode(2, "lk", 0.55, 25.0, 15.0, 3.5, active_from=None)
        cutin2 = self._mode(2, "cutin", 0.45, 25.0, 15.0, 0.0, active_from=4, chance=0.45)
        lk3 = self._mode(3, "lk", 0.55, 40.0, 15.0, -3.5, active_from=None)
        cutin3 = self._mode(3, "cutin", 0.45, 40.0, 15.0, 0.0, active_from=4, chance=0.45)
        return [
            ACCProcessedPrediction(1, REL_EGO_LANE, [lead], {}, {}, {}),
            ACCProcessedPrediction(2, REL_LEFT_ADJACENT, [lk2, cutin2], {}, {}, {}),
            ACCProcessedPrediction(3, REL_RIGHT_ADJACENT, [lk3, cutin3], {}, {}, {}),
        ]

    def _build(self, targets, num_modes=8, **kwargs):
        return build_multitarget_lead_prediction(
            targets, ego_state=np.array([0.0, 15.0]), horizon=self.HORIZON, desired_speed=20.0,
            num_modes=num_modes, reference_beta=0.6, vanish_threshold=0.1, **kwargs)

    def test_needs_leadmerge(self):
        with self.assertRaises(ValueError):
            self._build(self._shadow_targets(), merge_first=True)

    def test_off_is_leadmerge_unchanged(self):
        merged, _ = self._build(self._shadow_targets(), lead_merge=True)
        off, _ = self._build(self._shadow_targets(), lead_merge=True, merge_first=False)
        self.assertEqual(off.mode_names, merged.mode_names)
        for name in self.CELLS:
            np.testing.assert_array_equal(getattr(off, name), getattr(merged, name))

    def test_classes_are_ordered_by_their_summed_probability(self):
        merged, _ = self._build(self._shadow_targets(), lead_merge=True)
        first, metadata = self._build(self._shadow_targets(), lead_merge=True, merge_first=True)
        # leadmerge keeps each class at its most likely member's rank
        self.assertEqual(merged.mode_names, ["1:lk", "1:lk+3:cutin", "1:lk+2:cutin"])
        np.testing.assert_allclose(merged.probabilities, [0.3025, 0.2475, 0.45])
        self.assertEqual(first.mode_names, ["1:lk+2:cutin", "1:lk", "1:lk+3:cutin"])
        np.testing.assert_allclose(first.probabilities, [0.45, 0.3025, 0.2475])
        self.assertEqual(metadata[0]["merge_first"], {"joint_modes": 4, "joint_modes_used": 4, "classes": 3})
        # the same classes with the same cells, only reordered
        order = [merged.mode_names.index(name) for name in first.mode_names]
        for name in self.CELLS:
            np.testing.assert_array_equal(getattr(first, name), getattr(merged, name)[order])

    def test_the_cap_keeps_the_most_likely_classes(self):
        # cap 2: leadmerge keeps the two most likely joint modes, (2 lk, 3 lk) and (2 lk, 3 cut-in), and loses
        # class "1:lk+2:cutin" (0.45) whose members rank 3rd and 4th; mergefirst keeps it
        capped, _ = self._build(self._shadow_targets(), num_modes=2, lead_merge=True)
        self.assertEqual(capped.mode_names, ["1:lk", "1:lk+3:cutin"])
        np.testing.assert_allclose(capped.probabilities, [0.55, 0.45])
        first, _ = self._build(self._shadow_targets(), num_modes=2, lead_merge=True, merge_first=True)
        self.assertEqual(first.mode_names, ["1:lk+2:cutin", "1:lk"])
        np.testing.assert_allclose(first.probabilities, [0.45 / 0.7525, 0.3025 / 0.7525])

    def _busy_targets(self):
        """Six targets with several modes each: lane-keeping modes that never lead, gated cut-ins, a cut-in that
        stays behind the ego, a cut-in excluded by entrypass, and cut-ins that shadow one another."""
        targets = [ACCProcessedPrediction(1, REL_EGO_LANE, [self._mode(1, "lk", 1.0, 70.0, 16.0, 0.0)], {}, {}, {})]
        rng = np.random.default_rng(20261103)
        for vid in range(2, 7):
            side = REL_LEFT_ADJACENT if vid % 2 else REL_RIGHT_ADJACENT
            start = float(rng.uniform(-4.0, 45.0))
            speed = float(rng.uniform(11.0, 17.0))
            if vid == 5:    # behind the ego at every step: its cut-ins never pass the lead filter
                start, speed = -40.0, 11.0
            if vid == 6:    # enters 5.5-7.3 m behind the planned ego front: entrypass excludes both cut-ins
                start, speed = 2.0, 12.0
            p = rng.dirichlet([2.0, 1.0, 1.0])
            first_entry = 5 if vid == 6 else int(rng.integers(2, 9))
            second_entry = None if vid == 4 else (8 if vid == 6 else int(rng.integers(3, 12)))
            modes = [self._mode(vid, "lk", float(p[0]), start, speed, 3.5, active_from=None),
                     self._mode(vid, "cutin", float(p[1]), start, speed, 0.0, active_from=first_entry,
                                chance=float(min(p[1], 0.6)))]
            modes.append(self._mode(vid, "cutin", float(p[2]), start, speed, 0.0, active_from=second_entry,
                                    chance=float(min(p[2], 0.6))))
            modes[-1].scenario_label = "cutin/LK"
            targets.append(ACCProcessedPrediction(vid, side, modes, {}, {}, {}))
        return targets

    def test_pooling_equals_merging_every_joint_mode(self):
        entry_plan = {vid: 15.0 * self.DT * np.arange(self.HORIZON + 1) for vid in range(2, 7)}
        kwargs = dict(lead_merge=True, entry_plan=entry_plan, entry_clearance=4.5)
        first, metadata = self._build(self._busy_targets(), merge_first=True, **kwargs)
        # brute force: leadmerge over all 3^5 = 243 joint modes, then ranked by the summed probability
        every, _ = self._build(self._busy_targets(), num_modes=243, **kwargs)
        ranked = sorted(range(every.num_modes), key=lambda idx: -every.probabilities[idx])[:8]
        self.assertGreater(metadata[0]["merge_first"]["classes"], 8)
        self.assertLess(metadata[0]["merge_first"]["joint_modes"], 243)
        self.assertEqual(sum(d["excluded"] for d in metadata[0]["entry_pass"] if d["vehicle_id"] == 6), 2)
        self.assertEqual(first.mode_names, [every.mode_names[idx] for idx in ranked])
        np.testing.assert_allclose(first.probabilities, every.probabilities[ranked] / every.probabilities[ranked].sum(),
                                   rtol=1e-12)
        for name in self.CELLS:
            np.testing.assert_array_equal(getattr(first, name), getattr(every, name)[ranked])

    def test_policy_token(self):
        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_leadmerge"
        self.assertFalse(ACCNairSMPCAgent._parse_merge_first(base))
        self.assertTrue(ACCNairSMPCAgent._parse_merge_first(base + "_mergefirst"))
        self.assertTrue(ACCNairSMPCAgent._parse_merge_first(base + "_mergefirst_safebranch"))
        self.assertFalse(ACCNairSMPCAgent._parse_merge_first(base + "_mergefirstx"))


if __name__ == "__main__":
    unittest.main()
