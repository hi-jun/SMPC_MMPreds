import os
import sys
import unittest
from dataclasses import replace
from unittest import mock

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

HORIZON = 15
DT = 0.2
EGO = np.array([100.0, 15.0])     # ego front at s = 100 m, 15 m/s


def mode(vehicle_id, name, probability, start_s, speed, d, active_from=0, chance=float("nan")):
    steps = HORIZON + 1
    frenet = np.array([[start_s + speed * DT * k, d, speed] for k in range(steps)])
    mask = np.zeros(steps, dtype=bool)
    if active_from is not None:
        mask[active_from:] = True
    return ACCModePrediction(vehicle_id=vehicle_id, mode_name=name, probability=probability, frenet=frenet,
                             active_mask=mask, raw_mode_indices=[1], ego_lane_membership_split_step=0,
                             chance_confidence=chance)


def scene(tv_s):
    """Vehicle 1 leads in the ego lane 60 m ahead at 15 m/s.  Vehicle 2 is the TV in the right lane at ``tv_s`` (its
    current position, rear + L like the ego's front) at 18 m/s: lane keeping (0.4, never in the lane) or a cut-in in
    the ego lane from step 3 (0.6).  Vehicle 3 keeps the left lane far ahead (never a lead)."""
    lead = mode(1, "lk", 1.0, EGO[0] + 60.0, 15.0, 0.0)
    lk2 = mode(2, "lk", 0.4, tv_s, 18.0, -3.5, active_from=None)
    cut2 = mode(2, "cutin", 0.6, tv_s, 18.0, 0.0, active_from=3, chance=0.6)
    lk3 = mode(3, "lk", 1.0, EGO[0] + 80.0, 15.0, 3.5, active_from=None)
    return [ACCProcessedPrediction(1, REL_EGO_LANE, [lead], {}, {}, {}),
            ACCProcessedPrediction(2, REL_RIGHT_ADJACENT, [lk2, cut2], {}, {}, {}),
            ACCProcessedPrediction(3, REL_LEFT_ADJACENT, [lk3], {}, {}, {})]


def build(targets, **kwargs):
    return build_multitarget_lead_prediction(targets, ego_state=EGO, horizon=HORIZON, desired_speed=20.0,
                                             num_modes=8, reference_beta=0.6, vanish_threshold=0.1, **kwargs)


def arrays(prediction):
    return (prediction.means, prediction.active_mask, prediction.lead_vehicle_ids, prediction.lead_mode_probability,
            prediction.chance_confidence, prediction.clearance_scale, np.asarray(prediction.probabilities))


class TestAheadOnly(unittest.TestCase):
    """``aheadonly``: ISA-MPC's obstacle rule -- a target is an obstacle only while its current position is at or
    ahead of the ego's."""

    def assert_same(self, a, b):
        self.assertEqual(a.mode_names, b.mode_names)
        for x, y in zip(arrays(a), arrays(b)):
            np.testing.assert_array_equal(x, y)

    def test_a_tv_beside_the_ego_never_leads(self):
        """TV 2 m behind the ego's front now but faster: without the rule its cut-in leads from step 3 (its predicted
        position is past the ego's current one); with it, no cell is the TV's and the metadata names it."""
        targets = scene(EGO[0] - 2.0)
        plain, _ = build(targets)
        self.assertIn(2, plain.lead_vehicle_ids)
        rule, meta = build(targets, ahead_only=True)
        self.assertNotIn(2, rule.lead_vehicle_ids)
        self.assertEqual(meta[0]["ahead_only"], [2])
        # the lead keeps its cells in every scenario
        self.assertTrue((rule.lead_vehicle_ids == 1).all())

    def test_the_same_as_switching_the_targets_modes_off(self):
        """The rule equals the behind target's modes made inactive by hand (plain, leadmerge, leadmerge + mergefirst)."""
        targets = scene(EGO[0] - 2.0)
        off = [targets[0], replace(targets[1], mode_predictions=[
            replace(m, active_mask=np.zeros_like(m.active_mask)) for m in targets[1].mode_predictions]), targets[2]]
        for kw in ({}, {"lead_merge": True}, {"lead_merge": True, "merge_first": True}):
            rule, _ = build(targets, ahead_only=True, **kw)
            hand, _ = build(off, **kw)
            self.assert_same(rule, hand)

    def test_at_or_ahead_of_the_ego_nothing_changes(self):
        """A target exactly at the ego's position (>= rule) or ahead of it is unchanged, with an empty metadata list."""
        for tv_s in (EGO[0], EGO[0] + 5.0):
            targets = scene(tv_s)
            for kw in ({}, {"lead_merge": True, "merge_first": True}):
                plain, _ = build(targets, **kw)
                rule, meta = build(targets, ahead_only=True, **kw)
                self.assert_same(plain, rule)
                self.assertEqual(meta[0]["ahead_only"], [])

    def test_it_releases_once_the_tv_is_ahead(self):
        """The next tick with the TV 0.1 m ahead of the ego's front: its cut-in leads again, as without the rule."""
        targets = scene(EGO[0] + 0.1)
        rule, _ = build(targets, ahead_only=True)
        plain, _ = build(targets)
        self.assertIn(2, rule.lead_vehicle_ids)
        self.assert_same(plain, rule)

    def test_off_by_default(self):
        targets = scene(EGO[0] - 2.0)
        default, meta = build(targets)
        explicit, _ = build(targets, ahead_only=False)
        self.assert_same(default, explicit)
        self.assertNotIn("ahead_only", meta[0])


class TestToken(unittest.TestCase):

    def test_parse(self):
        parse = ACCNairSMPCAgent._parse_ahead_only
        self.assertTrue(parse("acc_nair_smpc_stdan_3int_modes8_trajcutin_aheadonly"))
        self.assertTrue(parse("acc_nair_smpc_stdan_3int_aheadonly_bcts"))
        self.assertFalse(parse("acc_nair_smpc_stdan_3int_modes8_trajcutin"))
        self.assertFalse(parse("acc_x_aheadonlyx"))
        self.assertFalse(parse("acc_x_xaheadonly"))

    def test_needs_stdan_3int(self):
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config="acc_nair_smpc_const_lead_fixed_risk_nominal_safe_distance_aheadonly")
        self.assertIn("aheadonly", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
