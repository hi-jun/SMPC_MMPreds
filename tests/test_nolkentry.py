import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_RIGHT_ADJACENT,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402
import test_stdan_3int_acc_integration as base  # noqa: E402  (its gate fixture; a module import keeps its tests out)

REF = 0.6
S = np.array([20.0, 22.0, 24.0])


def raw(prob, lk_d, llc_d, rlc_d):
    """Right-adjacent target: raw mode 1 (LLC) is the lane change toward the ego lane."""
    return {
        "vehicle_id": 10,
        "raw_intention_prob": np.asarray(prob, dtype=float),
        "pred_traj_frenet": np.stack([np.column_stack((S, np.asarray(d, dtype=float))) for d in (lk_d, llc_d, rlc_d)]),
        "raw_pred_vel": np.ones((3, 3, 2)),
        "signed_t_cross": 1.0,
        "valid_mask": np.ones((3, 3), dtype=bool),
    }


def summary(processed):
    return [(m.scenario_label, round(m.probability, 12), tuple(m.raw_mode_indices), tuple(m.active_mask))
            for m in processed.mode_predictions]


class TestNoLaneKeepingEntry(unittest.TestCase):

    def setUp(self):
        self.gate = base.TestCutInProbabilityGate()

    def process(self, r, **kwargs):
        return self.gate._process(r, cutin_chance_ref=REF, lk_entry_as_cutin=True, **kwargs)

    def test_a_lane_keeping_trajectory_that_drifts_in_never_leads(self):
        r = raw([0.5, 0.3, 0.2], [2.0, 0.4, 0.2], [0.0, 0.0, 0.0], [3.5, 3.5, 3.5])
        plain = self.process(r)
        self.assertIn("cutin/LK", [m.scenario_label for m in plain.mode_predictions])
        self.assertAlmostEqual(plain.acc_mode_prob["cutin"], 0.8, places=9)
        p = self.process(r, no_lk_entry=True)
        self.assertEqual([m.scenario_label for m in p.mode_predictions], ["lk", "cutin"])
        lk, cut = p.mode_predictions
        self.assertAlmostEqual(lk.probability, 0.7, places=9)
        self.assertFalse(lk.active_mask[1:].any(), "the lane-keeping hypothesis stays out of the ego's path")
        self.assertAlmostEqual(cut.probability, 0.3, places=9)
        np.testing.assert_array_equal(cut.active_mask, [False, True, True, True])
        self.assertAlmostEqual(p.acc_mode_prob["cutin"], 0.3, places=9)
        self.assertTrue(p.branch_info["no_lk_entry"])
        self.assertFalse(p.branch_info["mode_lane_membership"][[0, 2], 1:].any())

    def test_the_away_direction_never_leads_either(self):
        r = raw([0.6, 0.3, 0.1], [3.5, 3.5, 3.5], [3.5, 0.0, 0.0], [0.3, 0.2, 0.1])
        self.assertIn("cutin/RLC", [m.scenario_label for m in self.process(r).mode_predictions])
        p = self.process(r, no_lk_entry=True)
        self.assertEqual([m.scenario_label for m in p.mode_predictions], ["lk", "cutin"])
        self.assertFalse(p.mode_predictions[0].active_mask[1:].any())

    def test_nothing_changes_without_an_entering_lane_keeping_trajectory(self):
        r = raw([0.6, 0.3, 0.1], [3.5, 3.5, 3.5], [3.5, 0.0, 0.0], [3.5, 3.5, 3.5])
        self.assertEqual(summary(self.process(r)), summary(self.process(r, no_lk_entry=True)))

    def test_ego_lane_vehicles_are_untouched(self):
        r = raw([0.6, 0.3, 0.1], [0.0, 0.0, 0.0], [0.0, 1.0, 3.5], [0.0, -1.0, -3.5])
        a = self.gate._process(r, current_d=0.0, relation=REL_EGO_LANE, cutin_chance_ref=REF)
        b = self.gate._process(r, current_d=0.0, relation=REL_EGO_LANE, cutin_chance_ref=REF, no_lk_entry=True)
        self.assertEqual(summary(a), summary(b))

    def test_occin_spread_of_a_likely_lane_keeping_trajectory(self):
        """occin: a likely lane-keeping trajectory whose mean stays out but whose lateral spread reaches the path
        (0.9 * 0.2 > 0.05) is a cut-in mode of probability 0.9 with trajcutin alone, and out with nolkentry."""
        r = raw([0.9, 0.08, 0.02], [3.5, 3.4, 3.3], [3.5, 1.0, 0.0], [3.5, 3.6, 3.7])
        pin = np.array([[0.0, 0.0, 0.2, 0.3], [0.0, 0.3, 0.9, 1.0], [0.0, 0.0, 0.0, 0.0]])
        kw = dict(mode_lane_probabilities=pin, occupancy_risk=0.05, relation=REL_RIGHT_ADJACENT)
        plain = self.process(r, **kw)
        lkc = [m for m in plain.mode_predictions if m.scenario_label == "cutin/LK"]
        self.assertEqual(len(lkc), 1)
        self.assertAlmostEqual(lkc[0].probability, 0.9, places=9)
        np.testing.assert_array_equal(lkc[0].active_mask, [False, False, True, True])
        p = self.process(r, no_lk_entry=True, **kw)
        self.assertEqual([m.scenario_label for m in p.mode_predictions], ["lk", "cutin"])
        np.testing.assert_array_equal(p.mode_predictions[1].active_mask, [False, False, True, True])   # 0.08 * 0.9 > 0.05
        self.assertAlmostEqual(p.acc_mode_prob["cutin"], 0.08, places=9)


class TestToken(unittest.TestCase):

    def test_parse(self):
        parse = ACCNairSMPCAgent._parse_no_lk_entry
        self.assertTrue(parse("acc_nair_smpc_stdan_3int_modes8_trajcutin_nolkentry"))
        self.assertTrue(parse("acc_nair_smpc_stdan_3int_nolkentry_bcts"))
        self.assertFalse(parse("acc_nair_smpc_stdan_3int_modes8_trajcutin"))
        self.assertFalse(parse("acc_x_nolkentryx"))
        self.assertFalse(parse("acc_x_xnolkentry"))

    def test_needs_stdan_3int(self):
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config="acc_nair_smpc_const_lead_fixed_risk_nominal_safe_distance_nolkentry")
        self.assertIn("nolkentry", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
