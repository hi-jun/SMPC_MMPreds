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
    process_vehicle_prediction,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


HORIZON = 6
DT = 0.2


def raw(probs, d_lk, d_llc, d_rlc, s0=20.0, speed=10.0):
    """Three trajectories (LK, LLC, RLC) with the given lateral offsets per step 1..HORIZON (0 = ego-lane centre)."""
    s = s0 + speed * DT * np.arange(1, HORIZON + 1)
    traj = np.stack([np.column_stack((s, np.asarray(d, dtype=float))) for d in (d_lk, d_llc, d_rlc)])
    vel = np.zeros((3, HORIZON, 2))
    vel[:, :, 0] = speed
    return {"vehicle_id": 1, "raw_intention_prob": np.asarray(probs, dtype=float), "pred_traj_frenet": traj,
            "raw_pred_vel": vel, "signed_t_cross": 1.0, "valid_mask": np.ones((3, HORIZON), dtype=bool)}


def modes(raw_prediction, relation, d0=0.0, **kwargs):
    processed = process_vehicle_prediction(
        raw_prediction, relation, current_frenet_state=np.array([20.0, d0, 10.0]), horizon=HORIZON, dt=DT,
        ego_lane_threshold=0.5, **kwargs)
    return {mode.mode_name: mode for mode in processed.mode_predictions}


IN = [0.0] * HORIZON


class TestLkHold(unittest.TestCase):
    """``lkhold`` (final-check item 4, 2026-11-03): the ego-lane lead's lane-keeping mode stays active at every step
    when its trajectory is no departure by the 3-step rule."""

    def test_a_wobble_out_at_the_last_two_steps_keeps_the_lead(self):
        # LLC is the most likely staying trajectory: out at steps 5-6 only, so the 3-step rule keeps it as lane keeping
        r = raw([0.3, 0.7, 0.0], IN, [0.0, 0.0, 0.0, 0.0, 3.5, 3.5], IN)
        plain = modes(r, REL_EGO_LANE, no_cutout=True)["lk"]
        self.assertEqual(list(plain.active_mask), [True, True, True, True, True, False, False])
        held = modes(r, REL_EGO_LANE, no_cutout=True, lk_hold=True)["lk"]
        self.assertTrue(held.active_mask.all())
        np.testing.assert_array_equal(held.frenet, plain.frenet)
        self.assertEqual(held.probability, plain.probability)

    def test_a_wobble_out_at_middle_steps_keeps_the_lead(self):
        r = raw([1.0, 0.0, 0.0], [0.0, 3.5, 3.5, 0.0, 0.0, 0.0], IN, IN)
        self.assertEqual(list(modes(r, REL_EGO_LANE)["lk"].active_mask), [True, True, False, False, True, True, True])
        self.assertTrue(modes(r, REL_EGO_LANE, lk_hold=True)["lk"].active_mask.all())

    def test_a_departure_keeps_its_cells_off(self):
        # the lane-keeping trajectory itself ends outside for three steps: a departure, not a wobble
        r = raw([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.5, 3.5, 3.5], IN, IN)
        plain = modes(r, REL_EGO_LANE)["lk"]
        held = modes(r, REL_EGO_LANE, lk_hold=True)["lk"]
        np.testing.assert_array_equal(held.active_mask, plain.active_mask)
        self.assertEqual(list(held.active_mask), [True, True, True, True, False, False, False])

    def test_adjacent_vehicles_are_untouched(self):
        r = raw([0.5, 0.0, 0.5], [3.5] * HORIZON, [3.5] * HORIZON, [3.5, 3.5, 1.0, 0.3, 0.0, 0.0])
        plain = modes(r, REL_LEFT_ADJACENT, d0=3.5, lk_entry_as_cutin=True)
        held = modes(r, REL_LEFT_ADJACENT, d0=3.5, lk_entry_as_cutin=True, lk_hold=True)
        self.assertEqual(sorted(plain), sorted(held))
        for name in plain:
            np.testing.assert_array_equal(held[name].active_mask, plain[name].active_mask)

    def test_the_token_is_a_whole_token(self):
        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        self.assertFalse(ACCNairSMPCAgent._parse_lk_hold(base))
        self.assertTrue(ACCNairSMPCAgent._parse_lk_hold(base + "_lkhold"))
        self.assertTrue(ACCNairSMPCAgent._parse_lk_hold(base + "_lkhold_rmove1000"))
        self.assertFalse(ACCNairSMPCAgent._parse_lk_hold(base + "_lkholdx"))


if __name__ == "__main__":
    unittest.main()
