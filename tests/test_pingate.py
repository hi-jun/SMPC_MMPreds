import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_LEFT_ADJACENT,
    occupancy_memberships,
    process_vehicle_prediction,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


HORIZON = 8
DT = 0.2
BASE = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        "_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6")
#: P_in of a trajectory whose lateral spread reaches the ego's path gradually (step 0 = now, not used)
RAMP = np.array([0.0, 0.01, 0.05, 0.12, 0.25, 0.45, 0.7, 0.9, 1.0])
OUT = [3.5] * HORIZON


def raw(probs, d_lk, d_llc, d_rlc, s0=20.0, speed=10.0):
    """Three trajectories (LK, LLC, RLC) with the given lateral offsets per step 1..HORIZON (0 = ego-lane centre)."""
    s = s0 + speed * DT * np.arange(1, HORIZON + 1)
    traj = np.stack([np.column_stack((s, np.asarray(d, dtype=float))) for d in (d_lk, d_llc, d_rlc)])
    vel = np.zeros((3, HORIZON, 2))
    vel[:, :, 0] = speed
    return {"vehicle_id": 7, "raw_intention_prob": np.asarray(probs, dtype=float), "pred_traj_frenet": traj,
            "raw_pred_vel": vel, "signed_t_cross": 1.0, "valid_mask": np.ones((3, HORIZON), dtype=bool)}


def process(raw_prediction, relation, d0=3.5, **kwargs):
    return process_vehicle_prediction(
        raw_prediction, relation, current_frenet_state=np.array([20.0, d0, 10.0]), horizon=HORIZON, dt=DT,
        ego_lane_threshold=0.5, cutin_probability_threshold=0.1, **kwargs)


def by_label(processed):
    return {mode.scenario_label: mode for mode in processed.mode_predictions}


class TestPinGate(unittest.TestCase):
    """``pingate``: with ``occin<eps>`` a trajectory with ``p > eps`` is in at step k >= 1 iff ``P_in(k) > eps``."""

    def test_the_step_no_longer_moves_with_the_probability(self):
        memberships = np.zeros((1, HORIZON + 1), dtype=bool)
        first = {}
        for p in (0.95, 0.5, 0.2, 0.12, 0.1, 0.05):
            occ = occupancy_memberships(memberships, RAMP[None, :], np.array([p]), 0.1, pin_gate=True)[0]
            first[p] = int(np.flatnonzero(occ)[0]) if occ.any() else None
            if occ.any():
                self.assertTrue(occ[first[p]:].all())
        # P_in passes 0.1 at step 3 (0.12) for every p above eps; occin's product gave 3 / 4 / 6 / 7
        self.assertEqual(first, {0.95: 3, 0.5: 3, 0.2: 3, 0.12: 3, 0.1: None, 0.05: None})

    def test_off_is_the_product_and_the_bound_holds_where_it_is_out(self):
        rng = np.random.default_rng(9)
        for _ in range(200):
            mem = rng.random((3, HORIZON + 1)) < 0.5
            pin = rng.random((3, HORIZON + 1))
            pin[rng.random((3, HORIZON + 1)) < 0.1] = np.nan
            p = rng.dirichlet(np.ones(3))
            off = occupancy_memberships(mem, pin, p, 0.05)
            expected = mem.copy()
            expected[:, 1:] = p[:, None] * np.nan_to_num(pin[:, 1:], nan=0.0) > 0.05
            np.testing.assert_array_equal(off, expected)
            on = occupancy_memberships(mem, pin, p, 0.05, pin_gate=True)
            np.testing.assert_array_equal(on[:, 0], mem[:, 0])
            # every step the product rule switches on, the gate does too; every step left out keeps p P_in <= eps
            self.assertTrue(np.all(on[:, 1:] | ~off[:, 1:]))
            prod = p[:, None] * np.nan_to_num(pin[:, 1:], nan=0.0)
            self.assertTrue(np.all(prod[~on[:, 1:]] <= 0.05))

    def test_in_processing(self):
        # left-adjacent vehicle: RLC (0.65) is its cut-in, its spread reaches the path from step 2; the LK
        # trajectory's (0.3) reaches 0.3 from step 4; LLC (0.05) never
        r = raw([0.3, 0.05, 0.65], OUT, OUT, [3.5, 2.0, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0])
        mean = np.zeros((3, HORIZON + 1), dtype=bool)
        mean[2, 3:] = True
        pin = np.zeros((3, HORIZON + 1))
        pin[2] = RAMP
        pin[0, 4:] = 0.3
        kw = dict(mode_lane_memberships=mean, lk_entry_as_cutin=True, mode_lane_probabilities=pin,
                  occupancy_risk=0.1)
        occin = by_label(process(r, REL_LEFT_ADJACENT, **kw))
        gate = by_label(process(r, REL_LEFT_ADJACENT, occupancy_pin_gate=True, **kw))
        # the cut-in: step 4 under the product (0.65 * 0.25), step 3 under the gate (0.12 > 0.1)
        self.assertEqual(int(np.flatnonzero(occin["cutin"].active_mask)[0]), 4)
        self.assertEqual(int(np.flatnonzero(gate["cutin"].active_mask)[0]), 3)
        # the LK trajectory: 0.3 * 0.3 <= 0.1 keeps it lane keeping; the gate (0.3 > 0.1, 0.3 > 0.1) makes it
        # trajcutin's cutin/LK from step 4, unless nolkentry keeps lane keeping out of the path
        self.assertNotIn("cutin/LK", occin)
        np.testing.assert_array_equal(np.flatnonzero(gate["cutin/LK"].active_mask), [4, 5, 6, 7, 8])
        nolk = by_label(process(r, REL_LEFT_ADJACENT, occupancy_pin_gate=True, no_lk_entry=True, **kw))
        self.assertNotIn("cutin/LK", nolk)
        self.assertFalse(nolk["lk"].active_mask.any())
        np.testing.assert_array_equal(nolk["cutin"].active_mask, gate["cutin"].active_mask)

    def test_needs_occin(self):
        r = raw([0.3, 0.05, 0.65], OUT, OUT, OUT)
        with self.assertRaises(ValueError):
            process(r, REL_LEFT_ADJACENT, lk_entry_as_cutin=True, occupancy_pin_gate=True)
        with self.assertRaises(ValueError):    # raised before the adapter is used at all
            STDAN3IntACCAdapter.predict_acc(
                None, ego_state=np.array([0.0, 10.0]), target_states_frenet={}, target_relations={}, trackings={},
                frenet_handler=None, horizon=HORIZON, desired_speed=20.0, num_modes=8, occupancy_pin_gate=True)

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_occupancy_pin_gate
        self.assertTrue(parse(BASE + "_bcts_occin0.05_pingate_nolkentry"))
        self.assertTrue(parse("pingate"))
        self.assertFalse(parse(BASE + "_bcts_occin0.05_nolkentry"))
        self.assertFalse(parse("acc_x_xpingate_ra0"))
        self.assertFalse(parse("acc_x_pingatex"))
        self.assertFalse(parse(BASE + "_latgate"))


if __name__ == "__main__":
    unittest.main()
