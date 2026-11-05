import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    build_multitarget_lead_prediction,
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


OUT = [3.5] * HORIZON


class TestOccupancyActivation(unittest.TestCase):
    """``occin<eps>``: a cut-in hypothesis's cell at step k >= 1 is active iff ``p * P_in(k) > eps``."""

    def test_the_activation_step_moves_with_the_probability(self):
        memberships = np.zeros((1, HORIZON + 1), dtype=bool)
        first = {}
        for p in (0.95, 0.5, 0.2, 0.12, 0.1):
            occ = occupancy_memberships(memberships, RAMP[None, :], np.array([p]), 0.1)[0]
            first[p] = int(np.flatnonzero(occ)[0]) if occ.any() else None
            # once the product passes eps it stays on along a rising P_in
            if occ.any():
                self.assertTrue(occ[first[p]:].all())
        self.assertEqual(first, {0.95: 3, 0.5: 4, 0.2: 6, 0.12: 7, 0.1: None})   # p <= eps never switches on

    def test_step_zero_keeps_the_measured_test(self):
        memberships = np.zeros((2, HORIZON + 1), dtype=bool)
        memberships[1, 0] = True
        occ = occupancy_memberships(memberships, np.ones((2, HORIZON + 1)), np.array([0.05, 0.9]), 0.1)
        self.assertFalse(occ[0, 0])
        self.assertTrue(occ[1, 0])
        self.assertFalse(occ[0, 1:].any())
        self.assertTrue(occ[1, 1:].all())

    def test_without_spread_it_is_the_mean_test_with_the_gate(self):
        # P_in an indicator (sigma -> 0): in iff the mean is in and p > eps, the 0.1 gate's limit at eps = 0.1
        rng = np.random.default_rng(5)
        for _ in range(100):
            mean = rng.random((3, HORIZON + 1)) < 0.5
            p = rng.dirichlet(np.ones(3))
            occ = occupancy_memberships(mean, mean.astype(float), p, 0.1)
            np.testing.assert_array_equal(occ[:, 1:], mean[:, 1:] & (p[:, None] > 0.1))


class TestOccupancyInProcessing(unittest.TestCase):

    def setUp(self):
        # left-adjacent vehicle: RLC is its cut-in; its mean trajectory enters at step 3; the LK trajectory's mean
        # enters at step 5 (trajcutin makes it cutin/LK), LLC never
        self.raw = raw([0.3, 0.05, 0.65], [3.5, 3.5, 3.5, 3.5, 0.3, 0.2, 0.1, 0.0], OUT,
                       [3.5, 2.0, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0])
        steps = HORIZON + 1
        self.mean = np.zeros((3, steps), dtype=bool)
        self.mean[0, 5:] = True
        self.mean[2, 3:] = True
        self.pin = np.zeros((3, steps))
        self.pin[2] = RAMP                                     # the cut-in's spread reaches the path from step 2
        self.pin[0, 4:] = 0.3                                   # the LK trajectory's: at most 0.3

    def test_mean_test_and_trajcutin_today(self):
        modes = by_label(process(self.raw, REL_LEFT_ADJACENT, mode_lane_memberships=self.mean, lk_entry_as_cutin=True))
        self.assertEqual(sorted(modes), ["cutin", "cutin/LK", "lk"])
        np.testing.assert_array_equal(modes["cutin"].active_mask, self.mean[2])
        np.testing.assert_array_equal(modes["cutin/LK"].active_mask, self.mean[0])

    def test_occupancy_replaces_the_mean_test(self):
        processed = process(self.raw, REL_LEFT_ADJACENT, mode_lane_memberships=self.mean, lk_entry_as_cutin=True,
                            mode_lane_probabilities=self.pin, occupancy_risk=0.1)
        modes = by_label(processed)
        # LK: 0.3 * 0.3 = 0.09 <= 0.1, so it is no cut-in hypothesis any more and stays lane keeping (inactive)
        self.assertEqual(sorted(modes), ["cutin", "lk"])
        self.assertEqual(list(modes["lk"].raw_mode_indices), [0, 1])
        self.assertFalse(modes["lk"].active_mask.any())
        # RLC (0.65): active from the first step with 0.65 * P_in > 0.1, step 4 (P_in 0.25), one step later than its
        # mean trajectory
        expected = np.zeros(HORIZON + 1, dtype=bool)
        expected[4:] = True
        np.testing.assert_array_equal(modes["cutin"].active_mask, expected)
        self.assertAlmostEqual(modes["cutin"].probability, 0.65)
        info = processed.branch_info["occupancy"]
        np.testing.assert_array_equal(info["mode_lane_membership_mean"], self.mean)
        # against today's cells: cutin step 3 and cutin/LK steps 5..8 suppressed, nothing activated
        self.assertEqual((info["activated"], info["suppressed"]), (0, 1 + 4))

    def test_an_lk_trajectory_can_become_a_cutin_hypothesis(self):
        # the LK trajectory's spread reaches the path although its mean never enters: 0.3 * 0.5 > 0.1
        mean = self.mean.copy()
        mean[0] = False
        pin = self.pin.copy()
        pin[0, 6:] = 0.5
        modes = by_label(process(self.raw, REL_LEFT_ADJACENT, mode_lane_memberships=mean, lk_entry_as_cutin=True,
                                 mode_lane_probabilities=pin, occupancy_risk=0.1))
        self.assertEqual(sorted(modes), ["cutin", "cutin/LK", "lk"])
        np.testing.assert_array_equal(np.flatnonzero(modes["cutin/LK"].active_mask), [6, 7, 8])

    def test_the_gate_is_replaced(self):
        # a cut-in of p = 0.08 < 0.1: the gate would switch it off; with eps = 0.05 its certain occupancy keeps it on
        r = raw([0.92, 0.0, 0.08], OUT, OUT, [3.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mean = np.zeros((3, HORIZON + 1), dtype=bool)
        mean[2, 2:] = True
        gated = by_label(process(r, REL_LEFT_ADJACENT, mode_lane_memberships=mean, lk_entry_as_cutin=True))
        self.assertFalse(gated["cutin"].active_mask.any())
        occ = by_label(process(r, REL_LEFT_ADJACENT, mode_lane_memberships=mean, lk_entry_as_cutin=True,
                               mode_lane_probabilities=mean.astype(float), occupancy_risk=0.05))
        np.testing.assert_array_equal(occ["cutin"].active_mask, mean[2])

    def test_ego_lane_targets_are_unchanged(self):
        r = raw([0.8, 0.1, 0.1], [0.0] * HORIZON, [0.0] * HORIZON, [0.0, 0.0, 0.0, 0.0, 0.0, 3.5, 3.5, 3.5])
        mean = np.ones((3, HORIZON + 1), dtype=bool)
        mean[2, 6:] = False
        plain = by_label(process(r, REL_EGO_LANE, d0=0.0, mode_lane_memberships=mean, lk_entry_as_cutin=True,
                                 no_cutout=True))
        occ = process(r, REL_EGO_LANE, d0=0.0, mode_lane_memberships=mean, lk_entry_as_cutin=True, no_cutout=True,
                      mode_lane_probabilities=np.zeros((3, HORIZON + 1)), occupancy_risk=0.1)
        self.assertIsNone(occ.branch_info["occupancy"])
        occ = by_label(occ)
        self.assertEqual(sorted(plain), sorted(occ))
        for name in plain:
            np.testing.assert_array_equal(occ[name].active_mask, plain[name].active_mask)

    def test_entrypass_reads_the_activated_step(self):
        # entrypass evaluates a hypothesis at its first active step, which occin moves (3 -> 4 here)
        for kwargs, step in (({}, 3), ({"mode_lane_probabilities": self.pin, "occupancy_risk": 0.1}, 4)):
            processed = process(self.raw, REL_LEFT_ADJACENT, mode_lane_memberships=self.mean,
                                lk_entry_as_cutin=True, cutin_chance_ref=0.6, full_footprint=True, **kwargs)
            _, metadata = build_multitarget_lead_prediction(
                [processed], ego_state=[0.0, 10.0], horizon=HORIZON, desired_speed=20.0, num_modes=8,
                reference_beta=0.6, vanish_threshold=0.1, entry_plan={7: 10.0 * DT * np.arange(HORIZON + 1)},
                entry_clearance=4.5, lead_merge=True, merge_first=True)
            decision = [d for d in metadata[0]["entry_pass"] if d["label"] == "cutin"][0]
            self.assertEqual(decision["entry_step"], step)

    def test_needs_trajcutin_and_the_probabilities(self):
        with self.assertRaises(ValueError):
            process(self.raw, REL_LEFT_ADJACENT, mode_lane_memberships=self.mean,
                    mode_lane_probabilities=self.pin, occupancy_risk=0.1)
        with self.assertRaises(ValueError):
            process(self.raw, REL_LEFT_ADJACENT, mode_lane_memberships=self.mean, lk_entry_as_cutin=True,
                    occupancy_risk=0.1)

    def test_predict_acc_needs_the_probabilistic_test(self):
        # raised before the adapter is used at all
        with self.assertRaises(ValueError):
            STDAN3IntACCAdapter.predict_acc(
                None, ego_state=np.array([0.0, 10.0]), target_states_frenet={}, target_relations={}, trackings={},
                frenet_handler=None, horizon=HORIZON, desired_speed=20.0, num_modes=8, occupancy_risk=0.1)


class TestOccupancyToken(unittest.TestCase):

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_occupancy_risk
        self.assertEqual(parse(BASE + "_occin0.1"), 0.1)
        self.assertEqual(parse(BASE + "_occin0.05_cvar0.5_2"), 0.05)
        self.assertIsNone(parse(BASE))
        self.assertIsNone(parse("acc_x_occin_ra0"))
        self.assertIsNone(parse("acc_x_xoccin0.1"))
        self.assertIsNone(parse("acc_x_occin0.1x"))

    def test_the_carla_agent_refuses_it(self):
        vehicle = mock.MagicMock()
        for policy, message in ((BASE + "_occin0.1", "CARLA agent has none"),
                                (BASE.replace("_trajcutin", "") + "_occin0.1", "trajcutin")):
            with self.assertRaises(ValueError) as caught:
                ACCNairSMPCAgent(vehicle, None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8, smpc_config=policy)
            self.assertIn(message, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
