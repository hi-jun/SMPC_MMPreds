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
    entry_ego_displacement,
)
from utils.acc_nair_smpc import (  # noqa: E402
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestLeadMerge(unittest.TestCase):
    """``leadmerge``: joint modes with the same effective lead at every step (and the same cells) are one
    scenario, with their probabilities summed."""

    HORIZON = 15
    DT = 0.2

    @staticmethod
    def _mode(vehicle_id, name, probability, start_s, speed, d, active_from=0, chance=float("nan")):
        steps = TestLeadMerge.HORIZON + 1
        frenet = np.array([[start_s + speed * TestLeadMerge.DT * k, d, speed] for k in range(steps)])
        mask = np.zeros(steps, dtype=bool)
        if active_from is not None:
            mask[active_from:] = True
        return ACCModePrediction(
            vehicle_id=vehicle_id, mode_name=name, probability=probability, frenet=frenet, active_mask=mask,
            raw_mode_indices=[1], ego_lane_membership_split_step=0, chance_confidence=chance)

    def _targets(self):
        """Vehicle 1 leads in the ego lane at 15 m/s, 60 m ahead.  Vehicle 2 keeps its lane (never a lead,
        0.6) or cuts in 25 m ahead at 13 m/s from step 6 (a cut-in chance cell, 0.4).  Vehicle 3 keeps its
        lane (0.7) or has a cut-in mode under the gate (0.3); neither ever leads, so vehicle 3's two modes
        give duplicate scenarios."""
        lead = self._mode(1, "lk", 1.0, 60.0, 15.0, 0.0)
        lk2 = self._mode(2, "lk", 0.6, 25.0, 13.0, 3.5, active_from=None)
        cutin2 = self._mode(2, "cutin", 0.4, 25.0, 13.0, 0.0, active_from=6, chance=0.4)
        lk3 = self._mode(3, "lk", 0.7, 40.0, 15.0, -3.5, active_from=None)
        cutin3 = self._mode(3, "cutin", 0.3, 40.0, 15.0, 0.0, active_from=None, chance=0.3)
        return [
            ACCProcessedPrediction(1, REL_EGO_LANE, [lead], {}, {}, {}),
            ACCProcessedPrediction(2, REL_LEFT_ADJACENT, [lk2, cutin2], {}, {}, {}),
            ACCProcessedPrediction(3, REL_RIGHT_ADJACENT, [lk3, cutin3], {}, {}, {}),
        ]

    def _build(self, **kwargs):
        return build_multitarget_lead_prediction(
            self._targets(), ego_state=np.array([0.0, 15.0]), horizon=self.HORIZON, desired_speed=20.0,
            num_modes=8, reference_beta=0.6, vanish_threshold=0.1, **kwargs)

    def test_off_by_default(self):
        plain, _ = self._build()
        off, _ = self._build(lead_merge=False)
        self.assertEqual(plain.num_modes, 4)
        self.assertEqual(off.mode_names, plain.mode_names)
        np.testing.assert_array_equal(off.probabilities, plain.probabilities)
        np.testing.assert_array_equal(off.means, plain.means)

    def test_duplicates_become_one_scenario_with_the_summed_probability(self):
        plain, _ = self._build()
        merged, metadata = self._build(lead_merge=True)
        self.assertEqual(plain.mode_names, ["1:lk+2:lk+3:lk", "1:lk+2:cutin+3:lk", "1:lk+2:lk+3:cutin",
                                            "1:lk+2:cutin+3:cutin"])
        self.assertEqual(merged.mode_names, ["1:lk", "1:lk+2:cutin"])
        np.testing.assert_allclose(merged.probabilities, [0.6, 0.4])
        self.assertEqual(metadata[0]["merged_mode_names"], ["1:lk+2:lk+3:lk", "1:lk+2:lk+3:cutin"])
        self.assertEqual(metadata[1]["mode_names"], "1:lk+2:cutin")
        # each class keeps its most likely member's cells
        for name in ("means", "covariances", "active_mask", "clearance_scale", "chance_confidence",
                     "lead_vehicle_ids", "lead_mode_lk"):
            np.testing.assert_array_equal(getattr(merged, name), getattr(plain, name)[[0, 1]])
        self.assertEqual(merged.policy_tree.policy_split_steps, (6,))
        self.assertEqual(merged.k_group_map.shape, (2, self.HORIZON))
        # entrypass reads the class names: vehicle 2's plan is the class in which its cut-in does not lead
        plan = np.vstack([np.full(self.HORIZON + 1, 15.0), np.full(self.HORIZON + 1, 10.0)])
        displacement = entry_ego_displacement([2], self.HORIZON, self.DT, 15.0,
                                              (merged.mode_names, merged.probabilities, plan, 0.0))
        np.testing.assert_allclose(displacement[2], 15.0 * self.DT * np.arange(self.HORIZON + 1))

    def test_classes_told_apart_only_by_branch_markers_keep_their_members_names(self):
        # Vehicle 3's cut-in enters from step 8, so it can lead (it leads when vehicle 2 keeps its lane) and
        # gets branchdec markers; when vehicle 2 cuts in too, 3 is shadowed, and (2 cut-in, 3 cut-in) has the
        # leads of (2 cut-in, 3 lk) but different keys -- two classes under one lead-sequence name
        targets = self._targets()
        targets[2].mode_predictions[1] = self._mode(3, "cutin", 0.3, 40.0, 15.0, 0.0, active_from=8, chance=0.3)
        merged, _ = build_multitarget_lead_prediction(
            targets, ego_state=np.array([0.0, 15.0]), horizon=self.HORIZON, desired_speed=20.0, num_modes=8,
            reference_beta=0.6, vanish_threshold=0.1, lead_merge=True, branch_floor_by_vehicle={3: 3})
        self.assertEqual(merged.num_modes, 4)
        self.assertEqual(len(set(merged.mode_names)), 4)
        self.assertIn("1:lk+2:cutin+3:cutin", merged.mode_names)

    def test_a_cutin_that_never_leads_gets_no_branch_marker(self):
        # vehicle 3's cut-in never leads (no in-lane step), so a branchdec floor for it changes nothing
        merged, _ = self._build(lead_merge=True)
        floored, _ = self._build(lead_merge=True, branch_floor_by_vehicle={3: 3})
        self.assertEqual(floored.mode_names, merged.mode_names)
        np.testing.assert_array_equal(floored.probabilities, merged.probabilities)

    def _config(self):
        """The current design's QP: dopt100, jinput, r_a 0, l1slack1e6 (fullfoot is the cells' factor 1)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=4,
            controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        config.__post_init__()
        return config

    def test_the_merged_problem_gives_the_same_command_and_plan(self):
        plain, _ = self._build()
        merged, _ = self._build(lead_merge=True)
        solutions = []
        for prediction in (plain, merged):
            solution = NairACCSMPC(self._config()).solve(
                np.array([0.0, 15.0]), prediction, previous_accel_cmd=0.0, command_dt=0.1, ego_accel=0.0)
            self.assertNotEqual(solution.solve_path, "fallback")
            solutions.append(solution)
        full, small = solutions
        self.assertLess(float(np.max(full.slack)), 1e-6)       # the hard rows hold: the slack count is moot
        self.assertLess(float(np.min(full.u_nominal[1])), -0.05)   # the cut-in branch brakes
        self.assertLess(abs(full.action - small.action), 1e-4)
        np.testing.assert_allclose(small.x_nominal, full.x_nominal[[0, 1]], atol=1e-3)
        np.testing.assert_allclose(full.x_nominal[2], full.x_nominal[0], atol=1e-4)   # duplicates share the plan

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_lead_merge
        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        self.assertFalse(parse(base + "_rmove1000"))
        self.assertTrue(parse(base + "_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge"))
        self.assertTrue(parse(base + "_leadmerge_safelayer"))
        self.assertFalse(parse(base + "_noleadmerge"))

        class _Vehicle:
            def get_world(self):
                return None

        with self.assertRaisesRegex(ValueError, "stdan_3int"):
            ACCNairSMPCAgent(_Vehicle(), None,
                             smpc_config="acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance_leadmerge_rmove1000")


if __name__ == "__main__":
    unittest.main()
