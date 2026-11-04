"""``passnocap`` (2026-11-04, maintainer's request after the ``passmiqp`` result): no acceleration cap on the pass
branch of the pass / yield MIQP, so the ego may plan to accelerate past a merging car within the design's own limits;
the candidate pre-filter then reads the maximum-acceleration displacement."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from test_passmiqp import DT, HORIZON, L, PrefilterTests, design_config, prediction, solve  # noqa: E402
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    build_multitarget_lead_prediction,
    pass_cap_displacement,
    pass_maxacc_displacement,
)
from utils.acc_nair_smpc import PASS_MIQP_FAR_BEHIND_M, PASS_MIQP_SLOTS  # noqa: E402


class ConfigAndParseTests(unittest.TestCase):

    def test_needs_passmiqp(self):
        with self.assertRaises(ValueError):
            design_config(pass_miqp_no_cap=True)
        design_config(pass_miqp_margin=0.0, pass_miqp_no_cap=True)

    def test_agent_parses_the_token(self):
        parse = ACCNairSMPCAgent._parse_pass_no_cap
        self.assertFalse(parse("acc_..._entrypass0_passmiqp"))
        self.assertTrue(parse("acc_..._entrypass0_passmiqp_passnocap"))
        self.assertTrue(parse("acc_..._passmiqp2_passnocap_rmove1000"))
        self.assertFalse(parse("acc_..._xpassnocap"))


class DisplacementTests(unittest.TestCase):

    def test_maxacc_displacement_bounds_the_cap_displacement(self):
        # from a non-negative a_0 the maximum-acceleration displacement is at or above the no-acceleration one
        for v0, a0 in ((10.0, 0.0), (10.0, 1.0), (19.5, 0.5)):
            reach = pass_maxacc_displacement(HORIZON, DT, v0, a0, 20.0, 2.0, 5.0)
            cap = pass_cap_displacement(HORIZON, DT, v0, a0, 20.0)
            self.assertTrue(np.all(reach >= cap - 1e-12))
            self.assertTrue(np.all(np.diff(reach) >= 0.0))
        # while braking (a_0 < 0) the jerk limit keeps it below the held speed's at first (the cap displacement
        # holds the speed at once; the true reach cannot), and above it later
        reach = pass_maxacc_displacement(HORIZON, DT, 15.0, -2.0, 20.0, 2.0, 5.0)
        cap = pass_cap_displacement(HORIZON, DT, 15.0, -2.0, 20.0)
        self.assertLess(reach[1], cap[1])
        self.assertGreater(reach[-1], cap[-1])
        # from a_0 = 0 at jerk 5 the acceleration reaches 2 after 0.4 s: the exact triple integrator
        reach = pass_maxacc_displacement(HORIZON, DT, 10.0, 0.0, 20.0, 2.0, 5.0)
        self.assertAlmostEqual(reach[1], 10.0 * DT + 5.0 * DT ** 3 / 6.0, places=12)
        self.assertAlmostEqual(reach[2], reach[1] + (10.0 + 0.1) * DT + 0.5 * 1.0 * DT ** 2 + 5.0 * DT ** 3 / 6.0, places=12)
        self.assertAlmostEqual(reach[-1] - reach[-2], 15.2 * DT + 0.5 * 2.0 * DT ** 2, places=9)   # v(2.8 s) = 15.2
        # capped at v_max: the last steps advance by v_max dt
        reach = pass_maxacc_displacement(HORIZON, DT, 19.8, 2.0, 20.0, 2.0, 5.0)
        self.assertAlmostEqual(reach[-1] - reach[-2], 20.0 * DT, places=12)
        # no jerk bound: a_max at once (the jerk over the first step is a_max / dt)
        reach = pass_maxacc_displacement(2, DT, 10.0, 0.0, 20.0, 2.0, None)
        self.assertAlmostEqual(reach[1], 10.0 * DT + (2.0 / DT) * DT ** 3 / 6.0, places=12)


class NoCapPrefilterTests(PrefilterTests):
    """The pre-filter with ``pass_plan``: the candidate band from the maximum-acceleration displacement."""

    def _build(self, targets, ego_s=100.0, margin=0.0, slots=PASS_MIQP_SLOTS, num_modes=8, no_cap=False):
        disp = pass_cap_displacement(HORIZON, DT, 10.0, 0.0, 20.0)
        extra = {}
        if no_cap:
            reach = pass_maxacc_displacement(HORIZON, DT, 10.0, 0.0, 20.0, 2.0, 5.0)
            extra = {"pass_plan": {t.vehicle_id: reach for t in targets}}
        return build_multitarget_lead_prediction(
            targets, ego_state=[ego_s, 10.0], horizon=HORIZON, desired_speed=20.0, num_modes=num_modes,
            reference_beta=0.6, vanish_threshold=0.1, entry_plan={t.vehicle_id: disp for t in targets},
            entry_clearance=L, entry_margin=PASS_MIQP_FAR_BEHIND_M, pass_margin=margin, pass_slots=slots,
            lead_merge=True, merge_first=True, **extra)

    def test_without_pass_plan_the_decisions_carry_the_same_gap(self):
        pred, meta = self._build(self._targets(100.0, [-3.0, 2.0]))
        for d in meta[0]["entry_pass"]:
            self.assertEqual(d["pass_gap_m"], d["entry_gap_m"])

    def test_a_pass_reachable_only_by_accelerating_becomes_a_candidate(self):
        # entry at step 3 (0.6 s): the max-acceleration displacement gains 5 * 0.6^3 / 6 ... up to 0.1 m there,
        # so a gap of +0.05 m (rear just ahead of the ego's front at the held speed) is reachable, +1 m is not
        reach = pass_maxacc_displacement(HORIZON, DT, 10.0, 0.0, 20.0, 2.0, 5.0)
        extra = float(reach[3] - 10.0 * DT * 3)
        self.assertGreater(extra, 0.05)
        self.assertLess(extra, 1.0)
        pred_cap, meta_cap = self._build(self._targets(100.0, [0.05, 1.0, -12.0]))
        pred_free, meta_free = self._build(self._targets(100.0, [0.05, 1.0, -12.0]), no_cap=True)
        cap = {d["vehicle_id"]: d for d in meta_cap[0]["entry_pass"]}
        free = {d["vehicle_id"]: d for d in meta_free[0]["entry_pass"]}
        self.assertFalse(cap[20]["candidate"])                     # with the cap: yields
        self.assertTrue(free[20]["candidate"])                     # without it: a candidate
        self.assertAlmostEqual(free[20]["pass_gap_m"], 0.05 - extra, places=9)
        self.assertEqual(free[20]["entry_gap_m"], cap[20]["entry_gap_m"])   # the no-acceleration gap is kept
        self.assertFalse(free[21]["candidate"] or free[21]["excluded"])    # +1 m: unreachable, yields
        self.assertTrue(free[22]["excluded"] and not free[22]["candidate"])  # far behind: the entry_plan rule
        self.assertIsNone(pred_cap.pass_candidates or None)
        self.assertEqual([c["vehicle_id"] for c in pred_free.pass_candidates], [20])


class NoCapSolveTests(unittest.TestCase):

    def test_a_pass_that_needs_acceleration_is_chosen_without_the_cap(self):
        # the TV's rear 1 m ahead of the ego's front at step 8 (1.6 s) under the held speed: the ego can get
        # ahead only by accelerating (about 0.5 m/s^2 on average); with the cap it yields, without it passes
        pred = prediction(0.0, 15.0, 1.0, entry_step=8, next_gap=80.0)
        _, capped = solve(design_config(pass_miqp_margin=0.0), pred)
        _, free = solve(design_config(pass_miqp_margin=0.0, pass_miqp_no_cap=True), pred)
        self.assertEqual([d["pass"] for d in capped.pass_decisions], [False])
        self.assertEqual([d["pass"] for d in free.pass_decisions], [True])
        self.assertGreater(free.pass_accel_max, 0.5)                        # planned acceleration above a_0 = 0
        self.assertGreater(float(np.max(free.u_nominal[1, :8])), 0.5)
        self.assertLess(free.slack_max, 1e-6)
        self.assertGreater(free.action, capped.action)
        # the pass row holds on the free plan: the ego's front ahead of the TV's rear at the entry step
        self.assertGreaterEqual(free.x_nominal[1, 8, 0] + L - pred.means[1, 8, 0], -1e-6)
        # the free pass stays within the design's limits
        self.assertLessEqual(float(np.max(free.u_nominal)), 2.0 + 1e-6)
        self.assertLessEqual(float(np.max(np.abs(free.j_nominal))), 5.0 + 1e-6)

    def test_an_unreachable_pass_still_yields(self):
        # +4 m at step 4: even the maximum acceleration gains under 0.4 m by then; the hard pass row cannot be
        # met, so the MIQP yields (with the slack, as the plain QP does)
        pred = prediction(0.0, 15.0, 4.0, entry_step=4, next_gap=80.0)
        _, free = solve(design_config(pass_miqp_margin=0.0, pass_miqp_no_cap=True), pred)
        _, base = solve(design_config(), prediction(0.0, 15.0, 4.0, entry_step=4, next_gap=80.0, slot=False))
        self.assertEqual([d["pass"] for d in free.pass_decisions], [False])
        self.assertAlmostEqual(free.action, base.action, places=4)

    def test_a_pass_that_needs_no_acceleration_is_free_to_accelerate(self):
        # the TV enters 3 m behind: both pass; without the cap the plan may accelerate before the pass step
        # (little here: the first tick's reference still eases toward the TV), with the cap it is held at a_0 = 0
        pred = prediction(0.0, 15.0, -3.0, next_gap=80.0)
        _, capped = solve(design_config(pass_miqp_margin=0.0), pred)
        _, free = solve(design_config(pass_miqp_margin=0.0, pass_miqp_no_cap=True), pred)
        self.assertTrue(capped.pass_decisions[0]["pass"] and free.pass_decisions[0]["pass"])
        self.assertLessEqual(float(np.max(capped.u_nominal[1, :4])), 1e-6)
        self.assertGreater(float(np.max(free.u_nominal[1, :4])), 1e-3)
        self.assertGreater(free.pass_accel_max, capped.pass_accel_max)

    def test_no_candidate_ticks_solve_the_plain_qp(self):
        pred = prediction(0.0, 15.0, 30.0, next_gap=30.0, slot=False)
        _, free = solve(design_config(pass_miqp_margin=0.0, pass_miqp_no_cap=True), pred)
        _, base = solve(design_config(), pred)
        self.assertEqual(free.action, base.action)
        np.testing.assert_array_equal(free.policy.h, base.policy.h)


if __name__ == "__main__":
    unittest.main()
