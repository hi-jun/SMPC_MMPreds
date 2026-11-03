"""``passmiqp<m>`` (final-check item 8, 2026-11-04): the pass / yield choice for an entering cut-in hypothesis made
inside the optimization, one binary per candidate in one MIQP per tick (big-M rows, Gurobi through CasADi's
``discrete`` flags), in place of ``entrypass``'s test on a remembered plan."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, ROOT)

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    PASS_MIQP_FAR_BEHIND_M,
    PASS_MIQP_SLOTS,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    ACCModePrediction,
    ACCProcessedPrediction,
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    build_multitarget_lead_prediction,
    pass_cap_displacement,
)

HORIZON = 15
DT = 0.2
TICK = 0.1
L = 4.5
D0 = 3.0
TAU = 1.3


def design_config(**overrides):
    """The current design's QP (dopt100lkr, jinput, r_a 0, l1slack1e6; fullfoot is the prediction's factor 1)."""
    config = original_nair_acc_config(
        horizon=HORIZON, dt=DT, desired_speed=20.0, num_modes=2, controller_variant=VARIANT_FIXED_RISK,
        gurobi_time_limit=0.15)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move, config.r_jerk, config.r_a = 1000.0, 40.0, 0.0
    config.standoff_shortfall_weight = 100.0
    config.standoff_shortfall_on_lk = True
    config.standoff_shortfall_lk_cutin_only = True
    config.jerk_input = True
    config.slack_l1_weight = 1.0e6
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def prediction(s_ego, v_ego, entry_gap, entry_step=4, lead_v=15.0, next_gap=None, next_v=15.0, slot=True):
    """Mode 0: the ego-lane lead (lane keeping, vehicle 10) ``next_gap`` ahead, or no lead; mode 1: the TV
    (vehicle 11) cutting in from ``entry_step``, its rear ``entry_gap`` ahead of the ego's front at that step
    under the held speed ``v_ego`` (the cap displacement at a_0 <= 0), both at ``lead_v``.  With ``slot`` the
    TV's cells carry binary slot 0 and the lane-keeping lead as their next lead."""
    steps = HORIZON + 1
    t = DT * np.arange(steps)
    means = np.zeros((2, steps, 2))
    active = np.zeros((2, steps), dtype=bool)
    s_tv_entry = s_ego + v_ego * t[entry_step] + entry_gap + L
    means[1, :, 0] = s_tv_entry + lead_v * (t - t[entry_step])
    means[1, :, 1] = lead_v
    active[1, entry_step:] = True
    confidence = np.full((2, steps), np.nan)
    confidence[1] = 0.6
    lk = np.zeros((2, steps), dtype=bool)
    ids = np.full((2, steps), -1)
    ids[1] = 11
    if next_gap is not None:
        means[0, :, 0] = s_ego + next_gap + L + next_v * t
        means[0, :, 1] = next_v
        active[0] = True
        lk[0] = True
        ids[0] = 10
        means[1, :entry_step] = means[0, :entry_step]      # the lane-keeping lead leads mode 1 before the entry
        active[1, :entry_step] = True
        lk[1, :entry_step] = True
        ids[1, :entry_step] = 10
        confidence[1, :entry_step] = np.nan
    else:
        means[0, :, 0] = s_ego
        means[0, :, 1] = 20.0
    extra = {}
    if slot:
        pass_slot = np.full((2, steps), -1)
        pass_slot[1, entry_step:] = 0
        next_means = np.zeros((2, steps, 2))
        next_means[:, :, 0], next_means[:, :, 1] = s_ego, 20.0
        next_active = np.zeros((2, steps), dtype=bool)
        next_lk = np.zeros((2, steps), dtype=bool)
        next_ids = np.full((2, steps), -1)
        if next_gap is not None:
            next_means[1, entry_step:] = means[0, entry_step:]
            next_active[1, entry_step:] = True
            next_lk[1, entry_step:] = True
            next_ids[1, entry_step:] = 10
        extra = dict(pass_slot=pass_slot, next_means=next_means, next_active_mask=next_active,
                     next_covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)),
                     next_lead_mode_lk=next_lk, next_lead_vehicle_ids=next_ids,
                     pass_candidates=[dict(vehicle_id=11, label="cutin", entry_step=entry_step,
                                           entry_gap_m=entry_gap, slot=0)])
    return MultimodalLeadPrediction(
        means=means, probabilities=np.array([0.5, 0.5]),
        covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)), active_mask=active,
        chance_confidence=confidence, lead_mode_lk=lk, lead_vehicle_ids=ids, clearance_scale=np.ones((2, steps)),
        mode_names=["10:lk", "10:lk+11:cutin"], **extra)


def solve(config, pred, v_ego=15.0, a0=0.0, controller=None):
    controller = NairACCSMPC(config) if controller is None else controller
    solution = controller.solve(np.array([0.0, v_ego]), pred, previous_accel_cmd=a0, command_dt=TICK, ego_accel=a0)
    return controller, solution


class ConfigTests(unittest.TestCase):

    def test_invalid_combinations(self):
        with self.assertRaises(ValueError):
            design_config(pass_miqp_margin=-1.0)
        with self.assertRaises(ValueError):
            design_config(pass_miqp_margin=0.0, standoff_shortfall_weight=None, standoff_shortfall_on_lk=False,
                          standoff_shortfall_lk_cutin_only=False)
        with self.assertRaises(ValueError):
            design_config(pass_miqp_margin=0.0, relative_brake_decel=3.0)
        with self.assertRaises(ValueError):
            design_config(pass_miqp_margin=0.0, floor_risk=0.3)
        with self.assertRaises(ValueError):
            design_config(pass_miqp_margin=0.0, controller_variant=VARIANT_MULTIMODAL_OL)
        with self.assertRaises(ValueError):
            design_config(pass_miqp_margin=0.0, optimize_k=True)
        design_config(pass_miqp_margin=0.0)

    def test_agent_parses_the_token(self):
        parse = ACCNairSMPCAgent._parse_pass_miqp
        self.assertIsNone(parse("acc_nair_smpc_stdan_3int_cutin_chance0.6_dopt100lkr_entrypass0"))
        self.assertEqual(parse("acc_..._entrypass0_passmiqp"), 0.0)
        self.assertEqual(parse("acc_..._entrypass0_passmiqp_rmove1000"), 0.0)
        self.assertEqual(parse("acc_..._passmiqp2_jinput"), 2.0)
        self.assertEqual(parse("acc_..._passmiqp1.5"), 1.5)
        self.assertIsNone(parse("acc_..._xpassmiqp"))


class PrefilterTests(unittest.TestCase):

    def test_cap_displacement(self):
        d = pass_cap_displacement(HORIZON, DT, 10.0, -2.0, 20.0)      # braking now: the speed held
        np.testing.assert_allclose(d, 10.0 * DT * np.arange(HORIZON + 1))
        d = pass_cap_displacement(HORIZON, DT, 10.0, 1.0, 20.0)       # accelerating: the acceleration held
        t = DT * np.arange(HORIZON + 1)
        np.testing.assert_allclose(d, 10.0 * t + 0.5 * t ** 2)
        d = pass_cap_displacement(HORIZON, DT, 19.0, 2.0, 20.0)       # capped at v_max
        self.assertLess(d[-1], 19.0 * 3.0 + 0.5 * 2.0 * 9.0)
        self.assertAlmostEqual(d[-1] - d[-2], 20.0 * DT, places=12)

    def _targets(self, ego_s, gaps, lead_gap=40.0):
        """An ego-lane lead (vehicle 1) and one adjacent vehicle per entry gap (vehicles 20, 21, ...), each with
        a lane-keeping mode (p 0.5) and a cut-in mode (p 0.5) in the lane from step 3, its rear ``gap`` ahead of
        the ego's front there under the held speed 10 m/s."""
        steps = HORIZON + 1
        t = DT * np.arange(steps)
        targets = []
        frenet = np.zeros((steps, 3))
        frenet[:, 0], frenet[:, 2] = ego_s + lead_gap + L + 10.0 * t, 10.0
        targets.append(ACCProcessedPrediction(1, REL_EGO_LANE, [ACCModePrediction(
            1, "lk", 1.0, frenet, np.ones(steps, dtype=bool), [0], 0)], {}, {}, {}))
        for idx, gap in enumerate(gaps):
            vid = 20 + idx
            frenet = np.zeros((steps, 3))
            frenet[:, 0], frenet[:, 2] = ego_s + 10.0 * t[3] + gap + L + 10.0 * (t - t[3]), 10.0
            active = np.zeros(steps, dtype=bool)
            active[3:] = True
            modes = [ACCModePrediction(vid, "lk", 0.5, frenet, np.zeros(steps, dtype=bool), [0], 3),
                     ACCModePrediction(vid, "cutin", 0.5, frenet, active, [1], 3, chance_confidence=0.5)]
            targets.append(ACCProcessedPrediction(vid, REL_LEFT_ADJACENT, modes, {}, {}, {}))
        return targets

    def _build(self, targets, ego_s=100.0, margin=0.0, slots=PASS_MIQP_SLOTS, num_modes=8):
        disp = pass_cap_displacement(HORIZON, DT, 10.0, 0.0, 20.0)
        return build_multitarget_lead_prediction(
            targets, ego_state=[ego_s, 10.0], horizon=HORIZON, desired_speed=20.0, num_modes=num_modes,
            reference_beta=0.6, vanish_threshold=0.1, entry_plan={t.vehicle_id: disp for t in targets},
            entry_clearance=L, entry_margin=PASS_MIQP_FAR_BEHIND_M, pass_margin=margin, pass_slots=slots,
            lead_merge=True, merge_first=True)

    def test_candidate_band_and_next_lead(self):
        # gaps: far behind (passes without a row), in the band (candidate), ahead (yields, no slot)
        pred, meta = self._build(self._targets(100.0, [-12.0, -3.0, 2.0]))
        decisions = meta[0]["entry_pass"]
        by_vid = {d["vehicle_id"]: d for d in decisions}
        self.assertTrue(by_vid[20]["excluded"] and not by_vid[20]["candidate"])
        self.assertTrue(by_vid[21]["candidate"] and by_vid[21]["slot"] == 0 and not by_vid[21]["excluded"])
        self.assertFalse(by_vid[22]["candidate"] or by_vid[22]["excluded"])
        self.assertEqual([c["vehicle_id"] for c in pred.pass_candidates], [21])
        # the candidate's cells carry slot 0 and, as the next lead, the nearest non-candidate: the yielding
        # vehicle 22 in the scenarios where it cuts in too (its cut-in cell with its confidence), the ego-lane
        # lead 1 (lane keeping, no confidence) elsewhere; the yielding vehicle's own cells carry no slot
        cand_cells = pred.pass_slot == 0
        self.assertTrue(cand_cells.any())
        self.assertTrue(np.all(pred.lead_vehicle_ids[cand_cells] == 21))
        self.assertTrue(np.all(pred.next_active_mask[cand_cells]))
        next_ids = pred.next_lead_vehicle_ids[cand_cells]
        self.assertEqual(set(next_ids.tolist()), {1, 22})
        np.testing.assert_array_equal(pred.next_lead_mode_lk[cand_cells], next_ids == 1)
        np.testing.assert_array_equal(np.isnan(pred.next_chance_confidence[cand_cells]), next_ids == 1)
        self.assertTrue(np.all(pred.next_chance_confidence[cand_cells][next_ids == 22] == 0.5))
        self.assertFalse(np.any(pred.pass_slot[pred.lead_vehicle_ids == 22] >= 0))
        self.assertFalse(np.any(pred.next_active_mask[~cand_cells]))
        # the excluded far-behind vehicle leads nowhere
        self.assertFalse(np.any(pred.lead_vehicle_ids == 20))

    def test_margin_moves_the_band_and_slots_overflow(self):
        pred, meta = self._build(self._targets(100.0, [-3.0, -1.0]), margin=2.0)
        by_vid = {d["vehicle_id"]: d for d in meta[0]["entry_pass"]}
        self.assertTrue(by_vid[20]["candidate"])          # -3 <= -2
        self.assertFalse(by_vid[21]["candidate"])         # -1 > -2: yields
        # four candidates, two slots: the two nearest the ego's front get them, the others pass without a row
        pred, meta = self._build(self._targets(100.0, [-8.0, -2.0, -5.0, -1.0]), slots=2)
        by_vid = {d["vehicle_id"]: d for d in meta[0]["entry_pass"]}
        self.assertEqual((by_vid[23]["slot"], by_vid[21]["slot"]), (0, 1))
        self.assertTrue(by_vid[22]["excluded"] and by_vid[22].get("overflow") and not by_vid[22]["candidate"])
        self.assertTrue(by_vid[20]["excluded"] and by_vid[20].get("overflow"))
        self.assertEqual(sorted(c["slot"] for c in pred.pass_candidates), [0, 1])

    def test_without_pass_margin_nothing_changes(self):
        targets = self._targets(100.0, [-3.0])
        disp = pass_cap_displacement(HORIZON, DT, 10.0, 0.0, 20.0)
        pred, meta = build_multitarget_lead_prediction(
            targets, ego_state=[100.0, 10.0], horizon=HORIZON, desired_speed=20.0, num_modes=8, reference_beta=0.6,
            vanish_threshold=0.1, entry_plan={t.vehicle_id: disp for t in targets}, entry_clearance=L,
            entry_margin=0.0, lead_merge=True, merge_first=True)
        self.assertIsNone(pred.pass_slot)
        self.assertNotIn("candidate", meta[0]["entry_pass"][0])


class SolveTests(unittest.TestCase):

    def test_no_candidate_ticks_solve_the_plain_qp(self):
        pred = prediction(0.0, 15.0, 30.0, next_gap=30.0, slot=False)
        _, with_token = solve(design_config(pass_miqp_margin=0.0), pred)
        _, without = solve(design_config(), pred)
        self.assertEqual(with_token.action, without.action)
        np.testing.assert_array_equal(with_token.policy.h, without.policy.h)
        self.assertEqual(with_token.pass_decisions, [])
        self.assertTrue(np.isnan(with_token.pass_solve_s))

    def test_pass_when_the_vehicle_enters_behind(self):
        # the TV's rear 3 m behind the ego's front at its entry step: yield would need 7.5 m + sigma of gap at
        # once (slack 1e6 per metre), pass needs nothing; the MIQP passes and the relaxed rows use no slack
        config = design_config(pass_miqp_margin=0.0)
        _, sol = solve(config, prediction(0.0, 15.0, -3.0, next_gap=40.0))
        self.assertEqual([d["pass"] for d in sol.pass_decisions], [True])
        self.assertEqual(sol.pass_n_pass, 1)
        self.assertLess(sol.slack_max, 1e-6)
        self.assertGreaterEqual(sol.action, -1e-6)
        self.assertTrue(sol.feasible)
        # the same scene without the token: the cut-in cells are hard, the solve leans on the slack
        _, base = solve(design_config(), prediction(0.0, 15.0, -3.0, next_gap=40.0, slot=False))
        self.assertGreater(base.slack_max, 1.0)
        self.assertLess(base.action, sol.action)
        # the pass row holds on the plan: the ego's front ahead of the TV's rear at the entry step
        k = 4
        self.assertGreaterEqual(sol.x_nominal[1, k, 0] + L - prediction(0.0, 15.0, -3.0).means[1, k, 0], -1e-6)
        # the relaxed main row is violated by the plan (big-M did the work, not the slack)
        gap = prediction(0.0, 15.0, -3.0).means[1, k, 0] - sol.x_nominal[1, k, 0]
        self.assertLess(gap, L + D0)

    def test_yield_when_pass_is_impossible_without_acceleration(self):
        # the TV's rear 2 m ahead at its entry step: the ego cannot get ahead without accelerating, which the
        # cap forbids, so it yields (z = 1) and the solve equals the plain QP's on the same cells
        config = design_config(pass_miqp_margin=0.0)
        _, sol = solve(config, prediction(0.0, 15.0, 2.0, next_gap=40.0))
        self.assertEqual([d["pass"] for d in sol.pass_decisions], [False])
        _, base = solve(design_config(), prediction(0.0, 15.0, 2.0, next_gap=40.0, slot=False))
        self.assertAlmostEqual(sol.action, base.action, places=4)
        self.assertAlmostEqual(sol.slack_max, base.slack_max, places=3)
        self.assertTrue(np.isnan(sol.pass_accel_max))

    def test_yield_when_both_are_feasible_and_yield_is_cheap(self):
        # the TV 30 m ahead: yield costs nothing; the pass row (ego ahead of it) would cost the slack of the
        # next-lead rows (none here) ... but cannot be met without acceleration anyway; z = 1
        config = design_config(pass_miqp_margin=0.0)
        _, sol = solve(config, prediction(0.0, 15.0, 30.0, next_gap=60.0))
        self.assertEqual([d["pass"] for d in sol.pass_decisions], [False])
        self.assertLess(sol.slack_max, 1e-6)

    def test_next_lead_rows_bind_on_pass(self):
        # the TV enters behind (pass); the ego-lane lead sits 1 m inside d_safe at 15 m/s: on the pass branch
        # the next-lead row is the lead's hard standoff, so the plan brakes and leans on the slack by about the
        # deficit, as the plain QP does for the same lead
        config = design_config(pass_miqp_margin=0.0)
        d_safe = L + D0 + TAU * 15.0
        _, sol = solve(config, prediction(0.0, 15.0, -3.0, next_gap=d_safe - L - 1.0))
        self.assertEqual([d["pass"] for d in sol.pass_decisions], [True])
        self.assertLess(sol.action, -0.4)
        self.assertGreater(sol.slack_max, 0.5)
        _, base = solve(design_config(), prediction(0.0, 15.0, 30.0, next_gap=d_safe - L - 1.0, slot=False))
        self.assertAlmostEqual(sol.action, base.action, places=3)
        # the next-lead margin replaces the relaxed cell's margin in the solution's diagnostics
        self.assertLess(sol.chance_margin_min, 0.0)
        self.assertGreater(sol.chance_margin_min, -2.0)

    def test_pass_never_plans_more_acceleration_than_it_has(self):
        config = design_config(pass_miqp_margin=0.0)
        for a0 in (0.0, 0.8):
            _, sol = solve(config, prediction(0.0, 15.0, -3.0, next_gap=60.0), a0=a0)
            self.assertEqual([d["pass"] for d in sol.pass_decisions], [True])
            self.assertLessEqual(float(np.max(sol.u_nominal[1, :4])), a0 + 1e-6)
            self.assertAlmostEqual(sol.pass_accel_max, min(a0, float(np.max(sol.u_nominal[1, :4]))), places=6)
        # without the TV the ego accelerates toward its set speed from a_0 = 0
        _, free = solve(design_config(), prediction(0.0, 15.0, 30.0, next_gap=60.0, slot=False))
        self.assertGreater(float(np.max(free.u_nominal[0, :4])), 0.1)

    def test_margin_makes_a_marginal_pass_a_yield(self):
        # the TV's rear 1 m behind: passes at m = 0, yields at m = 2 (the pass row asks 2 m)
        _, sol0 = solve(design_config(pass_miqp_margin=0.0), prediction(0.0, 15.0, -1.0, next_gap=60.0))
        _, sol2 = solve(design_config(pass_miqp_margin=2.0), prediction(0.0, 15.0, -1.0, next_gap=60.0))
        self.assertTrue(sol0.pass_decisions[0]["pass"])
        self.assertFalse(sol2.pass_decisions[0]["pass"])

    def test_reference_follows_the_last_decision(self):
        # after a pass the candidate's cells take the next lead's reference: the speed reference no longer
        # eases toward the slow TV, so the second solve tracks the set speed (a_0 = 0 keeps the cap at 0)
        config = design_config(pass_miqp_margin=0.0)
        controller, first = solve(config, prediction(0.0, 15.0, -3.0, next_gap=80.0, lead_v=8.0))
        self.assertTrue(first.pass_decisions[0]["pass"])
        ref_first = first.reference.v_ref[1, -1]
        second = controller.solve(np.array([0.0, 15.0]), prediction(0.0, 15.0, -3.0, next_gap=80.0, lead_v=8.0),
                                  previous_accel_cmd=0.0, command_dt=TICK, ego_accel=0.0)
        self.assertTrue(second.pass_decisions[0]["pass"])
        self.assertGreater(second.reference.v_ref[1, -1], ref_first + 1.0)
        self.assertEqual(controller._pass_last, {(11, "cutin"): True})

    def test_fallback_clears_the_decisions(self):
        config = design_config(pass_miqp_margin=0.0, gurobi_time_limit=1.0e-9)
        _, sol = solve(config, prediction(0.0, 15.0, -3.0, next_gap=40.0))
        self.assertEqual(sol.solve_path, "fallback")
        self.assertEqual(sol.pass_decisions, [])


if __name__ == "__main__":
    unittest.main()
