"""Unit tests for the provably-correct ACC baseline (utils/pcacc.py), no CARLA."""
import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.pcacc import (  # noqa: E402
    PATH_BOUNDED,
    PATH_CRUISE,
    PATH_EMERGENCY,
    PATH_ICS,
    PATH_NOMINAL,
    PATH_RECAPTURE,
    TABLE_I_A_MIN,
    TABLE_I_JERK_MAX,
    LaneMembershipTracker,
    PCACCConfig,
    PCACCController,
    VehicleObs,
    ego_brake_distance,
    emergency_accel,
    emergency_config,
    is_inevitable_collision,
    lead_brake_distance,
    relevant_vehicles,
    safe_distance,
    standoff,
)

DT_CMD = 0.05
PAPER = "acc_pcacc_paperdsafe_amin10_amax3_jlim10"   # the paper's d_safe and Table I limits


def _numeric_brake_distance(v, a, cfg, step=1.0e-4):
    """Fine-grained reference for ego_brake_distance (zero-order-hold profile)."""
    jerks = cfg.profile_jerks
    s = 0.0
    t = 0.0
    while v > 0.0:
        k = int(t / cfg.dt + 1.0e-9)
        j = jerks[k] if k < len(jerks) else cfg.jerk_min
        if a <= cfg.a_min:
            j = 0.0
        s += v * step + 0.5 * a * step * step
        v += a * step
        a = max(cfg.a_min, a + j * step)
        t += step
    return s


class SafeDistanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PCACCConfig()
        self.paper = PCACCConfig.from_policy_string(PAPER)

    def test_standoff_is_the_forks(self):
        # AGENTS.md §3: L + d0 + tau v centre to centre, so d0 + tau v on the bumper gap.
        self.assertEqual((self.cfg.vehicle_length, self.cfg.d0, self.cfg.time_headway), (4.5, 3.0, 1.3))
        self.assertAlmostEqual(standoff(17.0, self.cfg), 3.0 + 1.3 * 17.0, places=9)
        ctrl = PCACCController(self.cfg)
        self.assertAlmostEqual(ctrl._d_safe(17.0, 0.0, 15.0, DT_CMD), standoff(17.0, self.cfg), places=9)
        paper = PCACCController(self.paper)
        self.assertAlmostEqual(paper._d_safe(17.0, 0.0, 15.0, DT_CMD),
                               safe_distance(17.0, 0.0, 15.0, self.paper, DT_CMD), places=9)

    def test_lead_brakes_at_a_p_min(self):
        self.assertAlmostEqual(lead_brake_distance(15.0, self.cfg), 15.0 ** 2 / 21.0, places=9)

    def test_ego_brake_distance_matches_numeric_integration(self):
        for v, a in ((5.0, 0.0), (12.0, 0.0), (17.0, 0.0), (17.0, -2.0), (17.0, 2.0)):
            self.assertAlmostEqual(
                ego_brake_distance(v, a, self.cfg), _numeric_brake_distance(v, a, self.cfg),
                delta=0.02, msg="v=%s a=%s" % (v, a))

    def test_ego_brake_distance_monotone(self):
        d = [ego_brake_distance(v, 0.0, self.cfg) for v in (5.0, 10.0, 15.0, 20.0)]
        self.assertTrue(all(np.diff(d) > 0.0))
        self.assertGreater(ego_brake_distance(15.0, 1.0, self.cfg), ego_brake_distance(15.0, -1.0, self.cfg))
        self.assertEqual(ego_brake_distance(0.0, 0.0, self.cfg), 0.0)

    def test_safe_distance_magnitudes(self):
        # Fig. 8(d): about 15-17 m at 12 m/s with the lead at the same speed (Table I limits).
        cfg = self.paper
        self.assertTrue(14.0 <= safe_distance(12.0, 0.0, 12.0, cfg, DT_CMD) <= 19.0)
        d_fast_lead = safe_distance(17.0, 0.0, 17.0, cfg, DT_CMD)
        d_slow_lead = safe_distance(17.0, 0.0, 12.0, cfg, DT_CMD)
        self.assertGreater(d_slow_lead, d_fast_lead)
        self.assertGreater(safe_distance(20.0, 0.0, 15.0, cfg, DT_CMD),
                           safe_distance(15.0, 0.0, 15.0, cfg, DT_CMD))

    def test_condition_seven_uses_the_planned_first_step(self):
        # Braking in the first interval shortens the stopping distance.
        self.assertLess(safe_distance(17.0, 0.0, 15.0, self.paper, DT_CMD, u0=-10.0),
                        safe_distance(17.0, 0.0, 15.0, self.paper, DT_CMD, u0=0.0))

    def test_condition_seven_on_the_standoff_uses_the_next_state(self):
        ctrl = PCACCController(self.cfg)
        gap = standoff(17.0, self.cfg) + 0.05
        # Closing on a 15 m/s lead at 17 m/s, the gap after 0.05 s is 0.1 m smaller.
        self.assertTrue(ctrl._next_state_unsafe(gap, 17.0, 0.0, 15.0, 0.0, DT_CMD))
        self.assertFalse(ctrl._next_state_unsafe(gap, 17.0, 0.0, 17.0, 0.0, DT_CMD))
        # The paper's check: the gap now against Prop. IV.5 with the planned jerk.
        paper = PCACCController(self.paper)
        d = safe_distance(17.0, 0.0, 15.0, self.paper, DT_CMD, u0=-10.0)
        self.assertFalse(paper._next_state_unsafe(d + 0.01, 17.0, 0.0, 15.0, -10.0, DT_CMD))
        self.assertTrue(paper._next_state_unsafe(d - 0.01, 17.0, 0.0, 15.0, -10.0, DT_CMD))

    def test_emergency_profile(self):
        cfg = self.cfg
        a_1s = emergency_accel(1.0, 0.0, cfg)
        self.assertAlmostEqual(a_1s, sum(cfg.profile_jerks) * cfg.dt, places=9)
        self.assertAlmostEqual(emergency_accel(DT_CMD, 0.0, cfg), cfg.profile_jerks[0] * DT_CMD, places=9)
        samples = [emergency_accel(t, 0.5, cfg) for t in np.arange(0.0, 3.0, DT_CMD)]
        self.assertTrue(all(np.diff(samples) <= 1.0e-12))
        self.assertEqual(samples[-1], cfg.a_min)

    def test_inevitable_collision_state(self):
        self.assertFalse(is_inevitable_collision(17.5, 17.0, 0.0, 15.0, self.cfg))
        self.assertTrue(is_inevitable_collision(2.0, 17.0, 0.0, 5.0, self.cfg))

    def test_policy_tokens(self):
        cfg = PCACCConfig.from_policy_string("acc_pcacc_amin10_amax3_jlim10_apmin10.5_tc3_acutin1.5")
        self.assertEqual((cfg.a_min, cfg.a_max, cfg.jerk_max, cfg.jerk_min), (-10.0, 3.0, 10.0, -10.0))
        self.assertEqual((cfg.a_p_min, cfg.t_clear_s, cfg.a_cutin), (-10.5, 3.0, -1.5))
        self.assertFalse(cfg.paper_d_safe)
        default = PCACCConfig.from_policy_string("acc_pcacc")
        self.assertEqual((default.a_min, default.a_max, default.jerk_max, default.a_p_min), (-3.0, 2.0, 5.0, -10.5))
        self.assertEqual((default.d0, default.time_headway, default.vehicle_length), (3.0, 1.3, 4.5))
        self.assertFalse(default.paper_d_safe)
        self.assertTrue(PCACCConfig.from_policy_string(PAPER).paper_d_safe)
        PCACCConfig.from_policy_string("acc_pcacc_apmin3")   # a_p_min -3 <= a_min -3 (Spec. III.5)
        with self.assertRaises(ValueError):
            PCACCConfig.from_policy_string("acc_pcacc_apmin2")   # lead braking weaker than the ego's


class ModeSwitchingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PCACCConfig()
        self.ctrl = PCACCController(self.cfg)

    def test_cruise_without_vehicles(self):
        dec = self.ctrl.step(0.0, 14.0, 0.0, 17.0, [], DT_CMD)
        self.assertEqual(dec.path, PATH_CRUISE)
        self.assertAlmostEqual(dec.accel, self.cfg.cruise_gain * 3.0)
        self.assertTrue(dec.feasible)

    def test_nominal_when_the_gap_is_safe(self):
        d = standoff(15.0, self.cfg)
        dec = self.ctrl.step(0.0, 15.0, 0.0, 17.0, [VehicleObs(1, d + 5.0, 15.0, True)], DT_CMD)
        self.assertEqual(dec.path, PATH_NOMINAL)
        self.assertEqual(dec.lead_vid, 1)
        self.assertLess(abs(dec.accel), 0.5)
        self.assertTrue(dec.feasible)

    def test_emergency_follows_the_jerk_profile_and_releases(self):
        cfg = self.cfg
        d = standoff(15.0, cfg)
        unsafe = [VehicleObs(1, d - 3.0, 15.0, True)]
        dec1 = self.ctrl.step(0.0, 15.0, 0.0, 17.0, unsafe, DT_CMD)
        self.assertEqual(dec1.path, PATH_EMERGENCY)
        self.assertFalse(dec1.feasible)
        self.assertAlmostEqual(dec1.accel, emergency_accel(DT_CMD, 0.0, cfg), places=9)
        dec2 = self.ctrl.step(DT_CMD, 15.0, dec1.accel, 17.0, unsafe, DT_CMD)
        self.assertEqual(dec2.path, PATH_EMERGENCY)
        self.assertAlmostEqual(dec2.accel, emergency_accel(2 * DT_CMD, 0.0, cfg), places=9)
        dec3 = self.ctrl.step(2 * DT_CMD, 15.0, dec2.accel, 17.0, [VehicleObs(1, d + 10.0, 15.0, True)], DT_CMD)
        self.assertEqual(dec3.path, PATH_NOMINAL)
        self.assertNotIn(1, self.ctrl._emergency)

    def test_cut_in_inside_the_safe_distance_is_recaptured(self):
        cfg = self.cfg
        v, v_p = 17.0, 15.0
        d = standoff(v, cfg)
        self.ctrl.step(0.0, v, 0.0, 17.0, [VehicleObs(2, d - 8.0, v_p, False)], DT_CMD)
        dec = self.ctrl.step(DT_CMD, v, 0.0, 17.0, [VehicleObs(2, d - 8.0, v_p, True)], DT_CMD)
        self.assertEqual(dec.path, PATH_RECAPTURE)
        self.assertIn(2, self.ctrl._cutin_since)
        self.assertLess(dec.accel, 0.0)
        self.assertGreaterEqual(dec.accel, cfg.jerk_min * DT_CMD)
        extra = dec.per_vehicle[2].extra
        self.assertAlmostEqual(extra["v_p_tc"], max(0.0, v_p + cfg.a_cutin * cfg.t_clear_s), places=6)
        # Safe distance restored (ego slower than the lead) -> safety keeping again.
        d_restored = standoff(12.0, cfg)
        dec = self.ctrl.step(2 * DT_CMD, 12.0, dec.accel, 17.0, [VehicleObs(2, d_restored + 2.0, v_p, True)], DT_CMD)
        self.assertNotIn(2, self.ctrl._cutin_since)
        self.assertEqual(dec.path, PATH_NOMINAL)

    def test_first_sight_inside_the_lane_is_not_a_cut_in(self):
        d = standoff(15.0, self.cfg)
        dec = self.ctrl.step(0.0, 15.0, 0.0, 17.0, [VehicleObs(7, d - 3.0, 15.0, True)], DT_CMD)
        self.assertEqual(dec.path, PATH_EMERGENCY)
        self.assertNotIn(7, self.ctrl._cutin_since)

    def test_entry_outside_the_safe_distance_is_not_a_cut_in(self):
        d = standoff(17.0, self.cfg)
        self.ctrl.step(0.0, 17.0, 0.0, 17.0, [VehicleObs(3, d + 2.0, 15.0, False)], DT_CMD)
        dec = self.ctrl.step(DT_CMD, 17.0, 0.0, 17.0, [VehicleObs(3, d + 2.0, 15.0, True)], DT_CMD)
        self.assertEqual(dec.path, PATH_NOMINAL)
        self.assertNotIn(3, self.ctrl._cutin_since)

    def test_cut_in_is_judged_against_the_forks_standoff(self):
        # At 12 m/s the paper's d_safe (Table I limits) is shorter than the fork's
        # 3 + 1.3 v: a vehicle entering between the two is a cut-in here, not there.
        paper_cfg = PCACCConfig.from_policy_string(PAPER)
        d_paper = safe_distance(12.0, 0.0, 12.0, paper_cfg, DT_CMD)
        d_fork = standoff(12.0, self.cfg)
        self.assertLess(d_paper, d_fork)
        gap = 0.5 * (d_paper + d_fork)
        for cfg, path, d in ((self.cfg, PATH_RECAPTURE, d_fork), (paper_cfg, PATH_NOMINAL, d_paper)):
            ctrl = PCACCController(cfg)
            ctrl.step(0.0, 12.0, 0.0, 17.0, [VehicleObs(8, gap, 12.0, False)], DT_CMD)
            dec = ctrl.step(DT_CMD, 12.0, 0.0, 17.0, [VehicleObs(8, gap, 12.0, True)], DT_CMD)
            self.assertEqual(dec.path, path)
            self.assertAlmostEqual(dec.per_vehicle[8].d_safe, d, places=9)

    def test_keeping_constraint_is_linear_in_the_step_speed(self):
        # (3) on the fork's standoff: ds_k >= d0 + tau v_k at every step, v_k = v_p - dv_k.
        cfg = self.cfg
        qp = self.ctrl._qp_keep
        v, v_p = 16.0, 15.0
        kw = dict(dv_lo=np.full(qp.N, v_p - cfg.v_max), dv_hi=np.full(qp.N, v_p - cfg.v_min),
                  a_lo=cfg.a_min, a_hi=cfg.a_max, ds_min=np.full(qp.N, standoff(v_p, cfg)))
        feasible, _, X = qp.solve([standoff(v, cfg) + 5.0, v_p - v, 0.0], standoff(v, cfg),
                                  ds_dv_coef=cfg.time_headway, **kw)
        self.assertTrue(feasible)
        ds, dv = X[0::3], X[1::3]
        self.assertTrue(np.all(ds + 1.0e-6 >= cfg.d0 + cfg.time_headway * (v_p - dv)))
        # 1 m outside the standoff of the lead's speed but 0.3 m inside its own,
        # closing at 1 m/s: infeasible per step in v_k, feasible against the
        # lead-speed constant.
        x0 = [standoff(v_p, cfg) + 1.0, v_p - v, 0.0]
        self.assertFalse(qp.solve(x0, standoff(v, cfg), ds_dv_coef=cfg.time_headway, **kw)[0])
        self.assertTrue(qp.solve(x0, standoff(v, cfg), **kw)[0])

    def test_inevitable_collision_state_brakes_fully(self):
        self.ctrl.step(0.0, 17.0, 0.0, 17.0, [VehicleObs(4, 2.0, 5.0, False)], DT_CMD)
        dec = self.ctrl.step(DT_CMD, 17.0, 0.0, 17.0, [VehicleObs(4, 2.0, 5.0, True)], DT_CMD)
        self.assertEqual(dec.path, PATH_ICS)
        self.assertAlmostEqual(dec.accel, self.cfg.jerk_min * DT_CMD, places=9)

    def test_selector_takes_the_minimum(self):
        d = standoff(15.0, self.cfg)
        near = VehicleObs(5, d - 3.0, 15.0, True)
        far = VehicleObs(6, d + 20.0, 12.0, True)
        dec = self.ctrl.step(0.0, 15.0, 0.0, 17.0, [far, near], DT_CMD)
        self.assertEqual(dec.lead_vid, 5)
        self.assertEqual(dec.accel, min(v.accel for v in dec.per_vehicle.values()))

    def test_relevant_vehicle_filter(self):
        v = 15.0
        reach = ego_brake_distance(v, 0.0, self.cfg) + 5.0
        vehicles = [VehicleObs(1, 20.0, 10.0, True), VehicleObs(2, 25.0, 12.0, True),
                    VehicleObs(3, 26.0, 8.0, True), VehicleObs(4, reach + 50.0, 5.0, True)]
        kept = [veh.vid for veh in relevant_vehicles(vehicles, v, 0.0, self.cfg, DT_CMD)]
        self.assertEqual(kept, [1, 3])


class CutInRuleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PCACCConfig()

    def _first_prediction(self, lateral, heading, adjacent=True):
        tracker = LaneMembershipTracker(self.cfg)
        for k in range(len(lateral)):
            t = k * DT_CMD
            if tracker.update(9, t, False, adjacent, lateral[k], heading[k]).predicted:
                return t
        return None

    def test_rule_fires_half_a_second_after_both_thresholds(self):
        t = np.arange(0.0, 4.0, DT_CMD)
        lateral = np.where(t < 1.0, 0.0, 1.5 * (t - 1.0))     # > 0.4 m from t = 1.30
        heading = np.where(t >= 0.8, 0.05, 0.0)
        self.assertAlmostEqual(self._first_prediction(lateral, heading), 1.80, places=6)

    def test_rule_needs_both_lateral_and_heading(self):
        t = np.arange(0.0, 4.0, DT_CMD)
        lateral = np.where(t < 1.0, 0.0, 1.5 * (t - 1.0))
        self.assertIsNone(self._first_prediction(lateral, np.full(t.size, 0.02)))
        self.assertIsNone(self._first_prediction(np.full(t.size, 0.3), np.full(t.size, 0.05)))

    def test_rule_ignores_motion_away_from_the_ego_lane(self):
        t = np.arange(0.0, 4.0, DT_CMD)
        lateral = -np.where(t < 1.0, 0.0, 1.5 * (t - 1.0))
        self.assertIsNone(self._first_prediction(lateral, np.full(t.size, -0.05)))
        self.assertIsNone(self._first_prediction(np.full(t.size, 1.0), np.full(t.size, 0.05), adjacent=False))

    def test_dropout_restarts_the_window(self):
        tracker = LaneMembershipTracker(self.cfg)
        predicted = {}
        for k in range(80):
            t = k * DT_CMD
            theta = 0.0 if k == 20 else 0.05    # t = 1.0: one sample below theta_c
            predicted[round(t, 2)] = tracker.update(9, t, False, True, 1.0, theta).predicted
        self.assertFalse(predicted[0.45])
        self.assertTrue(predicted[0.5])
        self.assertTrue(predicted[0.95])
        self.assertFalse(predicted[1.0])
        self.assertFalse(predicted[1.5])
        self.assertTrue(predicted[1.55])    # window restarted at 1.05

    def test_membership_leave_lag(self):
        cfg = self.cfg
        tracker = LaneMembershipTracker(cfg)
        self.assertTrue(tracker.update(1, 0.0, True, False, None, None).member)
        self.assertTrue(tracker.update(1, 0.5, False, True, -1.0, -0.1).member)       # 0.5 s after leaving
        self.assertTrue(tracker.update(1, cfg.delta_leave_s, False, True, -1.0, -0.1).member)
        self.assertFalse(tracker.update(1, cfg.delta_leave_s + DT_CMD, False, True, -1.0, -0.1).member)


TICK = 0.1     # the NGSIM replay's command interval


def _issue(action, prev, dt, lo, hi, jerk):
    """The NGSIM replay's command limiter (``baselines._limit``) on ``command_bounds``."""
    action = float(np.clip(action, lo, hi))
    if jerk is None:
        return action
    limited = np.clip(action, prev - jerk * dt, prev + jerk * dt)
    return float(np.clip(limited, lo, hi))


def _closed_loop(policy, v0, gap0, lead_accel, T, v_set=20.0, dt=TICK):
    """One ego-lane lead (member from the start), the ideal double integrator, the
    replay's limiter.  Returns per tick (t, v, gap, path, issued command)."""
    ctrl = PCACCController(PCACCConfig.from_policy_string(policy))
    s_e, v, a, s_l, v_l = 0.0, v0, 0.0, gap0, v0
    rows = []
    for k in range(int(round(T / dt))):
        t = k * dt
        dec = ctrl.step(t, v, a, v_set, [VehicleObs(1, s_l - s_e, v_l, True)], dt)
        cmd = _issue(dec.accel, a, dt, *ctrl.command_bounds(dec.path, a, dt))
        rows.append((t, v, s_l - s_e, dec.path, cmd))
        v_next = max(0.0, v + cmd * dt)
        s_e += 0.5 * (v + v_next) * dt
        v, a = v_next, cmd
        vl_next = max(0.0, v_l + lead_accel(t) * dt)
        s_l += 0.5 * (v_l + vl_next) * dt
        v_l = vl_next
    return rows


class PhysicalCheckTests(unittest.TestCase):
    """``physcheck``: (3), (7) and the fail-safe the paper's (Table I's emergency
    limits); the reference d0 + tau v and the comfort limits the port's."""

    def setUp(self):
        self.cfg = PCACCConfig()
        self.phys = PCACCConfig.from_policy_string("acc_pcacc_physcheck")
        self.em = emergency_config(self.phys)

    def test_tokens(self):
        for policy in ("acc_pcacc", "acc_pcacc_paperdsafe_amin10_amax3_jlim10", "acc_pcacc_physcheckx",
                       "acc_pcacc_reachfixed"):
            cfg = PCACCConfig.from_policy_string(policy)
            self.assertFalse(cfg.physical_check or cfg.reach_nonneg_accel, policy)
            self.assertIs(emergency_config(cfg), cfg)
        both = PCACCConfig.from_policy_string("acc_pcacc_physcheck_reachfix")
        self.assertTrue(both.physical_check and both.reach_nonneg_accel)
        self.assertTrue(PCACCConfig.from_policy_string("acc_pcacc_reachfix").reach_nonneg_accel)
        # the comfort side keeps the fork's limits; the fail-safe gets Table I's
        self.assertEqual((self.phys.a_min, self.phys.a_max, self.phys.jerk_max), (-3.0, 2.0, 5.0))
        self.assertEqual((self.em.a_min, self.em.a_max, self.em.jerk_max), (TABLE_I_A_MIN, 2.0, TABLE_I_JERK_MAX))
        self.assertEqual((TABLE_I_A_MIN, TABLE_I_JERK_MAX), (-10.0, 10.0))
        with self.assertRaises(ValueError):
            PCACCConfig.from_policy_string("acc_pcacc_physcheck_paperdsafe")
        with self.assertRaises(ValueError):
            PCACCConfig.from_policy_string("acc_pcacc_physcheck_apmin9")   # lead braking weaker than the fail-safe

    def test_condition_seven_and_constraint_three_are_physical(self):
        ctrl = PCACCController(self.phys)
        d = safe_distance(15.0, 0.0, 13.0, self.em, TICK, u0=-1.0)
        self.assertFalse(ctrl._next_state_unsafe(d + 0.01, 15.0, 0.0, 13.0, -1.0, TICK))
        self.assertTrue(ctrl._next_state_unsafe(d - 0.01, 15.0, 0.0, 13.0, -1.0, TICK))
        # 1 m inside d0 + tau v at equal speed: unsafe for the port's check, safe for the physical one
        gap = standoff(15.0, self.cfg) - 1.0
        self.assertTrue(PCACCController(self.cfg)._next_state_unsafe(gap, 15.0, 0.0, 15.0, 0.0, TICK))
        self.assertFalse(ctrl._next_state_unsafe(gap, 15.0, 0.0, 15.0, 0.0, TICK))
        dec = ctrl.step(0.0, 15.0, 0.0, 20.0, [VehicleObs(1, gap, 15.0, True)], TICK)
        self.assertEqual(dec.path, PATH_NOMINAL)
        self.assertAlmostEqual(dec.per_vehicle[1].d_safe, standoff(15.0, self.cfg), places=9)   # the reference
        self.assertLess(dec.accel, 0.0)                    # opening the gap toward d0 + tau v
        # (3): no plan below Prop. IV.5's distance of the current state
        qp, N = ctrl._qp_keep, ctrl.N_keep
        d_now = safe_distance(15.0, 0.0, 15.0, self.em, TICK)
        _, _, X = qp.solve([gap, 0.0, 0.0], standoff(15.0, self.cfg), ds_min=np.full(N, d_now),
                           dv_lo=np.full(N, 15.0 - self.phys.v_max), dv_hi=np.full(N, 15.0),
                           a_lo=self.phys.a_min, a_hi=self.phys.a_max)
        self.assertTrue(np.all(X[0::3] >= d_now - 1.0e-6))
        # inside the physical distance: the fail-safe on Table I's profile
        dec = ctrl.step(0.1, 15.0, 0.0, 20.0, [VehicleObs(2, d_now - 1.0, 15.0, True)], TICK)
        self.assertEqual(dec.path, PATH_EMERGENCY)
        self.assertAlmostEqual(dec.accel, emergency_accel(TICK, 0.0, self.em), places=12)

    def test_follows_the_standoff_without_the_fail_safe(self):
        # A lead slowing at -2 m/s^2 for 3 s from 15 m/s: the port engages its fail-safe (its
        # check (7) is the tracked standoff); physcheck brakes on the nominal path only and settles
        # on d0 + tau v of the new speed.
        lead = lambda t: -2.0 if 2.0 <= t < 5.0 else 0.0
        port = _closed_loop("acc_pcacc", 15.0, standoff(15.0, self.cfg), lead, 25.0)
        phys = _closed_loop("acc_pcacc_physcheck", 15.0, standoff(15.0, self.cfg), lead, 25.0)
        self.assertGreater(sum(r[3] == PATH_EMERGENCY for r in port), 0)
        self.assertEqual(sum(r[3] == PATH_EMERGENCY for r in phys), 0)
        self.assertGreater(min(r[4] for r in phys), min(r[4] for r in port))
        t, v, gap = phys[-1][:3]
        self.assertAlmostEqual(gap, standoff(v, self.cfg), delta=0.05)

    def test_fail_safe_reaches_table_i_and_hands_back(self):
        # A lead braking at -7 m/s^2 for 1.5 s from 15 m/s at d0 + tau v.  The port's fail-safe
        # stops at -3 and nearly hits it; physcheck's runs down Table I's profile (its own jerk
        # passes the limiter unchanged), then the nominal path takes over below a_min and
        # releases at the comfort jerk bound instead of braking to standstill.
        lead = lambda t: -7.0 if 1.0 <= t < 2.5 else 0.0
        port = _closed_loop("acc_pcacc", 15.0, standoff(15.0, self.cfg), lead, 12.0)
        phys = _closed_loop("acc_pcacc_physcheck", 15.0, standoff(15.0, self.cfg), lead, 12.0)
        self.assertGreaterEqual(min(r[4] for r in port), self.cfg.a_min)
        self.assertLess(min(r[2] for r in port), 1.0)
        cmds = np.array([r[4] for r in phys])
        paths = [r[3] for r in phys]
        self.assertLess(cmds.min(), -6.0)
        self.assertGreaterEqual(cmds.min(), TABLE_I_A_MIN)
        self.assertGreater(min(r[2] for r in phys), 5.0)
        em = [i for i, p in enumerate(paths) if p == PATH_EMERGENCY]
        self.assertTrue(em)
        steps = np.diff(cmds[em[0]:em[-1] + 1])
        self.assertTrue(np.all(steps <= 1.0e-12) and np.all(steps >= -TABLE_I_JERK_MAX * TICK - 1.0e-12))
        # after the fail-safe: up at exactly the comfort jerk bound until a_min, no stop
        k = em[-1] + 1
        self.assertNotEqual(paths[k], PATH_EMERGENCY)
        j = k
        while cmds[j] < self.phys.a_min - 1.0e-9:
            self.assertAlmostEqual(cmds[j] - cmds[j - 1], self.phys.jerk_max * TICK, places=9)
            j += 1
        self.assertGreater(phys[-1][1], 4.0)

    def test_nominal_qp_recovers_from_below_a_min(self):
        ctrl = PCACCController(self.phys)
        v, a = 10.0, -8.0
        gap = standoff(v, self.cfg) - 2.0      # inside the port's reach at a = -8, outside Prop. IV.5's distance
        self.assertGreater(gap, safe_distance(v, a, 10.0, self.em, TICK, u0=self.phys.jerk_max))
        dec = ctrl.step(0.0, v, a, 20.0, [VehicleObs(1, gap, 10.0, True)], TICK)
        self.assertEqual(dec.path, PATH_NOMINAL)
        self.assertTrue(dec.feasible)
        self.assertAlmostEqual(dec.accel, a + self.phys.jerk_max * TICK, places=6)
        self.assertEqual(ctrl.command_bounds(PATH_NOMINAL, a, TICK)[0], a + self.phys.jerk_max * TICK)
        # the port's QP cannot start there (acceleration bound -3 unreachable at jerk 5)
        port = PCACCController(self.cfg)
        dec = port.step(0.0, v, a, 20.0, [VehicleObs(1, gap, 10.0, True)], TICK)
        self.assertEqual(dec.path, PATH_EMERGENCY)
        self.assertFalse(dec.per_vehicle[1].extra["nominal_feasible"])

    def test_command_bounds(self):
        port, phys = PCACCController(self.cfg), PCACCController(self.phys)
        for path in (PATH_CRUISE, PATH_NOMINAL, PATH_EMERGENCY, PATH_RECAPTURE, PATH_ICS):
            self.assertEqual(port.command_bounds(path, -1.0, TICK), (-3.0, 2.0, 5.0))
            self.assertEqual(port.command_bounds(path, -3.0, TICK), (-3.0, 2.0, 5.0))
            if path != PATH_EMERGENCY:
                self.assertEqual(phys.command_bounds(path, -1.0, TICK), (-3.0, 2.0, 5.0))
                self.assertAlmostEqual(phys.command_bounds(path, -6.0, TICK)[0], -5.5, places=12)
        self.assertEqual(port.command_bounds(PATH_BOUNDED, -1.0, TICK), (-3.0, 2.0, None))
        self.assertEqual(phys.command_bounds(PATH_BOUNDED, -6.0, TICK), (-3.0, 2.0, None))
        self.assertEqual(phys.command_bounds(PATH_EMERGENCY, -1.0, TICK), (-10.0, 2.0, 10.0))

    def test_cut_in_recapture_and_ics_stay_the_ports(self):
        # Entering between Prop. IV.5's distance and d0 + tau v is a cut-in (Def. II.3 on the standoff).
        v = 12.0
        d_phys, d_fork = safe_distance(v, 0.0, v, self.em, TICK), standoff(v, self.cfg)
        self.assertLess(d_phys, d_fork)
        gap = 0.5 * (d_phys + d_fork)
        outs = []
        for cfg in (self.cfg, self.phys):
            ctrl = PCACCController(cfg)
            ctrl.step(0.0, v, 0.0, 20.0, [VehicleObs(8, gap, v, False)], TICK)
            dec = ctrl.step(TICK, v, 0.0, 20.0, [VehicleObs(8, gap, v, True)], TICK)
            self.assertEqual(dec.path, PATH_RECAPTURE)
            self.assertAlmostEqual(dec.per_vehicle[8].d_safe, d_fork, places=9)
            outs.append(dec.accel)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(is_inevitable_collision(2.0, 17.0, 0.0, 5.0, self.cfg),
                         is_inevitable_collision(2.0, 17.0, 0.0, 5.0, self.phys))


class ReachFixTests(unittest.TestCase):
    """``reachfix``: rule 2's one-step advance from max(a, 0)."""

    def setUp(self):
        self.cfg = PCACCConfig()
        self.fix = PCACCConfig.from_policy_string("acc_pcacc_reachfix")

    def test_reach_does_not_shrink_while_braking(self):
        v = 8.0
        lead = [VehicleObs(1, standoff(v, self.cfg), v, True)]
        kept = lambda cfg, a: [veh.vid for veh in relevant_vehicles(lead, v, a, cfg, TICK)]
        self.assertEqual(kept(self.cfg, -3.0), [])        # the port drops the lead at d0 + tau v while braking
        self.assertEqual(kept(self.cfg, 0.0), [1])
        for a in (-3.0, -1.0, 0.0):
            self.assertEqual(kept(self.fix, a), [1])
        # the reach at a < 0 is the port's at a = 0; at a > 0 the port's
        for a, a_port in ((-3.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (1.0, 1.0)):
            for gap in np.arange(0.5, 60.0, 0.5):
                far = [VehicleObs(1, float(gap), v, True)]
                self.assertEqual([x.vid for x in relevant_vehicles(far, v, a, self.fix, TICK)],
                                 [x.vid for x in relevant_vehicles(far, v, a_port, self.cfg, TICK)])

    def test_stopping_distance_stays_the_comfort_profiles_with_physcheck(self):
        both = PCACCConfig.from_policy_string("acc_pcacc_physcheck_reachfix")
        for v in (3.0, 6.0, 12.0):
            for gap in np.arange(0.5, 40.0, 0.25):
                veh = [VehicleObs(1, float(gap), v, True)]
                self.assertEqual(len(relevant_vehicles(veh, v, -2.0, both, TICK)),
                                 len(relevant_vehicles(veh, v, -2.0, self.fix, TICK)))
        # ... which contains the reach with Table I's stopping distance
        em = emergency_config(both)
        for v in (3.0, 6.0, 12.0, 20.0):
            self.assertGreater(ego_brake_distance(v, 0.5, self.cfg), ego_brake_distance(v, 0.5, em))

    def test_no_cruise_fail_safe_flicker_in_stop_and_go(self):
        # A lead at 6 m/s slowing and speeding up by 2 m/s: the port's reach drops the lead while
        # the ego brakes, which alternates cruise and its fail-safe; reachfix keeps it.
        lead = lambda t: -1.0 if 2.0 <= t < 4.0 else (1.0 if 8.0 <= t < 10.0 else 0.0)
        port = _closed_loop("acc_pcacc", 6.0, 10.0, lead, 30.0)
        fix = _closed_loop("acc_pcacc_reachfix", 6.0, 10.0, lead, 30.0)
        flips = lambda rows: sum(rows[i][3] != rows[i - 1][3] for i in range(1, len(rows)))
        self.assertGreater(flips(port), 10)
        self.assertGreater(sum(r[3] == PATH_CRUISE for r in port), 0)
        self.assertLessEqual(flips(fix), 2)
        self.assertEqual(sum(r[3] == PATH_CRUISE for r in fix), 0)


if __name__ == "__main__":
    unittest.main()
