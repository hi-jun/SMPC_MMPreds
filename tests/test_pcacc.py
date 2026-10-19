"""Unit tests for the provably-correct ACC baseline (utils/pcacc.py), no CARLA."""
import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.pcacc import (  # noqa: E402
    PATH_CRUISE,
    PATH_EMERGENCY,
    PATH_ICS,
    PATH_NOMINAL,
    PATH_RECAPTURE,
    LaneMembershipTracker,
    PCACCConfig,
    PCACCController,
    VehicleObs,
    ego_brake_distance,
    emergency_accel,
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


if __name__ == "__main__":
    unittest.main()
