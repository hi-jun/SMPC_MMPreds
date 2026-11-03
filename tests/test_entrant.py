"""``entrant<b>`` (final-check item 12, 2026-11-04): the safety layer covers close entrants (vehicles that joined the
ego's path inside its safe distance, Def. II.3's cut-ins) under the braking bound a_c = -b, with a trajectory check
in place of the ICS test, and with ``safebranch`` they lead the fail-safe branch too."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    FAILSAFE_BRANCH_MARGIN_M,
    FailsafeBranch,
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    failsafe_branch_lead,
    failsafe_entrant_terminal,
    failsafe_recovery_floor,
    original_nair_acc_config,
)
from utils.pcacc import VehicleObs, _time_to_standstill  # noqa: E402
from utils.safety_layer import (  # noqa: E402
    FAILSAFE_DECEL,
    FAILSAFE_JERK,
    LEAD_DECEL,
    PATH_ENTRANT,
    PATH_FAILSAFE,
    PATH_ICS,
    PATH_NOMINAL,
    SafetyLayer,
    _min_gap_until_the_ego_stops,
    entrant_margin,
    held_command_safe_distance,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402

HORIZON = 15
DT = 0.2
TICK = 0.1
L = 4.5
A_C = -2.0


def plant(s, v, a, dt):
    v_next = max(0.0, v + a * dt)
    return s + 0.5 * (v + v_next) * dt, v_next


def brute_min_gap(gap, v_e, a_e, j_e, t_jerk, a_hold, v_l, a_l, t_max=math.inf, step=2.0e-4):
    """``_min_gap_until_the_ego_stops`` by fine-step simulation (continuous time, both clipped at standstill)."""
    s_e = s_l = 0.0
    g_min, t = gap, 0.0
    if j_e == 0.0:
        t_jerk = math.inf
    while t < min(t_max, 60.0):
        if v_e <= 0.0 and a_e <= 0.0 and (t >= t_jerk or j_e == 0.0):
            break
        jerk = j_e if t < t_jerk else 0.0
        v_e_next = max(0.0, v_e + a_e * step + 0.5 * jerk * step ** 2)
        s_e += 0.5 * (v_e + v_e_next) * step
        v_e = v_e_next
        a_e = a_e + jerk * step
        if t + step >= t_jerk > t:
            a_e, jerk = a_hold, 0.0
        v_l_next = max(0.0, v_l + a_l * step)
        s_l += 0.5 * (v_l + v_l_next) * step
        v_l = v_l_next
        t += step
        g_min = min(g_min, gap + s_l - s_e)
    return g_min


class MarginTests(unittest.TestCase):

    def test_min_gap_matches_a_fine_simulation(self):
        rng = np.random.default_rng(20261104)
        for trial in range(300):
            gap = rng.uniform(0.0, 30.0)
            v_e, v_l = rng.uniform(0.0, 25.0), rng.uniform(0.0, 25.0)
            a_e = rng.uniform(-8.0, 2.0)
            a_l = rng.uniform(-10.5, -0.5)
            if trial % 3 == 0:          # the held command (no jerk) over an interval
                exact = _min_gap_until_the_ego_stops(gap, v_e, a_e, 0.0, 0.0, a_e, v_l, a_l, t_max=0.1)
                brute = brute_min_gap(gap, v_e, a_e, 0.0, 0.0, a_e, v_l, a_l, t_max=0.1)
            else:                       # the fail-safe from (v_e, a_e)
                t_jerk = (a_e - FAILSAFE_DECEL) / FAILSAFE_JERK
                exact = _min_gap_until_the_ego_stops(gap, v_e, a_e, -FAILSAFE_JERK, t_jerk, FAILSAFE_DECEL, v_l, a_l)
                brute = brute_min_gap(gap, v_e, a_e, -FAILSAFE_JERK, t_jerk, FAILSAFE_DECEL, v_l, a_l)
            self.assertAlmostEqual(exact, brute, delta=5.0e-3, msg="trial %d" % trial)
            self.assertLessEqual(exact, gap + 1e-9)

    def test_entrant_margin_against_a_slower_and_a_faster_entrant(self):
        cfg = SafetyLayer(2.0, entrant_decel=A_C).cfg
        # equal speeds: the ego's fail-safe (-8) stops far inside the entrant's stopping distance at -2
        self.assertGreater(entrant_margin(1.0, 15.0, 0.0, 15.0, A_C, cfg, TICK), 0.0)
        # 10 m/s faster than the entrant 3 m ahead: unavoidable
        self.assertLess(entrant_margin(3.0, 15.0, 0.0, 5.0, A_C, cfg, TICK), 0.0)
        # the margin falls with the held command and with the gap
        m = [entrant_margin(6.0, 15.0, a, 8.0, A_C, cfg, TICK) for a in (-3.0, 0.0, 2.0)]
        self.assertTrue(m[0] > m[1] > m[2])
        self.assertGreater(entrant_margin(8.0, 15.0, 0.0, 8.0, A_C, cfg, TICK), entrant_margin(6.0, 15.0, 0.0, 8.0, A_C, cfg, TICK))
        # stopped behind a stopped entrant: only the stopping tick's term
        self.assertAlmostEqual(entrant_margin(0.05, 0.0, 0.0, 0.0, A_C, cfg, TICK), 0.05 - 8.0 * TICK ** 2 / 8.0, places=9)


class LayerTests(unittest.TestCase):

    def test_without_the_bound_the_layer_is_unchanged(self):
        # the ICS test of 2026-11-02 for the cut-in set (tests/test_safety_layer.py covers it; here the flag)
        layer = SafetyLayer(2.0)
        self.assertIsNone(layer.entrant_decel)
        layer.step(0.0, 15.0, 0.0, 0.5, [VehicleObs(7, 2.0, 5.0, False)], TICK)
        dec = layer.step(0.1, 15.0, 0.5, 0.5, [VehicleObs(7, 1.0, 5.0, True)], TICK)
        self.assertEqual(dec.path, PATH_ICS)
        self.assertTrue(math.isnan(dec.entrant_margin))
        with self.assertRaises(ValueError):
            SafetyLayer(2.0, entrant_decel=2.0)

    def test_entrant_check_replaces_the_ics_test(self):
        layer = SafetyLayer(2.0, entrant_decel=A_C)
        layer.step(0.0, 15.0, 0.0, 0.5, [VehicleObs(7, 2.0, 5.0, False)], TICK)
        dec = layer.step(0.1, 15.0, 0.5, 0.5, [VehicleObs(7, 1.0, 5.0, True)], TICK)   # 10 m/s slower, 1 m ahead
        self.assertEqual(dec.cutin_vids, [7])
        self.assertEqual(dec.new_entrant_vids, [7])
        self.assertEqual(dec.unavoidable_vids, [7])                # unavoidable at entry: the uncovered case
        self.assertEqual(dec.new_unavoidable_vids, [7])
        self.assertEqual(dec.path, PATH_ENTRANT)
        self.assertAlmostEqual(dec.accel, 0.5 - FAILSAFE_JERK * TICK)   # the fail-safe, as the ICS braking was
        self.assertLess(dec.entrant_margin, 0.0)

    def test_avoidable_entrant_passes_a_mild_nominal_and_fails_an_acceleration(self):
        layer = SafetyLayer(2.0, entrant_decel=A_C)
        layer.step(0.0, 15.0, 0.0, 0.0, [VehicleObs(7, 6.0, 12.0, False)], TICK)
        dec = layer.step(0.1, 15.0, 0.0, 0.0, [VehicleObs(7, 6.0, 12.0, True)], TICK)  # 3 m/s slower, 6 m: entrant
        self.assertEqual((dec.cutin_vids, dec.checked_vids), ([7], []))
        self.assertEqual(dec.unavoidable_vids, [])
        self.assertEqual(dec.path, PATH_NOMINAL)
        self.assertGreater(dec.entrant_margin, 0.0)
        # from a closer state an accelerating nominal fails while the fail-safe still passes
        dec = layer.step(0.2, 15.0, 0.0, 2.0, [VehicleObs(7, 2.5, 12.0, True)], TICK)
        self.assertEqual(dec.path, PATH_ENTRANT)
        self.assertEqual(dec.unavoidable_vids, [])
        self.assertAlmostEqual(dec.accel, -FAILSAFE_JERK * TICK)

    def _entrant_loop(self, layer, v0, vl0, gap0, nominal, entrant_accel, steps, dt=TICK):
        """An ego at ``v0`` with a vehicle beside it that becomes a path member at tick 1, ``gap0`` ahead at
        ``vl0``, and then brakes by ``entrant_accel(k)``. Returns (min gap, paths, the entry decision)."""
        s_e, v_e, s_l, v_l, prev = 0.0, v0, gap0, vl0, 0.0
        gap_min, entry = gap0, None
        paths = {PATH_NOMINAL: 0, PATH_FAILSAFE: 0, PATH_ICS: 0, PATH_ENTRANT: 0}
        for k in range(steps):
            gap = s_l - s_e
            dec = layer.step(k * dt, v_e, prev, nominal(k, v_e, gap), [VehicleObs(1, gap, v_l, k >= 1)], dt)
            if k == 1:
                entry = dec
            paths[dec.path] += 1
            s_e, v_e = plant(s_e, v_e, dec.accel, dt)
            s_l, v_l = plant(s_l, v_l, entrant_accel(k), dt)
            prev = dec.accel
            gap_min = min(gap_min, s_l - s_e)
        return gap_min, paths, entry

    def test_adversarial_nominal_against_an_entrant_braking_at_the_bound(self):
        # an entrant 6 m ahead at the ego's speed (inside the 7.8 m safe distance) that brakes at -2 for good,
        # the nominal always accelerating: covered, no contact; without the bound the ICS test acts too late
        for v in (10.0, 15.0, 20.0):
            cfg = SafetyLayer(2.0).cfg
            gap0 = 0.8 * held_command_safe_distance(v, 0.0, v, cfg, TICK)
            gap_new, paths_new, entry = self._entrant_loop(
                SafetyLayer(2.0, entrant_decel=A_C), v, v, gap0, lambda k, ve, g: 2.0, lambda k: A_C, int(12 / TICK))
            self.assertEqual(entry.cutin_vids, [1])
            self.assertEqual(entry.unavoidable_vids, [])
            self.assertGreaterEqual(gap_new, 0.0, "v %g" % v)
            self.assertGreater(paths_new[PATH_ENTRANT], 0)
            gap_old, paths_old, _ = self._entrant_loop(
                SafetyLayer(2.0), v, v, gap0, lambda k, ve, g: 2.0, lambda k: A_C, int(12 / TICK))
            self.assertLess(gap_old, gap_new)
            self.assertGreater(paths_old[PATH_ICS], 0)

    def test_random_entrants_within_the_bound_never_collide_when_avoidable(self):
        rng = np.random.default_rng(1104)
        n_avoidable = n_unavoidable = 0
        for trial in range(300):
            v0, vl0 = rng.uniform(0.0, 25.0), rng.uniform(0.0, 25.0)
            cfg = SafetyLayer(2.0).cfg
            d = held_command_safe_distance(v0, 0.0, vl0, cfg, TICK)
            gap0 = rng.uniform(0.3, max(0.6, d))            # inside the safe distance where that is positive
            acc_seq = rng.uniform(A_C, 2.0, size=8)        # the entrant never brakes harder than a_c ...
            nom_seq = rng.uniform(-3.0, 2.0, size=8)
            steps = int(15 / TICK)
            gap_min, paths, entry = self._entrant_loop(
                SafetyLayer(2.0, entrant_decel=A_C), v0, vl0, gap0,
                lambda k, ve, g: float(nom_seq[min(7, k * 8 // steps)]),
                lambda k: float(acc_seq[min(7, k * 8 // steps)]), steps)
            if entry.cutin_vids and not entry.unavoidable_vids:
                n_avoidable += 1
                self.assertGreaterEqual(gap_min, -1.0e-9, "trial %d" % trial)   # ... so it is covered
            elif entry.unavoidable_vids:
                n_unavoidable += 1
                self.assertNotEqual(entry.path, PATH_NOMINAL)                  # mitigation at once
            else:
                self.assertGreaterEqual(gap_min, -1.0e-9, "trial %d (checked)" % trial)
        self.assertGreater(n_avoidable, 50)
        self.assertGreater(n_unavoidable, 10)

    def test_entrant_becomes_a_checked_member_at_the_safe_distance(self):
        # a faster entrant pulls away past the safe distance (the nominal holds the speed), is then covered
        # under -10.5 and brakes at it
        layer = SafetyLayer(2.0, entrant_decel=A_C)
        s_e, v_e, s_l, v_l, prev = 0.0, 15.0, 4.0, 18.0, 0.0
        became_checked = False
        gap_min = 4.0
        for k in range(200):
            gap = s_l - s_e
            dec = layer.step(k * TICK, v_e, prev, 0.0, [VehicleObs(1, gap, v_l, k >= 1)], TICK)
            if k >= 1 and dec.checked_vids == [1]:
                became_checked = True
            lead_a = LEAD_DECEL if (became_checked and k > 60) else 0.0
            s_e, v_e = plant(s_e, v_e, dec.accel, TICK)
            s_l, v_l = plant(s_l, v_l, lead_a, TICK)
            prev = dec.accel
            gap_min = min(gap_min, s_l - s_e)
        self.assertTrue(became_checked)
        self.assertGreaterEqual(gap_min, -1.0e-9)


def design_config(entrant=A_C, **overrides):
    config = original_nair_acc_config(
        horizon=HORIZON, dt=DT, desired_speed=20.0, num_modes=2, controller_variant=VARIANT_FIXED_RISK)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move = 1000.0
    config.r_jerk = 40.0
    config.r_a = 0.0
    config.standoff_shortfall_weight = 100.0
    config.jerk_input = True
    config.slack_l1_weight = 1.0e6
    config.failsafe_branch = True
    config.failsafe_recovery = True
    config.failsafe_entrant_decel = entrant
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def far_prediction(s_ego, lead_v=20.0):
    steps = HORIZON + 1
    means = np.zeros((2, steps, 2))
    means[0, :, 0] = s_ego + 80.0 + lead_v * DT * np.arange(steps)
    means[1, :, 0] = s_ego + 80.0 + lead_v * DT * np.arange(steps)
    means[:, :, 1] = lead_v
    confidence = np.full((2, steps), np.nan)
    confidence[1] = 0.4
    return MultimodalLeadPrediction(
        means=means, probabilities=np.array([0.6, 0.4]),
        covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)), active_mask=np.ones((2, steps), dtype=bool),
        chance_confidence=confidence, lead_mode_lk=np.array([[True] * steps, [False] * steps]),
        lead_vehicle_ids=np.array([[10] * steps, [11] * steps]), clearance_scale=np.ones((2, steps)),
        mode_names=["10:lk", "11:cutin"])


class BranchTests(unittest.TestCase):

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            design_config(failsafe_branch=False)
        with self.assertRaises(ValueError):
            design_config(entrant=2.0)

    def test_lead_per_member_bound_and_the_terminal_values(self):
        config = design_config()
        branch = FailsafeBranch.from_members([VehicleObs(1, 30.0, 20.0, True)], [VehicleObs(2, 6.0, 15.0, True)], A_C)
        self.assertEqual((branch.n_entrants, branch.gaps.size), (1, 2))
        np.testing.assert_allclose(branch.decels, [LEAD_DECEL, A_C])
        lead, stop = failsafe_branch_lead(branch, config)
        t = DT * np.arange(HORIZON + 1)
        entrant = 6.0 + L + np.where(t < 15.0 / 2.0, 15.0 * t - 1.0 * t ** 2, 15.0 ** 2 / 4.0)
        member = 30.0 + L + np.where(t < 20.0 / 10.5, 20.0 * t - 0.5 * 10.5 * t ** 2, 20.0 ** 2 / 21.0)
        np.testing.assert_allclose(lead, np.minimum(entrant, member), rtol=1e-12)
        self.assertAlmostEqual(stop, min(6.0 + L + 15.0 ** 2 / 4.0, 30.0 + L + 400.0 / 21.0), places=9)
        v_term, a_term = failsafe_entrant_terminal(branch, config)
        self.assertAlmostEqual(v_term, 15.0 - 2.0 * HORIZON * DT)
        self.assertEqual(a_term, A_C)
        # without an entrant the rows are inactive; without the bound entrants are not taken
        self.assertEqual(failsafe_entrant_terminal(FailsafeBranch.from_members([VehicleObs(1, 30.0, 20.0, True)]), config),
                         (1.0e6, config.a_max))
        plain = FailsafeBranch.from_members([VehicleObs(1, 30.0, 20.0, True)], [VehicleObs(2, 6.0, 15.0, True)], None)
        self.assertEqual((plain.n_entrants, plain.gaps.size), (0, 1))

    def test_branch_with_an_entrant_ends_no_faster_than_it(self):
        config = design_config()
        controller = NairACCSMPC(config)
        branch = FailsafeBranch.from_members([], [VehicleObs(2, 6.0, 20.0, True)], A_C)
        solution = controller.solve(np.array([0.0, 20.0]), far_prediction(0.0), previous_accel_cmd=0.0,
                                    command_dt=TICK, ego_accel=0.0, failsafe_branch=branch)
        self.assertNotEqual(solution.solve_path, "fallback")
        self.assertEqual((solution.failsafe_present, solution.failsafe_members, solution.failsafe_entrants), (True, 1, 1))
        self.assertLess(solution.failsafe_slack_max, 1e-6)                 # a hard solution exists ...
        self.assertGreaterEqual(solution.failsafe_entrant_margin, -1e-6)   # ... with both terminal rows
        fs_x = solution.failsafe_x_nominal
        self.assertLessEqual(fs_x[HORIZON, 1], 20.0 - 2.0 * HORIZON * DT + 1e-6)
        self.assertLessEqual(fs_x[HORIZON, 2], A_C + 1e-6)
        self.assertGreaterEqual(solution.failsafe_margin_min, -1e-6)
        # the same solve without an entrant: the terminal rows do not bind and the margin is NaN
        plain = controller.solve(np.array([0.0, 20.0]), far_prediction(0.0), previous_accel_cmd=0.0,
                                 command_dt=TICK, ego_accel=0.0, failsafe_branch=FailsafeBranch.from_members([]))
        self.assertTrue(math.isnan(plain.failsafe_entrant_margin))
        self.assertEqual(plain.failsafe_entrants, 0)

    def test_entrant_rows_only_while_the_branch_holds_an_entrant(self):
        # without the bound, and with it while no entrant is in the branch, the QP has no entrant rows and the
        # solution is the paper design's bit for bit; with an entrant a second cached problem carries the rows
        off = NairACCSMPC(design_config(entrant=None))
        on = NairACCSMPC(design_config())
        member = [VehicleObs(1, 30.0, 20.0, True)]
        for branch in (FailsafeBranch.from_members([]), FailsafeBranch.from_members(member)):
            a = off.solve(np.array([0.0, 20.0]), far_prediction(0.0), previous_accel_cmd=0.0, command_dt=TICK,
                          ego_accel=0.0, failsafe_branch=branch)
            b = on.solve(np.array([0.0, 20.0]), far_prediction(0.0), previous_accel_cmd=0.0, command_dt=TICK,
                         ego_accel=0.0, failsafe_branch=FailsafeBranch.from_members(branch.gaps.size and member or [], [], A_C))
            self.assertEqual(a.action, b.action)
            np.testing.assert_array_equal(a.u_nominal, b.u_nominal)
        self.assertEqual(len(on._feedback_scalar_chance_qp_cache), 1)
        problem = next(iter(on._feedback_scalar_chance_qp_cache.values()))
        self.assertIsNone(problem.fs_term_v_param)
        on.solve(np.array([0.0, 20.0]), far_prediction(0.0), previous_accel_cmd=0.0, command_dt=TICK, ego_accel=0.0,
                 failsafe_branch=FailsafeBranch.from_members(member, [VehicleObs(2, 6.0, 15.0, True)], A_C))
        self.assertEqual(len(on._feedback_scalar_chance_qp_cache), 2)
        self.assertEqual(sum(p.fs_term_v_param is not None for p in on._feedback_scalar_chance_qp_cache.values()), 1)

    def test_closed_loop_with_an_entrant_keeps_the_gap(self):
        # ego at 20 m/s; a vehicle enters the path 8 m ahead at 17 m/s at tick 5 and brakes at -2 from tick 30;
        # the MPC's hypotheses never see it; with the bound the branch plans the contingency and the layer
        # checks it: no contact, no fallback, and the fail-safe is issued no more often than without the branch
        def run(with_bound):
            config = design_config(entrant=A_C if with_bound else None)
            controller = NairACCSMPC(config)
            layer = SafetyLayer(config.a_max, entrant_decel=A_C)
            s_e, v_e, s_l, v_l, prev = 0.0, 20.0, 8.0, 17.0, 0.0
            gap_min, acted, fallbacks = 8.0, 0, 0
            for k in range(120):
                t = k * TICK
                obs = [VehicleObs(1, s_l - s_e, v_l, k >= 5)]
                mpc_prev = prev if prev >= config.a_min else max(prev, failsafe_recovery_floor(v_e, config, TICK))
                checked, cutins = layer.classify(t, v_e, prev, obs, TICK)
                branch = FailsafeBranch.from_members(checked, cutins, config.failsafe_entrant_decel)
                solution = controller.solve(np.array([s_e, v_e]), far_prediction(s_e), previous_accel_cmd=mpc_prev,
                                            command_dt=TICK, ego_accel=mpc_prev, failsafe_branch=branch)
                fallbacks += solution.solve_path == "fallback"
                decision = layer.decide(t, v_e, prev, solution.action, checked, cutins, TICK)
                acted += decision.path != PATH_NOMINAL
                s_e, v_e = plant(s_e, v_e, decision.accel, TICK)
                s_l, v_l = plant(s_l, v_l, A_C if k >= 30 else 0.0, TICK)
                prev = decision.accel
                gap_min = min(gap_min, s_l - s_e)
            return gap_min, acted, fallbacks
        gap_with, acted_with, fb_with = run(True)
        gap_without, acted_without, fb_without = run(False)
        self.assertEqual((fb_with, fb_without), (0, 0))
        self.assertGreaterEqual(gap_with, 0.0)
        self.assertGreaterEqual(gap_without, 0.0)
        self.assertLessEqual(acted_with, acted_without)


class TokenTests(unittest.TestCase):

    def test_whole_token(self):
        parse = ACCNairSMPCAgent._parse_entrant_decel
        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        self.assertEqual(parse(base + "_safelayer_safebranch_entrant2"), -2.0)
        self.assertEqual(parse(base + "_entrant3.5_safelayer"), -3.5)
        self.assertIsNone(parse(base + "_safelayer_safebranch"))
        self.assertIsNone(parse(base + "_entrants2"))
        self.assertIsNone(parse(base + "_entrant"))

    def test_needs_safelayer(self):
        class _Vehicle:
            def get_world(self):
                return None

        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        with self.assertRaisesRegex(ValueError, "safelayer"):
            ACCNairSMPCAgent(_Vehicle(), None, smpc_config=base + "_jinput_ra0_fullfoot_l1slack1e6_entrant2")


if __name__ == "__main__":
    unittest.main()
