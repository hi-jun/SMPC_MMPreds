"""Unified limits (2026-11-11, maintainer: every controller's acceleration and jerk bounds by ISO 15622 in the NGSIM
replay).  The options, all off by default:

* D-opt (``utils/acc_nair_smpc.py``, ``policies/acc_nair_smpc_agent.py``): ``amin<x>`` / ``amax<x>`` (the comfort
  acceleration bounds), ``jlimup<x>`` (the rising side's jerk bound apart from ``jlim<x>``), ``fsdecel<x>`` /
  ``fsjerk<x>`` (the safety layer's fail-safe, ``utils/safety_layer.py``);
* ISA-MPC (``policies/isampc_agent.py``): ``jerk_limit_up`` of the planner (``jlimup<x>`` through the agent);
* PC-ACC (``utils/pcacc.py``): ``jlimup<x>``, ``emamin<x>`` / ``emjlim<x>`` (the emergency limits apart from the
  comfort ones, also with ``paperdsafe``) and ``emics``.

Without them every problem is the one of before, number for number: here by explicit settings equal to the
defaults; against the code of 23b2866 by the separate check of the branch (notes folder)."""
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    FailsafeBranch,
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    _failsafe_limits,
    _jerk_limit_up,
    failsafe_recovery_command,
    failsafe_recovery_envelope,
    failsafe_recovery_floor,
    failsafe_terminal_coefficients,
    limit_accel_command,
    original_nair_acc_config,
)
from utils.pcacc import (  # noqa: E402
    PATH_BOUNDED,
    PATH_CRUISE,
    PATH_EMERGENCY,
    PATH_ICS,
    PATH_NOMINAL,
    PATH_RECAPTURE,
    TABLE_I_A_MIN,
    TABLE_I_JERK_MAX,
    PCACCConfig,
    PCACCController,
    VehicleObs,
    _BatchQP,
    ego_brake_distance,
    emergency_config,
    safe_distance,
)
from utils.safety_layer import (  # noqa: E402
    FAILSAFE_DECEL,
    FAILSAFE_JERK,
    LEAD_DECEL,
    PATH_FAILSAFE,
    SafetyLayer,
    held_command_safe_distance,
    safety_layer_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402
from policies.isampc_agent import ISAMPCPlanner  # noqa: E402

HORIZON = 15
DT = 0.2
TICK = 0.1
EGO = np.array([0.0, 12.0])
NP_W3 = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt3lkr_trajcutin_ctx0_latgate"
         "_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch_mergefirst"
         "_lkrsigma_lkhold_entrant2_bcts_occin0.1_nolkentry_floorrisk0.1_floorhyp_nopricesigma_lkonesided0.05")


def design_config(jerk_input=True, **overrides):
    """The design's QP (dopt100, r_jerk 40, r_a 0, l1slack1e6, jinput by default) with ``overrides``."""
    config = original_nair_acc_config(
        horizon=HORIZON, dt=DT, desired_speed=20.0, num_modes=2, controller_variant=VARIANT_FIXED_RISK)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move = 1000.0
    config.r_jerk = 40.0
    config.r_a = 0.0
    config.standoff_shortfall_weight = 100.0
    config.jerk_input = jerk_input
    config.slack_l1_weight = 1.0e6
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def prediction(s_ego=0.0, gap0=30.0, lead_v=12.0, p_cutin=0.4):
    """Mode 0 lane keeping without a lead, mode 1 a cut-in that leads at every step (as tests/test_jerk_input.py)."""
    steps = HORIZON + 1
    means = np.zeros((2, steps, 2))
    means[:, :, 0] = s_ego + gap0 + lead_v * DT * np.arange(steps)
    means[:, :, 1] = lead_v
    active = np.zeros((2, steps), dtype=bool)
    active[1] = True
    confidence = np.full((2, steps), np.nan)
    confidence[1] = p_cutin
    return MultimodalLeadPrediction(
        means=means, probabilities=np.array([1.0 - p_cutin, p_cutin]),
        covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)), active_mask=active,
        chance_confidence=confidence, lead_mode_lk=np.zeros((2, steps), dtype=bool),
        lead_vehicle_ids=np.where(active, 10, -1), clearance_scale=np.ones((2, steps)),
        mode_names=["10:lk", "10:cutin"])


def solve(config, ego=EGO, prev=0.0, ego_accel=None, pred=None, branch=None):
    controller = NairACCSMPC(config)
    solution = controller.solve(np.asarray(ego, dtype=float), prediction() if pred is None else pred,
                                previous_accel_cmd=prev, command_dt=TICK,
                                ego_accel=prev if ego_accel is None else ego_accel, failsafe_branch=branch)
    return controller, solution


def plant(s, v, a, dt):
    """The replay's ideal plant: the command held over the tick, speed clipped at 0, trapezoidal position."""
    v_next = max(0.0, v + a * dt)
    return s + 0.5 * (v + v_next) * dt, v_next


class TokenTests(unittest.TestCase):

    def test_whole_tokens_and_magnitudes(self):
        parse = ACCNairSMPCAgent._parse_limit_tokens
        self.assertEqual(parse(NP_W3), {})
        self.assertEqual(parse(NP_W3 + "_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10"),
                         {"a_min": -3.5, "a_max": 2.0, "jerk_limit_up": 5.0, "failsafe_decel": -8.0,
                          "failsafe_jerk": 10.0})
        self.assertEqual(parse("acc_nair_smpc_jlim2.5"), {})
        for policy in ("acc_nair_smpc_xamin3", "acc_nair_smpc_amin3x", "acc_nair_smpc_amaxx2", "acc_nair_smpc_jlimupx",
                       "acc_nair_smpc_safelayer_fsdecelx8"):
            self.assertEqual(parse(policy), {}, policy)
        # jlim<x> keeps its own parser, which jlimup<x> does not trigger
        self.assertIsNone(ACCNairSMPCAgent._parse_jerk_limit("acc_nair_smpc_jlimup5"))
        self.assertEqual(ACCNairSMPCAgent._parse_jerk_limit("acc_nair_smpc_jlimup5_jlim2.5"), 2.5)
        # the fail-safe tokens need safelayer
        for policy in ("acc_nair_smpc_fsdecel8", "acc_nair_smpc_fsjerk10"):
            with self.assertRaises(ValueError):
                parse(policy)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            design_config(jerk_limit=None, jerk_limit_up=5.0)
        with self.assertRaises(ValueError):
            design_config(jerk_limit_up=0.0)
        with self.assertRaises(ValueError):
            design_config(a_min=-3.5, failsafe_decel=-3.0)            # the fail-safe must brake at least as hard
        with self.assertRaises(ValueError):
            design_config(failsafe_jerk=0.0)
        with self.assertRaises(ValueError):                          # the terminal row's bound needs a_max <= |a_em|/2
            design_config(failsafe_branch=True, failsafe_recovery=True, a_max=4.5)
        design_config(failsafe_branch=True, failsafe_recovery=True, a_max=4.0)
        design_config(failsafe_branch=True, failsafe_recovery=True, a_max=2.0, failsafe_decel=-4.0)
        with self.assertRaises(ValueError):
            design_config(failsafe_branch=True, failsafe_recovery=True, a_max=2.0, failsafe_decel=-3.5)

    def test_failsafe_limits_follow_the_config(self):
        self.assertEqual(_failsafe_limits(), (FAILSAFE_DECEL, FAILSAFE_JERK, LEAD_DECEL))
        self.assertEqual(_failsafe_limits(design_config()), (FAILSAFE_DECEL, FAILSAFE_JERK, LEAD_DECEL))
        config = design_config(failsafe_decel=-6.0, failsafe_jerk=10.0)
        self.assertEqual(_failsafe_limits(config), (-6.0, 10.0, LEAD_DECEL))
        self.assertEqual(_jerk_limit_up(design_config()), 5.0)
        self.assertEqual(_jerk_limit_up(design_config(jerk_limit_up=2.0)), 2.0)
        self.assertEqual(_jerk_limit_up(design_config(jerk_limit=None)), math.inf)


class DoptTests(unittest.TestCase):

    def test_explicit_defaults_solve_the_same_problem(self):
        # jerk_limit_up equal to jerk_limit, and the fail-safe's module values given explicitly, are the defaults'
        # problems number for number (with the fail-safe branch, the recovery and a start below a_min)
        for jerk_input in (True, False):
            base = dict(jerk_input=jerk_input, failsafe_branch=True, failsafe_recovery=True)
            branch = FailsafeBranch.from_members([VehicleObs(1, 18.0, 12.0, True)])
            _, plain = solve(design_config(**base), prev=-6.0, branch=branch)
            _, same = solve(design_config(jerk_limit_up=5.0, failsafe_decel=FAILSAFE_DECEL,
                                          failsafe_jerk=FAILSAFE_JERK, **base), prev=-6.0, branch=branch)
            self.assertNotEqual(plain.solve_path, "fallback")
            self.assertEqual(plain.action, same.action)
            np.testing.assert_array_equal(plain.u_nominal, same.u_nominal)
            np.testing.assert_array_equal(plain.failsafe_u_nominal, same.failsafe_u_nominal)

    def test_jinput_rising_side_has_its_own_bound(self):
        # releasing from -2 m/s^2 with the lead far and a cheap jerk (r_jerk 0.1): the plan raises the acceleration
        # as fast as allowed
        far = prediction(gap0=80.0, lead_v=20.0)
        _, sym = solve(design_config(jerk_limit=2.5, r_jerk=0.1), ego=[0.0, 10.0], prev=-2.0, pred=far)
        _, up = solve(design_config(jerk_limit=2.5, jerk_limit_up=5.0, r_jerk=0.1), ego=[0.0, 10.0], prev=-2.0,
                      pred=far)
        for solution in (sym, up):
            self.assertNotEqual(solution.solve_path, "fallback")
        self.assertLessEqual(np.max(sym.j_nominal), 2.5 + 1e-6)
        self.assertGreater(np.max(up.j_nominal), 2.5 + 0.5)
        self.assertLessEqual(np.max(up.j_nominal), 5.0 + 1e-6)
        self.assertGreater(up.action, sym.action)
        # braking for a close, slower cut-in: the falling side stays at jerk_limit
        near = prediction(gap0=16.0, lead_v=6.0, p_cutin=0.9)
        _, brake = solve(design_config(jerk_limit=2.5, jerk_limit_up=5.0), ego=[0.0, 12.0], prev=0.0, pred=near)
        self.assertNotEqual(brake.solve_path, "fallback")
        self.assertGreaterEqual(np.min(brake.j_nominal), -2.5 - 1e-6)
        self.assertLess(np.min(brake.j_nominal), -2.5 + 1e-3)        # the falling bound binds
        self.assertAlmostEqual(brake.action, -2.5 * TICK, places=6)

    def test_acceleration_input_rows_have_the_asymmetric_bound(self):
        far = prediction(gap0=80.0, lead_v=20.0)
        config = design_config(jerk_input=False, jerk_limit=2.5, jerk_limit_up=5.0, r_jerk=0.01, r_move=0.01)
        _, solution = solve(config, ego=[0.0, 10.0], prev=-2.0, pred=far)
        self.assertNotEqual(solution.solve_path, "fallback")
        u = solution.u_nominal[0]
        self.assertLessEqual(u[0] - (-2.0), 5.0 * TICK + 1e-6)
        self.assertGreater(u[0] - (-2.0), 2.5 * TICK + 1e-3)          # rises faster than the falling bound allows
        steps = np.diff(u)
        self.assertLessEqual(np.max(steps), 5.0 * DT + 1e-6)
        self.assertGreaterEqual(np.min(steps), -2.5 * DT - 1e-6)

    def test_limiter_bounds_the_rise_apart(self):
        sym, up = design_config(jerk_limit=2.5), design_config(jerk_limit=2.5, jerk_limit_up=5.0)
        self.assertAlmostEqual(limit_accel_command(sym, 1.0, 0.0, TICK), 0.25)
        self.assertAlmostEqual(limit_accel_command(up, 1.0, 0.0, TICK), 0.5)
        self.assertAlmostEqual(limit_accel_command(up, -1.0, 0.0, TICK), -0.25)
        self.assertAlmostEqual(limit_accel_command(up, -9.0, -3.4, TICK), -3.0)      # comfort a_min, then the jerk
        self.assertAlmostEqual(limit_accel_command(design_config(a_min=-3.5), -9.0, -3.4, TICK), -3.5)
        # the default limiter's arithmetic, number for number
        cfg = design_config()
        for action, prev, dt in ((1.7, 0.4, 0.1), (-2.9, -0.3, 0.05), (0.3, 1.9, 0.1), (-5.0, -2.95, 0.1)):
            max_delta = float(cfg.jerk_limit) * max(float(dt), 1.0e-6)
            old = float(np.clip(np.clip(float(np.clip(action, cfg.a_min, cfg.a_max)),
                                        prev - max_delta, prev + max_delta), cfg.a_min, cfg.a_max))
            self.assertEqual(limit_accel_command(cfg, action, prev, dt), old)

    def test_agent_limiter_delegates(self):
        class Fake:
            controller_config = design_config(jerk_limit=2.5, jerk_limit_up=4.0, a_min=-3.5)
        for action, prev in ((2.0, 0.0), (-4.0, -1.0), (0.1, 0.05)):
            self.assertEqual(ACCNairSMPCAgent._limit_accel_command(Fake(), action, prev, TICK),
                             limit_accel_command(Fake.controller_config, action, prev, TICK))

    def test_recovery_envelope_follows_the_failsafe_jerk_and_the_rising_bound(self):
        config = design_config(failsafe_recovery=True, failsafe_jerk=10.0, jerk_limit=2.5, jerk_limit_up=4.0)
        a_lo, j_hi, j0_lo = failsafe_recovery_envelope(-8.0, config, TICK)
        # e(t) = min(a_min, -8 + 10 t): -7 after the command interval, a_min from 0.5 s on
        self.assertAlmostEqual(j0_lo, 10.0)
        np.testing.assert_allclose(a_lo[:4], [-6.0, -4.0, -3.0, -3.0])
        self.assertEqual(j_hi[1], 10.0)
        self.assertEqual(j_hi[3], 4.0)                                # the comfort part: the rising bound
        # the command window: the fail-safe's jerk up to a_min, the falling side at jerk_limit
        self.assertAlmostEqual(failsafe_recovery_command(0.0, -8.0, config, TICK), -7.0)
        self.assertAlmostEqual(failsafe_recovery_command(-9.0, -7.0, config, TICK), -7.25)
        self.assertAlmostEqual(failsafe_recovery_command(0.0, -3.2, config, TICK), -2.8)   # a_min, then j_up
        # the floor is the configured fail-safe deceleration at speed, and lies above it near standstill
        config6 = design_config(failsafe_recovery=True, failsafe_decel=-6.0, failsafe_jerk=10.0)
        self.assertEqual(failsafe_recovery_floor(15.0, config6, TICK), -6.0)
        self.assertGreater(failsafe_recovery_floor(0.5, config6, TICK), -6.0)

    def test_terminal_row_covers_the_configured_failsafe(self):
        for a_em, j_em in ((FAILSAFE_DECEL, FAILSAFE_JERK), (-8.0, 10.0), (-6.0, 10.0), (-8.0, 5.0)):
            config = design_config(failsafe_branch=True, failsafe_recovery=True, failsafe_decel=a_em,
                                   failsafe_jerk=j_em)
            c_t, d_0 = failsafe_terminal_coefficients(config)
            self.assertAlmostEqual(c_t, (config.a_max - a_em) / j_em + config.v_max / (2.0 * abs(a_em)))
            self.assertAlmostEqual(d_0, 2.0 * config.a_max ** 3 / (3.0 * j_em ** 2))
            cfg = SafetyLayer(config.a_max, failsafe_decel=a_em, failsafe_jerk=j_em).cfg
            for v in np.concatenate([[1e-4, 1e-3], np.linspace(0.0, 0.2, 11), np.linspace(0.0, config.v_max, 41)]):
                for a in np.linspace(a_em, config.a_max, 21):
                    self.assertLessEqual(ego_brake_distance(v, a, cfg), c_t * v + d_0 + 1e-9,
                                         "a_em %g j_em %g v %g a %g" % (a_em, j_em, v, a))

    def test_branch_uses_the_configured_failsafe(self):
        branch = FailsafeBranch.from_members([VehicleObs(1, 14.0, 20.0, True)])
        config = design_config(failsafe_branch=True, failsafe_recovery=True, failsafe_jerk=10.0)
        _, solution = solve(config, ego=[0.0, 20.0], prev=0.0, branch=branch,
                            pred=prediction(gap0=80.0, lead_v=20.0))
        self.assertTrue(solution.failsafe_present)
        fs_j, fs_x = solution.failsafe_u_nominal, solution.failsafe_x_nominal
        self.assertLess(float(np.min(fs_j[1:])), -5.0)               # beyond the comfort jerk ...
        self.assertGreaterEqual(float(np.min(fs_j)), -10.0 - 1e-6)   # ... within the configured fail-safe's
        self.assertLess(float(np.min(fs_x[:, 2])), -3.0)
        self.assertGreaterEqual(float(np.min(fs_x[:, 2])), FAILSAFE_DECEL - 1e-6)


class SafetyLayerTests(unittest.TestCase):

    def test_parameters_and_validation(self):
        self.assertEqual((safety_layer_config(2.0).a_min, safety_layer_config(2.0).jerk_max),
                         (FAILSAFE_DECEL, FAILSAFE_JERK))
        cfg = SafetyLayer(2.0, failsafe_decel=-8.0, failsafe_jerk=10.0).cfg
        self.assertEqual((cfg.a_min, cfg.jerk_max, cfg.a_p_min), (-8.0, 10.0, LEAD_DECEL))
        for kwargs in (dict(failsafe_decel=-11.0), dict(failsafe_decel=0.5), dict(failsafe_jerk=0.0)):
            with self.assertRaises(ValueError):
                SafetyLayer(2.0, **kwargs)
        # a gentler fail-safe needs a longer gap: 4.4 / 7.8 / 11.9 m at jerk 20, 6.3 / 10.6 / 15.8 m at jerk 10
        for v, plain, gentle in ((10.0, 4.44, 6.28), (15.0, 7.80, 10.64), (20.0, 11.91, 15.75)):
            self.assertAlmostEqual(held_command_safe_distance(v, 0.0, v, SafetyLayer(2.0).cfg, TICK), plain, delta=0.01)
            self.assertAlmostEqual(held_command_safe_distance(v, 0.0, v, cfg, TICK), gentle, delta=0.01)

    def test_failsafe_runs_on_the_configured_profile(self):
        layer = SafetyLayer(2.0, failsafe_decel=-8.0, failsafe_jerk=10.0)
        layer.step(0.0, 20.0, 0.0, 0.0, [VehicleObs(1, 60.0, 20.0, True)], TICK)
        accel, cmds = 0.0, []
        for k in range(1, 12):
            dec = layer.step(k * TICK, 20.0, accel, 0.0, [VehicleObs(1, 5.0, 20.0, True)], TICK)
            self.assertEqual(dec.path, PATH_FAILSAFE)
            accel = dec.accel
            cmds.append(accel)
        np.testing.assert_allclose(cmds[:8], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], atol=1e-9)
        self.assertEqual(min(cmds), -8.0)

    def test_guarantee_holds_for_the_configured_failsafe(self):
        # random leads within [a_p, 2] and an adversarial nominal: no contact when the run starts outside the
        # configured fail-safe's safe distance (Prop. IV.5 by induction, as tests/test_safety_layer.py)
        rng = np.random.default_rng(20261111)
        for trial in range(200):
            dt = 0.1 if trial % 2 else 0.05
            a_em, j_em = (-8.0, 10.0) if trial % 3 else (-6.0, 5.0)
            layer = SafetyLayer(2.0, failsafe_decel=a_em, failsafe_jerk=j_em)
            v0, vl0 = rng.uniform(0.0, 25.0), rng.uniform(0.0, 25.0)
            gap0 = max(0.0, held_command_safe_distance(v0, 0.0, vl0, layer.cfg, dt)) + rng.uniform(0.0, 3.0)
            lead_seq = rng.uniform(LEAD_DECEL, 2.0, size=10)
            nom_seq = rng.uniform(-3.5, 2.0, size=10)
            steps = int(15 / dt)
            s_e, v_e, s_l, v_l, prev, gap_min = 0.0, v0, gap0, vl0, 0.0, gap0
            for k in range(steps):
                gap = s_l - s_e
                dec = layer.step(k * dt, v_e, prev, float(nom_seq[min(9, k * 10 // steps)]),
                                 [VehicleObs(1, gap, v_l, True)], dt)
                self.assertGreaterEqual(dec.accel, a_em - 1e-9)
                s_e, v_e = plant(s_e, v_e, dec.accel, dt)
                s_l, v_l = plant(s_l, v_l, float(lead_seq[min(9, k * 10 // steps)]), dt)
                prev = dec.accel
                gap_min = min(gap_min, s_l - s_e)
            self.assertGreaterEqual(gap_min, -1e-9, "trial %d" % trial)


class ISAMPCTests(unittest.TestCase):

    def test_planner_bounds_the_rising_side_apart(self):
        with self.assertRaises(ValueError):
            ISAMPCPlanner(horizon=HORIZON, dt=DT, v_des=17.0, jerk_limit=None, jerk_limit_up=3.0)
        # a cheap snap (q_snap 1e-4): the plan raises the jerk state as far as allowed
        sym = ISAMPCPlanner(horizon=HORIZON, dt=DT, v_des=17.0, jerk_limit=2.5, q_snap=1e-4)
        up = ISAMPCPlanner(horizon=HORIZON, dt=DT, v_des=17.0, jerk_limit=2.5, jerk_limit_up=5.0, q_snap=1e-4)
        out_sym = sym.plan(100.0, 10.0, -2.0, 0.0, 0.0, [], command_dt=TICK)
        out_up = up.plan(100.0, 10.0, -2.0, 0.0, 0.0, [], command_dt=TICK)
        self.assertEqual((out_sym["status"], out_up["status"]), ("OPTIMAL", "OPTIMAL"))
        self.assertLessEqual(np.max(out_sym["X"][3, 1:]), 2.5 + 1e-6)
        self.assertGreater(np.max(out_up["X"][3, 1:]), 3.0)
        self.assertLessEqual(np.max(out_up["X"][3, 1:]), 5.0 + 1e-6)
        self.assertGreaterEqual(np.min(out_up["X"][3, 1:]), -2.5 - 1e-6)
        self.assertEqual(up._clip_jerk(7.0), 5.0)
        self.assertEqual(up._clip_jerk(-7.0), -2.5)

    def test_explicit_symmetric_bound_is_the_default_problem(self):
        for accel_input in (False, True):
            kw = dict(horizon=HORIZON, dt=DT, v_des=17.0, accel_input=accel_input, r_jerk=10.0, r_move=1000.0)
            plain, same = ISAMPCPlanner(**kw), ISAMPCPlanner(jerk_limit_up=5.0, **kw)
            a = plain.plan(100.0, 14.0, -1.0, -0.5, 0.0, [], command_dt=TICK)
            b = same.plan(100.0, 14.0, -1.0, -0.5, 0.0, [], command_dt=TICK)
            self.assertEqual((a["accel_raw"], a["jerk_next"]), (b["accel_raw"], b["jerk_next"]))
            np.testing.assert_array_equal(a["U"], b["U"])

    def test_accel_input_rows(self):
        planner = ISAMPCPlanner(horizon=HORIZON, dt=DT, v_des=17.0, accel_input=True, r_jerk=0.01, r_move=0.01,
                                jerk_limit=2.5, jerk_limit_up=5.0)
        out = planner.plan(100.0, 10.0, -2.0, 0.0, 0.0, [], command_dt=TICK)
        self.assertEqual(out["status"], "OPTIMAL")
        u = out["U"]
        self.assertLessEqual(u[0] + 2.0, 5.0 * TICK + 1e-6)
        self.assertGreater(u[0] + 2.0, 2.5 * TICK + 1e-3)
        self.assertLessEqual(np.max(np.diff(u)), 5.0 * DT + 1e-6)
        self.assertGreaterEqual(np.min(np.diff(u)), -2.5 * DT - 1e-6)


def issue(action, prev, dt, lo, hi, jerk, jerk_up=None):
    """The NGSIM replay's command limiter (``baselines._limit``) with the rising side's bound apart."""
    action = float(np.clip(action, lo, hi))
    if jerk is None:
        return action
    up = jerk if jerk_up is None else jerk_up
    return float(np.clip(np.clip(action, prev - jerk * dt, prev + up * dt), lo, hi))


def pcacc_loop(policy, v0, gap0, lead_accel, T, dt=TICK, v_set=20.0):
    """One ego-lane lead (member from the start), the ideal plant, the limiter on ``command_bounds`` and
    ``command_jerk_up``.  Returns per tick (t, v, gap, path, issued command)."""
    ctrl = PCACCController(PCACCConfig.from_policy_string(policy))
    s_e, v, a, s_l, v_l = 0.0, v0, 0.0, gap0, v0
    rows = []
    for k in range(int(round(T / dt))):
        t = k * dt
        dec = ctrl.step(t, v, a, v_set, [VehicleObs(1, s_l - s_e, v_l, True)], dt)
        cmd = issue(dec.accel, a, dt, *ctrl.command_bounds(dec.path, a, dt), ctrl.command_jerk_up(dec.path))
        rows.append((t, v, s_l - s_e, dec.path, cmd))
        v_next = max(0.0, v + cmd * dt)
        s_e += 0.5 * (v + v_next) * dt
        v, a = v_next, cmd
        vl_next = max(0.0, v_l + lead_accel(t) * dt)
        s_l += 0.5 * (v_l + vl_next) * dt
        v_l = vl_next
    return rows


PAPER = "acc_pcacc_paperdsafe_amin10_amax3_jlim10"
PAPER_ISO = "acc_pcacc_paperdsafe_amin3.5_amax2_jlim2.5_emamin8_emjlim10"


class PCACCTests(unittest.TestCase):

    def test_tokens_and_validation(self):
        cfg = PCACCConfig.from_policy_string("acc_pcacc_jlimup2")
        self.assertEqual((cfg.jerk_max, cfg.jerk_up, cfg.jerk_up_max), (5.0, 2.0, 2.0))
        self.assertFalse(cfg.split_limits)
        iso = PCACCConfig.from_policy_string(PAPER_ISO + "_emics")
        self.assertEqual((iso.a_min, iso.a_max, iso.jerk_max, iso.em_a_min, iso.em_jerk_max),
                         (-3.5, 2.0, 2.5, -8.0, 10.0))
        self.assertTrue(iso.split_limits and iso.ics_emergency and iso.paper_d_safe)
        em = emergency_config(iso)
        self.assertEqual((em.a_min, em.jerk_max, em.a_max), (-8.0, 10.0, 2.0))
        # each emergency number defaults to Table I's
        self.assertEqual(emergency_config(PCACCConfig.from_policy_string("acc_pcacc_emamin8")).jerk_max,
                         TABLE_I_JERK_MAX)
        self.assertEqual(emergency_config(PCACCConfig.from_policy_string("acc_pcacc_emjlim8")).a_min, TABLE_I_A_MIN)
        # tokens never match inside other tokens
        for policy in ("acc_pcacc", PAPER, "acc_pcacc_physcheck_reachfix"):
            cfg = PCACCConfig.from_policy_string(policy)
            self.assertEqual((cfg.jerk_up, cfg.em_a_min, cfg.em_jerk_max, cfg.ics_emergency), (None, None, None, False))
        for policy in ("acc_pcacc_emics", "acc_pcacc_emamin2", "acc_pcacc_emamin10_apmin9", "acc_pcacc_jlimup0",
                       "acc_pcacc_emjlim0"):
            with self.assertRaises(ValueError, msg=policy):
                PCACCConfig.from_policy_string(policy)
        PCACCConfig.from_policy_string("acc_pcacc_physcheck_emics")

    def test_defaults_keep_their_bounds(self):
        for policy, expect in (("acc_pcacc", (-3.0, 2.0, 5.0)), (PAPER, (-10.0, 3.0, 10.0))):
            ctrl = PCACCController(PCACCConfig.from_policy_string(policy))
            self.assertIs(ctrl.em_cfg, ctrl.cfg)
            for path in (PATH_CRUISE, PATH_NOMINAL, PATH_EMERGENCY, PATH_RECAPTURE, PATH_ICS):
                self.assertEqual(ctrl.command_bounds(path, -1.0, TICK), expect)
                self.assertEqual(ctrl.command_jerk_up(path), expect[2])
            self.assertIsNone(ctrl.command_jerk_up(PATH_BOUNDED))
            # the paper's d_safe of the unsplit controller is the comfort config's, as before
            self.assertEqual(ctrl._d_safe(15.0, 0.0, 13.0, TICK),
                             safe_distance(15.0, 0.0, 13.0, ctrl.cfg, TICK) if ctrl.cfg.paper_d_safe else 22.5)
        phys = PCACCController(PCACCConfig.from_policy_string("acc_pcacc_physcheck"))
        self.assertEqual(phys.command_bounds(PATH_EMERGENCY, -1.0, TICK), (TABLE_I_A_MIN, 2.0, TABLE_I_JERK_MAX))
        self.assertEqual(phys.command_jerk_up(PATH_EMERGENCY), TABLE_I_JERK_MAX)
        self.assertEqual(phys.command_bounds(PATH_ICS, -8.0, TICK), (-7.5, 2.0, 5.0))
        self.assertEqual(phys.command_jerk_up(PATH_ICS), 5.0)

    def test_batch_qp_rising_bound(self):
        qp_sym = _BatchQP(20, TICK, (5.0, 10.0, 50.0), 100.0, 2.5, "sym")
        qp_up = _BatchQP(20, TICK, (5.0, 10.0, 50.0), 100.0, 2.5, "up", jerk_up=5.0)
        args = dict(ds_min=np.full(20, -np.inf), dv_lo=np.full(20, -np.inf), dv_hi=np.full(20, np.inf),
                    a_lo=-3.5, a_hi=2.0)
        # far behind a fast lead from -3 m/s^2: the plan raises the acceleration at the rising bound
        ok_s, u_s, _ = qp_sym.solve([60.0, 8.0, -3.0], 20.0, **args)
        ok_u, u_u, _ = qp_up.solve([60.0, 8.0, -3.0], 20.0, **args)
        self.assertTrue(ok_s and ok_u)
        self.assertLessEqual(np.max(u_s), 2.5 + 1e-6)
        self.assertGreater(np.max(u_u), 2.5 + 0.5)
        self.assertLessEqual(np.max(u_u), 5.0 + 1e-6)
        self.assertGreaterEqual(np.min(u_u), -2.5 - 1e-6)

    def test_published_controller_with_split_limits(self):
        ctrl = PCACCController(PCACCConfig.from_policy_string(PAPER_ISO))
        em = ctrl.em_cfg
        # Def. II.1's safe distance comes from the fail-safe (-8, |j| 10), not from the comfort limits
        self.assertEqual(ctrl._d_safe(15.0, 0.0, 13.0, TICK), safe_distance(15.0, 0.0, 13.0, em, TICK))
        self.assertLess(ctrl._d_safe(15.0, 0.0, 15.0, TICK), safe_distance(15.0, 0.0, 15.0, ctrl.cfg, TICK) - 10.0)
        self.assertEqual(ctrl.command_bounds(PATH_EMERGENCY, -1.0, TICK), (-8.0, 2.0, 10.0))
        self.assertEqual(ctrl.command_bounds(PATH_NOMINAL, -6.0, TICK), (-5.75, 2.0, 2.5))   # back at the comfort jerk
        self.assertEqual(ctrl.command_bounds(PATH_ICS, -1.0, TICK), (-3.5, 2.0, 2.5))        # comfort without emics
        # a lead braking at -7 for 1.5 s at the safe distance: the fail-safe brakes on -8 / jerk 10, the comfort
        # side stays within [-3.5, 2] at |j| <= 2.5 (rising: 2.5 only after the fail-safe), no contact
        v0 = 15.0
        gap0 = safe_distance(v0, 0.0, v0, em, TICK) + 0.5
        rows = pcacc_loop(PAPER_ISO, v0, gap0, lambda t: -7.0 if 1.0 <= t < 2.5 else 0.0, 12.0)
        cmds = np.array([r[4] for r in rows])
        paths = [r[3] for r in rows]
        self.assertGreater(min(r[2] for r in rows), 0.0)
        self.assertGreaterEqual(cmds.min(), -8.0 - 1e-9)
        self.assertLess(cmds.min(), -3.5)
        self.assertIn(PATH_EMERGENCY, paths)
        steps = np.diff(cmds)
        for k, step in enumerate(steps, start=1):
            if paths[k] == PATH_EMERGENCY:
                self.assertGreaterEqual(step, -10.0 * TICK - 1e-9)
            elif paths[k] != PATH_BOUNDED:
                self.assertGreaterEqual(step, -2.5 * TICK - 1e-9, "tick %d %s" % (k, paths[k]))
                self.assertLessEqual(step, 2.5 * TICK + 1e-9, "tick %d %s" % (k, paths[k]))
                if cmds[k] >= -3.5 - 1e-9:
                    self.assertLessEqual(cmds[k], 2.0 + 1e-9)

    def test_emics_brakes_on_the_emergency_limits(self):
        cfg = PCACCConfig.from_policy_string(PAPER_ISO + "_emics")
        ctrl = PCACCController(cfg)
        self.assertEqual(ctrl.command_bounds(PATH_ICS, -1.0, TICK), (-8.0, 2.0, 10.0))
        self.assertEqual(ctrl.command_jerk_up(PATH_ICS), 10.0)
        self.assertEqual(ctrl._full_braking(-1.0, TICK), -2.0)
        self.assertEqual(ctrl._full_braking(-7.5, TICK), -8.0)
        # a cut-in 4 m ahead and 5 m/s slower at 15 m/s is an ICS on both limits: without emics the full braking is
        # the comfort jerk's first step, with it the emergency jerk's
        comfort, emics = PCACCController(PCACCConfig.from_policy_string(PAPER_ISO)), ctrl
        for c in (comfort, emics):
            c.step(0.0, 15.0, 0.0, 20.0, [VehicleObs(1, 40.0, 10.0, False)], TICK)
        dec_c = comfort.step(TICK, 15.0, 0.0, 20.0, [VehicleObs(1, 4.0, 10.0, True)], TICK)
        dec_e = emics.step(TICK, 15.0, 0.0, 20.0, [VehicleObs(1, 4.0, 10.0, True)], TICK)
        self.assertEqual(dec_c.path, PATH_ICS)
        self.assertAlmostEqual(dec_c.accel, -2.5 * TICK)
        self.assertEqual(dec_e.path, PATH_ICS)
        self.assertAlmostEqual(dec_e.accel, -10.0 * TICK)


if __name__ == "__main__":
    unittest.main()
