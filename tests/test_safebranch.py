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
    VARIANT_MULTIMODAL_OL,
    failsafe_branch_lead,
    failsafe_recovery_floor,
    failsafe_terminal_coefficients,
    original_nair_acc_config,
)
from utils.pcacc import VehicleObs, ego_brake_distance  # noqa: E402
from utils.safety_layer import (  # noqa: E402
    FAILSAFE_DECEL,
    FAILSAFE_JERK,
    LEAD_DECEL,
    PATH_FAILSAFE,
    PATH_NOMINAL,
    SafetyLayer,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


HORIZON = 15
DT = 0.2
TICK = 0.1
L = 4.5


def design_config(branch=True, **overrides):
    """The current design's QP (dopt100, jinput, r_a 0, l1slack1e6; fullfoot is the prediction's factor 1)
    with or without the fail-safe branch."""
    config = original_nair_acc_config(
        horizon=HORIZON, dt=DT, desired_speed=20.0, num_modes=2, controller_variant=VARIANT_FIXED_RISK)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move = 1000.0
    config.r_jerk = 40.0
    config.r_a = 0.0
    config.standoff_shortfall_weight = 100.0
    config.jerk_input = True
    config.slack_l1_weight = 1.0e6
    config.failsafe_branch = branch
    config.failsafe_recovery = True          # safelayer's exit from the fail-safe (item 7), as the agent sets it
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def far_prediction(s_ego, gap_lk=80.0, gap_cutin=80.0, lead_v=20.0):
    """Mode 0: an in-lane lane-keeping lead ``gap_lk`` ahead of ``s_ego``; mode 1: a cut-in chance cell
    ``gap_cutin`` ahead (probability 0.4, beta 0.4).  Both at ``lead_v``; with the default gaps neither row
    binds, so the MPC cruises and the hypotheses do not cover the layer's member."""
    steps = HORIZON + 1
    means = np.zeros((2, steps, 2))
    means[0, :, 0] = s_ego + gap_lk + lead_v * DT * np.arange(steps)
    means[1, :, 0] = s_ego + gap_cutin + lead_v * DT * np.arange(steps)
    means[:, :, 1] = lead_v
    confidence = np.full((2, steps), np.nan)
    confidence[1] = 0.4
    return MultimodalLeadPrediction(
        means=means,
        probabilities=np.array([0.6, 0.4]),
        covariances=np.tile(np.diag([0.25, 0.09]), (2, steps, 1, 1)),
        active_mask=np.ones((2, steps), dtype=bool),
        chance_confidence=confidence,
        lead_mode_lk=np.array([[True] * steps, [False] * steps]),
        lead_vehicle_ids=np.array([[10] * steps, [11] * steps]),
        clearance_scale=np.ones((2, steps)),
        mode_names=["10:lk", "11:cutin"],
    )


def solve(config, ego_v, branch=None, prev=0.0):
    controller = NairACCSMPC(config)
    solution = controller.solve(
        np.array([0.0, ego_v]), far_prediction(0.0), previous_accel_cmd=prev, command_dt=TICK, ego_accel=prev,
        failsafe_branch=branch)
    return controller, solution


def plant(s, v, a, dt):
    """The replay's ideal plant: the command held over the tick, speed clipped at 0, trapezoidal position."""
    v_next = max(0.0, v + a * dt)
    return s + 0.5 * (v + v_next) * dt, v_next


class LayerRefactorTests(unittest.TestCase):

    def test_step_is_classify_then_decide(self):
        # the same random observation sequence through two layers: step on one, classify + decide on the other
        rng = np.random.default_rng(20261103)
        one, two = SafetyLayer(2.0), SafetyLayer(2.0)
        n_non_nominal = 0
        for k in range(600):
            t = 0.1 * k
            v, a, a_nominal = rng.uniform(0.0, 22.0), rng.uniform(-8.0, 2.0), rng.uniform(-3.0, 2.0)
            vehicles = [VehicleObs(int(vid), rng.uniform(-2.0, 60.0), rng.uniform(0.0, 22.0), bool(rng.random() < 0.7))
                        for vid in rng.choice(6, size=rng.integers(0, 4), replace=False)]
            dec = one.step(t, v, a, a_nominal, vehicles, 0.1)
            checked, cutins = two.classify(t, v, a, vehicles, 0.1)
            dec2 = two.decide(t, v, a, a_nominal, checked, cutins, 0.1)
            self.assertEqual((dec.accel, dec.path, dec.vid, dec.margin, dec.checked_vids, dec.cutin_vids),
                             (dec2.accel, dec2.path, dec2.vid, dec2.margin, dec2.checked_vids, dec2.cutin_vids))
            self.assertEqual([veh.vid for veh in checked], dec.checked_vids)
            self.assertEqual([veh.vid for veh in cutins], dec.cutin_vids)
            n_non_nominal += dec.path != PATH_NOMINAL
        self.assertGreater(n_non_nominal, 20)        # the sequence exercised the fail-safe and the ICS


class BranchGeometryTests(unittest.TestCase):

    def test_lead_per_step_and_stop_point(self):
        config = design_config()
        branch = FailsafeBranch.from_members([VehicleObs(1, 12.0, 20.0, True), VehicleObs(2, 25.0, 5.0, True)])
        lead, stop = failsafe_branch_lead(branch, config)
        t = DT * np.arange(HORIZON + 1)
        decel = abs(LEAD_DECEL)
        one = 12.0 + L + np.where(t < 20.0 / decel, 20.0 * t - 0.5 * decel * t ** 2, 20.0 ** 2 / (2.0 * decel))
        two = 25.0 + L + np.where(t < 5.0 / decel, 5.0 * t - 0.5 * decel * t ** 2, 5.0 ** 2 / (2.0 * decel))
        np.testing.assert_allclose(lead, np.minimum(one, two), rtol=1e-12)
        self.assertEqual(lead[0], 12.0 + L)                     # the near member leads at first ...
        self.assertAlmostEqual(lead[-1], two[-1], places=12)    # ... the slow one once the near one has passed it
        self.assertAlmostEqual(stop, 25.0 + L + 25.0 / (2.0 * decel), places=12)   # the nearer stop point
        self.assertLess(stop, 12.0 + L + 400.0 / (2.0 * decel))
        # absent: 10^6 m ahead
        lead, stop = failsafe_branch_lead(None, config)
        self.assertTrue(np.all(lead == 1.0e6) and stop == 1.0e6)
        self.assertFalse(FailsafeBranch.from_members([]).present)

    def test_terminal_bound_covers_the_true_stopping_distance(self):
        # c_T v + D_0 against the layer's fail-safe stopping distance (ego_brake_distance with the layer's
        # constant-jerk profile) on a grid of (v, a); without D_0 the bound is below the true distance as
        # v -> 0 with a > 0, by at most D_0
        config = design_config()
        c_t, d_0 = failsafe_terminal_coefficients(config)
        self.assertAlmostEqual(c_t, (config.a_max - FAILSAFE_DECEL) / FAILSAFE_JERK + config.v_max / (2.0 * abs(FAILSAFE_DECEL)))
        self.assertAlmostEqual(d_0, 2.0 * config.a_max ** 3 / (3.0 * FAILSAFE_JERK ** 2))
        cfg = SafetyLayer(config.a_max).cfg
        worst = -np.inf
        for v in np.concatenate([[1e-4, 1e-3], np.linspace(0.0, 0.1, 21), np.linspace(0.0, config.v_max, 81)]):
            for a in np.linspace(FAILSAFE_DECEL, config.a_max, 41):
                true = ego_brake_distance(v, a, cfg)
                self.assertLessEqual(true, c_t * v + d_0 + 1e-9, "v %g a %g" % (v, a))
                worst = max(worst, true - c_t * v)
        self.assertGreater(worst, 0.01)              # D_0 is needed ...
        self.assertLessEqual(worst, d_0 + 1e-9)      # ... and suffices


class ControllerTests(unittest.TestCase):

    def test_config_validation(self):
        with self.assertRaises(ValueError):        # cached feedback QP only
            design_config(controller_variant=VARIANT_MULTIMODAL_OL, standoff_shortfall_weight=None,
                          jerk_input=False, slack_l1_weight=None)
        with self.assertRaises(ValueError):        # not with alag<tau>
            design_config(jerk_input=False, accel_lag_tau=0.5)
        with self.assertRaises(ValueError):        # not with with_k
            design_config(optimize_k=True)

    def test_without_the_token_nothing_is_added(self):
        controller, solution = solve(design_config(branch=False), 20.0)
        problem = next(iter(controller._feedback_scalar_chance_qp_cache.values()))
        for name in ("fs_u_var", "fs_slack_var", "fs_stop_slack_var", "fs_lead_param", "fs_stop_param"):
            self.assertIsNone(getattr(problem, name))
        self.assertFalse(solution.failsafe_present)
        self.assertEqual(solution.failsafe_members, 0)
        self.assertTrue(np.isnan(solution.failsafe_margin_min) and np.isnan(solution.failsafe_slack_max))
        self.assertIsNone(solution.failsafe_x_nominal)
        with self.assertRaises(ValueError):        # a branch needs the token
            controller.solve(np.array([0.0, 20.0]), far_prediction(0.0), previous_accel_cmd=0.0, command_dt=TICK,
                             ego_accel=0.0, failsafe_branch=FailsafeBranch.from_members([VehicleObs(1, 30.0, 20.0, True)]))
        # the QP key carries the token and nothing else
        controller_on, _ = solve(design_config(branch=True), 20.0)
        key_on = next(iter(controller_on._feedback_scalar_chance_qp_cache))
        key_off = next(iter(controller._feedback_scalar_chance_qp_cache))
        self.assertEqual(len(key_on), len(key_off))
        differ = [i for i, (a, b) in enumerate(zip(key_on, key_off)) if a != b]
        self.assertEqual(len(differ), 1)
        self.assertEqual((key_on[differ[0]], key_off[differ[0]]), (True, False))

    def test_absent_branch_only_regularises(self):
        # no checked member: the rows sit 10^6 m ahead; the command moves by the 1e-3 weight only
        _, off = solve(design_config(branch=False), 20.0)
        for branch in (None, FailsafeBranch.from_members([])):
            _, on = solve(design_config(branch=True), 20.0, branch)
            self.assertFalse(on.failsafe_present)
            self.assertEqual(on.failsafe_members, 0)
            self.assertTrue(np.isnan(on.failsafe_margin_min))
            self.assertEqual(on.failsafe_slack_max, 0.0)
            self.assertLess(abs(on.action - off.action), 1e-3)
            self.assertLess(float(np.max(np.abs(on.u_nominal - off.u_nominal))), 1e-2)

    def test_branch_shares_the_first_input_and_uses_the_emergency_limits_after_it(self):
        # a member 14 m ahead at the ego's speed: the branch must brake with the fail-safe's limits from step 1
        branch = FailsafeBranch.from_members([VehicleObs(1, 14.0, 20.0, True)])
        _, solution = solve(design_config(), 20.0, branch)
        self.assertTrue(solution.failsafe_present)
        self.assertEqual(solution.failsafe_members, 1)
        fs_j, fs_x = solution.failsafe_u_nominal, solution.failsafe_x_nominal
        self.assertAlmostEqual(fs_j[0], float(solution.j_nominal[0, 0]), places=6)         # shared j_0
        self.assertAlmostEqual(fs_j[0], float(solution.j_nominal[1, 0]), places=6)
        self.assertLessEqual(abs(fs_j[0]), 5.0 + 1e-6)                                  # within the comfort jerk
        self.assertLess(float(np.min(fs_j[1:])), -5.0)                                  # beyond it from step 1
        self.assertGreaterEqual(float(np.min(fs_j)), -FAILSAFE_JERK - 1e-6)
        self.assertLess(float(np.min(fs_x[:, 2])), -3.0)                                # beyond the comfort decel
        self.assertGreaterEqual(float(np.min(fs_x[:, 2])), FAILSAFE_DECEL - 1e-6)
        self.assertGreaterEqual(float(np.min(fs_x[:, 1])), -1e-6)
        # the other scenarios keep the comfort limits
        self.assertGreaterEqual(float(np.min(solution.u_nominal)), -3.0 - 1e-6)
        self.assertLessEqual(float(np.max(np.abs(solution.j_nominal))), 5.0 + 1e-6)
        # the branch holds the rows without slack, and the reported margin is the rows' (gap - 0.1 m)
        self.assertLess(solution.failsafe_slack_max, 1e-6)
        lead, stop = failsafe_branch_lead(branch, design_config())
        margins = lead[1:] - fs_x[1:, 0] - L - FAILSAFE_BRANCH_MARGIN_M
        self.assertAlmostEqual(solution.failsafe_margin_min, float(np.min(margins)), places=6)
        self.assertGreaterEqual(solution.failsafe_margin_min, -1e-6)
        c_t, d_0 = failsafe_terminal_coefficients(design_config())
        terminal = stop - L - FAILSAFE_BRANCH_MARGIN_M - d_0 - fs_x[HORIZON, 0] - c_t * fs_x[HORIZON, 1]
        self.assertAlmostEqual(solution.failsafe_terminal_margin, float(terminal), places=6)
        self.assertGreaterEqual(solution.failsafe_terminal_margin, -1e-6)

    def test_member_inside_the_guaranteed_gap_has_no_hard_solution_and_the_layer_brakes(self):
        # the layer first sees the vehicle far (covered), then 3 m ahead at the ego's speed: it stays a checked
        # member, the branch cannot be kept (slack), and the layer's check on the MPC's command fails
        layer = SafetyLayer(2.0)
        layer.step(0.0, 20.0, 0.0, 0.0, [VehicleObs(1, 60.0, 20.0, True)], TICK)
        checked, cutins = layer.classify(TICK, 20.0, 0.0, [VehicleObs(1, 3.0, 20.0, True)], TICK)
        self.assertEqual(([veh.vid for veh in checked], cutins), ([1], []))
        _, solution = solve(design_config(), 20.0, FailsafeBranch.from_members(checked))
        self.assertNotEqual(solution.solve_path, "fallback")
        self.assertGreater(solution.failsafe_slack_max, 1.0)
        self.assertLess(solution.failsafe_margin_min, -1.0)
        self.assertAlmostEqual(solution.action, -0.5, places=4)      # the exact penalty: the comfort jerk limit
        decision = layer.decide(TICK, 20.0, 0.0, solution.action, checked, cutins, TICK)
        self.assertEqual(decision.path, PATH_FAILSAFE)
        self.assertAlmostEqual(decision.accel, -FAILSAFE_JERK * TICK)

    def test_u0_passes_the_check_where_the_plain_command_fails(self):
        # equal speeds at 20 m/s: the plain MPC cruises (u0 = 0), which the layer's check accepts above 11.91 m;
        # the branch's command, the comfort jerk limit (-0.5), passes above 11.29 m.  At 11.6 m the plain command
        # fails and the fail-safe would issue -2.0 m/s^2 (jerk -20 over the tick); with the branch the command
        # brakes now and gently and passes (the branch itself has no hard solution here: it holds the shared
        # input for 0.2 s where the check holds it for 0.1 s)
        obs = [VehicleObs(1, 11.6, 20.0, True)]
        for with_branch in (False, True):
            layer = SafetyLayer(2.0)
            layer.step(0.0, 20.0, 0.0, 0.0, [VehicleObs(1, 60.0, 20.0, True)], TICK)
            checked, cutins = layer.classify(TICK, 20.0, 0.0, obs, TICK)
            _, solution = solve(design_config(branch=with_branch), 20.0,
                                FailsafeBranch.from_members(checked) if with_branch else None)
            decision = layer.decide(TICK, 20.0, 0.0, solution.action, checked, cutins, TICK)
            if with_branch:
                self.assertAlmostEqual(solution.action, -0.5, places=4)
                self.assertEqual(decision.path, PATH_NOMINAL)
                self.assertGreater(decision.margin, 0.2)
                self.assertGreater(solution.failsafe_slack_max, 0.5)
            else:
                self.assertGreater(solution.action, -0.01)
                self.assertEqual(decision.path, PATH_FAILSAFE)
                self.assertAlmostEqual(decision.accel, -2.0)

    def _closed_loop(self, with_branch, gap0, lead_decel, lead_v_min, ticks=90):
        """Ego at 20 m/s behind a lead at 20 m/s, ``gap0`` ahead, that brakes at ``lead_decel`` down to
        ``lead_v_min``.  The MPC's hypotheses never cover it (far leads); only the layer sees it, through the
        replay's protocol (after the layer's fail-safe the MPC starts from the issued command, raised at low
        speed to the recovery floor).  Returns per-tick rows (tick, bumper gap, command, path, branch slack
        or NaN)."""
        config = design_config(branch=with_branch)
        controller = NairACCSMPC(config)
        layer = SafetyLayer(config.a_max)
        s_e, v_e, s_l, v_l, prev = 0.0, 20.0, gap0, 20.0, 0.0
        rows = []
        for k in range(ticks):
            t = k * TICK
            obs = [VehicleObs(1, s_l - s_e, v_l, True)]
            mpc_prev = prev
            if prev < config.a_min:
                mpc_prev = max(prev, failsafe_recovery_floor(v_e, config, TICK))
            if with_branch:
                checked, cutins = layer.classify(t, v_e, prev, obs, TICK)
                solution = controller.solve(np.array([s_e, v_e]), far_prediction(s_e), previous_accel_cmd=mpc_prev,
                                            command_dt=TICK, ego_accel=mpc_prev,
                                            failsafe_branch=FailsafeBranch.from_members(checked))
                decision = layer.decide(t, v_e, prev, solution.action, checked, cutins, TICK)
                slack = solution.failsafe_slack_max if solution.failsafe_present else float("nan")
            else:
                solution = controller.solve(np.array([s_e, v_e]), far_prediction(s_e), previous_accel_cmd=mpc_prev,
                                            command_dt=TICK, ego_accel=mpc_prev)
                decision = layer.step(t, v_e, prev, solution.action, obs, TICK)
                slack = float("nan")
            self.assertNotEqual(solution.solve_path, "fallback", "tick %d" % k)
            rows.append((k, s_l - s_e, decision.accel, decision.path, slack))
            s_e, v_e = plant(s_e, v_e, decision.accel, TICK)
            s_l, v_l = plant(s_l, v_l, lead_decel if v_l > lead_v_min else 0.0, TICK)
            prev = decision.accel
        return rows

    @staticmethod
    def _stats(rows):
        acted = [r[0] for r in rows if r[3] != PATH_NOMINAL]
        brake = [r[0] for r in rows if r[2] <= -0.1]
        no_hard = [r[0] for r in rows if r[4] > 1e-4]
        feasible_but_failed = [r[0] for r in rows if r[4] <= 1e-4 and r[3] != PATH_NOMINAL]
        return {"n_acted": len(acted), "first_act": acted[0] if acted else None,
                "first_brake": brake[0] if brake else None, "a_min": min(r[2] for r in rows),
                "gap_min": min(r[1] for r in rows), "first_no_hard": no_hard[0] if no_hard else None,
                "feasible_but_failed": len(feasible_but_failed)}

    def test_closed_loop_brakes_earlier_and_more_gently_than_the_failsafe(self):
        # a lead 30 m ahead easing at -2 m/s^2 to 12 m/s: without the branch the MPC cruises until the check
        # fails and the fail-safe brakes at up to -4.6 m/s^2; with it the command brakes two ticks earlier,
        # within the comfort limits, passes the check at every tick and keeps a larger gap
        plain = self._stats(self._closed_loop(False, 30.0, -2.0, 12.0))
        branch = self._stats(self._closed_loop(True, 30.0, -2.0, 12.0))
        self.assertGreater(plain["n_acted"], 0)
        self.assertLess(plain["a_min"], -4.0)
        self.assertEqual(branch["n_acted"], 0)
        self.assertEqual(branch["feasible_but_failed"], 0)
        self.assertLess(branch["first_brake"], plain["first_brake"])
        self.assertGreaterEqual(branch["a_min"], -3.0 - 1e-6)
        self.assertLess(branch["a_min"], -2.0)
        self.assertGreater(branch["gap_min"], plain["gap_min"])
        self.assertGreater(plain["gap_min"], 0.0)

    def test_closed_loop_against_hard_sustained_lead_braking(self):
        # a lead 40 m ahead braking at -4 m/s^2 to 5 m/s: the shared input, comfort-limited, cannot keep the
        # branch feasible, so the branch reports "no hard solution" before the layer's check fails and the
        # fail-safe still runs (fewer ticks); whenever the branch had a hard solution the check passed
        plain = self._stats(self._closed_loop(False, 40.0, -4.0, 5.0))
        branch = self._stats(self._closed_loop(True, 40.0, -4.0, 5.0))
        self.assertGreater(plain["n_acted"], 0)
        self.assertGreater(branch["n_acted"], 0)
        self.assertLess(branch["n_acted"], plain["n_acted"])
        self.assertIsNotNone(branch["first_no_hard"])
        self.assertLess(branch["first_no_hard"], branch["first_act"])
        self.assertLessEqual(branch["first_brake"], plain["first_brake"])
        self.assertEqual(branch["feasible_but_failed"], 0)
        self.assertGreater(plain["gap_min"], 0.0)
        self.assertGreater(branch["gap_min"], 0.0)


class TokenTests(unittest.TestCase):

    def test_whole_token(self):
        parse = ACCNairSMPCAgent._parse_failsafe_branch
        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        self.assertTrue(parse(base + "_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_safebranch"))
        self.assertTrue(parse(base + "_safebranch_safelayer"))
        self.assertFalse(parse(base + "_safelayer_leadmerge"))
        self.assertFalse(parse(base + "_safebranches"))
        self.assertFalse(parse(base + "_nosafebranch"))

    def test_needs_safelayer(self):
        class _Vehicle:
            def get_world(self):
                return None

        base = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        with self.assertRaisesRegex(ValueError, "safelayer"):
            ACCNairSMPCAgent(_Vehicle(), None, smpc_config=base + "_jinput_ra0_fullfoot_l1slack1e6_leadmerge_safebranch")


if __name__ == "__main__":
    unittest.main()
