"""``failsafe_recovery`` (safelayer, final-check item 7, 2026-11-04): the MPC's exit from the safety layer's
fail-safe.  The fail-safe leaves the acceleration below a_min; the MPC starts from it as it is and plans the
recovery along the fastest envelope at the fail-safe's jerk, and the command limiter opens the same window.
Before, the first command after a -8 m/s^2 fail-safe was -3 whatever the plan (50 m/s^3 at 0.1 s)."""
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    _failsafe_recovery_speed_loss,
    failsafe_recovery_envelope,
    failsafe_recovery_floor,
    original_nair_acc_config,
)
from utils.pcacc import VehicleObs  # noqa: E402
from utils.safety_layer import FAILSAFE_DECEL, FAILSAFE_JERK, PATH_NOMINAL, SafetyLayer  # noqa: E402
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402

HORIZON = 15
DT = 0.2
TICK = 0.1
A_EM, J_EM = FAILSAFE_DECEL, FAILSAFE_JERK


def design_config(jerk_input=True, recovery=True, **overrides):
    """The current design's QP (dopt100, r_a 0, l1slack1e6; jinput by default) with or without the
    recovery."""
    config = original_nair_acc_config(
        horizon=HORIZON, dt=DT, desired_speed=20.0, num_modes=2, controller_variant=VARIANT_FIXED_RISK)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move = 1000.0
    config.r_jerk = 40.0
    config.r_a = 0.0
    config.standoff_shortfall_weight = 100.0
    config.jerk_input = jerk_input
    config.slack_l1_weight = 1.0e6
    config.failsafe_recovery = recovery
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def far_prediction(s_ego, lead_v=20.0):
    """Two far leads (80 m): no safety row binds, so the MPC only tracks the set speed."""
    steps = HORIZON + 1
    means = np.zeros((2, steps, 2))
    means[0, :, 0] = s_ego + 80.0 + lead_v * DT * np.arange(steps)
    means[1, :, 0] = s_ego + 80.0 + lead_v * DT * np.arange(steps)
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


def solve(config, ego_v, prev, command_dt=TICK, controller=None):
    controller = NairACCSMPC(config) if controller is None else controller
    solution = controller.solve(np.array([0.0, ego_v]), far_prediction(0.0), previous_accel_cmd=prev,
                                command_dt=command_dt, ego_accel=prev)
    return controller, solution


def plant(s, v, a, dt):
    v_next = max(0.0, v + a * dt)
    return s + 0.5 * (v + v_next) * dt, v_next


def limiter(config, action, prev, dt):
    """The agent's command limiter on a config (``ACCNairSMPCAgent._limit_accel_command``)."""
    fake = types.SimpleNamespace(controller_config=config)
    return ACCNairSMPCAgent._limit_accel_command(fake, action, prev, dt)


class EnvelopeTests(unittest.TestCase):

    def test_envelope_with_the_jerk_input(self):
        config = design_config()
        a_lo, j_hi, j0_lo = failsafe_recovery_envelope(-8.0, config, TICK)
        np.testing.assert_allclose(a_lo[:3], [-4.0, -3.0, -3.0])
        self.assertTrue(np.all(a_lo == -3.0) or np.all(a_lo[2:] == -3.0))
        np.testing.assert_allclose(j_hi[:3], [20.0, 20.0, 5.0])         # the fail-safe's jerk while below a_min
        self.assertEqual(j0_lo, 20.0)                                   # j_0 fixed: the command is -6
        a_lo, j_hi, j0_lo = failsafe_recovery_envelope(-3.5, config, TICK)
        np.testing.assert_allclose(a_lo, -3.0)
        np.testing.assert_allclose(j_hi, 5.0)                           # the command reaches a_min at the comfort jerk
        self.assertAlmostEqual(j0_lo, 5.0)
        a_lo, j_hi, j0_lo = failsafe_recovery_envelope(-8.0, config, 0.05)   # CARLA's command interval
        self.assertAlmostEqual(j0_lo, 20.0)                             # the command is -7
        np.testing.assert_allclose(a_lo[:2], [-4.0, -3.0])

    def test_envelope_with_the_acceleration_input(self):
        config = design_config(jerk_input=False)
        a_lo, j_hi, j0_lo = failsafe_recovery_envelope(-8.0, config, TICK)
        np.testing.assert_allclose(a_lo[:3], [-6.0, -3.0, -3.0])        # u_0 over the command interval, then dt
        np.testing.assert_allclose(j_hi[:3], [20.0, 20.0, 5.0])
        self.assertEqual(j0_lo, -5.0)

    def test_floor_keeps_the_planned_speed_non_negative(self):
        for jerk_input in (True, False):
            config = design_config(jerk_input=jerk_input)
            for cdt in (TICK, 0.05):
                self.assertEqual(failsafe_recovery_floor(10.0, config, cdt), A_EM)     # no clip at speed
                self.assertEqual(failsafe_recovery_floor(0.0, config, cdt), 0.0)        # a stopped car is held
                for v in np.linspace(0.05, 4.0, 40):
                    a = failsafe_recovery_floor(v, config, cdt)
                    self.assertGreaterEqual(a, A_EM)
                    self.assertLessEqual(a, 0.0)
                    self.assertLessEqual(_failsafe_recovery_speed_loss(a, config, cdt), v + 1e-9)
                    if a > A_EM + 1e-6:
                        self.assertGreater(_failsafe_recovery_speed_loss(a - 1e-3, config, cdt), v)
            # the design's floor is the fail-safe's deceleration from 1.6 m/s up (jinput, 0.1 s) and above a_min
            # below 0.9 m/s, the old -sqrt(2 j v) clip's regime
            if jerk_input:
                self.assertEqual(failsafe_recovery_floor(1.7, config, TICK), A_EM)
                self.assertGreater(failsafe_recovery_floor(1.5, config, TICK), A_EM)
                self.assertGreater(failsafe_recovery_floor(0.8, config, TICK), config.a_min)
                self.assertAlmostEqual(failsafe_recovery_floor(0.3, config, TICK), -1.75, delta=0.05)


class ControllerTests(unittest.TestCase):

    def test_bit_identical_outside_a_recovery(self):
        for jerk_input in (True, False):
            for prev in (0.0, -3.0, 1.5):
                _, off = solve(design_config(jerk_input=jerk_input, recovery=False), 15.0, prev)
                _, on = solve(design_config(jerk_input=jerk_input, recovery=True), 15.0, prev)
                self.assertEqual(on.action, off.action)
                np.testing.assert_array_equal(on.u_nominal, off.u_nominal)

    def test_rejects_the_open_loop_variant(self):
        with self.assertRaises(ValueError):
            design_config(controller_variant=VARIANT_MULTIMODAL_OL, standoff_shortfall_weight=None,
                          jerk_input=False, slack_l1_weight=None)

    def test_recovery_ramps_at_the_failsafe_jerk_up_to_a_min(self):
        config = design_config()
        controller = NairACCSMPC(config)
        for prev, expect in ((-8.0, -6.0), (-6.0, -4.0), (-4.0, -3.0), (-3.5, -3.0), (-7.0, -5.0)):
            _, solution = solve(config, 15.0, prev, controller=controller)
            self.assertNotEqual(solution.solve_path, "fallback")
            self.assertAlmostEqual(solution.action, expect, places=5)
            self.assertEqual(controller._ego_accel, prev)                # the start state is the issued command
            plan_a = np.asarray(solution.u_nominal[0], dtype=float)      # jinput: the planned a_1..a_N
            self.assertGreaterEqual(float(plan_a[1]), config.a_min - 1e-6)   # back in the comfort set by a_2
            jerks = np.asarray(solution.j_nominal[0], dtype=float)
            self.assertLessEqual(float(np.max(np.abs(jerks[2:]))), config.jerk_limit + 1e-6)
        _, solution = solve(config, 15.0, -8.0, command_dt=0.05, controller=controller)
        self.assertAlmostEqual(solution.action, -7.0, places=5)          # CARLA's interval: -8 -> -7

    def test_recovery_with_the_acceleration_input(self):
        config = design_config(jerk_input=False)
        controller = NairACCSMPC(config)
        for prev, lo, hi in ((-8.0, -6.0, -6.0), (-5.0, -3.0, -3.0), (-3.2, -3.0, -2.7)):
            _, solution = solve(config, 15.0, prev, controller=controller)
            self.assertNotEqual(solution.solve_path, "fallback")
            self.assertGreaterEqual(solution.action, lo - 1e-5)         # e(command_dt) reached ...
            self.assertLessEqual(solution.action, hi + 1e-5)            # ... beyond it at the comfort jerk
            plan = np.asarray(solution.u_nominal[0], dtype=float)
            self.assertGreaterEqual(float(plan[1]), config.a_min - 1e-6)
            self.assertLessEqual(float(np.max(np.diff(plan[1:]))), config.jerk_limit * DT + 1e-6)

    def test_low_speed_start_is_raised_to_the_floor_and_solves(self):
        config = design_config()
        controller = NairACCSMPC(config)
        for v in (0.3, 1.0, 2.0):
            start = max(-8.0, failsafe_recovery_floor(v, config, TICK))
            _, solution = solve(config, v, start, controller=controller)
            self.assertNotEqual(solution.solve_path, "fallback", "v %g" % v)
            self.assertGreaterEqual(float(np.min(solution.x_nominal[0, :, 1])), -1e-6)


class LimiterTests(unittest.TestCase):

    def test_window_follows_the_failsafe_below_a_min(self):
        config = design_config()
        self.assertEqual(limiter(config, -6.0, -8.0, TICK), -6.0)        # the ramp passes ...
        self.assertEqual(limiter(config, -3.0, -8.0, TICK), -6.0)        # ... and nothing faster
        self.assertEqual(limiter(config, -8.0, -8.0, TICK), -8.0)        # the fail-safe's deceleration is admissible
        self.assertEqual(limiter(config, -9.0, -8.0, TICK), -8.0)
        self.assertEqual(limiter(config, -3.0, -3.5, TICK), -3.0)
        self.assertEqual(limiter(config, -2.0, -3.2, TICK), -2.7)        # beyond a_min the comfort jerk
        self.assertEqual(limiter(config, -3.1, -3.2, TICK), -3.1)        # (reaching a_min is the MPC's row)
        self.assertEqual(limiter(config, -2.0, -3.0, TICK), -2.5)        # at a_min the nominal window is back
        self.assertEqual(limiter(config, -3.5, -3.0, TICK), -3.0)
        off = design_config(recovery=False)
        self.assertEqual(limiter(off, -6.0, -8.0, TICK), -3.0)           # the old clip without the flag


class ClosedLoopTests(unittest.TestCase):

    def _loop(self, lead_decel, ticks=160):
        """Ego at 20 m/s, a lead 25 m ahead at 20 m/s braking at ``lead_decel`` to a stop; the MPC's hypotheses
        never cover it, the layer does.  The replay's protocol: the MPC and the limiter start from the issued
        command, raised at low speed to the recovery floor."""
        config = design_config()
        controller = NairACCSMPC(config)
        layer = SafetyLayer(config.a_max)
        s_e, v_e, s_l, v_l, prev = 0.0, 20.0, 25.0, 20.0, 0.0
        rows = []
        for k in range(ticks):
            mpc_prev = prev
            if prev < config.a_min:
                mpc_prev = max(prev, failsafe_recovery_floor(v_e, config, TICK))
            solution = controller.solve(np.array([s_e, v_e]), far_prediction(s_e), previous_accel_cmd=mpc_prev,
                                        command_dt=TICK, ego_accel=mpc_prev)
            self.assertNotEqual(solution.solve_path, "fallback", "tick %d" % k)
            nominal = limiter(config, solution.action, prev, TICK)
            decision = layer.step(k * TICK, v_e, prev, nominal, [VehicleObs(1, s_l - s_e, v_l, True)], TICK)
            rows.append(dict(k=k, v=v_e, gap=s_l - s_e, prev=prev, mpc_prev=mpc_prev, a=decision.accel,
                             path=decision.path, nominal=nominal, raw=solution.action))
            s_e, v_e = plant(s_e, v_e, decision.accel, TICK)
            s_l, v_l = plant(s_l, v_l, lead_decel if v_l > 0.0 else 0.0, TICK)
            prev = decision.accel
        return rows

    def test_exit_from_the_failsafe_is_a_ramp_at_its_jerk(self):
        rows = self._loop(-8.0)
        acted = [r for r in rows if r["path"] != PATH_NOMINAL]
        self.assertGreater(len(acted), 0)
        self.assertLessEqual(min(r["a"] for r in rows), A_EM + 1e-9)          # the fail-safe reached -8
        self.assertGreater(min(r["gap"] for r in rows), 0.0)                  # the layer's guarantee
        changes = [(rows[i + 1]["a"] - rows[i]["a"]) / TICK for i in range(len(rows) - 1)]
        self.assertLessEqual(max(changes), J_EM + 1e-6)                       # no 50 m/s^3 release, ever
        # on every tick inside a recovery the MPC started from the issued command, except at low speed, where
        # the floor raised the start and the limiter held the command on the ramp; the issued command was the
        # MPC's plan wherever the start was the issued command
        inside = [r for r in rows if r["prev"] < -3.0 and r["path"] == PATH_NOMINAL]
        self.assertGreater(len(inside), 0)
        floored = [r for r in inside if r["mpc_prev"] != r["prev"]]
        self.assertTrue(all(r["v"] < 1.7 for r in floored))
        for r in floored:
            self.assertAlmostEqual(r["a"], min(-3.0, r["prev"] + J_EM * TICK), places=6)
        for r in inside:
            if r["mpc_prev"] == r["prev"]:
                self.assertAlmostEqual(r["a"], r["raw"], places=6)
                self.assertGreaterEqual(r["a"], min(-3.0, r["prev"] + J_EM * TICK) - 1e-6)
                self.assertLessEqual(r["a"], max(min(-3.0, r["prev"] + J_EM * TICK), r["prev"] + 5.0 * TICK) + 1e-6)
        # back inside the comfort set the comfort jerk applies
        after = [i for i in range(len(rows) - 1) if rows[i]["a"] >= -3.0 and rows[i]["path"] == PATH_NOMINAL
                 and rows[i + 1]["path"] == PATH_NOMINAL]
        self.assertTrue(all(abs(changes[i]) <= 5.0 + 1e-6 for i in after))


if __name__ == "__main__":
    unittest.main()
