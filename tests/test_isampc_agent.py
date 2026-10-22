import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from policies.isampc_agent import (  # noqa: E402
    ISAMPCAgent,
    ISAMPCPlanner,
    ZHOU_EPSILON,
    isampc_targets_from_bundle,
    lon_tracker_matrices,
    occupancy_box,
)
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    ACCModePrediction,
    ACCProcessedPrediction,
    REL_LEFT_ADJACENT,
)
from utils.acc_nair_smpc import acc_dynamics_matrices, original_nair_acc_config  # noqa: E402

N = 15
DT = 0.2
STEPS = DT * np.arange(1, N + 1)


def _two_mode_target(vehicle_id, s0, d0, v0, p_cutin, d_end=0.0, lane_change_s=2.5):
    """Mode 0 keeps d0, mode 1 moves linearly to d_end over lane_change_s."""
    probs = np.array([1.0 - p_cutin, p_cutin, 0.0])
    s = np.tile(s0 + v0 * STEPS, (3, 1))
    d = np.tile(np.full(N, d0), (3, 1))
    d[1] = d0 + (d_end - d0) * np.clip(STEPS / lane_change_s, 0.0, 1.0)
    return {"vehicle_id": vehicle_id, "relation": REL_LEFT_ADJACENT, "s0": s0, "d0": d0, "v0": v0,
            "probs": probs, "s": s, "d": d,
            "var_s": np.tile(0.3 * STEPS, (3, 1)), "var_d": np.tile(0.05 * STEPS, (3, 1))}


class TestOccupancyBox(unittest.TestCase):
    def test_single_mode_box_sits_on_the_mean(self):
        xb, yb, hs, hd, included = occupancy_box([1.0], [30.0], [3.5], [0.0], [0.0], ZHOU_EPSILON)
        self.assertLess(abs(xb - 30.0), 0.3)   # 0.55 m grid
        self.assertLess(abs(yb - 3.5), 0.1)    # 0.15 m grid
        self.assertEqual((hs, hd), (0.0, 0.0))
        self.assertEqual(included, [0])

    def test_minority_mode_below_epsilon_is_dropped(self):
        # 0.2 < 0.8 * 0.8: the cut-in component never reaches the level set
        xb, yb, hs, hd, included = occupancy_box([0.8, 0.2], [30.0, 30.0], [3.5, 0.0], [0.3, 0.3], [0.05, 0.05], ZHOU_EPSILON)
        self.assertEqual(included, [0])
        self.assertGreater(yb - hd, 2.0)
        # equal weights: both in, the box spans both lanes
        xb, yb, hs, hd, included = occupancy_box([0.5, 0.5], [30.0, 30.0], [3.5, 0.0], [0.3, 0.3], [0.05, 0.05], ZHOU_EPSILON)
        self.assertEqual(included, [0, 1])
        self.assertLess(yb - hd, 0.5)
        self.assertGreater(yb + hd, 3.0)

    def test_tracker_matrices_match_zhou_velocity_tracking(self):
        a, b = lon_tracker_matrices(0.32, (0.1029, 0.3423))
        np.testing.assert_allclose(a[1], [0.0, 1 - 0.1029 * 0.32 ** 2 / 2, 0.32 - 0.3423 * 0.32 ** 2 / 2])
        np.testing.assert_allclose(b, [0.0, 0.1029 * 0.32 ** 2 / 2, 0.1029 * 0.32])


class TestISAMPCPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0)

    def test_free_road_accelerates_toward_desired_speed(self):
        out = self.planner.plan(100.0, 14.0, 0.0, 0.0, 0.0, [])
        self.assertEqual(out["status"], "OPTIMAL")
        self.assertTrue(out["feasible"])
        self.assertEqual(out["v_ref"], 17.0)
        self.assertGreater(out["accel_raw"], 0.0)
        self.assertTrue(all(x is None for x in out["x_dv"]))
        self.assertTrue(np.isnan(out["min_gap"]))

    def test_in_lane_lead_bounds_the_reference_speed(self):
        lead = _two_mode_target(1, 125.0, 0.0, 12.0, 0.0)
        out = self.planner.plan(100.0, 17.0, 0.0, 0.0, 0.0, [lead])
        self.assertEqual(out["status"], "OPTIMAL")
        occ, = out["occupancies"]
        self.assertTrue(occ["in_ego_lane"].all())
        self.assertTrue(all(x is not None for x in out["x_dv"]))
        # rear edge = the predicted centre within the grid cell: no longitudinal
        # vehicle growth, the fork's L covers the vehicle
        self.assertLess(abs(out["x_dv"][0] - (125.0 + 12.0 * DT)), 0.3)
        self.assertLess(out["v_ref"], 12.0)
        self.assertEqual(out["binding"][0], 1)
        # the projection keeps the fork's standoff L + d0 + tau v = 7.5 + 1.3 v to every
        # rear edge under the closed-loop tracker, with equality at the binding step
        A, B = lon_tracker_matrices(DT)
        x, s_cl = np.array([100.0, 17.0, 0.0]), []
        for _ in range(N):
            x = A @ x + B * out["v_ref"]
            s_cl.append(x[0])
        surplus = np.asarray(out["x_dv"]) - np.asarray(s_cl) - (7.5 + 1.3 * out["v_ref"])
        self.assertGreaterEqual(float(surplus.min()), -1.0e-6)
        self.assertAlmostEqual(float(surplus[out["binding"][1] - 1]), 0.0, places=6)
        self.assertLess(out["accel_raw"], 0.0)
        self.assertGreaterEqual(float(np.min(out["rho"])), -1.0e-6)
        # the standoff is soft in the QP: the slack carries the planned gap's shortfall
        self.assertLess(out["min_gap"], 0.0)
        X = out["X"]
        shortfall = (7.5 + 1.3 * X[1, 1:]) - (np.asarray(out["x_dv"]) - X[0, 1:])
        np.testing.assert_allclose(out["rho"], np.maximum(shortfall, 0.0), atol=1.0e-2)
        self.assertAlmostEqual(out["min_gap"], -float(np.max(out["rho"])), places=2)

    def test_cutin_hypothesis_enters_only_above_epsilon_times_the_dominant_mode(self):
        # 2-mode weight ratio p / (1 - p) >= 0.8  <=>  p >= 0.444
        strong = self.planner.plan(100.0, 17.0, 0.0, 0.0, 0.0, [_two_mode_target(2, 122.0, 3.5, 15.0, 0.5)])
        occ, = strong["occupancies"]
        self.assertTrue(occ["in_ego_lane"].any())
        self.assertLess(strong["v_ref"], 17.0)
        self.assertLess(strong["accel_raw"], 0.0)
        weak = self.planner.plan(100.0, 17.0, 0.0, 0.0, 0.0, [_two_mode_target(2, 122.0, 3.5, 15.0, 0.4)])
        occ, = weak["occupancies"]
        self.assertFalse(occ["in_ego_lane"].any())
        self.assertEqual(weak["v_ref"], 17.0)
        self.assertIsNone(weak["binding"])
        self.assertAlmostEqual(weak["accel_raw"], 0.0, places=2)

    def test_vehicle_behind_the_ego_is_ignored(self):
        behind = _two_mode_target(3, 95.0, 0.0, 12.0, 0.0)
        out = self.planner.plan(100.0, 17.0, 0.0, 0.0, 0.0, [behind])
        self.assertEqual(out["occupancies"], [])
        self.assertEqual(out["v_ref"], 17.0)

    def test_plan_respects_the_fork_limits(self):
        lead = _two_mode_target(1, 120.0, 0.0, 8.0, 0.0)
        out = self.planner.plan(100.0, 17.0, -1.0, -3.0, 0.0, [lead])
        X = out["X"]
        self.assertTrue((X[1, 1:] >= -1.0e-6).all())
        self.assertTrue((X[2, 1:] >= -3.0 - 1.0e-6).all() and (X[2, 1:] <= 2.0 + 1.0e-6).all())
        self.assertTrue((np.abs(X[3, 1:]) <= 5.0 + 1.0e-6).all())
        np.testing.assert_allclose(X[:, 0], [100.0, 17.0, -1.0, -3.0])

    def test_command_and_jerk_state_advance_along_the_plan(self):
        lead = _two_mode_target(1, 125.0, 0.0, 12.0, 0.0)
        out = self.planner.plan(100.0, 17.0, -0.5, -2.0, 0.0, [lead], command_dt=0.05)
        u0 = float(out["U"][0])
        self.assertAlmostEqual(out["accel_raw"], -0.5 + (-2.0) * 0.05 + 0.5 * u0 * 0.05 ** 2)
        self.assertAlmostEqual(out["jerk_next"], max(-5.0, min(5.0, -2.0 + u0 * 0.05)))
        # one command interval of the chain never exceeds the 5 m/s^3 command jerk limit
        self.assertLessEqual(abs(out["accel_raw"] - (-0.5)), 5.0 * 0.05 + 1.0e-9)
        # the default interval is one MPC step: the planned a_1
        default = self.planner.plan(100.0, 17.0, -0.5, -2.0, 0.0, [lead])
        self.assertAlmostEqual(default["accel_raw"], float(default["X"][2, 1]), places=6)

    def test_closed_loop_with_plan_state_feedback_has_no_limit_cycle(self):
        """Plant v += a*0.05 behind a 12 m/s lead; the issued command must not cycle at the jerk limit."""
        planner = ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0)
        s, v, a, j = 100.0, 17.0, 0.0, 0.0
        s_lead, tau = 130.0, 0.05
        cmds = []
        for _ in range(int(4.0 / tau)):
            out = planner.plan(s, v, a, j, 0.0, [_two_mode_target(1, s_lead, 0.0, 12.0, 0.0)], command_dt=tau)
            a = float(np.clip(out["accel_raw"], -3.0, 2.0))
            j = out["jerk_next"]
            cmds.append(a)
            v = max(0.0, v + a * tau)
            s += v * tau
            s_lead += 12.0 * tau
        jerk = np.diff(cmds) / tau
        self.assertLess(np.abs(jerk).max(), 5.0)
        self.assertLessEqual(int((np.diff(np.sign(jerk[np.abs(jerk) > 1.0e-6])) != 0).sum()), 2)
        self.assertLess(min(cmds), -1.0)   # it does brake for the slower lead

    def test_fallback_when_the_qp_fails(self):
        planner = ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0)
        planner.solve_mpc = lambda x0, v_ref, x_dv: (None, "gurobi_failed: test")
        out = planner.plan(100.0, 17.0, 0.0, 0.0, 0.0, [_two_mode_target(1, 125.0, 0.0, 12.0, 0.0)])
        self.assertEqual(out["status"], "fallback")
        self.assertEqual(out["solve_path"], "fallback")
        self.assertFalse(out["feasible"])
        self.assertAlmostEqual(out["accel_raw"], max(-3.0, 0.5 * (out["v_ref"] - 17.0)))


class TestSnapPathUnchanged(unittest.TestCase):
    """``ainput`` leaves the default snap QP alone: these values are what the planner
    returned before ``ainput`` existed (9b1f721), bit for bit."""

    def test_plans_match_the_values_from_before_ainput(self):
        planner = ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0)
        out = planner.plan(100.0, 17.0, -0.5, -2.0, 0.1,
                           [_two_mode_target(1, 125.0, 0.0, 12.0, 0.0), _two_mode_target(2, 118.0, 3.5, 15.0, 0.5)],
                           command_dt=0.05)
        self.assertEqual((out["input"], out["status"], out["binding"]), ("snap", "OPTIMAL", (1, 14)))
        np.testing.assert_allclose(
            [out["accel_raw"], out["jerk_next"], out["v_ref"], out["cost"], out["min_gap"]],
            [-0.6003255096425583, -2.0130203857023323, 5.8747747917641515, 123.43449949942365, -11.951759761967029],
            rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            out["U"], [-0.2604077140466483, 0.4961410484620756, 1.0014406899009882, 1.2970614994458478,
                       1.4260816313958726, 1.4284669589804166, 1.3392473349016758, 1.1887185446703339,
                       1.0025835124314986, 0.8020226893398485, 0.6036898310858877, 0.4203959129056475,
                       0.26161000006806295, 0.13402920750474506, 0.04220291013688325], rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            out["rho"], [5.590325720543008, 6.007805125123353, 11.951760216667855, 11.901571373734079,
                         11.120698397274452, 10.719737136257027, 10.158886618260256, 9.448271171778279,
                         8.047708768584139, 7.066372142414104, 5.962494662968128, 4.943169784082447,
                         4.214265972965172, 3.3804659562617045, 1.8955592258047629], rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            out["X"][:, -1], [142.1673653970221, 10.10130873737003, -2.69233717346764, 0.2366568114359353],
            rtol=0.0, atol=1.0e-9)
        # free road, default command interval (one MPC step)
        out = planner.plan(100.0, 14.0, 0.3, 1.0, 0.0, [])
        np.testing.assert_allclose(
            [out["accel_raw"], out["jerk_next"], out["v_ref"], out["cost"]],
            [0.47575929000373507, 0.7575929000373506, 17.0, 6.860619469035709], rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            out["U"], [-1.212035499813247, -1.047533375753943, -0.8884966916174372, -0.7393738286769435,
                       -0.603187532503104, -0.48177237445369847, -0.375991915926534, -0.2859326273496437,
                       -0.21107310257973172, -0.15042836260259618, -0.10267002501444049, -0.0662238671579068,
                       -0.03934686929435711, -0.02018622575775883, -0.006823089328204256], rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(
            out["X"][:, -1], [144.99437652453724, 15.988981468522796, 0.47595426002799757, -0.24621507756589978],
            rtol=0.0, atol=1.0e-9)


class TestAccelInputPlanner(unittest.TestCase):
    """``ainput``: state (s, v), input a_k, the fork's double integrator and jerk terms."""

    @classmethod
    def setUpClass(cls):
        # what acc_isampc_ainput_rmove1000 gets: the parent config's r_jerk and rmove1000
        cls.planner = ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0, accel_input=True, r_jerk=10.0, r_move=1000.0)

    def test_plan_respects_the_fork_jerk_limits_and_bounds(self):
        # a lead 20 m ahead at 8 m/s; with near-zero jerk weights the plan brakes as hard
        # as the limits allow, so the limits are active
        loose = ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0, accel_input=True, r_jerk=1.0e-3, r_move=1.0e-3)
        lead = _two_mode_target(1, 120.0, 0.0, 8.0, 0.0)
        A, B = acc_dynamics_matrices(DT)
        for planner in (self.planner, loose):
            out = planner.plan(100.0, 17.0, -1.0, 0.0, 0.0, [lead], command_dt=0.05)
            self.assertEqual((out["input"], out["status"]), ("accel", "OPTIMAL"))
            X, U = out["X"], out["U"]
            self.assertEqual(X.shape, (2, N + 1))
            np.testing.assert_allclose(X[:, 0], [100.0, 17.0])
            self.assertTrue((U >= -3.0 - 1.0e-6).all() and (U <= 2.0 + 1.0e-6).all())
            self.assertTrue((X[1, 1:] >= -1.0e-6).all())
            delta = np.diff(np.concatenate(([-1.0], U)))
            # a_0 against the issued command over the command interval, a_k against a_{k-1} over dt
            self.assertLessEqual(abs(delta[0]), 5.0 * 0.05 + 1.0e-6)
            self.assertLessEqual(np.abs(delta[1:]).max(), 5.0 * DT + 1.0e-6)
            for k in range(N):
                np.testing.assert_allclose(X[:, k + 1], A @ X[:, k] + B[:, 0] * U[k], atol=1.0e-6)
            # the command is a_0 itself, no snap chain
            self.assertEqual(out["accel_raw"], float(U[0]))
        # both jerk limits and the lower bound bind in the loose plan
        self.assertAlmostEqual(delta[0], -0.25, places=4)
        self.assertGreater(np.abs(delta[1:]).max(), 0.99)
        self.assertAlmostEqual(float(U.min()), -3.0, places=4)

    def test_cost_is_zhou_terms_plus_the_fork_jerk_terms(self):
        """0.5 sum a_k^2 + (v_N - v_ref)^2 + 0.055 sum rho^2 + r_move (a_0 - u_prev)^2
        + r_jerk sum_{k>=1} ((a_k - a_{k-1}) / dt)^2, the last two as NairACCSMPC's QPs
        price them (step 0 not divided by any dt, later steps by the MPC dt)."""
        planner = ISAMPCPlanner(horizon=4, dt=DT, v_des=17.0, accel_input=True, r_jerk=10.0, r_move=1000.0)
        self.assertEqual(planner.weights,
                         {"q_accel": 0.5, "q_terminal_v": 1.0, "q_slack": 0.055, "r_jerk": 10.0, "r_move": 1000.0})

        def fork_jerk_terms(u, u_prev):
            previous, total = u_prev, 0.0
            for step, control in enumerate(u):
                move = control - previous
                total += 1000.0 * move ** 2 if step == 0 else 10.0 * (move / DT) ** 2
                previous = control
            return total

        def by_hand(u, v_end, rho, v_ref, u_prev):
            return (0.5 * np.sum(u ** 2) + (v_end - v_ref) ** 2 + 0.055 * np.sum(rho ** 2)
                    + fork_jerk_terms(u, u_prev))

        out = planner.plan(100.0, 15.0, -0.7, 0.0, 0.0, [_two_mode_target(1, 118.0, 0.0, 10.0, 0.0)],
                           command_dt=0.05)
        self.assertEqual(out["status"], "OPTIMAL")
        U, X = out["U"], out["X"]
        self.assertGreater(fork_jerk_terms(U, -0.7), 0.1)
        self.assertAlmostEqual(out["cost"], by_hand(U, X[1, 4], out["rho"], out["v_ref"], -0.7), places=6)
        # the QP's objective is that expression everywhere, not only at its optimum
        q = planner._qp
        u, x, rho = np.array([0.4, -1.1, 0.3, 1.5]), np.arange(10.0).reshape(2, 5), np.array([0.2, 0.0, 1.3, 0.7])
        value = float(q["opti"].value(q["cost"], [q["U"] == u, q["X"] == x, q["RHO"] == rho,
                                                  q["VREF"] == 12.0, q["UPREV"] == -0.7]))
        self.assertAlmostEqual(value, by_hand(u, x[1, 4], rho, 12.0, -0.7), places=6)

    def test_closed_loop_has_no_limit_cycle(self):
        """The snap test's loop with a_0 issued as the command: plant v += a*0.05 behind a 12 m/s lead."""
        s, v, a = 100.0, 17.0, 0.0
        s_lead, tau = 130.0, 0.05
        cmds = []
        for _ in range(int(4.0 / tau)):
            out = self.planner.plan(s, v, a, 0.0, 0.0, [_two_mode_target(1, s_lead, 0.0, 12.0, 0.0)], command_dt=tau)
            a = float(np.clip(out["accel_raw"], -3.0, 2.0))
            cmds.append(a)
            v = max(0.0, v + a * tau)
            s += v * tau
            s_lead += 12.0 * tau
        jerk = np.diff(cmds) / tau
        self.assertLess(np.abs(jerk).max(), 5.0)
        self.assertLessEqual(int((np.diff(np.sign(jerk[np.abs(jerk) > 1.0e-6])) != 0).sum()), 2)
        self.assertLess(min(cmds), -1.0)   # it does brake for the slower lead


class TestTargetsFromBundle(unittest.TestCase):
    def test_modes_and_sampled_variances(self):
        predictor_dt = 0.1
        n_full = 30
        traj = np.zeros((3, N, 2))
        traj[:, :, 0] = 50.0 + 15.0 * STEPS
        traj[:, :, 1] = 3.5
        traj[1, :, 1] = 3.5 - 3.5 * np.clip(STEPS / 2.5, 0.0, 1.0)
        cov = np.zeros((3, n_full, 2, 2))
        cov[:, :, 0, 0] = np.arange(n_full)          # var_s = full-rate index
        cov[:, :, 1, 1] = 10.0 + np.arange(n_full)
        raw = {"pred_traj_frenet": traj, "raw_intention_prob": np.array([0.3, 0.6, 0.1]),
               "full_pred_position_cov_frenet": cov}
        frenet = np.zeros((N + 1, 3))
        frenet[0] = [50.0, 3.5, 15.0]
        mode = ACCModePrediction(vehicle_id=7, mode_name="lk", probability=1.0, frenet=frenet,
                                 active_mask=np.zeros(N + 1, dtype=bool), raw_mode_indices=[0],
                                 ego_lane_membership_split_step=N)
        target = ACCProcessedPrediction(vehicle_id=7, relation_to_ego_lane=REL_LEFT_ADJACENT,
                                        mode_predictions=[mode], raw_mode_prob={}, acc_mode_prob={},
                                        branch_info={}, raw_prediction=raw)

        def sample_future(values, horizon, controller_dt):   # the adapter's _sample_future
            values = np.asarray(values)
            idx = [int(np.clip(int(round(step * controller_dt / predictor_dt)) - 1, 0, values.shape[1] - 1))
                   for step in range(1, int(horizon) + 1)]
            return values[:, idx, ...]

        targets = isampc_targets_from_bundle({"processed_targets": [target]}, N, DT, sample_future)
        t, = targets
        self.assertEqual(t["vehicle_id"], 7)
        self.assertEqual((t["s0"], t["d0"], t["v0"]), (50.0, 3.5, 15.0))
        np.testing.assert_allclose(t["probs"], [0.3, 0.6, 0.1])
        self.assertEqual(t["s"].shape, (3, N))
        np.testing.assert_allclose(t["var_s"][0], 2 * np.arange(N) + 1)       # steps 0.2 s -> indices 1, 3, ...
        np.testing.assert_allclose(t["var_d"][0], 10.0 + 2 * np.arange(N) + 1)
        self.assertEqual(isampc_targets_from_bundle(None, N, DT, sample_future), [])


class TestPolicyString(unittest.TestCase):
    def test_acc_isampc_always_uses_the_iaimm_kf_predictor(self):
        self.assertEqual(ISAMPCAgent._parse_predictor_type("acc_isampc"), "iaimm_kf")

    def test_ainput_takes_the_jerk_weights_from_the_parent_parsing(self):
        self.assertTrue(ISAMPCAgent._parse_accel_input("acc_isampc_ainput_rmove1000"))
        self.assertFalse(ISAMPCAgent._parse_accel_input("acc_isampc"))
        self.assertEqual(ISAMPCAgent._parse_r_move("acc_isampc_ainput_rmove1000"), 1000.0)
        self.assertIsNone(ISAMPCAgent._parse_r_jerk("acc_isampc_ainput_rmove1000"))
        # without the tokens the parent's config prices it: r_jerk 10, r_move = r_jerk / dt^2
        cfg = original_nair_acc_config(horizon=N, dt=DT)
        self.assertEqual(cfg.r_jerk, 10.0)
        self.assertAlmostEqual(cfg.r_move, 250.0)

    def test_qslack_sets_the_standoff_slack_weight(self):
        self.assertEqual(ISAMPCAgent._parse_q_slack("acc_isampc"), 0.055)
        self.assertEqual(ISAMPCAgent._parse_q_slack("acc_isampc_qslack5"), 5.0)
        self.assertEqual(ISAMPCAgent._parse_q_slack("acc_isampc_ainput_qslack0.5_rmove1000"), 0.5)
        self.assertEqual(ISAMPCPlanner(horizon=N, dt=DT, v_des=17.0, q_slack=5.0).weights["q_slack"], 5.0)


if __name__ == "__main__":
    unittest.main()
