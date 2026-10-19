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
        # rear edge = centre - half length (2.15 m) within the grid cell
        self.assertLess(abs(out["x_dv"][0] - (125.0 + 12.0 * DT - 2.15)), 0.3)
        self.assertLess(out["v_ref"], 12.0)
        self.assertEqual(out["binding"][0], 1)
        self.assertLess(out["accel_raw"], 0.0)
        self.assertGreaterEqual(float(np.min(out["rho"])), -1.0e-6)
        self.assertLess(out["min_gap"], 0.0)   # the 1.5 s headway is soft: slack carries the shortfall

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


if __name__ == "__main__":
    unittest.main()
