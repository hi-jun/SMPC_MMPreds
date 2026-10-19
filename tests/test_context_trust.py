import importlib.util
import json
import os
import pathlib
import pickle
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    CTX_LATERAL_CROSSING_M,
    CTX_LATERAL_NOISE_M,
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    REL_RIGHT_ADJACENT,
    lateral_observation,
    mix_cutin_probability,
    process_vehicle_prediction,
)


def _load_aggregator():
    path = os.path.join(ROOT, "scripts", "carla", "devel", "aggregate_cutin_table.py")
    spec = importlib.util.spec_from_file_location("aggregate_cutin_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestContextTrustMixing(unittest.TestCase):
    """``ctx<lambda>``: p_lambda = p_S [(1 - lambda) + lambda o_v] on the cut-in
    intention of an adjacent vehicle; the removed mass goes to lane keeping."""

    HORIZON = 3

    @staticmethod
    def _raw(intention_prob, relation=REL_LEFT_ADJACENT):
        """Adjacent target whose cut-in-direction trajectory enters the ego lane."""
        s = np.array([20.0, 22.0, 24.0])
        side = 3.5 if relation == REL_LEFT_ADJACENT else -3.5
        stay = np.column_stack((s, np.full_like(s, side)))
        enter = np.column_stack((s, np.array([side, 0.0, 0.0])))
        trajectories = [stay, stay, stay]
        trajectories[2 if relation == REL_LEFT_ADJACENT else 1] = enter
        return {
            "vehicle_id": 10,
            "raw_intention_prob": np.asarray(intention_prob, dtype=float),
            "pred_traj_frenet": np.stack(trajectories),
            "raw_pred_vel": np.ones((3, 3, 2)),
            "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, 3), dtype=bool),
        }

    def _process(self, raw, relation=REL_LEFT_ADJACENT, **kwargs):
        side = 3.5 if relation == REL_LEFT_ADJACENT else -3.5
        return process_vehicle_prediction(
            raw,
            relation,
            current_frenet_state=np.array([18.0, side, 8.0]),
            horizon=self.HORIZON,
            dt=0.2,
            ego_lane_threshold=0.5,
            **kwargs
        )

    @staticmethod
    def _mode(processed, name):
        return next((m for m in processed.mode_predictions if m.mode_name == name), None)

    def test_observation_clips_between_the_noise_and_the_crossing_offset(self):
        self.assertEqual(lateral_observation(-0.3), 0.0)
        self.assertEqual(lateral_observation(0.05), 0.0)
        self.assertEqual(lateral_observation(CTX_LATERAL_NOISE_M), 0.0)
        mid = 0.5 * (CTX_LATERAL_NOISE_M + CTX_LATERAL_CROSSING_M)
        self.assertAlmostEqual(lateral_observation(mid), 0.5)
        self.assertEqual(lateral_observation(CTX_LATERAL_CROSSING_M), 1.0)
        self.assertEqual(lateral_observation(1.5), 1.0)

    def test_mixing_moves_the_removed_mass_to_lane_keeping(self):
        raw = np.array([0.2, 0.1, 0.7])
        # Left-adjacent: RLC (index 2) is the cut-in direction; o_v = 0 inside the noise band.
        mixed, info = mix_cutin_probability(raw, REL_LEFT_ADJACENT, 0.5, 0.05)
        np.testing.assert_allclose(mixed, [0.55, 0.1, 0.35])
        self.assertAlmostEqual(float(np.sum(mixed)), 1.0)
        self.assertEqual(info, {"lambda": 0.5, "y_v": 0.05, "o_v": 0.0, "p_raw": 0.7, "p_mixed": 0.35})
        np.testing.assert_allclose(raw, [0.2, 0.1, 0.7], err_msg="the input is not modified")
        # Right-adjacent: LLC (index 1); o_v = 0.5 halfway between the two offsets.
        mid = 0.5 * (CTX_LATERAL_NOISE_M + CTX_LATERAL_CROSSING_M)
        mixed, info = mix_cutin_probability(np.array([0.2, 0.6, 0.2]), REL_RIGHT_ADJACENT, 1.0, mid)
        np.testing.assert_allclose(mixed, [0.5, 0.3, 0.2])
        self.assertAlmostEqual(info["o_v"], 0.5)
        # Past the crossing offset nothing is removed, and lambda = 0 removes nothing either.
        mixed, _ = mix_cutin_probability(np.array([0.2, 0.6, 0.2]), REL_RIGHT_ADJACENT, 1.0, 1.2)
        np.testing.assert_allclose(mixed, [0.2, 0.6, 0.2])
        mixed, _ = mix_cutin_probability(np.array([0.2, 0.6, 0.2]), REL_RIGHT_ADJACENT, 0.0, 0.0)
        np.testing.assert_allclose(mixed, [0.2, 0.6, 0.2])

    def test_vehicles_off_the_adjacent_lanes_and_without_an_offset_are_untouched(self):
        raw = np.array([0.2, 0.1, 0.7])
        for relation, offset in ((REL_EGO_LANE, 0.0), ("other", 0.0), (REL_LEFT_ADJACENT, None)):
            mixed, info = mix_cutin_probability(raw, relation, 1.0, offset)
            np.testing.assert_allclose(mixed, raw)
            self.assertEqual(info["lambda"], 1.0)
            self.assertIsNone(info["p_mixed"])

    def test_lambda_zero_is_identical_to_no_token(self):
        raw = self._raw([0.2, 0.1, 0.7])
        plain = self._process(raw, cutin_probability_threshold=0.1, cutin_chance_ref=0.6)
        zero = self._process(raw, cutin_probability_threshold=0.1, cutin_chance_ref=0.6,
                             context_trust=0.0, lateral_offset_m=0.0)
        self.assertEqual(plain.raw_mode_prob, zero.raw_mode_prob)
        self.assertEqual(plain.acc_mode_prob, zero.acc_mode_prob)
        self.assertEqual([m.probability for m in plain.mode_predictions],
                         [m.probability for m in zero.mode_predictions])
        self.assertEqual([m.clearance_scale for m in plain.mode_predictions],
                         [m.clearance_scale for m in zero.mode_predictions])
        self.assertIsNone(plain.branch_info["context_trust"])
        self.assertEqual(zero.branch_info["context_trust"]["p_mixed"], 0.7)

    def test_the_mixed_value_reaches_the_mapping_gate_and_chance_factor(self):
        raw = self._raw([0.2, 0.1, 0.7])
        # Full trust in the measurement and no lateral motion: the cut-in intention
        # is 0 and carries no mode; lane keeping holds the mass.
        none = self._process(raw, cutin_probability_threshold=0.1, cutin_chance_ref=0.6,
                             context_trust=1.0, lateral_offset_m=0.0)
        self.assertIsNone(self._mode(none, "cutin"))
        self.assertAlmostEqual(none.acc_mode_prob["cutin"], 0.0)
        self.assertAlmostEqual(none.raw_mode_prob["LK"], 0.9)
        self.assertAlmostEqual(none.raw_mode_prob["LLC"], 0.1, msg="away-direction intention untouched")
        # Halfway across the band at lambda = 0.5: 0.7 (0.5 + 0.5 * 0.5) = 0.525, which
        # the gate (0.1) passes and which is the chance confidence (below beta_ref).
        mid = 0.5 * (CTX_LATERAL_NOISE_M + CTX_LATERAL_CROSSING_M)
        half = self._process(raw, cutin_probability_threshold=0.1, cutin_chance_ref=0.6,
                             context_trust=0.5, lateral_offset_m=mid)
        cutin = self._mode(half, "cutin")
        self.assertAlmostEqual(cutin.probability, 0.525)
        self.assertAlmostEqual(half.branch_info["context_trust"]["p_mixed"], 0.525)
        self.assertTrue(cutin.active_mask.any())
        self.assertAlmostEqual(cutin.chance_confidence, 0.525)
        # A small mixed value is gated like a small raw one would be.
        tiny = self._process(raw, cutin_probability_threshold=0.1,
                             context_trust=0.9, lateral_offset_m=0.0)
        self.assertAlmostEqual(self._mode(tiny, "cutin").probability, 0.07)
        self.assertFalse(self._mode(tiny, "cutin").active_mask.any())
        # The trajectory is the predictor's whatever the mixing did.
        np.testing.assert_array_equal(cutin.frenet, self._mode(self._process(raw), "cutin").frenet)


class TestContextTrustToken(unittest.TestCase):
    """The agent reads ``ctx<lambda>``; it is a whole token and leaves the others alone."""

    BASE = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8"
            "_cutin_chance0.6_dopt100lkr_trajcutin")

    def test_the_agent_reads_the_ctx_token(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent as agent
        self.assertIsNone(agent._parse_context_trust(self.BASE + "_rmove1000"))
        self.assertEqual(agent._parse_context_trust(self.BASE + "_ctx0.5_rmove1000"), 0.5)
        self.assertEqual(agent._parse_context_trust(self.BASE + "_ctx1.0_rmove1000"), 1.0)
        self.assertEqual(agent._parse_context_trust(self.BASE + "_ctx0_rmove1000"), 0.0)
        self.assertEqual(agent._parse_context_trust(self.BASE + "_ctx1"), 1.0)
        # Only a whole token: ``ctx`` inside another token is not read.
        self.assertIsNone(agent._parse_context_trust(self.BASE + "_foctx0.5_rmove1000"))
        self.assertIsNone(agent._parse_context_trust(self.BASE + "_ctx0.5x_rmove1000"))

    def test_the_ctx_token_leaves_every_other_token_alone(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent as agent
        plain = self.BASE + "_branchdec7_sigmacap0.9_cutin_gate0.2_rmove1000"
        with_ctx = self.BASE + "_branchdec7_sigmacap0.9_cutin_gate0.2_ctx0.5_rmove1000"
        parsers = [name for name in dir(agent) if name.startswith("_parse_")
                   and name not in ("_parse_context_trust", "_parse_controller_num_modes")]
        self.assertGreater(len(parsers), 20)
        for name in parsers:
            self.assertEqual(getattr(agent, name)(plain), getattr(agent, name)(with_ctx), name)
        self.assertEqual(agent._parse_controller_num_modes(with_ctx, 2), 8)
        self.assertEqual(agent._parse_r_move(with_ctx), 1000.0)
        self.assertEqual(agent._parse_cutin_probability_threshold(with_ctx), 0.2)
        self.assertEqual(agent._parse_chance_sigma_cap(with_ctx), 0.9)
        self.assertEqual(agent._parse_branch_decision_brake(with_ctx), 7.0)
        self.assertEqual(agent._parse_context_trust(with_ctx), 0.5)

    def test_ctx_needs_stdan_3int_and_a_lambda_in_the_unit_interval(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent

        class _Vehicle:  # the check runs before anything else touches the vehicle
            def get_world(self):
                return None

        lstm = "acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance_ctx0.5_rmove1000"
        with self.assertRaisesRegex(ValueError, "stdan_3int"):
            ACCNairSMPCAgent(_Vehicle(), None, smpc_config=lstm)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            ACCNairSMPCAgent(_Vehicle(), None, smpc_config=self.BASE + "_ctx1.5_rmove1000")

    def test_lateral_offset_is_measured_from_the_lane_centre_toward_the_ego_lane(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent as agent

        class _Point:
            def __init__(self, x, y):
                self.x, self.y = x, y

        class _Waypoint:
            def __init__(self, x, y, yaw_deg):
                self.transform = type("Transform", (), {})()
                self.transform.location = _Point(x, y)
                self.transform.rotation = type("Rotation", (), {})()
                self.transform.rotation.yaw = yaw_deg

        # CARLA frame (left-handed): a lane heading +y (yaw 90 deg) has the
        # driver's right at -x, as carla.Transform.get_right_vector() gives.
        lane = _Waypoint(10.0, 0.0, 90.0)
        right = _Point(9.0, 5.0)    # 1 m to the right of the lane centre, 5 m along it
        left = _Point(10.4, -2.0)   # 0.4 m to the left
        self.assertAlmostEqual(agent._lateral_offset_toward_ego_lane(right, lane, REL_LEFT_ADJACENT), 1.0)
        self.assertAlmostEqual(agent._lateral_offset_toward_ego_lane(right, lane, REL_RIGHT_ADJACENT), -1.0)
        self.assertAlmostEqual(agent._lateral_offset_toward_ego_lane(left, lane, REL_LEFT_ADJACENT), -0.4)
        self.assertAlmostEqual(agent._lateral_offset_toward_ego_lane(left, lane, REL_RIGHT_ADJACENT), 0.4)


class TestContextTrustLossEvents(unittest.TestCase):
    """``ctx_miss`` / ``ctx_unnec`` of ``aggregate_cutin_table.py`` on synthetic runs."""

    DT = 0.05
    T0 = 15.0
    N = 240  # 12 s at 20 Hz

    @classmethod
    def setUpClass(cls):
        cls.agg = _load_aggregator()

    def _ego(self, cmd):
        t = self.T0 + self.DT * np.arange(self.N)
        v = np.full(self.N, 15.0)
        s = 100.0 + 15.0 * (t - self.T0)
        steps = [{"time_s": float(t[i]), "ego_s": float(s[i]), "ego_v": float(v[i]),
                  "accel_cmd": float(cmd[i]), "solve_path": "cached", "feasible": True}
                 for i in range(self.N)]
        ego = {"policy_log": {"steps": steps},
               # The aggregator maps the road onto -y: s_actor = ego_s + (y_ego - y_actor).
               "state_trajectory": np.column_stack((t, np.zeros(self.N), -s, np.zeros(self.N), v)),
               "lane_trajectory": [{"time_s": float(ti), "lane_id": -2} for ti in t]}
        return t, s, ego

    def _actor(self, t, s_actor, x, lane_id, trigger=None):
        n = len(t)
        entry = {
            "state_trajectory": np.column_stack((
                t, np.broadcast_to(x, n), -np.asarray(s_actor, dtype=float),
                np.zeros(n), np.full(n, 15.0))),
            "lane_trajectory": [{"time_s": float(ti), "lane_id": int(lane)}
                                for ti, lane in zip(t, np.broadcast_to(lane_id, n))],
        }
        if trigger is not None:
            entry["policy_log"] = {"trigger_time_s": trigger}
        return entry

    def _collect(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "cutin_0000"
            run_dir.mkdir()
            json.dump({"sweep_params": {"ego_speed": 15.0, "max_sim_time_s": 19}},
                      open(str(run_dir / "resolved_config.json"), "w"))
            json.dump({"ran_successfully": True}, open(str(run_dir / "metrics.json"), "w"))
            pickle.dump(data, open(str(run_dir / "scenario_result.pkl"), "wb"))
            return self.agg.collect_run("policy", "01_cutin_normal", run_dir, **kwargs)

    def test_a_cut_in_followed_by_intrusion_is_a_miss(self):
        t, s, ego = self._ego(np.zeros(self.N))
        # The TV drives 10 m ahead in the left lane (x 3.5, lane -3) and enters
        # the ego lane 5 s after t0: bumper gap 5.5 m against d_safe 22.5 m,
        # i.e. delta 75.6 %.
        entered = np.arange(self.N) >= 100
        tv = self._actor(t, s + 10.0, np.where(entered, 0.0, 3.5),
                         np.where(entered, -2, -3), trigger=self.T0 + 3.0)
        data = {"ego_0": ego, "target_cutin_0": tv, "_collision_log": []}
        row = self._collect(data)
        self.assertTrue(row["valid"], row["invalid_reason"])
        self.assertAlmostEqual(row["t_cross"], 5.0, places=2)
        self.assertAlmostEqual(row["delta_max"], 75.56, places=1)
        self.assertTrue(row["ctx_miss"])
        self.assertFalse(row["ctx_unnec"])
        lenient = self._collect(data, ctx_delta_thr=0.9)
        self.assertFalse(lenient["ctx_miss"])
        # The summary counts the events per cell.
        agg = self.agg.aggregate([row, lenient])
        stats = agg[("01_cutin_normal", "policy")]
        self.assertEqual((stats["_ctx_miss"], stats["_ctx_unnec"], stats["_ctx_n"]), (1, 0, 2))
        self.assertEqual(stats["ctx_miss"][0], 0.5)
        self.assertIn("| 1/2 | 0/2 |", self.agg.markdown_table(agg))

    def test_braking_with_no_lead_inside_d_safe_and_no_entry_is_unnecessary(self):
        cmd = np.zeros(self.N)
        cmd[60:70] = -1.0   # brakes 3 s after t0
        t, s, ego = self._ego(cmd)
        tv = self._actor(t, s + 10.0, 3.5, -3)   # stays in the left lane
        row = self._collect({"ego_0": ego, "target_cutin_0": tv, "_collision_log": []})
        self.assertIsNone(row["t_cross"])
        self.assertTrue(row["ctx_unnec"])
        self.assertFalse(row["ctx_miss"])
        # The same braking behind an in-lane lead inside d_safe (bumper 5.5 m) is covered ...
        near = self._actor(t, s + 10.0, 0.0, -2)
        covered = self._collect({"ego_0": ego, "target_cutin_0": tv, "traffic_0": near,
                                 "_collision_log": []})
        self.assertFalse(covered["ctx_unnec"])
        # ... but a lead outside d_safe (bumper 35.5 m > 22.5 m) does not cover it.
        far = self._actor(t, s + 40.0, 0.0, -2)
        uncovered = self._collect({"ego_0": ego, "target_cutin_0": tv, "traffic_0": far,
                                   "_collision_log": []})
        self.assertTrue(uncovered["ctx_unnec"])
        # A braking command above the onset threshold is not braking.
        mild = np.zeros(self.N)
        mild[60:70] = -0.2
        _, _, calm = self._ego(mild)
        self.assertFalse(self._collect({"ego_0": calm, "target_cutin_0": tv,
                                        "_collision_log": []})["ctx_unnec"])
        # Without a cut-in vehicle neither event is defined.
        none = self._collect({"ego_0": ego, "_collision_log": []})
        self.assertIsNone(none["ctx_miss"])
        self.assertIsNone(none["ctx_unnec"])


if __name__ == "__main__":
    unittest.main()
