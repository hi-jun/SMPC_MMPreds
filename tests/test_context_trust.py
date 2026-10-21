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
    CTX_GATE_P_MIN,
    CTX_GATE_VY_MIN_HISTORY_S,
    CTX_GATE_VY_MIN_MPS,
    CTX_GATE_VY_WINDOW_S,
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
        self.assertEqual(info, {"lambda": 0.5, "y_v": 0.05, "o_v": 0.0, "p_raw": 0.7, "p_mixed": 0.35,
                                "v_y": None, "trigger": None, "latched": None})
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


class TestLatchedLateralGate(unittest.TestCase):
    """``latgate``: g_v = 1 once the vehicle is past y_0, moving toward the ego lane
    at >= 0.2 m/s and predicted at p_S >= 0.9, held while it stays past y_0."""

    RAW = np.array([0.05, 0.05, 0.9])   # left-adjacent: RLC (index 2) is the cut-in direction
    Y_ON = CTX_LATERAL_NOISE_M + 0.05
    Y_OFF = CTX_LATERAL_NOISE_M - 0.05

    def _mix(self, raw=None, y_v=None, v_y=None, latched=False, lam=1.0, relation=REL_LEFT_ADJACENT):
        return mix_cutin_probability(self.RAW if raw is None else raw, relation, lam,
                                     self.Y_ON if y_v is None else y_v,
                                     lateral_speed_mps=v_y, latched=latched, lateral_gate=True)

    def test_trigger_truth_table(self):
        p_on, p_off = CTX_GATE_P_MIN, CTX_GATE_P_MIN - 0.01
        v_on, v_off = CTX_GATE_VY_MIN_MPS, CTX_GATE_VY_MIN_MPS - 0.01
        for y_v in (self.Y_ON, self.Y_OFF, CTX_LATERAL_NOISE_M):
            for v_y in (v_on, v_off, None, 1.5):
                for p in (p_on, p_off, 0.999):
                    raw = np.array([(1.0 - p) / 2.0, (1.0 - p) / 2.0, p])
                    _, info = self._mix(raw, y_v=y_v, v_y=v_y)
                    expect = (y_v > CTX_LATERAL_NOISE_M) and (v_y is not None and v_y >= CTX_GATE_VY_MIN_MPS) \
                        and (p >= CTX_GATE_P_MIN)
                    self.assertEqual(info["trigger"], expect, (y_v, v_y, p))
                    self.assertEqual(info["latched"], expect)
                    self.assertEqual(info["o_v"], 1.0 if expect else 0.0)
                    self.assertEqual(info["v_y"], v_y)
        # The right-adjacent vehicle's cut-in direction is LLC (index 1).
        _, info = self._mix(np.array([0.05, 0.9, 0.05]), v_y=0.5, relation=REL_RIGHT_ADJACENT)
        self.assertTrue(info["trigger"])
        _, info = self._mix(np.array([0.05, 0.05, 0.9]), v_y=0.5, relation=REL_RIGHT_ADJACENT)
        self.assertFalse(info["trigger"], "p_S is the cut-in-direction intention, not the other one")

    def test_the_latch_holds_past_the_noise_offset_and_releases_below_it(self):
        # Latched last tick, no trigger now (slow, or p dropped): still past y_0 -> held.
        for raw, v_y in ((self.RAW, 0.0), (np.array([0.6, 0.1, 0.3]), 1.0), (self.RAW, None)):
            mixed, info = self._mix(raw, v_y=v_y, latched=True)
            self.assertFalse(info["trigger"])
            self.assertTrue(info["latched"])
            self.assertEqual(info["o_v"], 1.0)
            np.testing.assert_allclose(mixed, raw, err_msg="lambda = 1 with g_v = 1 keeps p_S")
        # Back inside the noise band: released even with lateral speed and a confident predictor.
        mixed, info = self._mix(y_v=self.Y_OFF, v_y=1.0, latched=True)
        self.assertFalse(info["latched"])
        self.assertEqual(info["o_v"], 0.0)
        np.testing.assert_allclose(mixed, [0.95, 0.05, 0.0])
        # Not latched and no trigger: nothing is believed.
        _, info = self._mix(v_y=0.0, latched=False)
        self.assertFalse(info["latched"])
        self.assertEqual(info["o_v"], 0.0)

    def test_lambda_zero_is_unchanged_by_the_gate(self):
        for latched in (False, True):
            for v_y in (None, 0.0, 1.0):
                mixed, info = self._mix(v_y=v_y, latched=latched, lam=0.0)
                np.testing.assert_array_equal(mixed, self.RAW)
                self.assertEqual(info["p_mixed"], info["p_raw"])

    def test_the_removed_mass_goes_to_lane_keeping(self):
        raw = np.array([0.2, 0.1, 0.7])
        for lam in (0.3, 0.5, 1.0):
            for latched, v_y in ((False, 0.0), (True, 0.0), (False, 1.0)):
                mixed, info = self._mix(raw, v_y=v_y, latched=latched, lam=lam)
                g = info["o_v"]
                self.assertIn(g, (0.0, 1.0))
                self.assertAlmostEqual(mixed[2], 0.7 * ((1.0 - lam) + lam * g))
                self.assertAlmostEqual(mixed[0], 0.2 + 0.7 - mixed[2])
                self.assertAlmostEqual(mixed[1], 0.1, msg="away-direction intention untouched")
                self.assertAlmostEqual(float(np.sum(mixed)), 1.0)
        np.testing.assert_allclose(raw, [0.2, 0.1, 0.7], err_msg="the input is not modified")

    def test_without_latgate_the_ramp_is_bit_identical_to_the_previous_function(self):
        def ramp(raw_probs, relation, context_trust, lateral_offset_m):
            # mix_cutin_probability as committed before latgate (2026-10-19), verbatim.
            probs = np.asarray(raw_probs, dtype=float).copy()
            cutin_idx = {REL_LEFT_ADJACENT: 2, REL_RIGHT_ADJACENT: 1}.get(relation)
            if cutin_idx is None or lateral_offset_m is None or probs.size < 3:
                return probs
            o_v = lateral_observation(lateral_offset_m)
            p_raw = float(probs[cutin_idx])
            p_mixed = p_raw * ((1.0 - float(context_trust)) + float(context_trust) * o_v)
            probs[cutin_idx] = p_mixed
            probs[0] += p_raw - p_mixed
            return probs

        raws = [np.array([0.2, 0.1, 0.7]), np.array([0.6, 0.3, 0.1]), np.array([1 / 3, 1 / 3, 1 / 3]),
                np.array([0.01, 0.98, 0.01])]
        n = 0
        for relation in (REL_LEFT_ADJACENT, REL_RIGHT_ADJACENT, REL_EGO_LANE):
            for lam in (0.0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0):
                for y_v in list(np.linspace(-0.5, 1.5, 41)) + [None, CTX_LATERAL_NOISE_M, CTX_LATERAL_CROSSING_M]:
                    for raw in raws:
                        expected = ramp(raw, relation, lam, y_v)
                        # The gate inputs are ignored while the gate is off.
                        for kwargs in ({}, {"lateral_speed_mps": 1.0, "latched": True, "lateral_gate": False}):
                            mixed, info = mix_cutin_probability(raw, relation, lam, y_v, **kwargs)
                            np.testing.assert_array_equal(mixed, expected)
                            self.assertIsNone(info["trigger"])
                            self.assertIsNone(info["latched"])
                            n += 1
        self.assertGreater(n, 5000)

    def test_the_gate_reaches_the_processed_prediction(self):
        raw = TestContextTrustMixing._raw([0.05, 0.05, 0.9])
        process = TestContextTrustMixing()._process
        common = dict(cutin_probability_threshold=0.1, cutin_chance_ref=0.6, context_trust=1.0,
                      lateral_offset_m=self.Y_ON, lateral_gate=True)
        unlatched = process(raw, lateral_speed_mps=0.0, **common)
        self.assertIsNone(TestContextTrustMixing._mode(unlatched, "cutin"))
        self.assertAlmostEqual(unlatched.raw_mode_prob["LK"], 0.95)
        info = unlatched.branch_info["context_trust"]
        self.assertEqual((info["v_y"], info["trigger"], info["latched"], info["o_v"]), (0.0, False, False, 0.0))
        triggered = process(raw, lateral_speed_mps=0.5, **common)
        self.assertAlmostEqual(TestContextTrustMixing._mode(triggered, "cutin").probability, 0.9)
        self.assertTrue(triggered.branch_info["context_trust"]["trigger"])
        held = process(raw, lateral_speed_mps=0.0, lateral_latched=True, **common)
        self.assertAlmostEqual(TestContextTrustMixing._mode(held, "cutin").probability, 0.9)
        info = held.branch_info["context_trust"]
        self.assertEqual((info["trigger"], info["latched"]), (False, True))

    def test_the_agent_reads_latgate_as_a_whole_token_that_needs_ctx(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent
        agent = ACCNairSMPCAgent
        base = TestContextTrustToken.BASE
        self.assertTrue(agent._parse_lateral_gate(base + "_ctx1_latgate_rmove1000"))
        self.assertTrue(agent._parse_lateral_gate(base + "_ctx0.5_latgate"))
        self.assertFalse(agent._parse_lateral_gate(base + "_ctx1_rmove1000"))
        self.assertFalse(agent._parse_lateral_gate(base + "_ctx1_latgate2_rmove1000"))
        self.assertFalse(agent._parse_lateral_gate(base + "_ctx1_xlatgate_rmove1000"))
        self.assertFalse(agent._parse_lateral_gate(base + "_ctx1_latgatex_rmove1000"))
        # Every other parser reads the same with and without the token.
        plain, with_gate = base + "_ctx0.5_rmove1000", base + "_ctx0.5_latgate_rmove1000"
        for name in dir(agent):
            if name.startswith("_parse_") and name not in ("_parse_lateral_gate", "_parse_controller_num_modes"):
                self.assertEqual(getattr(agent, name)(plain), getattr(agent, name)(with_gate), name)

        class _Vehicle:  # the check runs before anything else touches the vehicle
            def get_world(self):
                return None

        with self.assertRaisesRegex(ValueError, "ctx<lambda>"):
            ACCNairSMPCAgent(_Vehicle(), None, smpc_config=base + "_latgate_rmove1000")
        with self.assertRaisesRegex(ValueError, "stdan_3int"):
            ACCNairSMPCAgent(_Vehicle(), None,
                             smpc_config="acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance_ctx1_latgate_rmove1000")

    def test_a_lane_flip_does_not_create_lateral_speed(self):
        from policies.acc_nair_smpc_agent import ACCNairSMPCAgent as agent

        class _Vec:
            def __init__(self, x, y):
                self.x, self.y = x, y

        class _Waypoint:
            def __init__(self, x, y, yaw_deg):
                self.transform = type("Transform", (), {})()
                self.transform.location = _Vec(x, y)
                self.transform.rotation = type("Rotation", (), {})()
                self.transform.rotation.yaw = yaw_deg

        # CARLA frame: a lane heading +y (yaw 90 deg) has the driver's right at -x.
        lane_a, lane_b = _Waypoint(10.0, 0.0, 90.0), _Waypoint(6.5, 0.0, 90.0)   # 3.5 m apart
        dt = 0.05
        # Drifting right at 0.3 m/s while driving at 10 m/s; the lane assignment
        # flips from lane_a to lane_b halfway through the window.
        samples, offsets = [], []
        for k in range(12):
            t = 100.0 + k * dt
            wp = lane_a if k < 6 else lane_b
            loc = _Vec(10.0 - 0.3 * k * dt, 10.0 * k * dt)
            samples.append((t, agent._lateral_speed_left(_Vec(-0.3, 10.0), wp)))
            offsets.append(agent._lateral_offset_toward_ego_lane(loc, wp, REL_LEFT_ADJACENT))
        self.assertTrue(all(abs(v + 0.3) < 1e-9 for _, v in samples), "left speed is -0.3 at every tick")
        # A left-adjacent vehicle going right moves toward the ego lane: +0.3 m/s ...
        v_y = agent._lateral_speed_toward_ego_lane(samples, samples[-1][0], REL_LEFT_ADJACENT)
        self.assertAlmostEqual(v_y, 0.3)
        self.assertAlmostEqual(agent._lateral_speed_toward_ego_lane(samples, samples[-1][0], REL_RIGHT_ADJACENT), -0.3)
        # ... while the offset difference across the flip would read 3.5 m / 0.05 s = 70 m/s.
        jumps = np.diff(offsets) / dt
        self.assertGreater(np.abs(jumps).max(), 60.0)
        # Only the samples in (t - window, t] count; fewer than the minimum history gives None.
        n_min = int(round(CTX_GATE_VY_MIN_HISTORY_S / dt)) + 1
        self.assertIsNone(agent._lateral_speed_toward_ego_lane(samples[:n_min - 1], samples[n_min - 2][0],
                                                               REL_LEFT_ADJACENT))
        self.assertAlmostEqual(agent._lateral_speed_toward_ego_lane(samples[:n_min], samples[n_min - 1][0],
                                                                    REL_LEFT_ADJACENT), 0.3)
        # Stale samples outside the window are ignored, even if they are the only ones.
        later = samples[-1][0] + CTX_GATE_VY_WINDOW_S + 1.0
        self.assertIsNone(agent._lateral_speed_toward_ego_lane(samples, later, REL_LEFT_ADJACENT))
        # A speed-up lateral burst averaged over the window.
        burst = [(100.0 + k * dt, -1.0 if k >= 6 else 0.0) for k in range(12)]
        self.assertAlmostEqual(agent._lateral_speed_toward_ego_lane(burst, burst[-1][0], REL_LEFT_ADJACENT),
                               6.0 / 10.0)


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
