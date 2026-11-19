"""pzjoint (2026-11-19, student: "S의 π를 결합 확률로 바꾸고 문턱을 없앤 뒤 다시 돌리기"): with pzrisk<eps>, a lead cell
seen past ego-lane vehicles that have left the path (the next vehicle ahead S behind a departing preceding vehicle V)
takes the joint probability p_V * p_S as its lead-mode probability and leads iff that times its own Z exceeds eps; no
separate 0.1 probability gate is applied."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import REL_EGO_LANE  # noqa: E402
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402
import test_pzrisk as PZ  # noqa: E402   (the module, so its TestCases are not collected again here)

AGENT, DT, EPS, HORIZON, StraightRoad, raw_target = PZ.AGENT, PZ.DT, PZ.EPS, PZ.HORIZON, PZ.StraightRoad, PZ.raw_target
V, S = 5, 6


class TestPzJointParser(unittest.TestCase):

    def test_parser(self):
        self.assertTrue(ACCNairSMPCAgent._parse_pz_joint(AGENT + "_pzjoint"))
        self.assertFalse(ACCNairSMPCAgent._parse_pz_joint(AGENT))
        self.assertFalse(ACCNairSMPCAgent._parse_pz_joint(AGENT + "_pzjointx"))
        with self.assertRaises(ValueError):          # needs pzrisk<eps>
            ACCNairSMPCAgent._parse_pz_joint(AGENT.replace("_pzrisk0.1", "_zconf0.95") + "_pzjoint")


class TestPzJoint(unittest.TestCase):
    """V (the preceding vehicle, 20 m) with a cut-out hypothesis whose trajectory leaves the lane, S (the next vehicle
    ahead, 45 m) keeping its lane at lateral offset ``s_y``; both ego-lane targets."""

    HALF = PZ.TestPzRiskActivation.HALF

    def _predict(self, pz_joint, v_probs=(0.7, 0.3, 0.0), s_y=0.0, threshold=0.1):
        """``pz_joint`` None omits the argument (the adapter's default)."""
        adapter = STDAN3IntACCAdapter(load_model=False)
        n = adapter.out_length
        raws = {V: raw_target(V, list(v_probs), ((0.0, 0.0), (0.0, 3.5), (0.0, -3.5)), 20.0, 9.0, 0.4, 0.2,
                              (0.3, 0.1, -0.2), n),
                S: raw_target(S, [1.0, 0.0, 0.0], ((s_y, s_y), (s_y, s_y), (s_y, s_y)), 45.0, 6.0, 0.4, 0.2,
                              (0.0, 0.0, 0.0), n)}
        adapter.predict_raw = lambda target_ids, trackings, model_yaw=None: {tid: dict(raws[tid]) for tid in target_ids}
        relations = {V: REL_EGO_LANE, S: REL_EGO_LANE}
        membership, floor_lateral = PZ.TestPzRiskActivation._lane_fns(self, relations)
        extra = {} if pz_joint is None else {"pz_joint": pz_joint}
        result = adapter.predict_acc(
            ego_state=np.array([0.0, 10.0]), target_states_frenet={V: [20.0, 0.0, 9.0], S: [45.0, s_y, 6.0]},
            target_relations=relations, trackings={}, frenet_handler=StraightRoad(), horizon=HORIZON,
            desired_speed=20.0, num_modes=8, controller_dt=DT, lane_membership_fn=membership,
            cutin_probability_threshold=threshold, cutin_chance_ref=0.6, lk_entry_as_cutin=True, full_footprint=True,
            lead_merge=True, merge_first=True, lk_hold=True, no_lk_entry=True, ahead_only=True,
            floor_lateral_fn=floor_lateral, pz_risk=EPS, **extra)
        modes = {(t.vehicle_id, m.scenario_label): m for t in result["processed_targets"] for m in t.mode_predictions}
        return result["prediction"], modes

    def _expected_s_cells(self, modes, p_v):
        """Steps k >= 1 at which, in V's cut-out scenario, V has left (its cut-out cell is off) and S is on: S is the
        nearest candidate there; the joint rule switches it on iff p_V p_S Z_S(k) > eps."""
        v_out, s_lk = modes[(V, "cutout")], modes[(S, "lk")]
        steps = [k for k in range(1, HORIZON + 1) if not v_out.active_mask[k] and s_lk.active_mask[k]]
        on = {k: p_v * s_lk.probability * s_lk.floor_lateral[k, 0] > EPS for k in steps}
        return steps, on

    def _scenario(self, prediction, label):
        return [i for i, name in enumerate(prediction.mode_names) if label in name]

    def test_s_takes_the_joint_probability(self):
        prediction, modes = self._predict(True)
        p_v = modes[(V, "cutout")].probability
        steps, on = self._expected_s_cells(modes, p_v)
        self.assertTrue(steps and all(on.values()))       # S in its lane: Z near 1, p_V 0.3: on
        hit = 0
        for j in range(prediction.num_modes):
            for k in steps:
                if prediction.lead_vehicle_ids[j, k] == S:
                    hit += 1
                    self.assertAlmostEqual(prediction.lead_mode_probability[j, k], p_v * modes[(S, "lk")].probability)
        self.assertGreater(hit, 0)
        # every active cell satisfies the joint pi > eps
        pi = prediction.lead_mode_probability * prediction.floor_lateral[..., 0]
        self.assertTrue(np.all(pi[:, 1:][prediction.active_mask[:, 1:]] > EPS))

    def test_without_pzjoint_s_keeps_its_own_probability(self):
        prediction, modes = self._predict(False)
        cells = prediction.lead_vehicle_ids == S
        self.assertTrue(cells.any())
        np.testing.assert_allclose(prediction.lead_mode_probability[cells], modes[(S, "lk")].probability)

    def test_the_joint_rule_switches_s_off_where_the_gate_did_not(self):
        # S drifting to the lane edge (Z about 0.5) behind a departure of p_V 0.15: the 0.1 gate keeps S
        # (p_V >= 0.1), the joint rule switches it off (0.15 * 1 * Z <= 0.1)
        v_probs = (0.85, 0.15, 0.0)
        joint, modes = self._predict(True, v_probs=v_probs, s_y=1.75)
        steps, on = self._expected_s_cells(modes, modes[(V, "cutout")].probability)
        self.assertTrue(steps)
        old, _ = self._predict(False, v_probs=v_probs, s_y=1.75)
        for k in steps:
            self.assertEqual(any(joint.lead_vehicle_ids[:, k] == S), on[k])
        self.assertTrue(any(not v for v in on.values()))
        self.assertTrue(any((old.lead_vehicle_ids[:, k] == S).any() for k in steps if not on[k]))

    def test_an_unlikely_departure_never_hands_the_lane_to_s(self):
        # p_V 0.08 < eps: the joint p can never pass, as the 0.1 gate did before
        joint, modes = self._predict(True, v_probs=(0.92, 0.08, 0.0))
        self.assertFalse((joint.lead_vehicle_ids[:, 1:] == S).any())

    def test_no_separate_gate(self):
        # the ego-lane lead's lane keeping below 0.1 (p 0.05): the gate switched it off at every step, step 0 included;
        # with pzjoint it keeps its measured step 0 and pzrisk's p Z > eps (off) after it
        _, old = self._predict(False, v_probs=(0.05, 0.95, 0.0))
        _, joint = self._predict(True, v_probs=(0.05, 0.95, 0.0))
        self.assertFalse(np.asarray(old[(V, "lk")].active_mask).any())
        lk = np.asarray(joint[(V, "lk")].active_mask, dtype=bool)
        self.assertTrue(lk[0])
        self.assertFalse(lk[1:].any())

    def test_off_by_default(self):
        # pz_joint=False is the adapter's default: the same prediction as without the argument
        a, _ = self._predict(False)
        b, _ = self._predict(None)
        self.assertEqual(a.mode_names, b.mode_names)
        for name in ("active_mask", "lead_mode_probability", "lead_vehicle_ids", "means", "chance_confidence",
                     "clearance_scale", "floor_lateral"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_needs_pzrisk(self):
        adapter = STDAN3IntACCAdapter(load_model=False)
        with self.assertRaises(ValueError):
            adapter.predict_acc(ego_state=np.array([0.0, 10.0]), target_states_frenet={}, target_relations={},
                                trackings={}, frenet_handler=StraightRoad(), horizon=HORIZON, desired_speed=20.0,
                                num_modes=8, pz_joint=True)


if __name__ == "__main__":
    unittest.main()
