import json
import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_LEFT_ADJACENT,
    REL_RIGHT_ADJACENT,
    calibrate_intention_probabilities,
    calibrated_lane_change_probability,
    load_probability_calibration,
    restricted_cubic_spline,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


BASE = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        "_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6")
MAPS = {"stdan": os.path.join(ROOT, "predictor", "stdan_3int_signed_tcross_velint", "pcal_ngsim_20261105.json"),
        "iaimm": os.path.join(ROOT, "predictor", "iaimm_kf", "pcal_ngsim_20261105.json")}
DESIGN = {"p_clip": [1e-4, 1 - 1e-4], "o_clip": [-2.0, 2.5], "v_clip": [-1.5, 2.0],
          "z_knots": [-4.6, -2.94, -1.39, 0.0, 1.39, 2.94, 4.6],
          "o_knots": [-0.6, -0.2, 0.2, 0.6, 1.0, 1.4], "v_knots": [-0.4, -0.1, 0.1, 0.3, 0.6, 1.0]}
N_TERMS = 6 + 5 + 5 + 6     # splines of z, o, v; toward, away and the four z products


def side_map(z=0.0, o=0.0, v=0.0, toward=0.0, b=0.0):
    """A map whose linear predictor is z * logit p + o * offset + v * speed + toward * moved_toward + b."""
    coef = np.zeros(N_TERMS)
    coef[0], coef[6], coef[11], coef[16] = z, o, v, toward
    return dict(DESIGN, mu=np.zeros(N_TERMS), sd=np.ones(N_TERMS), coef=coef, b=b)


def logit(p):
    return np.log(p / (1 - p))


class TestRestrictedCubicSpline(unittest.TestCase):

    def test_linear_beyond_the_outer_knots_and_smooth_at_the_knots(self):
        knots = DESIGN["z_knots"]
        f = lambda x: restricted_cubic_spline(x, knots)  # noqa: E731
        self.assertEqual(f(1.7)[0], 1.7)
        for x in (-8.0, 7.0):          # second difference 0 outside [t_1, t_K]
            np.testing.assert_allclose(f(x - 0.5) - 2 * f(x) + f(x + 0.5), 0.0, atol=1e-12)
        h = 1e-6
        for t in knots:                # continuous value and slope across every knot
            np.testing.assert_allclose(f(t - h), f(t + h), atol=1e-5)
            np.testing.assert_allclose((f(t) - f(t - h)) / h, (f(t + h) - f(t)) / h, atol=1e-4)


class TestCalibrateIntentionProbabilities(unittest.TestCase):

    STATE = dict(offset_m=0.0, lateral_speed_mps=0.0, lane_change=0, has_left=True, has_right=True)

    def test_identity_map_and_lane_keeping_takes_the_rest(self):
        cal = {"LLC": side_map(z=1.0), "RLC": side_map(z=1.0)}
        out = calibrate_intention_probabilities([0.5, 0.3, 0.2], cal, **self.STATE)
        np.testing.assert_allclose(out, [0.5, 0.3, 0.2], atol=1e-12)

    def test_no_lane_on_a_side_gives_zero(self):
        cal = {"LLC": side_map(z=1.0), "RLC": side_map(z=1.0)}
        out = calibrate_intention_probabilities([0.2, 0.6, 0.2], cal, **dict(self.STATE, has_left=False))
        np.testing.assert_allclose(out, [0.8, 0.0, 0.2], atol=1e-12)

    def test_inputs_are_signed_toward_each_side(self):
        # the offset term raises a side's probability when the vehicle sits toward that side
        cal = {"LLC": side_map(o=4.0, b=-2.0), "RLC": side_map(o=4.0, b=-2.0)}
        right = calibrate_intention_probabilities([0.4, 0.3, 0.3], cal, **dict(self.STATE, offset_m=1.0))
        left = calibrate_intention_probabilities([0.4, 0.3, 0.3], cal, **dict(self.STATE, offset_m=-1.0))
        self.assertGreater(right[2], 0.8)
        self.assertLess(right[1], 0.01)
        np.testing.assert_allclose(left[[0, 2, 1]], right, atol=1e-12)
        # a net lane change to the right is "toward" for RLC, "away" for LLC
        cal = {"LLC": side_map(toward=5.0, b=-3.0), "RLC": side_map(toward=5.0, b=-3.0)}
        moved = calibrate_intention_probabilities([0.4, 0.3, 0.3], cal, **dict(self.STATE, lane_change=1))
        self.assertAlmostEqual(moved[2], 1 / (1 + np.exp(-2.0)), places=12)
        self.assertAlmostEqual(moved[1], 1 / (1 + np.exp(3.0)), places=12)

    def test_unknown_speed_and_offset_read_as_zero(self):
        cal = {"LLC": side_map(z=1.0, o=1.0, v=1.0), "RLC": side_map(z=1.0, o=1.0, v=1.0)}
        a = calibrate_intention_probabilities([0.5, 0.3, 0.2], cal, **dict(self.STATE, lateral_speed_mps=None,
                                                                            offset_m=float("nan")))
        np.testing.assert_allclose(a, [0.5, 0.3, 0.2], atol=1e-12)

    def test_a_sum_above_one_is_scaled(self):
        cal = {"LLC": side_map(b=5.0), "RLC": side_map(b=5.0)}
        out = calibrate_intention_probabilities([0.4, 0.3, 0.3], cal, **self.STATE)
        np.testing.assert_allclose(out, [0.0, 0.5, 0.5], atol=1e-12)


@unittest.skipUnless(all(os.path.exists(p) for p in MAPS.values()), "the shipped maps are not in the tree")
class TestShippedMaps(unittest.TestCase):
    """The maps of 2026-11-05: every fold loads, reproduces the fit's own predictions, and is monotone in p."""

    def test_reproduces_the_fit(self):
        for name, path in MAPS.items():
            data = json.loads(open(path).read())
            for row in data["check_rows"]:
                cal = load_probability_calibration(path, recording=row["fold"])
                q = calibrated_lane_change_probability(row["p"], row["o"], row["v"], row["toward"], row["away"],
                                                       cal[row["side"]])
                self.assertAlmostEqual(q, row["q"], places=12, msg="%s %s" % (name, row))

    def test_monotone_in_the_predictor_probability(self):
        p = np.r_[0.001, np.linspace(0.01, 0.99, 99), 0.999]
        for name, path in MAPS.items():
            for fold in (None, 1, 2, 3, 4, 5, 6):
                cal = load_probability_calibration(path, recording=fold)
                for side in ("LLC", "RLC"):
                    for o in (-0.5, 0.0, 0.5, 1.0, 1.5):
                        for v in (-0.3, 0.0, 0.2, 0.5, 1.0):
                            for toward, away in ((0, 0), (1, 0), (0, 1)):
                                q = [calibrated_lane_change_probability(x, o, v, toward, away, cal[side]) for x in p]
                                self.assertTrue(np.all(np.diff(q) >= -1e-9),
                                                (name, fold, side, o, v, toward, away))


class TestPredictAccHook(unittest.TestCase):
    """``predict_acc(intention_calibration=)`` replaces the probabilities every step after the model reads."""

    @staticmethod
    def adapter():
        adapter = STDAN3IntACCAdapter(load_model=False)
        n = adapter.out_length
        s = 20.0 + 10.0 * adapter.dt * np.arange(1, n + 1)
        lateral = [np.full(n, 3.5), np.full(n, 7.0), np.linspace(3.5, 0.0, n)]   # LK, LLC away, RLC into the lane
        raw = {"vehicle_id": 7, "raw_traj_xy": np.stack([np.column_stack((s, d)) for d in lateral]),
               "raw_pred_vel": np.tile([10.0, 0.0], (3, n, 1)), "raw_intention_prob": np.array([0.5, 0.1, 0.4]),
               "signed_t_cross": 1.0, "valid_mask": np.ones((3, n), dtype=bool)}
        adapter.predict_raw = mock.MagicMock(side_effect=lambda ids, trackings, model_yaw=None: {
            7: {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in raw.items()}})
        return adapter

    @staticmethod
    def predict(adapter, **kwargs):
        from test_stdan_3int_acc_integration import StraightFrenetHandler
        return adapter.predict_acc(
            ego_state=np.array([0.0, 10.0]), target_states_frenet={7: np.array([20.0, 3.5, 10.0])},
            target_relations={7: REL_LEFT_ADJACENT}, trackings={}, frenet_handler=StraightFrenetHandler(),
            horizon=4, desired_speed=20.0, num_modes=8, controller_dt=0.2, cutin_probability_threshold=0.1,
            **kwargs)["processed_targets"][0]

    def test_calibrated_probabilities_reach_the_modes(self):
        plain = self.predict(self.adapter())
        self.assertAlmostEqual(plain.acc_mode_prob["cutin"], 0.4)
        self.assertNotIn("raw_intention_prob_uncal", plain.raw_prediction)
        calls = []

        def calibrate(target_id, probs):
            calls.append((target_id, np.array(probs)))
            return np.array([0.75, 0.05, 0.2])

        target = self.predict(self.adapter(), intention_calibration=calibrate)
        self.assertEqual(calls[0][0], 7)
        np.testing.assert_allclose(calls[0][1], [0.5, 0.1, 0.4])
        self.assertAlmostEqual(target.acc_mode_prob["cutin"], 0.2)
        np.testing.assert_allclose(target.raw_prediction["raw_intention_prob_uncal"], [0.5, 0.1, 0.4])


class TestToken(unittest.TestCase):

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_probability_calibration
        self.assertTrue(parse(BASE + "_pcal"))
        self.assertTrue(parse(BASE + "_pcal_safelayer"))
        self.assertFalse(parse(BASE))
        self.assertFalse(parse("acc_x_pcalx"))
        self.assertFalse(parse("acc_x_xpcal"))

    def test_the_carla_agent_refuses_it(self):
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=BASE + "_pcal")
        self.assertIn("NGSIM", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
