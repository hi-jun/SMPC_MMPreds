import json
import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_LEFT_ADJACENT,
    bcts_intention_probabilities,
    load_bcts_calibration,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


BASE = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
        "_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6")
FILES = {"stdan": os.path.join(ROOT, "predictor", "stdan_3int_signed_tcross_velint", "bcts_ngsim_20261106.json"),
         "iaimm": os.path.join(ROOT, "predictor", "iaimm_kf", "bcts_ngsim_20261106.json")}


def cal(T=1.0, b_left=0.0, b_right=0.0):
    return {"T": T, "b_left": b_left, "b_right": b_right}


class TestBCTS(unittest.TestCase):

    def test_identity(self):
        q = bcts_intention_probabilities([0.5, 0.1, 0.4], cal())
        np.testing.assert_allclose(q, [0.5, 0.1, 0.4], atol=1e-15)

    def test_a_bias_scales_the_odds_against_lane_keeping(self):
        p = np.array([0.6, 0.3, 0.1])
        q = bcts_intention_probabilities(p, cal(b_left=np.log(0.05), b_right=np.log(0.2)))
        self.assertAlmostEqual(q[1] / q[0], 0.05 * p[1] / p[0], places=12)
        self.assertAlmostEqual(q[2] / q[0], 0.2 * p[2] / p[0], places=12)
        self.assertAlmostEqual(q.sum(), 1.0, places=15)

    def test_the_temperature_flattens(self):
        p = np.array([0.81, 0.18, 0.01])
        q = bcts_intention_probabilities(p, cal(T=2.0))
        np.testing.assert_allclose(q, np.sqrt(p) / np.sqrt(p).sum(), atol=1e-15)

    def test_clipped_and_normalised_input(self):
        q = bcts_intention_probabilities([2.0, 0.0, 2.0], cal(T=1.05, b_left=-3.0, b_right=-3.85))
        self.assertTrue(np.all(np.isfinite(q)) and q[1] > 0.0)
        self.assertAlmostEqual(q.sum(), 1.0, places=15)

    def test_monotone_in_the_predictor_probability(self):
        c = cal(T=1.05, b_left=-3.0, b_right=-3.85)
        for p_right in (0.0, 0.1, 0.4):
            q = [bcts_intention_probabilities([1.0 - p - p_right, p, p_right], c)[1]
                 for p in np.linspace(0.0, 1.0 - p_right, 51)]
            self.assertTrue(np.all(np.diff(q) > 0.0), p_right)


@unittest.skipUnless(all(os.path.exists(p) for p in FILES.values()), "the shipped files are not in the tree")
class TestShippedFiles(unittest.TestCase):
    """The files of 2026-11-06: every fold loads and reproduces the fit's own predictions."""

    def test_reproduces_the_fit(self):
        for name, path in FILES.items():
            with open(path) as f:
                data = json.load(f)
            for row in data["check_rows"]:
                q = bcts_intention_probabilities(row["raw"], load_bcts_calibration(path, recording=row["fold"]))
                np.testing.assert_allclose(q, row["q"], rtol=0.0, atol=1e-12, err_msg="%s %s" % (name, row))

    def test_every_fold_and_the_checkpoint(self):
        for name, path in FILES.items():
            for fold in (None, 1, 2, 3, 4, 5, 6):
                c = load_bcts_calibration(path, recording=fold)
                self.assertEqual(sorted(c), ["T", "b_left", "b_right"])
                self.assertTrue(0.5 < c["T"] < 2.0 and c["b_left"] < 0.0 and c["b_right"] < 0.0, (name, fold, c))
        with open(FILES["stdan"]) as f:
            ckpt = json.load(f)["meta"]["checkpoint"]
        self.assertTrue(ckpt.endswith("checkpoints_out30_no_tcross/best_model.pt"), ckpt)


class TestPredictAccHook(unittest.TestCase):

    def test_bcts_reaches_the_modes(self):
        from test_pcal import TestPredictAccHook as Hook       # its scripted one-target adapter
        c = cal(T=1.05, b_left=-3.0, b_right=-3.85)
        q = bcts_intention_probabilities([0.5, 0.1, 0.4], c)
        target = Hook.predict(Hook.adapter(),
                              intention_calibration=lambda tid, probs: bcts_intention_probabilities(probs, c))
        self.assertEqual(target.relation_to_ego_lane, REL_LEFT_ADJACENT)
        self.assertAlmostEqual(target.acc_mode_prob["cutin"], q[2])     # left-adjacent: the RLC mode cuts in
        np.testing.assert_allclose(target.raw_prediction["raw_intention_prob_uncal"], [0.5, 0.1, 0.4])


class TestToken(unittest.TestCase):

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_bcts_calibration
        self.assertTrue(parse(BASE + "_bcts"))
        self.assertTrue(parse(BASE + "_bcts_occin0.05"))
        self.assertFalse(parse(BASE))
        self.assertFalse(parse(BASE + "_pcal"))
        self.assertFalse(parse("acc_x_bctsx"))
        self.assertFalse(parse("acc_x_xbcts"))

    def test_the_carla_agent_refuses_it(self):
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=BASE + "_bcts")
        self.assertIn("bcts", str(caught.exception))
        self.assertIn("NGSIM", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
