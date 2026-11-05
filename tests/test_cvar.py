import os
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from utils.acc_nair_smpc import (  # noqa: E402
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    confidence_quantile,
    original_nair_acc_config,
    safety_std,
    shortfall_cvar,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


class TestCutinCvar(unittest.TestCase):
    """``cvar<eps>_<c>``: the dopt price on the cut-in chance cells' shortfall replaced by the per-step bound
    ``CVaR_eps(short_k) <= c`` over the scenarios (Rockafellar-Uryasev rows with their own l1 slack)."""

    HORIZON = 15
    DT = 0.2
    COMMAND_DT = 0.1
    COV = np.diag([0.25, 0.09])
    BASE = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_dopt100lkr_trajcutin"
            "_ctx0_latgate_entrypass0_nocutout_lksigma_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6")

    def _config(self, cvar=None, **overrides):
        """The current design's QP (dopt100, jinput, r_a 0, l1slack1e6; fullfoot is the prediction's factor 1)."""
        config = original_nair_acc_config(
            horizon=self.HORIZON, dt=self.DT, desired_speed=20.0, num_modes=2, controller_variant=VARIANT_FIXED_RISK)
        config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
        config.r_move = 1000.0
        config.r_jerk = 40.0
        config.r_a = 0.0
        config.standoff_shortfall_weight = 100.0
        config.jerk_input = True
        config.slack_l1_weight = 1.0e6
        if cvar is not None:
            config.cutin_cvar_risk, config.cutin_cvar_bound = cvar
        for name, value in overrides.items():
            setattr(config, name, value)
        config.__post_init__()
        return config

    def _prediction(self, probabilities, gaps, lead_v=15.0, k_on=10, lk=None):
        """One scenario per entry of ``probabilities``; a scenario with a gap is a cut-in chance cell (factor 1,
        beta = min(P, 0.6)) ``gap`` ahead at the same speed, active from ``k_on``; ``None`` = no lead.  ``lk``:
        (scenario, gap) of an ego-lane lane-keeping lead (hard standoff, priced with lk) active at every step."""
        steps = self.HORIZON + 1
        n = len(probabilities)
        means = np.zeros((n, steps, 2))
        means[:, :, 1] = lead_v
        active = np.zeros((n, steps), dtype=bool)
        confidence = np.full((n, steps), np.nan)
        lead_lk = np.zeros((n, steps), dtype=bool)
        ids = np.full((n, steps), -1)
        for j, gap in enumerate(gaps):
            if gap is None:
                continue
            means[j, :, 0] = gap + lead_v * self.DT * np.arange(steps)
            active[j, k_on:] = True
            confidence[j] = min(float(probabilities[j]), 0.6)
            ids[j] = 11 + j
        if lk is not None:
            j, gap = lk
            means[j, :, 0] = gap + lead_v * self.DT * np.arange(steps)
            active[j] = True
            lead_lk[j] = True
            ids[j] = 10
        return MultimodalLeadPrediction(
            means=means, probabilities=np.asarray(probabilities, dtype=float),
            covariances=np.tile(self.COV, (n, steps, 1, 1)), active_mask=active, chance_confidence=confidence,
            lead_mode_lk=lead_lk, lead_vehicle_ids=ids, clearance_scale=np.ones((n, steps)),
            mode_names=["%d:mode%d" % (10 + j, j) for j in range(n)])

    def _solve(self, config, prediction, ego_v=15.0, floor_memory=True):
        controller = NairACCSMPC(config)
        if floor_memory:
            # the reference asks for the floor on the cut-in cells, so that only the rows (not the tracking terms)
            # hold the ego back from them
            controller.reference_adapter.previous_planned_standoff = {
                name: np.full(self.HORIZON + 1, 7.5) for name in prediction.mode_names}
        solution = controller.solve(np.array([0.0, ego_v]), prediction, previous_accel_cmd=0.0,
                                    command_dt=self.COMMAND_DT, ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        return controller, solution

    def _needed(self, config, prediction, solution, mode, beta):
        """max(0, d_safe(v_k) + q(beta) sigma - gap_k) on the scenario's active cells (the shortfall the plan needs)."""
        gap = prediction.means[mode, :, 0] - solution.x_nominal[mode, :, 0]
        full = 7.5 + 1.3 * solution.x_nominal[mode, :, 1] + confidence_quantile(beta) * safety_std(self.COV, config)
        return np.maximum(0.0, full - gap)[prediction.active_mask[mode]]

    def test_token_parse(self):
        parse = ACCNairSMPCAgent._parse_cutin_cvar
        self.assertEqual(parse(self.BASE + "_cvar0.5_2"), (0.5, 2.0))
        self.assertEqual(parse(self.BASE + "_occin0.1_cvar0.7_4_safelayer"), (0.7, 4.0))
        self.assertEqual(parse("acc_x_cvar.5_1.5"), (0.5, 1.5))
        self.assertIsNone(parse(self.BASE))
        self.assertIsNone(parse("acc_x_cvar0.5"))           # both parts
        self.assertIsNone(parse("acc_x_xcvar0.5_2"))
        self.assertIsNone(parse("acc_x_cvar0.5_2x_ra0"))

    def test_config_validation(self):
        for cvar in ((0.0, 2.0), (1.5, 2.0), (0.5, -1.0)):
            with self.assertRaises(ValueError):
                self._config(cvar)
        with self.assertRaises(ValueError):        # both fields together
            self._config(cutin_cvar_risk=0.5)
        with self.assertRaises(ValueError):        # needs dopt<w>
            self._config((0.5, 2.0), standoff_shortfall_weight=None)
        with self.assertRaises(ValueError):        # not with relbrake / floorrisk / passmiqp
            self._config((0.5, 2.0), relative_brake_decel=3.0)
        with self.assertRaises(ValueError):
            self._config((0.5, 2.0), floor_risk=0.2)
        with self.assertRaises(ValueError):
            self._config((0.5, 2.0), pass_miqp_margin=0.0)
        with self.assertRaises(ValueError):        # cached feedback QP only
            self._config((0.5, 2.0), controller_variant=VARIANT_MULTIMODAL_OL)
        self._config((1.0, 0.0))                   # eps 1 (the expectation) and c 0 are valid

    def test_agent_needs_fullfoot(self):
        vehicle = mock.MagicMock()
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(vehicle, None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=self.BASE.replace("_fullfoot", "") + "_cvar0.5_2")
        self.assertIn("fullfoot", str(caught.exception))

    def test_exact_cvar(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            n = int(rng.integers(1, 6))
            p = rng.dirichlet(np.ones(n + 1))[:n]           # mass left out: shortfall 0
            x = rng.uniform(0.0, 5.0, size=(n, 3))
            cells = rng.random((n, 3)) < 0.7
            eps = float(rng.uniform(0.05, 1.0))
            got = shortfall_cvar(x, p, cells, eps)
            for k in range(3):
                # brute force: the mean of the upper eps tail of the discrete distribution
                values = np.append(np.where(cells[:, k], x[:, k], 0.0), 0.0)
                mass = np.append(p, 1.0 - p.sum())
                order = np.argsort(-values)
                left, total = eps, 0.0
                for i in order:
                    take = min(left, mass[i])
                    total += take * values[i]
                    left -= take
                self.assertAlmostEqual(got[k], total / eps, places=9)
        # a lone shortfall x with probability P: x * min(1, P / eps)
        np.testing.assert_allclose(shortfall_cvar(np.array([[0.0], [4.0]]), [0.8, 0.2], [[False], [True]], 0.5),
                                   [4.0 * 0.2 / 0.5])

    def test_allowed_shortfall_closed_form(self):
        # a lone hypothesis 22 m ahead at the ego's speed, entering at 2 s: d_safe + q sigma is ~27.3 m, so without
        # braking the plan would need ~5.3 m of shortfall; the bound lets it give up c * max(1, eps / P)
        for eps, bound in ((0.5, 2.0), (0.7, 1.0)):
            for p in (0.8, 0.5, 0.4, 0.2):
                config = self._config((eps, bound))
                prediction = self._prediction([1.0 - p, p], [None, 22.0])
                _, solution = self._solve(config, prediction)
                needed = self._needed(config, prediction, solution, 1, min(p, 0.6))
                self.assertAlmostEqual(needed.max(), bound * max(1.0, eps / p), delta=2e-3, msg=(eps, bound, p))
                self.assertAlmostEqual(solution.cvar_max, bound, delta=2e-3)
                self.assertGreaterEqual(solution.cvar_active_rows, 1)
                self.assertLess(solution.cvar_slack_max, 1e-6)
                self.assertLess(solution.slack_max, 1e-6)
                self.assertAlmostEqual(solution.cvar_short_max, needed.max(), delta=1e-6)
                self.assertEqual(solution.cvar_cells, int(prediction.active_mask[1].sum()))
                self.assertEqual(solution.cvar_rows, int(prediction.active_mask[1].sum()))

    def test_two_hypotheses_share_the_budget(self):
        # two hypotheses of 0.3 each at the same steps: alone each could give up 2 * 0.5 / 0.3 = 3.33 m, together
        # the upper 0.5 tail is all shortfall, so CVaR = the shortfall and each keeps it at 2 m
        config = self._config((0.5, 2.0))
        prediction = self._prediction([0.4, 0.3, 0.3], [None, 22.0, 22.0])
        _, solution = self._solve(config, prediction)
        for mode in (1, 2):
            self.assertAlmostEqual(self._needed(config, prediction, solution, mode, 0.3).max(), 2.0, delta=2e-3)
        self.assertLess(solution.cvar_slack_max, 1e-6)

    def test_no_price_on_the_cvar_cells_and_the_lk_price_kept(self):
        # with a bound that never binds the problem is the design's with no price on the cut-in chance cells:
        # pricep2 with a zero cell probability there gives the same objective (the lk cell keeps its price)
        prediction = self._prediction([0.5, 0.5], [None, 22.0], lk=(0, 26.0))
        prediction.lead_mode_probability = np.where(prediction.lead_mode_lk, 1.0, 0.0)
        _, design = self._solve(self._config(standoff_shortfall_on_lk=True), prediction)
        _, unpriced = self._solve(self._config(standoff_shortfall_on_lk=True, standoff_shortfall_cell_price=True),
                                  prediction)
        _, token = self._solve(self._config((0.5, 100.0), standoff_shortfall_on_lk=True), prediction)
        self.assertGreater(abs(design.action - unpriced.action), 1e-3)     # the price on the cut-in cell mattered
        self.assertAlmostEqual(token.action, unpriced.action, delta=1e-4)
        np.testing.assert_allclose(token.x_nominal, unpriced.x_nominal, atol=1e-3)
        self.assertEqual(token.cvar_active_rows, 0)

    def test_tick_without_a_cutin_cell_solves_the_design_qp(self):
        # only a lane-keeping lead: no CVaR cell, so the token's solve is the design's, bit for bit
        prediction = self._prediction([0.5, 0.5], [None, None], lk=(1, 24.0))
        for floor_memory in (False, True):
            controller_d, design = self._solve(self._config(), prediction, floor_memory=floor_memory)
            controller_t, token = self._solve(self._config((0.5, 2.0)), prediction, floor_memory=floor_memory)
            self.assertEqual(design.action, token.action)
            np.testing.assert_array_equal(design.x_nominal, token.x_nominal)
            self.assertEqual(token.cvar_cells, 0)
            self.assertTrue(np.isnan(token.cvar_max))
            self.assertFalse(controller_t._cvar_rows())

    def test_slack_instead_of_infeasibility(self):
        # 15 m ahead at the ego's speed from step 8: the jerk-limited ego cannot open the gap to d_safe - 2 m in
        # time, so the rows need their slack; the solve must not fall back
        config = self._config((0.5, 2.0))
        prediction = self._prediction([0.2, 0.8], [None, 15.0], k_on=8)
        _, solution = self._solve(config, prediction)
        self.assertGreater(max(solution.cvar_slack_max, solution.slack_max), 0.1)
        self.assertGreater(solution.cvar_max, 2.0)

    def test_needed_shortfall_feeds_the_remembered_standoff(self):
        # the remembered standoff on the cut-in cells is d_safe(v_k) - needed (the plan's own gap where it falls
        # short), not the solver's arbitrary value
        config = self._config((0.5, 2.0))
        prediction = self._prediction([0.6, 0.4], [None, 22.0])
        controller, solution = self._solve(config, prediction)
        cells = prediction.active_mask[1]
        gap = prediction.means[1, :, 0] - solution.x_nominal[1, :, 0]
        sigma_term = confidence_quantile(0.4) * safety_std(self.COV, config)
        expected = np.minimum(7.5 + 1.3 * solution.x_nominal[1, :, 1], gap - sigma_term)
        np.testing.assert_allclose(solution.planned_standoff[1, cells], expected[cells], atol=1e-6)
        np.testing.assert_allclose(controller.reference_adapter.previous_planned_standoff["11:mode1"][cells],
                                   expected[cells], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
