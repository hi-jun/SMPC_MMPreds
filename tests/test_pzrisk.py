import os
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.stats import norm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    process_vehicle_prediction,
)
from utils.acc_nair_smpc import (  # noqa: E402
    PZ_RISK_B_MIN,
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    one_row_pz_risk_terms,
    one_row_sigma_term,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


HORIZON = 15
DT = 0.2
TAU = 1.3
L_D0 = 4.5 + 3.0
EPS = 0.1
# the final design's string (onerow with its cap, mean-path entry, aheadonly, noegonoise) with zconf0.95 -> pzrisk0.1
AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
         "_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_leadmerge_mergefirst_lkhold_nolkentry_amin3.5_amax2"
         "_jlim2.5_jlimup5_onerow_pzrisk0.1_snap160_aheadonly_noegonoise")


def config_for(eps=EPS, num_modes=1, **overrides):
    """The final design's controller with pzrisk<eps> in place of zconf0.95."""
    config = original_nair_acc_config(horizon=HORIZON, dt=DT, desired_speed=20.0, num_modes=num_modes,
                                      controller_variant=VARIANT_FIXED_RISK)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move, config.r_jerk, config.r_a = 1000.0, 40.0, 0.0
    config.standoff_shortfall_weight = 10.0
    config.jerk_input = True
    config.snap_weight = 160.0
    config.slack_l1_weight = 1.0e6
    config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
    config.process_noise_cov = ((0.0, 0.0), (0.0, 0.0))
    config.one_row = True
    config.one_row_pz_risk = eps
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def expected_term(p, z, lower, upper, rho, sigma_s, eps=EPS):
    """The maintainer's formula written out independently: b = 1 - eps / (p Z), the truncated-normal moments of the
    lateral coordinate on [lower, upper], T = Phi^-1(b) sigma_c - rho sigma_s E_z."""
    b = max(1.0 - eps / (p * z), PZ_RISK_B_MIN)
    mass = norm.cdf(upper) - norm.cdf(lower)
    e_z = (norm.pdf(lower) - norm.pdf(upper)) / mass
    v_z = 1.0 + (lower * norm.pdf(lower) - upper * norm.pdf(upper)) / mass - e_z ** 2
    sigma_c = np.sqrt(rho ** 2 * sigma_s ** 2 * v_z + sigma_s ** 2 * (1.0 - rho ** 2))
    return norm.ppf(b) * sigma_c - rho * sigma_s * e_z


def lead_prediction(gap0, lead_v, floor_lateral=None, p=1.0, cov=np.diag([0.25, 0.09]), chance=False):
    steps = HORIZON + 1
    means = np.zeros((1, steps, 2))
    means[0, :, 0] = gap0 + lead_v * DT * np.arange(steps)
    means[0, :, 1] = lead_v
    return MultimodalLeadPrediction(
        means=means, probabilities=np.array([1.0]), covariances=np.tile(cov, (1, steps, 1, 1)),
        active_mask=np.ones((1, steps), dtype=bool),
        chance_confidence=np.full((1, steps), 0.6 if chance else np.nan),
        sigma_confidence=np.full((1, steps), np.nan), lead_mode_lk=np.full((1, steps), not chance),
        lead_vehicle_ids=np.full((1, steps), 10, dtype=int), clearance_scale=np.ones((1, steps)),
        lead_mode_probability=np.full((1, steps), p), mode_names=["10:lk"], floor_lateral=floor_lateral)


def lateral_stats(z, lower, upper, rho):
    steps = HORIZON + 1
    lateral = np.zeros((1, steps, 4))
    lateral[0, :, 0], lateral[0, :, 1], lateral[0, :, 2], lateral[0, :, 3] = z, lower, upper, rho
    return lateral


class TestPzRiskTokens(unittest.TestCase):

    def test_parser(self):
        parse = ACCNairSMPCAgent._parse_pz_risk
        self.assertEqual(parse(AGENT), 0.1)
        self.assertEqual(parse("onerow_pzrisk0.05_noegonoise"), 0.05)
        self.assertIsNone(parse(AGENT.replace("_pzrisk0.1", "")))
        self.assertIsNone(parse("acc_x_xpzrisk0.1_onerow_noegonoise"))
        for bad in ("acc_x_pzrisk_onerow_noegonoise", "acc_x_pzrisk0.1x_onerow_noegonoise"):
            with self.assertRaises(ValueError):          # a pzrisk token without a readable eps
                parse(bad)

    def test_parser_refusals(self):
        parse = ACCNairSMPCAgent._parse_pz_risk
        for extra in ("_zconf0.95", "_floorsigma", "_flooreps0.05", "_occin0.1", "_nocap"):
            with self.assertRaises(ValueError) as caught:
                parse(AGENT + extra)
            self.assertIn("not with", str(caught.exception))
        for missing in ("_onerow", "_noegonoise"):
            with self.assertRaises(ValueError) as caught:
                parse(AGENT.replace(missing, ""))
            self.assertIn("needs " + missing[1:], str(caught.exception))

    def test_config_validation(self):
        self.assertEqual(config_for().one_row_pz_risk, EPS)
        for overrides in (dict(one_row=False), dict(eps=0.0), dict(eps=1.0), dict(one_row_z_conf=0.95),
                          dict(one_row_floor_sigma=True, one_row_z_conf=0.95), dict(one_row_floor_eps=0.05),
                          dict(one_row_no_cap=True), dict(process_noise_cov=((0.01, 0.0), (0.0, 0.01)))):
            with self.assertRaises(ValueError):
                config_for(**overrides)

    def test_the_carla_agent_refuses_it(self):
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=AGENT)
        self.assertIn("NGSIM replay only", str(caught.exception))


class TestPzRiskTerms(unittest.TestCase):
    """``one_row_pz_risk_terms``: T = Phi^-1(1 - eps / (p Z)) sigma_c - rho sigma_s E_z per active cell."""

    def test_against_the_formula(self):
        config = config_for()
        steps = HORIZON + 1
        z = np.linspace(0.95, 0.3, steps)
        lower, upper, rho = np.linspace(-3.0, -0.4, steps), np.linspace(1.0, 2.5, steps), np.linspace(-0.6, 0.5, steps)
        p, sd = 0.7, 0.8
        prediction = lead_prediction(30.0, 10.0, lateral_stats(z, lower, upper, rho), p=p,
                                     cov=np.diag([sd ** 2, 0.09]), chance=True)
        terms, guarded = one_row_pz_risk_terms(prediction, config)
        for k in range(steps):
            self.assertAlmostEqual(terms[0, k], expected_term(p, z[k], lower[k], upper[k], rho[k], sd), places=10)
        self.assertFalse(guarded.any())
        # the controller's row term is the same, cell by cell
        controller = NairACCSMPC(config)
        for k in range(steps):
            self.assertEqual(controller._cell_tightening(prediction, None, 0, k), terms[0, k])

    def test_the_breach_probability_is_eps(self):
        # Monte Carlo of the claim: per cell P(hypothesis, overlap, s_lead < mu_s - T) = eps; with rho 0 the
        # conditional law is exactly Gaussian, with rho != 0 it is the moment-matched Gaussian (close, not exact)
        config = config_for()
        rng = np.random.default_rng(5)
        n = 1000000
        for p, rho, lower, upper, sd in ((0.6, 0.0, -1.2, 0.9, 1.1), (0.9, 0.5, -0.8, 1.6, 0.7)):
            z_mass = norm.cdf(upper) - norm.cdf(lower)
            prediction = lead_prediction(30.0, 10.0, lateral_stats(z_mass, lower, upper, rho), p=p,
                                         cov=np.diag([sd ** 2, 0.09]), chance=True)
            t = float(one_row_pz_risk_terms(prediction, config)[0][0, 5])
            u = rng.standard_normal(n)
            s = sd * (rho * u + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n))      # s_lead - mu_s
            hyp = rng.random(n) < p
            breach = hyp & (u > lower) & (u < upper) & (s < -t)
            self.assertAlmostEqual(float(np.mean(breach)), EPS, delta=0.002 if rho == 0.0 else 0.006)

    def test_the_lead_lane_keeping(self):
        # p 1, Z 1, rho 0: z = Phi^-1(1 - eps) = 1.2816 in place of zconf0.95's 1.645
        config = config_for()
        prediction = lead_prediction(30.0, 10.0, lateral_stats(1.0, -30.0, 30.0, 0.0), p=1.0)
        terms, _ = one_row_pz_risk_terms(prediction, config)
        np.testing.assert_allclose(terms, norm.ppf(0.9) * 0.5, atol=1.0e-12)
        self.assertAlmostEqual(norm.ppf(1.0 - EPS / 0.95), 1.2521, places=4)     # at p 0.95

    def test_the_guard(self):
        # pi at or below eps (step 0 of an unlikely ego-lane mode): b = PZ_RISK_B_MIN, T = Phi^-1(1e-6) sigma
        config = config_for()
        steps = HORIZON + 1
        z = np.full(steps, 1.0)
        z[3], z[4] = 0.05, EPS / 0.5 + 1.0e-9            # pi 0.025, and pi 1e-9 above eps
        prediction = lead_prediction(30.0, 10.0, lateral_stats(z, -6.0, 6.0, 0.0), p=0.5)
        with np.errstate(all="raise"):
            terms, guarded = one_row_pz_risk_terms(prediction, config)
        np.testing.assert_array_equal(np.flatnonzero(guarded[0]), [3, 4])
        self.assertAlmostEqual(terms[0, 3], norm.ppf(PZ_RISK_B_MIN) * 0.5, places=10)
        self.assertTrue(np.all(np.isfinite(terms)))
        prediction.active_mask[0, 10:] = False             # inactive cells: no term
        terms, guarded = one_row_pz_risk_terms(prediction, config)
        self.assertTrue(np.all(terms[0, 10:] == 0.0) and not guarded[0, 10:].any())

    def test_prediction_without_the_statistics(self):
        config = config_for()
        # the replay's no-target placeholder (no cut-in chance cell): pi 1 without conditioning
        prediction = lead_prediction(1000.0, 20.0)
        prediction.lead_mode_probability = None
        terms, _ = one_row_pz_risk_terms(prediction, config)
        np.testing.assert_allclose(terms, norm.ppf(0.9) * 0.5, atol=1.0e-12)
        with self.assertRaises(ValueError):                # a cut-in chance cell without them raises
            one_row_pz_risk_terms(lead_prediction(30.0, 10.0, chance=True), config)


class TestPzRiskSolve(unittest.TestCase):

    def test_the_capped_row(self):
        # one capped row: gap >= L + d0 + tau v - short + T - xi with 0 <= short <= tau v; 0.3 m inside the floor
        # L + d0 + T at step 0 the emergency slack takes the 0.3 m
        config = config_for()
        steps = HORIZON + 1
        z = np.full(steps, 0.8)
        lateral = lateral_stats(z, -2.0, 0.8, 0.3)
        lateral[0, 4:, 0] = 0.12                            # pi 0.108: b 0.074, a relaxed row
        prediction = lead_prediction(0.0, 10.0, lateral, p=0.9, chance=True)
        terms, guarded = one_row_pz_risk_terms(prediction, config)
        self.assertLess(terms[0, 6], terms[0, 2])
        prediction.means[0, :, 0] += L_D0 + float(terms[0, 0]) - 0.3
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, 10.0]), prediction, previous_accel_cmd=0.0, command_dt=0.1,
                                    ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.3, places=4)
        self.assertEqual((solution.pz_risk_cells, solution.pz_risk_guarded), (steps, 0))
        # the solved cells: slack = max(0, L + d0 + T - gap) through the cap, short <= tau v
        gap = prediction.means[0, :, 0] - solution.x_nominal[0, :, 0]
        np.testing.assert_allclose(solution.slack[0], np.maximum(0.0, L_D0 + terms[0] - gap), atol=2.0e-4)

    def test_memo_follows_the_prediction_object(self):
        config = config_for()
        controller = NairACCSMPC(config)
        a = lead_prediction(30.0, 10.0, lateral_stats(1.0, -30.0, 30.0, 0.0), p=1.0)
        b = lead_prediction(30.0, 10.0, lateral_stats(0.5, -30.0, 30.0, 0.0), p=1.0)
        ta = controller._cell_tightening(a, None, 0, 3)
        tb = controller._cell_tightening(b, None, 0, 3)
        self.assertAlmostEqual(ta, norm.ppf(0.9) * 0.5, places=12)
        self.assertAlmostEqual(tb, norm.ppf(0.8) * 0.5, places=12)

    def test_default_unchanged(self):
        # without the token the row's term is zconf's z sigma, and floor_lateral is not read
        config = config_for(eps=None, one_row_z_conf=0.95)
        self.assertIsNone(config.one_row_pz_risk)
        plain = lead_prediction(30.0, 10.0, p=1.0)
        with_stats = lead_prediction(30.0, 10.0, lateral_stats(0.2, -1.0, 1.0, 0.4), p=1.0)
        controller = NairACCSMPC(config)
        for k in (0, 5, HORIZON):
            self.assertEqual(controller._cell_tightening(with_stats, None, 0, k),
                             one_row_sigma_term(plain, 0, k, config))
        s_plain = NairACCSMPC(config).solve(np.array([0.0, 10.0]), plain, previous_accel_cmd=0.0, command_dt=0.1,
                                            ego_accel=0.0)
        s_stats = NairACCSMPC(config).solve(np.array([0.0, 10.0]), with_stats, previous_accel_cmd=0.0,
                                            command_dt=0.1, ego_accel=0.0)
        self.assertEqual(s_plain.action, s_stats.action)
        np.testing.assert_array_equal(s_plain.x_nominal, s_stats.x_nominal)
        self.assertEqual((s_stats.pz_risk_cells, s_stats.pz_risk_guarded), (0, 0))


def raw_target(vehicle_id, probs, y_paths, x0, speed, sd_x, sd_y, rho, out_length):
    """Three trajectories (LK, LLC, RLC) along global x at ``speed``, lateral position per trajectory ramping from
    ``y_paths[i][0]`` to ``y_paths[i][1]`` over 3 s, with a 2-D position covariance (global frame) per trajectory."""
    t = 0.1 * np.arange(1, out_length + 1)
    traj = np.zeros((3, out_length, 2))
    cov = np.zeros((3, out_length, 2, 2))
    for i, (y0, y1) in enumerate(y_paths):
        traj[i, :, 0] = x0 + speed * t
        traj[i, :, 1] = y0 + (y1 - y0) * np.clip(t / 3.0, 0.0, 1.0)
        sx, sy = sd_x * (1.0 + t), sd_y * (1.0 + 0.5 * t)
        cov[i, :, 0, 0], cov[i, :, 1, 1] = sx ** 2, sy ** 2
        cov[i, :, 0, 1] = cov[i, :, 1, 0] = rho[i] * sx * sy
    vel = np.zeros((3, out_length, 2))
    vel[:, :, 0] = speed
    return {"vehicle_id": vehicle_id, "raw_traj_xy": traj, "raw_pred_vel": vel,
            "raw_intention_prob": np.asarray(probs, dtype=float), "signed_t_cross": 1.0,
            "valid_mask": np.ones((3, out_length), dtype=bool), "mask_true_count": 1, "model_yaw": 0.0,
            "raw_position_cov_global": cov, "raw_velocity_cov_global": 0.01 * np.tile(np.eye(2), (3, out_length, 1, 1))}


class StraightRoad:
    """A straight Frenet reference along global x (heading 0), as the adapter reads it."""

    def __init__(self):
        s = np.linspace(-50.0, 200.0, 501)
        self.trajectory = np.column_stack((s, s, np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)))


class TestPzRiskActivation(unittest.TestCase):
    """The adapter and the postprocess: each mode's cells at steps >= 1 switched on by p Z > eps with its own
    trajectory's Z, nolkentry's trajectories kept out, step 0 the measured test, the cells carrying their own Z."""

    HALF = 1.8

    def _lane_fns(self, relations):
        def membership(target_id, xy):
            out = np.zeros((xy.shape[0], xy.shape[1] + 1), dtype=bool)
            out[:, 0] = relations[target_id] == REL_EGO_LANE
            out[:, 1:] = np.abs(xy[..., 1]) < self.HALF
            return out

        def floor_lateral(target_id, xy, cov):
            sd = np.sqrt(cov[..., 1, 1])
            probs = np.zeros((xy.shape[0], xy.shape[1] + 1))
            lower, upper = np.full(probs.shape, np.nan), np.full(probs.shape, np.nan)
            probs[:, 0] = float(relations[target_id] == REL_EGO_LANE)
            lower[:, 1:], upper[:, 1:] = (-self.HALF - xy[..., 1]) / sd, (self.HALF - xy[..., 1]) / sd
            probs[:, 1:] = norm.cdf(upper[:, 1:]) - norm.cdf(lower[:, 1:])
            return probs, lower, upper
        return membership, floor_lateral

    def _predict(self, pz_risk, with_floor=True, lead_probs=(0.95, 0.03, 0.02), cutin_probs=(0.55, 0.05, 0.4)):
        adapter = STDAN3IntACCAdapter(load_model=False)
        n = adapter.out_length
        raws = {
            # the ego-lane lead 30 m ahead: lane keeping (a wobble toward the lane edge), a left lane change out
            # (a cut-out) and a right one out
            5: raw_target(5, list(lead_probs), ((0.0, 1.9), (0.0, 3.5), (0.0, -3.5)), 30.0, 9.0, 0.4, 0.2,
                          (0.3, 0.1, -0.2), n),
            # a left-adjacent vehicle 12 m ahead: its right lane change is the cut-in, LK stays, LLC goes away
            7: raw_target(7, list(cutin_probs), ((3.5, 3.5), (3.5, 7.0), (3.5, 0.0)), 12.0, 10.0, 0.5, 0.25,
                          (0.2, 0.0, -0.4), n),
        }
        adapter.predict_raw = lambda target_ids, trackings, model_yaw=None: {
            tid: dict(raws[tid]) for tid in target_ids}
        relations = {5: REL_EGO_LANE, 7: REL_LEFT_ADJACENT}
        membership, floor_lateral = self._lane_fns(relations)
        result = adapter.predict_acc(
            ego_state=np.array([0.0, 10.0]), target_states_frenet={5: [30.0, 0.0, 9.0], 7: [12.0, 3.5, 10.0]},
            target_relations=relations, trackings={}, frenet_handler=StraightRoad(), horizon=HORIZON,
            desired_speed=20.0, num_modes=8, controller_dt=DT, lane_membership_fn=membership,
            cutin_probability_threshold=0.1, cutin_chance_ref=0.6, lk_entry_as_cutin=True, full_footprint=True,
            lead_merge=True, merge_first=True, lk_hold=True, no_lk_entry=True, ahead_only=True,
            floor_lateral_fn=floor_lateral if with_floor else None, pz_risk=pz_risk)
        return adapter, raws, floor_lateral, result

    def test_modes_switch_on_by_p_times_z(self):
        adapter, raws, floor_lateral, result = self._predict(EPS)
        for target in result["processed_targets"]:
            raw = raws[target.vehicle_id]
            xy = adapter._sample_future(raw["raw_traj_xy"], HORIZON, DT)
            cov = adapter._sample_future(raw["raw_position_cov_global"], HORIZON, DT)
            z = floor_lateral(target.vehicle_id, xy, cov)[0]
            p_raw = np.asarray(raw["raw_intention_prob"]) / np.sum(raw["raw_intention_prob"])
            for mode in target.mode_predictions:
                rep = max(mode.raw_mode_indices, key=lambda idx: p_raw[idx])
                active = np.asarray(mode.active_mask, dtype=bool)
                np.testing.assert_array_equal(mode.floor_lateral[:, 0], z[rep])        # its own Z, not P_lat
                if target.relation_to_ego_lane == REL_LEFT_ADJACENT and mode.mode_name == "lk":
                    self.assertFalse(active.any())                                    # nolkentry: never leads
                    continue
                np.testing.assert_array_equal(active[1:], mode.probability * z[rep, 1:] > EPS)
                self.assertEqual(active[0], target.relation_to_ego_lane == REL_EGO_LANE)     # measured at step 0
        modes = {(t.vehicle_id, m.scenario_label): m for t in result["processed_targets"] for m in t.mode_predictions}
        # the lead's cut-out hypothesis (p 0.05) never leads after step 0, its lane keeping (p 0.95) stays on through
        # the wobble past the lane edge (Z about 0.4 there), the cut-in (p 0.4) switches on before its mean enters
        self.assertTrue(modes[(5, "lk")].active_mask.all())
        self.assertFalse(modes[(5, "cutout")].active_mask[1:].any())
        cutin = np.asarray(modes[(7, "cutin")].active_mask, dtype=bool)
        _, _, _, plain = self._predict(None)
        plain_cutin = [m for t in plain["processed_targets"] for m in t.mode_predictions
                       if (t.vehicle_id, m.scenario_label) == (7, "cutin")][0]
        self.assertLess(np.argmax(cutin), np.argmax(np.asarray(plain_cutin.active_mask, dtype=bool)))

    def test_cells_satisfy_pi_above_eps(self):
        _, _, _, result = self._predict(EPS)
        prediction = result["prediction"]
        active = prediction.active_mask
        pi = prediction.lead_mode_probability * prediction.floor_lateral[..., 0]
        self.assertTrue(np.all(pi[:, 1:][active[:, 1:]] > EPS))
        self.assertTrue(active[:, 1:].any())
        # and the controller's T on those cells is finite and unguarded
        terms, guarded = one_row_pz_risk_terms(prediction, config_for(num_modes=prediction.num_modes))
        self.assertTrue(np.all(np.isfinite(terms)) and not guarded[:, 1:].any())

    def test_the_gate_stays(self):
        # a cut-in hypothesis below the 0.1 gate never leads (pi <= p < eps anyway)
        _, _, _, result = self._predict(EPS, cutin_probs=(0.9, 0.04, 0.06))
        for target in result["processed_targets"]:
            for mode in target.mode_predictions:
                if mode.mode_name == "cutin":
                    self.assertFalse(np.asarray(mode.active_mask).any())

    def test_off_by_default(self):
        # without pz_risk: the mean-trajectory activation and P_lat in the first column (flooreps's), whether or not
        # the statistics are computed
        _, _, _, plain = self._predict(None, with_floor=False)
        _, _, _, stats = self._predict(None, with_floor=True)
        a, b = plain["prediction"], stats["prediction"]
        self.assertEqual(a.mode_names, b.mode_names)
        np.testing.assert_array_equal(a.active_mask, b.active_mask)
        np.testing.assert_array_equal(a.means, b.means)
        for target in stats["processed_targets"]:
            self.assertIsNone(target.branch_info["pz_risk"])
            for mode in target.mode_predictions:      # P_lat: the same value for every mode of the vehicle
                np.testing.assert_array_equal(mode.floor_lateral[:, 0], target.mode_predictions[0].floor_lateral[:, 0])

    def test_requirements(self):
        with self.assertRaises(ValueError):          # the adapter needs floor_lateral_fn
            self._predict(EPS, with_floor=False)
        raw = {"vehicle_id": 7, "raw_intention_prob": np.array([0.5, 0.1, 0.4]),
               "pred_traj_frenet": np.zeros((3, HORIZON, 2)), "raw_pred_vel": np.zeros((3, HORIZON, 2))}
        stats = tuple(np.zeros((3, HORIZON + 1)) for _ in range(4))
        with self.assertRaises(ValueError):          # the postprocess needs floor_lateral
            process_vehicle_prediction(raw, REL_LEFT_ADJACENT, np.array([12.0, 3.5, 10.0]), horizon=HORIZON, dt=DT,
                                       pz_risk=EPS)
        with self.assertRaises(ValueError):          # not with occin<eps>
            process_vehicle_prediction(raw, REL_LEFT_ADJACENT, np.array([12.0, 3.5, 10.0]), horizon=HORIZON, dt=DT,
                                       floor_lateral=stats, pz_risk=EPS, occupancy_risk=0.1, lk_entry_as_cutin=True,
                                       mode_lane_probabilities=np.zeros((3, HORIZON + 1)))


if __name__ == "__main__":
    unittest.main()
