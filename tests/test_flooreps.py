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
    build_multitarget_lead_prediction,
    process_vehicle_prediction,
    rotate_covariances_to_frenet,
)
from utils.acc_nair_smpc import (  # noqa: E402
    FLOOR_EPS_B_MIN,
    MultimodalLeadPrediction,
    NairACCSMPC,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    floor_eps_tightening,
    one_row_floor_terms,
    original_nair_acc_config,
    safety_std,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402


HORIZON = 15
DT = 0.2
TAU = 1.3
L_D0 = 4.5 + 3.0
AGENT = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
         "_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_leadmerge_mergefirst_lkhold_nolkentry_amin3.5_amax2"
         "_jlim2.5_jlimup5_onerow_zconf0.95_snap160_aheadonly_noegonoise_floorsigma_flooreps0.05")


def config_for(eps=0.05, num_modes=1, **overrides):
    """The final design's controller with floorsigma + flooreps<eps>."""
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
    config.one_row_z_conf = 0.95
    config.one_row_floor_sigma = True
    config.one_row_floor_eps = eps
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


class TestFloorEpsFormula(unittest.TestCase):
    """``floor_eps_tightening``: ``T = Phi^-1(b) sigma_c - rho sigma_s E_z``, ``b = clip(1 - eps / P_lat, 0.1, 1 - eps)``,
    with the truncated-normal moments of the lateral coordinate given the overlap."""

    #: the maintainer's reference (2026-11-18; Monte Carlo 0.99 % at the bound): two modes, eps 0.01, half-width 1.8 m,
    #: half-length 4.5 m, no d0 and no clip
    MODES = ((0.3, 15.0, 0.8, 2.0, 0.5, -0.4), (0.7, 20.0, 3.5, 2.0, 0.3, 0.0))   # p, mu_x, mu_y, sd_x, sd_y, rho
    HALF_W, HALF_L, EPS = 1.8, 4.5, 0.01

    def _bounds(self):
        lower = np.array([(-self.HALF_W - m[2]) / m[4] for m in self.MODES])
        upper = np.array([(self.HALF_W - m[2]) / m[4] for m in self.MODES])
        return lower, upper

    def test_the_maintainers_reference(self):
        lower, upper = self._bounds()
        p = np.array([m[0] for m in self.MODES])
        p_lat = float(np.sum(p * (norm.cdf(upper) - norm.cdf(lower))))
        self.assertAlmostEqual(p_lat, 0.293, places=3)
        self.assertAlmostEqual(self.EPS / p_lat, 0.0341, places=4)
        floor, b_raw = floor_eps_tightening(p_lat, lower[0], upper[0], self.MODES[0][5], self.MODES[0][3], self.EPS)
        self.assertAlmostEqual(float(b_raw), 1.0 - self.EPS / p_lat, places=12)
        bound = self.MODES[0][1] - self.HALF_L - float(floor)      # the ego centre's bound
        self.assertAlmostEqual(bound, 6.93, places=2)
        # Monte Carlo (fewer samples than the reference's 2e6): the collision probability at the bound is about eps
        rng = np.random.default_rng(1)
        n = 400000
        first = rng.random(n) < p[0]
        hits = 0
        for k, (_, mx, my, sx, sy, r) in enumerate(self.MODES):
            m = int(np.sum(first if k == 0 else ~first))
            xy = rng.multivariate_normal([mx, my], [[sx * sx, r * sx * sy], [r * sx * sy, sy * sy]], size=m)
            hits += int(np.sum((np.abs(xy[:, 0] - bound) < self.HALF_L) & (np.abs(xy[:, 1]) < self.HALF_W)))
        self.assertAlmostEqual(hits / n, self.EPS, delta=0.0006)

    def test_the_conditional_moments(self):
        # the shift rho sigma_s E_z and sigma_c against samples of a correlated pair truncated on the lateral one
        rho, sigma_s, lower, upper = 0.6, 2.0, -1.0, 0.5
        floor, _ = floor_eps_tightening(1.0, lower, upper, rho, sigma_s, 0.5)       # b = 0.5: T = -shift
        rng = np.random.default_rng(2)
        z_d = rng.standard_normal(1000000)
        s = sigma_s * (rho * z_d + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(z_d.size))
        kept = s[(z_d > lower) & (z_d < upper)]
        self.assertAlmostEqual(-float(floor), float(np.mean(kept)), delta=0.01)
        floor_b, _ = floor_eps_tightening(1.0, lower, upper, rho, sigma_s, 1.0 - norm.cdf(1.0))   # b = Phi(1)
        self.assertAlmostEqual(float(floor_b - floor), float(np.std(kept)), delta=0.01)    # T = sigma_c - shift

    def test_full_overlap_without_correlation(self):
        # P_lat 1, rho 0: z = Phi^-1(0.95) at eps 0.05 and T = 1.645 sigma (with the ego's term, safety_std's)
        floor, b_raw = floor_eps_tightening(1.0, -1.0, 2.0, 0.0, 0.5, 0.05)
        self.assertAlmostEqual(float(b_raw), 0.95, places=12)
        self.assertAlmostEqual(float(floor), norm.ppf(0.95) * 0.5, places=12)
        floor, _ = floor_eps_tightening(1.0, np.nan, np.nan, 0.0, 0.5, 0.05, ego_variance=0.164 ** 2)
        self.assertAlmostEqual(float(floor), norm.ppf(0.95) * np.hypot(0.5, 0.164), places=12)

    def test_rho_zero_keeps_the_unconditioned_moments(self):
        # rho 0: mu_c = mu_s and sigma_c = sigma_s whatever the overlap, so T = Phi^-1(b) sigma_s
        for lower, upper in ((-5.2, 2.0), (0.3, 0.9), (-np.inf, 0.0)):
            floor, _ = floor_eps_tightening(0.6, lower, upper, 0.0, 1.7, 0.05)
            self.assertAlmostEqual(float(floor), norm.ppf(1.0 - 0.05 / 0.6) * 1.7, places=12)

    def test_no_conditioning_below_the_overlap_threshold(self):
        # the own trajectory overlaps with Z < 1e-6: rho := 0
        lower, upper = 5.0, 6.0
        self.assertLess(norm.cdf(upper) - norm.cdf(lower), 1.0e-6)
        floor, _ = floor_eps_tightening(0.5, lower, upper, -0.8, 2.0, 0.05)
        self.assertAlmostEqual(float(floor), norm.ppf(0.9) * 2.0, places=12)

    def test_the_lower_clip(self):
        # 1 - eps / P_lat below 0.1 (and P_lat 0, b_raw -inf): b = 0.1, z = Phi^-1(0.1) = -1.28
        p_lat = np.array([0.04, 0.0, 0.06, 0.5])
        with np.errstate(all="raise"):
            floor, b_raw = floor_eps_tightening(p_lat, np.nan, np.nan, 0.0, 1.0, 0.05)
        np.testing.assert_array_equal(b_raw < FLOOR_EPS_B_MIN, [True, True, False, False])
        self.assertEqual(b_raw[1], -np.inf)
        np.testing.assert_allclose(floor[:2], norm.ppf(0.1), atol=1.0e-12)
        self.assertAlmostEqual(float(floor[2]), norm.ppf(1.0 - 0.05 / 0.06), places=12)
        self.assertAlmostEqual(float(floor[3]), norm.ppf(0.9), places=12)


def lead_prediction(gap0, lead_v, floor_lateral=None, cov=np.diag([0.25, 0.09])):
    steps = HORIZON + 1
    means = np.zeros((1, steps, 2))
    means[0, :, 0] = gap0 + lead_v * DT * np.arange(steps)
    means[0, :, 1] = lead_v
    return MultimodalLeadPrediction(
        means=means, probabilities=np.array([1.0]), covariances=np.tile(cov, (1, steps, 1, 1)),
        active_mask=np.ones((1, steps), dtype=bool), chance_confidence=np.full((1, steps), np.nan),
        sigma_confidence=np.full((1, steps), np.nan), lead_mode_lk=np.ones((1, steps), dtype=bool),
        lead_vehicle_ids=np.full((1, steps), 10, dtype=int), clearance_scale=np.ones((1, steps)),
        lead_mode_probability=np.ones((1, steps)), mode_names=["10:lk"], floor_lateral=floor_lateral)


class TestFloorEpsInTheController(unittest.TestCase):

    def test_token_parse_and_validation(self):
        parse = ACCNairSMPCAgent._parse_floor_eps
        self.assertEqual(parse(AGENT), 0.05)
        self.assertEqual(parse("onerow_floorsigma_flooreps0.01_ra0"), 0.01)
        self.assertIsNone(parse(AGENT.replace("_flooreps0.05", "")))
        self.assertIsNone(parse("acc_x_flooreps_ra0"))
        self.assertIsNone(parse("acc_x_xflooreps0.05"))
        self.assertIsNone(parse("acc_x_flooreps0.05x"))
        self.assertEqual(config_for().one_row_floor_eps, 0.05)
        with self.assertRaises(ValueError):                    # needs floorsigma
            config_for(one_row_floor_sigma=False)
        for bad in (0.0, 0.9, 1.2):
            with self.assertRaises(ValueError):
                config_for(eps=bad)
        config_for(one_row_z_conf=None)                       # floorsigma's sigma term is flooreps's

    def test_the_carla_agent_refuses_it(self):
        with self.assertRaises(ValueError) as caught:
            ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=0.2, N=15, N_modes=8,
                             smpc_config=AGENT)
        self.assertIn("NGSIM replay only", str(caught.exception))

    def test_floor_terms_from_the_cells_statistics(self):
        config = config_for()
        steps = HORIZON + 1
        lateral = np.zeros((1, steps, 4))
        lateral[0, :, 0] = np.linspace(1.0, 0.02, steps)          # P_lat falling: the clip binds at the end
        lateral[0, :, 1], lateral[0, :, 2], lateral[0, :, 3] = -1.0, 2.5, -0.5
        lateral[0, 0, 1:3] = np.nan                                 # step 0: no spread
        prediction = lead_prediction(30.0, 10.0, floor_lateral=lateral)
        terms, clipped = one_row_floor_terms(prediction, config)
        expected, b_raw = floor_eps_tightening(lateral[0, :, 0], lateral[0, :, 1], lateral[0, :, 2], lateral[0, :, 3],
                                               0.5, 0.05)
        np.testing.assert_allclose(terms[0], expected, atol=1.0e-12)
        np.testing.assert_array_equal(clipped[0], b_raw < FLOOR_EPS_B_MIN)
        self.assertTrue(clipped[0, -1] and not clipped[0, 0])
        prediction.active_mask[0, 10:] = False                     # inactive cells: no floor
        terms, clipped = one_row_floor_terms(prediction, config)
        self.assertTrue(np.all(terms[0, 10:] == 0.0) and not clipped[0, 10:].any())

    def test_prediction_without_the_statistics(self):
        config = config_for()
        # the replay's no-target placeholder (no cut-in chance cell): P_lat 1 without conditioning
        terms, _ = one_row_floor_terms(lead_prediction(1000.0, 20.0), config)
        np.testing.assert_allclose(terms, norm.ppf(0.95) * safety_std(np.diag([0.25, 0.09]), config), atol=1.0e-12)
        # with a cut-in chance cell it raises
        prediction = lead_prediction(30.0, 10.0)
        prediction.chance_confidence[0, 5] = 0.6
        with self.assertRaises(ValueError):
            one_row_floor_terms(prediction, config)

    def test_the_floor_row_and_the_counts(self):
        steps = HORIZON + 1
        lateral = np.zeros((1, steps, 4))
        lateral[0, :, 0] = 0.5
        lateral[0, :, 1], lateral[0, :, 2], lateral[0, :, 3] = -0.5, 3.0, 0.7
        lateral[0, 3:, 0] = 0.03                                    # steps 3..: the lower clip
        config = config_for()
        prediction = lead_prediction(0.0, 10.0, floor_lateral=lateral)
        floor0 = L_D0 + float(one_row_floor_terms(prediction, config)[0][0, 0])
        prediction.means[0, :, 0] += floor0 - 0.3                   # 0.3 m inside the floor at step 0
        controller = NairACCSMPC(config)
        solution = controller.solve(np.array([0.0, 10.0]), prediction, previous_accel_cmd=0.0, command_dt=0.1,
                                    ego_accel=0.0)
        self.assertNotEqual(solution.solve_path, "fallback")
        self.assertAlmostEqual(float(solution.slack[0, 0]), 0.3, places=4)
        self.assertEqual((solution.floor_eps_cells, solution.floor_eps_clipped), (steps, steps - 3))
        self.assertEqual((solution.floor_eps_cutin_cells, solution.floor_eps_cutin_clipped), (0, 0))

    def test_the_trim_keeps_the_statistics(self):
        steps = HORIZON + 6
        means = np.zeros((1, steps, 2))
        means[0, :, 0] = 40.0 + 2.0 * np.arange(steps)
        means[0, :, 1] = 10.0
        lateral = np.random.default_rng(3).random((1, steps, 4))
        prediction = MultimodalLeadPrediction(means=means, probabilities=np.array([1.0]), floor_lateral=lateral)
        trimmed = NairACCSMPC(config_for())._trim_prediction(prediction)
        np.testing.assert_array_equal(trimmed.floor_lateral, lateral[:, :HORIZON + 1])


def raw_target(vehicle_id, probs, y_paths, x0, speed, sd_x, sd_y, rho, out_length):
    """Three trajectories (LK, LLC, RLC) along global x at ``speed``, lateral position per trajectory ramping from
    ``y_paths[i][0]`` to ``y_paths[i][1]``, with a 2-D position covariance (global frame) per trajectory."""
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


class TestFloorEpsPipeline(unittest.TestCase):
    """The adapter and the postprocess: every target's P_lat summed over all its trajectories, each cell's own
    trajectory's bounds and correlation, carried to the cells its mode leads."""

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

    def _predict(self, with_floor):
        adapter = STDAN3IntACCAdapter(load_model=False)
        n = adapter.out_length
        raws = {
            # the ego-lane lead 30 m ahead: lane keeping, a left lane change out (a cut-out), a right one out
            5: raw_target(5, [0.8, 0.12, 0.08], ((0.0, 0.0), (0.0, 3.5), (0.0, -3.5)), 30.0, 9.0, 0.4, 0.2,
                          (0.3, 0.1, -0.2), n),
            # a left-adjacent vehicle 12 m ahead: its right lane change is the cut-in, LK stays, LLC goes away
            7: raw_target(7, [0.55, 0.05, 0.4], ((3.5, 3.5), (3.5, 7.0), (3.5, 0.0)), 12.0, 10.0, 0.5, 0.25,
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
            floor_lateral_fn=floor_lateral if with_floor else None)
        return adapter, raws, floor_lateral, result

    def test_cells_carry_their_leads_statistics(self):
        adapter, raws, floor_lateral, result = self._predict(True)
        prediction = result["prediction"]
        self.assertEqual(prediction.floor_lateral.shape, (prediction.num_modes, HORIZON + 1, 4))
        modes = {(t.vehicle_id, m.scenario_label): (t, m) for t in result["processed_targets"]
                 for m in t.mode_predictions}
        checked = set()
        for j in range(prediction.num_modes):
            for k in range(HORIZON + 1):
                if not prediction.active_mask[j, k]:
                    self.assertTrue(np.all(np.isnan(prediction.floor_lateral[j, k])))
                    continue
                vid, label = prediction.lead_keys[j][k][:2]
                target, mode = modes[(vid, label)]
                raw = raws[vid]
                xy = adapter._sample_future(raw["raw_traj_xy"], HORIZON, DT)
                cov = adapter._sample_future(raw["raw_position_cov_global"], HORIZON, DT)
                z, lower, upper = floor_lateral(vid, xy, cov)
                p = np.asarray(raw["raw_intention_prob"]) / np.sum(raw["raw_intention_prob"])
                rep = max(mode.raw_mode_indices, key=lambda idx: p[idx])
                # P_lat over all three trajectories (nolkentry switches two off for activation, not here)
                self.assertAlmostEqual(prediction.floor_lateral[j, k, 0], float(np.sum(p * z[:, k])), places=12)
                np.testing.assert_array_equal(prediction.floor_lateral[j, k, 1:3], [lower[rep, k], upper[rep, k]])
                # psi 0: rho is the global covariance's correlation (0 at step 0)
                expected_rho = 0.0 if k == 0 else (cov[rep, k - 1, 0, 1]
                                                   / np.sqrt(cov[rep, k - 1, 0, 0] * cov[rep, k - 1, 1, 1]))
                self.assertAlmostEqual(prediction.floor_lateral[j, k, 3], expected_rho, places=12)
                checked.add((vid, label))
        # the lead's lane keeping and its cut-out (in the lane at the first steps), and the cut-in hypothesis
        self.assertEqual(checked, {(5, "lk"), (5, "cutout"), (7, "cutin")})

    def test_off_without_the_function(self):
        _, _, _, plain = self._predict(False)
        _, _, _, floor = self._predict(True)
        self.assertIsNone(plain["prediction"].floor_lateral)
        for target in plain["processed_targets"]:
            self.assertTrue(all(mode.floor_lateral is None for mode in target.mode_predictions))
        # the scenarios, their probabilities and cells are the same with the statistics
        a, b = plain["prediction"], floor["prediction"]
        self.assertEqual(a.mode_names, b.mode_names)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        np.testing.assert_array_equal(a.active_mask, b.active_mask)
        np.testing.assert_array_equal(a.means, b.means)
        np.testing.assert_array_equal(a.covariances, b.covariances)

    def test_the_correlation_with_a_heading(self):
        # s = cos(psi) x + sin(psi) y (the Frenet rotation the rows' variance comes from) against y
        rng = np.random.default_rng(4)
        a = rng.normal(size=(5, 7, 2, 2))
        cov = a @ np.swapaxes(a, -1, -2) + 0.1 * np.eye(2)
        psi = rng.uniform(-0.3, 0.3, size=(5, 7))
        var_s = rotate_covariances_to_frenet(cov, psi)[..., 0, 0]
        rho = STDAN3IntACCAdapter._longitudinal_lateral_correlation(cov, psi, var_s)
        for idx in np.ndindex(psi.shape):
            m = np.array([[np.cos(psi[idx]), np.sin(psi[idx])], [0.0, 1.0]])
            sy = m @ cov[idx] @ m.T
            self.assertAlmostEqual(sy[0, 0], var_s[idx], places=10)
            self.assertAlmostEqual(rho[idx], sy[0, 1] / np.sqrt(sy[0, 0] * sy[1, 1]), places=10)
        self.assertEqual(float(STDAN3IntACCAdapter._longitudinal_lateral_correlation(
            np.zeros((2, 2)), 0.0, 0.0)), 0.0)

    def test_process_vehicle_prediction_needs_matching_shapes(self):
        raw = {"vehicle_id": 7, "raw_intention_prob": np.array([0.5, 0.1, 0.4]),
               "pred_traj_frenet": np.zeros((3, HORIZON, 2)), "raw_pred_vel": np.zeros((3, HORIZON, 2))}
        bad = tuple(np.zeros((3, HORIZON)) for _ in range(4))
        with self.assertRaises(ValueError):
            process_vehicle_prediction(raw, REL_LEFT_ADJACENT, np.array([12.0, 3.5, 10.0]), horizon=HORIZON, dt=DT,
                                       floor_lateral=bad)
        processed = process_vehicle_prediction(
            raw, REL_LEFT_ADJACENT, np.array([12.0, 3.5, 10.0]), horizon=HORIZON, dt=DT,
            floor_lateral=tuple(np.zeros((3, HORIZON + 1)) for _ in range(4)))
        prediction, _ = build_multitarget_lead_prediction([processed], ego_state=[0.0, 10.0], horizon=HORIZON,
                                                          desired_speed=20.0, num_modes=8)
        self.assertEqual(prediction.floor_lateral.shape, (prediction.num_modes, HORIZON + 1, 4))


if __name__ == "__main__":
    unittest.main()
