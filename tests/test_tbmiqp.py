"""``tbmiqp<w>`` (2026-11-16, maintainer; after Wang, He & Althoff, IV 2026): the step ``t_b`` from which the scenarios
that differ in a vehicle's cut-in hypothesis may use different jerks is chosen by the optimiser, one MIQP per tick
(monotone binaries per vehicle, big-M rows in place of the fixed share rows, a penalty on branching), in place of
today's split at the first step the hypothesis leads.

The scenes of ``tests/data/tbmiqp_scenes.json`` are recorded controller inputs (the prediction, the ego state and the
command context of one tick) of the NGSIM reproduction episodes, from one replay of the current design with the token.
Here a fresh controller solves them at the episode's set speed, without the replay's warm start and remembered plan.
No recorded tick has two adjacent vehicles with a cut-in hypothesis, so the two-vehicle scene puts two recorded
hypotheses into one scene (``two_vehicle_scene``)."""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "carla"))
sys.path.insert(0, ROOT)

from utils.acc_nair_smpc import (  # noqa: E402
    BRANCH_TIME_SLOTS,
    MultimodalLeadPrediction,
    NairACCSMPC,
    PolicySharingTree,
    SAFETY_CONFIDENCE_CHANCE,
    VARIANT_FIXED_RISK,
    _branch_time_transitive,
    branch_time_structure,
    failsafe_recovery_envelope,
    original_nair_acc_config,
)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (  # noqa: E402
    ACCModePrediction,
    ACCProcessedPrediction,
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    build_multitarget_lead_prediction,
    build_policy_sharing_tree_from_effective_leads,
)

HORIZON = 15
DT = 0.2
with open(os.path.join(ROOT, "tests", "data", "tbmiqp_scenes.json")) as handle:
    SCENES = json.load(handle)
#: the current design as the CARLA agent reads it (AGENTS.md §1 without ``bcts``, which the agent refuses)
DESIGN = ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_chancecells_dopt10_trajcutin_ctx0_latgate"
          "_nocutout_rmove1000_rjerk40_jinput_ra0_fullfoot_l1slack1e6_safelayer_leadmerge_mergefirst_lkhold_entrant2"
          "_nolkentry_amin3.5_amax2_jlim2.5_jlimup5_fsdecel8_fsjerk10_onerow_zconf0.95_snap160_aheadonly")


#: how far the jerks of two scenarios tied by a binary at 0 may differ [m/s^3]: the rows' feasibility tolerance (1e-4)
#: on a row whose largest coefficient is the big-M, 7.5 (measured: up to 7.5e-4; with the binaries fixed, below 1e-6)
TIED_TOL = 1.0e-3


def design_config(v_set=20.0, weight=None, **overrides):
    """The current NGSIM design's controller (onerow w 10 with the cap, zconf0.95, snap160, the unified limits,
    safelayer's recovery), with ``tbmiqp<weight>`` when a weight is given."""
    config = original_nair_acc_config(
        horizon=HORIZON, dt=DT, desired_speed=v_set, num_modes=8, controller_variant=VARIANT_FIXED_RISK,
        gurobi_time_limit=0.15)
    config.safety_constraint_mode = SAFETY_CONFIDENCE_CHANCE
    config.r_move, config.r_jerk, config.r_a = 1000.0, 40.0, 0.0
    config.standoff_shortfall_weight = 10.0
    config.jerk_input = True
    config.snap_weight = 160.0
    config.slack_l1_weight = 1.0e6
    config.a_min, config.a_max, config.jerk_limit, config.jerk_limit_up = -3.5, 2.0, 2.5, 5.0
    config.failsafe_recovery = True
    config.failsafe_decel, config.failsafe_jerk = -8.0, 10.0
    config.one_row, config.one_row_z_conf = True, 0.95
    config.branch_time_miqp_weight = weight
    for name, value in overrides.items():
        setattr(config, name, value)
    config.__post_init__()
    return config


def branch_penalty(weight, p_cut, t_b):
    """The token's penalty for one vehicle that branches at ``t_b``: ``w P (N - t_b)``."""
    return weight * p_cut * (HORIZON - t_b)


def _nan(rows):
    return np.array([[np.nan if value is None else value for value in row] for row in rows], dtype=float)


def scene_prediction(scene, tree=None):
    """The recorded prediction; ``tree`` replaces its policy tree (built from the lead keys, as the predictor does)."""
    keys = [[tuple(key) for key in scenario] for scenario in scene["lead_keys"]]
    if tree is None:
        tree = build_policy_sharing_tree_from_effective_leads(keys, HORIZON, scene["mode_names"])
    return MultimodalLeadPrediction(
        means=np.array(scene["means"]), probabilities=np.array(scene["probabilities"]),
        covariances=np.array(scene["covariances"]), mode_names=list(scene["mode_names"]), policy_tree=tree,
        active_mask=np.array(scene["active_mask"], dtype=bool), chance_confidence=_nan(scene["chance_confidence"]),
        sigma_confidence=_nan(scene["sigma_confidence"]), lead_mode_lk=np.array(scene["lead_mode_lk"], dtype=bool),
        lead_vehicle_ids=np.array(scene["lead_vehicle_ids"], dtype=int),
        lead_mode_probability=np.array(scene["lead_mode_probability"]), lead_keys=keys)


def split_tree(prediction, step):
    """The prediction with a policy tree that shares every input before ``step`` and none from it on."""
    tree = PolicySharingTree.single_split(prediction.num_modes, HORIZON, step)
    return MultimodalLeadPrediction(
        means=prediction.means, probabilities=prediction.probabilities, covariances=prediction.covariances,
        mode_names=prediction.mode_names, policy_tree=tree, active_mask=prediction.active_mask,
        chance_confidence=prediction.chance_confidence, sigma_confidence=prediction.sigma_confidence,
        lead_mode_lk=prediction.lead_mode_lk, lead_vehicle_ids=prediction.lead_vehicle_ids,
        lead_mode_probability=prediction.lead_mode_probability, lead_keys=prediction.lead_keys)


def solve(controller, scene, prediction, **command):
    """One solve of the scene from an empty memory (no warm start, no remembered plan), so that two controllers given
    the same scene solve the same problem."""
    controller.last_solution = None
    controller.reference_adapter.previous_planned_speed = None
    controller.reference_adapter.previous_planned_standoff = {}
    controller._last_issued_cmd = scene["last_issued_cmd"]
    context = dict(previous_accel_cmd=scene["previous_accel_cmd"], command_dt=scene["command_dt"],
                   ego_accel=scene["ego_accel"])
    context.update(command)
    return controller.solve(np.array(scene["ego_state"]), prediction, **context)


def fix_branch_steps(controller, steps):
    """Fix the binaries: the vehicle of slot s branches at ``steps[s]`` (1..N, N = never)."""
    def bounds(plan):
        fixed = np.zeros((BRANCH_TIME_SLOTS, HORIZON - 1))
        for vehicle in plan["vehicles"]:
            if vehicle["slot"] >= 0:
                fixed[vehicle["slot"], steps[vehicle["slot"]] - 1:] = 1.0
        return fixed, fixed.copy()
    controller._branch_time_bounds = bounds


def decisions_by_slot(solution):
    return sorted((d for d in solution.branch_time_decisions if d["slot"] >= 0), key=lambda d: d["slot"])


def _hypothesis(scene, row, first, ego_s):
    """The cut-in hypothesis leading ``scene``'s scenario ``row`` from step ``first``, as an adjacent target at the
    recorded offset from the ego: its lane-keeping mode (never in the ego lane) and its cut-in mode.  Before its entry
    the cut-in trajectory is out of the lane (inactive); those steps hold the entry speed backwards."""
    means, covariances = np.array(scene["means"])[row], np.array(scene["covariances"])[row]
    vehicle_id, p_cut = scene["lead_vehicle_ids"][row][first], scene["lead_mode_probability"][row][first]
    frenet = np.zeros((HORIZON + 1, 3))
    frenet[:, 0] = means[:, 0] - scene["ego_state"][0] + ego_s
    frenet[:, 2] = means[:, 1]
    for step in range(first - 1, -1, -1):
        frenet[step, 0] = frenet[step + 1, 0] - DT * frenet[first, 2]
        frenet[step, 2] = frenet[first, 2]
    active = np.zeros(HORIZON + 1, dtype=bool)
    active[first:] = True
    covariances[:first] = 0.0
    lane_keeping = ACCModePrediction(
        vehicle_id=vehicle_id, mode_name="lk", probability=1.0 - p_cut, frenet=frenet.copy(),
        active_mask=np.zeros(HORIZON + 1, dtype=bool), raw_mode_indices=[0, 2], ego_lane_membership_split_step=first)
    cutin = ACCModePrediction(
        vehicle_id=vehicle_id, mode_name="cutin", probability=p_cut, frenet=frenet, active_mask=active,
        raw_mode_indices=[1], ego_lane_membership_split_step=first, lead_covariance=covariances,
        chance_confidence=scene["chance_confidence"][row][first])
    return ACCProcessedPrediction(vehicle_id, REL_LEFT_ADJACENT, [lane_keeping, cutin], {}, {}, {})


def _lead(scene):
    """The ego-lane lead of the scene's lane-keeping scenario (row 0 of 'cutin')."""
    means = np.array(scene["means"])[0]
    frenet = np.zeros((HORIZON + 1, 3))
    frenet[:, 0], frenet[:, 2] = means[:, 0], means[:, 1]
    vehicle_id = scene["lead_vehicle_ids"][0][0]
    mode = ACCModePrediction(
        vehicle_id=vehicle_id, mode_name="lk", probability=1.0, frenet=frenet,
        active_mask=np.ones(HORIZON + 1, dtype=bool), raw_mode_indices=[0, 1, 2], ego_lane_membership_split_step=0,
        lead_covariance=np.array(scene["covariances"])[0])
    return ACCProcessedPrediction(vehicle_id, REL_EGO_LANE, [mode], {}, {}, {})


def _joint(scene, targets):
    """The joint prediction as the design builds it (leadmerge, mergefirst, aheadonly, the 0.1 gate)."""
    return build_multitarget_lead_prediction(
        targets, ego_state=np.array(scene["ego_state"]), horizon=HORIZON, desired_speed=scene["v_set"], num_modes=8,
        reference_beta=0.6, vanish_threshold=0.1, lead_merge=True, merge_first=True, ahead_only=True)[0]


def rebuilt_cutin_scene():
    """Scene 'cutin' rebuilt from its lead and its hypothesis through the predictor's joint builder."""
    scene = SCENES["cutin"]
    return _joint(scene, [_lead(scene), _hypothesis(scene, 1, 5, scene["ego_state"][0])])


def two_vehicle_scene():
    """Scene 'cutin' (lead 1443, the TV 1446 cutting in from step 5 with 49.8 %) with a second adjacent vehicle: the
    hypothesis recorded in 'other_vehicle' (vehicle 3261 from step 8 with 80.6 %), at its recorded offset from the ego.
    Three scenarios: 1446 cuts in (3261 behind it either way), only 3261 cuts in, neither."""
    scene, other = SCENES["cutin"], SCENES["other_vehicle"]
    ego_s = scene["ego_state"][0]
    return _joint(scene, [_lead(scene), _hypothesis(scene, 1, 5, ego_s), _hypothesis(other, 0, 8, ego_s)])


_CACHE = {}


def controller(scene, weight=None, **overrides):
    """One controller per configuration, shared by the tests (its QPs are built once)."""
    key = ("controller", scene["v_set"], weight, tuple(sorted(overrides.items())))
    if key not in _CACHE:
        _CACHE[key] = NairACCSMPC(design_config(scene["v_set"], weight, **overrides))
    return _CACHE[key]


def fixed_solutions(name):
    """Per branching step t = 1..N the solution of scene ``name`` with the binaries fixed at t and no penalty (w = 0),
    index t - 1; for 'two_vehicles' a dict over (t of slot 0, t of slot 1)."""
    key = ("fixed", name)
    if key not in _CACHE:
        scene = SCENES["cutin" if name == "two_vehicles" else name]
        prediction = two_vehicle_scene() if name == "two_vehicles" else scene_prediction(scene)
        fixed = NairACCSMPC(design_config(scene["v_set"], 0.0, allow_solver_fallback=False))
        steps = range(1, HORIZON + 1)
        if name == "two_vehicles":
            grid = [(first, second) for first in steps for second in steps]
        else:
            grid = [(step,) for step in steps]
        solutions = {}
        for point in grid:
            fix_branch_steps(fixed, list(point) + [HORIZON] * (BRANCH_TIME_SLOTS - len(point)))
            solutions[point] = solve(fixed, scene, prediction)
        _CACHE[key] = solutions if name == "two_vehicles" else [solutions[(step,)] for step in steps]
    return _CACHE[key]


def assert_cost_close(test, first, second):
    """Costs equal within the solvers' tolerance: 5e-4 relative (the QP's 1e-4 and the MIP gap 1e-4 are relative to an
    objective without its constant) plus 0.05 (the 1e6 l1 weight on slacks of 1e-8 m)."""
    test.assertLessEqual(abs(first - second), 5.0e-4 * max(abs(first), abs(second)) + 0.05)


class ParserAndRefusalTests(unittest.TestCase):

    def test_parser(self):
        parse = ACCNairSMPCAgent._parse_branch_time_miqp
        self.assertEqual(parse(DESIGN + "_tbmiqp100"), 100.0)
        self.assertEqual(parse(DESIGN + "_tbmiqp1e4"), 1.0e4)
        self.assertEqual(parse("jinput_tbmiqp0.5_onerow"), 0.5)
        self.assertEqual(parse(DESIGN + "_tbmiqp0"), 0.0)
        self.assertIsNone(parse(DESIGN))
        self.assertIsNone(parse("acc_x_xtbmiqp5"))
        for bad in ("acc_x_tbmiqp", "acc_x_tbmiqp5x", "acc_x_tbmiqp-1"):       # the weight is required
            with self.assertRaises(ValueError):
                parse(bad)

    def test_config(self):
        self.assertIsNone(original_nair_acc_config(horizon=HORIZON, dt=DT).branch_time_miqp_weight)
        self.assertIsNone(design_config().branch_time_miqp_weight)
        self.assertEqual(design_config(weight=0.0).branch_time_miqp_weight, 0.0)
        for overrides in (
                dict(weight=-1.0), dict(weight=float("inf")),
                dict(jerk_input=False, snap_weight=None),                   # its big-M is the jerk bounds' width
                dict(jerk_limit=None, jerk_limit_up=None),
                dict(one_row=False, one_row_z_conf=None),
                dict(optimize_k=True),
                dict(pass_miqp_margin=0.0), dict(cutin_cvar_risk=0.1, cutin_cvar_bound=0.5),
                dict(failsafe_branch=True)):
            overrides.setdefault("weight", 100.0)
            with self.assertRaises(ValueError, msg=str(overrides)):
                design_config(**overrides)

    def _agent(self, policy):
        return ACCNairSMPCAgent(mock.MagicMock(), None, nominal_speed_mps=20.0, dt=DT, N=HORIZON, N_modes=8,
                                smpc_config=policy)

    def test_agent_refusals(self):
        token = "_tbmiqp100"
        for policy, message in (
                (DESIGN.replace("_snap160", "").replace("_jinput", "") + token, "tbmiqp<w> needs jinput"),
                (DESIGN.replace("_onerow_zconf0.95", "") + token, "tbmiqp<w> needs jinput"),
                (DESIGN + "_branchdec3" + token, "not with branchdec<a>"),
                (DESIGN + "_best_mode" + token, "not with best_mode"),
                (DESIGN + "_with_k" + token, "not with with_k"),
                (DESIGN + "_tbmiqp", "needs its weight"),
                # onerow, which the token needs, refuses these before the token's own check does
                (DESIGN + "_passmiqp" + token, "passmiqp<m>"),
                (DESIGN + "_passmiqp_passnocap" + token, "passmiqp<m>"),
                (DESIGN + "_cvar0.1_0.5" + token, "cvar<eps>_<c>"),
                (DESIGN + "_safebranch" + token, "safebranch")):
            with self.assertRaises(ValueError, msg=policy) as caught:
                self._agent(policy)
            self.assertIn(message, str(caught.exception))


def keyed_prediction(keys, p_cut):
    """A prediction with what ``branch_time_structure`` reads: the lead keys (the policy tree built from them) and the
    hypothesis probability ``p_cut[vehicle]`` on the cut-in cells.  Returns it with today's groups."""
    steps = HORIZON + 1
    cell_p = np.ones((len(keys), steps))
    for scenario, sequence in enumerate(keys):
        for step, key in enumerate(sequence):
            if str(key[1]).startswith("cutin"):
                cell_p[scenario, step] = p_cut[key[0]]
    tree = build_policy_sharing_tree_from_effective_leads(keys, HORIZON)
    prediction = MultimodalLeadPrediction(
        means=np.zeros((len(keys), steps, 2)), probabilities=np.full(len(keys), 1.0 / len(keys)), policy_tree=tree,
        lead_mode_probability=cell_p, lead_keys=keys)
    return prediction, tree.policy_groups_by_step


def sequence(*parts):
    """Lead keys for steps 0..N from (key, count) parts."""
    keys = [key for key, count in parts for _ in range(count)]
    assert len(keys) == HORIZON + 1
    return keys


LEAD, LEAD2, TV_I, TV_J = (1, "lk"), (2, "lk"), (7, "cutin"), (8, "cutin")


class StructureTests(unittest.TestCase):
    """``branch_time_structure``: which pairs a vehicle's binaries couple, the slots, and the fall-back to today's
    rule."""

    def test_one_vehicle(self):
        keys = [sequence((LEAD, 16)), sequence((LEAD, 5), (TV_I, 11))]
        plan = branch_time_structure(*keyed_prediction(keys, {7: 0.3}))
        self.assertEqual(plan["vehicles"], [
            {"vehicle_id": 7, "labels": ("cutin",), "p_cut": 0.3, "k_on": 5, "slot": 0}])
        slots, free = plan["pairs"][(0, 1)]
        np.testing.assert_array_equal(slots, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(free, np.zeros(HORIZON))
        self.assertFalse(plan["today_rule"])
        self.assertTrue(plan["today_exact"])
        self.assertEqual(plan["n_other_pairs"], 0)

    def test_nothing_to_branch(self):
        prediction, groups = keyed_prediction([sequence((LEAD, 16))], {})
        self.assertIsNone(branch_time_structure(prediction, groups))                     # one scenario
        # two scenarios that no cut-in hypothesis tells apart
        self.assertIsNone(branch_time_structure(*keyed_prediction(
            [sequence((LEAD, 16)), sequence((LEAD, 5), (LEAD2, 11))], {})))
        # the same cut-in hypothesis in every scenario
        self.assertIsNone(branch_time_structure(*keyed_prediction(
            [sequence((LEAD, 3), (TV_I, 13)), sequence((LEAD, 3), (TV_I, 13))], {7: 1.0})))
        # a prediction without lead keys (the synthetic fallback prediction) keeps today's tree
        prediction, groups = keyed_prediction([sequence((LEAD, 16)), sequence((LEAD, 5), (TV_I, 11))], {7: 0.3})
        prediction.lead_keys = None
        self.assertIsNone(branch_time_structure(prediction, groups))

    def test_two_vehicles(self):
        # neither, i from step 3, j from step 6, both (j nearer once it is in)
        keys = [sequence((LEAD, 16)), sequence((LEAD, 3), (TV_I, 13)), sequence((LEAD, 6), (TV_J, 10)),
                sequence((LEAD, 3), (TV_I, 3), (TV_J, 10))]
        plan = branch_time_structure(*keyed_prediction(keys, {7: 0.7, 8: 0.4}))
        self.assertEqual([(v["vehicle_id"], v["k_on"], v["slot"]) for v in plan["vehicles"]], [(7, 3, 0), (8, 6, 1)])
        expected = {(0, 1): [1, 0], (0, 2): [0, 1], (0, 3): [1, 1], (1, 2): [1, 1], (1, 3): [0, 1], (2, 3): [1, 0]}
        for pair, slots in expected.items():                                    # D(a, b) per pair
            np.testing.assert_array_equal(plan["pairs"][pair][0][:2], slots, err_msg=str(pair))
            np.testing.assert_array_equal(plan["pairs"][pair][1], np.zeros(HORIZON))
        self.assertEqual(plan["n_other_pairs"], 0)
        self.assertFalse(plan["today_rule"])
        self.assertTrue(plan["today_exact"])
        # sharing stays transitive for every pattern of the binaries
        self.assertTrue(_branch_time_transitive(4, plan["pairs"], 2, HORIZON))

    def test_two_labels_of_one_vehicle_share_its_binaries(self):
        keys = [sequence((LEAD, 16)), sequence((LEAD, 4), (TV_I, 12)), sequence((LEAD, 6), ((7, "cutin/LK"), 10))]
        prediction, groups = keyed_prediction(keys, {7: 0.3})
        prediction.lead_mode_probability[2, 6:] = 0.2
        plan = branch_time_structure(prediction, groups)
        self.assertEqual(len(plan["vehicles"]), 1)
        self.assertEqual(plan["vehicles"][0]["labels"], ("cutin", "cutin/LK"))
        self.assertAlmostEqual(plan["vehicles"][0]["p_cut"], 0.5)                 # the vehicle cuts in either way
        self.assertEqual(plan["vehicles"][0]["k_on"], 4)
        for pair in ((0, 1), (0, 2), (1, 2)):
            np.testing.assert_array_equal(plan["pairs"][pair][0], [1.0, 0.0, 0.0])

    def test_slots(self):
        # four vehicles: the three most likely get binaries, the fourth keeps today's split
        vehicles = {11: (0.2, 4), 12: (0.5, 6), 13: (0.15, 2), 14: (0.9, 9)}
        keys = [sequence((LEAD, 16))] + [
            sequence((LEAD, first), ((vid, "cutin"), 16 - first)) for vid, (_, first) in vehicles.items()]
        plan = branch_time_structure(*keyed_prediction(keys, {vid: p for vid, (p, _) in vehicles.items()}))
        self.assertEqual([(v["vehicle_id"], v["slot"]) for v in plan["vehicles"]],
                         [(14, 0), (12, 1), (11, 2), (13, -1)])
        slots, free = plan["pairs"][(0, 3)]                                       # lane keeping against vehicle 13
        np.testing.assert_array_equal(slots, np.zeros(BRANCH_TIME_SLOTS))
        np.testing.assert_array_equal(free, (np.arange(HORIZON) >= 2).astype(float))
        slots, free = plan["pairs"][(1, 3)]                                       # vehicle 11 against vehicle 13
        np.testing.assert_array_equal(slots, [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(free, (np.arange(HORIZON) >= 2).astype(float))
        self.assertTrue(plan["today_exact"])

    def test_other_pairs_keep_todays_rule(self):
        # scenario 2 differs from the others from step 0 for another reason (another lead): its pairs keep today's
        # rule, and sharing stays transitive
        keys = [sequence((LEAD, 16)), sequence((LEAD, 5), (TV_I, 11)), sequence((LEAD2, 16))]
        prediction, groups = keyed_prediction(keys, {7: 0.3})
        plan = branch_time_structure(prediction, groups)
        self.assertEqual(plan["n_other_pairs"], 2)
        self.assertFalse(plan["today_rule"])
        for pair in ((0, 2), (1, 2)):
            np.testing.assert_array_equal(plan["pairs"][pair][0], np.zeros(BRANCH_TIME_SLOTS))
            np.testing.assert_array_equal(plan["pairs"][pair][1], np.ones(HORIZON))

    def test_falls_back_when_sharing_would_not_be_transitive(self):
        # scenario 2 leaves lane keeping at step 5 for another reason, where the cut-in also starts: with the binary
        # at 0 from step 5 on, 0 ~ 1 and 1 ~ 2 would be tied while today's rule separates 0 and 2
        keys = [sequence((LEAD, 16)), sequence((LEAD, 5), (TV_I, 11)), sequence((LEAD, 5), (LEAD2, 11))]
        plan = branch_time_structure(*keyed_prediction(keys, {7: 0.3}))
        self.assertEqual(plan["n_other_pairs"], 1)
        self.assertTrue(plan["today_rule"])

    def test_today_exact(self):
        # vehicle j (in a and b) hides i until step 6, while i alone leads from step 3: fixing i's binaries at its
        # first lead step, 3, unties a and b earlier than today's rule does
        keys = [sequence((LEAD, 2), (TV_J, 4), (TV_I, 10)), sequence((LEAD, 2), (TV_J, 14)),
                sequence((LEAD, 3), (TV_I, 13))]
        plan = branch_time_structure(*keyed_prediction(keys, {7: 0.6, 8: 0.5}))
        self.assertEqual(plan["n_other_pairs"], 0)
        self.assertFalse(plan["today_rule"])
        self.assertFalse(plan["today_exact"])


class OffTests(unittest.TestCase):
    """Without the token, and on a tick the token does not act on, the problem is the plain QP."""

    def test_without_the_token_the_problem_has_no_binaries(self):
        scene = SCENES["cutin"]
        plain = NairACCSMPC(design_config(scene["v_set"]))
        solution = solve(plain, scene, scene_prediction(scene))
        problems = list(plain._feedback_scalar_chance_qp_cache.values())
        self.assertEqual(len(problems), 1)
        self.assertIsNone(problems[0].tb_delta_var)
        self.assertIsNone(problems[0].tb_pair_params)
        self.assertEqual(len(problems[0].sharing_params), 1)                      # today's share rows, one pair
        self.assertIsNone(solution.branch_time_decisions)
        self.assertEqual(solution.branch_time_status, "")

    def _assert_identical(self, first, second):
        self.assertEqual(first.action, second.action)
        self.assertEqual(first.cost, second.cost)
        for name in ("x_nominal", "j_nominal", "u_nominal", "slack", "planned_standoff"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)

    def test_no_vehicle_told_apart_solves_the_plain_qp(self):
        scene = SCENES["no_cutin"]                                                # one scenario: the lead only
        prediction = scene_prediction(scene)
        with_token = NairACCSMPC(design_config(scene["v_set"], 100.0))
        solution = solve(with_token, scene, prediction)
        self.assertEqual(solution.branch_time_decisions, [])
        self.assertEqual(solution.branch_time_status, "")
        self.assertTrue(np.isnan(solution.branch_time_solve_s))
        self.assertTrue(all(p.tb_delta_var is None for p in with_token._feedback_scalar_chance_qp_cache.values()))
        self._assert_identical(solution, solve(controller(scene), scene, prediction))

    def test_todays_rule_tick_solves_the_plain_qp(self):
        # scene 'cutin' with its lane-keeping scenario doubled, the copy keyed to another lead from step 5: sharing
        # would not be transitive (StructureTests), so the tick keeps today's rule
        scene = SCENES["cutin"]
        per_scenario = ("means", "covariances", "active_mask", "chance_confidence", "sigma_confidence", "lead_mode_lk",
                        "lead_vehicle_ids", "lead_mode_probability", "lead_keys")
        variant = {name: (value[:1] + value if name in per_scenario else value) for name, value in scene.items()}
        variant["probabilities"] = [0.5 * scene["probabilities"][0]] * 2 + scene["probabilities"][1:]
        variant["mode_names"] = ["copy"] + scene["mode_names"]
        variant["lead_keys"][0] = scene["lead_keys"][0][:5] + [[999, "lk"]] * 11
        prediction = scene_prediction(variant)
        with_token = controller(scene, 100.0)
        solution = solve(with_token, scene, prediction)
        self.assertEqual(solution.branch_time_status, "today_rule")
        self.assertEqual([d["t_b"] for d in solution.branch_time_decisions], [-1])
        self.assertTrue(np.isnan(solution.branch_time_solve_s))
        self._assert_identical(solution, solve(controller(scene), scene, prediction))


class FixedStepTests(unittest.TestCase):
    """The binaries fixed at a step: the coupling rows are today's share rows for a tree that splits there."""

    def test_fixed_step_is_the_tree_split_at_that_step(self):
        for name in ("cutin", "cutin_likely", "other_vehicle"):
            scene = SCENES[name]
            prediction = scene_prediction(scene)
            plain = controller(scene, allow_solver_fallback=False)
            for step, fixed in enumerate(fixed_solutions(name), start=1):
                with self.subTest(scene=name, step=step):
                    self.assertEqual(fixed.branch_time_status, "OPTIMAL")
                    self.assertEqual(decisions_by_slot(fixed)[0]["t_b"], step)
                    self.assertEqual(fixed.branch_time_penalty, 0.0)
                    tree = solve(plain, scene, split_tree(prediction, step))
                    np.testing.assert_allclose(fixed.j_nominal, tree.j_nominal, atol=1.0e-4)
                    self.assertAlmostEqual(fixed.action, tree.action, places=6)
                    assert_cost_close(self, fixed.cost, tree.cost)

    def test_fixed_at_todays_split_is_todays_qp(self):
        for name in ("cutin", "cutin_likely", "other_vehicle"):
            scene = SCENES[name]
            today = solve(controller(scene, allow_solver_fallback=False), scene, scene_prediction(scene))
            split = today.first_policy_split_step
            fixed = fixed_solutions(name)[split - 1]
            with self.subTest(scene=name, split=split):
                self.assertEqual(decisions_by_slot(fixed)[0]["k_on"], split)     # today's split step, as logged
                self.assertTrue(fixed.branch_time_today_exact)
                np.testing.assert_allclose(fixed.j_nominal, today.j_nominal, atol=1.0e-4)
                self.assertAlmostEqual(fixed.action, today.action, places=6)
                assert_cost_close(self, fixed.cost, today.cost)

    def test_later_branching_never_costs_less(self):
        for name in ("cutin", "cutin_likely", "other_vehicle"):
            costs = np.array([solution.cost for solution in fixed_solutions(name)])
            self.assertTrue(np.all(np.diff(costs) >= -5.0e-4 * costs[:-1] - 0.05), msg=name)

    def test_previous_input_is_the_same_in_every_scenario(self):
        # so equal deviations h are equal jerks (the share rows and the coupling rows compare the same thing)
        scene = SCENES["cutin"]
        prediction = scene_prediction(scene)
        ctl = NairACCSMPC(design_config(scene["v_set"], 3.0))
        context = dict(previous_accel_cmd=scene["previous_accel_cmd"], command_dt=scene["command_dt"],
                       ego_accel=scene["ego_accel"])
        first = ctl.solve(np.array(scene["ego_state"]), prediction, **context)
        second = ctl.solve(np.array(scene["ego_state"]), prediction, **context)   # warm-started from the first plan
        for solution in (first, second):
            prev_u = solution.reference.prev_u
            np.testing.assert_array_equal(prev_u, np.tile(prev_u[:1], (prev_u.shape[0], 1)))
        self.assertTrue(np.any(second.reference.prev_u != 0.0))


class BigMTests(unittest.TestCase):

    def test_big_m_is_the_width_of_the_jerk_bounds(self):
        scene = SCENES["cutin"]
        prediction = scene_prediction(scene)
        ctl = NairACCSMPC(design_config(scene["v_set"], 0.0, allow_solver_fallback=False))
        fix_branch_steps(ctl, [1] * BRANCH_TIME_SLOTS)
        solve(ctl, scene, prediction)
        np.testing.assert_array_equal(ctl._branch_time_bigm(), np.full(HORIZON, 2.5 + 5.0))
        # inside a recovery from the safety layer's fail-safe the rising bound is the fail-safe's jerk on the first
        # steps: the big-M follows the bounds of that solve
        recovering = solve(ctl, scene, prediction, previous_accel_cmd=-6.0, ego_accel=-6.0)
        self.assertIsNotNone(ctl._recovery_bounds)
        _, j_hi, _ = failsafe_recovery_envelope(ctl._ego_accel, ctl.config, scene["command_dt"])
        np.testing.assert_array_equal(ctl._branch_time_bigm(), j_hi + 2.5)
        self.assertEqual(ctl._branch_time_bigm()[1], 10.0 + 2.5)
        self.assertEqual(ctl._branch_time_bigm()[-1], 5.0 + 2.5)
        # with every binary at 1 the rows cut nothing off: the plan is the one of the tree that splits at step 1
        plain = controller(scene, allow_solver_fallback=False)
        tree = solve(plain, scene, split_tree(prediction, 1), previous_accel_cmd=-6.0, ego_accel=-6.0)
        np.testing.assert_allclose(recovering.j_nominal, tree.j_nominal, atol=1.0e-4)


class MIQPTests(unittest.TestCase):
    """The MIQP's choice against the brute force over the branching steps."""

    def _miqp(self, name, weight):
        scene = SCENES["cutin" if name == "two_vehicles" else name]
        prediction = two_vehicle_scene() if name == "two_vehicles" else scene_prediction(scene)
        ctl = controller(scene, weight, allow_solver_fallback=False)
        return ctl, solve(ctl, scene, prediction)

    def _assert_brute_force(self, name, weight, expected=None):
        ctl, solution = self._miqp(name, weight)
        decisions = decisions_by_slot(solution)
        chosen = tuple(d["t_b"] for d in decisions)
        fixed = fixed_solutions(name)
        totals = {}
        for point, fixed_solution in (fixed.items() if isinstance(fixed, dict)
                                      else (((step,), s) for step, s in enumerate(fixed, start=1))):
            totals[point] = fixed_solution.cost + sum(
                branch_penalty(weight, d["p_cut"], step) for d, step in zip(decisions, point))
        best = min(totals, key=totals.get)
        with self.subTest(scene=name, weight=weight):
            self.assertEqual(solution.branch_time_status, "OPTIMAL")
            assert_cost_close(self, solution.cost, totals[best])                  # the MIQP optimum is the minimum
            assert_cost_close(self, totals[chosen], totals[best])                 # and its steps attain it
            self.assertAlmostEqual(solution.branch_time_penalty, sum(
                branch_penalty(weight, d["p_cut"], d["t_b"]) for d in decisions), places=9)
            if expected is not None:
                self.assertEqual(best, expected)
                self.assertEqual(chosen, expected)
        return solution

    def test_scene_rebuilt_through_the_joint_builder_is_the_recorded_one(self):
        # the way the two-vehicle scene is put together reproduces a recorded scene exactly
        rebuilt, recorded = rebuilt_cutin_scene(), scene_prediction(SCENES["cutin"])
        for name in ("means", "covariances", "probabilities", "active_mask", "chance_confidence", "lead_mode_lk",
                     "lead_vehicle_ids", "lead_mode_probability"):
            np.testing.assert_array_equal(getattr(rebuilt, name), getattr(recorded, name), err_msg=name)
        self.assertEqual(rebuilt.lead_keys, recorded.lead_keys)
        np.testing.assert_array_equal(rebuilt.policy_tree.policy_groups_by_step,
                                      recorded.policy_tree.policy_groups_by_step)

    def test_brute_force_cutin(self):
        # a cut-in 1.4 s before t_cross: the TV's hypothesis (49.8 %) leads from step 5
        scene = SCENES["cutin"]
        today = solve(controller(scene, allow_solver_fallback=False), scene, scene_prediction(scene))
        cells = np.asarray(scene["lead_vehicle_ids"]) == 1446
        # its rows are active: today's plan gives up part of the time gap on the TV's cells (a priced shortfall)
        d_safe = 4.5 + 3.0 + 1.3 * today.x_nominal[:, :, 1]
        self.assertGreater(np.max((d_safe - today.planned_standoff)[cells]), 0.1)
        self._assert_brute_force("cutin", 1.0, expected=(1,))
        self._assert_brute_force("cutin", 3.0, expected=(1,))
        self._assert_brute_force("cutin", 10.0, expected=(HORIZON,))
        self._assert_brute_force("cutin", 100.0, expected=(HORIZON,))

    def test_brute_force_likely_cutin(self):
        for weight in (1.0, 3.0, 10.0, 100.0):
            self._assert_brute_force("cutin_likely", weight)

    def test_brute_force_other_vehicle(self):
        # after the cut-in, another adjacent vehicle's hypothesis (80.6 %, it does not cut in within the episode)
        for weight in (1.0, 3.0, 10.0, 100.0):
            self._assert_brute_force("other_vehicle", weight)

    def test_brute_force_two_vehicles(self):
        prediction = two_vehicle_scene()
        self.assertEqual(prediction.num_modes, 3)
        plan = branch_time_structure(prediction, prediction.policy_tree.policy_groups_by_step)
        self.assertEqual([(v["vehicle_id"], v["k_on"], v["slot"]) for v in plan["vehicles"]],
                         [(3261, 8, 0), (1446, 5, 1)])
        self.assertTrue(plan["today_exact"])
        # the grid of both steps, 15 x 15
        self._assert_brute_force("two_vehicles", 1.0, expected=(HORIZON, 1))
        self._assert_brute_force("two_vehicles", 10.0, expected=(HORIZON, 1))
        self._assert_brute_force("two_vehicles", 100.0, expected=(HORIZON, HORIZON))

    def test_weight_zero_branches_at_step_one(self):
        for name in ("cutin", "cutin_likely", "other_vehicle", "two_vehicles"):
            ctl, solution = self._miqp(name, 0.0)
            fixed = fixed_solutions(name)
            first = fixed[(1, 1)] if isinstance(fixed, dict) else fixed[0]
            with self.subTest(scene=name):
                self.assertEqual(solution.branch_time_penalty, 0.0)
                assert_cost_close(self, solution.cost, first.cost)               # the cost of branching at step 1
                np.testing.assert_allclose(solution.j_nominal, first.j_nominal, atol=1.0e-3)
        # where branching at step 1 is clearly cheaper than at step 2, the binaries say so
        for name in ("cutin", "cutin_likely"):
            fixed = fixed_solutions(name)
            self.assertGreater(fixed[1].cost - fixed[0].cost, 1.0)
            self.assertEqual([d["t_b"] for d in decisions_by_slot(self._miqp(name, 0.0)[1])], [1])

    def test_large_weight_never_branches(self):
        for name in ("cutin", "cutin_likely", "other_vehicle", "two_vehicles"):
            ctl, solution = self._miqp(name, 1.0e4)
            fixed = fixed_solutions(name)
            never = fixed[(HORIZON, HORIZON)] if isinstance(fixed, dict) else fixed[-1]
            with self.subTest(scene=name):
                self.assertEqual({d["t_b"] for d in decisions_by_slot(solution)}, {HORIZON})
                self.assertEqual(solution.branch_time_penalty, 0.0)
                assert_cost_close(self, solution.cost, never.cost)
                # one plan for every scenario
                np.testing.assert_allclose(solution.j_nominal, np.tile(solution.j_nominal[:1],
                                                                       (solution.j_nominal.shape[0], 1)), atol=TIED_TOL)

    def test_binaries_are_monotone_and_couple_the_jerks(self):
        # (scene, weight, steps tied by a binary at 0): 'cutin' at w 10 never branches; in the two-vehicle scene at
        # w 10 vehicle 3261 never branches, which ties the scenarios it alone tells apart (1 and 2) at every step
        for name, weight, expected_tied in (("cutin", 10.0, HORIZON - 1), ("two_vehicles", 10.0, HORIZON - 1),
                                            ("two_vehicles", 0.0, None)):
            ctl, solution = self._miqp(name, weight)
            delta, plan = ctl._tb_result["delta"], ctl._tb_plan
            used = [v["slot"] for v in plan["vehicles"] if v["slot"] >= 0]
            with self.subTest(scene=name, weight=weight):
                self.assertTrue(np.all(np.isin(delta, (0.0, 1.0))))
                self.assertTrue(np.all(np.diff(delta, axis=1) >= 0.0))             # delta_{k+1} >= delta_k
                self.assertTrue(np.all(delta[[s for s in range(BRANCH_TIME_SLOTS) if s not in used]] == 0.0))
                for decision in decisions_by_slot(solution):                      # t_b = min{k : delta_k = 1}, or N
                    ones = np.flatnonzero(delta[decision["slot"]])
                    self.assertEqual(decision["t_b"], HORIZON if ones.size == 0 else int(ones[0]) + 1)
                jerk = solution.j_nominal
                np.testing.assert_allclose(jerk[:, 0], jerk[0, 0], atol=1.0e-6)   # the first input is always shared
                tied = 0
                for (a, b), (slots, free) in plan["pairs"].items():
                    for step in range(1, HORIZON):
                        if free[step] + float(np.dot(slots, delta[:, step - 1])) == 0.0:
                            tied += 1
                            self.assertLessEqual(abs(jerk[a, step] - jerk[b, step]), TIED_TOL)
                        else:                                                     # never tighter than the jerk bounds
                            self.assertLessEqual(abs(jerk[a, step] - jerk[b, step]), 2.5 + 5.0)
                if expected_tied is not None:
                    self.assertEqual(tied, expected_tied)

    def test_non_monotone_binaries_have_no_solution(self):
        scene = SCENES["cutin"]
        ctl = NairACCSMPC(design_config(scene["v_set"], 0.0, allow_solver_fallback=False))

        def bounds(plan):
            lo, hi = np.zeros((BRANCH_TIME_SLOTS, HORIZON - 1)), np.zeros((BRANCH_TIME_SLOTS, HORIZON - 1))
            hi[0] = 1.0
            lo[0, 2] = 1.0       # delta_3 = 1
            hi[0, 4] = 0.0       # delta_5 = 0
            return lo, hi
        ctl._branch_time_bounds = bounds
        with self.assertRaises(RuntimeError):
            solve(ctl, scene, scene_prediction(scene))
        self.assertNotEqual(ctl._tb_result["status"], "OPTIMAL")
        self.assertIsNone(ctl._tb_result["delta"])

    def test_log_fields(self):
        scene = SCENES["cutin"]
        ctl, solution = self._miqp("cutin", 3.0)
        self.assertEqual(len(solution.branch_time_decisions), 1)
        decision = solution.branch_time_decisions[0]
        self.assertEqual(set(decision), {"vehicle_id", "labels", "p_cut", "k_on", "slot", "t_b"})
        self.assertEqual(decision["vehicle_id"], 1446)
        self.assertEqual(decision["p_cut"], scene["lead_mode_probability"][1][5])     # the gate's and zconf's p
        self.assertEqual(decision["k_on"], 5)
        self.assertEqual(decision["k_on"], solution.first_policy_split_step)          # today's split step
        self.assertEqual(solution.branch_time_status, "OPTIMAL")
        self.assertGreater(solution.branch_time_solve_s, 0.0)
        self.assertTrue(solution.branch_time_today_exact)

    def test_non_optimal_status_falls_back(self):
        scene = SCENES["cutin"]
        ctl = NairACCSMPC(design_config(scene["v_set"], 3.0, gurobi_time_limit=1.0e-6))
        solution = solve(ctl, scene, scene_prediction(scene))
        self.assertEqual(solution.solve_path, "fallback")
        self.assertEqual(solution.branch_time_status, "TIME_LIMIT")
        self.assertEqual([d["t_b"] for d in solution.branch_time_decisions], [-1])
        self.assertTrue(np.isfinite(solution.branch_time_solve_s))
        self.assertTrue(np.isnan(solution.branch_time_penalty))


if __name__ == "__main__":
    unittest.main()
