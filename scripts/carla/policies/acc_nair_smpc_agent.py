import json
import os
import re
import sys
import time

import carla
import numpy as np

CARLA_ROOT = os.getenv("CARLA_ROOT")
if CARLA_ROOT is None:
    raise ValueError("CARLA_ROOT must be defined.")

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
rootdir = os.path.abspath(__file__).split('scripts')[0]
sys.path.append(rootdir)
from rasterizer.agent_history import AgentHistory
from utils import frenet_trajectory_handler as fth
from utils.acc_nair_smpc import (
    BRAKE_DISTANCE_BOUND_HARD_BAND,
    BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL,
    FailsafeBranch,
    failsafe_recovery_floor,
    limit_accel_command,
    MultimodalLeadPrediction,
    NairACCSMPC,
    PASS_MIQP_FAR_BEHIND_M,
    PASS_MIQP_SLOTS,
    PolicySharingTree,
    RISK_FIXED,
    RISK_OPTIMIZED_ETA,
    RISK_PROBABILITY_WEIGHTED,
    SAFETY_BRAKE_DISTANCE,
    SAFETY_CONFIDENCE_CHANCE,
    SAFETY_NOMINAL_SAFE_DISTANCE,
    SAFETY_SCALAR_CHANCE,
    VARIANT_FIXED_RISK,
    VARIANT_MULTIMODAL_OL,
    VARIANT_PROPOSED,
    make_constant_accel_lead_prediction,
    original_nair_acc_config,
    required_standoff,
    safety_function,
)
from utils.low_level_control import (
    IdealLongitudinalActuator, LowLevelControl, lane_keeping_steer)
from utils.synthetic_lk_cutin_prediction import (
    SyntheticCutInConfig,
    SyntheticLaneKeepingCutInPredictor,
)
from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import relation_from_waypoints
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import (
    CTX_GATE_VY_MIN_HISTORY_S,
    CTX_GATE_VY_WINDOW_S,
    INTENTION_NAMES,
    REL_EGO_LANE,
    REL_LEFT_ADJACENT,
    REL_OTHER,
    REL_RIGHT_ADJACENT,
    decision_point_branch_step,
    entry_ego_displacement,
    pass_cap_displacement,
    pass_maxacc_displacement,
)
from predictor.stdan_vel.acc_adapter import STDANVelACCAdapter
from predictor.lstm.acc_adapter import LSTMACCAdapter
from predictor.lstm_vel.acc_adapter import STDAN_LIKE_SHRINK, LSTMVelACCAdapter
from predictor.iaimm_kf.acc_adapter import IAIMMKFACCAdapter


class ACCNairSMPCAgent(object):
    """CARLA adapter for the 1D ACC Nair-style multimodal SMPC controller."""

    # Predicted seconds-to-ego-lane at which the gap re-establishment clock starts.
    # Waiting for the lane assignment to flip is too late: the braking is over by
    # then. One second ahead lands the relaxation on the deceleration itself.
    GAP_RECOVERY_ONSET_TLC_S = 1.0
    # Below this probability a predicted mode is dropped outright, for every
    # STDAN policy (an explicit ``cutin_gate<value>`` overrides it).  For the
    # chance constraint it is Benciolini's vanishing: the ellipse collapses to
    # a point as beta -> 0 and a point can be driven around, whereas the 1-D
    # counterpart still orders the ego behind the hypothetical cut-in even at
    # zero standoff.  For the plain multimodal policy a 1-2 % hypothesis held
    # at full standoff kept the ego behind a lead that had already left the
    # lane (cut-out, 2026-09-05).  Lane-keeping neighbours score <= 0.05 in the
    # 2026-09-04 sweep; 0.1 keeps a margin above that.
    CUTIN_CHANCE_VANISH_BELOW = 0.1

    def __init__(
            self,
            vehicle,
            goal_location,
            nominal_speed_mps=8.0,
            dt=0.2,
            N=8,
            N_modes=2,
            smpc_config="proposed"):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.DT = dt
        self.N = N
        self.nominal_speed = nominal_speed_mps
        self.goal_reached = False
        self.smpc_config = str(smpc_config)
        self.mode_probabilities = self._parse_mode_probabilities(smpc_config)
        self.predictor_type = self._parse_predictor_type(smpc_config)
        self.best_mode_only = self._parse_best_mode_only(smpc_config)
        threshold = self._parse_cutin_probability_threshold(smpc_config)
        if threshold is None:
            threshold = (self.CUTIN_CHANCE_VANISH_BELOW
                         if self.predictor_type in ("stdan_3int", "stdan_vel", "iaimm_kf") else 0.0)
        self.cutin_probability_threshold = threshold
        self.cutin_clearance_ramp_ref = self._parse_cutin_clearance_ramp_ref(smpc_config)
        self.cutin_chance_ref = self._parse_cutin_chance_ref(smpc_config)
        self.intention_lpf_s = self._parse_intention_lpf_s(smpc_config)
        self.track_smooth_samples = self._parse_track_smooth_samples(smpc_config)
        if self.cutin_chance_ref > 0.0 and self.cutin_clearance_ramp_ref > 0.0:
            raise ValueError(
                "cutin_ramp and cutin_chance both set the cut-in standoff scale; use one")
        self.chance_sigma_cap = self._parse_chance_sigma_cap(smpc_config)
        self.standoff_shortfall_weight = self._parse_standoff_shortfall_weight(smpc_config)
        self.standoff_shortfall_on_lk = self._parse_standoff_shortfall_on_lk(smpc_config)
        self.standoff_shortfall_lk_cutin_only = self._parse_standoff_shortfall_lk_cutin_only(smpc_config)
        self.standoff_shortfall_cell_price = self._parse_standoff_shortfall_cell_price(smpc_config)
        if self.cutin_chance_ref <= 0.0 and (
                self.chance_sigma_cap is not None or self.standoff_shortfall_weight is not None):
            raise ValueError(
                "sigmacap and dopt modify the cutin_chance constraint; add cutin_chance<beta_ref>")
        if self.standoff_shortfall_on_lk and self.predictor_type != "stdan_3int":
            # Only the stdan_3int joint prediction marks lane-keeping cells; any
            # other predictor would silently price cut-in cells only.
            raise ValueError("dopt<w>lk needs the stdan_3int predictor")
        if self.standoff_shortfall_cell_price and self.predictor_type != "stdan_3int":
            # Only the stdan_3int joint prediction gives the cell probability; any
            # other predictor would silently price as plain dopt<w>.
            raise ValueError("pricep2 needs the stdan_3int predictor")
        self.lk_entry_as_cutin = self._parse_lk_entry_as_cutin(smpc_config)
        if self.lk_entry_as_cutin and self.predictor_type != "stdan_3int":
            raise ValueError("trajcutin needs the stdan_3int predictor")
        self.no_cutout = self._parse_no_cutout(smpc_config)
        self.lk_sigma = self._parse_lk_sigma(smpc_config)
        if (self.no_cutout or self.lk_sigma) and self.predictor_type != "stdan_3int":
            raise ValueError("nocutout and lksigma need the stdan_3int predictor")
        if self.lk_sigma and not self.cutin_chance_ref:
            # The sigma term takes beta_j = min(p, beta_ref) from cutin_chance.
            raise ValueError("lksigma needs cutin_chance<beta_ref>")
        self.full_footprint = self._parse_full_footprint(smpc_config)
        if self.full_footprint:
            if self.predictor_type != "stdan_3int":
                raise ValueError("fullfoot needs the stdan_3int predictor")
            if not self.cutin_chance_ref or self.standoff_shortfall_weight is None:
                # It replaces dopt's alpha_j on the footprint of cut-in chance cells.
                raise ValueError("fullfoot needs cutin_chance<beta_ref> and dopt<w>")
        self.entry_pass = self._parse_entry_pass(smpc_config)
        self.entry_pass_margin = self._parse_entry_pass_margin(smpc_config)
        if self.entry_pass and self.predictor_type != "stdan_3int":
            raise ValueError("entrypass needs the stdan_3int predictor")
        # passmiqp<m> (2026-11-04, item 8): the pass / yield choice inside the optimization; it replaces
        # entrypass's test on the remembered plan by a pre-filter on the no-acceleration displacement
        self.pass_miqp_margin = self._parse_pass_miqp(smpc_config)
        if self.pass_miqp_margin is not None:
            if self.predictor_type != "stdan_3int":
                raise ValueError("passmiqp needs the stdan_3int predictor")
            if not self.cutin_chance_ref or self.standoff_shortfall_weight is None:
                raise ValueError("passmiqp needs cutin_chance<beta_ref> and dopt<w>")
        # passnocap (2026-11-04): no acceleration cap on the pass branch; the candidate test reads the
        # maximum-acceleration displacement
        self.pass_no_cap = self._parse_pass_no_cap(smpc_config)
        if self.pass_no_cap and self.pass_miqp_margin is None:
            raise ValueError("passnocap needs passmiqp<m>")
        self.lead_merge = self._parse_lead_merge(smpc_config)
        if self.lead_merge and self.predictor_type != "stdan_3int":
            raise ValueError("leadmerge needs the stdan_3int predictor")
        self.merge_first = self._parse_merge_first(smpc_config)
        if self.merge_first and not self.lead_merge:
            raise ValueError("mergefirst needs leadmerge")
        self.lk_hold = self._parse_lk_hold(smpc_config)
        if self.lk_hold and self.predictor_type != "stdan_3int":
            raise ValueError("lkhold needs the stdan_3int predictor")
        self.no_lk_entry = self._parse_no_lk_entry(smpc_config)
        if self.no_lk_entry and self.predictor_type != "stdan_3int":
            raise ValueError("nolkentry needs the stdan_3int predictor")
        # occin<eps> (2026-11-05): a cut-in cell is active where p * P(body overlaps the ego's path) > eps, which
        # needs a probabilistic ego-path test of the predicted trajectories (predict_acc(lane_membership_prob_fn=));
        # only the NGSIM replay has one, and inside the agent a predictor exception would be swallowed silently
        # (the synthetic prediction, AGENTS.md 8.12), so the token is refused here
        self.occupancy_risk = self._parse_occupancy_risk(smpc_config)
        if self._parse_occupancy_pin_gate(smpc_config) and self.occupancy_risk is None:
            raise ValueError("pingate needs occin<eps>")
        if self.occupancy_risk is not None:
            if self.predictor_type != "stdan_3int":
                raise ValueError("occin<eps> needs the stdan_3int predictor")
            if not self.lk_entry_as_cutin:
                raise ValueError("occin<eps> needs trajcutin")
            raise ValueError("occin<eps> needs a probabilistic ego-path test of the predicted trajectories "
                             "(lane_membership_prob_fn); the CARLA agent has none (NGSIM replay only)")
        # pcal (2026-11-05): the calibration maps are fitted on natural NGSIM traffic and read NGSIM's lateral
        # features; the replay applies them, and CARLA has neither a map nor those features
        if self._parse_probability_calibration(smpc_config):
            raise ValueError("pcal's calibration maps are fitted on NGSIM traffic (NGSIM replay only); "
                             "CARLA has no map")
        # bcts (2026-11-06): the same for the bias-corrected temperature scaling fitted on NGSIM
        if self._parse_bcts_calibration(smpc_config):
            raise ValueError("bcts's calibration is fitted on NGSIM traffic (NGSIM replay only); CARLA has none")
        # entrypass: (time_s, scenario names, probabilities, planned speeds) of the last solve
        self._entry_previous_plan = None
        self.branch_decision_brake = self._parse_branch_decision_brake(smpc_config)
        if self.branch_decision_brake is not None and self.predictor_type != "stdan_3int":
            raise ValueError("branchdec needs the stdan_3int predictor")
        self.context_trust = self._parse_context_trust(smpc_config)
        if self.context_trust is not None:
            if self.predictor_type != "stdan_3int":
                raise ValueError("ctx needs the stdan_3int predictor")
            if not 0.0 <= self.context_trust <= 1.0:
                raise ValueError("ctx<lambda> takes lambda in [0, 1]")
        self.lateral_gate = self._parse_lateral_gate(smpc_config)
        if self.lateral_gate and self.context_trust is None:
            raise ValueError("latgate replaces ctx<lambda>'s offset ramp; add ctx<lambda>")
        self.cutin_clearance_tlc_ref = self._parse_cutin_clearance_tlc_ref(smpc_config)
        self.gap_recovery_s = self._parse_gap_recovery_s(smpc_config)
        # Drive the ego as the exact double integrator the MPC models.  The
        # throttle/brake maps in LowLevelControl realise about 1.5x the
        # commanded acceleration plus 0.9 m/s^2 at zero command on this
        # vehicle, which turns the following loop into a limit cycle; the
        # ``carla_actuator`` token keeps that path for comparison.
        self.ideal_actuator = "carla_actuator" not in self.smpc_config
        self._ideal_actuator = IdealLongitudinalActuator(vehicle) if self.ideal_actuator else None
        # vehicle id -> time the gap re-establishment clock started for it
        self._ego_lane_entry_time = {}
        # vehicle id -> gap / full standoff when that clock started; the
        # relaxation accepts the gap the vehicle actually arrived with
        self._gap_recovery_start_scale = {}
        # vehicle id -> predicted seconds until it reaches the ego lane, from the
        # previous cycle.  Read one step late because the recovery scale has to be
        # decided before the predictor that produces it runs.
        self._predicted_lane_entry_s = {}
        # latgate: vehicle id -> [(time_s, lateral speed along its lane's left
        # normal)] over the last CTX_GATE_VY_WINDOW_S, for every target; and
        # vehicle id -> whether its lateral gate was latched at the previous tick
        # (only vehicles processed as adjacent targets at that tick).
        self._ctx_lateral_speed_samples = {}
        self._ctx_latched = {}
        self.enable_wandb_logging = os.getenv("ACC_NAIR_WANDB", "0") == "1"
        self._wandb = None
        self.goal_location = goal_location
        self._frenet_traj = None
        self._stdan_model_yaw = 0.0
        self._ego_route_debug = {"source": "uninitialized"}

        variant = self._parse_variant(smpc_config)
        self.controller_config = original_nair_acc_config(
            horizon=N,
            dt=dt,
            desired_speed=nominal_speed_mps,
            num_modes=self._parse_controller_num_modes(smpc_config, N_modes),
            controller_variant=variant,
            gurobi_time_limit=0.15,
        )
        self.controller_config.safety_constraint_mode = self._parse_safety_constraint_mode(smpc_config)
        self.controller_config.brake_distance_bound_mode = (
            self._parse_brake_distance_bound_mode(smpc_config)
        )
        self.controller_config.risk_allocation_mode = self._parse_risk_allocation_mode(smpc_config)
        self.controller_config.optimize_k = self._parse_optimize_k(smpc_config)
        r_jerk = self._parse_r_jerk(smpc_config)
        if r_jerk is not None:
            self.controller_config.r_jerk = r_jerk
        r_move = self._parse_r_move(smpc_config)
        if r_move is not None:
            self.controller_config.r_move = r_move
        jerk_limit = self._parse_jerk_limit(smpc_config)
        if jerk_limit is not None:
            self.controller_config.jerk_limit = jerk_limit
        # amin<x> amax<x> jlimup<x> fsdecel<x> fsjerk<x> (2026-11-11, unified limits)
        for name, value in self._parse_limit_tokens(smpc_config).items():
            setattr(self.controller_config, name, value)
        r_a = self._parse_r_a(smpc_config)
        if r_a is not None:
            self.controller_config.r_a = r_a
        self.controller_config.accel_lag_tau = self._parse_accel_lag_tau(smpc_config)
        self.controller_config.jerk_input = self._parse_jerk_input(smpc_config)
        self.controller_config.relative_brake_decel = self._parse_relative_brake_decel(smpc_config)
        self.controller_config.slack_l1_weight = self._parse_slack_l1_weight(smpc_config)
        self.controller_config.terminal_cost_scale = self._parse_terminal_cost_scale(smpc_config)
        self.controller_config.floor_risk = self._parse_floor_risk(smpc_config)
        if self.controller_config.floor_risk is not None and not self.full_footprint:
            # Its floor replaces the one fullfoot gives a cut-in chance cell, L + d0 + q(beta_j) sigma.
            raise ValueError("floorrisk<eps> needs fullfoot (with cutin_chance<beta_ref> and dopt<w>)")
        self.controller_config.floor_risk_hypothesis = self._parse_floor_risk_hypothesis(smpc_config)
        if self.controller_config.floor_risk_hypothesis and self.controller_config.floor_risk is None:
            raise ValueError("floorhyp needs floorrisk<eps>")
        self.controller_config.floor_risk_top = self._parse_floor_risk_top(smpc_config)
        if self.controller_config.floor_risk_top and not self.controller_config.floor_risk_hypothesis:
            raise ValueError("floortop needs floorhyp")
        self.controller_config.lane_keep_risk = self._parse_lane_keep_risk(smpc_config)
        if self.controller_config.lane_keep_risk is not None and not (
                self._parse_lk_sigma(smpc_config) and self._parse_no_cutout(smpc_config)):
            # lksigma gives the lane-keeping cells their sigma term, nocutout their p = 1 (so z always exists)
            raise ValueError("lkonesided<eps> needs lksigma and nocutout")
        self.controller_config.price_row_no_sigma = self._parse_price_row_no_sigma(smpc_config)
        self.controller_config.failsafe_branch = self._parse_failsafe_branch(smpc_config)
        if self.controller_config.failsafe_branch and not self._parse_safety_layer(smpc_config):
            # The branch is the layer's fail-safe against the layer's checked members.
            raise ValueError("safebranch needs safelayer")
        # safelayer: the MPC starts from the fail-safe's acceleration and plans the recovery (item 7, 2026-11-04)
        self.controller_config.failsafe_recovery = self._parse_safety_layer(smpc_config)
        # entrant<b> (item 12, 2026-11-04): the layer covers close entrants under the braking bound -b; with
        # safebranch they lead the branch too
        entrant_decel = self._parse_entrant_decel(smpc_config)
        if entrant_decel is not None and not self._parse_safety_layer(smpc_config):
            raise ValueError("entrant<b> needs safelayer")
        self.controller_config.failsafe_entrant_decel = (
            entrant_decel if self.controller_config.failsafe_branch else None)
        self.controller_config.standoff_shortfall_weight = self.standoff_shortfall_weight
        self.controller_config.standoff_shortfall_on_lk = self.standoff_shortfall_on_lk
        self.controller_config.standoff_shortfall_lk_cutin_only = self.standoff_shortfall_lk_cutin_only
        self.controller_config.standoff_shortfall_lk_release_sigma = self._parse_lk_release_sigma(smpc_config)
        if self.controller_config.standoff_shortfall_lk_release_sigma and not self.lk_sigma:
            # without lksigma the lane-keeping cells have no sigma term, so the token would change nothing
            raise ValueError("lkrsigma needs lksigma")
        self.controller_config.standoff_shortfall_cell_price = self.standoff_shortfall_cell_price
        self.controller_config.pass_miqp_margin = self.pass_miqp_margin
        self.controller_config.pass_miqp_no_cap = self.pass_no_cap
        # cvar<eps>_<c> (2026-11-05): a CVaR bound on the cut-in chance cells' shortfall in place of their price
        cutin_cvar = self._parse_cutin_cvar(smpc_config)
        if cutin_cvar is not None and not self.full_footprint:
            # its hard floor is fullfoot's L + d0 + q(beta_j) sigma (with cutin_chance<beta_ref> and dopt<w>)
            raise ValueError("cvar<eps>_<c> needs fullfoot (with cutin_chance<beta_ref> and dopt<w>)")
        self.controller_config.cutin_cvar_risk, self.controller_config.cutin_cvar_bound = (
            (None, None) if cutin_cvar is None else cutin_cvar)
        # onerow (2026-11-12): one priced row on every active lead cell, no floor row, no sigma term
        self.controller_config.one_row = self._parse_one_row(smpc_config)
        if self.controller_config.one_row:
            if (not self.cutin_chance_ref or self.standoff_shortfall_weight is None or self.standoff_shortfall_on_lk
                    or not self.full_footprint):
                raise ValueError("onerow needs plain dopt<w> (not dopt<w>lk or dopt<w>lkr), chancecells and fullfoot")
            conflicts = [token for token, present in (
                ("floorrisk<eps>", self.controller_config.floor_risk is not None),
                ("floorhyp", self.controller_config.floor_risk_hypothesis),
                ("floortop", self.controller_config.floor_risk_top),
                ("nopricesigma", self.controller_config.price_row_no_sigma),
                ("lksigma", self.lk_sigma),
                ("lkonesided<eps>", self.controller_config.lane_keep_risk is not None),
                ("lkrsigma", self.controller_config.standoff_shortfall_lk_release_sigma),
                ("cvar<eps>_<c>", cutin_cvar is not None),
                ("passmiqp<m>", self.pass_miqp_margin is not None),
                ("relbrake<b>", self.controller_config.relative_brake_decel is not None),
                ("safebranch", self.controller_config.failsafe_branch)) if present]
            if conflicts:
                raise ValueError("onerow has one row per cell without a sigma term or a floor row; not with "
                                 + ", ".join(conflicts))
        # zconf<b> (2026-11-12): onerow's row with the sigma term z sigma, z = Phi^-1(min(p, b)); the reference keeps d_safe
        self.controller_config.one_row_z_conf = self._parse_one_row_z_conf(smpc_config)
        if self.controller_config.one_row_z_conf is not None and not self.controller_config.one_row:
            raise ValueError("zconf<b> needs onerow")
        # nocap (2026-11-12): onerow's shortfall without the cap tau v
        self.controller_config.one_row_no_cap = self._parse_one_row_no_cap(smpc_config)
        if self.controller_config.one_row_no_cap and not self.controller_config.one_row:
            raise ValueError("nocap needs onerow")
        self.controller_config.__post_init__()
        self.controller = NairACCSMPC(self.controller_config)
        # safelayer (2026-11-02): PC-ACC's safety layer around the MPC's command, on PC-ACC's measured lane membership
        self.safety_layer = None
        if self._parse_safety_layer(smpc_config):
            from utils.pcacc import LaneMembershipTracker
            from utils.safety_layer import SafetyLayer
            self.safety_layer = SafetyLayer(self.controller_config.a_max, entrant_decel=entrant_decel,
                                            failsafe_decel=self.controller_config.failsafe_decel,
                                            failsafe_jerk=self.controller_config.failsafe_jerk)
            self._safety_tracker = LaneMembershipTracker(self.safety_layer.cfg)
            self._safety_clock = 0.0
        self.synthetic_predictor = SyntheticLaneKeepingCutInPredictor(
            SyntheticCutInConfig(
                horizon=N,
                dt=dt,
                probabilities=self.mode_probabilities,
                ego_lane_d=0.0,
            )
        )
        self.stdan_predictor = None
        self.stdan_history_secs = None
        self.stdan_history_closeness_eps = None
        self.agent_history = None
        if self.predictor_type in ("stdan_3int", "stdan_vel", "lstm", "lstm_vel", "iaimm_kf"):
            if self.predictor_type == "iaimm_kf":
                # Tuning knobs as a JSON dict, e.g. '{"lat_speedup": 2, "proj_horizon_s": 8}';
                # keys are IAIMMKFACCAdapter constructor arguments.
                self.stdan_predictor = IAIMMKFACCAdapter(
                    history=3.0,
                    future=max(3.0, float(N) * float(dt)),
                    dt=0.1,
                    call_dt=0.05,          # the 20 Hz control tick, not the MPC dt: sets the KF cadence
                    lane_width=2.0 * self.EGO_LANE_HALF_WIDTH_M,
                    **json.loads(os.getenv("ACC_NAIR_IAIMM_CFG", "{}")),
                )
            elif self.predictor_type == "lstm_vel":
                self.stdan_predictor = LSTMVelACCAdapter(
                    ckpt_path=os.getenv("ACC_NAIR_LSTM_VEL_CKPT",
                                        "predictor/lstm_vel/ckpt/best_model.pt"),
                    history=3.0,
                    future=5.0,
                    dt=0.1,
                    anchor_target=(STDAN_LIKE_SHRINK if "stdanlike" in str(smpc_config)
                                   else None),
                )
            elif self.predictor_type == "lstm":
                self.stdan_predictor = LSTMACCAdapter(
                    ckpt_path=os.getenv("ACC_NAIR_LSTM_CKPT", "predictor/lstm/ckpt/best_model.pt"),
                    history=3.0,
                    future=5.0,
                    dt=0.1,
                )
            elif self.predictor_type == "stdan_vel":
                self.stdan_predictor = STDANVelACCAdapter(
                    ckpt_path="predictor/stdan_vel/checkpoints/0113_ratio211/best_model.pt",
                    history=3.0,
                    future=5.0,
                    dt=0.1,
                    mc_dropout=False,
                )
            else:
                stdan_ckpt = os.getenv(
                    "ACC_NAIR_STDAN_CKPT",
                    os.getenv(
                        "ACC_NAIR_STDAN_3INT_CKPT",
                        "predictor/stdan_3int_signed_tcross_velint/checkpoints_out30/best_model.pt",
                    ),
                )
                self.stdan_predictor = STDAN3IntACCAdapter(
                    ckpt_path=stdan_ckpt,
                    history=3.0,
                    future=max(3.0, float(N) * float(dt)),
                    dt=0.1,
                    mc_dropout=False,
                    intention_lpf_s=self.intention_lpf_s,
                )
            self.stdan_history_secs = [
                round(i * self.stdan_predictor.dt, 2)
                for i in range(self.stdan_predictor.full_length - 1, -1, -1)
            ]
            self.stdan_history_closeness_eps = self.stdan_predictor.dt / 2.0 + 1e-6
            history_max_length = max(
                100,
                int(np.ceil((max(self.stdan_history_secs) + self.stdan_predictor.dt) / max(self.DT, 1.0e-6))) + 5,
            )
            self.agent_history = AgentHistory(self.world.get_actors(), history_max_length=history_max_length)
        self._low_level_control = LowLevelControl(vehicle)
        self.policy_log = []
        self._prev_actual_accel_sample = None
        self._prev_accel_cmd = 0.0
        self._prev_accel_cmd_time_s = None

    def done(self):
        return self.goal_reached

    @staticmethod
    def _build_current_lane_reference(
            init_waypoint,
            goal_location,
            ego_location,
            vehicle_yaw_deg,
            step_distance=1.0):
        if init_waypoint is None:
            raise RuntimeError("Could not find a driving waypoint for ACC ego route.")

        try:
            goal_distance = ego_location.distance(goal_location)
        except Exception:
            goal_distance = 0.0
        target_distance = max(80.0, float(goal_distance) + 30.0)

        direction, direction_score = ACCNairSMPCAgent._select_lane_chain_direction(
            init_waypoint,
            goal_location,
            vehicle_yaw_deg,
            step_distance,
        )
        if direction_score < 0.5:
            way_s, way_xy, way_yaw = ACCNairSMPCAgent._build_straight_heading_reference(
                ego_location,
                vehicle_yaw_deg,
                target_distance,
                step_distance,
            )
            debug = {
                "source": "straight_heading_fallback",
                "direction": direction,
                "direction_score": float(direction_score),
            }
            return way_s, way_xy, way_yaw, debug

        waypoints = [init_waypoint]
        distance = 0.0
        current = init_waypoint
        while distance < target_distance:
            candidates = ACCNairSMPCAgent._lane_chain_candidates(
                current, direction, step_distance)
            if not candidates:
                break
            next_waypoint = ACCNairSMPCAgent._select_current_lane_candidate(
                current, candidates)
            distance += next_waypoint.transform.location.distance(
                current.transform.location)
            waypoints.append(next_waypoint)
            current = next_waypoint

        if len(waypoints) < 2:
            raise RuntimeError("Could not extend ACC ego current-lane route.")
        way_s, way_xy, way_yaw = ACCNairSMPCAgent._waypoints_to_directional_frenet_reference(
            waypoints)
        ego_start_xy = np.array([ego_location.x, -ego_location.y], dtype=float)
        way_xy = way_xy + (ego_start_xy - way_xy[0])
        if not ACCNairSMPCAgent._reference_aligns_with_vehicle_heading(
                way_xy, vehicle_yaw_deg):
            way_s, way_xy, way_yaw = ACCNairSMPCAgent._build_straight_heading_reference(
                ego_location,
                vehicle_yaw_deg,
                target_distance,
                step_distance,
            )
            debug = {
                "source": "straight_heading_alignment_fallback",
                "direction": direction,
                "direction_score": float(direction_score),
            }
            return way_s, way_xy, way_yaw, debug

        debug = {
            "source": "current_lane_waypoints",
            "direction": direction,
            "direction_score": float(direction_score),
        }
        return way_s, way_xy, way_yaw, debug

    @staticmethod
    def _select_lane_chain_direction(
            init_waypoint,
            goal_location,
            vehicle_yaw_deg,
            step_distance):
        start = init_waypoint.transform.location
        del goal_location
        yaw = np.radians(float(vehicle_yaw_deg))
        heading_vec = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)

        scores = []
        for direction in ("next", "previous"):
            candidates = ACCNairSMPCAgent._lane_chain_candidates(
                init_waypoint, direction, step_distance)
            if not candidates:
                continue
            waypoint = ACCNairSMPCAgent._select_current_lane_candidate(
                init_waypoint, candidates)
            loc = waypoint.transform.location
            step_vec = np.array([
                float(loc.x - start.x),
                float(loc.y - start.y),
            ], dtype=float)
            denom = max(
                np.linalg.norm(heading_vec) * np.linalg.norm(step_vec),
                1.0e-6,
            )
            scores.append((float(np.dot(heading_vec, step_vec) / denom), direction))

        if not scores:
            raise RuntimeError("Could not find next/previous ACC ego lane waypoints.")
        score, direction = max(scores, key=lambda item: item[0])
        return direction, score

    @staticmethod
    def _lane_chain_candidates(waypoint, direction, step_distance):
        if direction == "next":
            return waypoint.next(step_distance)
        if direction == "previous" and hasattr(waypoint, "previous"):
            return waypoint.previous(step_distance)
        return []

    @staticmethod
    def _select_current_lane_candidate(current_waypoint, next_waypoints):
        same_lane = [
            waypoint for waypoint in next_waypoints
            if (waypoint.road_id == current_waypoint.road_id
                and waypoint.lane_id == current_waypoint.lane_id)
        ]
        if same_lane:
            return same_lane[0]

        same_lane_id = [
            waypoint for waypoint in next_waypoints
            if waypoint.lane_id == current_waypoint.lane_id
        ]
        if same_lane_id:
            return same_lane_id[0]

        return next_waypoints[0]

    @staticmethod
    def _waypoints_to_directional_frenet_reference(waypoints):
        way_xy = np.array([
            [
                waypoint.transform.location.x,
                -waypoint.transform.location.y,
            ]
            for waypoint in waypoints
        ], dtype=float)
        diff_xy = np.diff(way_xy, axis=0)
        segment_lengths = np.sqrt(np.sum(np.square(diff_xy), axis=1))
        valid = segment_lengths > 1.0e-6
        if not np.any(valid):
            raise RuntimeError("ACC ego current-lane route has no valid segments.")
        way_s = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        segment_yaw = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])
        way_yaw = np.empty(way_xy.shape[0], dtype=float)
        way_yaw[:-1] = segment_yaw
        way_yaw[-1] = segment_yaw[-1]
        return way_s, way_xy, way_yaw

    @staticmethod
    def _build_straight_heading_reference(
            ego_location,
            vehicle_yaw_deg,
            target_distance,
            step_distance):
        way_s = np.arange(0.0, target_distance + step_distance, step_distance)
        psi = -fth.fix_angle(np.radians(float(vehicle_yaw_deg)))
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        start_xy = np.array([ego_location.x, -ego_location.y], dtype=float)
        way_xy = start_xy[None, :] + way_s[:, None] * heading_rhs[None, :]
        way_yaw = np.full(way_s.shape, psi, dtype=float)
        return way_s, way_xy, way_yaw

    @staticmethod
    def _reference_aligns_with_vehicle_heading(way_xy, vehicle_yaw_deg):
        if len(way_xy) < 2:
            return False
        psi = -fth.fix_angle(np.radians(float(vehicle_yaw_deg)))
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        step_rhs = way_xy[1] - way_xy[0]
        denom = max(np.linalg.norm(heading_rhs) * np.linalg.norm(step_rhs), 1.0e-6)
        return float(np.dot(heading_rhs, step_rhs) / denom) >= 0.5

    def _ensure_frenet_reference_initialized(self):
        if self._frenet_traj is not None:
            return

        carla_map = self.world.get_map()
        vehicle_transform = self.vehicle.get_transform()
        init_waypoint = carla_map.get_waypoint(
            vehicle_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        way_s, way_xy, way_yaw, self._ego_route_debug = self._build_current_lane_reference(
            init_waypoint,
            self.goal_location,
            vehicle_transform.location,
            vehicle_transform.rotation.yaw,
        )
        self._frenet_traj = fth.FrenetTrajectoryHandler(
            way_s, way_xy, way_yaw, s_resolution=0.5)
        self._stdan_model_yaw = float(self._frenet_traj.trajectory[0, 3])

    def run_step(self, pred_dict):
        del pred_dict

        x, y, psi, speed = self._ego_state_rhs()
        self._ensure_frenet_reference_initialized()
        s_ego, ey, epsi = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
        z0 = np.array([x, y, psi, speed])
        u0 = np.array([self.controller_config.a_min, 0.0])
        solve_time = np.nan
        is_feasible = False
        time_s = self._elapsed_seconds()
        previous_accel_cmd = self._prev_accel_cmd
        command_dt = self._command_dt(time_s)
        raw_action = self.controller_config.a_min
        command_jerk = None

        if self.goal_reached or self._frenet_traj.reached_trajectory_end(s_ego, resolution=5.0):
            self.goal_reached = True
            raw_action = self.controller_config.a_min
            action = self._limit_accel_command(raw_action, previous_accel_cmd, command_dt)
            v_des = max(0.0, speed + action * self.DT)
            df_des = 0.0
            u0 = np.array([action, df_des])
        else:
            prediction_bundle = self._prediction_bundle(s_ego, speed)
            safety_members = None
            failsafe_branch = None
            if self.safety_layer is not None and self.controller_config.failsafe_branch:
                # safebranch: the layer observes and classifies before the solve, which takes its checked
                # members as the fail-safe branch's lead; the same classification decides after the solve
                # (the layer is never classified twice in a tick)
                safety_members = self._classify_safety_layer(
                    time_s, s_ego, ey, speed, previous_accel_cmd, command_dt)
                failsafe_branch = FailsafeBranch.from_members(
                    safety_members[0], safety_members[1], self.controller_config.failsafe_entrant_decel)
            # safelayer: after a fail-safe the MPC starts from the issued command, raised at low speed to the
            # floor from which the recovery keeps v >= 0 (_mpc_previous_accel); the limiter below keeps its
            # window around the issued command (its recovery window admits the MPC's ramp)
            mpc_previous_accel = self._mpc_previous_accel(previous_accel_cmd, speed, command_dt)
            start = time.time()
            solution = self.controller.solve(
                np.array([s_ego, speed]),
                prediction_bundle["prediction"],
                previous_accel_cmd=mpc_previous_accel,
                command_dt=command_dt,
                failsafe_branch=failsafe_branch,
            )
            solve_time = time.time() - start
            if self.entry_pass and time_s is not None:
                planned = prediction_bundle["prediction"]
                self._entry_previous_plan = (
                    time_s, list(planned.mode_names or []), np.asarray(planned.probabilities, dtype=float),
                    np.asarray(solution.x_nominal, dtype=float)[:, :, 1])
            raw_action = float(solution.action)
            action = self._limit_accel_command(raw_action, previous_accel_cmd, command_dt)
            safety = None
            if self.safety_layer is not None:
                safety = self._apply_safety_layer(time_s, s_ego, ey, speed, previous_accel_cmd, action, command_dt,
                                                  members=safety_members)
                action = safety.accel
            df_des = self._lane_keeping_steer(ey, epsi)
            v_des = np.clip(speed + action * self.DT,
                            self.controller_config.v_min,
                            self.controller_config.v_max)
            u0 = np.array([action, df_des])
            is_feasible = bool(solution.feasible)
            min_gap = self._min_gap(solution, prediction_bundle["prediction"])
            actual_accel = self._actual_longitudinal_accel()
            actual_jerk = self._actual_longitudinal_jerk(time_s, actual_accel)
            command_jerk = self._command_jerk(action, command_dt)
            self.policy_log.append({
                "time_s": time_s,
                "status": solution.status,
                "feasible": is_feasible,
                "solve_time": solve_time,
                "ego_s": float(s_ego),
                "ego_v": float(speed),
                "accel_cmd_raw": raw_action,
                "accel_cmd": action,
                "command_dt": command_dt,
                "command_jerk": command_jerk,
                "command_jerk_limit": self.controller_config.jerk_limit,
                # jinput: the issued jerk j_0 (the command is a_0 + j_0 * command_dt); None otherwise
                "jerk_cmd": None if solution.j_nominal is None else float(solution.j_nominal[0, 0]),
                "actual_accel": actual_accel,
                "actual_jerk": actual_jerk,
                "steer_cmd": float(df_des),
                "target_s": float(prediction_bundle["selected_target_state"][0]),
                "target_d": float(prediction_bundle["selected_target_state"][1]),
                "target_v": float(prediction_bundle["selected_target_state"][2]),
                "mode_names": prediction_bundle["prediction"].mode_names,
                "mode_probabilities": prediction_bundle["prediction"].probabilities.tolist(),
                "prediction_num_modes": int(prediction_bundle["prediction"].num_modes),
                "controller_config_num_modes": int(self.controller_config.num_modes),
                "safety_constraint_mode": solution.safety_constraint_mode,
                "brake_distance_bound_mode": self.controller_config.brake_distance_bound_mode,
                "brake_distance_hard_band_width": (
                    self.controller_config.brake_distance_hard_band_width
                ),
                "risk_allocation_mode": solution.risk_allocation_mode,
                "optimize_k": bool(self.controller_config.optimize_k),
                "solver_message": solution.solver_message,
                "solve_path": solution.solve_path,
                "solve_timing": dict(solution.timing_breakdown or {}),
                "num_decision_variables": int(solution.num_decision_variables),
                "baseline_decision_variables": int(solution.baseline_decision_variables),
                "policy_group_count": int(solution.policy_group_count),
                "policy_split_steps": list(solution.policy_split_steps),
                "first_policy_split_step": (
                    -1 if solution.first_policy_split_step is None
                    else int(solution.first_policy_split_step)
                ),
                "chance_margin_min": float(solution.chance_margin_min),
                "slack_max": float(solution.slack_max),
                # floorrisk<eps>: the cut-in chance cells whose floor the rule removed / lowered / raised (0
                # without it), and the plan's smallest margin to the old floor L + d0 + q(beta_j) sigma
                # (NaN without such a cell)
                "n_floor_free": int(solution.n_floor_free),
                "n_floor_lowered": int(solution.n_floor_lowered),
                "n_floor_raised": int(solution.n_floor_raised),
                "floor_old_margin_min": float(solution.floor_old_margin_min),
                "tightening_max": float(solution.tightening_max),
                "clearance_scale_now": self._clearance_scale_now(prediction_bundle["prediction"]),
                "predictor_time": prediction_bundle.get("debug", {}).get("predictor_time"),
                "predictor_type": self.predictor_type,
                "ego_route_debug": self._ego_route_debug,
                "stdan_debug": prediction_bundle.get("debug", {}),
                "min_gap": min_gap,
                "prediction_covariance": self._prediction_covariance_debug(
                    prediction_bundle["prediction"]
                ),
                "acc_reference": self._acc_reference_debug(solution),
            })
            if solution.planned_standoff is not None:
                # dopt<w>: the standoff the solve allowed per (mode, step), None off the dopt cells
                self.policy_log[-1]["planned_standoff"] = [
                    [None if np.isnan(value) else value for value in row]
                    for row in np.asarray(solution.planned_standoff, dtype=float).round(4).tolist()
                ]
            if safety is not None:
                # safelayer: the decision; accel_cmd above is the issued command, nominal_accel the MPC's limited one
                self.policy_log[-1]["safety_layer"] = {
                    "path": safety.path,
                    "vid": safety.vid,
                    "margin": safety.margin,
                    "nominal_accel": self._safety_nominal_accel,
                    "checked_vids": safety.checked_vids,
                    "cutin_vids": safety.cutin_vids,
                    # entrant<b>: the close entrants' margin for the nominal command, the entrants the fail-safe
                    # cannot avoid, and the vehicles that became entrants this tick (those unavoidable at once)
                    "entrant_margin": safety.entrant_margin,
                    "unavoidable_vids": safety.unavoidable_vids,
                    "new_entrant_vids": safety.new_entrant_vids,
                    "new_unavoidable_vids": safety.new_unavoidable_vids,
                }
            if self.pass_miqp_margin is not None:
                # passmiqp: the candidates and their decisions, the MIQP's solver time and iteration count
                self.policy_log[-1]["pass_miqp"] = {
                    "decisions": list(solution.pass_decisions or []),
                    "n_pass": int(solution.pass_n_pass),
                    "accel_max": float(solution.pass_accel_max),
                    "solve_s": float(solution.pass_solve_s),
                    "iter_count": int(solution.pass_iter_count),
                }
            if self.controller_config.failsafe_branch:
                # safebranch: the fail-safe branch of this solve (NairACCSolution.failsafe_*)
                self.policy_log[-1]["failsafe_branch"] = {
                    "present": bool(solution.failsafe_present),
                    "members": int(solution.failsafe_members),
                    "margin_min": float(solution.failsafe_margin_min),
                    "terminal_margin": float(solution.failsafe_terminal_margin),
                    "slack_max": float(solution.failsafe_slack_max),
                    "entrants": int(solution.failsafe_entrants),
                    "entrant_margin": float(solution.failsafe_entrant_margin),
                }
            self._log_wandb(self.policy_log[-1], solution, prediction_bundle)

        self._remember_accel_cmd(time_s, action)
        control = self._low_level_control.update(speed, action, v_des, df_des)
        if self.ideal_actuator:
            # Steering only: with throttle on a pinned vehicle the speed keeps
            # a two-tick +/-0.06 m/s cycle at 7 m/s once excited (it appeared
            # behind the 7 m/s second lead, 2026-09-07); with the engine idle
            # the pin converges to the integrated command exactly.
            control.throttle = 0.0
            control.brake = 0.0
            self._ideal_actuator.update(action)
        return control, z0, u0, is_feasible, solve_time

    def get_cut_in_log(self):
        return {
            "controller": "acc_nair_smpc",
            "mode_names": ["lane_keeping", "cutin"],
            "mode_probabilities": list(self.mode_probabilities),
            "cutin_chance_ref": float(self.cutin_chance_ref),
            "intention_lpf_s": float(self.intention_lpf_s),
            "track_smooth_samples": int(self.track_smooth_samples),
            "cutin_probability_threshold": float(self.cutin_probability_threshold),
            "steps": self.policy_log,
        }

    def _prediction_bundle(self, s_ego, speed):
        if self.predictor_type == "constant_velocity_lead":
            return self._constant_velocity_lead_prediction_bundle(s_ego, speed)
        if self.predictor_type in ("stdan_3int", "stdan_vel", "lstm", "lstm_vel", "iaimm_kf"):
            bundle = self._stdan_prediction_bundle(s_ego, speed)
            if bundle is not None:
                return self._maybe_collapse_bundle_to_best_mode(bundle)
        target_state = self._selected_target_state(s_ego)
        synthetic = self.synthetic_predictor.predict(
            ego_state=np.array([s_ego, speed]),
            target_state=target_state,
            probabilities=self.mode_probabilities,
        )
        return self._maybe_collapse_bundle_to_best_mode({
            "prediction": synthetic.prediction,
            "selected_target_state": target_state,
            "debug": {"fallback": self.predictor_type != "synthetic"},
        })

    def _constant_velocity_lead_prediction_bundle(self, s_ego, speed):
        lead_state, lead_debug = self._selected_ego_lane_lead_state(s_ego, speed)
        horizon = int(self.controller_config.horizon)
        if lead_debug["selected_vehicle_id"] is None:
            means = np.zeros((1, horizon + 1, 2), dtype=float)
            means[:, :, 0] = float(s_ego)
            means[:, :, 1] = float(self.nominal_speed)
            prediction = MultimodalLeadPrediction(
                means=means,
                probabilities=np.array([1.0]),
                covariances=np.zeros((1, horizon + 1, 2, 2), dtype=float),
                mode_names=["default_no_ego_lane_lead"],
                active_mask=np.zeros((1, horizon + 1), dtype=bool),
            )
        else:
            prediction = make_constant_accel_lead_prediction(
                s0=float(lead_state[0]),
                v0=float(lead_state[2]),
                accelerations=np.array([0.0]),
                probabilities=np.array([1.0]),
                horizon=horizon,
                dt=float(self.controller_config.dt),
                covariance=np.diag([0.25, 0.09]),
                mode_names=["constant_velocity_lead"],
            )
        return {
            "prediction": prediction,
            "selected_target_state": lead_state,
            "debug": {
                "fallback": False,
                "predictor_type": self.predictor_type,
                "predictor_time": 0.0,
                "prediction_num_modes": int(prediction.num_modes),
                "controller_config_num_modes": int(self.controller_config.num_modes),
                "constant_velocity_lead": lead_debug,
            },
        }

    def _maybe_collapse_bundle_to_best_mode(self, bundle):
        if not self.best_mode_only:
            return bundle
        prediction = bundle["prediction"]
        if prediction.num_modes <= 1:
            bundle.setdefault("debug", {})["best_mode_only"] = True
            return bundle
        best_idx = int(np.argmax(prediction.probabilities))
        mode_names = prediction.mode_names or [f"mode_{idx}" for idx in range(prediction.num_modes)]
        policy_tree = None
        if prediction.policy_tree is not None:
            policy_tree = PolicySharingTree(
                policy_groups_by_step=prediction.policy_tree.policy_groups_by_step[
                    best_idx: best_idx + 1
                ].copy(),
                policy_split_steps=(),
                mode_names=[mode_names[best_idx]],
            )
        collapsed = MultimodalLeadPrediction(
            means=prediction.means[best_idx: best_idx + 1].copy(),
            probabilities=np.array([1.0]),
            covariances=prediction.covariances[best_idx: best_idx + 1].copy(),
            T=None if prediction.T is None else prediction.T[best_idx: best_idx + 1].copy(),
            c=None if prediction.c is None else prediction.c[best_idx: best_idx + 1].copy(),
            mode_names=[mode_names[best_idx]],
            policy_tree=policy_tree,
            active_mask=prediction.active_mask[best_idx: best_idx + 1].copy(),
            k_group_map=(
                None if prediction.k_group_map is None
                else prediction.k_group_map[best_idx: best_idx + 1].copy()
            ),
            k_group_names=prediction.k_group_names,
        )
        collapsed_bundle = dict(bundle)
        collapsed_bundle["prediction"] = collapsed
        debug = dict(collapsed_bundle.get("debug", {}))
        debug.update({
            "best_mode_only": True,
            "best_mode_source_num_modes": int(prediction.num_modes),
            "best_mode_index": best_idx,
            "best_mode_probability": float(prediction.probabilities[best_idx]),
            "best_mode_name": mode_names[best_idx],
            "prediction_num_modes": int(collapsed.num_modes),
            "controller_config_num_modes": int(self.controller_config.num_modes),
        })
        collapsed_bundle["debug"] = debug
        return collapsed_bundle

    def _stdan_prediction_bundle(self, s_ego, speed):
        self._update_agent_history()
        trackings = self._stdan_trackings()
        target_states = {}
        target_relations = {}
        target_current_rhs_xy = {}
        # Every other vehicle's (s, d, v), for the decision-point branch rule:
        # a target's own lead is a traffic actor, not a prediction target.
        other_states = {}
        # Adjacent targets' measured lateral offset toward the ego lane, for
        # the context trust mixing (``ctx<lambda>``); empty when the token is off.
        target_lateral = {}
        # latgate: their lateral speed toward the ego lane (None = too little
        # history), from samples recorded for every target whatever its relation.
        target_lateral_speed = {}
        now = self._elapsed_seconds() if self.lateral_gate else None
        nearest_target = None
        carla_map = self.world.get_map()
        ego_wp = self.world.get_map().get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            role = actor.attributes.get("role_name", "")
            x, y, psi, actor_speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv < s_ego - 5.0:
                continue
            other_states[actor.id] = np.array([s_tv, d_tv, actor_speed], dtype=float)
            if "target" not in role:
                continue
            target_states[actor.id] = np.array([s_tv, d_tv, actor_speed], dtype=float)
            target_current_rhs_xy[actor.id] = np.array([x, y], dtype=float)
            target_wp = carla_map.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            # Only the same-lane test is taken from the waypoints: it compares
            # road_id/lane_id directly and is exact.  Which *side* an adjacent lane
            # sits on comes from the Frenet offset instead, because CARLA's
            # get_left_lane()/get_right_lane() are posed in the simulator's
            # left-handed frame and disagreed with the sign of d in 8 of 10 runs of
            # the 2026-09-02 sweep -- and disagreed with each other across runs of
            # identical geometry.  Everything downstream (lane_occupancy_from_d, the
            # LLC/RLC to cut-in mapping, the controller's own Frenet state) reads d
            # with +d to the left, so the relation has to be posed the same way.
            relation = relation_from_waypoints(ego_wp, target_wp)
            if relation != REL_EGO_LANE:
                relation = self._relation_from_frenet_offset(d_tv)
            target_relations[actor.id] = relation
            if now is not None and target_wp is not None:
                self._record_lateral_speed_sample(
                    actor.id, now, self._lateral_speed_left(actor.get_velocity(), target_wp))
            if (self.context_trust is not None and target_wp is not None
                    and relation in (REL_LEFT_ADJACENT, REL_RIGHT_ADJACENT)):
                target_lateral[actor.id] = self._lateral_offset_toward_ego_lane(
                    actor.get_location(), target_wp, relation)
                if now is not None:
                    target_lateral_speed[actor.id] = self._lateral_speed_toward_ego_lane(
                        self._ctx_lateral_speed_samples.get(actor.id, []), now, relation)
            if nearest_target is None or s_tv - s_ego < nearest_target[0]:
                nearest_target = (s_tv - s_ego, target_states[actor.id])
        if now is not None:
            # A target that left the loop (behind the ego, gone) starts over.
            for vehicle_id in list(self._ctx_lateral_speed_samples):
                if vehicle_id not in target_states:
                    del self._ctx_lateral_speed_samples[vehicle_id]

        if not target_states:
            return None

        def lane_membership_fn(target_id, sampled_raw_traj_xy):
            return self._stdan_ego_lane_memberships_from_carla_waypoints(
                ego_wp,
                target_current_rhs_xy.get(target_id),
                sampled_raw_traj_xy,
            )

        gap_recovery_elapsed, gap_recovery_start_scale = self._gap_recovery_elapsed(
            target_relations, target_states, np.array([s_ego, speed]))
        branch_floor_by_vehicle, branch_decision_debug = self._decision_point_branch_floors(
            target_states, target_relations, other_states)
        entry_plan = None
        entry_margin = self.entry_pass_margin
        pass_kw = {}
        if self.pass_miqp_margin is not None:
            # passmiqp: the entry test is a pre-filter on the no-acceleration displacement from the previous
            # command (the ideal actuator's acceleration); far-behind entries pass without a row
            displacement = pass_cap_displacement(self.N, self.DT, speed, self._prev_accel_cmd,
                                                 self.controller_config.v_max)
            entry_plan = {int(vid): displacement for vid in target_states.keys()}
            entry_margin = PASS_MIQP_FAR_BEHIND_M
            pass_kw = {"pass_margin": self.pass_miqp_margin, "pass_slots": PASS_MIQP_SLOTS}
            if self.pass_no_cap:
                # passnocap: the candidate test against the maximum-acceleration displacement
                cfg = self.controller_config
                reach = pass_maxacc_displacement(
                    self.N, self.DT, speed, self._prev_accel_cmd, cfg.v_max, cfg.a_max,
                    cfg.jerk_limit if cfg.jerk_limit_up is None else cfg.jerk_limit_up)   # the rise: jlimup<x>
                pass_kw["pass_plan"] = {int(vid): reach for vid in target_states.keys()}
        elif self.entry_pass:
            now_s = self._elapsed_seconds()
            previous = self._entry_previous_plan
            entry_plan = entry_ego_displacement(
                target_states.keys(), self.N, self.DT, speed,
                None if previous is None or now_s is None else previous[1:] + (now_s - previous[0],))
        predictor_start = time.time()
        try:
            result = self.stdan_predictor.predict_acc(
                ego_state=np.array([s_ego, speed]),
                target_states_frenet=target_states,
                target_relations=target_relations,
                trackings=trackings,
                frenet_handler=self._frenet_traj,
                horizon=self.N,
                desired_speed=self.nominal_speed,
                num_modes=self.controller_config.num_modes,
                controller_dt=self.DT,
                lane_membership_fn=lane_membership_fn,
                model_yaw=self._stdan_model_yaw,
                cutin_probability_threshold=self.cutin_probability_threshold,
                cutin_clearance_ramp_ref=self.cutin_clearance_ramp_ref,
                cutin_chance_ref=self.cutin_chance_ref,
                cutin_chance_sigma_cap=self.chance_sigma_cap,
                cutin_clearance_tlc_ref=self.cutin_clearance_tlc_ref,
                gap_recovery_elapsed=gap_recovery_elapsed,
                gap_recovery_s=self.gap_recovery_s,
                gap_recovery_start_scale=gap_recovery_start_scale,
                lk_entry_as_cutin=self.lk_entry_as_cutin,
                branch_floor_by_vehicle=branch_floor_by_vehicle,
                context_trust=self.context_trust,
                lateral_offsets=target_lateral,
                lateral_speeds=target_lateral_speed,
                lateral_latched=self._ctx_latched,
                lateral_gate=self.lateral_gate,
                no_cutout=self.no_cutout,
                lk_sigma=self.lk_sigma,
                full_footprint=self.full_footprint,
                lead_merge=self.lead_merge,
                merge_first=self.merge_first,
                lk_hold=self.lk_hold,
                no_lk_entry=self.no_lk_entry,
                **({"entry_plan": entry_plan, "entry_clearance": self.controller_config.vehicle_length,
                    "entry_margin": entry_margin} if entry_plan is not None else {}),
                **pass_kw,
            )
            predictor_time = time.time() - predictor_start
        except Exception as exc:
            return {
                "prediction": self.synthetic_predictor.predict(
                    np.array([s_ego, speed]), nearest_target[1], self.mode_probabilities
                ).prediction,
                "selected_target_state": nearest_target[1],
                "debug": {
                    "fallback": True,
                    "reason": repr(exc),
                    "predictor_time": time.time() - predictor_start,
                    "controller_config_num_modes": int(self.controller_config.num_modes),
                },
            }
        selected_target_state = nearest_target[1] if nearest_target is not None else np.array([s_ego, 0.0, speed])
        result["selected_target_state"] = selected_target_state
        self._cache_predicted_lane_entry(result.get("processed_targets", []))
        if self.lateral_gate:
            self._update_lateral_latch(result.get("processed_targets", []))
        result["debug"] = {
            "num_targets": len(target_states),
            "relations": target_relations,
            "stdan_model_yaw": self._stdan_model_yaw,
            # Which weights produced these predictions.  The checkpoint comes
            # from an environment variable, so runs made in different shells
            # can silently use different predictors; the log has to say which.
            "stdan_ckpt": str(getattr(self.stdan_predictor, "ckpt_path", "")),
            "gap_recovery_start_scale": dict(self._gap_recovery_start_scale),
            "predictor_time": predictor_time,
            "prediction_num_modes": int(result["prediction"].num_modes),
            "controller_config_num_modes": int(self.controller_config.num_modes),
            "scenario_metadata": result.get("scenario_metadata", []),
            "branch_decision": branch_decision_debug,
            "processed_targets": self._stdan_processed_target_debug(
                result.get("processed_targets", [])),
        }
        return result

    # Nair et al.'s traffic-light example decides where its TV must choose:
    # the point from which a 0.7 g brake just stops at the light.  Here a
    # target closing on its own lead must choose (change lane or brake) at the
    # gap from which that brake just settles at its minimum following gap.
    BRANCH_DECISION_MIN_GAP_M = 4.5 + 2.5   # SPEED_CAP_VEHICLE_LENGTH_M + SPEED_CAP_MIN_LONGITUDINAL_GAP_M

    def _decision_point_branch_floors(self, target_states, target_relations, other_states):
        """Per adjacent target: the step before which its cut-in and lane-keeping
        scenarios keep one policy, from its gap and closing speed to the vehicle
        ahead of it in its own lane (``decision_point_branch_step``).  Empty
        when the token is off or no target has a lead ahead."""
        floors = {}
        debug = {}
        if self.branch_decision_brake is None:
            return floors, debug
        for target_id, state in target_states.items():
            if target_relations.get(target_id) == REL_EGO_LANE:
                continue
            s_tv, d_tv, v_tv = float(state[0]), float(state[1]), float(state[2])
            lead = None
            for other_id, other in other_states.items():
                if other_id == target_id:
                    continue
                gap = float(other[0]) - s_tv
                if gap <= 0.0 or abs(float(other[1]) - d_tv) > self.EGO_LANE_HALF_WIDTH_M:
                    continue
                if lead is None or gap < lead[0]:
                    lead = (gap, float(other[2]), other_id)
            if lead is None:
                continue
            step = decision_point_branch_step(
                lead[0], v_tv - lead[1], self.DT, self.branch_decision_brake,
                self.BRANCH_DECISION_MIN_GAP_M)
            debug[int(target_id)] = {
                "lead_id": int(lead[2]), "gap": round(lead[0], 3),
                "closing_speed": round(v_tv - lead[1], 3),
                "step": None if step is None else int(step),
            }
            if step is not None:
                floors[int(target_id)] = int(step)
        return floors, debug

    def _acc_reference_debug(self, solution):
        reference = solution.reference
        return {
            "s_ref": np.asarray(reference.s_ref, dtype=float).round(4).tolist(),
            "v_ref": np.asarray(reference.v_ref, dtype=float).round(4).tolist(),
            "a_ref": np.asarray(reference.a_ref, dtype=float).round(4).tolist(),
            "prev_u": np.asarray(reference.prev_u, dtype=float).round(4).tolist(),
            "x_nominal": np.asarray(solution.x_nominal, dtype=float).round(4).tolist(),
            "u_nominal": np.asarray(solution.u_nominal, dtype=float).round(4).tolist(),
            # jinput: the planned jerk per (mode, step); None otherwise
            "j_nominal": (None if solution.j_nominal is None
                          else np.asarray(solution.j_nominal, dtype=float).round(4).tolist()),
        }

    @staticmethod
    def _clearance_scale_now(prediction):
        """Standoff scale applied to the lead the ego has right now, for plotting.

        Only modes active at step 0 count, so a low-probability cut-in
        hypothesis about a next-lane vehicle does not show up as a relaxation
        of the gap to the vehicle actually ahead.
        """
        active_now = np.asarray(prediction.active_mask, dtype=bool)[:, 0]
        if not active_now.any():
            return float("nan")
        return float(np.min(np.asarray(prediction.clearance_scale, dtype=float)[active_now, 0]))

    def _prediction_covariance_debug(self, prediction):
        covariances = np.asarray(prediction.covariances, dtype=float)
        if covariances.ndim != 4 or covariances.shape[-2:] != (2, 2):
            return {}
        diag = np.maximum(np.diagonal(covariances, axis1=2, axis2=3), 0.0)
        var_s = diag[:, :, 0]
        var_v = diag[:, :, 1]
        std_s = np.sqrt(var_s)
        std_v = np.sqrt(var_v)
        mode_names = prediction.mode_names or [
            f"mode_{idx}" for idx in range(prediction.num_modes)
        ]
        return {
            "mode_names": [str(name) for name in mode_names],
            "active_mask": np.asarray(prediction.active_mask, dtype=bool).tolist(),
            "var_s": np.asarray(var_s, dtype=float).round(6).tolist(),
            "var_v": np.asarray(var_v, dtype=float).round(6).tolist(),
            "std_s": np.asarray(std_s, dtype=float).round(6).tolist(),
            "std_v": np.asarray(std_v, dtype=float).round(6).tolist(),
        }

    def _stdan_processed_target_debug(self, processed_targets):
        summaries = []
        for target in processed_targets:
            raw = target.raw_prediction or {}
            traj_frenet = np.asarray(raw.get("pred_traj_frenet", []), dtype=float)
            full_traj_frenet = np.asarray(raw.get("full_pred_traj_frenet", []), dtype=float)
            sampled_xy = np.asarray(raw.get("sampled_raw_traj_xy", []), dtype=float)
            full_xy = np.asarray(raw.get("raw_traj_xy", []), dtype=float)
            lead_covariance = np.asarray(raw.get("pred_lead_covariance", []), dtype=float)
            mode_d = {}
            for mode_idx, mode_name in enumerate(INTENTION_NAMES):
                if traj_frenet.ndim < 3 or mode_idx >= traj_frenet.shape[0] or traj_frenet.shape[1] == 0:
                    continue
                d_values = traj_frenet[mode_idx, :, 1]
                mode_d[mode_name] = {
                    "start": float(d_values[0]),
                    "end": float(d_values[-1]),
                    "min": float(np.min(d_values)),
                    "max": float(np.max(d_values)),
                    "delta": float(d_values[-1] - d_values[0]),
                }
            occupancy = target.branch_info.get("ego_lane_occupancy_mask", [])
            mode_membership_arr = np.asarray(
                target.branch_info.get("mode_lane_membership", []),
                dtype=bool,
            )
            pred_traj_frenet = {}
            full_pred_traj_frenet = {}
            pred_traj_xy = {}
            full_pred_traj_xy = {}
            mode_lane_membership = {}
            predictor_covariance = {}
            for mode_idx, mode_name in enumerate(INTENTION_NAMES):
                if traj_frenet.ndim == 3 and mode_idx < traj_frenet.shape[0]:
                    pred_traj_frenet[mode_name] = np.asarray(
                        traj_frenet[mode_idx], dtype=float
                    ).round(4).tolist()
                if full_traj_frenet.ndim == 3 and mode_idx < full_traj_frenet.shape[0]:
                    full_pred_traj_frenet[mode_name] = np.asarray(
                        full_traj_frenet[mode_idx], dtype=float
                    ).round(4).tolist()
                if sampled_xy.ndim == 3 and mode_idx < sampled_xy.shape[0]:
                    pred_traj_xy[mode_name] = np.asarray(
                        sampled_xy[mode_idx], dtype=float
                    ).round(4).tolist()
                if full_xy.ndim == 3 and mode_idx < full_xy.shape[0]:
                    full_pred_traj_xy[mode_name] = np.asarray(
                        full_xy[mode_idx], dtype=float
                    ).round(4).tolist()
                if mode_membership_arr.ndim == 2 and mode_idx < mode_membership_arr.shape[0]:
                    mode_lane_membership[mode_name] = [
                        bool(value)
                        for value in mode_membership_arr[mode_idx].reshape(-1)
                    ]
                if (
                        lead_covariance.ndim == 4
                        and mode_idx < lead_covariance.shape[0]
                        and lead_covariance.shape[-2:] == (2, 2)):
                    diag = np.maximum(np.diagonal(
                        lead_covariance[mode_idx], axis1=1, axis2=2), 0.0)
                    predictor_covariance[mode_name] = {
                        "var_s": np.asarray(diag[:, 0], dtype=float).round(6).tolist(),
                        "var_v": np.asarray(diag[:, 1], dtype=float).round(6).tolist(),
                        "std_s": np.sqrt(diag[:, 0]).round(6).tolist(),
                        "std_v": np.sqrt(diag[:, 1]).round(6).tolist(),
                    }
            summaries.append({
                "vehicle_id": int(target.vehicle_id),
                "relation_to_ego_lane": target.relation_to_ego_lane,
                "acc_mode_raw_indices": {
                    mode.scenario_label: [int(idx) for idx in mode.raw_mode_indices]
                    for mode in target.mode_predictions
                },
                "raw_mode_prob": {
                    str(key): float(value)
                    for key, value in target.raw_mode_prob.items()
                },
                "acc_mode_prob": {
                    str(key): float(value)
                    for key, value in target.acc_mode_prob.items()
                },
                "signed_t_cross": float(raw.get("signed_t_cross", 0.0)),
                "raw_longitudinal_prob": {
                    str(idx): float(value)
                    for idx, value in enumerate(np.asarray(
                        raw.get("raw_longitudinal_prob", []), dtype=float).reshape(-1))
                },
                "selected_longitudinal_idx": raw.get("selected_longitudinal_idx"),
                "selected_longitudinal_mode": raw.get("selected_longitudinal_mode"),
                "ego_lane_membership_split_step": int(target.branch_info.get(
                    "ego_lane_membership_split_step",
                    target.branch_info.get("branch_step", -1),
                )),
                "branch_step": int(target.branch_info.get("branch_step", -1)),
                "ego_lane_start_idx": int(target.branch_info.get("ego_lane_start_idx", 999)),
                "ego_lane_end_idx": int(target.branch_info.get("ego_lane_end_idx", -1)),
                "lane_membership_source": str(target.branch_info.get(
                    "lane_membership_source", "")),
                "trajectory_suggested_cutin_raw_idx": target.branch_info.get(
                    "trajectory_suggested_cutin_raw_idx"),
                "used_trajectory_aware_cutin_mapping": bool(target.branch_info.get(
                    "used_trajectory_aware_cutin_mapping", False)),
                "cutin_chance_confidences": target.branch_info.get(
                    "cutin_chance_confidences", {}),
                "cutin_probability_threshold": float(target.branch_info.get(
                    "cutin_probability_threshold", 0.0)),
                "gated_cutin_modes": list(target.branch_info.get("gated_cutin_modes", [])),
                "ego_lane_occupancy_mask": [
                    bool(value) for value in np.asarray(occupancy, dtype=bool).reshape(-1)
                ],
                "mode_lane_membership": mode_lane_membership,
                "mode_d": mode_d,
                "pred_traj_frenet": pred_traj_frenet,
                "full_pred_traj_frenet": full_pred_traj_frenet,
                "pred_traj_xy": pred_traj_xy,
                "full_pred_traj_xy": full_pred_traj_xy,
                "predictor_covariance": predictor_covariance,
            })
            ctx = target.branch_info.get("context_trust")
            if ctx is not None:
                # ctx<lambda>: the measured lateral offset toward the ego lane,
                # its observation, and the cut-in intention before/after mixing.
                summaries[-1].update({
                    "ctx_lambda": ctx["lambda"],
                    "ctx_y_v": ctx["y_v"],
                    "ctx_o_v": ctx["o_v"],
                    "ctx_p_raw": ctx["p_raw"],
                    "ctx_p_mixed": ctx["p_mixed"],
                    # latgate: lateral speed toward the ego lane, the trigger
                    # and the latch (None without the token).
                    "ctx_v_y": ctx["v_y"],
                    "ctx_trigger": ctx["trigger"],
                    "ctx_latched": ctx["latched"],
                })
        return summaries

    def _stdan_ego_lane_memberships_from_carla_waypoints(
            self,
            ego_wp,
            current_rhs_xy,
            sampled_raw_traj_xy):
        sampled = np.asarray(sampled_raw_traj_xy, dtype=float)
        if sampled.ndim != 3:
            raise ValueError("sampled_raw_traj_xy must have shape (num_modes, horizon, 2)")
        memberships = np.zeros((sampled.shape[0], sampled.shape[1] + 1), dtype=bool)
        current_on_ego_lane = self._rhs_xy_on_ego_lane(ego_wp, current_rhs_xy)
        memberships[:, 0] = current_on_ego_lane
        for mode_idx in range(sampled.shape[0]):
            for step_idx, point in enumerate(sampled[mode_idx], start=1):
                memberships[mode_idx, step_idx] = self._rhs_xy_on_ego_lane(ego_wp, point)
        return memberships

    def _rhs_xy_on_ego_lane(self, ego_wp, rhs_xy):
        if ego_wp is None or rhs_xy is None:
            return False
        point = np.asarray(rhs_xy, dtype=float).reshape(-1)
        if point.size < 2:
            return False
        location = carla.Location(x=float(point[0]), y=-float(point[1]), z=ego_wp.transform.location.z)
        waypoint = self.world.get_map().get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return False
        return waypoint.lane_id == ego_wp.lane_id

    def _cache_predicted_lane_entry(self, processed_targets):
        """Remember how soon each target is predicted to reach the ego lane."""
        if self.gap_recovery_s <= 0.0:
            return
        entries = {}
        for target in processed_targets:
            start_idx = getattr(target, "ego_lane_start_idx", None)
            if start_idx is None:
                branch = getattr(target, "branch_info", None) or {}
                start_idx = branch.get("ego_lane_start_idx")
            if not isinstance(start_idx, (int, float)):
                continue
            entries[getattr(target, "vehicle_id", None)] = float(start_idx) * self.DT
        self._predicted_lane_entry_s = entries

    def _gap_recovery_elapsed(self, target_relations, target_states=None, ego_state=None):
        """Seconds each vehicle has been treated as a lead, for the gap re-establishment.

        Also returns, per vehicle, the gap it had as a fraction of the full
        standoff when its clock started -- the relaxation starts from there,
        so a vehicle arriving outside the standoff is not relaxed at all.

        The clock starts when the vehicle is *about* to be in the way, not when it
        finishes merging: by the time the lane assignment flips, the ego has
        already done its braking and relaxing the standoff then changes nothing.
        Production ACC switches target on recognition for the same reason.

        Entries are dropped once a vehicle is neither in lane nor imminent, so one
        that merges, leaves and merges again starts its recovery over.
        """
        if self.gap_recovery_s <= 0.0:
            return None, None
        now = self._elapsed_seconds()
        tracked = {
            vehicle_id for vehicle_id, relation in target_relations.items()
            if relation == REL_EGO_LANE
        }
        for vehicle_id, entry_s in self._predicted_lane_entry_s.items():
            if entry_s <= self.GAP_RECOVERY_ONSET_TLC_S:
                tracked.add(vehicle_id)
        for vehicle_id in list(self._ego_lane_entry_time):
            if vehicle_id not in tracked:
                del self._ego_lane_entry_time[vehicle_id]
                self._gap_recovery_start_scale.pop(vehicle_id, None)
        elapsed = {}
        for vehicle_id in tracked:
            if vehicle_id not in self._ego_lane_entry_time:
                self._ego_lane_entry_time[vehicle_id] = now
                if target_states is not None and ego_state is not None and vehicle_id in target_states:
                    gap = float(target_states[vehicle_id][0]) - float(ego_state[0])
                    full = required_standoff(ego_state, self.controller_config)
                    self._gap_recovery_start_scale[vehicle_id] = float(
                        np.clip(gap / max(full, 1.0e-6), 0.0, 1.0))
            elapsed[vehicle_id] = max(0.0, now - self._ego_lane_entry_time[vehicle_id])
        return elapsed, dict(self._gap_recovery_start_scale)

    def _record_lateral_speed_sample(self, vehicle_id, time_s, lateral_speed_left):
        """Append a target's left-normal lateral speed and drop samples older
        than the ``latgate`` window."""
        samples = self._ctx_lateral_speed_samples.setdefault(vehicle_id, [])
        samples[:] = [(t, v) for t, v in samples if t > float(time_s) - CTX_GATE_VY_WINDOW_S]
        samples.append((float(time_s), float(lateral_speed_left)))

    def _update_lateral_latch(self, processed_targets):
        """``latgate``: keep the latch of every target processed as adjacent this
        tick; any other vehicle starts unlatched when it becomes adjacent again."""
        latched = {}
        for target in processed_targets:
            ctx = target.branch_info.get("context_trust")
            if ctx is not None and ctx.get("latched") is not None:
                latched[int(target.vehicle_id)] = bool(ctx["latched"])
        self._ctx_latched = latched

    def _update_agent_history(self):
        if self.agent_history is None:
            return
        snapshot = self.world.get_snapshot()
        if snapshot is not None:
            self.agent_history.update(snapshot, self.world)

    def _stdan_trackings(self):
        if self.agent_history is None:
            return {}
        trackings = {}
        for actor in self.world.get_actors().filter("vehicle*"):
            actor_history = self.agent_history.vehicles.get(actor.id)
            if actor_history is None or not actor_history.time_history:
                continue
            times = np.asarray(actor_history.time_history, dtype=float)
            poses = np.asarray(actor_history.pose_history, dtype=np.float64)
            current_time = float(times[-1])
            history = []
            for hsec in self.stdan_history_secs:
                query_time = current_time - float(hsec)
                if query_time < times[0] - self.stdan_history_closeness_eps:
                    continue
                idx = int(np.argmin(np.abs(times - query_time)))
                if abs(float(times[idx]) - query_time) <= self.stdan_history_closeness_eps:
                    history.append(poses[idx])
            if history:
                trackings[actor.id] = self._smooth_track(
                    np.asarray(history, dtype=np.float64))
        return trackings

    def _smooth_track(self, history):
        """국소 2차 평활(Savitzky-Golay 형)을 이력 좌표에 건다. 기본은 꺼짐.

        예측기가 보는 위치에는 CARLA 타이어 모델이 남기는 밀리미터급 잔떨림이
        있고, 오프라인 절제(2026-09-10)에서 **완전히 매끈한 합성 이력을 넣으면
        의도확률의 틱간 변화가 정확히 0** 이 나왔다 -- 진동은 전부 이 잔떨림이
        모델을 통과해 증폭된 것이다. 배포 스택이라면 원시 단일 프레임 자세가
        아니라 추적 필터의 추정치를 예측기에 주므로, 이 평활은 시뮬 아티팩트를
        가리는 것이 아니라 빠져 있던 추적 단계를 세우는 쪽이다.

        차수 2 라 등가속 운동을 정확히 재현한다 -- 우리 시나리오의 차선변경도
        0.5 s 창 안에서는 등가속에 가까우므로 지연이 생기지 않는다. 창이 이력
        안에서만 앞뒤를 보므로 미래 정보를 쓰지 않는다.
        """
        w = int(self.track_smooth_samples)
        if w < 2 or history.shape[0] < 2 * w + 1:
            return history
        out = history.copy()
        n = history.shape[0]
        idx = np.arange(n)
        for col in range(min(2, history.shape[1])):      # x, y 만 (yaw 는 그대로)
            y = history[:, col]
            sm = np.empty(n)
            for i in range(n):
                lo, hi = max(0, i - w), min(n, i + w + 1)
                k = idx[lo:hi] - i
                sm[i] = np.polyval(np.polyfit(k, y[lo:hi], 2), 0.0)
            out[:, col] = sm
        return out

    def _ego_state_rhs(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        x = transform.location.x
        y = -transform.location.y
        psi = -fth.fix_angle(np.radians(transform.rotation.yaw))
        speed = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2))
        return x, y, psi, speed

    def _selected_target_state(self, s_ego):
        candidates = []
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            role = actor.attributes.get("role_name", "")
            if "target" not in role:
                continue
            x, y, psi, speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv >= s_ego - 5.0:
                candidates.append((s_tv - s_ego, s_tv, d_tv, speed))

        if not candidates:
            return np.array([s_ego + 1000.0, 3.5, self.nominal_speed])

        _, s_tv, d_tv, speed = sorted(candidates, key=lambda item: item[0])[0]
        return np.array([s_tv, d_tv, speed], dtype=float)

    def _selected_ego_lane_lead_state(self, s_ego, speed):
        carla_map = self.world.get_map()
        ego_wp = carla_map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        candidates = []
        for actor in self.world.get_actors().filter("vehicle*"):
            if actor.id == self.vehicle.id:
                continue
            x, y, psi, actor_speed = self._actor_state_rhs(actor)
            s_tv, d_tv, _ = self._frenet_traj.convert_global_to_frenet_frame(x, y, psi)
            if s_tv < s_ego - 1.0:
                continue
            target_wp = carla_map.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            relation = relation_from_waypoints(ego_wp, target_wp)
            if relation != REL_EGO_LANE and abs(d_tv) > self.EGO_LANE_HALF_WIDTH_M:
                continue
            role = actor.attributes.get("role_name", "")
            candidates.append((s_tv - s_ego, actor.id, role, s_tv, d_tv, actor_speed))

        if not candidates:
            return (
                np.array([s_ego + 1000.0, 0.0, speed], dtype=float),
                {
                    "selected_vehicle_id": None,
                    "selected_role": None,
                    "candidate_count": 0,
                    "source": "carla_waypoint_ego_lane",
                },
            )

        gap, actor_id, role, s_tv, d_tv, actor_speed = sorted(candidates, key=lambda item: item[0])[0]
        return (
            np.array([s_tv, d_tv, actor_speed], dtype=float),
            {
                "selected_vehicle_id": int(actor_id),
                "selected_role": role,
                "candidate_count": len(candidates),
                "gap": float(gap),
                "source": "carla_waypoint_ego_lane",
            },
        )

    @staticmethod
    def _parse_predictor_type(smpc_config):
        config = str(smpc_config)
        if "const_lead" in config or "constant_lead" in config:
            return "constant_velocity_lead"
        if "iaimm" in config:
            return "iaimm_kf"
        if "lstm_vel" in config:
            return "lstm_vel"
        if "lstm" in config:
            return "lstm"
        if "stdan_vel" in config:
            return "stdan_vel"
        if "stdan_3int" in config:
            return "stdan_3int"
        return "synthetic"

    @staticmethod
    def _parse_best_mode_only(smpc_config):
        return "best_mode" in str(smpc_config)

    @staticmethod
    def _parse_controller_num_modes(smpc_config, requested_num_modes):
        config = str(smpc_config)
        if "best_mode" in config or "const_lead" in config or "constant_lead" in config or "lstm" in config:
            return 1
        match = re.search(r"(?:num_modes|modes|mode)[_-]?(\d+)", config)
        if match:
            return max(1, int(match.group(1)))
        return max(1, int(requested_num_modes))

    @staticmethod
    def _parse_variant(smpc_config):
        config = str(smpc_config).lower()
        if config in ("open_loop", "multimodal_ol", "acc_nair_smpc_ol") or "open_loop" in config or "multimodal_ol" in config or "_ol" in config:
            return VARIANT_MULTIMODAL_OL
        if "fixed_risk" in config:
            return VARIANT_FIXED_RISK
        return VARIANT_PROPOSED

    @staticmethod
    def _parse_safety_constraint_mode(smpc_config):
        config = str(smpc_config).lower()
        if re.search(r"cutin_chance[0-9]", config) or re.search(r"(?:^|_)chancecells(?=_|$)", config):
            return SAFETY_CONFIDENCE_CHANCE
        if (
                "brake_distance" in config
                or "braking_distance" in config
                or "d_brake" in config
                or "dbrake" in config
                or "collision_distance" in config):
            return SAFETY_BRAKE_DISTANCE
        if "scalar_chance" in config or "scalar_probability" in config or "optimized_eta" in config:
            return SAFETY_SCALAR_CHANCE
        return SAFETY_NOMINAL_SAFE_DISTANCE

    @staticmethod
    def _parse_brake_distance_bound_mode(smpc_config):
        config = str(smpc_config).lower()
        if (
                "brake_hard_band" in config
                or "brake_distance_hard_band" in config
                or "dbrake_hard_band" in config):
            return BRAKE_DISTANCE_BOUND_HARD_BAND
        return BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL

    @staticmethod
    def _parse_risk_allocation_mode(smpc_config):
        config = str(smpc_config)
        if "optimized_eta" in config:
            return RISK_OPTIMIZED_ETA
        if "probability_weighted" in config:
            return RISK_PROBABILITY_WEIGHTED
        return RISK_FIXED

    @staticmethod
    def _parse_optimize_k(smpc_config):
        config = str(smpc_config).lower()
        if (
                "no_k" in config
                or "zero_k" in config
                or "fixed_k" in config
                or "disable_k" in config):
            return False
        return (
            "with_k" in config
            or "optimize_k" in config
            or "enable_k" in config
            or "k_opt" in config
            or "opt_k" in config
            or "kopt" in config
        )

    @staticmethod
    def _parse_cutin_probability_threshold(smpc_config):
        """Read ``cutin_gate<value>`` from the config string (None when absent).

        Modes below this probability generate no collision-avoidance
        constraint, so the ego keeps speed instead of yielding to an unlikely
        merge.  Absent, STDAN policies use ``CUTIN_CHANCE_VANISH_BELOW``.  See
        ``gate_unlikely_cutin_modes`` and ``chance_cutout_clearance``.
        """
        match = re.search(r"cutin_gate([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_cutin_clearance_ramp_ref(smpc_config):
        """Read ``cutin_ramp<value>`` from the config string (default: disabled).

        The value is the reference confidence at which a cut-in mode keeps its
        present standoff; less likely modes shrink theirs continuously.  See
        ``cutin_clearance_scale``.
        """
        match = re.search(r"cutin_ramp([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_track_smooth_samples(smpc_config):
        """Read ``track_smooth<half-window in samples>`` (default: off).

        Half-width of the local quadratic fit applied to the tracked history
        before it reaches the predictor.  ``track_smooth2`` is +/-2 samples,
        a 0.5 s window at the predictor's 0.1 s spacing.  See ``_smooth_track``.
        """
        match = re.search(r"track_smooth([0-9]+)", str(smpc_config))
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parse_r_move(smpc_config):
        """Read ``rmove<value>`` from the config string (None = r_jerk / dt²).

        Weight on ``(u_0 - u_prev)^2``, the only transition the vehicle
        actually executes.  Separate from ``r_jerk`` so the ratio between
        "do not jump from the command that is going out" and "keep the planned
        profile smooth" is a chosen number rather than ``(dt / command_dt)^2``.
        """
        match = re.search(r"rmove([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_r_jerk(smpc_config):
        """Read ``rjerk<value>`` from the config string (None = keep the default).

        Sweeping the jerk weight needs a knob because it is shared by all four
        policies.  It became worth sweeping when 8ff4cf8 put step 0's jerk on
        the command period: the same r_jerk now buys a 16x heavier penalty on
        the tick-to-tick command jump than it did, which helps the cut-in cells
        and costs the cut-out cells their anticipation.
        """
        match = re.search(r"rjerk([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_jerk_limit(smpc_config):
        """Read ``jlim<value>`` from the config string (None = keep the default 5 m/s³).

        The hard jerk bound [m/s³] on the first input (over the command
        period), on every planned input change and on the issued command.
        ``rmove0_jlim1.5`` replaces the step-0 move cost with upstream Nair's
        ±1.5 m/s³ rate bound (hard here, soft upstream), 2026-10-22.
        """
        match = re.search(r"jlim([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    #: ``_parse_limit_tokens``: token -> (NairACCConfig field, sign of the value); the number is a magnitude
    LIMIT_TOKENS = (
        ("amin", "a_min", -1.0),
        ("amax", "a_max", 1.0),
        ("jlimup", "jerk_limit_up", 1.0),
        ("fsdecel", "failsafe_decel", -1.0),
        ("fsjerk", "failsafe_jerk", 1.0),
    )

    @classmethod
    def _parse_limit_tokens(cls, smpc_config):
        """Read the limit tokens of 2026-11-11 (unified limits) from the config string: a dict of the
        ``NairACCConfig`` fields to set, empty without any (every string before them).

        * ``amin<x>`` / ``amax<x>``: the comfort acceleration bounds ``a_min = -x`` / ``a_max = x`` [m/s^2]
          (default -3 / 2, ``original_nair_acc_config``), PC-ACC's token names.
        * ``jlimup<x>``: ``jerk_limit_up = x`` [m/s^3], the bound on the rising side of the acceleration;
          ``jlim<x>`` (``_parse_jerk_limit``) then bounds the falling side only.
        * ``fsdecel<x>`` / ``fsjerk<x>``: the safety layer's fail-safe, ``failsafe_decel = -x`` [m/s^2] and
          ``failsafe_jerk = x`` [m/s^3] (default -8 / 20, ``utils/safety_layer.py``); need ``safelayer``.

        Whole-token match; the numbers are magnitudes."""
        out = {}
        for token, name, sign in cls.LIMIT_TOKENS:
            match = re.search(r"(?:^|_)" + token + r"([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
            if match:
                out[name] = sign * float(match.group(1))
        if ("failsafe_decel" in out or "failsafe_jerk" in out) and not cls._parse_safety_layer(smpc_config):
            raise ValueError("fsdecel<x> / fsjerk<x> set the safety layer's fail-safe; they need safelayer")
        return out

    @staticmethod
    def _parse_r_a(smpc_config):
        """Read ``ra<value>`` from the config string (None = keep the default 1).

        The weight of ``r_a (u - a_ref)^2``, with ``a_ref = 0`` the only cost on
        the command's size.  ``ra0`` drops it (2026-10-29).  Matched as a whole
        token, so ``trajcutin`` and ``cutin_ramp`` never read as one.
        """
        match = re.search(r"(?:^|_)ra([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_accel_lag_tau(smpc_config):
        """Read ``alag<tau>`` from the config string (None = the double integrator).

        The time constant [s] of the first-order lag from the commanded to the
        actual acceleration in the controller's prediction model
        (``NairACCConfig.accel_lag_tau``), 2026-10-29.  CARLA's ideal actuator
        has no lag; the agent passes no measured acceleration, so the model
        starts from the previous command.
        """
        match = re.search(r"alag([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_jerk_input(smpc_config):
        """Read ``jinput`` from the config string (default: off), as a whole token.

        The jerk as the MPC input (``NairACCConfig.jerk_input``), 2026-10-29: the
        state is [s, v, a], ``|j_k| <= jerk_limit``, ``r_jerk j_k^2`` on every
        step, and the command is ``a_0 + j_0 * command_dt``.  ``rmove<w>`` has no
        role with it.  Not with ``alag<tau>``.
        """
        return re.search(r"(?:^|_)jinput(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_relative_brake_decel(smpc_config):
        """Read ``relbrake<b>`` from the config string (None = the plain ``tau * v``).

        The deceleration ``b`` [m/s^2] of the relative-speed safe distance
        ``L + d0 + max(0, tau v + (v^2 - v_l^2) / (2 b))`` on every lead cell
        (``NairACCConfig.relative_brake_decel``), 2026-10-30.  Whole-token match.
        """
        match = re.search(r"(?:^|_)relbrake([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_slack_l1_weight(smpc_config):
        """Read ``l1slack<w>`` from the config string (None = the quadratic slack cost).

        The weight ``w`` of the exact emergency-slack penalty ``w * sum(slack)``
        (``NairACCConfig.slack_l1_weight``), 2026-10-31.  Whole-token match;
        ``l1slack1e6`` reads as 1e6.
        """
        match = re.search(r"(?:^|_)l1slack([0-9]*\.?[0-9]+(?:e[0-9]+)?)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_terminal_cost_scale(smpc_config):
        """Read ``termcost<c>`` from the config string (None = no terminal cost).

        The scale ``c`` of the terminal cost-to-go
        ``c * 3 * P_j * [e_N, a_N] (P - Q) [e_N, a_N]^T``
        (``NairACCConfig.terminal_cost_scale``), 2026-11-02.  Whole-token match;
        ``termcost0.5`` reads as 0.5.  ``jinput`` only.
        """
        match = re.search(r"(?:^|_)termcost([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_floor_risk(smpc_config):
        """Read ``floorrisk<eps>`` from the config string (None = the hard floor of the cut-in chance cells).

        The risk ``eps`` of the chance constraint ``P_j P(gap_jk < L + d0) <= eps``
        that sets a cut-in chance cell's floor, ``L + d0 + Phi^-1(1 - eps / P_j) sigma``
        (``NairACCConfig.floor_risk``), 2026-11-02.  Whole-token match;
        ``floorrisk0.2`` reads as 0.2.  Needs ``fullfoot``.
        """
        match = re.search(r"(?:^|_)floorrisk([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_floor_risk_hypothesis(smpc_config):
        """Read ``floorhyp`` from the config string (default: off).

        With ``floorrisk<eps>``: the floor's ``Phi^-1(1 - eps / p)`` takes ``p``, the probability of the
        cell's own hypothesis, in place of the scenario probability ``P_j``
        (``NairACCConfig.floor_risk_hypothesis``), 2026-11-09. Whole-token match.
        """
        return re.search(r"(?:^|_)floorhyp(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_floor_risk_top(smpc_config):
        """Read ``floortop`` from the config string (default: off).

        With ``floorhyp``: a cut-in chance cell's sigma term in its priced row and the reference is the floor's
        one-sided ``z sigma``, so D = the floor + tau v (``NairACCConfig.floor_risk_top``), 2026-11-10.
        Whole-token match.
        """
        return re.search(r"(?:^|_)floortop(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_lane_keep_risk(smpc_config):
        """Read ``lkonesided<eps>`` from the config string (None = off).

        With ``lksigma`` and ``nocutout``: a lane-keeping cell's sigma term is the one-sided ``z sigma``,
        ``z = Phi^-1(1 - eps)`` at its ``p = 1``, in place of ``q(min(p, beta_ref)) sigma``
        (``NairACCConfig.lane_keep_risk``), 2026-11-10. Whole-token match; ``lkonesided0.05`` reads as 0.05.
        """
        match = re.search(r"(?:^|_)lkonesided([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_price_row_no_sigma(smpc_config):
        """Read ``nopricesigma`` from the config string (default: off).

        With ``floorrisk<eps>``: a priced cell asks for its standoff on the mean gap, without a sigma term; its
        floor row keeps the sigma term (``NairACCConfig.price_row_no_sigma``), 2026-11-10. Whole-token match.
        """
        return re.search(r"(?:^|_)nopricesigma(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_one_row(smpc_config):
        """Read ``onerow`` from the config string (default: off).

        With plain ``dopt<w>``: one row on every active lead cell, ``gap >= L + d0 + tau v - short - xi`` with
        ``0 <= short <= tau v`` priced at ``w P_j short^2``, no sigma term, no floor row, no hard lane-keeping row,
        and the reference's plain ``d_safe`` (``NairACCConfig.one_row``), 2026-11-12. Whole-token match.
        """
        return re.search(r"(?:^|_)onerow(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_one_row_z_conf(smpc_config):
        """Read ``zconf<b>`` from the config string (None = off).

        With ``onerow``: each active cell's row gets back a sigma term ``z sigma``, ``z = Phi^-1(min(p, b))`` with ``p``
        the probability of the cell's own lead mode (``NairACCConfig.one_row_z_conf``), 2026-11-12. Whole-token match;
        ``zconf0.95`` reads as 0.95.
        """
        match = re.search(r"(?:^|_)zconf([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_one_row_no_cap(smpc_config):
        """Read ``nocap`` from the config string (default: off).

        With ``onerow``: no upper bound on an active cell's shortfall, so its row is a pure soft constraint priced
        ``w P_j short^2`` (``NairACCConfig.one_row_no_cap``), 2026-11-12. Whole-token match (``passnocap`` is not it).
        """
        return re.search(r"(?:^|_)nocap(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_safety_layer(smpc_config):
        """Read ``safelayer`` from the config string (default: off).

        PC-ACC's safety layer (``utils/safety_layer.py``) around the MPC's command, 2026-11-02. Whole-token match.
        """
        return re.search(r"(?:^|_)safelayer(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_failsafe_branch(smpc_config):
        """Read ``safebranch`` from the config string (default: off).

        The safety layer's fail-safe as a branch of the scenario tree (``NairACCConfig.failsafe_branch``),
        2026-11-03. Needs ``safelayer``. Whole-token match.
        """
        return re.search(r"(?:^|_)safebranch(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_entrant_decel(smpc_config):
        """Read ``entrant<b>`` from the config string (default: off, ``None``).

        The safety layer covers close entrants (vehicles that joined the ego's path inside its safe distance,
        the code's cut-in set) under the braking bound a_c = -b m/s^2 (``utils/safety_layer.py``,
        ``SafetyLayer(entrant_decel=)``); with ``safebranch`` they lead the fail-safe branch too
        (``NairACCConfig.failsafe_entrant_decel``). ``entrant2`` reads as -2.0. Needs ``safelayer``. Whole-token
        match, 2026-11-04.
        """
        match = re.search(r"(?:^|_)entrant([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return -float(match.group(1)) if match else None

    @staticmethod
    def _parse_lead_merge(smpc_config):
        """Read ``leadmerge`` from the config string (default: off).

        One scenario per class of joint modes with the same effective lead at every step, its probability
        summed (``acc_postprocess._merge_duplicate_scenarios``), 2026-11-03. Whole-token match.
        """
        return re.search(r"(?:^|_)leadmerge(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_merge_first(smpc_config):
        """Read ``mergefirst`` from the config string (default: off).

        With ``leadmerge``: every joint mode is merged before the ``modes<N>`` cap, the classes are ordered by their
        summed probability and the N most likely kept (``acc_postprocess.build_multitarget_lead_prediction``),
        2026-11-03. Whole-token match.
        """
        return re.search(r"(?:^|_)mergefirst(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_lk_hold(smpc_config):
        """Read ``lkhold`` from the config string (default: off).

        The ego-lane lead's lane-keeping mode stays active at every step when its trajectory is no departure by
        the 3-step rule (``acc_postprocess.ends_outside_ego_lane``): a wobble out of the lane at a middle step or at
        the last one or two no longer switches the lead's cells off (final-check item 4, 2026-11-03). Whole-token
        match.
        """
        return re.search(r"(?:^|_)lkhold(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_no_lk_entry(smpc_config):
        """Read ``nolkentry`` from the config string (default: off).

        An adjacent vehicle's lane-keeping and away-direction trajectories never enter the ego's path at steps 1..
        (``acc_postprocess.process_vehicle_prediction(no_lk_entry=)``): only its lane change toward the ego lane can
        lead, so ``trajcutin`` finds no entering lane-keeping trajectory (2026-11-06). Whole-token match.
        """
        return re.search(r"(?:^|_)nolkentry(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_occupancy_risk(smpc_config):
        """Read ``occin<eps>`` from the config string (default: off, ``None``).

        A cut-in hypothesis's cell at step k >= 1 is active iff ``p_j * P_in,j(k) > eps``, ``P_in,j(k)`` the
        probability that the vehicle's body overlaps the ego's path laterally at step k under trajectory j
        (``acc_postprocess.occupancy_memberships``), in place of the mean-trajectory lane test and the 0.1
        gate (2026-11-05). ``occin0.1`` reads as 0.1. Needs ``stdan_3int``, ``trajcutin`` and a probabilistic
        lane test, which only the NGSIM replay has. Whole-token match.
        """
        match = re.search(r"(?:^|_)occin([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_occupancy_pin_gate(smpc_config):
        """Read ``pingate`` from the config string (default: off).

        With ``occin<eps>``: a trajectory with ``p_j > eps`` is in the ego's path at step k >= 1 iff
        ``P_in,j(k) > eps``, in place of ``p_j * P_in,j(k) > eps``, so the step does not depend on the
        probability (``acc_postprocess.occupancy_memberships(pin_gate=)``), 2026-11-09. NGSIM replay only, as
        ``occin``. Whole-token match.
        """
        return re.search(r"(?:^|_)pingate(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_probability_calibration(smpc_config):
        """Read ``pcal`` from the config string (default: off).

        Every target's intention probabilities are replaced, before the mode mapping, by calibrated probabilities of
        each lane change within 3 s from the predictor's probability and the vehicle's measured lateral state
        (``acc_postprocess.calibrate_intention_probabilities``), with maps fitted on natural NGSIM traffic
        (2026-11-05). NGSIM replay only. Whole-token match.
        """
        return re.search(r"(?:^|_)pcal(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_bcts_calibration(smpc_config):
        """Read ``bcts`` from the config string (default: off).

        Every target's intention probabilities are replaced, before the mode mapping, by bias-corrected temperature
        scaling of the predictor's own (``acc_postprocess.bcts_intention_probabilities``), fitted to the predictor's
        intention label on natural NGSIM traffic (2026-11-06). NGSIM replay only; not with ``pcal``. Whole-token
        match.
        """
        return re.search(r"(?:^|_)bcts(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_cutin_cvar(smpc_config):
        """Read ``cvar<eps>_<c>`` from the config string (default: off, ``None``), as ``(eps, c)``.

        The dopt price on the cut-in chance cells' shortfall replaced by the CVaR bound
        ``CVaR_eps(short_k) <= c`` [m] per step over the scenarios (``NairACCConfig.cutin_cvar_risk`` /
        ``cutin_cvar_bound``, 2026-11-05). ``cvar0.5_2`` reads as (0.5, 2.0). Needs ``cutin_chance<beta_ref>``,
        ``dopt<w>`` and ``fullfoot``. Whole-token match (both parts).
        """
        match = re.search(r"(?:^|_)cvar([0-9]*\.?[0-9]+)_([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return (float(match.group(1)), float(match.group(2))) if match else None

    @staticmethod
    def _parse_intention_lpf_s(smpc_config):
        """Read ``intent_lpf<seconds>`` from the config string (default: off).

        Time constant of the first-order lag on the predictor's intention
        probabilities.  The probability weights each mode's reference-tracking
        cost as well as scaling its standoff, so filtering it is the one place
        that reaches both.  See ``STDAN3IntACCAdapter._filter_intention``.
        """
        match = re.search(r"intent_lpf([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_cutin_chance_ref(smpc_config):
        """Read ``cutin_chance<value>`` from the config string (default: disabled).

        The value is the reference confidence ``beta_ref`` of the Benciolini
        confidence chance constraint on cut-in modes; the token also switches
        the controller to ``confidence_chance`` safety constraints.  Modes below
        ``CUTIN_CHANCE_VANISH_BELOW`` vanish as for every STDAN policy.  See
        ``chance_cutin_clearance``.

        ``chancecells`` (2026-11-10, maintainer: a name without the beta_ref the NGSIM design no longer uses) is the
        same switch with ``beta_ref`` 0.6, i.e. ``cutin_chance0.6``: the cut-in cells become chance cells (the cells
        that ``dopt<w>`` prices and ``floorrisk<eps>`` floors) and the sigma terms are on.  Whole-token match; not
        together with ``cutin_chance<value>``.
        """
        match = re.search(r"cutin_chance([0-9]*\.?[0-9]+)", str(smpc_config))
        cells = re.search(r"(?:^|_)chancecells(?=_|$)", str(smpc_config)) is not None
        if match and cells:
            raise ValueError("chancecells is cutin_chance0.6 under another name; use one of them")
        if cells:
            return 0.6
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_chance_sigma_cap(smpc_config):
        """Read ``sigmacap<value>`` from the config string (None: ``beta_ref`` caps it).

        The chance constraint's sigma term is then asked for confidence
        ``min(p, value)`` while the standoff factor keeps ``min(p, beta_ref)``.
        Needs ``cutin_chance<beta_ref>``.  See ``chance_cutin_clearance``.
        """
        match = re.search(r"sigmacap([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_standoff_shortfall_weight(smpc_config):
        """Read ``dopt<w>`` from the config string (None = off).

        On cut-in chance cells ``alpha_j`` scales the footprint ``L + d0`` only and
        the optimizer picks how much of the time gap ``tau * v`` to keep, at a
        cost ``w * P_j * shortfall^2``.  Needs ``cutin_chance<beta_ref>``.  See
        ``NairACCConfig.standoff_shortfall_weight``.
        """
        match = re.search(r"dopt([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_standoff_shortfall_on_lk(smpc_config):
        """Read the ``lk`` suffix of ``dopt<w>lk`` (False = cut-in cells only).

        With it the shortfall is priced on the ego-lane lead's lane-keeping
        cells too, so a cut-in keeps its price once it is in the lane.  See
        ``NairACCConfig.standoff_shortfall_on_lk``.
        """
        return re.search(r"dopt[0-9]*\.?[0-9]+lk", str(smpc_config)) is not None

    @staticmethod
    def _parse_standoff_shortfall_lk_cutin_only(smpc_config):
        """Read the ``r`` of ``dopt<w>lkr`` (False = every in-lane lead is priced).

        With it a lane-keeping cell is priced only while its lead came in as a
        cut-in and the gap to it has not yet recovered the full standoff.  See
        ``NairACCConfig.standoff_shortfall_lk_cutin_only``.
        """
        return re.search(r"dopt[0-9]*\.?[0-9]+lkr", str(smpc_config)) is not None

    @staticmethod
    def _parse_lk_release_sigma(smpc_config):
        """Read ``lkrsigma`` (default: off): with ``dopt<w>lkr`` and ``lksigma``, the release test also asks for
        the step-0 sigma term. See ``NairACCConfig.standoff_shortfall_lk_release_sigma``. Whole-token match."""
        return re.search(r"(?:^|_)lkrsigma(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_standoff_shortfall_cell_price(smpc_config):
        """Read ``pricep2`` (False = off).

        With it a ``dopt<w>`` cell's shortfall costs ``w * P_j * p_cell * short^2``,
        ``p_cell`` the cell's own mode probability, so an unlikely cut-in plans a
        shorter standoff.  See ``NairACCConfig.standoff_shortfall_cell_price``.
        """
        return "pricep2" in str(smpc_config)

    @staticmethod
    def _parse_branch_decision_brake(smpc_config):
        """Read ``branchdec<a>`` from the config string (None = off).

        With it the policy tree splits a cut-in hypothesis from lane keeping at
        the target's decision point -- the step at which a brake at ``a`` m/s^2
        would just settle behind its own lead -- instead of at the step its
        predicted trajectory enters the ego lane.  See
        ``acc_postprocess.decision_point_branch_step``.
        """
        match = re.search(r"branchdec([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_context_trust(smpc_config):
        """Read ``ctx<lambda>`` from the config string (None = off).

        Context trust ``lambda`` in [0, 1]: an adjacent vehicle's cut-in
        intention is mixed with its measured lateral motion toward the ego
        lane, ``p = p_S [(1 - lambda) + lambda o_v]``, before the mode mapping.
        The token is matched between ``_`` separators so that it can neither
        catch nor be caught by another token.  See
        ``acc_postprocess.mix_cutin_probability``.
        """
        match = re.search(r"(?:^|_)ctx([0-9]*\.?[0-9]+)(?=_|$)", str(smpc_config))
        return float(match.group(1)) if match else None

    @staticmethod
    def _parse_lateral_gate(smpc_config):
        """Read ``latgate`` from the config string (False = ``ctx``'s offset ramp).

        With it ``ctx<lambda>`` reads a latched lateral gate instead of the
        offset ramp: a vehicle counts as moving laterally once it is past the
        noise offset, moving toward the ego lane at >= 0.2 m/s over the last
        0.5 s and predicted to cut in with p_S >= 0.9, and stays so while it is
        past the noise offset.  Whole token only, like ``ctx<lambda>``.  See
        ``acc_postprocess.mix_cutin_probability``.
        """
        return re.search(r"(?:^|_)latgate(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_entry_pass(smpc_config):
        """Read ``entrypass`` from the config string (False = the current-position lead filter only).

        With it a cut-in hypothesis whose vehicle enters the ego lane beside or
        behind the ego's position planned at the previous tick (negative bumper
        gap at its first in-lane step) never leads.  See
        ``acc_postprocess.entry_ego_displacement`` and
        ``build_multitarget_lead_prediction``.
        """
        return "entrypass" in str(smpc_config)

    @staticmethod
    def _parse_entry_pass_margin(smpc_config):
        """Read the margin [m] of ``entrypass<m>`` (``entrypass`` alone: 0).

        A cut-in hypothesis is dropped only when its entry bumper gap to the
        planned ego is below ``-m``; ``entrypass9`` asks for the whole vehicle
        (4.5 m) behind the ego's rear.
        """
        match = re.search(r"entrypass([0-9]+(?:\.[0-9]+)?)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_pass_miqp(smpc_config):
        """Read ``passmiqp<m>`` from the config string (default: off, ``None``; ``passmiqp`` alone: m = 0).

        The pass / yield choice for an entering cut-in hypothesis made inside the optimization, one binary per
        candidate in one MIQP per tick (``NairACCConfig.pass_miqp_margin``, 2026-11-04, final-check item 8):
        on the pass branch the ego's front must be at least m metres ahead of the vehicle's rear at the step
        it would lead, without planning more acceleration than it has.  It replaces ``entrypass``'s test on
        the remembered plan (that token may stay in the string).  Whole-token match.
        """
        match = re.search(r"(?:^|_)passmiqp([0-9]*\.?[0-9]*)(?=_|$)", str(smpc_config))
        if match is None:
            return None
        return float(match.group(1)) if match.group(1) else 0.0

    @staticmethod
    def _parse_pass_no_cap(smpc_config):
        """Read ``passnocap`` from the config string (default: off).

        With ``passmiqp<m>``: no acceleration cap on the pass branch, so the MIQP may plan to accelerate past
        a merging car within the design's own limits; the candidate test reads the maximum-acceleration
        displacement (``NairACCConfig.pass_miqp_no_cap``, 2026-11-04). Whole-token match.
        """
        return re.search(r"(?:^|_)passnocap(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_lk_entry_as_cutin(smpc_config):
        """Read ``trajcutin`` from the config string (False = map by label).

        With it an adjacent vehicle's lane-keeping-labelled trajectory that
        enters the ego lane within the horizon is a cut-in mode of its own.  See
        ``acc_postprocess._mode_specs_for_relation``.
        """
        return "trajcutin" in str(smpc_config)

    @staticmethod
    def _parse_no_cutout(smpc_config):
        """Read ``nocutout`` from the config string (2026-10-28).

        With it an ego-lane vehicle gets no cut-out mode: it keeps its lane with
        probability 1 and its full standoff.  See
        ``acc_postprocess._mode_specs_for_relation``.
        """
        return "nocutout" in str(smpc_config)

    @staticmethod
    def _parse_lk_sigma(smpc_config):
        """Read ``lksigma`` from the config string (2026-10-28).

        With it the ego-lane lead's lane-keeping cells also get the confidence
        chance constraint's sigma term, ``q(min(p_lk, beta_ref)) * sigma``; the
        standoff stays hard.  See ``acc_postprocess.chance_cutout_clearance``.
        """
        return "lksigma" in str(smpc_config)

    @staticmethod
    def _parse_full_footprint(smpc_config):
        """Read ``fullfoot`` from the config string (2026-10-29).

        With it a cut-in chance cell keeps the whole footprint ``L + d0`` under
        ``dopt<w>`` (factor 1 in place of ``alpha_j``); the sigma term keeps
        ``beta_j``.  Whole token only.  See
        ``acc_postprocess.chance_cutin_clearance``.
        """
        return re.search(r"(?:^|_)fullfoot(?=_|$)", str(smpc_config)) is not None

    @staticmethod
    def _parse_cutin_clearance_tlc_ref(smpc_config):
        """Read ``cutin_tlc<value>`` from the config string (default: disabled).

        The value is the time-to-lane-crossing at which a cut-in mode stops
        reserving standoff; it keeps the full standoff at zero.  See
        ``tlc_cutin_clearance``.
        """
        match = re.search(r"cutin_tlc([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_gap_recovery_s(smpc_config):
        """Read ``gap_recover<value>`` from the config string (default: disabled).

        The value is how long the standoff takes to return to its full time gap
        after a vehicle enters the ego lane.  See ``gap_reestablish_clearance``.
        """
        match = re.search(r"gap_recover([0-9]*\.?[0-9]+)", str(smpc_config))
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _parse_mode_probabilities(smpc_config):
        if "cutin_high" in str(smpc_config):
            return [0.3, 0.7]
        return [0.6, 0.4]

    # Half a lane: a vehicle whose centre is within it is in the ego lane even
    # when the waypoint test fails because the two vehicles sit on different
    # road segments (Town04 road 47 -> 1073), which otherwise drops an in-lane
    # lead for a second or two.
    EGO_LANE_HALF_WIDTH_M = 1.75

    @classmethod
    def _relation_from_frenet_offset(cls, d_tv):
        if d_tv > cls.EGO_LANE_HALF_WIDTH_M:
            return REL_LEFT_ADJACENT
        if d_tv < -cls.EGO_LANE_HALF_WIDTH_M:
            return REL_RIGHT_ADJACENT
        return REL_EGO_LANE

    @staticmethod
    def _lateral_offset_toward_ego_lane(location, waypoint, relation):
        """Signed cross-track offset [m] of a vehicle's centre from its own lane
        centre, positive toward the ego lane (``ctx<lambda>``'s ``y_v``).

        The lane centre is the CARLA map's waypoint, not +/-3.5 m from the ego
        path: near a Town04 segment boundary (y ~ 47) adjacent traffic runs
        2.0 m from the ego path.  Both poses are taken into the repository's
        right-handed frame (y -> -y, yaw -> -yaw), where +d is to the left of
        the lane heading; a left-adjacent vehicle moves toward the ego lane by
        going right, so its sign flips.
        """
        centre = waypoint.transform.location
        psi = -fth.fix_angle(np.radians(waypoint.transform.rotation.yaw))
        dx = float(location.x) - float(centre.x)
        dy = -float(location.y) + float(centre.y)
        d_left = -np.sin(psi) * dx + np.cos(psi) * dy
        return float(-d_left if relation == REL_LEFT_ADJACENT else d_left)

    @staticmethod
    def _lateral_speed_left(velocity, waypoint):
        """A vehicle's velocity [m/s] along the left normal of its own lane's
        heading, in the right-handed frame of ``_lateral_offset_toward_ego_lane``.

        The projection reads only the lane heading, so it is continuous when
        the lane assignment flips; differencing the offset from the lane centre
        would jump by a lane width there.  The ego path's Frenet d is not used
        either: near the Town04 segment boundary it shifts ~1.5 m against the
        adjacent lanes, and a position chord on a curve would add R(1 - cos).
        """
        psi = -fth.fix_angle(np.radians(waypoint.transform.rotation.yaw))
        vx = float(velocity.x)
        vy = -float(velocity.y)
        return float(-np.sin(psi) * vx + np.cos(psi) * vy)

    @staticmethod
    def _lateral_speed_toward_ego_lane(samples, time_s, relation):
        """``latgate``'s v_y: the mean of the ``(time_s, left-normal speed)``
        samples in the last ``CTX_GATE_VY_WINDOW_S``, signed toward the ego lane
        like ``_lateral_offset_toward_ego_lane``; None with less than
        ``CTX_GATE_VY_MIN_HISTORY_S`` of history.
        """
        window = [(t, v) for t, v in samples
                  if float(time_s) - CTX_GATE_VY_WINDOW_S < t <= float(time_s)]
        if not window or float(time_s) - window[0][0] < CTX_GATE_VY_MIN_HISTORY_S - 1.0e-6:
            return None
        v_left = float(np.mean([v for _, v in window]))
        return float(-v_left if relation == REL_LEFT_ADJACENT else v_left)

    @staticmethod
    def _lane_keeping_steer(ey, epsi):
        return lane_keeping_steer(ey, epsi)

    def _min_gap(self, solution, prediction):
        min_gap = np.inf
        linearization_speed = float(solution.x_nominal[0, 0, 1])
        for mode in range(prediction.num_modes):
            for step in range(prediction.horizon + 1):
                if not prediction.active_mask[mode, step]:
                    continue
                min_gap = min(
                    min_gap,
                    safety_function(
                        solution.x_nominal[mode, step],
                        prediction.means[mode, step],
                        self.controller_config,
                        linearization_speed=linearization_speed,
                    ),
                )
        return float(min_gap) if np.isfinite(min_gap) else np.nan

    def _actor_state_rhs(self, actor):
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        x = transform.location.x
        y = -transform.location.y
        psi = -fth.fix_angle(np.radians(transform.rotation.yaw))
        speed = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2))
        return x, y, psi, speed

    def _command_dt(self, time_s):
        fallback_dt = max(float(self.DT), 1.0e-6)
        if time_s is None or self._prev_accel_cmd_time_s is None:
            return fallback_dt
        dt = float(time_s) - float(self._prev_accel_cmd_time_s)
        if not np.isfinite(dt) or dt <= 1.0e-6:
            return fallback_dt
        return dt

    def _limit_accel_command(self, action, previous_accel_cmd, command_dt):
        """Clip to the limits, then the command jerk limit around the previous command
        (``utils.acc_nair_smpc.limit_accel_command``; ``jlimup<x>`` bounds the rising side apart).

        ``safelayer`` (``failsafe_recovery``): while the previous command lies below ``a_min`` (the layer's
        fail-safe) the window is the fail-safe's jerk around it, within its deceleration and ``a_max``, so the
        MPC's recovery ramp passes as planned instead of being clipped to ``a_min`` at once."""
        return limit_accel_command(self.controller_config, action, previous_accel_cmd, command_dt)

    def _observe_safety_layer(self, time_s, s_ego, ey, command_dt):
        """``safelayer``: PC-ACC's perception this tick (``policies/pcacc_agent.observe_preceding_vehicles``):
        waypoint lanes, the paper's lateral cut-in rule and leave lag, and physical bumper gaps. An adjacent
        vehicle counts only while its body overlaps the ego's laterally. Advances the layer's clock; once per
        tick."""
        from policies.pcacc_agent import observe_preceding_vehicles
        # the membership rule's clocks need a time even when the snapshot has none (PC-ACC's agent does the same)
        self._safety_clock = float(time_s) if time_s is not None else self._safety_clock + float(command_dt)
        ego_waypoint = self.world.get_map().get_waypoint(
            self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        vehicles, _ = observe_preceding_vehicles(
            self.world, self.vehicle, self._frenet_traj, self._safety_tracker, self.safety_layer.cfg,
            self._safety_clock, s_ego, ego_waypoint,
            ego_path=(float(ey), float(self.vehicle.bounding_box.extent.y)))
        return vehicles

    def _classify_safety_layer(self, time_s, s_ego, ey, speed, previous_accel_cmd, command_dt):
        """``safebranch``: observe and classify (``SafetyLayer.classify``) before the MPC solve. Returns
        ``(checked, cutins)``, to be given to ``_apply_safety_layer`` after the solve."""
        vehicles = self._observe_safety_layer(time_s, s_ego, ey, command_dt)
        return self.safety_layer.classify(self._safety_clock, speed, previous_accel_cmd, vehicles, command_dt)

    def _apply_safety_layer(self, time_s, s_ego, ey, speed, previous_accel_cmd, action, command_dt, members=None):
        """``safelayer`` (2026-11-02): PC-ACC's safety layer (``utils/safety_layer.py``) around the limited MPC
        command, on PC-ACC's perception (``_observe_safety_layer``). Returns the layer's ``SafetyDecision``.
        With ``members``, this tick's ``(checked, cutins)`` from ``_classify_safety_layer`` (``safebranch``),
        the layer only decides; otherwise it observes, classifies and decides (``SafetyLayer.step``)."""
        self._safety_nominal_accel = float(action)
        if members is None:
            vehicles = self._observe_safety_layer(time_s, s_ego, ey, command_dt)
            decision = self.safety_layer.step(
                self._safety_clock, speed, previous_accel_cmd, action, vehicles, command_dt)
        else:
            checked, cutins = members
            decision = self.safety_layer.decide(
                self._safety_clock, speed, previous_accel_cmd, action, checked, cutins, command_dt)
        return decision

    def _mpc_previous_accel(self, previous_accel_cmd, speed, command_dt):
        """``safelayer``: the previous command the MPC starts from (item 7, 2026-11-04).

        After the layer's fail-safe that command lies below ``a_min``. The MPC takes it as it is and plans the
        recovery (``NairACCConfig.failsafe_recovery``); at low speed it is raised to
        ``failsafe_recovery_floor``, below which the recovery cannot keep v >= 0 (the limiter then holds the
        command on the ramp). Otherwise it is unchanged."""
        cfg = self.controller_config
        if not cfg.failsafe_recovery or float(previous_accel_cmd) >= cfg.a_min:
            return previous_accel_cmd
        return float(max(previous_accel_cmd, failsafe_recovery_floor(speed, cfg, command_dt)))

    def _command_jerk(self, action, command_dt):
        if self._prev_accel_cmd_time_s is None:
            return None
        dt = max(float(command_dt), 1.0e-6)
        return float((float(action) - float(self._prev_accel_cmd)) / dt)

    def _remember_accel_cmd(self, time_s, action):
        self._prev_accel_cmd = float(action)
        self._prev_accel_cmd_time_s = None if time_s is None else float(time_s)

    def _actual_longitudinal_accel(self):
        _, _, psi, _ = self._ego_state_rhs()
        accel = self.vehicle.get_acceleration()
        accel_rhs = np.array([float(accel.x), -float(accel.y)], dtype=float)
        heading_rhs = np.array([np.cos(psi), np.sin(psi)], dtype=float)
        return float(np.dot(accel_rhs, heading_rhs))

    def _actual_longitudinal_jerk(self, time_s, actual_accel):
        if time_s is None or actual_accel is None:
            return None
        time_s = float(time_s)
        actual_accel = float(actual_accel)
        previous = self._prev_actual_accel_sample
        self._prev_actual_accel_sample = (time_s, actual_accel)
        if previous is None:
            return None
        prev_time, prev_accel = previous
        dt = time_s - prev_time
        if dt <= 1.0e-6:
            return None
        return float((actual_accel - prev_accel) / dt)

    def _elapsed_seconds(self):
        snapshot = self.world.get_snapshot()
        if snapshot is None or snapshot.timestamp is None:
            return None
        return snapshot.timestamp.elapsed_seconds

    def _log_wandb(self, row, solution, prediction_bundle):
        if not self.enable_wandb_logging:
            return
        try:
            import wandb
            if wandb.run is None:
                wandb.init(project="carla", name=f"acc_nair_{self.predictor_type}", reinit=True)
            payload = {
                "acc/time_s": row["time_s"],
                "acc/ego_s": row["ego_s"],
                "acc/ego_v": row["ego_v"],
                "acc/accel_cmd": row["accel_cmd"],
                "acc/accel_cmd_raw": row.get("accel_cmd_raw"),
                "acc/command_jerk": row.get("command_jerk"),
                "acc/command_jerk_limit": row.get("command_jerk_limit"),
                "acc/min_gap": row["min_gap"],
                "acc/feasible": float(row["feasible"]),
                "acc/solve_time": row["solve_time"],
                "acc/chance_margin_min": solution.chance_margin_min,
                "acc/first_policy_split_step": (
                    -1 if solution.first_policy_split_step is None
                    else int(solution.first_policy_split_step)
                ),
                "acc/policy_branch_step": (
                    -1 if solution.policy_branch_step is None
                    else int(solution.policy_branch_step)
                ),
            }
            for target in prediction_bundle.get("processed_targets", []):
                prefix = f"stdan/{target.vehicle_id}"
                for key, value in target.raw_mode_prob.items():
                    payload[f"{prefix}/raw_{key}"] = value
                for key, value in target.acc_mode_prob.items():
                    payload[f"{prefix}/acc_{key}"] = value
                payload[f"{prefix}/ego_lane_membership_split_step"] = target.branch_info.get(
                    "ego_lane_membership_split_step",
                    target.branch_info["branch_step"],
                )
                payload[f"{prefix}/branch_step"] = target.branch_info["branch_step"]
                payload[f"{prefix}/ego_lane_start_idx"] = target.branch_info["ego_lane_start_idx"]
                payload[f"{prefix}/ego_lane_end_idx"] = target.branch_info["ego_lane_end_idx"]
                if target.raw_prediction is not None:
                    payload[f"{prefix}/signed_t_cross"] = target.raw_prediction.get("signed_t_cross", 0.0)
            wandb.log(payload)
        except Exception:
            return
