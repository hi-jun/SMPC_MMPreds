"""
ACC-oriented multimodal SMPC controller inspired by Nair et al.

This module keeps the mode/policy structure of the 2D SMPC_MMPreds
controller, but reduces the planning problem to 1D Frenet ACC:

    ego state x = [s, v]^T, input u = a
    lead state o = [s_lead, v_lead]^T

The old longitudinal ACC MPC in ``multimodal/controller/mpc_controller.py``
already contains the desired-speed ramp and time-gap lead-vehicle reference
logic.  The ``OldACCReferenceAdapter`` below exposes that logic in a small
    standalone form so this Nair-style controller can reuse it without importing
    the OSQP-based controller during unit tests.

The optimization backend follows the original SMPC_MMPreds implementation:
CasADi Opti builds a convex QP and delegates it to Gurobi.  This replaces the
earlier SciPy/SLSQP prototype, which was too slow for receding-horizon use.
"""

from dataclasses import dataclass
import glob
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


def _configure_gurobi_version():
    if os.getenv("GUROBI_VERSION"):
        return

    gurobi_home = os.getenv("GUROBI_HOME")
    if not gurobi_home:
        return

    for lib_path in glob.glob(os.path.join(gurobi_home, "lib", "libgurobi*.so")):
        match = re.match(r"libgurobi(\d+)\.so$", os.path.basename(lib_path))
        if match:
            os.environ["GUROBI_VERSION"] = match.group(1)
            return


_configure_gurobi_version()

import casadi as ca


NX = 2
NU = 1
NO = 2

VARIANT_PROPOSED = "proposed"
VARIANT_FIXED_RISK = "fixed_risk"
VARIANT_MULTIMODAL_OL = "multimodal_ol"

SAFETY_NOMINAL_SAFE_DISTANCE = "nominal_safe_distance"
SAFETY_SCALAR_CHANCE = "scalar_chance"
SAFETY_BRAKE_DISTANCE = "brake_distance"
SAFETY_CONFIDENCE_CHANCE = "confidence_chance"
CHANCE_SAFETY_MODES = (SAFETY_SCALAR_CHANCE, SAFETY_CONFIDENCE_CHANCE)

BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL = "reachable_interval"
BRAKE_DISTANCE_BOUND_HARD_BAND = "hard_band"

RISK_FIXED = "fixed"
RISK_PROBABILITY_WEIGHTED = "probability_weighted"
RISK_OPTIMIZED_ETA = "optimized_eta"

DEFAULT_CDF_APPROX_SEGMENTS = (
    ((0.02, 1.35), (0.508, 0.91)),
    ((1.35, 2.0), (0.91, 0.978)),
)


@dataclass
class NairACCConfig:
    """Configuration for the 1D ACC adaptation."""

    horizon: int = 10
    dt: float = 0.2
    desired_speed: float = 20.0
    d0: float = 3.0
    time_headway: float = 1.3
    vehicle_length: float = 4.5
    v_min: float = 0.0
    v_max: float = 35.0
    a_min: float = -8.0
    a_max: float = 3.0
    jerk_limit: Optional[float] = 4.0
    epsilon: float = 0.05
    fixed_risk_level: Optional[float] = None
    eta_max: float = 0.49
    eta_std_min: float = 1.0e-7
    eta_std_max: float = 3.0
    cdf_approx_segments: Tuple[
        Tuple[Tuple[float, float], Tuple[float, float]], ...
    ] = DEFAULT_CDF_APPROX_SEGMENTS
    use_variable_risk: bool = False
    safety_constraint_mode: str = SAFETY_NOMINAL_SAFE_DISTANCE
    brake_distance_bound_mode: str = BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL
    brake_distance_hard_band_width: float = 4.0
    risk_allocation_mode: str = RISK_FIXED
    num_modes: int = 2
    controller_variant: str = VARIANT_PROPOSED
    process_noise_cov: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.01, 0.0),
        (0.0, 0.04),
    )
    tv_prediction_cov: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (0.25, 0.0),
        (0.0, 0.09),
    )
    q_s: float = 0.5
    q_v: float = 8.0
    r_a: float = 0.5
    r_jerk: float = 1.0
    slack_weight: float = 5000.0
    solver_name: str = "gurobi"
    gurobi_output: bool = False
    gurobi_time_limit: Optional[float] = None
    allow_solver_fallback: bool = True
    max_slsqp_iter: int = 120
    optimizer_ftol: float = 1.0e-4
    feedback_bound: float = 2.0
    disturbance_feedback_bound: float = 0.5
    optimize_k: bool = False

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.num_modes < 1:
            raise ValueError("num_modes must be positive")
        if self.controller_variant not in (
            VARIANT_PROPOSED,
            VARIANT_FIXED_RISK,
            VARIANT_MULTIMODAL_OL,
        ):
            raise ValueError("invalid controller_variant")
        if self.safety_constraint_mode not in (
            SAFETY_NOMINAL_SAFE_DISTANCE,
            SAFETY_SCALAR_CHANCE,
            SAFETY_BRAKE_DISTANCE,
            SAFETY_CONFIDENCE_CHANCE,
        ):
            raise ValueError("invalid safety_constraint_mode")
        if self.brake_distance_bound_mode not in (
            BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL,
            BRAKE_DISTANCE_BOUND_HARD_BAND,
        ):
            raise ValueError("invalid brake_distance_bound_mode")
        if self.brake_distance_hard_band_width <= 0.0:
            raise ValueError("brake_distance_hard_band_width must be positive")
        if self.risk_allocation_mode not in (
            RISK_FIXED,
            RISK_PROBABILITY_WEIGHTED,
            RISK_OPTIMIZED_ETA,
        ):
            raise ValueError("invalid risk_allocation_mode")
        if (
                self.risk_allocation_mode == RISK_OPTIMIZED_ETA
                and self.safety_constraint_mode != SAFETY_SCALAR_CHANCE):
            raise ValueError("optimized_eta risk allocation requires scalar_chance safety constraints")
        if self.eta_std_min < 0.0:
            raise ValueError("eta_std_min must be non-negative")
        if self.eta_std_max <= self.eta_std_min:
            raise ValueError("eta_std_max must be larger than eta_std_min")
        if self.jerk_limit is not None and self.jerk_limit < 0.0:
            raise ValueError("jerk_limit must be non-negative or None")
        if self.solver_name != "gurobi":
            raise ValueError("NairACCSMPC currently supports only the gurobi solver backend")


def original_nair_acc_config(
        horizon=10,
        dt=0.2,
        desired_speed=15.0,
        num_modes=2,
        controller_variant=VARIANT_PROPOSED,
        gurobi_time_limit=None,
        max_slsqp_iter=120):
    """
    Map the original SMPC_MMPreds Nair controller defaults to the 1D ACC port.

    Source defaults are from ``scripts/carla/utils/mpc_utils.py::SMPC_MMPreds``:
    N=10, DT=0.2, V_MIN=0, V_MAX=15, A_MIN=-3, A_MAX=2,

    ``v_max`` deviates from that source and is raised to 20.  The source value of
    15 sits below the speeds the ACC scenarios ask for: an ego spawned at 16 m/s
    leaves the desired speed permanently above the state bound, so the tracking
    error never reaches zero and the cost keeps commanding acceleration.  In the
    2026-09-02 sweep every ego-16 run collided under all three policies for this
    reason, while every ego-14 run was collision free.
    A_DOT_MIN=-1.5, A_DOT_MAX=1.5, TIGHTENING=1.64,
    NOISE_STD=[0.1, .1, .01, .1, .01], Q=[5, 2.5, 10, 1], R=[10, 1000].

    ``jerk_limit`` deviates from that source's 1.5 and is raised to 10.  At 1.5
    the command needs 2 s to reach ``a_min``, which is longer than the 1.0-1.3 s
    time-to-collision a +6 m/s cut-in leaves: every policy collided in those
    cells for a reason that has nothing to do with how well it predicts.  The
    comfort metrics are reported on the commanded jerk, which the same limit
    bounds, so the looser limit widens the range the controllers can differ in
    rather than hiding the difference.
    """
    tightening = 1.64
    return NairACCConfig(
        horizon=horizon,
        dt=dt,
        desired_speed=desired_speed,
        v_min=0.0,
        v_max=20.0,
        a_min=-3.0,
        a_max=2.0,
        jerk_limit=10.0,
        epsilon=1.0 - norm.cdf(tightening),
        fixed_risk_level=1.0 - norm.cdf(tightening),
        eta_max=0.49,
        use_variable_risk=False,
        safety_constraint_mode=SAFETY_NOMINAL_SAFE_DISTANCE,
        risk_allocation_mode=RISK_FIXED,
        num_modes=num_modes,
        controller_variant=controller_variant,
        process_noise_cov=((0.1 ** 2, 0.0), (0.0, 0.1 ** 2)),
        tv_prediction_cov=((0.01 ** 2, 0.0), (0.0, 0.01 ** 2)),
        q_s=0.0,
        q_v=1.0,
        r_a=10.0,
        r_jerk=10.0,
        slack_weight=5000.0,
        solver_name="gurobi",
        gurobi_output=False,
        gurobi_time_limit=gurobi_time_limit,
        allow_solver_fallback=True,
        max_slsqp_iter=max_slsqp_iter,
        optimizer_ftol=1.0e-4,
        feedback_bound=2.0,
        disturbance_feedback_bound=0.5,
    )


@dataclass
class ACCReference:
    """Mode-dependent ACC reference plus the input trajectory used for linearization."""

    s_ref: np.ndarray
    v_ref: np.ndarray
    a_ref: np.ndarray
    prev_u: np.ndarray

    @property
    def x_ref(self) -> np.ndarray:
        return np.stack((self.s_ref, self.v_ref), axis=-1)


@dataclass
class PolicySharingTree:
    """
    Non-anticipative ego-policy sharing metadata.

    ``policy_groups_by_step[j, k]`` is the policy group used by joint scenario
    ``j`` at control step ``k``.  Modes with the same group at a step share h
    and M.  K uses a separate target-local grouping.
    """

    policy_groups_by_step: np.ndarray
    policy_split_steps: Tuple[int, ...] = ()
    mode_names: Optional[List[str]] = None

    def __post_init__(self):
        self.policy_groups_by_step = np.asarray(self.policy_groups_by_step, dtype=int)
        if self.policy_groups_by_step.ndim != 2:
            raise ValueError("policy_groups_by_step must have shape (J, N)")
        if np.any(self.policy_groups_by_step < 0):
            raise ValueError("policy_groups_by_step entries must be non-negative")
        horizon = self.horizon
        split_steps = []
        for step in self.policy_split_steps:
            step = int(step)
            if 0 <= step < horizon and step not in split_steps:
                split_steps.append(step)
        self.policy_split_steps = tuple(split_steps)

    @property
    def num_modes(self) -> int:
        return self.policy_groups_by_step.shape[0]

    @property
    def horizon(self) -> int:
        return self.policy_groups_by_step.shape[1]

    @property
    def first_policy_split_step(self) -> Optional[int]:
        if not self.policy_split_steps:
            return None
        return int(self.policy_split_steps[0])

    @property
    def branch_step(self) -> int:
        """Compatibility alias for older diagnostics/tests."""
        first = self.first_policy_split_step
        return self.horizon if first is None else int(first)

    @property
    def mode_groups_by_step(self) -> np.ndarray:
        """Compatibility alias for the old PredictionTree field name."""
        return self.policy_groups_by_step

    def trimmed(self, horizon):
        if horizon > self.horizon:
            raise ValueError("cannot trim policy sharing tree to a longer horizon")
        return PolicySharingTree(
            policy_groups_by_step=self.policy_groups_by_step[:, :horizon],
            policy_split_steps=tuple(step for step in self.policy_split_steps if step < horizon),
            mode_names=self.mode_names,
        )

    @staticmethod
    def single_split(num_modes, horizon, split_step, mode_names=None):
        split_step = int(np.clip(split_step, 0, horizon))
        groups = np.zeros((num_modes, horizon), dtype=int)
        group_id = 0
        for step in range(horizon):
            if step < split_step:
                groups[:, step] = group_id
                group_id += 1
            else:
                for mode in range(num_modes):
                    groups[mode, step] = group_id
                    group_id += 1
        return PolicySharingTree(
            policy_groups_by_step=groups,
            policy_split_steps=() if split_step >= horizon else (split_step,),
            mode_names=mode_names,
        )

    @staticmethod
    def single_branch(num_modes, horizon, branch_step, mode_names=None):
        """Compatibility factory for older callers."""
        return PolicySharingTree.single_split(
            num_modes=num_modes,
            horizon=horizon,
            split_step=branch_step,
            mode_names=mode_names,
        )


PredictionTree = PolicySharingTree


@dataclass
class MultimodalLeadPrediction:
    """
    Multimodal LTV prediction for the effective lead vehicle.

    The first axis indexes complete joint-mode sequences/leaf scenarios over
    the horizon.  With one target vehicle this is the target's mode index; with
    multiple targets it is the Cartesian-product joint mode retained by the
    prediction adapter.
    """

    means: np.ndarray
    probabilities: np.ndarray
    covariances: Optional[np.ndarray] = None
    T: Optional[np.ndarray] = None
    c: Optional[np.ndarray] = None
    mode_names: Optional[List[str]] = None
    policy_tree: Optional[PolicySharingTree] = None
    prediction_tree: Optional[PolicySharingTree] = None
    active_mask: Optional[np.ndarray] = None
    clearance_scale: Optional[np.ndarray] = None
    # Benciolini confidence beta_j per (mode, step); NaN marks a deterministic
    # cell (ego-lane lead) that gets no sigma-based tightening.
    chance_confidence: Optional[np.ndarray] = None
    k_group_map: Optional[np.ndarray] = None
    k_group_names: Optional[List[str]] = None

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=float)
        if self.means.ndim != 3 or self.means.shape[2] != NO:
            raise ValueError("means must have shape (J, N+1, 2)")

        self.probabilities = np.asarray(self.probabilities, dtype=float).reshape(-1)
        if self.probabilities.shape[0] != self.means.shape[0]:
            raise ValueError("probabilities must have one entry per mode")
        prob_sum = float(np.sum(self.probabilities))
        if prob_sum <= 0.0:
            self.probabilities = np.ones_like(self.probabilities) / self.probabilities.size
        else:
            self.probabilities = self.probabilities / prob_sum

        horizon = self.means.shape[1] - 1
        if self.covariances is None:
            self.covariances = np.zeros((self.num_modes, horizon + 1, NO, NO))
        else:
            self.covariances = np.asarray(self.covariances, dtype=float)
            expected = (self.num_modes, horizon + 1, NO, NO)
            if self.covariances.shape != expected:
                raise ValueError("covariances must have shape %s" % (expected,))

        if self.T is None or self.c is None:
            self.T, self.c = fit_ltv_from_means(self.means)
        if self.policy_tree is None and self.prediction_tree is not None:
            self.policy_tree = self.prediction_tree
        elif self.policy_tree is not None and self.prediction_tree is None:
            self.prediction_tree = self.policy_tree
        elif (
                self.policy_tree is not None
                and self.prediction_tree is not None
                and self.policy_tree is not self.prediction_tree):
            if (
                    not np.array_equal(
                        self.policy_tree.policy_groups_by_step,
                        self.prediction_tree.policy_groups_by_step,
                    )
                    or tuple(self.policy_tree.policy_split_steps)
                    != tuple(self.prediction_tree.policy_split_steps)):
                raise ValueError("policy_tree and prediction_tree must not disagree")
        if self.policy_tree is not None:
            if self.policy_tree.num_modes != self.num_modes:
                raise ValueError("policy_tree mode count must match prediction modes")
            if self.policy_tree.horizon != horizon:
                raise ValueError("policy_tree horizon must match prediction horizon")
        if self.active_mask is None:
            self.active_mask = np.ones((self.num_modes, horizon + 1), dtype=bool)
        else:
            self.active_mask = np.asarray(self.active_mask, dtype=bool)
            expected = (self.num_modes, horizon + 1)
            if self.active_mask.shape != expected:
                raise ValueError("active_mask must have shape %s" % (expected,))
        if self.clearance_scale is None:
            self.clearance_scale = np.ones((self.num_modes, horizon + 1), dtype=float)
        else:
            self.clearance_scale = np.asarray(self.clearance_scale, dtype=float)
            expected = (self.num_modes, horizon + 1)
            if self.clearance_scale.shape != expected:
                raise ValueError("clearance_scale must have shape %s" % (expected,))
            if np.any(self.clearance_scale < 0.0) or np.any(self.clearance_scale > 1.0):
                raise ValueError("clearance_scale entries must lie in [0, 1]")
        if self.chance_confidence is None:
            self.chance_confidence = np.full((self.num_modes, horizon + 1), np.nan)
        else:
            self.chance_confidence = np.asarray(self.chance_confidence, dtype=float)
            expected = (self.num_modes, horizon + 1)
            if self.chance_confidence.shape != expected:
                raise ValueError("chance_confidence must have shape %s" % (expected,))
            finite = self.chance_confidence[np.isfinite(self.chance_confidence)]
            if np.any(finite < 0.0) or np.any(finite > 1.0):
                raise ValueError("finite chance_confidence entries must lie in [0, 1]")
        if self.k_group_map is not None:
            self.k_group_map = np.asarray(self.k_group_map, dtype=int)
            expected = (self.num_modes, horizon)
            if self.k_group_map.shape != expected:
                raise ValueError("k_group_map must have shape %s" % (expected,))
            if np.any(self.k_group_map < 0):
                raise ValueError("k_group_map entries must be non-negative")

    @property
    def num_modes(self) -> int:
        return self.means.shape[0]

    @property
    def num_joint_modes(self) -> int:
        return self.means.shape[0]

    @property
    def horizon(self) -> int:
        return self.means.shape[1] - 1


@dataclass
class NairACCPolicy:
    """
    Feedback policy adapted from Nair et al. to scalar ACC acceleration.

    h[j,k] is the mode-dependent open-loop acceleration deviation.
    M[j,k,l] maps ego process disturbance w_l to acceleration at k.
    K[j,k] maps lead prediction deviation (o_k - mu_k) to acceleration.
    """

    h: np.ndarray
    M: np.ndarray
    K: np.ndarray

    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=float)
        self.M = np.asarray(self.M, dtype=float)
        self.K = np.asarray(self.K, dtype=float)
        if self.h.ndim != 3 or self.h.shape[2] != NU:
            raise ValueError("h must have shape (J, N, 1)")
        if self.M.shape != (self.h.shape[0], self.h.shape[1], self.h.shape[1], NX):
            raise ValueError("M must have shape (J, N, N, 2)")
        if self.K.shape != (self.h.shape[0], self.h.shape[1], NO):
            raise ValueError("K must have shape (J, N, 2)")

    @property
    def num_modes(self) -> int:
        return self.h.shape[0]

    @property
    def horizon(self) -> int:
        return self.h.shape[1]

    def acceleration_deviation(
            self,
            mode: int,
            step: int,
            tv_deviation: Sequence[float],
            ego_disturbances: Optional[Sequence[Sequence[float]]] = None) -> float:
        correction = float(self.h[mode, step, 0])
        correction += float(np.dot(self.K[mode, step], np.asarray(tv_deviation, dtype=float)))
        if ego_disturbances is not None:
            for idx, disturbance in enumerate(ego_disturbances[:step]):
                correction += float(np.dot(self.M[mode, step, idx], disturbance))
        return correction


@dataclass
class NairACCSolution:
    """Controller output and diagnostics."""

    action: float
    status: str
    feasible: bool
    cost: float
    policy: NairACCPolicy
    reference: ACCReference
    x_nominal: np.ndarray
    u_nominal: np.ndarray
    safety_values: np.ndarray
    risk_levels: np.ndarray
    solver_message: str
    num_decision_variables: int = 0
    baseline_decision_variables: int = 0
    policy_split_steps: Tuple[int, ...] = ()
    first_policy_split_step: Optional[int] = None
    policy_group_count: int = 0
    safety_constraint_mode: str = SAFETY_NOMINAL_SAFE_DISTANCE
    risk_allocation_mode: str = RISK_FIXED
    risk_min: float = 0.0
    risk_max: float = 0.0
    tightening_min: float = 0.0
    tightening_max: float = 0.0
    chance_margin_min: float = 0.0
    eta_levels: Optional[np.ndarray] = None
    safe_probability_levels: Optional[np.ndarray] = None
    optimized_risk_levels: Optional[np.ndarray] = None
    target_safe_probability: float = 0.0
    eta_min: float = 0.0
    eta_max: float = 0.0
    safe_probability_min: float = 0.0
    safe_probability_max: float = 0.0
    optimized_risk_min: float = 0.0
    optimized_risk_max: float = 0.0
    solve_path: str = ""
    timing_breakdown: Optional[Dict[str, float]] = None

    @property
    def policy_branch_step(self) -> Optional[int]:
        """Compatibility alias for old logs."""
        return self.first_policy_split_step


@dataclass
class PolicyVariableLayout:
    group_map: np.ndarray
    group_steps: List[int]
    k_group_map: np.ndarray
    k_group_names: Optional[List[str]]
    k_group_count: int
    k_decision_variables: int
    num_decision_variables: int
    baseline_decision_variables: int
    policy_split_steps: Tuple[int, ...] = ()
    first_policy_split_step: Optional[int] = None

    @property
    def policy_group_count(self) -> int:
        return len(self.group_steps)

    @property
    def branch_step(self) -> Optional[int]:
        """Compatibility alias for old diagnostics."""
        return self.first_policy_split_step


@dataclass
class CachedOpenLoopFixedQP:
    opti: Any
    h_var: Any
    slack_var: Any
    x0_param: Any
    prev_u_param: Any
    previous_accel_param: Any
    command_dt_param: Any
    lead_s_param: Any
    lead_v_param: Any
    x_ref_s_param: Any
    x_ref_v_param: Any
    a_ref_param: Any
    probability_param: Any
    tightening_param: Any
    clearance_scale_param: Any
    brake_slope_param: Any
    brake_intercept_param: Any
    brake_v_min_param: Any
    brake_v_max_param: Any
    objective: Any


@dataclass
class CachedFeedbackScalarChanceQP:
    opti: Any
    h_var: Any
    k_var: Any
    k_group_capacity: int
    k_select_params: List[Any]
    slack_var: Any
    eta_var: Any
    rho_var: Any
    x0_param: Any
    prev_u_param: Any
    previous_accel_param: Any
    command_dt_param: Any
    lead_s_param: Any
    lead_v_param: Any
    x_ref_s_param: Any
    x_ref_v_param: Any
    a_ref_param: Any
    probability_param: Any
    tightening_param: Any
    clearance_scale_param: Any
    safety_std_param: Any
    brake_slope_param: Any
    brake_intercept_param: Any
    brake_v_min_param: Any
    brake_v_max_param: Any
    lead_dev_s_params: List[Any]
    lead_dev_v_params: List[Any]
    sharing_params: List[Tuple[int, int, Any]]
    objective: Any
    optimized_eta: bool
    optimize_k: bool


class OldACCReferenceAdapter:
    """
    Reference generator extracted from the older longitudinal MPC controller.

    It preserves the old controller's ACC intent:
    desired-speed ramp, time-gap policy ``d0 + tau*v``, lead-speed limiting,
    and acceleration/velocity clipping.  The result is exposed as
    ``s_ref``, ``v_ref`` and ``a_ref`` for tracking; the controller later
    attaches ``prev_u`` as the nominal input trajectory corrected by the policy.
    """

    def __init__(self, config):
        self.config = config
        self.previous_planned_speed = None

    def generate(self, ego_state, lead_prediction):
        s_ego = float(ego_state[0])
        v_ego = float(ego_state[1])
        horizon = self.config.horizon
        dt = self.config.dt

        base_v_ref = self._old_speed_ramp(v_ego)
        base_s_ref = np.zeros(horizon + 1)
        base_s_ref[0] = s_ego
        for step in range(horizon):
            base_s_ref[step + 1] = base_s_ref[step] + base_v_ref[step] * dt

        num_modes = lead_prediction.num_modes
        s_ref = np.tile(base_s_ref, (num_modes, 1))
        v_ref = np.tile(base_v_ref, (num_modes, 1))

        if self.previous_planned_speed is None:
            planned_v = np.full(horizon + 1, v_ego)
        else:
            planned_v = np.asarray(self.previous_planned_speed, dtype=float)
            if planned_v.shape[0] != horizon + 1:
                planned_v = np.full(horizon + 1, v_ego)

        for mode in range(num_modes):
            for step in range(horizon + 1):
                if not lead_prediction.active_mask[mode, step]:
                    continue
                lead_s = lead_prediction.means[mode, step, 0]
                lead_v = lead_prediction.means[mode, step, 1]
                safe_gap = self.config.vehicle_length + self.config.d0
                safe_gap += self.config.time_headway * planned_v[step]
                # Relax the reference by the same probability-driven factor the
                # collision constraint uses. Leaving it out makes the tracking
                # cost demand the full standoff while the constraint allows a
                # scaled one, so the ego brakes harder than the risk warrants.
                safe_gap *= float(lead_prediction.clearance_scale[mode, step])
                # Same reasoning for the sigma term: the constraint below adds
                # it, so the reference has to ask for it too.
                if self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
                    safe_gap += confidence_tightening(
                        lead_prediction.covariances[mode, step],
                        lead_prediction.chance_confidence[mode, step],
                        self.config,
                    )
                target_s = lead_s - safe_gap

                if target_s < s_ref[mode, step]:
                    s_ref[mode, step] = target_s

                v_ref[mode, step] = min(
                    v_ref[mode, step],
                    self._lead_limited_speed(
                        lead_s, lead_v, safe_gap, base_s_ref[step], base_v_ref[step]),
                )

        v_ref = np.clip(v_ref, self.config.v_min, self.config.v_max)

        a_ref = np.zeros((num_modes, horizon))
        for mode in range(num_modes):
            a_ref[mode] = np.diff(v_ref[mode]) / dt
        a_ref = np.clip(a_ref, self.config.a_min, self.config.a_max)
        return ACCReference(s_ref=s_ref, v_ref=v_ref, a_ref=a_ref, prev_u=a_ref.copy())

    @staticmethod
    def _lead_limited_speed(lead_s, lead_v, safe_gap, ego_s_at_step, free_speed):
        """Desired speed eased toward a lead's, by how much of the distance to
        it is still usable.

        The ego position is where it is *predicted* to be at this step, not
        where it is now: measuring a step-k gap from the current position
        inflates it by everything the ego covers in between, which keeps the
        blend near 1 and leaves the reference at the desired speed even where
        the lead is predicted to be in lane.
        """
        distance_to_lead = max(lead_s - ego_s_at_step, 1.0e-6)
        usable_gap = max((lead_s - safe_gap) - ego_s_at_step, 0.0)
        blend = float(np.clip(usable_gap / distance_to_lead, 0.0, 1.0))
        return blend * free_speed + (1.0 - blend) * lead_v

    def remember_planned_speed(self, x_nominal):
        self.previous_planned_speed = np.asarray(x_nominal[0, :, 1], dtype=float)

    def _old_speed_ramp(self, v_ego):
        reach_time = 2.0
        reach_steps = max(1, int(reach_time / self.config.dt))
        horizon = self.config.horizon
        v_ref = np.zeros(horizon + 1)
        if v_ego < self.config.desired_speed:
            ramp_steps = min(reach_steps, horizon + 1)
            v_ref[:ramp_steps] = np.linspace(v_ego, self.config.desired_speed, ramp_steps)
            if ramp_steps < horizon + 1:
                v_ref[ramp_steps:] = self.config.desired_speed
        else:
            v_ref[:] = self.config.desired_speed
        return v_ref


class NairACCSMPC:
    """Nair-style multimodal SMPC reduced to longitudinal ACC."""

    def __init__(self, config=None):
        self.config = config if config is not None else NairACCConfig()
        self.config.__post_init__()
        self.reference_adapter = OldACCReferenceAdapter(self.config)
        self.last_solution = None
        self._previous_accel_cmd = 0.0
        self._command_dt = float(self.config.dt)
        self._open_loop_fixed_qp_cache = {}
        self._feedback_scalar_chance_qp_cache = {}

    @property
    def A(self):
        return acc_dynamics_matrices(self.config.dt)[0]

    @property
    def B(self):
        return acc_dynamics_matrices(self.config.dt)[1]

    def initial_policy(self, num_modes=None):
        modes = num_modes if num_modes is not None else self.config.num_modes
        h = np.zeros((modes, self.config.horizon, NU))
        M = np.zeros((modes, self.config.horizon, self.config.horizon, NX))
        K = np.zeros((modes, self.config.horizon, NO))
        return NairACCPolicy(h=h, M=M, K=K)

    def solve(
            self,
            ego_state,
            lead_prediction,
            warm_start=None,
            previous_accel_cmd=None,
            command_dt=None):
        solve_start = time.perf_counter()
        timing_breakdown = {}
        self._set_command_context(previous_accel_cmd, command_dt)

        phase_start = time.perf_counter()
        prediction = self._trim_prediction(lead_prediction)
        timing_breakdown["trim_prediction_s"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reference = self.reference_adapter.generate(ego_state, prediction)
        timing_breakdown["reference_generate_s"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        reference = self._with_linearization_input(reference, prediction)
        timing_breakdown["reference_linearization_s"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        policy0 = warm_start if warm_start is not None else self.initial_policy(prediction.num_modes)
        timing_breakdown["initial_policy_s"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        risks = allocate_risk(prediction.probabilities, self.config)
        timing_breakdown["risk_allocation_s"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        if self.config.controller_variant == VARIANT_MULTIMODAL_OL:
            solution = self._solve_open_loop(ego_state, prediction, reference, risks, policy0)
        else:
            solution = self._solve_feedback(ego_state, prediction, reference, risks, policy0)
        timing_breakdown["feedback_or_ol_solve_s"] = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        self.reference_adapter.remember_planned_speed(solution.x_nominal)
        timing_breakdown["remember_planned_speed_s"] = time.perf_counter() - phase_start
        timing_breakdown["controller_total_s"] = time.perf_counter() - solve_start

        solution_timing = dict(solution.timing_breakdown or {})
        solution_timing.update(timing_breakdown)
        solution.timing_breakdown = solution_timing
        self.last_solution = solution
        return solution

    def compute_acceleration(self, ego_state, lead_prediction):
        return float(self.solve(ego_state, lead_prediction).action)

    def _set_command_context(self, previous_accel_cmd, command_dt):
        if previous_accel_cmd is None:
            previous_accel_cmd = 0.0
        previous_accel_cmd = float(previous_accel_cmd)
        if not np.isfinite(previous_accel_cmd):
            previous_accel_cmd = 0.0
        self._previous_accel_cmd = float(np.clip(
            previous_accel_cmd,
            self.config.a_min,
            self.config.a_max,
        ))

        if command_dt is None:
            command_dt = self.config.dt
        command_dt = float(command_dt)
        if not np.isfinite(command_dt) or command_dt <= 1.0e-6:
            command_dt = self.config.dt
        self._command_dt = max(float(command_dt), 1.0e-6)

    def _command_context(self):
        return self._previous_accel_cmd, self._command_dt

    def _jerk_limit_value(self):
        if self.config.jerk_limit is None:
            return None
        return float(self.config.jerk_limit)

    def _add_symbolic_jerk_constraints(self, opti, control, previous_input, dt):
        jerk_limit = self._jerk_limit_value()
        if jerk_limit is None:
            return
        opti.subject_to(control - previous_input <= jerk_limit * dt)
        opti.subject_to(previous_input - control <= jerk_limit * dt)

    def _rate_limited_accel(self, accel, previous_input, dt):
        accel = float(np.clip(accel, self.config.a_min, self.config.a_max))
        jerk_limit = self._jerk_limit_value()
        if jerk_limit is None:
            return accel
        dt = max(float(dt), 1.0e-6)
        max_delta = jerk_limit * dt
        return float(np.clip(
            accel,
            previous_input - max_delta,
            previous_input + max_delta,
        ))

    def _brake_speed_intervals(self, x0, num_modes):
        horizon = self.config.horizon
        x0_v = float(np.asarray(x0, dtype=float).reshape(-1)[1])
        if self.config.brake_distance_bound_mode == BRAKE_DISTANCE_BOUND_REACHABLE_INTERVAL:
            steps = np.arange(horizon + 1, dtype=float)
            v_min = np.clip(
                x0_v + steps * self.config.dt * self.config.a_min,
                self.config.v_min,
                self.config.v_max,
            )
            v_max = np.clip(
                x0_v + steps * self.config.dt * self.config.a_max,
                self.config.v_min,
                self.config.v_max,
            )
            v_min[0] = x0_v
            v_max[0] = x0_v
            v_min = np.tile(v_min.reshape((1, horizon + 1)), (num_modes, 1))
            v_max = np.tile(v_max.reshape((1, horizon + 1)), (num_modes, 1))
            return v_min, v_max

        centers = np.full((num_modes, horizon + 1), x0_v, dtype=float)
        if self.last_solution is not None and self.last_solution.status == "optimal":
            previous = np.asarray(self.last_solution.x_nominal, dtype=float)
            if previous.shape == (num_modes, horizon + 1, NX):
                centers = previous[:, :, 1].copy()
        centers[:, 0] = x0_v
        half_width = float(self.config.brake_distance_hard_band_width)
        v_min = np.maximum(self.config.v_min, centers - half_width)
        v_max = np.minimum(self.config.v_max, centers + half_width)
        v_min[:, 0] = x0_v
        v_max[:, 0] = x0_v
        return v_min, v_max

    def _brake_distance_bound_data(self, x0, num_modes):
        v_min, v_max = self._brake_speed_intervals(x0, num_modes)
        slope, intercept = brake_distance_upper_bound_coefficients(
            v_min,
            v_max,
            self.config,
        )
        return v_min, v_max, slope, intercept

    def _with_linearization_input(self, reference, prediction):
        prev_u = self._shifted_previous_input(prediction.num_modes, self.config.horizon)
        if prev_u is None:
            base = np.asarray(reference.a_ref[0], dtype=float).reshape((1, self.config.horizon))
            prev_u = np.tile(base, (prediction.num_modes, 1))
        return ACCReference(
            s_ref=reference.s_ref,
            v_ref=reference.v_ref,
            a_ref=reference.a_ref,
            prev_u=np.clip(prev_u, self.config.a_min, self.config.a_max),
        )

    def _shifted_previous_input(self, num_modes, horizon):
        if (
                self.last_solution is None
                or not self.last_solution.feasible
                or self.last_solution.status != "optimal"):
            return None
        previous = np.asarray(self.last_solution.u_nominal, dtype=float)
        if previous.ndim != 2 or previous.shape[1] != horizon:
            return None
        base = previous[0]
        shifted = np.empty(horizon)
        if horizon > 1:
            shifted[:-1] = base[1:]
        shifted[-1] = base[-1]
        return np.tile(shifted.reshape((1, horizon)), (num_modes, 1))

    def apply_policy(
            self,
            policy,
            mode,
            step,
            reference_acceleration,
            lead_realization,
            lead_mean,
            ego_disturbances=None):
        tv_deviation = np.asarray(lead_realization, dtype=float) - np.asarray(lead_mean, dtype=float)
        accel = float(reference_acceleration)
        accel += policy.acceleration_deviation(mode, step, tv_deviation, ego_disturbances)
        return float(np.clip(accel, self.config.a_min, self.config.a_max))

    def _trim_prediction(self, prediction):
        if prediction.horizon == self.config.horizon:
            return prediction
        if prediction.horizon < self.config.horizon:
            raise ValueError("lead prediction horizon is shorter than controller horizon")
        means = prediction.means[:, :self.config.horizon + 1, :]
        cov = prediction.covariances[:, :self.config.horizon + 1, :, :]
        T = prediction.T[:, :self.config.horizon, :, :] if prediction.T is not None else None
        c = prediction.c[:, :self.config.horizon, :] if prediction.c is not None else None
        policy_tree = (
            prediction.policy_tree.trimmed(self.config.horizon)
            if prediction.policy_tree is not None
            else None
        )
        k_group_map = (
            prediction.k_group_map[:, :self.config.horizon]
            if prediction.k_group_map is not None
            else None
        )
        return MultimodalLeadPrediction(
            means=means,
            probabilities=prediction.probabilities,
            covariances=cov,
            T=T,
            c=c,
            mode_names=prediction.mode_names,
            policy_tree=policy_tree,
            active_mask=prediction.active_mask[:, :self.config.horizon + 1],
            clearance_scale=prediction.clearance_scale[:, :self.config.horizon + 1],
            chance_confidence=prediction.chance_confidence[:, :self.config.horizon + 1],
            k_group_map=k_group_map,
            k_group_names=prediction.k_group_names,
        )

    def _solve_open_loop(self, ego_state, prediction, reference, risks, policy0):
        policy0.K[:] = 0.0
        policy0.M[:] = 0.0
        return self._solve_feedback(ego_state, prediction, reference, risks, policy0, open_loop=True)

    def _solve_feedback(self, ego_state, prediction, reference, risks, policy0, open_loop=False):
        solve_start = time.perf_counter()
        timing_breakdown = {}
        x0 = np.asarray(ego_state, dtype=float)
        phase_start = time.perf_counter()
        layout = self._policy_variable_layout(prediction, open_loop=open_loop)
        timing_breakdown["policy_layout_s"] = time.perf_counter() - phase_start
        solve_path = "fallback"

        try:
            if self._can_use_cached_open_loop_fixed_qp(open_loop, prediction):
                if self.config.safety_constraint_mode == SAFETY_NOMINAL_SAFE_DISTANCE:
                    solve_path = "cached_open_loop_safe_distance_qp"
                elif self.config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
                    solve_path = "cached_open_loop_brake_distance_qp"
                elif self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
                    solve_path = "cached_open_loop_confidence_chance_qp"
                else:
                    solve_path = "cached_open_loop_fixed_qp"
                policy, slack, cost, status, message, layout, optimized_eta_data = (
                    self._solve_open_loop_fixed_qp_cached(
                        x0, prediction, reference, risks, policy0, layout
                    )
                )
            elif self._can_use_cached_feedback_scalar_chance_qp(open_loop, prediction):
                if self.config.safety_constraint_mode == SAFETY_NOMINAL_SAFE_DISTANCE:
                    solve_path = "cached_feedback_safe_distance_qp"
                elif self.config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
                    solve_path = "cached_feedback_brake_distance_qp"
                elif self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
                    solve_path = "cached_feedback_confidence_chance_qp"
                else:
                    solve_path = "cached_feedback_scalar_chance_qp"
                policy, slack, cost, status, message, layout, optimized_eta_data = (
                    self._solve_feedback_scalar_chance_qp_cached(
                        x0, prediction, reference, risks, policy0, layout
                    )
                )
            else:
                solve_path = "generic_qp_gurobi"
                phase_start = time.perf_counter()
                samples = self._build_samples(prediction)
                timing_breakdown["sample_build_s"] = time.perf_counter() - phase_start
                policy, slack, cost, status, message, layout, optimized_eta_data = self._solve_qp_gurobi(
                    x0, prediction, reference, risks, policy0, samples, open_loop, layout
                )
        except Exception as exc:
            if not self.config.allow_solver_fallback:
                raise
            solve_path = "fallback"
            phase_start = time.perf_counter()
            policy, slack = self._fallback_policy(x0, prediction, reference)
            timing_breakdown["fallback_policy_s"] = time.perf_counter() - phase_start
            status = "fallback"
            message = "gurobi_failed: %r" % (exc,)
            cost = float("nan")
            optimized_eta_data = None
        timing_breakdown.update(self._timings_from_solver_message(message))

        phase_start = time.perf_counter()
        x_nominal = np.zeros((prediction.num_modes, self.config.horizon + 1, NX))
        u_nominal = np.zeros((prediction.num_modes, self.config.horizon))
        safety_values = np.zeros((prediction.num_modes, self.config.horizon + 1))
        tightening_values = np.zeros((prediction.num_modes, self.config.horizon + 1))
        chance_margin_values = np.zeros((prediction.num_modes, self.config.horizon + 1))
        eta_levels = None
        safe_probability_levels = None
        optimized_risk_levels = None
        target_safe_probability = 0.0
        risk_diagnostics = risks
        if optimized_eta_data is not None:
            eta_levels = np.asarray(optimized_eta_data["eta_levels"], dtype=float).reshape(-1)
            safe_probability_levels = np.asarray(
                optimized_eta_data["safe_probability_levels"], dtype=float
            ).reshape(-1)
            target_safe_probability = float(optimized_eta_data["target_safe_probability"])
            optimized_risk_levels = 1.0 - safe_probability_levels
            risk_diagnostics = np.tile(
                optimized_risk_levels.reshape((-1, 1)),
                (1, self.config.horizon + 1),
            )
        for mode in range(prediction.num_modes):
            x_nominal[mode], u_nominal[mode] = self._rollout_policy(
                x0,
                mode,
                policy,
                reference,
                prediction,
                prediction.means[mode],
                np.zeros((self.config.horizon, NX)),
            )
            for step in range(self.config.horizon + 1):
                if prediction.active_mask[mode, step]:
                    safety_values[mode, step] = safety_function(
                        x_nominal[mode, step],
                        prediction.means[mode, step],
                        self.config,
                        linearization_speed=x0[1],
                        clearance_scale=float(prediction.clearance_scale[mode, step]),
                    )
                    if eta_levels is not None:
                        tightening_values[mode, step] = (
                            eta_levels[mode] * safety_std(prediction.covariances[mode, step], self.config)
                        )
                    else:
                        tightening_values[mode, step] = self._cell_tightening(
                            prediction, risks, mode, step
                        )
                    chance_margin_values[mode, step] = (
                        safety_values[mode, step] - tightening_values[mode, step]
                    )
                else:
                    safety_values[mode, step] = np.inf
                    tightening_values[mode, step] = 0.0
                    chance_margin_values[mode, step] = np.inf
        timing_breakdown["solution_postprocess_s"] = time.perf_counter() - phase_start

        first_accel = reference.prev_u[0, 0] + policy.h[0, 0, 0]
        action = float(np.clip(first_accel, self.config.a_min, self.config.a_max))

        if self.config.safety_constraint_mode in CHANCE_SAFETY_MODES:
            feasible = bool(np.all(
                chance_margin_values[prediction.active_mask]
                + slack[prediction.active_mask]
                + 1.0e-7 >= 0.0
            ))
        else:
            feasible = bool(np.all(safety_values[prediction.active_mask] + 1.0e-7 >= 0.0))
        timing_breakdown["feedback_total_s"] = time.perf_counter() - solve_start
        return NairACCSolution(
            action=action,
            status=status,
            feasible=feasible,
            cost=cost,
            policy=policy,
            reference=reference,
            x_nominal=x_nominal,
            u_nominal=u_nominal,
            safety_values=safety_values,
            risk_levels=risk_diagnostics,
            solver_message=message,
            num_decision_variables=layout.num_decision_variables,
            baseline_decision_variables=layout.baseline_decision_variables,
            policy_split_steps=layout.policy_split_steps,
            first_policy_split_step=layout.first_policy_split_step,
            policy_group_count=layout.policy_group_count,
            safety_constraint_mode=self.config.safety_constraint_mode,
            risk_allocation_mode=self.config.risk_allocation_mode,
            risk_min=float(np.min(risk_diagnostics)),
            risk_max=float(np.max(risk_diagnostics)),
            tightening_min=float(np.min(tightening_values)),
            tightening_max=float(np.max(tightening_values)),
            chance_margin_min=(
                float(np.min(chance_margin_values[prediction.active_mask]))
                if np.any(prediction.active_mask) else float("inf")
            ),
            eta_levels=eta_levels,
            safe_probability_levels=safe_probability_levels,
            optimized_risk_levels=optimized_risk_levels,
            target_safe_probability=target_safe_probability,
            eta_min=float(np.min(eta_levels)) if eta_levels is not None else 0.0,
            eta_max=float(np.max(eta_levels)) if eta_levels is not None else 0.0,
            safe_probability_min=(
                float(np.min(safe_probability_levels)) if safe_probability_levels is not None else 0.0
            ),
            safe_probability_max=(
                float(np.max(safe_probability_levels)) if safe_probability_levels is not None else 0.0
            ),
            optimized_risk_min=(
                float(np.min(optimized_risk_levels)) if optimized_risk_levels is not None else 0.0
            ),
            optimized_risk_max=(
                float(np.max(optimized_risk_levels)) if optimized_risk_levels is not None else 0.0
            ),
            solve_path=solve_path,
            timing_breakdown=timing_breakdown,
        )

    @staticmethod
    def _timings_from_solver_message(message):
        timings = {}
        for token in str(message).split(";"):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "t_wall_solver" or key.startswith("diag_"):
                try:
                    timings[key] = float(value)
                except ValueError:
                    continue
        return timings

    def _symbolic_safety_margin(
            self,
            lead_s,
            ego_s,
            ego_v,
            brake_slope=0.0,
            brake_intercept=0.0,
            tightening=0.0,
            slack=0.0,
            clearance_scale=1.0):
        if self.config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
            d_brake = brake_slope * ego_v + brake_intercept
            clearance = self.config.d0 + d_brake
        else:
            clearance = self.config.d0 + self.config.time_headway * ego_v
        # ``clearance_scale`` multiplies the whole deterministic standoff, footprint
        # included.  Benciolini et al. (T-IV 2023) Remark 5: leaving the footprint
        # outside the confidence scaling keeps a fixed exclusion zone alive even for
        # a disbelieved mode.  ``tightening`` stays outside because the risk
        # allocation (or the cell's confidence) already carries the mode
        # probability.
        return (
            lead_s
            - ego_s
            - clearance_scale * (self.config.vehicle_length + clearance)
            - tightening
            + slack
        )

    def _symbolic_min_clearance_margin(self, lead_s, ego_s, slack=0.0, clearance_scale=1.0):
        return (
            lead_s
            - ego_s
            - clearance_scale * (self.config.vehicle_length + self.config.d0)
            + slack
        )

    def _add_symbolic_safety_constraints(
            self,
            opti,
            lead_s,
            ego_s,
            ego_v,
            brake_slope=0.0,
            brake_intercept=0.0,
            brake_v_min=None,
            brake_v_max=None,
            enforce_brake_speed_band=True,
            tightening=0.0,
            slack=0.0,
            clearance_scale=1.0):
        opti.subject_to(
            self._symbolic_safety_margin(
                lead_s,
                ego_s,
                ego_v,
                brake_slope=brake_slope,
                brake_intercept=brake_intercept,
                tightening=tightening,
                slack=slack,
                clearance_scale=clearance_scale,
            )
            >= 0.0
        )
        if self.config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
            opti.subject_to(
                self._symbolic_min_clearance_margin(
                    lead_s, ego_s, slack=slack, clearance_scale=clearance_scale
                )
                >= 0.0
            )
            if (
                    self.config.brake_distance_bound_mode == BRAKE_DISTANCE_BOUND_HARD_BAND
                    and enforce_brake_speed_band):
                opti.subject_to(ego_v >= brake_v_min)
                opti.subject_to(ego_v <= brake_v_max)

    def _cell_tightening(self, prediction, risks, mode, step):
        """Numeric safety-margin tightening for one (mode, step) cell.

        ``scalar_chance`` draws it from the risk allocation, ``confidence_chance``
        from the cell's Benciolini confidence, and the deterministic modes use
        none.
        """
        if self.config.safety_constraint_mode == SAFETY_SCALAR_CHANCE:
            return chance_tightening(
                prediction.covariances[mode, step], risks[mode, step], self.config
            )
        if self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
            return confidence_tightening(
                prediction.covariances[mode, step],
                prediction.chance_confidence[mode, step],
                self.config,
            )
        return 0.0

    def _can_use_cached_open_loop_fixed_qp(self, open_loop, prediction):
        if not open_loop:
            return False
        if self.config.safety_constraint_mode in CHANCE_SAFETY_MODES:
            return self.config.risk_allocation_mode == RISK_FIXED
        if self.config.safety_constraint_mode in (
                SAFETY_NOMINAL_SAFE_DISTANCE,
                SAFETY_BRAKE_DISTANCE):
            return self.config.risk_allocation_mode != RISK_OPTIMIZED_ETA
        return False

    def _cached_safety_lead_s(self, x0, prediction):
        lead_s = np.asarray(prediction.means[:, :, 0], dtype=float).copy()
        inactive = ~np.asarray(prediction.active_mask, dtype=bool)
        if np.any(inactive):
            lead_s[inactive] = float(np.asarray(x0, dtype=float).reshape(-1)[0] + 1.0e6)
        return lead_s

    def _open_loop_fixed_qp_key(self, num_modes):
        return (
            int(num_modes),
            int(self.config.horizon),
            self.config.safety_constraint_mode,
            float(self.config.dt),
            float(self.config.a_min),
            float(self.config.a_max),
            float(self.config.v_min),
            float(self.config.v_max),
            None if self.config.jerk_limit is None else float(self.config.jerk_limit),
            float(self.config.vehicle_length),
            float(self.config.d0),
            float(self.config.time_headway),
            self.config.brake_distance_bound_mode,
            float(self.config.brake_distance_hard_band_width),
            float(self.config.q_s),
            float(self.config.q_v),
            float(self.config.r_a),
            float(self.config.r_jerk),
            float(self.config.slack_weight),
            self.config.solver_name,
            float(self.config.optimizer_ftol),
            None if self.config.gurobi_time_limit is None else float(self.config.gurobi_time_limit),
        )

    def _solve_open_loop_fixed_qp_cached(self, x0, prediction, reference, risks, policy0, layout):
        phase_start = time.perf_counter()
        problem = self._get_open_loop_fixed_qp(prediction.num_modes)
        get_problem_s = time.perf_counter() - phase_start
        horizon = self.config.horizon
        num_modes = prediction.num_modes

        phase_start = time.perf_counter()
        tightening = np.zeros((num_modes, horizon + 1))
        if self.config.safety_constraint_mode in CHANCE_SAFETY_MODES:
            for mode in range(num_modes):
                for step in range(horizon + 1):
                    if not prediction.active_mask[mode, step]:
                        continue
                    tightening[mode, step] = self._cell_tightening(prediction, risks, mode, step)
        tightening_build_s = time.perf_counter() - phase_start

        opti = problem.opti
        phase_start = time.perf_counter()
        previous_accel_cmd, command_dt = self._command_context()
        brake_v_min, brake_v_max, brake_slope, brake_intercept = (
            self._brake_distance_bound_data(x0, num_modes)
        )
        opti.set_value(problem.x0_param, np.asarray(x0, dtype=float).reshape((NX, 1)))
        opti.set_value(problem.prev_u_param, np.asarray(reference.prev_u[0], dtype=float).reshape((horizon, 1)))
        opti.set_value(problem.previous_accel_param, previous_accel_cmd)
        opti.set_value(problem.command_dt_param, command_dt)
        opti.set_value(problem.lead_s_param, self._cached_safety_lead_s(x0, prediction))
        opti.set_value(problem.lead_v_param, prediction.means[:, :, 1])
        opti.set_value(problem.x_ref_s_param, reference.x_ref[:, :, 0])
        opti.set_value(problem.x_ref_v_param, reference.x_ref[:, :, 1])
        opti.set_value(problem.a_ref_param, reference.a_ref)
        opti.set_value(problem.probability_param, prediction.probabilities.reshape((num_modes, 1)))
        opti.set_value(problem.tightening_param, tightening)
        opti.set_value(problem.clearance_scale_param, prediction.clearance_scale)
        opti.set_value(problem.brake_slope_param, brake_slope)
        opti.set_value(problem.brake_intercept_param, brake_intercept)
        opti.set_value(problem.brake_v_min_param, brake_v_min)
        opti.set_value(problem.brake_v_max_param, brake_v_max)
        set_value_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        opti.set_initial(problem.h_var, np.asarray(policy0.h[0, :, 0], dtype=float).reshape((horizon, 1)))
        opti.set_initial(problem.slack_var, 0.0)
        set_initial_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        sol = opti.solve()
        stats = opti.stats()
        opti_solve_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        h_values = np.asarray(sol.value(problem.h_var), dtype=float).reshape(horizon)
        slack_value = float(sol.value(problem.slack_var))
        h = np.tile(h_values.reshape((1, horizon, NU)), (num_modes, 1, 1))
        k = np.zeros((num_modes, horizon, NO))
        m = np.zeros((num_modes, horizon, horizon, NX))
        slack = np.full((num_modes, horizon + 1), slack_value, dtype=float)
        extract_solution_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        opti.set_initial(problem.h_var, h_values.reshape((horizon, 1)))
        opti.set_initial(problem.slack_var, slack_value)
        store_warmstart_s = time.perf_counter() - phase_start

        status = str(stats.get("return_status", "unknown"))
        solver_time = stats.get("t_wall_solver", None)
        if self.config.safety_constraint_mode == SAFETY_NOMINAL_SAFE_DISTANCE:
            cache_marker = "cached_open_loop_safe_distance_qp=1"
        elif self.config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
            cache_marker = "cached_open_loop_brake_distance_qp=1"
        elif self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
            cache_marker = "cached_open_loop_confidence_chance_qp=1"
        else:
            cache_marker = "cached_open_loop_fixed_qp=1"
        if solver_time is None:
            message = "gurobi_return_status=%s;%s" % (status, cache_marker)
        else:
            message = "gurobi_return_status=%s;t_wall_solver=%.6f;%s" % (
                status,
                float(solver_time),
                cache_marker,
            )
        message += (
            ";diag_get_problem=%.6f"
            ";diag_tightening=%.6f"
            ";diag_set_value=%.6f"
            ";diag_set_initial=%.6f"
            ";diag_opti_solve=%.6f"
            ";diag_extract=%.6f"
            ";diag_store_warmstart=%.6f"
        ) % (
            get_problem_s,
            tightening_build_s,
            set_value_s,
            set_initial_s,
            opti_solve_s,
            extract_solution_s,
            store_warmstart_s,
        )
        cost = float(sol.value(problem.objective))
        return NairACCPolicy(h=h, M=m, K=k), slack, cost, "optimal", message, layout, None

    def _get_open_loop_fixed_qp(self, num_modes):
        key = self._open_loop_fixed_qp_key(num_modes)
        cached = self._open_loop_fixed_qp_cache.get(key)
        if cached is not None:
            return cached

        horizon = self.config.horizon
        opti = ca.Opti("conic")
        h_var = opti.variable(horizon)
        slack_var = opti.variable()
        x0_param = opti.parameter(NX)
        prev_u_param = opti.parameter(horizon)
        previous_accel_param = opti.parameter()
        command_dt_param = opti.parameter()
        lead_s_param = opti.parameter(num_modes, horizon + 1)
        lead_v_param = opti.parameter(num_modes, horizon + 1)
        x_ref_s_param = opti.parameter(num_modes, horizon + 1)
        x_ref_v_param = opti.parameter(num_modes, horizon + 1)
        a_ref_param = opti.parameter(num_modes, horizon)
        probability_param = opti.parameter(num_modes)
        tightening_param = opti.parameter(num_modes, horizon + 1)
        clearance_scale_param = opti.parameter(num_modes, horizon + 1)
        brake_slope_param = opti.parameter(num_modes, horizon + 1)
        brake_intercept_param = opti.parameter(num_modes, horizon + 1)
        brake_v_min_param = opti.parameter(num_modes, horizon + 1)
        brake_v_max_param = opti.parameter(num_modes, horizon + 1)

        opti.subject_to(slack_var >= 0.0)

        states_s = [x0_param[0]]
        states_v = [x0_param[1]]
        controls = []
        for step in range(horizon):
            control = prev_u_param[step] + h_var[step]
            controls.append(control)
            opti.subject_to(control >= self.config.a_min)
            opti.subject_to(control <= self.config.a_max)
            if step == 0:
                self._add_symbolic_jerk_constraints(
                    opti,
                    control,
                    previous_accel_param,
                    command_dt_param,
                )
            else:
                self._add_symbolic_jerk_constraints(
                    opti,
                    control,
                    controls[step - 1],
                    self.config.dt,
                )
            next_s = states_s[-1] + self.config.dt * states_v[-1] + 0.5 * self.config.dt ** 2 * control
            next_v = states_v[-1] + self.config.dt * control
            states_s.append(next_s)
            states_v.append(next_v)
            opti.subject_to(next_v >= self.config.v_min)
            opti.subject_to(next_v <= self.config.v_max)

        for mode in range(num_modes):
            for step in range(horizon + 1):
                self._add_symbolic_safety_constraints(
                    opti,
                    lead_s_param[mode, step],
                    states_s[step],
                    states_v[step],
                    brake_slope=brake_slope_param[mode, step],
                    brake_intercept=brake_intercept_param[mode, step],
                    brake_v_min=brake_v_min_param[mode, step],
                    brake_v_max=brake_v_max_param[mode, step],
                    enforce_brake_speed_band=step > 0,
                    tightening=tightening_param[mode, step],
                    slack=slack_var,
                    clearance_scale=clearance_scale_param[mode, step],
                )

        objective = self.config.slack_weight * slack_var ** 2
        sample_count = 3.0
        for mode in range(num_modes):
            probability = probability_param[mode]
            for step in range(horizon + 1):
                s_error = states_s[step] - x_ref_s_param[mode, step]
                v_error = states_v[step] - x_ref_v_param[mode, step]
                objective += sample_count * probability * (
                    self.config.q_s * s_error ** 2 + self.config.q_v * v_error ** 2
                )

            previous_input = previous_accel_param
            for step in range(horizon):
                control = controls[step]
                input_error = control - a_ref_param[mode, step]
                jerk = (control - previous_input) / self.config.dt
                objective += sample_count * probability * (
                    self.config.r_a * input_error ** 2 + self.config.r_jerk * jerk ** 2
                )
                previous_input = control

        opti.minimize(objective)
        p_opts, s_opts = self._gurobi_options()
        opti.solver(self.config.solver_name, p_opts, s_opts)

        problem = CachedOpenLoopFixedQP(
            opti=opti,
            h_var=h_var,
            slack_var=slack_var,
            x0_param=x0_param,
            prev_u_param=prev_u_param,
            previous_accel_param=previous_accel_param,
            command_dt_param=command_dt_param,
            lead_s_param=lead_s_param,
            lead_v_param=lead_v_param,
            x_ref_s_param=x_ref_s_param,
            x_ref_v_param=x_ref_v_param,
            a_ref_param=a_ref_param,
            probability_param=probability_param,
            tightening_param=tightening_param,
            clearance_scale_param=clearance_scale_param,
            brake_slope_param=brake_slope_param,
            brake_intercept_param=brake_intercept_param,
            brake_v_min_param=brake_v_min_param,
            brake_v_max_param=brake_v_max_param,
            objective=objective,
        )
        self._open_loop_fixed_qp_cache[key] = problem
        return problem

    def _can_use_cached_feedback_scalar_chance_qp(self, open_loop, prediction):
        if open_loop:
            return False
        if self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
            return self.config.risk_allocation_mode == RISK_FIXED
        if self.config.safety_constraint_mode == SAFETY_SCALAR_CHANCE:
            return self.config.risk_allocation_mode in (RISK_FIXED, RISK_OPTIMIZED_ETA)
        if self.config.safety_constraint_mode in (
                SAFETY_NOMINAL_SAFE_DISTANCE,
                SAFETY_BRAKE_DISTANCE):
            return self.config.risk_allocation_mode != RISK_OPTIMIZED_ETA
        return False

    def _feedback_scalar_chance_qp_key(self, num_modes):
        return (
            int(num_modes),
            int(self.config.horizon),
            self.config.safety_constraint_mode,
            float(self.config.dt),
            float(self.config.a_min),
            float(self.config.a_max),
            float(self.config.v_min),
            float(self.config.v_max),
            None if self.config.jerk_limit is None else float(self.config.jerk_limit),
            float(self.config.vehicle_length),
            float(self.config.d0),
            float(self.config.time_headway),
            self.config.brake_distance_bound_mode,
            float(self.config.brake_distance_hard_band_width),
            float(self.config.q_s),
            float(self.config.q_v),
            float(self.config.r_a),
            float(self.config.r_jerk),
            float(self.config.slack_weight),
            float(self.config.feedback_bound),
            float(self.config.disturbance_feedback_bound),
            bool(self.config.optimize_k),
            self.config.risk_allocation_mode,
            self.config.solver_name,
            float(self.config.optimizer_ftol),
            None if self.config.gurobi_time_limit is None else float(self.config.gurobi_time_limit),
        )

    def _solve_feedback_scalar_chance_qp_cached(self, x0, prediction, reference, risks, policy0, layout):
        phase_start = time.perf_counter()
        problem = self._get_feedback_scalar_chance_qp(prediction.num_modes)
        get_problem_s = time.perf_counter() - phase_start
        horizon = self.config.horizon
        num_modes = prediction.num_modes

        phase_start = time.perf_counter()
        tightening = np.zeros((num_modes, horizon + 1))
        safety_stds = np.zeros((num_modes, horizon + 1))
        if self.config.safety_constraint_mode in CHANCE_SAFETY_MODES:
            for mode in range(num_modes):
                for step in range(horizon + 1):
                    if not prediction.active_mask[mode, step]:
                        continue
                    safety_stds[mode, step] = safety_std(prediction.covariances[mode, step], self.config)
                    tightening[mode, step] = self._cell_tightening(prediction, risks, mode, step)

        std = np.sqrt(np.maximum(np.diagonal(prediction.covariances, axis1=2, axis2=3), 0.0))
        lead_dev_s = [
            np.zeros((num_modes, horizon)),
            -std[:, :horizon, 0],
            std[:, :horizon, 0],
        ]
        lead_dev_v = [
            np.zeros((num_modes, horizon)),
            -std[:, :horizon, 1],
            std[:, :horizon, 1],
        ]
        precompute_s = time.perf_counter() - phase_start

        opti = problem.opti
        phase_start = time.perf_counter()
        previous_accel_cmd, command_dt = self._command_context()
        brake_v_min, brake_v_max, brake_slope, brake_intercept = (
            self._brake_distance_bound_data(x0, num_modes)
        )
        opti.set_value(problem.x0_param, np.asarray(x0, dtype=float).reshape((NX, 1)))
        opti.set_value(problem.prev_u_param, np.asarray(reference.prev_u, dtype=float))
        opti.set_value(problem.previous_accel_param, previous_accel_cmd)
        opti.set_value(problem.command_dt_param, command_dt)
        opti.set_value(problem.lead_s_param, self._cached_safety_lead_s(x0, prediction))
        opti.set_value(problem.lead_v_param, prediction.means[:, :, 1])
        opti.set_value(problem.x_ref_s_param, reference.x_ref[:, :, 0])
        opti.set_value(problem.x_ref_v_param, reference.x_ref[:, :, 1])
        opti.set_value(problem.a_ref_param, reference.a_ref)
        opti.set_value(problem.probability_param, prediction.probabilities.reshape((num_modes, 1)))
        opti.set_value(problem.tightening_param, tightening)
        opti.set_value(problem.clearance_scale_param, prediction.clearance_scale)
        opti.set_value(problem.safety_std_param, safety_stds)
        opti.set_value(problem.brake_slope_param, brake_slope)
        opti.set_value(problem.brake_intercept_param, brake_intercept)
        opti.set_value(problem.brake_v_min_param, brake_v_min)
        opti.set_value(problem.brake_v_max_param, brake_v_max)
        for sample_idx in range(3):
            opti.set_value(problem.lead_dev_s_params[sample_idx], lead_dev_s[sample_idx])
            opti.set_value(problem.lead_dev_v_params[sample_idx], lead_dev_v[sample_idx])
        k_slot_by_step = []
        if problem.optimize_k:
            for step in range(horizon):
                step_groups = sorted(set(int(group) for group in layout.k_group_map[:, step]))
                if len(step_groups) > problem.k_group_capacity:
                    raise ValueError("too many K groups at one step for cached QP")
                slot_by_group = {group_id: slot for slot, group_id in enumerate(step_groups)}
                k_slot_by_step.append(slot_by_group)
                selector = np.zeros((num_modes, problem.k_group_capacity), dtype=float)
                for mode in range(num_modes):
                    selector[mode, slot_by_group[int(layout.k_group_map[mode, step])]] = 1.0
                opti.set_value(problem.k_select_params[step], selector)
        else:
            k_slot_by_step = [{} for _ in range(horizon)]
        for mode_a, mode_b, share_param in problem.sharing_params:
            share_mask = np.zeros(horizon)
            for step in range(horizon):
                if int(layout.group_map[mode_a, step]) == int(layout.group_map[mode_b, step]):
                    share_mask[step] = 1.0
            opti.set_value(share_param, share_mask)
        set_value_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        h_initial = np.asarray(policy0.h[:, :, 0], dtype=float).copy()
        k_initial = np.asarray(policy0.K, dtype=float).copy()
        for group_id in range(layout.policy_group_count):
            rep_mode, rep_step = self._representative_mode_step(layout, group_id)
            locations = np.argwhere(layout.group_map == group_id)
            for mode, step in locations:
                h_initial[int(mode), int(step)] = h_initial[rep_mode, rep_step]
                if problem.optimize_k:
                    k_initial[int(mode), int(step)] = k_initial[rep_mode, rep_step]

        opti.set_initial(problem.h_var, h_initial)
        opti.set_initial(problem.slack_var, np.zeros((num_modes, horizon + 1)))
        if problem.optimize_k:
            for slot in range(problem.k_group_capacity):
                for step in range(horizon):
                    groups_for_slot = [
                        group_id for group_id, candidate_slot in k_slot_by_step[step].items()
                        if candidate_slot == slot
                    ]
                    if groups_for_slot:
                        group_id = groups_for_slot[0]
                        locations = np.argwhere(layout.k_group_map[:, step] == group_id)
                        rep_mode = int(locations[0, 0])
                        opti.set_initial(problem.k_var[slot][step], k_initial[rep_mode, step])
                    else:
                        opti.set_initial(problem.k_var[slot][step], np.zeros(NO))
        if problem.optimized_eta:
            initial_eta = np.full(
                (num_modes, 1),
                np.clip(1.64, self.config.eta_std_min, self.config.eta_std_max),
            )
            initial_rho = np.full(
                (num_modes, 1),
                min(1.0, optimized_eta_target_probability(self.config)),
            )
            opti.set_initial(problem.eta_var, initial_eta)
            opti.set_initial(problem.rho_var, initial_rho)
        set_initial_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        sol = opti.solve()
        stats = opti.stats()
        opti_solve_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        h_values = np.asarray(sol.value(problem.h_var), dtype=float).reshape((num_modes, horizon))
        h = h_values[:, :, None]
        k = np.zeros((num_modes, horizon, NO))
        m = np.zeros((num_modes, horizon, horizon, NX))
        if problem.optimize_k:
            for mode in range(num_modes):
                for step in range(horizon):
                    group_id = int(layout.k_group_map[mode, step])
                    slot = k_slot_by_step[step][group_id]
                    k[mode, step] = np.asarray(sol.value(problem.k_var[slot][step]), dtype=float).reshape(NO)
        slack = np.asarray(sol.value(problem.slack_var), dtype=float).reshape((num_modes, horizon + 1))

        opti.set_initial(problem.h_var, h_values)
        opti.set_initial(problem.slack_var, slack)

        optimized_eta_data = None
        if problem.optimized_eta:
            eta_levels = np.asarray(sol.value(problem.eta_var), dtype=float).reshape(num_modes)
            safe_probability_levels = np.asarray(sol.value(problem.rho_var), dtype=float).reshape(num_modes)
            optimized_eta_data = {
                "eta_levels": eta_levels,
                "safe_probability_levels": safe_probability_levels,
                "target_safe_probability": optimized_eta_target_probability(self.config),
            }
        extract_solution_s = time.perf_counter() - phase_start

        status = str(stats.get("return_status", "unknown"))
        solver_time = stats.get("t_wall_solver", None)
        if self.config.safety_constraint_mode == SAFETY_NOMINAL_SAFE_DISTANCE:
            cache_marker = "cached_feedback_safe_distance_qp=1"
        elif self.config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
            cache_marker = "cached_feedback_brake_distance_qp=1"
        elif self.config.safety_constraint_mode == SAFETY_CONFIDENCE_CHANCE:
            cache_marker = "cached_feedback_confidence_chance_qp=1"
        else:
            cache_marker = "cached_feedback_scalar_chance_qp=1"
        if solver_time is None:
            message = "gurobi_return_status=%s;%s" % (status, cache_marker)
        else:
            message = "gurobi_return_status=%s;t_wall_solver=%.6f;%s" % (
                status,
                float(solver_time),
                cache_marker,
            )
        cost = float(sol.value(problem.objective))
        message += (
            ";diag_get_problem=%.6f"
            ";diag_precompute=%.6f"
            ";diag_set_value=%.6f"
            ";diag_set_initial=%.6f"
            ";diag_opti_solve=%.6f"
            ";diag_extract=%.6f"
        ) % (
            get_problem_s,
            precompute_s,
            set_value_s,
            set_initial_s,
            opti_solve_s,
            extract_solution_s,
        )
        return NairACCPolicy(h=h, M=m, K=k), slack, cost, "optimal", message, layout, optimized_eta_data

    def _get_feedback_scalar_chance_qp(self, num_modes):
        key = self._feedback_scalar_chance_qp_key(num_modes)
        cached = self._feedback_scalar_chance_qp_cache.get(key)
        if cached is not None:
            return cached

        horizon = self.config.horizon
        optimized_eta = self.config.risk_allocation_mode == RISK_OPTIMIZED_ETA
        optimize_k = bool(self.config.optimize_k)
        k_group_capacity = int(num_modes) if optimize_k else 0
        opti = ca.Opti("conic")
        h_var = opti.variable(num_modes, horizon)
        k_var = (
            [[opti.variable(NO) for _ in range(horizon)] for _ in range(k_group_capacity)]
            if optimize_k
            else None
        )
        k_select_params = (
            [opti.parameter(num_modes, k_group_capacity) for _ in range(horizon)]
            if optimize_k
            else []
        )
        slack_var = opti.variable(num_modes, horizon + 1)
        eta_var = opti.variable(num_modes) if optimized_eta else None
        rho_var = opti.variable(num_modes) if optimized_eta else None
        x0_param = opti.parameter(NX)
        prev_u_param = opti.parameter(num_modes, horizon)
        previous_accel_param = opti.parameter()
        command_dt_param = opti.parameter()
        lead_s_param = opti.parameter(num_modes, horizon + 1)
        lead_v_param = opti.parameter(num_modes, horizon + 1)
        x_ref_s_param = opti.parameter(num_modes, horizon + 1)
        x_ref_v_param = opti.parameter(num_modes, horizon + 1)
        a_ref_param = opti.parameter(num_modes, horizon)
        probability_param = opti.parameter(num_modes)
        tightening_param = opti.parameter(num_modes, horizon + 1)
        clearance_scale_param = opti.parameter(num_modes, horizon + 1)
        safety_std_param = opti.parameter(num_modes, horizon + 1)
        brake_slope_param = opti.parameter(num_modes, horizon + 1)
        brake_intercept_param = opti.parameter(num_modes, horizon + 1)
        brake_v_min_param = opti.parameter(num_modes, horizon + 1)
        brake_v_max_param = opti.parameter(num_modes, horizon + 1)
        lead_dev_s_params = [opti.parameter(num_modes, horizon) for _ in range(3)]
        lead_dev_v_params = [opti.parameter(num_modes, horizon) for _ in range(3)]
        sharing_params = []

        opti.subject_to(ca.vec(slack_var) >= 0.0)
        if optimize_k:
            for group_id in range(k_group_capacity):
                for step in range(horizon):
                    opti.subject_to(k_var[group_id][step] <= self.config.feedback_bound)
                    opti.subject_to(k_var[group_id][step] >= -self.config.feedback_bound)

        for mode in range(num_modes):
            for step in range(horizon):
                opti.subject_to(h_var[mode, step] >= self.config.a_min - prev_u_param[mode, step])
                opti.subject_to(h_var[mode, step] <= self.config.a_max - prev_u_param[mode, step])

        for mode_a in range(num_modes):
            for mode_b in range(mode_a + 1, num_modes):
                share_param = opti.parameter(horizon)
                sharing_params.append((mode_a, mode_b, share_param))
                for step in range(horizon):
                    share = share_param[step]
                    opti.subject_to(share * (h_var[mode_a, step] - h_var[mode_b, step]) == 0.0)

        first_input = prev_u_param[0, 0] + h_var[0, 0]
        for mode in range(1, num_modes):
            opti.subject_to(prev_u_param[mode, 0] + h_var[mode, 0] == first_input)

        if optimized_eta:
            opti.subject_to(eta_var >= self.config.eta_std_min)
            opti.subject_to(eta_var <= self.config.eta_std_max)
            opti.subject_to(rho_var >= 0.5)
            opti.subject_to(rho_var <= 1.0)
            for slope, intercept in cdf_lower_bound_lines(self.config):
                opti.subject_to(rho_var <= slope * eta_var + intercept)
            total_safe_probability = 0.0
            for mode in range(num_modes):
                total_safe_probability += probability_param[mode] * rho_var[mode]
            opti.subject_to(total_safe_probability >= optimized_eta_target_probability(self.config))

        objective = self.config.slack_weight * ca.sumsqr(slack_var)
        mean_states_by_mode = []
        for mode in range(num_modes):
            sample_states = []
            for sample_idx in range(3):
                state_s = x0_param[0]
                state_v = x0_param[1]
                states_s = [state_s]
                states_v = [state_v]
                inputs = []
                for step in range(horizon):
                    delta_u = h_var[mode, step]
                    if optimize_k:
                        k_feedback = ca.DM.zeros(NO, 1)
                        for group_id in range(k_group_capacity):
                            k_feedback += k_select_params[step][mode, group_id] * k_var[group_id][step]
                        delta_u += k_feedback[0] * lead_dev_s_params[sample_idx][mode, step]
                        delta_u += k_feedback[1] * lead_dev_v_params[sample_idx][mode, step]
                    control = prev_u_param[mode, step] + delta_u
                    inputs.append(control)
                    opti.subject_to(control >= self.config.a_min)
                    opti.subject_to(control <= self.config.a_max)
                    if step == 0:
                        self._add_symbolic_jerk_constraints(
                            opti,
                            control,
                            previous_accel_param,
                            command_dt_param,
                        )
                    else:
                        self._add_symbolic_jerk_constraints(
                            opti,
                            control,
                            inputs[step - 1],
                            self.config.dt,
                        )
                    state_s = state_s + self.config.dt * state_v + 0.5 * self.config.dt ** 2 * control
                    state_v = state_v + self.config.dt * control
                    states_s.append(state_s)
                    states_v.append(state_v)
                    opti.subject_to(state_v >= self.config.v_min)
                    opti.subject_to(state_v <= self.config.v_max)
                sample_states.append((states_s, states_v, inputs))

                previous_input = previous_accel_param
                for step in range(horizon + 1):
                    s_error = states_s[step] - x_ref_s_param[mode, step]
                    v_error = states_v[step] - x_ref_v_param[mode, step]
                    objective += probability_param[mode] * (
                        self.config.q_s * s_error ** 2 + self.config.q_v * v_error ** 2
                    )
                    if step < horizon:
                        control = inputs[step]
                        input_error = control - a_ref_param[mode, step]
                        jerk = (control - previous_input) / self.config.dt
                        objective += probability_param[mode] * (
                            self.config.r_a * input_error ** 2 + self.config.r_jerk * jerk ** 2
                        )
                        previous_input = control
            mean_states_by_mode.append(sample_states[0])

        for mode in range(num_modes):
            states_s, states_v, _ = mean_states_by_mode[mode]
            for step in range(horizon + 1):
                if optimized_eta:
                    tightening = safety_std_param[mode, step] * eta_var[mode]
                else:
                    tightening = tightening_param[mode, step]
                self._add_symbolic_safety_constraints(
                    opti,
                    lead_s_param[mode, step],
                    states_s[step],
                    states_v[step],
                    brake_slope=brake_slope_param[mode, step],
                    brake_intercept=brake_intercept_param[mode, step],
                    brake_v_min=brake_v_min_param[mode, step],
                    brake_v_max=brake_v_max_param[mode, step],
                    enforce_brake_speed_band=step > 0,
                    tightening=tightening,
                    slack=slack_var[mode, step],
                    clearance_scale=clearance_scale_param[mode, step],
                )

        opti.minimize(objective)
        p_opts, s_opts = self._gurobi_options()
        opti.solver(self.config.solver_name, p_opts, s_opts)

        problem = CachedFeedbackScalarChanceQP(
            opti=opti,
            h_var=h_var,
            k_var=k_var,
            k_group_capacity=k_group_capacity,
            k_select_params=k_select_params,
            slack_var=slack_var,
            eta_var=eta_var,
            rho_var=rho_var,
            x0_param=x0_param,
            prev_u_param=prev_u_param,
            previous_accel_param=previous_accel_param,
            command_dt_param=command_dt_param,
            lead_s_param=lead_s_param,
            lead_v_param=lead_v_param,
            x_ref_s_param=x_ref_s_param,
            x_ref_v_param=x_ref_v_param,
            a_ref_param=a_ref_param,
            probability_param=probability_param,
            tightening_param=tightening_param,
            clearance_scale_param=clearance_scale_param,
            safety_std_param=safety_std_param,
            brake_slope_param=brake_slope_param,
            brake_intercept_param=brake_intercept_param,
            brake_v_min_param=brake_v_min_param,
            brake_v_max_param=brake_v_max_param,
            lead_dev_s_params=lead_dev_s_params,
            lead_dev_v_params=lead_dev_v_params,
            sharing_params=sharing_params,
            objective=objective,
            optimized_eta=optimized_eta,
            optimize_k=optimize_k,
        )
        self._feedback_scalar_chance_qp_cache[key] = problem
        return problem

    def _solve_qp_gurobi(self, x0, prediction, reference, risks, policy0, samples, open_loop, layout):
        phase_start = time.perf_counter()
        horizon = self.config.horizon
        num_modes = prediction.num_modes
        num_joint_modes = prediction.num_joint_modes
        previous_accel_cmd, command_dt = self._command_context()
        brake_v_min, brake_v_max, brake_slope, brake_intercept = (
            self._brake_distance_bound_data(x0, num_modes)
        )
        opti = ca.Opti("conic")

        h_var = []
        m_var = []
        for group_id, step in enumerate(layout.group_steps):
            h_group = opti.variable()
            mode, _ = self._representative_mode_step(layout, group_id)
            opti.set_initial(h_group, policy0.h[mode, step, 0])
            if open_loop:
                m_group = None
            else:
                m_group = opti.variable(step, NX) if step > 0 else None
                if m_group is not None:
                    opti.set_initial(m_group, policy0.M[mode, step, :step, :])
            h_var.append(h_group)
            m_var.append(m_group)
        k_var = None
        if not open_loop and self.config.optimize_k:
            k_var = []
            for step in range(horizon):
                step_vars = {}
                for group_id in sorted(set(int(group) for group in layout.k_group_map[:, step])):
                    k_group = opti.variable(NO)
                    locations = np.argwhere(layout.k_group_map[:, step] == group_id)
                    rep_mode = int(locations[0, 0])
                    opti.set_initial(k_group, policy0.K[rep_mode, step])
                    step_vars[group_id] = k_group
                k_var.append(step_vars)
        slack_var = opti.variable() if open_loop else opti.variable(num_modes, horizon + 1)

        if open_loop:
            opti.set_initial(slack_var, 0.0)
        else:
            opti.set_initial(slack_var, np.zeros((num_modes, horizon + 1)))
        use_optimized_eta = (
            self.config.safety_constraint_mode == SAFETY_SCALAR_CHANCE
            and self.config.risk_allocation_mode == RISK_OPTIMIZED_ETA
        )
        joint_eta_var = None
        joint_rho_var = None
        if use_optimized_eta:
            joint_eta_var = opti.variable(num_joint_modes)
            joint_rho_var = opti.variable(num_joint_modes)
            initial_eta = np.full(
                (num_joint_modes, 1),
                np.clip(1.64, self.config.eta_std_min, self.config.eta_std_max),
            )
            initial_rho = np.full(
                (num_joint_modes, 1),
                min(1.0, optimized_eta_target_probability(self.config)),
            )
            opti.set_initial(joint_eta_var, initial_eta)
            opti.set_initial(joint_rho_var, initial_rho)
            opti.subject_to(joint_eta_var >= self.config.eta_std_min)
            opti.subject_to(joint_eta_var <= self.config.eta_std_max)
            opti.subject_to(joint_rho_var >= 0.5)
            opti.subject_to(joint_rho_var <= 1.0)
            for slope, intercept in cdf_lower_bound_lines(self.config):
                opti.subject_to(joint_rho_var <= slope * joint_eta_var + intercept)
            total_safe_probability = 0.0
            for joint_mode in range(num_joint_modes):
                total_safe_probability += (
                    float(prediction.probabilities[joint_mode]) * joint_rho_var[joint_mode]
                )
            opti.subject_to(total_safe_probability >= optimized_eta_target_probability(self.config))

        def slack_at(mode, step):
            return slack_var if open_loop else slack_var[mode, step]

        objective = self.config.slack_weight * ca.sumsqr(slack_var)

        opti.subject_to(ca.vec(slack_var) >= 0.0)
        if k_var is not None:
            for step_vars in k_var:
                for k_group in step_vars.values():
                    opti.subject_to(k_group <= self.config.feedback_bound)
                    opti.subject_to(k_group >= -self.config.feedback_bound)
        if not open_loop:
            for group_id in range(layout.policy_group_count):
                if m_var[group_id] is not None:
                    opti.subject_to(ca.vec(m_var[group_id]) <= self.config.disturbance_feedback_bound)
                    opti.subject_to(ca.vec(m_var[group_id]) >= -self.config.disturbance_feedback_bound)

        for mode in range(num_modes):
            for step in range(horizon):
                prev_u = float(reference.prev_u[mode, step])
                h_group = h_var[layout.group_map[mode, step]]
                opti.subject_to(h_group >= self.config.a_min - prev_u)
                opti.subject_to(h_group <= self.config.a_max - prev_u)

        # Prediction-tree sharing already enforces non-anticipative first input.
        # Keep the explicit equality only for layouts whose first-step groups differ.
        if len(set(int(layout.group_map[mode, 0]) for mode in range(num_modes))) > 1:
            first_input = float(reference.prev_u[0, 0]) + h_var[layout.group_map[0, 0]]
            for mode in range(1, num_modes):
                opti.subject_to(float(reference.prev_u[mode, 0]) + h_var[layout.group_map[mode, 0]] == first_input)

        if self.config.safety_constraint_mode in CHANCE_SAFETY_MODES:
            process_zero = np.zeros((horizon, NX))
            for mode in range(num_modes):
                states, _ = self._rollout_symbolic(
                    x0,
                    mode,
                    h_var,
                    k_var,
                    m_var,
                    layout,
                    reference,
                    prediction,
                    prediction.means[mode],
                    process_zero,
                )
                for step in range(horizon + 1):
                    if not prediction.active_mask[mode, step]:
                        continue
                    state = states[step]
                    lead = prediction.means[mode, step]
                    if use_optimized_eta:
                        tightening = (
                            safety_std(prediction.covariances[mode, step], self.config)
                            * joint_eta_var[mode]
                        )
                    else:
                        tightening = self._cell_tightening(prediction, risks, mode, step)
                    self._add_symbolic_safety_constraints(
                        opti,
                        float(lead[0]),
                        state[0],
                        state[1],
                        tightening=tightening,
                        slack=slack_at(mode, step),
                        clearance_scale=float(prediction.clearance_scale[mode, step]),
                    )

        x_ref = reference.x_ref
        use_sample_zero_safety = (
            self.config.safety_constraint_mode in (
                SAFETY_NOMINAL_SAFE_DISTANCE,
                SAFETY_BRAKE_DISTANCE,
            )
        )
        for mode in range(num_modes):
            for sample_idx, (lead_sample, process_sample) in enumerate(samples[mode]):
                states, inputs = self._rollout_symbolic(
                    x0, mode, h_var, k_var, m_var, layout, reference, prediction, lead_sample, process_sample
                )
                for step in range(horizon + 1):
                    state = states[step]
                    if use_sample_zero_safety and sample_idx == 0 and prediction.active_mask[mode, step]:
                        lead = prediction.means[mode, step]
                        self._add_symbolic_safety_constraints(
                            opti,
                            float(lead[0]),
                            state[0],
                            state[1],
                            brake_slope=brake_slope[mode, step],
                            brake_intercept=brake_intercept[mode, step],
                            brake_v_min=brake_v_min[mode, step],
                            brake_v_max=brake_v_max[mode, step],
                            enforce_brake_speed_band=step > 0,
                            slack=slack_at(mode, step),
                            clearance_scale=float(prediction.clearance_scale[mode, step]),
                        )
                    if step > 0:
                        opti.subject_to(state[1] >= self.config.v_min)
                        opti.subject_to(state[1] <= self.config.v_max)

                    s_error = state[0] - float(x_ref[mode, step, 0])
                    v_error = state[1] - float(x_ref[mode, step, 1])
                    objective += prediction.probabilities[mode] * (
                        self.config.q_s * s_error ** 2 + self.config.q_v * v_error ** 2
                    )

                previous_input = previous_accel_cmd
                for step in range(horizon):
                    control = inputs[step]
                    opti.subject_to(control >= self.config.a_min)
                    opti.subject_to(control <= self.config.a_max)
                    if step == 0:
                        self._add_symbolic_jerk_constraints(
                            opti,
                            control,
                            previous_accel_cmd,
                            command_dt,
                        )
                    else:
                        self._add_symbolic_jerk_constraints(
                            opti,
                            control,
                            previous_input,
                            self.config.dt,
                        )
                    input_error = control - float(reference.a_ref[mode, step])
                    jerk = (control - previous_input) / self.config.dt
                    objective += prediction.probabilities[mode] * (
                        self.config.r_a * input_error ** 2 + self.config.r_jerk * jerk ** 2
                    )
                    previous_input = control

        opti.minimize(objective)
        p_opts, s_opts = self._gurobi_options()
        opti.solver(self.config.solver_name, p_opts, s_opts)
        build_problem_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        sol = opti.solve()
        stats = opti.stats()
        opti_solve_s = time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        h = np.zeros((num_modes, horizon, NU))
        k = np.zeros((num_modes, horizon, NO))
        m = np.zeros((num_modes, horizon, horizon, NX))
        for mode in range(num_modes):
            for step in range(horizon):
                group_id = int(layout.group_map[mode, step])
                h[mode, step, 0] = float(sol.value(h_var[group_id]))
                if k_var is not None:
                    k_group_id = int(layout.k_group_map[mode, step])
                    k[mode, step] = np.asarray(sol.value(k_var[step][k_group_id]), dtype=float).reshape(NO)
                if m_var[group_id] is not None:
                    m[mode, step, :step, :] = np.asarray(sol.value(m_var[group_id]), dtype=float).reshape((step, NX))
        if open_loop:
            slack = np.full(
                (num_modes, horizon + 1),
                float(sol.value(slack_var)),
                dtype=float,
            )
        else:
            slack = np.asarray(sol.value(slack_var), dtype=float).reshape((num_modes, horizon + 1))
        optimized_eta_data = None
        if use_optimized_eta:
            eta_levels = np.asarray(sol.value(joint_eta_var), dtype=float).reshape(num_joint_modes)
            safe_probability_levels = np.asarray(sol.value(joint_rho_var), dtype=float).reshape(num_joint_modes)
            optimized_eta_data = {
                "eta_levels": eta_levels,
                "safe_probability_levels": safe_probability_levels,
                "target_safe_probability": optimized_eta_target_probability(self.config),
            }
        for step in range(horizon):
            if step + 1 < horizon:
                m[:, step, step + 1:, :] = 0.0
        extract_solution_s = time.perf_counter() - phase_start

        status = str(stats.get("return_status", "unknown"))
        message = (
            "gurobi_return_status=%s;diag_build_problem=%.6f;diag_opti_solve=%.6f;diag_extract=%.6f"
            % (status, build_problem_s, opti_solve_s, extract_solution_s)
        )
        cost = float(sol.value(objective))
        return NairACCPolicy(h=h, M=m, K=k), slack, cost, "optimal", message, layout, optimized_eta_data

    def _gurobi_options(self):
        p_opts = {"error_on_fail": 0}
        s_opts = {
            "OutputFlag": int(bool(self.config.gurobi_output)),
            "FeasibilityTol": self.config.optimizer_ftol,
            "OptimalityTol": self.config.optimizer_ftol,
            "BarConvTol": self.config.optimizer_ftol,
            "BarQCPConvTol": self.config.optimizer_ftol,
            "PSDTol": max(self.config.optimizer_ftol, 1.0e-9),
        }
        if self.config.gurobi_time_limit is not None:
            s_opts["TimeLimit"] = float(self.config.gurobi_time_limit)
        return p_opts, s_opts

    def _policy_variable_layout(self, prediction, open_loop=False):
        num_modes = prediction.num_modes
        num_joint_modes = prediction.num_joint_modes
        horizon = self.config.horizon
        if open_loop:
            raw_map = np.tile(np.arange(horizon, dtype=int), (num_modes, 1))
            policy_split_steps = ()
            first_policy_split_step = None
        elif prediction.policy_tree is None:
            raw_map = np.arange(num_modes * horizon, dtype=int).reshape((num_modes, horizon))
            policy_split_steps = ()
            first_policy_split_step = None
        else:
            raw_map = np.asarray(prediction.policy_tree.policy_groups_by_step, dtype=int)
            policy_split_steps = tuple(int(step) for step in prediction.policy_tree.policy_split_steps)
            first_policy_split_step = prediction.policy_tree.first_policy_split_step
            if raw_map.shape != (num_modes, horizon):
                raise ValueError("policy_tree groups must have shape (J, N)")

        group_map = np.zeros_like(raw_map)
        group_steps = []
        remap = {}
        for step in range(horizon):
            for mode in range(num_modes):
                raw_id = int(raw_map[mode, step])
                if raw_id not in remap:
                    remap[raw_id] = len(group_steps)
                    group_steps.append(step)
                group_id = remap[raw_id]
                if group_steps[group_id] != step:
                    raise ValueError("policy groups cannot span multiple time steps")
                group_map[mode, step] = group_id

        if open_loop or not self.config.optimize_k:
            k_group_map = np.zeros((num_modes, horizon), dtype=int)
            k_group_names = []
            k_group_count = 0
            k_decision_variables = 0
        else:
            k_group_map, k_group_names = self._k_variable_group_map(prediction)
            k_group_count = (
                int(np.max(k_group_map)) + 1
                if k_group_map.size > 0
                else 0
            )
            k_decision_variables = 0
            for step in range(horizon):
                k_decision_variables += len(set(int(group) for group in k_group_map[:, step])) * NO

        risk_variable_count = 0
        if (
                self.config.safety_constraint_mode == SAFETY_SCALAR_CHANCE
                and self.config.risk_allocation_mode == RISK_OPTIMIZED_ETA):
            risk_variable_count = 2 * num_joint_modes

        if open_loop:
            baseline_decision_variables = (
                horizon * NU
                + 1
                + risk_variable_count
            )
            num_decision_variables = (
                len(group_steps) * NU
                + 1
                + risk_variable_count
            )
        else:
            baseline_decision_variables = (
                num_modes * horizon * NU
                + num_modes * horizon * NO
                + num_modes * horizon * horizon * NX
                + num_modes * (horizon + 1)
                + risk_variable_count
            )
            num_decision_variables = (
                len(group_steps) * NU
                + k_decision_variables
                + sum(step * NX for step in group_steps)
                + num_modes * (horizon + 1)
                + risk_variable_count
            )
        return PolicyVariableLayout(
            group_map=group_map,
            group_steps=group_steps,
            k_group_map=k_group_map,
            k_group_names=k_group_names,
            k_group_count=k_group_count,
            k_decision_variables=k_decision_variables,
            num_decision_variables=num_decision_variables,
            baseline_decision_variables=baseline_decision_variables,
            policy_split_steps=policy_split_steps,
            first_policy_split_step=first_policy_split_step,
        )

    def _k_variable_group_map(self, prediction):
        num_modes = prediction.num_modes
        horizon = self.config.horizon
        if prediction.k_group_map is not None:
            raw_map = np.asarray(prediction.k_group_map, dtype=int)
            if raw_map.shape != (num_modes, horizon):
                raise ValueError("k_group_map must have shape (J, N)")
            raw_names = prediction.k_group_names
        else:
            raw_map = np.tile(np.arange(num_modes, dtype=int).reshape((num_modes, 1)), (1, horizon))
            raw_names = [f"mode_{mode}" for mode in range(num_modes)]

        remap = {}
        names = []
        k_group_map = np.zeros_like(raw_map)
        for step in range(horizon):
            for mode in range(num_modes):
                raw_id = int(raw_map[mode, step])
                if raw_id not in remap:
                    remap[raw_id] = len(remap)
                    if raw_names is not None and 0 <= raw_id < len(raw_names):
                        names.append(str(raw_names[raw_id]))
                    else:
                        names.append(f"k_group_{raw_id}")
                k_group_map[mode, step] = remap[raw_id]
        return k_group_map, names

    def _representative_mode_step(self, layout, group_id):
        locations = np.argwhere(layout.group_map == group_id)
        if locations.shape[0] == 0:
            raise ValueError("empty policy group")
        mode, step = locations[0]
        return int(mode), int(step)

    def _rollout_symbolic(
            self, x0, mode, h_var, k_var, m_var, layout, reference, prediction, lead_sample, process_sample):
        horizon = self.config.horizon
        A = ca.DM(self.A)
        B = ca.DM(self.B)
        state = ca.DM(np.asarray(x0, dtype=float).reshape((NX, 1)))
        states = [state]
        inputs = []

        for step in range(horizon):
            lead_dev = lead_sample[step] - prediction.means[mode, step]
            group_id = int(layout.group_map[mode, step])
            delta_u = h_var[group_id]
            if k_var is not None:
                k_group_id = int(layout.k_group_map[mode, step])
                k_group = k_var[step][k_group_id]
                delta_u += k_group[0] * float(lead_dev[0])
                delta_u += k_group[1] * float(lead_dev[1])
            if m_var[group_id] is not None:
                for hist_step in range(step):
                    delta_u += m_var[group_id][hist_step, 0] * float(process_sample[hist_step, 0])
                    delta_u += m_var[group_id][hist_step, 1] * float(process_sample[hist_step, 1])

            control = float(reference.prev_u[mode, step]) + delta_u
            process = ca.DM(process_sample[step].reshape((NX, 1)))
            state = ca.mtimes(A, state) + B * control + process
            inputs.append(control)
            states.append(state)
        return states, inputs

    def _build_samples(self, prediction):
        samples = []
        horizon = self.config.horizon
        process_zero = np.zeros((horizon, NX))
        for mode in range(prediction.num_modes):
            mode_samples = [(prediction.means[mode], process_zero)]
            std = np.sqrt(np.maximum(np.diagonal(prediction.covariances[mode], axis1=1, axis2=2), 0.0))
            close = prediction.means[mode].copy()
            far = prediction.means[mode].copy()
            close[:, 0] -= std[:, 0]
            close[:, 1] -= std[:, 1]
            far[:, 0] += std[:, 0]
            far[:, 1] += std[:, 1]
            inactive = ~prediction.active_mask[mode]
            close[inactive] = prediction.means[mode, inactive]
            far[inactive] = prediction.means[mode, inactive]
            mode_samples.append((close, process_zero))
            mode_samples.append((far, process_zero))
            samples.append(mode_samples)
        return samples

    def _constraints(self, x0, prediction, reference, risks, layout, samples):
        constraints = []
        horizon = self.config.horizon

        def make_safety(mode, step, lead_sample, process_sample):
            def fun(z):
                policy, slack = self._unpack(z, layout, prediction.num_modes)
                states, _ = self._rollout_policy(
                    x0, mode, policy, reference, prediction, lead_sample, process_sample
                )
                value = safety_function(
                    states[step],
                    lead_sample[step],
                    self.config,
                    linearization_speed=x0[1],
                    clearance_scale=float(prediction.clearance_scale[mode, step]),
                )
                return float(value + slack[mode, step])
            return fun

        def make_v_lower(mode, step, lead_sample, process_sample):
            def fun(z):
                policy, _ = self._unpack(z, layout, prediction.num_modes)
                states, _ = self._rollout_policy(
                    x0, mode, policy, reference, prediction, lead_sample, process_sample
                )
                return float(states[step, 1] - self.config.v_min)
            return fun

        def make_v_upper(mode, step, lead_sample, process_sample):
            def fun(z):
                policy, _ = self._unpack(z, layout, prediction.num_modes)
                states, _ = self._rollout_policy(
                    x0, mode, policy, reference, prediction, lead_sample, process_sample
                )
                return float(self.config.v_max - states[step, 1])
            return fun

        def make_u_lower(mode, step, lead_sample, process_sample):
            def fun(z):
                policy, _ = self._unpack(z, layout, prediction.num_modes)
                _, inputs = self._rollout_policy(
                    x0, mode, policy, reference, prediction, lead_sample, process_sample
                )
                return float(inputs[step] - self.config.a_min)
            return fun

        def make_u_upper(mode, step, lead_sample, process_sample):
            def fun(z):
                policy, _ = self._unpack(z, layout, prediction.num_modes)
                _, inputs = self._rollout_policy(
                    x0, mode, policy, reference, prediction, lead_sample, process_sample
                )
                return float(self.config.a_max - inputs[step])
            return fun

        for mode in range(prediction.num_modes):
            for sample_idx, (lead_sample, process_sample) in enumerate(samples[mode]):
                for step in range(horizon + 1):
                    if sample_idx == 0 and prediction.active_mask[mode, step]:
                        constraints.append({"type": "ineq", "fun": make_safety(mode, step, lead_sample, process_sample)})
                    constraints.append({"type": "ineq", "fun": make_v_lower(mode, step, lead_sample, process_sample)})
                    constraints.append({"type": "ineq", "fun": make_v_upper(mode, step, lead_sample, process_sample)})
                for step in range(horizon):
                    constraints.append({"type": "ineq", "fun": make_u_lower(mode, step, lead_sample, process_sample)})
                    constraints.append({"type": "ineq", "fun": make_u_upper(mode, step, lead_sample, process_sample)})

        if len(set(int(layout.group_map[mode, 0]) for mode in range(prediction.num_modes))) > 1:
            for mode in range(1, prediction.num_modes):
                constraints.append({
                    "type": "eq",
                    "fun": self._make_first_input_nac(layout, prediction.num_modes, mode, reference),
                })
        return constraints

    def _make_first_input_nac(self, layout, num_modes, mode, reference):
        def fun(z):
            policy, _ = self._unpack(z, layout, num_modes)
            first = reference.prev_u[0, 0] + policy.h[0, 0, 0]
            candidate = reference.prev_u[mode, 0] + policy.h[mode, 0, 0]
            return float(first - candidate)
        return fun

    def _rollout_policy(self, x0, mode, policy, reference, prediction, lead_sample, process_sample):
        states = np.zeros((self.config.horizon + 1, NX))
        inputs = np.zeros(self.config.horizon)
        states[0] = x0
        process_history = []
        for step in range(self.config.horizon):
            lead_dev = lead_sample[step] - prediction.means[mode, step]
            delta_u = policy.acceleration_deviation(mode, step, lead_dev, process_history)
            accel = reference.prev_u[mode, step] + delta_u
            accel = float(np.clip(accel, self.config.a_min, self.config.a_max))
            inputs[step] = accel
            states[step + 1] = acc_step(states[step], accel, self.config.dt) + process_sample[step]
            process_history.append(process_sample[step])
        return states, inputs

    def _fallback_policy(self, x0, prediction, reference):
        policy = self.initial_policy(prediction.num_modes)
        slack = np.zeros((prediction.num_modes, self.config.horizon + 1))
        previous_accel_cmd, command_dt = self._command_context()
        for mode in range(prediction.num_modes):
            states = np.zeros((self.config.horizon + 1, NX))
            states[0] = x0
            previous_input = previous_accel_cmd
            for step in range(self.config.horizon):
                if prediction.active_mask[mode, step]:
                    gap = safety_function(
                        states[step],
                        prediction.means[mode, step],
                        self.config,
                        linearization_speed=x0[1],
                        clearance_scale=float(prediction.clearance_scale[mode, step]),
                    )
                    desired = reference.a_ref[mode, step] - 0.25 * min(0.0, gap)
                else:
                    desired = reference.prev_u[mode, step]
                step_dt = command_dt if step == 0 else self.config.dt
                accel = self._rate_limited_accel(desired, previous_input, step_dt)
                policy.h[mode, step, 0] = accel - reference.prev_u[mode, step]
                previous_input = accel
                states[step + 1] = acc_step(
                    states[step],
                    accel,
                    self.config.dt,
                )
            for step in range(self.config.horizon + 1):
                if prediction.active_mask[mode, step]:
                    slack[mode, step] = max(
                        0.0,
                        -safety_function(
                            states[step],
                            prediction.means[mode, step],
                            self.config,
                            linearization_speed=x0[1],
                            clearance_scale=float(prediction.clearance_scale[mode, step]),
                        ),
                    )
        policy.K[:] = 0.0
        policy.M[:] = 0.0
        return policy, slack


def acc_dynamics_matrices(dt):
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    B = np.array([[0.5 * dt ** 2], [dt]], dtype=float)
    return A, B


def acc_step(state, acceleration, dt):
    state = np.asarray(state, dtype=float)
    return np.array([
        state[0] + dt * state[1] + 0.5 * dt ** 2 * acceleration,
        state[1] + dt * acceleration,
    ])


def rollout_dynamics(initial_state, accelerations, dt):
    accelerations = np.asarray(accelerations, dtype=float)
    states = np.zeros((accelerations.shape[0] + 1, NX))
    states[0] = np.asarray(initial_state, dtype=float)
    for step, acceleration in enumerate(accelerations):
        states[step + 1] = acc_step(states[step], acceleration, dt)
    return states


def brake_deceleration(config):
    return max(abs(float(config.a_min)), 1.0e-9)


def exact_brake_distance(ego_speed, config):
    decel = brake_deceleration(config)
    return ego_speed ** 2 / (2.0 * decel)


def brake_distance_upper_bound_coefficients(v_min, v_max, config):
    v_min = np.asarray(v_min, dtype=float)
    v_max = np.asarray(v_max, dtype=float)
    if np.any(v_min > v_max + 1.0e-9):
        raise ValueError("brake-distance interval must satisfy v_min <= v_max")
    decel = brake_deceleration(config)
    slope = (v_min + v_max) / (2.0 * decel)
    intercept = -(v_min * v_max) / (2.0 * decel)
    return slope, intercept


def upper_bound_brake_distance(ego_speed, config, v_min, v_max):
    slope, intercept = brake_distance_upper_bound_coefficients(v_min, v_max, config)
    return slope * ego_speed + intercept


def clearance_gap(ego_state, lead_state, config):
    ego_state = np.asarray(ego_state, dtype=float)
    lead_state = np.asarray(lead_state, dtype=float)
    return float(
        lead_state[0]
        - ego_state[0]
        - config.vehicle_length
    )


def required_standoff(ego_state, config):
    """Deterministic distance the ego must keep behind the lead, footprint included."""
    ego_state = np.asarray(ego_state, dtype=float)
    if config.safety_constraint_mode == SAFETY_BRAKE_DISTANCE:
        clearance = config.d0 + exact_brake_distance(ego_state[1], config)
    else:
        clearance = config.d0 + config.time_headway * ego_state[1]
    return float(config.vehicle_length + clearance)


def safety_function(ego_state, lead_state, config, linearization_speed=None, clearance_scale=1.0):
    del linearization_speed
    ego_state = np.asarray(ego_state, dtype=float)
    lead_state = np.asarray(lead_state, dtype=float)
    return float(
        lead_state[0]
        - ego_state[0]
        - float(clearance_scale) * required_standoff(ego_state, config)
    )


def safety_std(lead_covariance, config):
    lead_covariance = np.asarray(lead_covariance, dtype=float)
    process_covariance = np.asarray(config.process_noise_cov, dtype=float)
    h_lead = np.array([1.0, 0.0])
    h_ego = np.array([-1.0, -config.time_headway])
    variance = float(h_lead.dot(lead_covariance).dot(h_lead))
    variance += float(h_ego.dot(process_covariance).dot(h_ego))
    return float(np.sqrt(max(variance, 0.0)))


def chance_tightening(lead_covariance, risk, config):
    risk = float(np.clip(risk, 1.0e-6, 0.49))
    return float(norm.ppf(1.0 - risk) * safety_std(lead_covariance, config))


def confidence_quantile(beta):
    """Half-width, in standard deviations, of the interval holding probability ``beta``.

    The 1-D counterpart of the confidence ellipse of Benciolini et al. (T-IV
    2023), eq. (17): a Gaussian puts probability ``beta`` inside
    ``mu +/- q(beta) sigma`` with ``q(beta) = Phi^-1((1 + beta) / 2)``.  It vanishes
    as ``beta -> 0`` and diverges as ``beta -> 1``, so the caller caps ``beta``
    (their Remark 4).  The paper's ``sqrt(-2 ln(1 - beta))`` is the quantile of
    its 2-D ellipse, not of a longitudinal-only interval.
    """
    beta = float(np.clip(beta, 0.0, 1.0 - 1.0e-9))
    return float(norm.ppf(0.5 * (1.0 + beta)))


def confidence_tightening(lead_covariance, beta, config):
    """``q(beta) * safety_std``: the sigma term of the confidence chance constraint.

    ``beta`` is the per-cell confidence in ``MultimodalLeadPrediction.chance_confidence``.
    NaN marks a deterministic cell and contributes nothing.
    """
    if not np.isfinite(beta):
        return 0.0
    return float(confidence_quantile(beta) * safety_std(lead_covariance, config))


def cdf_lower_bound_lines(config):
    lines = []
    for eta_pair, rho_pair in config.cdf_approx_segments:
        eta0, eta1 = eta_pair
        rho0, rho1 = rho_pair
        if eta1 <= eta0:
            raise ValueError("CDF approximation eta breakpoints must be increasing")
        slope = (rho1 - rho0) / (eta1 - eta0)
        intercept = rho0 - slope * eta0
        lines.append((float(slope), float(intercept)))
    return lines


def optimized_eta_target_probability(config):
    if config.fixed_risk_level is not None:
        return float(1.0 - config.fixed_risk_level)
    return float(1.0 - config.epsilon)


def scalar_chance_margin(ego_state, lead_mean, lead_covariance, risk, config):
    return safety_function(ego_state, lead_mean, config) - chance_tightening(
        lead_covariance, risk, config
    )


def allocate_risk(probabilities, config):
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    prob_sum = float(np.sum(probabilities))
    if prob_sum <= 0.0:
        probabilities = np.ones_like(probabilities) / probabilities.size
    else:
        probabilities = probabilities / prob_sum
    horizon = config.horizon
    if config.risk_allocation_mode == RISK_OPTIMIZED_ETA:
        base = (
            config.fixed_risk_level
            if config.fixed_risk_level is not None
            else config.epsilon
        )
        return np.full((probabilities.shape[0], horizon + 1), float(base))

    if config.risk_allocation_mode == RISK_PROBABILITY_WEIGHTED:
        base = (
            config.fixed_risk_level
            if config.fixed_risk_level is not None
            else config.epsilon / (horizon + 1)
        )
        risks = np.zeros((probabilities.shape[0], horizon + 1))
        for mode, probability in enumerate(probabilities):
            risks[mode, :] = float(base) / (probabilities.shape[0] * max(probability, 1.0e-6))
        return np.clip(risks, 1.0e-6, config.eta_max)

    if config.fixed_risk_level is not None:
        base = float(config.fixed_risk_level)
        return np.full((probabilities.shape[0], horizon + 1), base)

    total_constraints = probabilities.shape[0] * (horizon + 1)
    if not config.use_variable_risk:
        return np.full((probabilities.shape[0], horizon + 1), config.epsilon / total_constraints)

    # TODO: connect this lightweight allocation to the full Nair eta optimization.
    # For now, allocate more risk to less probable modes while respecting eta_max.
    inv_prob = 1.0 / np.maximum(probabilities, 1.0e-6)
    mode_weights = inv_prob / np.sum(inv_prob)
    risks = np.zeros((probabilities.shape[0], horizon + 1))
    for mode in range(probabilities.shape[0]):
        risks[mode, :] = config.epsilon * mode_weights[mode] / (horizon + 1)
    return np.clip(risks, 1.0e-6, config.eta_max)


def fit_ltv_from_means(means):
    means = np.asarray(means, dtype=float)
    num_modes, horizon_plus_one, _ = means.shape
    horizon = horizon_plus_one - 1
    T = np.zeros((num_modes, horizon, NO, NO))
    c = np.zeros((num_modes, horizon, NO))
    for mode in range(num_modes):
        for step in range(horizon):
            T[mode, step] = np.eye(NO)
            c[mode, step] = means[mode, step + 1] - means[mode, step]
    return T, c


def make_constant_accel_lead_prediction(
        s0,
        v0,
        accelerations,
        probabilities,
        horizon,
        dt,
        covariance=None,
        mode_names=None):
    accelerations = np.asarray(accelerations, dtype=float).reshape(-1)
    num_modes = accelerations.shape[0]
    means = np.zeros((num_modes, horizon + 1, NO))
    T = np.zeros((num_modes, horizon, NO, NO))
    c = np.zeros((num_modes, horizon, NO))
    for mode, accel in enumerate(accelerations):
        means[mode, 0] = np.array([s0, v0], dtype=float)
        for step in range(horizon):
            T[mode, step] = np.array([[1.0, dt], [0.0, 1.0]])
            c[mode, step] = np.array([0.5 * dt ** 2 * accel, dt * accel])
            means[mode, step + 1] = T[mode, step].dot(means[mode, step]) + c[mode, step]
            means[mode, step + 1, 1] = max(0.0, means[mode, step + 1, 1])

    if covariance is None:
        covariance = np.diag([0.25, 0.09])
    covariance = np.asarray(covariance, dtype=float)
    covariances = np.tile(covariance, (num_modes, horizon + 1, 1, 1))
    return MultimodalLeadPrediction(
        means=means,
        probabilities=np.asarray(probabilities, dtype=float),
        covariances=covariances,
        T=T,
        c=c,
        mode_names=mode_names,
    )


def policy_dimensions(policy):
    return {
        "h": policy.h.shape,
        "M": policy.M.shape,
        "K": policy.K.shape,
    }
