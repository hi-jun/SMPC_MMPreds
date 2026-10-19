"""Provably-correct and comfortable ACC (Althoff, Maierhofer & Pek, T-IV 2021).

The paper's longitudinal controller, used here as a rule-based baseline for the
multimodal ACC: no predictor, measured states only.  The CARLA adapter is
``policies/pcacc_agent.py``; this module is simulator-free.

Paper -> code (equation / section -> function):

* Def. II.1 / Prop. IV.5 safe distance
  ``d_safe = s(dt_cmd) + s_brake(x_hat) - v_p^2 / (2 |a_p_min|)``   -> ``safe_distance``
  The ego runs the intended input for one command interval, then the fail-safe
  braking (Fig. 6 jerk profile, held at a_min) -> ``ego_brake_distance``; the
  lead brakes at a_p_min immediately -> ``lead_brake_distance``.
* (2)-(6) nominal MPC: point-mass x_n = [ds, dv, a], input jerk, Q/r of
  Table I, lead at constant speed                                -> ``_BatchQP`` (``PCACCController._safety_keeping``)
* (7) safe use of the nominal controller: the next state's fail-safe must
  stop behind the fully braking lead                             -> ``safe_distance`` with the planned u0
* Sec. IV-B emergency controller: the predefined jerk profile    -> ``emergency_accel``
* Sec. II-C lane-change prediction, Sec. III-C delta_leave       -> ``LaneMembershipTracker``
* Sec. III-C relevant vehicles (faster leads, beyond stopping)   -> ``relevant_vehicles``
* Def. II.3 cut-in, Spec. III.1 clearing time, Prop. V.1         -> ``PCACCController._safety_recapturing``
  terminal constraint ``ds(t_c) >= d_safe(x(t_c), v_p + a_cutin t_c)`` with the
  lead braking at a_cutin inside the horizon
* Sec. V-B inevitable collision state / acceleration-bounded     -> ``is_inevitable_collision`` / bounded ``_BatchQP``
* Alg. 1 selector ``min(a_1 .. a_N)``                             -> ``PCACCController.step``

Choices where the paper is open (see the agent's docstring for the rest):
t_c is a constant (``t_clear_s``); x_hat for the MPC's d_safe is the zero-jerk
propagation; the ego's acceleration state is the last issued command; the
terminal safe distance of Prop. V.1 is linearised in the ego's terminal speed
around the lead's speed at t_c; the cruise controller is a proportional speed
law that also caps every following command at the set speed.

Safe distance and limits (2026-10-19, the maintainer's choice for the thesis
comparison): by default this fork's, not the paper's.  The standoff is
``d_safe(v) = d0 + tau v`` on the bumper gap (``standoff``; L = 4.5 m, d0 = 3 m,
tau = 1.3 s, AGENTS.md §3), the gap being centre to centre minus L exactly as
the aggregator scores delta, so delta = 0 means this standoff is held.  It
replaces Def. II.1 / Prop. IV.5 wherever the paper's d_safe is a standoff: the
nominal QP's reference and constraint (3), per step and linear in the ego's
speed (``ds_k >= d0 + tau v_k``); condition (7), which engages the fail-safe
when the next state's gap is below ``d_safe(v_next)``; Def. II.3; and the
terminal constraint of Prop. V.1 (linear in v, so exact).  The relevant-vehicle
reach and the inevitable collision state stay physical.  The limits default to
a in [-3, 2] m/s^2 and |j| <= 5 m/s^3 (Table I's through
``amin10_amax3_jlim10``); a_p_min, a_cut-in and t_c are unchanged.
``paperdsafe`` restores the paper's d_safe everywhere, on the physical bumper
gap.
"""
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np


# Fig. 6, read off the plot: zero-order-hold jerk of the first ten 0.1 s steps as
# fractions of |j_min| (the profile then stays at j_min).  Monotonically
# decreasing, so Prop. IV.2 applies.
JERK_PROFILE_FRACTIONS = (0.005, 0.01, 0.03, 0.05, 0.075, 0.12, 0.20, 0.30, 0.43, 0.60)

PATH_CRUISE = "cruise"          # no relevant vehicle: cruise controller
PATH_NOMINAL = "nominal"        # safety-keeping nominal MPC, verified by (7)
PATH_EMERGENCY = "emergency"    # safety-keeping fail-safe jerk profile
PATH_RECAPTURE = "recapture"    # safety-recapturing nominal MPC (Prop. V.1)
PATH_BOUNDED = "bounded"        # acceleration-bounded recapture (no jerk bounds)
PATH_ICS = "ics"                # inevitable collision state: full braking

_POLICY_TOKENS = {
    # token -> (config field, sign).  ``amin3`` is a_min = -3, ``jlim5`` j = +/-5.
    "amin": ("a_min", -1.0),
    "amax": ("a_max", 1.0),
    "jlim": ("jerk_max", 1.0),
    "apmin": ("a_p_min", -1.0),
    "acutin": ("a_cutin", -1.0),
    "tc": ("t_clear_s", 1.0),
}


@dataclass
class PCACCConfig:
    """Table I of the paper, plus the values it leaves open.  The limits and the
    safe distance default to this fork's (module docstring)."""
    jerk_max: float = 5.0         # j_bar [m/s^3]; j_min = -jerk_max  (Table I: 10)
    a_max: float = 2.0            # a_bar [m/s^2]  (constant: v_S is not given; Table I: 3)
    a_min: float = -3.0           # a_underbar  (Table I: -10)
    v_max: float = 51.0           # v_bar (sensor-range / limit speed)
    v_min: float = 0.0
    sensor_range_m: float = 200.0  # s_sr
    dt: float = 0.1               # Delta t: MPC step and jerk-profile hold
    a_p_min: float = -10.5        # a_p_underbar: lead full braking (Spec. III.5)
    rule_window_s: float = 0.5    # lambda
    s_c: float = 0.4              # lateral deviation threshold [m]
    theta_c: float = 0.035        # heading deviation threshold [rad]
    q_keep: Tuple[float, float, float] = (5.0, 10.0, 50.0)    # Q_k
    r_keep: float = 100.0                                       # r_k
    q_recap: Tuple[float, float, float] = (10.0, 30.0, 50.0)  # Q_r
    r_recap: float = 100.0                                      # r_r
    horizon_keep_s: float = 6.0   # h_k
    a_cutin: float = -2.0         # a_cut-in
    delta_leave_s: float = 1.0    # delta_leave
    # Open in the paper.
    t_clear_s: float = 4.0        # clearing time t_c (user-defined there)
    cruise_gain: float = 0.5      # cruise controller a = K (v_set - v)
    jerk_profile_fractions: Tuple[float, ...] = JERK_PROFILE_FRACTIONS
    # This fork's safe distance (AGENTS.md §3): ``d0 + tau v`` on the bumper gap,
    # the gap being centre to centre minus ``vehicle_length``.
    d0: float = 3.0
    time_headway: float = 1.3     # tau
    vehicle_length: float = 4.5   # L
    paper_d_safe: bool = False    # ``paperdsafe``: Def. II.1 / Prop. IV.5 instead

    @property
    def jerk_min(self):
        return -float(self.jerk_max)

    @property
    def profile_jerks(self):
        return [-float(f) * float(self.jerk_max) for f in self.jerk_profile_fractions]

    @classmethod
    def from_policy_string(cls, policy):
        """``acc_pcacc`` plus optional ``amin<x>`` ``amax<x>`` ``jlim<x>`` ``apmin<x>``
        ``acutin<x>`` ``tc<x>`` tokens (magnitudes; the sign is the field's;
        ``amin10_amax3_jlim10`` is Table I) and ``paperdsafe`` (the paper's d_safe)."""
        cfg = cls()
        for token, (name, sign) in _POLICY_TOKENS.items():
            m = re.search(r"(?:^|_)" + token + r"([0-9]*\.?[0-9]+)(?:_|$)", str(policy))
            if m:
                setattr(cfg, name, sign * float(m.group(1)))
        cfg.paper_d_safe = re.search(r"(?:^|_)paperdsafe(?:_|$)", str(policy)) is not None
        if cfg.a_p_min > cfg.a_min:
            raise ValueError("Spec. III.5 needs a_p_min <= a_min (%.2f > %.2f)" % (cfg.a_p_min, cfg.a_min))
        return cfg

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- kinematics

def _advance(v, a, j, t):
    """(distance, v, a) after ``t`` seconds of constant jerk ``j`` from (v, a)."""
    return (v * t + 0.5 * a * t * t + j * t ** 3 / 6.0,
            v + a * t + 0.5 * j * t * t,
            a + j * t)


def _time_to_standstill(v, a, j):
    """Smallest t > 0 with v + a t + j t^2 / 2 = 0, or None."""
    if v <= 0.0:
        return 0.0
    if abs(j) < 1.0e-12:
        return None if a >= 0.0 else -v / a
    disc = a * a - 2.0 * j * v
    if disc < 0.0:
        return None
    roots = [(-a - math.sqrt(disc)) / j, (-a + math.sqrt(disc)) / j]
    roots = [t for t in roots if t > 0.0]
    return min(roots) if roots else None


def ego_brake_distance(v, a, cfg):
    """Distance until standstill under the fail-safe trajectory from (v, a):
    the Fig. 6 jerk profile (zero-order hold, ``cfg.dt`` per step), then j_min,
    until a_min is reached, then a_min."""
    v = float(v)
    a = float(a)
    if v <= 0.0:
        return 0.0
    dist = 0.0
    jerks = cfg.profile_jerks
    k = 0
    while a > cfg.a_min + 1.0e-12:
        j = jerks[k] if k < len(jerks) else cfg.jerk_min
        k += 1
        t_seg = cfg.dt if k <= len(jerks) else (cfg.a_min - a) / j
        if j < 0.0:
            t_seg = min(t_seg, (cfg.a_min - a) / j)
        t_stop = _time_to_standstill(v, a, j)
        if t_stop is not None and t_stop <= t_seg:
            return dist + _advance(v, a, j, t_stop)[0]
        s, v, a = _advance(v, a, j, t_seg)
        dist += s
    # constant deceleration a_min until standstill
    return dist + v * v / (2.0 * abs(cfg.a_min))


def ego_brake_time(v, a, cfg):
    """Seconds until standstill under the fail-safe trajectory (for the ICS check)."""
    v = float(v)
    a = float(a)
    t_total = 0.0
    jerks = cfg.profile_jerks
    k = 0
    while v > 0.0 and a > cfg.a_min + 1.0e-12:
        j = jerks[k] if k < len(jerks) else cfg.jerk_min
        k += 1
        t_seg = cfg.dt if k <= len(jerks) else (cfg.a_min - a) / j
        if j < 0.0:
            t_seg = min(t_seg, (cfg.a_min - a) / j)
        t_stop = _time_to_standstill(v, a, j)
        if t_stop is not None and t_stop <= t_seg:
            return t_total + t_stop
        _, v, a = _advance(v, a, j, t_seg)
        t_total += t_seg
    return t_total + max(0.0, v) / abs(cfg.a_min)


def lead_brake_distance(v_p, cfg):
    """Lead braking immediately at a_p_min (Def. II.1, u_p,brake)."""
    return max(0.0, float(v_p)) ** 2 / (2.0 * abs(cfg.a_p_min))


def safe_distance(v, a, v_p, cfg, dt_cmd, u0=0.0):
    """Prop. IV.5: distance the ego needs now so that, after one command
    interval under jerk ``u0``, its fail-safe stops behind the fully braking
    lead.  ``gap >= safe_distance(..., u0)`` is condition (7)."""
    s1, v1, a1 = _advance(float(v), float(a), float(u0), float(dt_cmd))
    v1 = max(0.0, v1)
    a1 = min(cfg.a_max, max(cfg.a_min, a1))
    return max(0.0, s1) + ego_brake_distance(v1, a1, cfg) - lead_brake_distance(v_p, cfg)


def standoff(v, cfg):
    """This fork's safe distance on the bumper gap, ``d0 + tau v`` (AGENTS.md §3:
    ``L + d0 + tau v`` centre to centre).  The aggregator scores delta against
    the same geometry, so delta = 0 means the controller holds exactly it."""
    return cfg.d0 + cfg.time_headway * float(v)


def emergency_accel(t_since, a_engage, cfg):
    """Acceleration of the fail-safe trajectory ``t_since`` seconds after it was
    engaged at acceleration ``a_engage``: the integrated Fig. 6 profile, j_min
    after it, held at a_min."""
    a = float(a_engage)
    t = 0.0
    for j in cfg.profile_jerks:
        seg = min(cfg.dt, max(0.0, t_since - t))
        a += j * seg
        t += cfg.dt
        if t >= t_since:
            break
    if t_since > t:
        a += cfg.jerk_min * (t_since - t)
    return max(cfg.a_min, a)


def is_inevitable_collision(gap, v, a, v_p, cfg, step=0.05):
    """Def. II.2 with the cut-in assumption of Sec. V-B1: the lead brakes at
    a_cutin, the ego brakes as hard as the model allows (j_min, then a_min).
    True when the ego still reaches the lead."""
    v = float(v)
    a = float(a)
    s_ego = 0.0
    s_lead = float(gap)
    v_lead = float(v_p)
    t_end = max(ego_brake_time(v, a, cfg), v_lead / abs(cfg.a_cutin)) + step
    t = 0.0
    while t < t_end:
        if s_lead - s_ego < 0.0:
            return True
        # ego: jerk j_min until a_min
        j = cfg.jerk_min if a > cfg.a_min else 0.0
        t_seg = step if j == 0.0 else min(step, (cfg.a_min - a) / j)
        t_stop = _time_to_standstill(v, a, j)
        if t_stop is not None and t_stop < t_seg:
            s_ego += _advance(v, a, j, t_stop)[0]
            v, a = 0.0, 0.0
        else:
            ds, v, a = _advance(v, a, j, t_seg)
            s_ego += ds
            a = max(cfg.a_min, a)
        # lead at a_cutin until standstill
        t_lead_stop = v_lead / abs(cfg.a_cutin) if v_lead > 0.0 else 0.0
        t_l = min(t_seg, t_lead_stop)
        s_lead += v_lead * t_l + 0.5 * cfg.a_cutin * t_l * t_l
        v_lead = max(0.0, v_lead + cfg.a_cutin * t_l)
        t += t_seg
        if v <= 0.0:
            return s_lead - s_ego < 0.0
    return s_lead - s_ego < 0.0


# --------------------------------------------------------------------------- lane membership

@dataclass
class LaneMembership:
    occupies: bool          # O(x_p) intersects L_ego now
    predicted: bool         # Sec. II-C rule true over the last lambda seconds
    member: bool            # occupies or predicted, or left less than delta_leave ago
    rule_ok_since: Optional[float] = None
    last_in_lane_s: Optional[float] = None


class LaneMembershipTracker:
    """Ego-lane membership of every other vehicle (Sec. II-C and III-C).

    ``update`` takes, for one vehicle at time ``t``: whether its occupancy
    intersects the ego lane, whether it drives in a lane adjacent to the ego
    lane, and its lateral and heading deviation from its own lane centre
    *toward* the ego lane (positive = toward).  The cut-in rule
    ``|s_lat| > s_c and |theta| > theta_c for all of [t - lambda, t]`` is
    applied with the signed-toward values, so a vehicle drifting away from the
    ego lane never counts.  A vehicle that stops intersecting keeps its
    membership for ``delta_leave`` seconds.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._rule_ok_since = {}
        self._last_in_lane = {}

    def update(self, vid, t, occupies, adjacent, s_lat_toward, theta_toward):
        cfg = self.cfg
        ok = bool(adjacent) and s_lat_toward is not None and theta_toward is not None \
            and float(s_lat_toward) > cfg.s_c and float(theta_toward) > cfg.theta_c
        if ok:
            since = self._rule_ok_since.setdefault(vid, float(t))
            predicted = float(t) - since >= cfg.rule_window_s - 1.0e-9
        else:
            self._rule_ok_since.pop(vid, None)
            since = None
            predicted = False
        in_lane = bool(occupies) or predicted
        if in_lane:
            self._last_in_lane[vid] = float(t)
        last_in = self._last_in_lane.get(vid)
        member = in_lane or (last_in is not None and float(t) - last_in <= cfg.delta_leave_s)
        return LaneMembership(bool(occupies), predicted, member, since, last_in)

    def forget(self, vid):
        self._rule_ok_since.pop(vid, None)
        self._last_in_lane.pop(vid, None)


# --------------------------------------------------------------------------- QP

class _BatchQP:
    """Dense batch QP of (2)-(6) for x_n = [ds, dv, a] with jerk input held over
    ``dt``: ds' = dv, dv' = a_lead - a, a' = u.  The lead acceleration per step
    is given at solve time (0 for the safety-keeping MPC, a_cutin for the
    recapturing one).  ``jerk_bound`` None drops (6) (acceleration-bounded
    controller).  Solved with qpOASES through CasADi."""

    def __init__(self, N, dt, q, r, jerk_bound, label):
        self.N = int(N)
        self.dt = float(dt)
        self.jerk_bound = None if jerk_bound is None else float(jerk_bound)
        A = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
        B = np.array([-dt ** 3 / 6.0, -0.5 * dt * dt, dt])
        N = self.N
        Sx = np.zeros((3 * N, 3))
        Su = np.zeros((3 * N, N))
        sx, su = np.eye(3), np.zeros((3, N))
        for k in range(N):
            su = A @ su
            su[:, k] += B
            sx = A @ sx
            Sx[3 * k:3 * k + 3] = sx
            Su[3 * k:3 * k + 3] = su
        self.A, self.Sx, self.Su = A, Sx, Su
        Qbar = np.kron(np.eye(N), np.diag(np.asarray(q, dtype=float)))
        self.QSu = Qbar @ Su
        self.H = 2.0 * (Su.T @ self.QSu + float(r) * np.eye(N)) + 1.0e-9 * np.eye(N)
        # constraint rows: every stacked state, plus one terminal row (Prop. V.1)
        self.rows = 3 * N + 1
        self.solver = ca.conic(
            label, "qpoases",
            {"h": ca.Sparsity.dense(N, N), "a": ca.Sparsity.dense(self.rows, N)},
            {"printLevel": "none", "print_time": False, "error_on_fail": False, "verbose": False})

    def _affine_term(self, a_lead):
        """Stacked contribution of the lead acceleration profile (per step)."""
        dt = self.dt
        Se = np.zeros(3 * self.N)
        se = np.zeros(3)
        for k in range(self.N):
            se = self.A @ se + np.array([0.5 * a_lead[k] * dt * dt, a_lead[k] * dt, 0.0])
            Se[3 * k:3 * k + 3] = se
        return Se

    def solve(self, x0, d_ref, ds_min, dv_lo, dv_hi, a_lo, a_hi, a_lead=None, terminal=None,
              ds_dv_coef=0.0):
        """``a_lead``: lead acceleration per step (default 0).  ``ds_dv_coef``:
        the gap rows read ds_k + ds_dv_coef * dv_k >= ds_min_k (a standoff
        linear in the ego's speed v_k = v_p - dv_k).  ``terminal`` =
        (step k, coef_dv, rhs): the extra row ds_k + coef_dv * dv_k >= rhs at
        step k in 1..N.  Returns (feasible, U, X) with X the stacked predicted
        states."""
        N = self.N
        x0 = np.asarray(x0, dtype=float)
        a_lead = np.zeros(N) if a_lead is None else np.asarray(a_lead, dtype=float)
        free = self.Sx @ x0 + self._affine_term(a_lead)
        x_ref = np.tile([float(d_ref), 0.0, 0.0], N)
        g = 2.0 * (self.QSu.T @ (free - x_ref))
        lba = np.full(self.rows, -np.inf)
        uba = np.full(self.rows, np.inf)
        idx_ds = np.arange(0, 3 * N, 3)
        idx_dv = idx_ds + 1
        idx_a = idx_ds + 2
        lba[idx_ds] = np.asarray(ds_min, dtype=float) - free[idx_ds]
        lba[idx_dv] = np.asarray(dv_lo, dtype=float) - free[idx_dv]
        uba[idx_dv] = np.asarray(dv_hi, dtype=float) - free[idx_dv]
        lba[idx_a] = float(a_lo) - free[idx_a]
        uba[idx_a] = float(a_hi) - free[idx_a]
        A_c = np.zeros((self.rows, N))
        A_c[:3 * N] = self.Su
        if ds_dv_coef:
            A_c[idx_ds] += ds_dv_coef * self.Su[idx_dv]
            lba[idx_ds] -= ds_dv_coef * free[idx_dv]
        if terminal is not None:
            k, coef_dv, rhs = terminal
            i_ds, i_dv = idx_ds[int(k) - 1], idx_dv[int(k) - 1]
            A_c[-1] = self.Su[i_ds] + coef_dv * self.Su[i_dv]
            lba[-1] = rhs - (free[i_ds] + coef_dv * free[i_dv])
        if self.jerk_bound is None:
            lbx, ubx = -np.inf, np.inf
        else:
            lbx, ubx = -self.jerk_bound, self.jerk_bound
        res = self.solver(h=ca.DM(self.H), g=ca.DM(g), a=ca.DM(A_c),
                          lba=ca.DM(lba), uba=ca.DM(uba), lbx=lbx, ubx=ubx)
        stats = self.solver.stats()
        feasible = bool(stats.get("success", False))
        U = np.array(res["x"], dtype=float).ravel()
        X = free + self.Su @ U
        return feasible, U, X


# --------------------------------------------------------------------------- controller

@dataclass
class VehicleObs:
    """One preceding vehicle as the controller sees it this tick."""
    vid: int
    gap: float          # bumper gap [m]: centre to centre minus cfg.vehicle_length (paperdsafe: the measured bumpers)
    speed: float        # measured speed [m/s]
    member: bool        # ego-lane member (LaneMembershipTracker)
    role: str = ""


@dataclass
class VehicleDecision:
    path: str
    accel: float
    d_safe: float
    gap: float
    speed: float
    feasible: bool
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    accel: float
    path: str
    feasible: bool
    lead_vid: Optional[int]
    a_cruise: float
    per_vehicle: Dict[int, VehicleDecision]
    keep_vids: List[int]
    recap_vids: List[int]
    jerk_limited: bool      # the command is subject to the jerk bound (not for ``bounded``)


def relevant_vehicles(vehicles, v, a, cfg, dt_cmd):
    """Sec. III-C: drop a vehicle when a nearer one is at most as fast (1) and
    when it is beyond the ego's one-step reach plus emergency stopping distance (2).
    ``vehicles`` are the safety-keeping candidates sorted by gap."""
    s1, v1, a1 = _advance(float(v), float(a), cfg.jerk_max, float(dt_cmd))
    reach = max(0.0, s1) + ego_brake_distance(max(0.0, v1), min(cfg.a_max, a1), cfg)
    out = []
    v_nearer_min = np.inf
    for veh in vehicles:
        drop = veh.gap > cfg.sensor_range_m or veh.speed >= v_nearer_min or veh.gap >= reach
        v_nearer_min = min(v_nearer_min, veh.speed)
        if not drop:
            out.append(veh)
    return out


class PCACCController:
    """Alg. 1 of the paper for one tick: ``step`` takes the ego state, the set
    speed and the observed preceding vehicles and returns the acceleration for
    the actuator plus per-vehicle diagnostics.  Keeps the per-vehicle state:
    membership history (cut-in detection), the recapturing clock and the
    engaged fail-safe trajectory."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.N_keep = max(1, int(round(cfg.horizon_keep_s / cfg.dt)))       # h_k / dt
        self.N_recap = max(1, int(math.ceil(cfg.t_clear_s / cfg.dt - 1.0e-9)))  # t_c / dt
        self._qp_keep = _BatchQP(self.N_keep, cfg.dt, cfg.q_keep, cfg.r_keep, cfg.jerk_max, "keep")
        self._qp_recap = _BatchQP(self.N_recap, cfg.dt, cfg.q_recap, cfg.r_recap, cfg.jerk_max, "recap")
        self._qp_bounded = _BatchQP(self.N_recap, cfg.dt, cfg.q_recap, cfg.r_recap, None, "bounded")
        self._was_member = {}
        self._cutin_since = {}      # vid -> time the vehicle cut in (Def. II.3)
        self._emergency = {}        # vid -> (t_engaged, a_engaged)

    def _command_from_jerk(self, a, u0, dt_cmd):
        return min(self.cfg.a_max, max(self.cfg.a_min, float(a) + float(u0) * float(dt_cmd)))

    def _d_safe(self, v, a, v_p, dt_cmd):
        """Safe distance at the current state: this fork's standoff of the ego's
        speed, or the paper's Prop. IV.5 distance with ``paperdsafe``."""
        if self.cfg.paper_d_safe:
            return safe_distance(v, a, v_p, self.cfg, dt_cmd)
        return standoff(v, self.cfg)

    def _next_state_unsafe(self, gap, v, a, v_p, u0, dt_cmd):
        """Condition (7) fails under the planned jerk ``u0``: one command interval
        on, the gap is below the safe distance of the state reached (the paper:
        its fail-safe no longer stops behind the fully braking lead; the fork:
        the standoff of the next speed, the lead at constant speed)."""
        cfg = self.cfg
        if cfg.paper_d_safe:
            return gap < safe_distance(v, a, v_p, cfg, dt_cmd, u0=u0)
        s1, v1, _ = _advance(float(v), float(a), float(u0), float(dt_cmd))
        return gap + float(v_p) * float(dt_cmd) - s1 < standoff(max(0.0, v1), cfg)

    # ---- Sec. IV: safety-keeping controller w.r.t. one vehicle
    def _safety_keeping(self, t, v, a, veh, dt_cmd):
        cfg = self.cfg
        N = self.N_keep
        d_ref = self._d_safe(v, a, veh.speed, dt_cmd)
        if cfg.paper_d_safe:
            # (3) with d_safe of the zero-jerk propagation: one constant
            ds_min, ds_dv_coef = np.full(N, d_ref), 0.0
        else:
            # (3) per step, linear in the ego's speed v_k = v_p - dv_k:
            # ds_k >= d0 + tau v_k  <=>  ds_k + tau dv_k >= d0 + tau v_p
            ds_min, ds_dv_coef = np.full(N, standoff(veh.speed, cfg)), cfg.time_headway
        feasible, U, X = self._qp_keep.solve(
            [veh.gap, veh.speed - v, a], d_ref,
            ds_min=ds_min, ds_dv_coef=ds_dv_coef,
            dv_lo=np.full(N, veh.speed - cfg.v_max), dv_hi=np.full(N, veh.speed - cfg.v_min),
            a_lo=cfg.a_min, a_hi=cfg.a_max)
        unsafe = True
        a_nom = None
        if feasible:
            a_nom = self._command_from_jerk(a, U[0], dt_cmd)
            unsafe = self._next_state_unsafe(veh.gap, v, a, veh.speed, U[0], dt_cmd)
        if not unsafe:
            self._emergency.pop(veh.vid, None)
            return VehicleDecision(PATH_NOMINAL, a_nom, d_ref, veh.gap, veh.speed, True,
                                   {"u0": float(U[0]), "ds_end": float(X[-3])})
        if veh.vid not in self._emergency:
            self._emergency[veh.vid] = (float(t), float(a))
        t_e, a_e = self._emergency[veh.vid]
        a_cmd = emergency_accel(float(t) - t_e + dt_cmd, a_e, cfg)
        return VehicleDecision(PATH_EMERGENCY, a_cmd, d_ref, veh.gap, veh.speed, False,
                               {"nominal_feasible": float(feasible), "t_engaged": t_e,
                                "a_nominal": np.nan if a_nom is None else a_nom})

    # ---- Sec. V: safety-recapturing controller w.r.t. one cut-in vehicle
    def _safety_recapturing(self, t, v, a, veh, dt_cmd):
        cfg = self.cfg
        d_now = self._d_safe(v, a, veh.speed, dt_cmd)
        if is_inevitable_collision(veh.gap, v, a, veh.speed, cfg):
            return VehicleDecision(PATH_ICS, self._command_from_jerk(a, cfg.jerk_min, dt_cmd),
                                   d_now, veh.gap, veh.speed, False, {"reason": 0.0})
        elapsed = float(t) - self._cutin_since[veh.vid]
        remaining = max(cfg.t_clear_s - elapsed, cfg.dt)
        N = self.N_recap
        k_c = min(N, max(1, int(math.ceil(remaining / cfg.dt - 1.0e-9))))   # deadline step
        # Lead braking at a_cutin until standstill (Prop. V.1 assumption).
        steps = np.arange(1, N + 1) * cfg.dt
        v_p_k = np.maximum(0.0, veh.speed + cfg.a_cutin * steps)
        a_lead = np.where(v_p_k > 0.0, cfg.a_cutin, 0.0)
        # Prop. V.1 terminal constraint ds(t_c) >= d_safe(x(t_c), v_p(t_c)),
        # linearised in the ego's terminal speed around the lead's speed at t_c
        # (exact for the fork's standoff, which is linear in that speed).
        v_p_tc = float(v_p_k[k_c - 1])
        d_tc = lambda vv: self._d_safe(max(0.0, vv), 0.0, v_p_tc, dt_cmd)
        d0 = d_tc(v_p_tc)
        lo, hi = max(0.0, v_p_tc - 0.5), v_p_tc + 0.5
        slope = (d_tc(hi) - d_tc(lo)) / (hi - lo)
        # ds_k >= d0 + slope (v_k - v_p_tc), v_k = v_p(t_c) - dv_k  ->  ds_k + slope dv_k >= d0
        terminal = (k_c, slope, d0)
        x0 = [veh.gap, veh.speed - v, a]
        for qp, path in ((self._qp_recap, PATH_RECAPTURE), (self._qp_bounded, PATH_BOUNDED)):
            feasible, U, X = qp.solve(
                x0, d_now, ds_min=np.full(N, -np.inf),
                dv_lo=v_p_k - cfg.v_max, dv_hi=v_p_k - cfg.v_min,
                a_lo=cfg.a_min, a_hi=cfg.a_max, a_lead=a_lead, terminal=terminal)
            if feasible:
                i = 3 * (k_c - 1)
                return VehicleDecision(path, self._command_from_jerk(a, U[0], dt_cmd), d_now,
                                       veh.gap, veh.speed, path == PATH_RECAPTURE,
                                       {"u0": float(U[0]), "t_remaining": remaining, "d_tc": d0,
                                        "v_p_tc": v_p_tc, "ds_tc": float(X[i]), "v_tc": float(v_p_tc - X[i + 1])})
        # Not in the paper: the clearing time cannot be met within the
        # acceleration bounds (t~_c > t_c).  Full braking, as in the ICS.
        return VehicleDecision(PATH_ICS, self._command_from_jerk(a, cfg.jerk_min, dt_cmd),
                               d_now, veh.gap, veh.speed, False, {"reason": 1.0, "t_remaining": remaining})

    # ---- Alg. 1
    def step(self, t, v, a, v_set, vehicles: Sequence[VehicleObs], dt_cmd):
        cfg = self.cfg
        t = float(t)
        seen = {veh.vid for veh in vehicles}
        for vid in list(self._was_member):
            if vid not in seen:
                self._was_member.pop(vid, None)
                self._cutin_since.pop(vid, None)
                self._emergency.pop(vid, None)
        members = []
        for veh in sorted(vehicles, key=lambda item: item.gap):
            # A vehicle first seen inside the lane has not entered it (Def. II.3
            # needs it outside just before): it is a safety-keeping vehicle.
            was = self._was_member.get(veh.vid, bool(veh.member))
            self._was_member[veh.vid] = bool(veh.member)
            if not veh.member:
                self._cutin_since.pop(veh.vid, None)
                self._emergency.pop(veh.vid, None)
                continue
            d_safe = self._d_safe(v, a, veh.speed, dt_cmd)
            if not was and veh.gap < d_safe:
                # Def. II.3: entered the ego lane inside the safe distance
                self._cutin_since.setdefault(veh.vid, t)
            elif veh.vid in self._cutin_since and veh.gap >= d_safe:
                # safe distance restored -> back to safety keeping (Fig. 2)
                self._cutin_since.pop(veh.vid, None)
            members.append(veh)
        recap = [veh for veh in members if veh.vid in self._cutin_since]
        keep = relevant_vehicles([veh for veh in members if veh.vid not in self._cutin_since],
                                 v, a, cfg, dt_cmd)
        per_vehicle = {}
        for veh in keep:
            per_vehicle[veh.vid] = self._safety_keeping(t, v, a, veh, dt_cmd)
        for veh in recap:
            per_vehicle[veh.vid] = self._safety_recapturing(t, v, a, veh, dt_cmd)
        a_cruise = min(cfg.a_max, max(cfg.a_min, cfg.cruise_gain * (float(v_set) - float(v))))
        accel, path, lead_vid, feasible = a_cruise, PATH_CRUISE, None, True
        for vid, dec in per_vehicle.items():
            if dec.accel < accel:
                accel, path, lead_vid, feasible = dec.accel, dec.path, vid, dec.feasible
        return Decision(float(accel), path, bool(feasible), lead_vid, float(a_cruise), per_vehicle,
                        [veh.vid for veh in keep], [veh.vid for veh in recap],
                        jerk_limited=path != PATH_BOUNDED)
