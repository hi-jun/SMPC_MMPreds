"""PC-ACC's safety layer around the ACC's nominal command (policy token ``safelayer``, 2026-11-02).

Althoff, Maierhofer & Pek, "Provably-Correct and Comfortable Adaptive Cruise Control", T-IV 2021: Alg. 1 with
the nominal controller replaced by this repository's MPC. Their Sec. I states that the nominal controller "can be
replaced without having to have the system re-certified".

Every command interval:

* **Cut-in classification (Def. II.3).** A measured vehicle that joins the ego's path with a gap below the safe
  distance of the current state is a cut-in until the gap reaches that distance again. This includes a vehicle seen
  for the first time already inside the path. Cut-ins are outside the guarantee.
* **Check (7).** For every other member: after the nominal command is held over the interval, the fail-safe braking
  must stop behind the vehicle braking at a_p from now (Prop. IV.4-IV.5).
* **Fail-safe.** If one check fails, the fail-safe is issued (or the nominal command, if it brakes harder). The
  fail-safe starts from the acceleration of the last interval and ends as soon as the nominal command passes again.
* **Full braking (Spec. III.3).** In an inevitable collision state against a cut-in vehicle braking at a_cut-in, the
  layer brakes fully.

Only measured states enter; the predictor and the MPC's feasibility play no role.

**Membership.** The callers give each vehicle's membership in the ego's path from PC-ACC's perception, with its 1 s
leave lag: the CARLA agent through ``policies/pcacc_agent.observe_preceding_vehicles``, the replay through
``PCACCReplay.observe``. A vehicle is in the path if it precedes the ego (its rear bumper is ahead of the ego's front)
and it is either in the same lane, or an adjacent-lane vehicle whose body overlaps the ego's laterally.

This differs from the paper, because a longitudinal controller can only avoid vehicles ahead that overlap it
laterally:
* the paper's lane occupancy is not used;
* the paper's lateral cut-in rule is not used;
* vehicles alongside the ego are not members.

On NGSIM each of these engaged the fail-safe or full braking for vehicles that the longitudinal controller cannot or
need not avoid:
* vehicles touching the marking 0.4-1.4 m beside the ego;
* a slow adjacent queue vehicle;
* vehicles merging alongside the ego (bumper gap down to -12 m). Braking for these turned a pass into a yield and
  created floor breaches (2026-11-02, the first final-set run).

**Guarantee (Prop. IV.5, by induction).** No collision with a path member that joined the path, or was first seen,
outside the safe distance. This holds provided that every member brakes no harder than |a_p|, the vehicle executes
the fail-safe, and the measured states are exact.

**Fail-safe parameters.** Constant jerk -20 m/s^3 from the current acceleration down to -8 m/s^2, then -8. This is
emergency braking, beyond the nominal comfort limits. a_p = -10.5 m/s^2 (Spec. III.5); a_cut-in = -2 m/s^2 (PC-ACC's).
Guaranteed bumper gap at equal speeds and zero acceleration: 4.4 / 7.8 / 11.9 m at 10 / 15 / 20 m/s.
``SafetyLayer(failsafe_decel=, failsafe_jerk=)`` (tokens ``fsdecel<x>`` / ``fsjerk<x>``, 2026-11-11) replace -8 and 20;
the check and the guarantee are then those of that fail-safe (at -8 m/s^2 and 10 m/s^3: 6.3 / 10.6 / 15.8 m).

**Plant model of the interval.** The command is held (zero-order hold), the speed clipped at 0, the position
trapezoidal, as in the NGSIM replay's ideal plant. The fail-safe is issued as the profile's value at the end of each
interval, so the executed braking is never weaker than the profile the check assumed. The requirement adds
|a_em| dt^2 / 8, the most the trapezoidal step can carry the ego past the continuous profile in the tick in which it
stops.

**Close entrants, token ``entrant<b>`` (2026-11-04, final-check item 12).** The vehicles of the cut-in set are close
entrants: they joined the path (or were first seen in it) inside the safe distance, whether or not they changed
lanes. With ``SafetyLayer(entrant_decel=-b)`` they are covered too, under the braking bound a_c = -b: every tick
each one is checked with ``entrant_margin`` (after the nominal command is held, the fail-safe keeps the bumper gap
>= 0 at every time against the entrant braking at a_c until standstill, a trajectory comparison because Prop. IV.4's
stopping-point test needs the lead to brake harder than the ego), the fail-safe is issued when a check fails
(``PATH_ENTRANT``), and the ICS test is dropped (an entrant the fail-safe cannot avoid fails the check for every
command, so the fail-safe is issued as mitigation). Guarantee, by the same induction with a_c in place of a_p: no
collision with a close entrant that the fail-safe could avoid when it entered, as long as it brakes no harder than
a_c while it is an entrant (and no harder than a_p after its gap has reached the safe distance, when it is a checked
member again). Not guaranteed: an entry the fail-safe cannot avoid at the entry tick (``new_unavoidable_vids``),
and vehicles not yet in the path. The exit from the fail-safe is the MPC's recovery (``NairACCConfig.failsafe_recovery``).
"""
import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np

from utils.pcacc import (
    PCACCConfig,
    VehicleObs,
    _time_to_standstill,
    ego_brake_distance,
    emergency_accel,
    is_inevitable_collision,
    lead_brake_distance,
)

FAILSAFE_DECEL = -8.0     # a_em [m/s^2]: the fail-safe's deceleration
FAILSAFE_JERK = 20.0      # |j_em| [m/s^3]: the fail-safe's constant jerk
LEAD_DECEL = -10.5        # a_p [m/s^2]: the hardest braking of an ego-lane member (Spec. III.5)
CUTIN_DECEL = -2.0        # a_cut-in [m/s^2]: a cut-in vehicle's braking in the ICS test (PC-ACC's)

PATH_NOMINAL = "nominal"
PATH_FAILSAFE = "failsafe"
PATH_ICS = "ics"
PATH_ENTRANT = "entrant"  # entrant<b>: the fail-safe because a close entrant's check failed
PATH_CODE = {PATH_NOMINAL: 0, PATH_FAILSAFE: 1, PATH_ICS: 2, PATH_ENTRANT: 3}


def _min_gap_until_the_ego_stops(gap, v_e, a_e, j_e, t_jerk, a_hold, v_l, a_l, t_max=math.inf):
    """The smallest gap over ``[0, t_max]`` between a vehicle ``gap`` ahead, moving at ``v_l`` with the constant
    acceleration ``a_l`` <= 0 until standstill, and the ego from ``(v_e, a_e)`` under the jerk ``j_e`` for
    ``t_jerk`` seconds, then the acceleration ``a_hold`` (jerk 0; ``j_e`` 0 holds ``a_e`` throughout), until the
    ego stands still (after which the gap cannot fall).  Exact: between the breakpoints (the end of the jerk
    phase, either vehicle's standstill) the gap is a polynomial of degree <= 3 in time, whose minimum lies at a
    segment end or at a root of the relative speed."""
    g, v_e, a_e, v_l, t = float(gap), max(0.0, float(v_e)), float(a_e), max(0.0, float(v_l)), 0.0
    j_e, t_jerk, a_hold, a_l = float(j_e), float(t_jerk), float(a_hold), float(a_l)
    if j_e == 0.0:
        t_jerk = math.inf
    g_min = g
    for _ in range(12):
        if v_e <= 0.0 and a_e <= 0.0 and (j_e <= 0.0 or t >= t_jerk) or t >= t_max:
            break
        stop_e = _time_to_standstill(v_e, a_e, j_e if t < t_jerk else 0.0)
        stop_l = (v_l / -a_l) if (a_l < 0.0 and v_l > 0.0) else math.inf
        tau = min([x for x in (stop_e if stop_e is not None else math.inf, stop_l, t_jerk - t, t_max - t)
                   if x > 0.0] + [math.inf])
        if not math.isfinite(tau):
            break
        jerk = j_e if t < t_jerk else 0.0
        accel_l = a_l if v_l > 0.0 else 0.0
        # g(s) = g + v_l s + a_l s^2 / 2 - v_e s - a_e s^2 / 2 - jerk s^3 / 6 on [0, tau]
        coeffs = [-jerk / 6.0, 0.5 * (accel_l - a_e), v_l - v_e, g]
        candidates = [0.0, tau]
        deriv = [-0.5 * jerk, accel_l - a_e, v_l - v_e]
        roots = np.roots(deriv) if abs(deriv[0]) > 1e-12 else (
            np.array([-deriv[2] / deriv[1]]) if abs(deriv[1]) > 1e-12 else np.array([]))
        for root in roots:
            if abs(root.imag) < 1e-9 and 0.0 < root.real < tau:
                candidates.append(float(root.real))
        g_min = min(g_min, min(float(np.polyval(coeffs, c)) for c in candidates))
        g = float(np.polyval(coeffs, tau))
        v_e = max(0.0, v_e + a_e * tau + 0.5 * jerk * tau ** 2)
        a_e = a_e + jerk * tau
        v_l = max(0.0, v_l + accel_l * tau)
        t += tau
        if t >= t_jerk and j_e != 0.0:
            a_e, j_e = a_hold, 0.0
        if v_e <= 0.0 and a_e <= 0.0:
            break
    return g_min


def entrant_margin(gap, v, a_cmd, v_p, a_c, cfg, dt):
    """``entrant<b>``: the margin [m] of the check on a close entrant ``gap`` ahead at ``v_p`` for the command
    ``a_cmd`` held over ``dt``: the smallest bumper gap, at any future time, between the ego (``a_cmd`` held,
    then the fail-safe from the state reached) and the entrant braking at ``a_c`` < 0 from now until standstill,
    minus the stopping tick's |a_em| dt^2 / 8 (as ``held_command_safe_distance``).  A trajectory comparison,
    because the stopping-point test of Prop. IV.4 needs the lead to brake harder than the ego, which ``a_c`` does
    not.  The check passes when the margin is >= 0."""
    v, dt, a_c = max(0.0, float(v)), float(dt), float(a_c)
    a_cmd = min(cfg.a_max, max(cfg.a_min, float(a_cmd)))
    # the interval: the command held (speed clipped at 0), the entrant braking at a_c
    m_interval = _min_gap_until_the_ego_stops(gap, v, a_cmd, 0.0, 0.0, a_cmd, v_p, a_c, t_max=dt)
    v1 = max(0.0, v + a_cmd * dt)
    s1 = 0.5 * (v + v1) * dt
    t_l = min(dt, v_p / -a_c) if v_p > 0.0 else 0.0
    gap1 = float(gap) + float(v_p) * t_l + 0.5 * a_c * t_l ** 2 - s1
    v_p1 = max(0.0, float(v_p) + a_c * dt)
    # then the fail-safe: the jerk j_em from a1 down to a_em, then a_em, until standstill
    t_jerk = max(0.0, (a_cmd - cfg.a_min) / cfg.jerk_max)
    m_failsafe = _min_gap_until_the_ego_stops(gap1, v1, a_cmd, cfg.jerk_min, t_jerk, cfg.a_min, v_p1, a_c)
    return min(m_interval, m_failsafe) - abs(cfg.a_min) * dt * dt / 8.0


def safety_layer_config(a_max, failsafe_decel=None, failsafe_jerk=None):
    """The layer's ``PCACCConfig``. The fail-safe is a constant-jerk profile: no Fig. 6 ramp, so no profile
    fractions. Gaps are physical bumper gaps (``paper_d_safe``). ``a_max`` is the nominal controller's upper
    acceleration bound. ``failsafe_decel`` (< 0) and ``failsafe_jerk`` (> 0), tokens ``fsdecel<x>`` /
    ``fsjerk<x>`` (2026-11-11), replace ``FAILSAFE_DECEL`` and ``FAILSAFE_JERK``; Spec. III.5 (a_p no weaker
    than the fail-safe) still has to hold."""
    a_em = FAILSAFE_DECEL if failsafe_decel is None else float(failsafe_decel)
    j_em = FAILSAFE_JERK if failsafe_jerk is None else float(failsafe_jerk)
    if not (a_em < 0.0 and j_em > 0.0):
        raise ValueError("the fail-safe needs a negative deceleration and a positive jerk (%r, %r)" % (a_em, j_em))
    if LEAD_DECEL > a_em:
        raise ValueError("Spec. III.5 needs a_p <= the fail-safe's deceleration (%.2f > %.2f)" % (LEAD_DECEL, a_em))
    return replace(PCACCConfig(), a_min=a_em, jerk_max=j_em, a_max=float(a_max),
                   a_p_min=LEAD_DECEL, a_cutin=CUTIN_DECEL, jerk_profile_fractions=(), paper_d_safe=True)


def held_command_safe_distance(v, a_cmd, v_p, cfg, dt):
    """Condition (7) on the plant: the bumper gap needed now such that, after ``a_cmd`` is held for ``dt`` (speed
    clipped at 0), the fail-safe stops behind a lead that brakes at a_p from now.

    The last term, |a_em| dt^2 / 8, is the most the replay's trapezoidal step moves the ego past the continuous
    profile in the tick in which it stops: 0.5 v dt against v^2 / (2 |a|), with v < |a| dt. It is 1 cm at
    dt = 0.1 s and 2.5 mm at 0.05 s."""
    v = max(0.0, float(v))
    dt = float(dt)
    v1 = max(0.0, v + float(a_cmd) * dt)
    s1 = 0.5 * (v + v1) * dt
    a1 = min(cfg.a_max, max(cfg.a_min, float(a_cmd)))
    stop_tick = abs(cfg.a_min) * dt * dt / 8.0
    return s1 + ego_brake_distance(v1, a1, cfg) - lead_brake_distance(v_p, cfg) + stop_tick


@dataclass
class SafetyDecision:
    accel: float                  # the command to issue
    path: str                     # PATH_NOMINAL, PATH_FAILSAFE, PATH_ICS or PATH_ENTRANT
    vid: Optional[int]            # the vehicle that set a non-nominal command
    margin: float                 # min over the checked members of gap - required gap for the nominal command [m]
    checked_vids: List[int] = field(default_factory=list)
    cutin_vids: List[int] = field(default_factory=list)
    # entrant<b>: min over the close entrants of their margin for the nominal command (NaN without the token or
    # an entrant); the entrants whose check fails for the fail-safe itself (unavoidable under a_c); the vehicles
    # that became close entrants this tick, and those of them unavoidable at that tick (the uncovered entries)
    entrant_margin: float = math.nan
    unavoidable_vids: List[int] = field(default_factory=list)
    new_entrant_vids: List[int] = field(default_factory=list)
    new_unavoidable_vids: List[int] = field(default_factory=list)


class SafetyLayer:
    """One run's layer: the membership history (Def. II.3), the cut-in set and the running fail-safe.

    ``entrant_decel`` (token ``entrant<b>``, a_c = -b, 2026-11-04): close entrants (the cut-in set) are covered too:
    every tick each one is checked with ``entrant_margin`` (the fail-safe after the held nominal command keeps the
    gap >= 0 against the entrant braking at a_c until standstill), and the fail-safe is issued when a check fails
    (``PATH_ENTRANT``); the ICS test is then not used (an entrant the fail-safe cannot avoid fails the check for
    every command, so the fail-safe is issued as mitigation).  Guarantee, by the induction of Prop. IV.5 with
    a_p replaced by a_c for the entrant: no collision with a close entrant that was avoidable under a_c when it
    entered, as long as it brakes no harder than a_c while it is an entrant (and no harder than a_p once its gap
    has reached the safe distance and it is a checked member).  ``None`` is the layer of 2026-11-02 (entrants
    get the ICS test only).

    ``failsafe_decel`` / ``failsafe_jerk`` (tokens ``fsdecel<x>`` / ``fsjerk<x>``, 2026-11-11): the fail-safe's
    deceleration and constant jerk in place of ``FAILSAFE_DECEL`` / ``FAILSAFE_JERK`` (``safety_layer_config``);
    the MPC must get the same values (``NairACCConfig.failsafe_decel`` / ``failsafe_jerk``)."""

    def __init__(self, a_max, entrant_decel=None, failsafe_decel=None, failsafe_jerk=None):
        self.cfg = safety_layer_config(a_max, failsafe_decel, failsafe_jerk)
        self.entrant_decel = None if entrant_decel is None else float(entrant_decel)
        if self.entrant_decel is not None and self.entrant_decel >= 0.0:
            raise ValueError("entrant_decel must be negative (a braking bound)")
        self._was_member = {}
        self._cutin = set()
        self._new_cutin = []      # the vehicles added to the cut-in set by the last classify
        self._engaged = None      # (time, acceleration) at which the running fail-safe started

    def step(self, t, v, a, a_nominal, vehicles: Sequence[VehicleObs], dt):
        """Return the ``SafetyDecision`` for one command interval: ``classify`` then ``decide``.

        * ``v``: the ego's speed.
        * ``a``: the acceleration of the last interval (the previous command).
        * ``a_nominal``: the nominal controller's command for this interval.
        * ``vehicles``: the measured preceding vehicles with their ego-lane membership
          (``pcacc.LaneMembershipTracker``) and physical bumper gaps.
        """
        checked, cutins = self.classify(t, v, a, vehicles, dt)
        return self.decide(t, v, a, a_nominal, checked, cutins, dt)

    def classify(self, t, v, a, vehicles: Sequence[VehicleObs], dt):
        """Update the membership history and the cut-in set (Def. II.3) for one command interval and return
        ``(checked, cutins)``: the members the guarantee covers, and the cut-ins, both as ``VehicleObs`` lists
        sorted by gap. Arguments as in ``step``. Call it once per interval: a second call advances the
        membership history again (``safebranch`` classifies before the MPC solve and decides after it)."""
        del t                                   # the classification does not depend on the time
        cfg = self.cfg
        v, a, dt = max(0.0, float(v)), float(a), float(dt)
        seen = {veh.vid for veh in vehicles}
        for vid in list(self._was_member):
            if vid not in seen:
                self._was_member.pop(vid, None)
                self._cutin.discard(vid)
        checked, cutins = [], []
        self._new_cutin = []
        for veh in sorted(vehicles, key=lambda item: item.gap):
            # A vehicle first seen inside the path counts as having just entered it: inside the safe distance it is a
            # cut-in like any other (the paper treats it as safety keeping and assumes a safe start).
            was = self._was_member.get(veh.vid, False)
            self._was_member[veh.vid] = bool(veh.member)
            if not veh.member:
                self._cutin.discard(veh.vid)
                continue
            d_now = held_command_safe_distance(v, a, veh.speed, cfg, dt)
            if not was and veh.gap < d_now:
                self._cutin.add(veh.vid)            # Def. II.3: entered the ego lane inside the safe distance
                self._new_cutin.append(veh.vid)
            elif veh.vid in self._cutin and veh.gap >= d_now:
                self._cutin.discard(veh.vid)        # safe distance restored -> covered by the guarantee
            (cutins if veh.vid in self._cutin else checked).append(veh)
        return checked, cutins

    def decide(self, t, v, a, a_nominal, checked, cutins, dt):
        """The ``SafetyDecision`` for one command interval from ``classify``'s members: check (7) on every
        checked member, the fail-safe when one fails, full braking in an ICS against a cut-in. Arguments as
        in ``step``."""
        cfg = self.cfg
        t, v, a, a_nominal, dt = float(t), max(0.0, float(v)), float(a), float(a_nominal), float(dt)
        accel, path, vid = a_nominal, PATH_NOMINAL, None
        margin, worst = math.inf, None
        for veh in checked:
            m = veh.gap - held_command_safe_distance(v, a_nominal, veh.speed, cfg, dt)
            if m < margin:
                margin, worst = m, veh.vid
        # entrant<b>: the close entrants' check for the nominal command, and for the fail-safe's own next command
        # (its failure means the entrant is unavoidable under a_c from this state)
        m_entrant, worst_entrant = math.inf, None
        unavoidable = []
        if self.entrant_decel is not None and cutins:
            t_e, a_e = self._engaged if self._engaged is not None else (t, a)
            a_failsafe = emergency_accel(t - t_e + dt, a_e, cfg)
            for veh in cutins:
                m = entrant_margin(veh.gap, v, a_nominal, veh.speed, self.entrant_decel, cfg, dt)
                if m < m_entrant:
                    m_entrant, worst_entrant = m, veh.vid
                if entrant_margin(veh.gap, v, a_failsafe, veh.speed, self.entrant_decel, cfg, dt) < 0.0:
                    unavoidable.append(veh.vid)
        if margin < 0.0 or m_entrant < 0.0:
            if self._engaged is None:
                self._engaged = (t, a)
            t_e, a_e = self._engaged
            accel = min(a_nominal, emergency_accel(t - t_e + dt, a_e, cfg))
            path, vid = (PATH_FAILSAFE, worst) if margin < 0.0 else (PATH_ENTRANT, worst_entrant)
        else:
            self._engaged = None
        if self.entrant_decel is None:
            for veh in cutins:
                if is_inevitable_collision(veh.gap, v, a, veh.speed, cfg):
                    full = max(cfg.a_min, a + cfg.jerk_min * dt)
                    if full < accel:
                        accel, path, vid = full, PATH_ICS, veh.vid
        return SafetyDecision(float(accel), path, vid, float(margin),
                              [veh.vid for veh in checked], [veh.vid for veh in cutins],
                              entrant_margin=float(m_entrant) if math.isfinite(m_entrant) else math.nan,
                              unavoidable_vids=unavoidable, new_entrant_vids=list(self._new_cutin),
                              new_unavoidable_vids=[vid_ for vid_ in self._new_cutin if vid_ in unavoidable])
