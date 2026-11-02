"""PC-ACC's safety layer around the ACC's nominal command (policy token ``safelayer``, 2026-11-02).

Althoff, Maierhofer & Pek, "Provably-Correct and Comfortable Adaptive Cruise Control", T-IV 2021: Alg. 1 with
the nominal controller replaced by this repository's MPC. Their Sec. I states that the nominal controller "can be
replaced without having to have the system re-certified".

Every command interval:

* **Cut-in classification (Def. II.3).** A measured vehicle that becomes an ego-lane member with a gap below the safe
  distance of the current state is a cut-in until the gap reaches that distance again. Cut-ins are outside the
  guarantee.
* **Check (7).** For every other member: after the nominal command is held over the interval, the fail-safe braking
  must stop behind the vehicle braking at a_p from now (Prop. IV.4-IV.5).
* **Fail-safe.** If one check fails, the fail-safe is issued (or the nominal command, if it brakes harder). The
  fail-safe starts from the acceleration of the last interval and ends as soon as the nominal command passes again.
* **Full braking (Spec. III.3).** In an inevitable collision state against a cut-in vehicle braking at a_cut-in, the
  layer brakes fully.

Only measured states enter; the predictor and the MPC's feasibility play no role.

**Membership.** The callers give each vehicle's membership in the ego's path from PC-ACC's perception, with its 1 s
leave lag: the CARLA agent through ``policies/pcacc_agent.observe_preceding_vehicles``, the replay through
``PCACCReplay.observe``. A vehicle is in the path if it is in the same lane, or if it is an adjacent-lane vehicle whose
body overlaps the ego's laterally.

This differs from the paper in two ways, because a 1-D longitudinal collision needs lateral body overlap:
* the paper's lane occupancy is not used;
* the paper's lateral cut-in rule is not used.

On NGSIM both engaged the fail-safe for vehicles that never entered: vehicles touching the marking 0.4-1.4 m beside
the ego, and a slow adjacent queue vehicle.

**Guarantee (Prop. IV.5, by induction).** No collision with an ego-lane member that did not cut in, provided that:
every member brakes no harder than |a_p|, the vehicle executes the fail-safe, the measured states are exact, and the
layer starts in a safe state.

**Fail-safe parameters.** Constant jerk -20 m/s^3 from the current acceleration down to -8 m/s^2, then -8. This is
emergency braking, beyond the nominal comfort limits. a_p = -10.5 m/s^2 (Spec. III.5); a_cut-in = -2 m/s^2 (PC-ACC's).
Guaranteed bumper gap at equal speeds and zero acceleration: 4.4 / 7.8 / 11.9 m at 10 / 15 / 20 m/s.

**Plant model of the interval.** The command is held (zero-order hold), the speed clipped at 0, the position
trapezoidal, as in the NGSIM replay's ideal plant. The fail-safe is issued as the profile's value at the end of each
interval, so the executed braking is never weaker than the profile the check assumed. The requirement adds
|a_em| dt^2 / 8, the most the trapezoidal step can carry the ego past the continuous profile in the tick in which it
stops.
"""
import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

from utils.pcacc import (
    PCACCConfig,
    VehicleObs,
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
PATH_CODE = {PATH_NOMINAL: 0, PATH_FAILSAFE: 1, PATH_ICS: 2}


def safety_layer_config(a_max):
    """The layer's ``PCACCConfig``. The fail-safe is a constant-jerk profile: no Fig. 6 ramp, so no profile
    fractions. Gaps are physical bumper gaps (``paper_d_safe``). ``a_max`` is the nominal controller's upper
    acceleration bound."""
    return replace(PCACCConfig(), a_min=FAILSAFE_DECEL, jerk_max=FAILSAFE_JERK, a_max=float(a_max),
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
    path: str                     # PATH_NOMINAL, PATH_FAILSAFE or PATH_ICS
    vid: Optional[int]            # the vehicle that set a non-nominal command
    margin: float                 # min over the checked members of gap - required gap for the nominal command [m]
    checked_vids: List[int] = field(default_factory=list)
    cutin_vids: List[int] = field(default_factory=list)


class SafetyLayer:
    """One run's layer: the membership history (Def. II.3), the cut-in set and the running fail-safe."""

    def __init__(self, a_max):
        self.cfg = safety_layer_config(a_max)
        self._was_member = {}
        self._cutin = set()
        self._engaged = None      # (time, acceleration) at which the running fail-safe started

    def step(self, t, v, a, a_nominal, vehicles: Sequence[VehicleObs], dt):
        """Return the ``SafetyDecision`` for one command interval.

        * ``v``: the ego's speed.
        * ``a``: the acceleration of the last interval (the previous command).
        * ``a_nominal``: the nominal controller's command for this interval.
        * ``vehicles``: the measured preceding vehicles with their ego-lane membership
          (``pcacc.LaneMembershipTracker``) and physical bumper gaps.
        """
        cfg = self.cfg
        t, v, a, a_nominal, dt = float(t), max(0.0, float(v)), float(a), float(a_nominal), float(dt)
        seen = {veh.vid for veh in vehicles}
        for vid in list(self._was_member):
            if vid not in seen:
                self._was_member.pop(vid, None)
                self._cutin.discard(vid)
        checked, cutins = [], []
        for veh in sorted(vehicles, key=lambda item: item.gap):
            # A vehicle first seen inside the lane has not entered it (Def. II.3 needs it outside just before).
            was = self._was_member.get(veh.vid, bool(veh.member))
            self._was_member[veh.vid] = bool(veh.member)
            if not veh.member:
                self._cutin.discard(veh.vid)
                continue
            d_now = held_command_safe_distance(v, a, veh.speed, cfg, dt)
            if not was and veh.gap < d_now:
                self._cutin.add(veh.vid)            # Def. II.3: entered the ego lane inside the safe distance
            elif veh.vid in self._cutin and veh.gap >= d_now:
                self._cutin.discard(veh.vid)        # safe distance restored -> covered by the guarantee
            (cutins if veh.vid in self._cutin else checked).append(veh)
        accel, path, vid = a_nominal, PATH_NOMINAL, None
        margin, worst = math.inf, None
        for veh in checked:
            m = veh.gap - held_command_safe_distance(v, a_nominal, veh.speed, cfg, dt)
            if m < margin:
                margin, worst = m, veh.vid
        if margin < 0.0:
            if self._engaged is None:
                self._engaged = (t, a)
            t_e, a_e = self._engaged
            accel = min(a_nominal, emergency_accel(t - t_e + dt, a_e, cfg))
            path, vid = PATH_FAILSAFE, worst
        else:
            self._engaged = None
        for veh in cutins:
            if is_inevitable_collision(veh.gap, v, a, veh.speed, cfg):
                full = max(cfg.a_min, a + cfg.jerk_min * dt)
                if full < accel:
                    accel, path, vid = full, PATH_ICS, veh.vid
        return SafetyDecision(float(accel), path, vid, float(margin),
                              [veh.vid for veh in checked], [veh.vid for veh in cutins])
