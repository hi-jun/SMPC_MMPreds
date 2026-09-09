import carla
import os
import sys
import numpy as np

def lane_keeping_steer(ey, epsi):
    """Front steering angle [rad] that holds a lane.

    ``ey`` is the lateral offset to the lane centre (+left) and ``epsi`` the
    heading error, both in the right-handed frame the agents use (y and yaw
    negated from CARLA).  Shared by the ACC ego and the fixed-lane traffic.
    """
    return float(np.clip(-0.15 * ey - 0.8 * epsi, -0.45, 0.45))


class LowLevelControl:
    def __init__(self, vehicle):
        # Control setup and parameters.
        self.control_prev = carla.VehicleControl()
        self.max_steer_angle = np.radians( vehicle.get_physics_control().wheels[0].max_steer_angle )
        self.alpha         = 0.4 # low-pass filter on actuation to simulate first order delay

        # Throttle Parameters
        self.k_v  = 0.9  # P gain on velocity tracking error
        self.thr_ff_map  = np.column_stack(([  2.5,  7.5,  12.5,  17.5],        # speed (m/s) -> steady state throttle
                                            [0.325, 0.45, 0.525, 0.625]))

        # Brake Parameters
        self.brake_accel_thresh = -2.0 # m/s^2, value below which the brake is activated
        self.brake_decel_map  = np.column_stack(([ 1.6,  3.9, 6.8,  7.1, 7.9],  # deceleration (m/s^2) -> steady state throttle (at 12 m/s^2)
                                                 [  0., 0.25, 0.5, 0.75, 1.0]))

    def update(self, v_curr, a_des, v_des, df_des):
        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False

        if a_des > self.brake_accel_thresh:
            control.throttle = self.k_v * (v_des - v_curr) + np.interp(v_des, self.thr_ff_map[:,0], self.thr_ff_map[:,1])
        else:
            control.brake    = np.interp( -a_des, self.brake_decel_map[:,0], self.brake_decel_map[:,1])

        # Simulated actuation delay, also used to avoid high frequency control inputs.
        if control.throttle > 0.0:
            control.throttle = self.alpha * control.throttle + (1. - self.alpha) * self.control_prev.throttle

        elif control.brake > 0.0:
            control.brake    = self.alpha * control.brake    + (1. - self.alpha) * self.control_prev.brake

        # Steering control.  Flipped sign due to Carla LHS convention.

        control.steer    = -df_des / self.max_steer_angle
        control.steer    = self.alpha * control.steer    + (1. - self.alpha) * self.control_prev.steer

        # Clip Carla control to limits.
        control.throttle = np.clip(control.throttle, 0.0, 1.0)
        control.brake    = np.clip(control.brake, 0.0, 1.0)
        control.steer    = np.clip(control.steer, -1.0, 1.0)

        self.control_prev = control

        return control


class IdealLongitudinalActuator:
    """Pins the speed along the heading to the integrated acceleration command.

    Drives the ACC ego (its default since 2026-09-03; the ``carla_actuator``
    policy token keeps the throttle/brake path) and, since 2026-09-07, the
    target vehicles: through the throttle/brake maps above a vehicle never
    holds its planned speed exactly (the maps realise about 1.5x the
    commanded acceleration plus 0.9 m/s^2 at zero command, and the 7 m/s
    FixedLaneSpeedAgent hunted by +/-0.15 m/s), and a predictor reads every
    such ripple as acceleration.

    ``set_target_velocity`` is applied before the physics step, which then
    moves the speed by whatever the tyres and drag do within the tick.  The
    difference between the speed set last tick and the speed measured now is
    that per-tick effect, and the value it had **one tick earlier** is added
    back in advance so the measured speed follows ``v += a*dt``.

    The compensation predicts the *next* tick's loss, so the estimator has to
    match the disturbance it is predicting.  Feeding this tick's loss straight
    back sustains a two-tick alternation once excited, and the two-tick mean
    used until 2026-09-10 only halved it: for a loss ``e_k = e + (-1)^k d``
    the mean is flat at ``e`` while the next tick's loss is ``e - (-1)^k d``,
    so ``d`` is passed through undamped every tick.  The same-phase value
    ``e_{k-1}`` equals ``e_{k+1}`` for that disturbance and cancels it
    exactly.  Numerically, against a +/-0.02 m/s two-tick disturbance with
    0.02 m/s of drag: no compensation 7.98 m/s (drag uncompensated), one-tick
    feedback 0.040 m/s of residual ripple, two-tick mean 0.020, same-phase
    0.000; with no alternating component both the mean and the same-phase
    value hold the speed exactly, and on an ``a = 1 m/s^2`` ramp the
    same-phase value tracks 0.0500 m/s per tick against the mean's 0.0507.
    Measured cause: a cut-in target at 11 m/s still carried a +0.055/-0.028
    two-tick saw-tooth under the mean (sign alternating on 92-100 % of ticks,
    |dv| 0.042 m/s) where the same target at 13 m/s was monotone -- the
    residue of the reference-snap limit cycle that 3a00ce0 fixed at 13 m/s
    but not at 11.  ``08bbaf2`` had answered the same ripple at the predictor
    input with a 0.2 s acceleration feature; that belongs here, at the
    vehicle, not in a feature the model was not trained on.  Only the
    component along the heading is pinned: the lateral velocity is left as
    the tyres made it, so steering turns the vehicle as before.  The callers
    apply no throttle: with the engine driving the wheels a pinned vehicle
    keeps a two-tick speed cycle for good once it is excited (+/-0.06 m/s at
    7 m/s, +/-0.01 at 13, P-term or feed-forward alike), with the engine idle
    the pin converges to the exact speed within 2 s (pin experiment,
    2026-09-07).
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self._v_set_prev = None
        self._eaten_prev = 0.0
        self._dt = None

    def _longitudinal_state(self):
        if self._dt is None:
            self._dt = self.vehicle.get_world().get_settings().fixed_delta_seconds or 0.05
        dt = self._dt
        yaw = np.radians(self.vehicle.get_transform().rotation.yaw)
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        velocity = self.vehicle.get_velocity()
        v_xy = np.array([velocity.x, velocity.y])
        v_lon = float(np.dot(v_xy, heading))
        return float(dt), heading, v_lon, v_xy - v_lon * heading

    def update(self, a_des):
        dt, heading, v_lon, v_lat = self._longitudinal_state()
        eaten = 0.0 if self._v_set_prev is None else self._v_set_prev - v_lon
        compensation = self._eaten_prev   # 같은 위상 1스텝 예측: e_{k-1} == e_{k+1}
        self._eaten_prev = eaten
        v_next = max(0.0, v_lon + float(a_des) * dt + compensation)
        target = v_next * heading + v_lat
        self.vehicle.set_target_velocity(carla.Vector3D(x=float(target[0]), y=float(target[1]), z=0.0))
        self._v_set_prev = v_next
        return v_next

    def track_speed(self, v_target, a_max=2.0):
        """Approach ``v_target`` at up to ``a_max``, then hold it exactly."""
        dt, _, v_lon, _ = self._longitudinal_state()
        a_des = np.clip((float(v_target) - v_lon) / dt, -float(a_max), float(a_max))
        return self.update(a_des)
