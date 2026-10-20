"""Zhou, Olofsson and Frisk's ISA-MPC (IEEE T-IV 2023) as a cruise controller.

A second thesis baseline next to the Nair-style controllers: the interaction-
and safety-aware MPC of ``interaction-safety-aware-motion-planning/Implementation/
CASE_1_ISAMPC_SIM/ISA_MPC.py`` restricted to lane keeping, fed by the fork's
IAIMM-KF port (``predictor/iaimm_kf``) exactly as the ``iaimm`` policies use it.

What of their planner survives when the lateral motion is frozen (the ego keeps
its lane; the manoeuvre candidates, the lane-tracking cost terms, ``Define_DV``'s
two-lane merge and the steering states go away):

1. ``SafetyAwareOccupancy`` / ``GMM_Model``: per target and step a Gaussian
   mixture over the predicted modes (weights p_j, diagonal variances from the
   IAIMM-KF), each component normalised to peak 1, cut at ``epsilon`` times the
   mixture's maximum (eps = 0.8 in CASE_1/2; eps = 1 is their deterministic MPC);
   the level set's bounding box, grown by half a vehicle, is the obstacle.  A mode
   whose weight is below eps times the dominant one leaves the box entirely, so
   this is a probability-gated worst case over modes, not a probability-weighted
   cost and not a policy tree (their ``policy-vs-sequence-optimization`` study is
   a separate MATLAB comparison; the planner itself is sequence based).
2. ``ProjectSpeed`` / ``construct_QP``: the reference speed is the largest
   v_ref <= v_des for which the ego's closed-loop velocity tracker (gains
   ``K_Lon_EV``) keeps the safe distance to every occupancy's rear edge at every
   step where the occupancy overlaps the ego laterally.  A scalar QP; solved in
   closed form here (an upper bound per step).
3. ``contruct_MT_MPC``: the longitudinal part of their moving-target MPC -- state
   (s, v, a, j), input snap, cost 0.5 snap^2 + 0.5 a^2 + (v_N - v_ref)^2 +
   0.055 rho^2, the safe distance as a soft constraint with slack rho >= 0, v >= 0.
   Their bicycle model is nonlinear (IPOPT); its longitudinal chain is linear, so
   the problem is a QP and runs through Gurobi like the fork's controllers.

Both headway layers use the fork's standoff instead of Zhou's two headways (the
maintainer's decision for the thesis comparison, 2026-10-19): d_safe(v) = L + d0 +
tau v with L = 4.5 m, d0 = 3 m, tau = 1.3 s (``NairACCConfig``) between Frenet
centre positions, which is exactly what ``aggregate_cutin_table.py`` scores delta
against (bumper gap = centre gap - L >= d0 + tau v  <=>  delta = 0).  It replaces
the projection's ``Th_QP`` = 2 s + l_veh/2, which becomes
v_ref <= (x_rear_k - B_k - (L + d0)) / (A_k + tau), and the MPC's soft ``Th_MPC``
= 1.5 s + l_veh/2, which becomes x_rear_k - s_k - (L + d0 + tau v_k) + rho_k >= 0.
x_rear is the level-set box's rear edge without the longitudinal vehicle growth
(zeta_l = 0: L already covers the vehicle, their zeta_l = 0.5 would add l_veh/2 on
top of it); the lateral growth and the lane test are unchanged.

Deviations from their code, all stated in the report: horizon N = 15 at 0.2 s
instead of 25 at 0.32 s (the fork's sweep setting and the predictor's 3 s future);
the fork's acceleration bounds [-3, 2] m/s^2 and 5 m/s^3 jerk limit replace their
|a| <= 6 and unbounded jerk (their formulation has no jerk limit); the level set's
box is read off the thresholded grid cells (their ``skimage.find_contours`` is not
in the environment, difference below one cell of 0.55 x 0.15 m); lane membership of
the box is an interval overlap with the ego lane instead of their three-mark lane
lookup; the ego's lateral position is its measured offset ``ey`` rather than a
lane-tracking rollout; the issued command is the plan's acceleration advanced by
one command interval (0.05 s) along the snap chain, and the (a, j) state of the
next solve is advanced with it -- their simulated EV *is* the model and takes the
planned state at Ts the same way.  (Issuing the first step's mean acceleration and
reading the jerk state back from tick-to-tick command differences was unstable:
a 20 Hz difference is four times the per-step rate the plan expects, and the loop
settled into a +/-3 m/s^2 cycle at the jerk limit, offline and in CARLA.)

``ainput`` (2026-10-20, ``acc_isampc_ainput_rmove1000``) is an appendix ablation:
acceleration as the input with the fork's own jerk penalty, to isolate the effect of
Zhou's snap input; the default snap path above is the baseline.  State (s, v), input
a_k (k = 0..N-1), the fork's double integrator (``acc_dynamics_matrices``).  Zhou's
terms stay -- 0.5 sum a_k^2 + (v_N - v_ref)^2 + 0.055 sum rho^2, the soft standoff
row, the speed projection, epsilon, the occupancy box -- and the snap cost becomes
``NairACCSMPC``'s jerk terms: r_move (a_0 - u_prev)^2 against the issued command
u_prev and r_jerk ((a_k - a_{k-1}) / dt)^2 for k >= 1, with r_jerk and r_move taken
from the parent agent's ``controller_config`` as D-opt gets them (r_jerk 10;
``rmove<w>`` sets r_move, which is otherwise the config's r_jerk / dt^2 = 250 at
dt = 0.2, fixed when the config is built, so ``rjerk<w>`` alone leaves it at 250),
and the fork's hard jerk limit on a_0 - u_prev over the command interval and on
a_k - a_{k-1} over dt.  The command is a_0 through the agent's limiter, as for D-opt.
D-opt's QP sums its whole tracking block, these terms included, over three identical
samples (no K feedback), so its absolute price is 3 r_jerk and 3 r_move against its
5000 slack weight; here they enter once.
"""
from __future__ import annotations

import math
import os
import sys
import time

import casadi as ca
import numpy as np

scriptdir = os.path.abspath(__file__).split('carla')[0] + 'carla/'
sys.path.append(scriptdir)
rootdir = os.path.abspath(__file__).split('scripts')[0]
sys.path.append(rootdir)
from policies.acc_nair_smpc_agent import ACCNairSMPCAgent  # noqa: E402  (also sets GUROBI_VERSION)
from utils.acc_nair_smpc import NairACCConfig, acc_dynamics_matrices  # noqa: E402
from predictor.stdan_3int_signed_tcross_velint.acc_postprocess import normalize_probabilities  # noqa: E402

# Zhou et al., CASE_1_ISAMPC_SIM/main.ipynb (opts_EV) -- CASE_3 sweeps epsilon over
# 0.1 / 0.4 / 1 and CASE_4 (highD replay) uses epsilon 0.1, Th 1 s / 0.5 s.
ZHOU_EPSILON = 0.8
ZHOU_L_VEH = 4.3            # their Th_QP = 2 s / Th_MPC = 1.5 s headways are replaced by the fork's standoff
ZHOU_W_VEH = 1.8
ZHOU_ZETA_L = 0.5           # occupancy grows by zeta * vehicle size; the planner uses zeta_l = 0 (L covers it)
ZHOU_ZETA_W = 0.5
ZHOU_K_LON_EV = (0.1029, 0.3423)
ZHOU_Q_SNAP = 0.5           # Q1
ZHOU_Q_ACCEL = 0.5          # Q3
ZHOU_Q_TERMINAL_V = 1.0     # Q6
ZHOU_Q_SLACK = 0.055        # Q7
GRID_HX = 0.55              # GMM_Model grid
GRID_HY = 0.15
VARSIGMA = 1.0e-4           # added to every variance (SafetyAwareOccupancy)


def lon_tracker_matrices(dt, k_lon=ZHOU_K_LON_EV):
    """Zhou's VelocityTracking longitudinal block: x = (s, v, a), input v_ref."""
    k1, k2 = float(k_lon[0]), float(k_lon[1])
    a = np.array([[1.0, dt, dt ** 2 / 2.0],
                  [0.0, 1.0 - k1 * dt ** 2 / 2.0, dt - k2 * dt ** 2 / 2.0],
                  [0.0, -k1 * dt, 1.0 - k2 * dt]])
    b = np.array([0.0, k1 * dt ** 2 / 2.0, k1 * dt])
    return a, b


def occupancy_box(probs, mu_s, mu_d, var_s, var_d, epsilon, hx=GRID_HX, hy=GRID_HY):
    """Zhou's ``GMM_Model`` for one step.

    Returns (centre_s, centre_d, half_s, half_d, included) of the bounding box of
    {sum_j p_j N_j / max N_j >= epsilon * max} on the 0.55 x 0.15 m grid, and the
    modes whose own peak clears the level.  Half sizes exclude the vehicle.
    """
    probs = np.asarray(probs, dtype=float).reshape(-1)
    modes = [j for j in range(probs.size) if probs[j] > 0.0]
    if not modes:
        raise ValueError("occupancy_box needs at least one mode with positive probability")
    mu_s = np.asarray(mu_s, dtype=float)[modes]
    mu_d = np.asarray(mu_d, dtype=float)[modes]
    sig_s = np.sqrt(np.maximum(np.asarray(var_s, dtype=float)[modes], 0.0) + VARSIGMA)
    sig_d = np.sqrt(np.maximum(np.asarray(var_d, dtype=float)[modes], 0.0) + VARSIGMA)
    xs = np.arange(mu_s.min() - 3.0 * sig_s.max() - hx, mu_s.max() + 3.0 * sig_s.max() + hx, hx)
    ys = np.arange(mu_d.min() - 3.0 * sig_d.max() - hy, mu_d.max() + 3.0 * sig_d.max() + hy, hy)
    dens = np.zeros((xs.size, ys.size))
    peaks = []
    for idx, j in enumerate(modes):
        log_comp = (-0.5 * ((xs - mu_s[idx]) / sig_s[idx]) ** 2)[:, None] \
            + (-0.5 * ((ys - mu_d[idx]) / sig_d[idx]) ** 2)[None, :]
        comp = np.exp(log_comp - log_comp.max())      # Pd / max(Pd), peak exactly 1
        dens += probs[j] * comp
        peaks.append((int(np.argmin(np.abs(xs - mu_s[idx]))), int(np.argmin(np.abs(ys - mu_d[idx])))))
    level = float(epsilon) * float(dens.max())
    mask = dens >= level
    xm = xs[mask.any(axis=1)]
    ym = ys[mask.any(axis=0)]
    included = [j for j, (ix, iy) in zip(modes, peaks) if dens[ix, iy] >= level]
    return (0.5 * float(xm.min() + xm.max()), 0.5 * float(ym.min() + ym.max()),
            0.5 * float(xm.max() - xm.min()), 0.5 * float(ym.max() - ym.min()), included)


class ISAMPCPlanner:
    """Cruise-only ISA-MPC: occupancy -> reference-speed projection -> moving-target QP."""

    def __init__(self, horizon, dt, v_des, a_min=-3.0, a_max=2.0, jerk_limit=5.0,
                 v_min=0.0, v_max=20.0, lane_half_width=1.75, epsilon=ZHOU_EPSILON,
                 vehicle_length=NairACCConfig.vehicle_length, d0=NairACCConfig.d0,
                 time_headway=NairACCConfig.time_headway, l_veh=ZHOU_L_VEH, w_veh=ZHOU_W_VEH,
                 zeta_l=0.0, zeta_w=ZHOU_ZETA_W, k_lon=ZHOU_K_LON_EV,
                 q_snap=ZHOU_Q_SNAP, q_accel=ZHOU_Q_ACCEL, q_terminal_v=ZHOU_Q_TERMINAL_V,
                 q_slack=ZHOU_Q_SLACK, gurobi_time_limit=0.15, solver_tol=1.0e-4,
                 accel_input=False, r_jerk=None, r_move=None):
        self.N = int(horizon)
        self.dt = float(dt)
        self.v_des = float(v_des)
        self.a_min, self.a_max = float(a_min), float(a_max)
        self.jerk_limit = None if jerk_limit is None else float(jerk_limit)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.lane_half_width = float(lane_half_width)
        self.epsilon = float(epsilon)
        # the fork's standoff d_safe(v) = L + d0 + tau v, in both headway layers
        self.vehicle_length, self.d0, self.time_headway = float(vehicle_length), float(d0), float(time_headway)
        self.l_veh, self.w_veh = float(l_veh), float(w_veh)
        self.zeta_l, self.zeta_w = float(zeta_l), float(zeta_w)
        self.weights = {"q_snap": float(q_snap), "q_accel": float(q_accel),
                        "q_terminal_v": float(q_terminal_v), "q_slack": float(q_slack)}
        # ``ainput``: acceleration is the input and the fork's jerk terms replace the snap cost
        self.accel_input = bool(accel_input)
        if self.accel_input:
            del self.weights["q_snap"]
            self.weights.update({"r_jerk": float(r_jerk), "r_move": float(r_move)})
        self._lon_a, self._lon_b = lon_tracker_matrices(self.dt, k_lon)
        self._gurobi_time_limit = gurobi_time_limit
        self._solver_tol = float(solver_tol)
        if self.accel_input:
            self._build_accel_qp()
        else:
            self._build_qp()

    # ------------------------------------------------------------------ occupancy
    def target_occupancy(self, target):
        """Per-step box of one target: dict of arrays over steps 1..N."""
        probs = np.asarray(target["probs"], dtype=float)
        s, d = np.asarray(target["s"], dtype=float), np.asarray(target["d"], dtype=float)
        var_s, var_d = np.asarray(target["var_s"], dtype=float), np.asarray(target["var_d"], dtype=float)
        x_bar, y_bar, dx, dy = (np.zeros(self.N) for _ in range(4))
        included = []
        for k in range(self.N):
            xb, yb, hs, hd, inc = occupancy_box(probs, s[:, k], d[:, k], var_s[:, k], var_d[:, k], self.epsilon)
            x_bar[k], y_bar[k] = xb, yb
            dx[k] = hs + self.zeta_l * self.l_veh
            dy[k] = hd + self.zeta_w * self.w_veh
            included.append(inc)
        in_ego_lane = (y_bar - dy <= self.lane_half_width) & (y_bar + dy >= -self.lane_half_width)
        return {"vehicle_id": target.get("vehicle_id"), "x_bar": x_bar, "y_bar": y_bar,
                "dx": dx, "dy": dy, "in_ego_lane": in_ego_lane, "included_modes": included}

    # ------------------------------------------------------------------ reference speed
    def project_speed(self, s, v, a, d_ego, occupancies):
        """Zhou's construct_QP in closed form: v_ref = max v <= v_des that keeps the
        fork's standoff L + d0 + tau v to every laterally overlapping occupancy's
        rear edge under the closed-loop tracker started at (s, v, a):
        x_rear_k - (A_k v + B_k) >= L + d0 + tau v."""
        x0 = np.array([s, v, a], dtype=float)
        A, B = np.zeros(self.N), np.zeros(self.N)
        acc, apow = np.zeros(3), np.eye(3)
        for k in range(self.N):
            acc = acc + apow @ self._lon_b
            apow = apow @ self._lon_a
            A[k] = acc[0]
            B[k] = (apow @ x0)[0]
        v_up, binding = self.v_des, None
        for occ in occupancies:
            sel = np.abs(float(d_ego) - occ["y_bar"]) < self.w_veh / 2.0 + occ["dy"]
            bound = (occ["x_bar"] - occ["dx"] - B - (self.vehicle_length + self.d0)) / (A + self.time_headway)
            for k in np.nonzero(sel)[0]:
                if bound[k] < v_up:
                    v_up, binding = float(bound[k]), (occ["vehicle_id"], int(k) + 1)
        return max(0.0, v_up), v_up, binding

    # ------------------------------------------------------------------ moving-target QP
    def _build_qp(self):
        N, T = self.N, self.dt
        opti = ca.Opti("conic")
        X = opti.variable(4, N + 1)          # s, v, a, j
        U = opti.variable(N)                 # snap
        RHO = opti.variable(N)
        X0 = opti.parameter(4)
        VREF = opti.parameter()
        XDV = opti.parameter(N)              # occupancy rear edge, steps 1..N
        opti.subject_to(X[:, 0] == X0)
        for k in range(N):
            # exact discretisation of the (s, v, a, j) chain under piecewise-constant
            # snap (their RK4 of this linear chain is exact); written row by row so no
            # zero coefficients reach Gurobi
            sk, vk, ak, jk = X[0, k], X[1, k], X[2, k], X[3, k]
            opti.subject_to(X[0, k + 1] == sk + T * vk + T ** 2 / 2.0 * ak + T ** 3 / 6.0 * jk + T ** 4 / 24.0 * U[k])
            opti.subject_to(X[1, k + 1] == vk + T * ak + T ** 2 / 2.0 * jk + T ** 3 / 6.0 * U[k])
            opti.subject_to(X[2, k + 1] == ak + T * jk + T ** 2 / 2.0 * U[k])
            opti.subject_to(X[3, k + 1] == jk + T * U[k])
        s, v, a, j = X[0, 1:], X[1, 1:], X[2, 1:], X[3, 1:]
        opti.subject_to(opti.bounded(self.v_min, v, self.v_max))
        opti.subject_to(opti.bounded(self.a_min, a, self.a_max))
        if self.jerk_limit is not None:
            opti.subject_to(opti.bounded(-self.jerk_limit, j, self.jerk_limit))
        opti.subject_to(v * self.time_headway - RHO.T <= XDV.T - s - (self.vehicle_length + self.d0))
        opti.subject_to(RHO >= 0.0)
        w = self.weights
        cost = (w["q_snap"] * ca.sumsqr(U) + w["q_accel"] * ca.sumsqr(a)
                + w["q_terminal_v"] * (X[1, N] - VREF) ** 2 + w["q_slack"] * ca.sumsqr(RHO))
        opti.minimize(cost)
        self._set_gurobi(opti)
        self._qp = {"opti": opti, "X": X, "U": U, "RHO": RHO, "X0": X0, "VREF": VREF, "XDV": XDV, "cost": cost}

    def _build_accel_qp(self):
        """``ainput``: the QP above with acceleration as the input and the fork's jerk terms.

        Dynamics are ``acc_dynamics_matrices``; the jerk terms and limits are
        ``NairACCSMPC``'s (cached feedback QP, ``_get_open_loop_fixed_qp``,
        ``_add_symbolic_jerk_constraints``): step 0 is priced r_move (a_0 - u_prev)^2
        against the issued command and bounded over the command interval, step k >= 1
        is priced r_jerk ((a_k - a_{k-1}) / dt)^2 and bounded over dt."""
        N, T = self.N, self.dt
        a_dyn, b_dyn = acc_dynamics_matrices(T)
        opti = ca.Opti("conic")
        X = opti.variable(2, N + 1)          # s, v
        U = opti.variable(N)                 # acceleration a_0 .. a_{N-1}
        RHO = opti.variable(N)
        X0 = opti.parameter(2)
        VREF = opti.parameter()
        XDV = opti.parameter(N)              # occupancy rear edge, steps 1..N
        UPREV = opti.parameter()             # the issued command the plan continues from
        CMD_DT = opti.parameter()            # the interval until a_0 is issued
        opti.subject_to(X[:, 0] == X0)
        for k in range(N):
            # x_{k+1} = A x_k + B a_k with A = [[1, dt], [0, 1]], B = [dt^2 / 2, dt]; row by
            # row so no zero coefficients reach Gurobi
            opti.subject_to(X[0, k + 1] == X[0, k] + float(a_dyn[0, 1]) * X[1, k] + float(b_dyn[0, 0]) * U[k])
            opti.subject_to(X[1, k + 1] == X[1, k] + float(b_dyn[1, 0]) * U[k])
        s, v = X[0, 1:], X[1, 1:]
        opti.subject_to(opti.bounded(self.v_min, v, self.v_max))
        opti.subject_to(opti.bounded(self.a_min, U, self.a_max))
        if self.jerk_limit is not None:
            opti.subject_to(U[0] - UPREV <= self.jerk_limit * CMD_DT)
            opti.subject_to(UPREV - U[0] <= self.jerk_limit * CMD_DT)
            opti.subject_to(opti.bounded(-self.jerk_limit * T, U[1:] - U[:-1], self.jerk_limit * T))
        opti.subject_to(v * self.time_headway - RHO.T <= XDV.T - s - (self.vehicle_length + self.d0))
        opti.subject_to(RHO >= 0.0)
        w = self.weights
        cost = (w["q_accel"] * ca.sumsqr(U) + w["q_terminal_v"] * (X[1, N] - VREF) ** 2
                + w["q_slack"] * ca.sumsqr(RHO)
                + w["r_move"] * (U[0] - UPREV) ** 2 + w["r_jerk"] * ca.sumsqr((U[1:] - U[:-1]) / T))
        opti.minimize(cost)
        self._set_gurobi(opti)
        self._qp = {"opti": opti, "X": X, "U": U, "RHO": RHO, "X0": X0, "VREF": VREF, "XDV": XDV,
                    "UPREV": UPREV, "CMD_DT": CMD_DT, "cost": cost}

    def _set_gurobi(self, opti):
        s_opts = {"OutputFlag": 0, "FeasibilityTol": self._solver_tol, "OptimalityTol": self._solver_tol,
                  "BarConvTol": self._solver_tol, "BarQCPConvTol": self._solver_tol,
                  "PSDTol": max(self._solver_tol, 1.0e-9)}
        if self._gurobi_time_limit is not None:
            s_opts["TimeLimit"] = float(self._gurobi_time_limit)
        opti.solver("gurobi", {"error_on_fail": 0}, s_opts)

    def solve_mpc(self, x0, v_ref, x_dv, u_prev=None, command_dt=None):
        q = self._qp
        q["opti"].set_value(q["X0"], np.asarray(x0, dtype=float))
        q["opti"].set_value(q["VREF"], float(v_ref))
        q["opti"].set_value(q["XDV"], np.asarray(x_dv, dtype=float))
        if self.accel_input:
            q["opti"].set_value(q["UPREV"], float(u_prev))
            q["opti"].set_value(q["CMD_DT"], float(command_dt))
        try:
            sol = q["opti"].solve()
        except Exception as exc:  # Gurobi failure -> the caller falls back
            return None, "gurobi_failed: %r" % (exc,)
        status = str(q["opti"].return_status())
        return ({"X": np.asarray(sol.value(q["X"]), dtype=float).reshape(-1, self.N + 1),
                 "U": np.asarray(sol.value(q["U"]), dtype=float).reshape(-1),
                 "rho": np.asarray(sol.value(q["RHO"]), dtype=float).reshape(-1),
                 "cost": float(sol.value(q["cost"])), "status": status},
                "gurobi_return_status=%s" % status)

    # ------------------------------------------------------------------ one tick
    def plan(self, s, v, a, j, d_ego, targets, command_dt=None):
        """targets: dicts with s0, probs (M,), s/d/var_s/var_d (M, N) over steps 1..N.

        ``accel_raw`` is the planned acceleration ``command_dt`` (default: one MPC
        step) into the plan and ``jerk_next`` the jerk state there; the caller
        feeds both back as the next solve's (a, j).  With ``accel_input`` ``a`` is
        the issued command the plan continues from (clipped to the bounds, as
        ``NairACCSMPC._set_command_context`` does), ``j`` is unused and
        ``accel_raw`` is the first planned acceleration a_0."""
        tau = self.dt if command_dt is None else max(float(command_dt), 1.0e-6)
        t0 = time.time()
        occupancies = [self.target_occupancy(t) for t in targets if float(t["s0"]) >= float(s)]
        x_dv = np.full(self.N, np.inf)
        for occ in occupancies:
            rear = occ["x_bar"] - occ["dx"]
            x_dv = np.where(occ["in_ego_lane"], np.minimum(x_dv, rear), x_dv)
        t1 = time.time()
        v_ref, v_up, binding = self.project_speed(s, v, a, d_ego, occupancies)
        t2 = time.time()
        x_dv_qp = np.where(np.isfinite(x_dv), x_dv, float(s) + 1.0e4)
        if self.accel_input:
            sol, message = self.solve_mpc([s, v], v_ref, x_dv_qp,
                                          u_prev=float(np.clip(a, self.a_min, self.a_max)), command_dt=tau)
        else:
            sol, message = self.solve_mpc([s, v, a, j], v_ref, x_dv_qp)
        t3 = time.time()
        out = {"input": "accel" if self.accel_input else "snap",
               "v_ref": float(v_ref), "v_up_raw": float(v_up), "v_pri": self.v_des, "binding": binding,
               "x_dv": [None if not np.isfinite(x) else float(x) for x in x_dv],
               "occupancies": occupancies, "solver_message": message,
               "timing": {"occupancy": t1 - t0, "projection": t2 - t1, "qp": t3 - t2}}
        if sol is None:
            # rule-based: approach the projected speed, the agent's limiter rates it
            accel = float(np.clip(0.5 * (v_ref - v), self.a_min, self.a_max))
            out.update({"accel_raw": accel, "jerk_next": self._clip_jerk((accel - a) / tau),
                        "status": "fallback", "solve_path": "fallback", "feasible": False,
                        "X": None, "U": None, "rho": None, "cost": float("nan"), "min_gap": float("nan")})
            return out
        X, U = sol["X"], sol["U"]
        margin = x_dv - X[0, 1:] - (self.vehicle_length + self.d0) - self.time_headway * X[1, 1:]
        finite = np.isfinite(margin)
        if self.accel_input:
            # a_0 is the command, as for the fork's controllers; no snap chain to advance
            accel_raw, jerk_next = float(U[0]), self._clip_jerk((U[0] - a) / tau)
        else:
            accel_raw, jerk_next = float(a + j * tau + 0.5 * U[0] * tau ** 2), self._clip_jerk(j + U[0] * tau)
        out.update({"accel_raw": accel_raw,
                    "jerk_next": jerk_next,
                    "status": sol["status"], "solve_path": "isampc_gurobi_qp",
                    "feasible": sol["status"].upper() == "OPTIMAL",
                    "X": X, "U": sol["U"], "rho": sol["rho"], "cost": sol["cost"],
                    "min_gap": float(np.min(margin[finite])) if finite.any() else float("nan")})
        return out

    def _clip_jerk(self, jerk):
        if self.jerk_limit is None:
            return float(jerk)
        return float(np.clip(jerk, -self.jerk_limit, self.jerk_limit))


def isampc_targets_from_bundle(bundle, horizon, controller_dt, sample_future):
    """Planner targets from the IAIMM-KF adapter's ``predict_acc`` output.

    Per target the raw manoeuvre modes (LK / LLC / RLC) with their probabilities,
    Frenet means over steps 1..N and the position variances sampled at the same
    steps (``sample_future`` is the adapter's ``_sample_future``)."""
    targets = []
    if bundle is None:
        return targets
    for target in bundle.get("processed_targets", []):
        raw = target.raw_prediction or {}
        traj = np.asarray(raw.get("pred_traj_frenet", []), dtype=float)
        if traj.ndim != 3 or traj.shape[1] < int(horizon) or not target.mode_predictions:
            continue
        probs = normalize_probabilities(raw.get("raw_intention_prob", np.ones(traj.shape[0])))
        cov = raw.get("full_pred_position_cov_frenet")
        if cov is None:
            var_s = var_d = np.zeros((traj.shape[0], int(horizon)))
        else:
            cov = np.asarray(cov, dtype=float)
            var_s = sample_future(np.maximum(cov[:, :, 0, 0], 0.0), horizon, controller_dt)
            var_d = sample_future(np.maximum(cov[:, :, 1, 1], 0.0), horizon, controller_dt)
        current = np.asarray(target.mode_predictions[0].frenet[0], dtype=float)
        targets.append({"vehicle_id": int(target.vehicle_id),
                        "relation": target.relation_to_ego_lane,
                        "s0": float(current[0]), "d0": float(current[1]), "v0": float(current[2]),
                        "probs": probs,
                        "s": traj[:, :int(horizon), 0], "d": traj[:, :int(horizon), 1],
                        "var_s": np.asarray(var_s, dtype=float)[:, :int(horizon)],
                        "var_d": np.asarray(var_d, dtype=float)[:, :int(horizon)]})
    return targets


class ISAMPCAgent(ACCNairSMPCAgent):
    """CARLA ego running the cruise-only ISA-MPC on IAIMM-KF predictions.

    Reuses the ACC agent's lane reference, lane-keeping steering, ideal actuator,
    prediction bundle (IAIMM-KF adapter, trackings, ``stdan_debug`` logging) and
    command limiter; only the planner and the per-step log differ.  The policy
    string is ``acc_isampc``; its predictor is always the IAIMM-KF.  ``ainput``
    selects the acceleration-input ablation QP, priced with the ``r_jerk`` /
    ``r_move`` the parent reads into ``controller_config`` (``rjerk<w>`` /
    ``rmove<w>``), as for D-opt.
    """

    def __init__(self, vehicle, goal_location, nominal_speed_mps=8.0, dt=0.2, N=8,
                 N_modes=2, smpc_config="acc_isampc"):
        super().__init__(vehicle, goal_location, nominal_speed_mps=nominal_speed_mps,
                         dt=dt, N=N, N_modes=N_modes, smpc_config=smpc_config)
        cfg = self.controller_config
        self.planner = ISAMPCPlanner(
            horizon=N, dt=dt, v_des=nominal_speed_mps,
            a_min=cfg.a_min, a_max=cfg.a_max, jerk_limit=cfg.jerk_limit,
            v_min=cfg.v_min, v_max=cfg.v_max, lane_half_width=self.EGO_LANE_HALF_WIDTH_M,
            vehicle_length=cfg.vehicle_length, d0=cfg.d0, time_headway=cfg.time_headway,
            gurobi_time_limit=cfg.gurobi_time_limit,
            accel_input=self._parse_accel_input(smpc_config), r_jerk=cfg.r_jerk, r_move=cfg.r_move)
        self._prev_command_jerk = 0.0

    @staticmethod
    def _parse_predictor_type(smpc_config):
        return "iaimm_kf"

    @staticmethod
    def _parse_accel_input(smpc_config):
        """Read ``ainput`` from the config string (False = Zhou's snap input)."""
        return "ainput" in str(smpc_config)

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

        if self.goal_reached or self._frenet_traj.reached_trajectory_end(s_ego, resolution=5.0):
            self.goal_reached = True
            action = self._limit_accel_command(self.controller_config.a_min, previous_accel_cmd, command_dt)
            v_des = max(0.0, speed + action * self.DT)
            df_des = 0.0
            u0 = np.array([action, df_des])
            next_command_jerk = self.planner._clip_jerk((action - previous_accel_cmd) / max(float(command_dt), 1.0e-6))
        else:
            bundle = self._stdan_prediction_bundle(s_ego, speed)
            debug = {} if bundle is None else dict(bundle.get("debug", {}))
            targets = ([] if debug.get("fallback") else
                       isampc_targets_from_bundle(bundle, self.N, self.DT, self.stdan_predictor._sample_future))
            start = time.time()
            plan = self.planner.plan(s_ego, speed, previous_accel_cmd, self._prev_command_jerk, ey, targets,
                                     command_dt=command_dt)
            solve_time = time.time() - start
            raw_action = float(plan["accel_raw"])
            action = self._limit_accel_command(raw_action, previous_accel_cmd, command_dt)
            next_command_jerk = float(plan["jerk_next"])
            df_des = self._lane_keeping_steer(ey, epsi)
            v_des = np.clip(speed + action * self.DT,
                            self.controller_config.v_min,
                            self.controller_config.v_max)
            u0 = np.array([action, df_des])
            is_feasible = bool(plan["feasible"])
            selected = (np.array([s_ego + 1000.0, 3.5, self.nominal_speed]) if bundle is None
                        else np.asarray(bundle["selected_target_state"], dtype=float))
            prediction = None if bundle is None else bundle.get("prediction")
            actual_accel = self._actual_longitudinal_accel()
            actual_jerk = self._actual_longitudinal_jerk(time_s, actual_accel)
            self.policy_log.append({
                "time_s": time_s,
                "status": plan["status"],
                "feasible": is_feasible,
                "solve_time": solve_time,
                "ego_s": float(s_ego),
                "ego_v": float(speed),
                "accel_cmd_raw": raw_action,
                "accel_cmd": action,
                "command_dt": command_dt,
                "command_jerk": self._command_jerk(action, command_dt),
                "command_jerk_limit": self.controller_config.jerk_limit,
                "actual_accel": actual_accel,
                "actual_jerk": actual_jerk,
                "steer_cmd": float(df_des),
                "target_s": float(selected[0]),
                "target_d": float(selected[1]),
                "target_v": float(selected[2]),
                "mode_names": [] if prediction is None else list(prediction.mode_names or []),
                "mode_probabilities": [] if prediction is None else prediction.probabilities.tolist(),
                "prediction_num_modes": 0 if prediction is None else int(prediction.num_modes),
                "controller_config_num_modes": int(self.controller_config.num_modes),
                "safety_constraint_mode": "isampc_soft_headway",
                "solver_message": plan["solver_message"],
                "solve_path": plan["solve_path"],
                "solve_timing": dict(plan["timing"]),
                "slack_max": float(np.max(plan["rho"])) if plan["rho"] is not None else float("nan"),
                "chance_margin_min": float("nan"),
                "clearance_scale_now": float("nan"),
                "predictor_time": debug.get("predictor_time"),
                "predictor_type": self.predictor_type,
                "ego_route_debug": self._ego_route_debug,
                "stdan_debug": debug,
                "min_gap": plan["min_gap"],
                "acc_reference": self._isampc_reference_debug(plan, s_ego, speed, previous_accel_cmd),
                "isampc": self._isampc_debug(plan, targets),
            })

        self._prev_command_jerk = next_command_jerk
        self._remember_accel_cmd(time_s, action)
        control = self._low_level_control.update(speed, action, v_des, df_des)
        if self.ideal_actuator:
            control.throttle = 0.0
            control.brake = 0.0
            self._ideal_actuator.update(action)
        return control, z0, u0, is_feasible, solve_time

    def get_cut_in_log(self):
        p = self.planner
        return {
            "controller": "acc_isampc",
            "policy": self.smpc_config,
            "predictor_type": self.predictor_type,
            "epsilon": p.epsilon,
            "vehicle_length": p.vehicle_length,
            "d0": p.d0,
            "time_headway": p.time_headway,
            "input": "accel" if p.accel_input else "snap",
            "r_jerk": p.weights.get("r_jerk"),
            "r_move": p.weights.get("r_move"),
            "weights": dict(p.weights),
            "l_veh": p.l_veh,
            "w_veh": p.w_veh,
            "zeta_l": p.zeta_l,
            "accel_bounds": [p.a_min, p.a_max],
            "jerk_limit": p.jerk_limit,
            "cutin_probability_threshold": float(self.cutin_probability_threshold),
            "steps": self.policy_log,
        }

    def _isampc_reference_debug(self, plan, s_ego, speed, previous_accel_cmd):
        """Same keys as the ACC agent's ``acc_reference`` so the plotting tools can read it."""
        n = self.N
        if plan["X"] is None:
            x_nom = np.tile(np.array([[s_ego, speed]]), (n + 1, 1))[None]
            u_nom = np.full((1, n), previous_accel_cmd)
        else:
            x_nom = plan["X"][:2].T[None]
            u_nom = (plan["X"][2, 1:] if plan["input"] == "snap" else plan["U"])[None]
        return {
            "s_ref": x_nom[:, :, 0].round(4).tolist(),
            "v_ref": [[plan["v_ref"]] * (n + 1)],
            "a_ref": [[0.0] * n],
            "prev_u": [previous_accel_cmd] * n,
            "x_nominal": x_nom.round(4).tolist(),
            "u_nominal": u_nom.round(4).tolist(),
        }

    def _isampc_debug(self, plan, targets):
        occ_rows = []
        for occ in plan["occupancies"]:
            occ_rows.append({
                "vehicle_id": occ["vehicle_id"],
                "x_bar": occ["x_bar"].round(3).tolist(),
                "y_bar": occ["y_bar"].round(3).tolist(),
                "dx": occ["dx"].round(3).tolist(),
                "dy": occ["dy"].round(3).tolist(),
                "in_ego_lane": [bool(v) for v in occ["in_ego_lane"]],
                "included_modes": [[int(j) for j in inc] for inc in occ["included_modes"]],
            })
        snap = plan["input"] == "snap"
        return {
            "input": plan["input"],
            "r_jerk": self.planner.weights.get("r_jerk"),
            "r_move": self.planner.weights.get("r_move"),
            "v_ref": plan["v_ref"],
            "v_up_raw": plan["v_up_raw"],
            "v_pri": plan["v_pri"],
            "binding": None if plan["binding"] is None else [plan["binding"][0], plan["binding"][1]],
            "x_dv": plan["x_dv"],
            "targets": [{"vehicle_id": t["vehicle_id"], "relation": t["relation"], "s0": round(t["s0"], 3),
                         "d0": round(t["d0"], 3), "probs": np.asarray(t["probs"]).round(4).tolist()}
                        for t in targets],
            "occupancies": occ_rows,
            "rho": None if plan["rho"] is None else plan["rho"].round(4).tolist(),
            "snap": None if plan["U"] is None or not snap else plan["U"].round(4).tolist(),
            "planned_v": None if plan["X"] is None else plan["X"][1].round(4).tolist(),
            "planned_a": None if plan["X"] is None else (plan["X"][2] if snap else plan["U"]).round(4).tolist(),
            "cost": plan["cost"],
        }
