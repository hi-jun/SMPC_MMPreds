"""IAIMM-KF predictor (Zhou, Olofsson, Frisk, IEEE T-IV 2023) plugged into the ACC pipeline.

A model-based counterpart of the STDAN predictor: for every target an interacting
multiple-model Kalman filter runs three lane-dependent manoeuvre models -- keep lane,
change left, change right -- each an LQR-style tracker of a reference speed and a lane
centre.  The reference speed of every manoeuvre is projected to be collision-free against
the predicted motion of higher-priority vehicles (the interaction), the manoeuvre
probability mixes the Kalman innovation with a manoeuvre cost, and the trajectory
uncertainty comes from re-simulating each manoeuvre with controller gains sampled from
the sets identified on highD.  Ported from
``interaction-safety-aware-motion-planning/Implementation/CASE_1_ISAMPC_SIM/IAIMM_KF.py``
with three changes: lanes are indexed relative to the ego lane (Frenet ``d``) instead of
a fixed three-lane road, the filter runs at the predictor call period instead of 0.32 s
(transition probabilities and process noise are rescaled accordingly), and the
collision-free speed projection is solved in closed form (only leaders bound the speed
from above) instead of with IPOPT.

The output is the STDAN raw-prediction dict (three "intentions" LK / LLC / RLC, global-xy
trajectories, position covariances), so everything downstream -- Frenet conversion,
lane membership, ACC mode mapping, the alpha relaxation -- is shared with the learned
predictor.  Only ``predict_raw`` differs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import scipy.io as sio

from predictor.stdan_3int_signed_tcross_velint.acc_adapter import STDAN3IntACCAdapter

DEFAULT_PARAMS = (
    Path(__file__).resolve().parents[2]
    / "interaction-safety-aware-motion-planning/Implementation/CASE_1_ISAMPC_SIM/Model_Parameters.mat"
)
MODE_LK, MODE_LLC, MODE_RLC = 0, 1, 2          # order of acc_postprocess.INTENTION_NAMES
LANE_OFFSET = {MODE_LK: 0, MODE_LLC: +1, MODE_RLC: -1}   # +d is left
# Zhou's centre-lane models: m3 keeps lane 2, m4 goes 2 -> 3 (+y), m2 goes 2 -> 1 (-y).
ZHOU_MODE_OF = {MODE_LK: "m3", MODE_LLC: "m4", MODE_RLC: "m2"}
ZHOU_TS = 0.32


def _stack_gains(field, width):
    arr = np.asarray(field)
    if arr.dtype == object:
        return np.array([np.asarray(k, dtype=float).reshape(-1)[:width] for k in arr.ravel()])
    return np.asarray(arr, dtype=float).reshape(-1, width)


def load_zhou_parameters(path):
    struct = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)["Model_Parameters"]
    params = {}
    for mode, name in ZHOU_MODE_OF.items():
        st = getattr(struct, name)
        params[mode] = {
            "lon": np.asarray(st.Lon, dtype=float).reshape(2),
            "lat": np.asarray(st.Lat, dtype=float).reshape(3),
            "kset_lon": _stack_gains(st.K_set_lon, 2),
            "kset_lat": _stack_gains(st.K_set_lat, 3) if hasattr(st, "K_set_lat") else None,
            "std_y": float(np.asarray(st.std_y).reshape(-1)[0]) if hasattr(st, "std_y") else None,
        }
    return params


def _lon_ab(ts, klon):
    a = np.array([[1.0, ts, ts ** 2 / 2],
                  [0.0, 1 - klon[0] * ts ** 2 / 2, ts - klon[1] * ts ** 2 / 2],
                  [0.0, -klon[0] * ts, 1 - klon[1] * ts]])
    b = np.array([0.0, klon[0] * ts ** 2 / 2, klon[0] * ts])
    return a, b


def _lat_ab(ts, klat):
    a = np.array([[1 - klat[0] * ts ** 3 / 6, ts - klat[1] * ts ** 3 / 6, ts ** 2 / 2 - klat[2] * ts ** 3 / 6],
                  [-klat[0] * ts ** 2 / 2, 1 - klat[1] * ts ** 2 / 2, ts - klat[2] * ts ** 2 / 2],
                  [-klat[0] * ts, -klat[1] * ts, 1 - klat[2] * ts]])
    b = np.array([ts ** 3 / 6 * klat[0], ts ** 2 / 2 * klat[0], ts * klat[0]])
    return a, b


def _rollout(x0, ts, klon, klat, v_ref, y_ref, n):
    """Zhou's VelocityTracking: n steps of the 6-state manoeuvre model, returns (6, n+1)."""
    a_lon, b_lon = _lon_ab(ts, klon)
    a_lat, b_lat = _lat_ab(ts, klat)
    out = np.zeros((6, n + 1))
    out[:, 0] = x0
    for k in range(1, n + 1):
        out[:3, k] = a_lon @ out[:3, k - 1] + b_lon * v_ref
        out[3:, k] = a_lat @ out[3:, k - 1] + b_lat * y_ref
    return out


def _f_e(ts, klon, klat, v_ref, y_ref):
    a_lon, b_lon = _lon_ab(ts, klon)
    a_lat, b_lat = _lat_ab(ts, klat)
    F = np.zeros((6, 6)); F[:3, :3] = a_lon; F[3:, 3:] = a_lat
    E = np.concatenate((b_lon * v_ref, b_lat * y_ref))
    return F, E


class IAIMMKFACCAdapter(STDAN3IntACCAdapter):
    """Same call contract as ``STDAN3IntACCAdapter``; the model is replaced by an IAIMM-KF."""

    mode_names = ["LK", "LLC", "RLC"]

    def __init__(self, history=3.0, future=3.0, dt=0.1, call_dt=0.05, lane_width=3.5,
                 params_path=None, k_sampling=30, seed=0, l_veh=4.3, w_veh=1.8, kf_dt=0.3,
                 lat_speedup=1.0, q_lat_scale=1.0, weight=(0.1, 0.3, 0.1, 0.5), trans_eps=0.25,
                 proj_horizon_s=None):
        """Tuning knobs (defaults = Zhou's values):
        lat_speedup   time-scales the lane-change models (k0, k1, k2 -> a^3 k0, a^2 k1, a k2), so a
                      lane change takes 1/a of the highD-identified duration;
        q_lat_scale   multiplies the lateral process noise (smaller = sharper lateral evidence);
        weight        manoeuvre-cost weights on ax^2, ay^2, (v_ref - v)^2, (y_ref - y)^2;
        trans_eps     per-0.32 s probability of switching manoeuvre;
        proj_horizon_s horizon of the collision-free speed projection and manoeuvre cost
                      (Zhou: 8 s); the output trajectories keep ``future``."""
        super().__init__(history=history, future=future, dt=dt, device="cpu", load_model=False, call_dt=call_dt)
        self.ckpt_path = Path(params_path) if params_path is not None else DEFAULT_PARAMS
        self.params = load_zhou_parameters(self.ckpt_path)
        a = float(lat_speedup)
        if a != 1.0:
            for mode in (MODE_LLC, MODE_RLC):
                pr = self.params[mode]
                pr["lat"] = pr["lat"] * np.array([a ** 3, a ** 2, a])
                if pr["kset_lat"] is not None:
                    pr["kset_lat"] = pr["kset_lat"] * np.array([a ** 3, a ** 2, a])
        self.lane_width = float(lane_width)
        self.k_sampling = int(k_sampling)
        self.l_veh = float(l_veh)
        self.w_veh = float(w_veh)
        self.rng = np.random.default_rng(seed)
        # The filter keeps Zhou's cadence (0.32 s; 0.3 s = 6 control periods here).  Updating it
        # every 0.05 s call would shrink the one-step difference between the manoeuvre models
        # by (0.05/0.32)^2 and leave the innovation unable to tell them apart.
        self.kf_dt = float(kf_dt)
        self.kf_every = max(1, int(round(self.kf_dt / float(call_dt))))
        scale = self.kf_dt / ZHOU_TS
        self.Q = np.diag([1.0, 0.5, 0.25, 0.1 * q_lat_scale, 0.1 * q_lat_scale, 0.0]) * scale
        self.R = np.eye(3) * 1e-5
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=float)
        self.weight = np.asarray(weight, dtype=float)       # ax^2, ay^2, (v_ref - v)^2, (y_ref - y)^2
        eps = float(trans_eps) * scale                      # Zhou's centre-lane matrix
        self.trans = np.full((3, 3), eps); np.fill_diagonal(self.trans, 1.0 - 2.0 * eps)
        self.n_proj = self.out_length if proj_horizon_s is None else max(self.out_length, int(round(proj_horizon_s / self.dt)))
        self._filters = {}
        self._ctx = None

    # ------------------------------------------------------------------ pipeline glue
    def predict_acc(self, ego_state, target_states_frenet, target_relations, trackings, frenet_handler,
                    horizon, desired_speed, num_modes, **kwargs):
        self._ctx = (np.asarray(ego_state, dtype=float), dict(target_states_frenet), frenet_handler)
        try:
            return super().predict_acc(ego_state, target_states_frenet, target_relations, trackings,
                                       frenet_handler, horizon, desired_speed, num_modes, **kwargs)
        finally:
            self._ctx = None

    def predict_raw(self, target_ids: Iterable[int], trackings: Dict[int, np.ndarray],
                    model_yaw: Optional[float] = None) -> Dict[int, dict]:
        if self._ctx is None:
            raise RuntimeError("predict_raw must be called through predict_acc")
        ego_state, states, fh = self._ctx
        n, tp = self.n_proj, self.dt
        ids = [int(t) for t in target_ids
               if t in trackings and t in states and self._has_enough_history(trackings[t])]
        meas = {t: np.array([states[t][0], states[t][2], states[t][1]], dtype=float) for t in ids}  # s, v, d
        steps = tp * np.arange(n + 1)
        cv = lambda s, v, d: np.stack((s + v * steps, np.full(n + 1, d)), axis=-1)
        cars = {"ego": (ego_state[0], ego_state[1], 0.0)}
        cars.update({t: (meas[t][0], meas[t][1], meas[t][2]) for t in ids})
        # Every other tracked vehicle (traffic, blockers) is an obstacle with a constant-velocity
        # prediction, as all vehicles are in Zhou's priority list; the ego appears in trackings
        # too and is recognised by its position.
        for vid, hist in trackings.items():
            if vid in ids:
                continue
            h = np.asarray(hist, dtype=float)
            if h.shape[0] < 2:
                continue
            s1, d1, _ = fh.convert_global_to_frenet_frame(float(h[-1, 0]), float(h[-1, 1]), float(h[-1, 2]))
            s0, _, _ = fh.convert_global_to_frenet_frame(float(h[-2, 0]), float(h[-2, 1]), float(h[-2, 2]))
            if abs(s1 - ego_state[0]) < 2.0 and abs(d1) < 1.0:
                continue
            cars[("other", vid)] = (s1, (s1 - s0) / self.dt, d1)
        order = self._priority(cars, n * tp)             # highest priority first
        obstacles = []
        out = {}
        for cid in order:
            if cid == "ego" or (isinstance(cid, tuple) and cid[0] == "other"):
                obstacles.append(cv(*cars[cid])); continue
            res = self._step_vehicle(cid, meas[cid], trackings[cid], fh, list(obstacles))
            out[cid] = self._to_raw(cid, res, fh, model_yaw)
            obstacles.append(res["fused_traj"])
        return out

    def _priority(self, cars, horizon_s):
        """Zhou's dynamic priority list: within a lane the leader ranks higher; across lanes the
        per-lane laggards are compared by their constant-velocity terminal position, and the
        one ending furthest back gets the lowest remaining rank."""
        lanes = {}
        for cid, (s, v, d) in cars.items():
            lanes.setdefault(int(round(d / self.lane_width)), []).append((s, s + v * horizon_s, cid))
        ranked = []                                       # lowest priority first
        while any(lanes.values()):
            laggards = [min(entries) for entries in lanes.values() if entries]
            pick = min(laggards, key=lambda e: e[1])
            for entries in lanes.values():
                if pick in entries:
                    entries.remove(pick)
            ranked.append(pick[2])
        return ranked[::-1]

    # ------------------------------------------------------------------ filter
    def _lateral_speed(self, history, fh):
        pts = np.asarray(history, dtype=float)[-2:]
        d = [fh.convert_global_to_frenet_frame(float(p[0]), float(p[1]), float(p[2]))[1] for p in pts]
        return (d[-1] - d[0]) / self.dt if len(d) == 2 else 0.0

    def _step_vehicle(self, tid, y, history, fh, leaders):
        n, tp, dt, w = self.n_proj, self.dt, self.kf_dt, self.lane_width
        lane = int(round(y[2] / w))
        f = self._filters.get(tid)
        if f is None or f["lane"] != lane:
            x0 = f["fused"].copy() if f is not None else np.array([y[0], y[1], 0.0, y[2], self._lateral_speed(history, fh), 0.0])
            x0[0], x0[1], x0[3] = y[0], y[1], y[2]
            f = {"mu": np.array([0.5, 0.25, 0.25]), "x": [x0.copy() for _ in range(3)],
                 "P": [np.eye(6) * 1e-6 for _ in range(3)], "lane": lane, "fused": x0.copy(),
                 "calls": self.kf_every}                     # force an update on the first call
        v_pri = float(y[1])                                  # free-flow reference: keep the present speed
        y_ref = [(lane + LANE_OFFSET[i]) * w for i in range(3)]
        f["calls"] += 1
        update = f["calls"] >= self.kf_every
        if update:
            f["calls"] = 0
            mu, xs, ps = f["mu"], f["x"], f["P"]
            # IMM mixing
            c = mu @ self.trans
            x_bar, p_bar = [], []
            for i in range(3):
                wgt = self.trans[:, i] * mu / c[i]
                xb = sum(wgt[j] * xs[j] for j in range(3))
                pb = sum(wgt[j] * (ps[j] + np.outer(xb - xs[j], xb - xs[j])) for j in range(3))
                x_bar.append(xb); p_bar.append(pb)
            # Kalman update per manoeuvre (prediction with the unprojected reference, as in Zhou)
            x_hat, p_hat, inno, S = [], [], [], []
            for i in range(3):
                pr = self.params[i]
                F, E = _f_e(dt, pr["lon"], pr["lat"], v_pri, y_ref[i])
                xp = F @ x_bar[i] + E
                pp = F @ p_bar[i] @ F.T + self.Q
                e = y - self.H @ xp
                s = self.H @ pp @ self.H.T + self.R
                k = pp @ self.H.T @ np.linalg.inv(s)
                x_hat.append(xp + k @ e); p_hat.append((np.eye(6) - k @ self.H) @ pp); inno.append(e); S.append(s)
        else:
            # between filter updates: keep the posterior modes, re-anchor the measured components
            x_hat = []
            for x in f["x"]:
                x = x.copy(); x[0], x[1], x[3] = y[0], y[1], y[2]; x_hat.append(x)
        # collision-free reference speed per manoeuvre (leaders only -> upper bound, closed form)
        ref = []
        for i in range(3):
            pr = self.params[i]
            a, b = _lon_ab(tp, pr["lon"])
            A, B = np.zeros(n), np.zeros(n)
            acc, apow = np.zeros(3), np.eye(3)
            for j in range(n):
                acc = acc + apow @ b
                apow = apow @ a
                A[j] = acc[0]; B[j] = (apow @ x_hat[i][:3])[0]
            y_mode = _rollout(x_hat[i], tp, pr["lon"], pr["lat"], v_pri, y_ref[i], n)[3, 1:]
            v_up = v_pri
            for tr in leaders:
                m = (np.abs(y_mode - tr[1:, 1]) <= self.w_veh) & (tr[1:, 0] - B > 0.0) & (A > 1e-9)
                if m.any():
                    v_up = min(v_up, float(np.min((tr[1:, 0][m] - self.l_veh - B[m]) / A[m])))
            ref.append(max(v_up, 0.0))
        if update:
            # manoeuvre cost and likelihood (Zhou's augmented innovation)
            t = np.arange(n + 1) * tp
            logl = np.zeros(3)
            for i in range(3):
                pr = self.params[i]
                X = _rollout(x_hat[i], tp, pr["lon"], pr["lat"], ref[i], y_ref[i], n)
                cost = (self.weight[0] * np.trapz(X[2] ** 2, t) + self.weight[1] * np.trapz(X[5] ** 2, t)
                        + self.weight[2] * (ref[i] - x_hat[i][1]) ** 2 + self.weight[3] * (y_ref[i] - x_hat[i][3]) ** 2 + 1e-4)
                s_aug = np.zeros((4, 4)); s_aug[:3, :3] = S[i]; s_aug[3, 3] = 0.1 * cost
                e_aug = np.append(inno[i], math.sqrt(cost))
                logl[i] = -0.5 * e_aug @ np.linalg.solve(s_aug, e_aug) - 0.5 * np.linalg.slogdet(2 * math.pi * s_aug)[1]
            logw = np.log(np.maximum(c, 1e-300)) + logl
            mu_k = np.exp(logw - logw.max()); mu_k /= mu_k.sum()
            f["mu"], f["x"], f["P"] = mu_k, x_hat, p_hat
        else:
            mu_k = f["mu"]
        m_star = int(np.argmax(mu_k))
        fused = sum(mu_k[i] * x_hat[i] for i in range(3))
        f["fused"] = fused
        pr = self.params[m_star]
        fused_X = _rollout(fused, tp, pr["lon"], pr["lat"], ref[m_star], y_ref[m_star], n)
        # per-manoeuvre trajectories and sampling-based variances (output horizon)
        n_out = self.out_length
        trajs, var_x, var_y = [], [], []
        for i in range(3):
            pr = self.params[i]
            trajs.append(_rollout(x_hat[i], tp, pr["lon"], pr["lat"], ref[i], y_ref[i], n_out))
            vx, vy = self._sample_variance(i, x_hat[i], ref[i], y_ref[i], n_out)
            var_x.append(vx); var_y.append(vy)
        self._filters[tid] = f
        return {"mu": mu_k, "trajs": trajs, "var_x": var_x, "var_y": var_y, "ref": ref,
                "fused_traj": np.stack((fused_X[0], fused_X[3]), axis=-1)}

    def _sample_variance(self, mode, x0, v_ref, y_ref, n):
        pr = self.params[mode]
        K = self.k_sampling
        klon = pr["kset_lon"][self.rng.integers(0, len(pr["kset_lon"]), K)]
        xs = np.tile(x0[:3], (K, 1)); pos_x = np.zeros((K, n + 1)); pos_x[:, 0] = x0[0]
        for k in range(1, n + 1):
            s, v, a = xs[:, 0], xs[:, 1], xs[:, 2]
            s2 = s + self.dt * v + self.dt ** 2 / 2 * a
            v2 = v + self.dt * a - klon[:, 0] * self.dt ** 2 / 2 * (v - v_ref) - klon[:, 1] * self.dt ** 2 / 2 * a
            a2 = a - klon[:, 0] * self.dt * (v - v_ref) - klon[:, 1] * self.dt * a
            xs = np.stack((s2, v2, a2), axis=-1); pos_x[:, k] = s2
        var_x = pos_x.var(axis=0)
        if pr["kset_lat"] is None:
            var_y = np.full(n + 1, pr["std_y"] ** 2); var_y[0] = 0.0
            return var_x, var_y
        klat = pr["kset_lat"][self.rng.integers(0, len(pr["kset_lat"]), K)]
        ys = np.tile(x0[3:], (K, 1)); pos_y = np.zeros((K, n + 1)); pos_y[:, 0] = x0[3]
        for k in range(1, n + 1):
            d, vd, ad = ys[:, 0], ys[:, 1], ys[:, 2]
            e = d - y_ref
            d2 = d + self.dt * vd + self.dt ** 2 / 2 * ad - self.dt ** 3 / 6 * (klat[:, 0] * e + klat[:, 1] * vd + klat[:, 2] * ad)
            vd2 = vd + self.dt * ad - self.dt ** 2 / 2 * (klat[:, 0] * e + klat[:, 1] * vd + klat[:, 2] * ad)
            ad2 = ad - self.dt * (klat[:, 0] * e + klat[:, 1] * vd + klat[:, 2] * ad)
            ys = np.stack((d2, vd2, ad2), axis=-1); pos_y[:, k] = d2
        return var_x, pos_y.var(axis=0)

    # ------------------------------------------------------------------ output
    def _to_raw(self, tid, res, fh, model_yaw):
        n = self.out_length
        traj = np.asarray(fh.trajectory, dtype=float)
        ds = float(traj[1, 0] - traj[0, 0])
        s = np.stack([tr[0, 1:] for tr in res["trajs"]])          # (3, n)
        d = np.stack([tr[3, 1:] for tr in res["trajs"]])
        idx = np.clip(np.rint((s - traj[0, 0]) / ds).astype(int), 0, traj.shape[0] - 1)
        psi = traj[idx, 3]
        x = traj[idx, 1] - np.sin(psi) * d
        y = traj[idx, 2] + np.cos(psi) * d
        raw_traj_xy = np.stack((x, y), axis=-1)
        raw_vel = np.stack((np.stack([tr[1, 1:] for tr in res["trajs"]]), np.stack([tr[4, 1:] for tr in res["trajs"]])), axis=-1)
        cov = np.zeros((3, n, 2, 2))
        for i in range(3):
            c, sn = np.cos(psi[i]), np.sin(psi[i])
            vx, vy = res["var_x"][i][1:], res["var_y"][i][1:]
            cov[i, :, 0, 0] = c * c * vx + sn * sn * vy
            cov[i, :, 1, 1] = sn * sn * vx + c * c * vy
            cov[i, :, 0, 1] = cov[i, :, 1, 0] = c * sn * (vx - vy)
        probs = self._filter_intention(int(tid), np.asarray(res["mu"], dtype=float))
        return {
            "vehicle_id": int(tid),
            "raw_traj_xy": raw_traj_xy,
            "raw_pred_vel": raw_vel,
            "raw_intention_prob": probs,
            "raw_intention_logits": None,
            "signed_t_cross": 0.0,
            "valid_mask": np.ones((3, n), dtype=bool),
            "mask_true_count": 0,
            "model_yaw": None if model_yaw is None else float(model_yaw),
            "raw_position_cov_global": cov,
            "raw_velocity_cov_global": np.zeros((3, n, 2, 2)),
            "iaimm_ref_speed": [float(v) for v in res["ref"]],
        }
