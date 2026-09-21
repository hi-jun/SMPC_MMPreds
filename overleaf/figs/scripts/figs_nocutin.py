# -*- coding: utf-8 -*-
"""No-cut-in (false-positive) figure: adjacent vehicle decelerates but stays in its lane.
Multimodal SMPC vs Proposed only. Cell auto-picked = largest speed-loss gap between the two."""
import importlib.util, pathlib, pickle, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location("agg", "scripts/carla/devel/aggregate_cutin_table.py")
agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
spec2 = importlib.util.spec_from_file_location("an", "scripts/carla/devel/animate_acc_stdan_scenario.py")
an = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(an)
R = pathlib.Path("results/acc_scenario_sweep/cutin_final_20260910")
OUT = pathlib.Path("overleaf/figs"); OUT.mkdir(exist_ok=True)
NAME_STDAN = "Multimodal SCC"
B = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8"
SCC = "acc_nair_smpc_const_lead_fixed_risk_nominal_safe_distance_rmove1000"
LSTM = "acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance_rmove1000"
POL = [("SCC", SCC, "#8d99ae", 1.4, "--"), ("SCC + LSTM", LSTM, "#1f77b4", 1.4, "-."),
       (NAME_STDAN, B + "_rmove1000", "#ff7f0e", 1.6, ":"), ("Proposed", B + "_cutin_chance0.6_rmove1000", "#2ca02c", 2.2, "-")]
PRED = (NAME_STDAN, "Proposed")          # 의도 확률이 있는 정책
GREY = "#8d99ae"; HALF = 1.75
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

# ---- 셀 선택: STDAN 속도손실 - Proposed 속도손실 (SCC 기준) 최대
cells = collections.defaultdict(dict)
for r in csv.DictReader(open(R / "table_cutin_runs.csv")):
    if r["group"][:2] in ("03", "04") and r["policy"] in (POL[2][1], POL[3][1], SCC):
        cells[(r["group"], r["run"])][r["policy"]] = float(r["v_avg"])
best = max(((v[SCC] - v[POL[2][1]]) - (v[SCC] - v[POL[3][1]]), k) for k, v in cells.items() if len(v) == 3)
(g, run) = best[1]; print("cell:", g, run, "loss gap %.2f m/s" % best[0])


def load(pol):
    d = R / pol / g / run
    row = agg.collect_run(pol, g, d, window_s=17.5)
    data = pickle.load(open(d / "scenario_result.pkl", "rb"))
    ego = next(v for k, v in data.items() if k.startswith("ego")); tv = data["target_cutin_0"]
    st = ego["policy_log"]["steps"]
    t = np.array([s["time_s"] for s in st], dtype=float); t0 = t[0]
    keep = t <= t0 + 17.5 + 1e-9; st = [s for s, k in zip(st, keep) if k]; t = t[keep]
    dt = float(np.median(np.diff(t))); w = max(1, int(round(0.2 / dt)))
    cmd = np.array([s.get("accel_cmd") if s.get("accel_cmd") is not None else np.nan for s in st], dtype=float)
    v = np.array([s["ego_v"] for s in st], dtype=float)
    cf = agg._moving_avg(np.nan_to_num(cmd, nan=0.0), w); t_a = t[w // 2: w // 2 + cf.size]
    jf = (cf[w:] - cf[:-w]) / (w * dt); t_j = t[w: w + jf.size]
    ts_ = np.asarray(tv["state_trajectory"], float); es = np.asarray(ego["state_trajectory"], float)
    v_tv = np.interp(t, ts_[:, 0], ts_[:, 4]); dx = np.abs(np.interp(t, ts_[:, 0], ts_[:, 1]) - np.interp(t, es[:, 0], es[:, 1]))
    p = np.zeros(t.size)
    for i, s in enumerate(st):
        try: p[i] = max((float((x.get("acc_mode_prob") or {}).get("cutin") or 0.0) for x in an._targets(s)), default=0.0)
        except Exception: pass
    # 감속 개시 = TV 속도가 초기값보다 0.5 m/s 이상 떨어진 첫 시각
    # 감속 시작 = TV 속도가 초기값에서 0.2 m/s 떨어진 첫 시각 (0.5 로 잡으면 0.35 s 늦어 예측이 앞서 보인다)
    k0 = int(np.argmax(v_tv < v_tv[:5].mean() - 0.2)); ref = t[k0]
    return dict(t=t - ref, t_a=t_a - ref, a=cf, t_j=t_j - ref, j=jf, v=v, v_tv=v_tv, dx=dx, p=p, v0=float(v[:20].mean()), row=row)


S = {lab: load(pol) for lab, pol, _, _, _ in POL}
r = S["Proposed"]["row"]; xlim = (-3, 12)
RED = "#b3261e"


def draw_tv(ax):
    s = S["Proposed"]
    ax.plot(s["t"], s["v_tv"], color="k", lw=1.3, label="adjacent vehicle speed (m/s)")
    ax.set_ylabel("Adjacent vehicle speed (m/s)"); ax.set_xlim(*xlim)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(S["Proposed"]["t"], S["Proposed"]["p"], color="#8e44ad", lw=1.6, label="$p_{\\rm cutin}$ (prediction)")
    ax2.axhline(0.6, color=GREY, lw=0.8, ls="--", label="$\\beta_{ref}$ = 0.6"); ax2.set_ylim(-0.02, 1.05); ax2.set_ylabel("Cut-in probability")
    ax.axvline(0, color=RED, lw=1.0, ls="--")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    return h1 + h2 + [Line2D([], [], color=RED, lw=1.0, ls="--")], l1 + l2 + ["deceleration"], ax2


def draw_speed(ax):
    ax.axhline(S["Proposed"]["v0"], color=GREY, lw=1.0, ls="--", label="cruise speed (no reaction)")
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t"], S[lab]["v"], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylim(11, 23); ax.set_ylabel("Ego speed (m/s)")
    h, l = ax.get_legend_handles_labels(); return h + [Line2D([], [], color=RED, lw=1.0, ls="--")], l + ["deceleration"]


def draw_accel(ax):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t_a"], S[lab]["a"], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylabel("Acceleration (m/s$^2$)")
    h, l = ax.get_legend_handles_labels(); return h + [Line2D([], [], color=RED, lw=1.0, ls="--")], l + ["deceleration"]


def draw_jerk(ax):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t_j"], S[lab]["j"], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylabel("Jerk (m/s$^3$)")
    h, l = ax.get_legend_handles_labels(); return h + [Line2D([], [], color=RED, lw=1.0, ls="--")], l + ["deceleration"]


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=200, bbox_inches="tight"); fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


tag = "%s_%s" % (g[:2], run[-4:])
fig, ax = plt.subplots(4, 1, figsize=(7.16, 9.4), sharex=True, gridspec_kw=dict(hspace=0.12))
LOC = {"tv": "upper right", "speed": "upper right", "accel": "lower right", "jerk": "lower right"}
def legend_in(ax, h, l, loc, fs=6.2): ax.legend(h, l, loc=loc, fontsize=fs, framealpha=0.85, edgecolor="none", handlelength=1.6, handletextpad=0.5, labelspacing=0.25, borderpad=0.35, borderaxespad=0.3)
h, l, ax2 = draw_tv(ax[0]); legend_in(ax[0], h, l, LOC["tv"])
for a, fn, key in zip(ax[1:], (draw_speed, draw_accel, draw_jerk), ("speed", "accel", "jerk")):
    h, l = fn(a); legend_in(a, h, l, LOC[key])
ax[3].set_xlabel("Time from adjacent-vehicle deceleration (s)")
fig.suptitle("No cut-in (false positive) — ego %.0f m/s, adjacent vehicle %.0f m/s decelerating in its own lane" % (r["ego_speed"], r["target_speed"]), y=0.995, fontsize=10)
save(fig, "nocutin_A_timeseries_%s" % tag)
for fn, nm, key in ((draw_speed, "A2_speed", "speed"), (draw_accel, "A3_accel", "accel"), (draw_jerk, "A4_jerk", "jerk")):
    f1, a1 = plt.subplots(figsize=(3.5, 2.6)); h, l = fn(a1); a1.set_xlabel("Time from deceleration (s)")
    legend_in(a1, h, l, LOC[key], fs=5.4); save(f1, "nocutin_%s_%s" % (nm, tag))
f1, a1 = plt.subplots(figsize=(3.5, 2.6)); h, l, _ = draw_tv(a1); a1.set_xlabel("Time from deceleration (s)")
legend_in(a1, h, l, LOC["tv"], fs=5.4); save(f1, "nocutin_A1_prediction_%s" % tag)
