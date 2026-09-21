# -*- coding: utf-8 -*-
"""Cut-in paper figures (v2). Chosen: A (3-panel + per-panel files) -> overleaf/figs.
Alternatives (B mechanism, C 2x2 panels, D bars, 02_0018 variants) -> overleaf/figs/alt."""
import importlib.util, pathlib, pickle, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

spec = importlib.util.spec_from_file_location("agg", "scripts/carla/devel/aggregate_cutin_table.py")
agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
spec2 = importlib.util.spec_from_file_location("an", "scripts/carla/devel/animate_acc_stdan_scenario.py")
an = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(an)

R = pathlib.Path("results/acc_scenario_sweep/cutin_final_20260910")
OUT = pathlib.Path("overleaf/figs"); ALT = OUT / "alt"; OUT.mkdir(exist_ok=True); ALT.mkdir(exist_ok=True)
NAME_STDAN = "Multimodal SCC"          # 한 곳에서 바꾼다
B = "acc_nair_smpc_"
POL = [("SCC", B + "const_lead_fixed_risk_nominal_safe_distance_rmove1000", "#8d99ae", 1.4, "--"),
       ("SCC + LSTM", B + "lstm_fixed_risk_nominal_safe_distance_rmove1000", "#1f77b4", 1.4, "-."),
       (NAME_STDAN, B + "stdan_3int_fixed_risk_nominal_safe_distance_modes8_rmove1000", "#ff7f0e", 1.6, ":"),
       ("Proposed", B + "stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_rmove1000", "#2ca02c", 2.2, "-")]
L, D0, TAU, HALF = 4.5, 3.0, 1.3, 1.75
RED, PRE = "#b3261e", "#e6ebf2"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5})


def load(pol, group, run, window_s=17.5):
    d = R / pol / group / run
    row = agg.collect_run(pol, group, d, window_s=window_s)
    data = pickle.load(open(d / "scenario_result.pkl", "rb"))
    ego = next(v for k, v in data.items() if k.startswith("ego"))
    tv = data["target_cutin_0"]
    st = ego["policy_log"]["steps"]
    t = np.array([s["time_s"] for s in st], dtype=float); t0 = t[0]
    keep = t <= t0 + window_s + 1e-9
    st = [s for s, k in zip(st, keep) if k]; t = t[keep]
    dt = float(np.median(np.diff(t))); w = max(1, int(round(0.2 / dt)))
    cmd = np.array([s.get("accel_cmd") if s.get("accel_cmd") is not None else np.nan for s in st], dtype=float)
    v = np.array([s["ego_v"] for s in st], dtype=float)
    cf = agg._moving_avg(np.nan_to_num(cmd, nan=0.0), w); t_a = t[w // 2: w // 2 + cf.size]
    jf = (cf[w:] - cf[:-w]) / (w * dt); t_j = t[w: w + jf.size]
    es = np.asarray(ego["state_trajectory"], float); ts_ = np.asarray(tv["state_trajectory"], float)
    ex = np.interp(t, es[:, 0], es[:, 1]); ey = np.interp(t, es[:, 0], es[:, 2])
    tx = np.interp(t, ts_[:, 0], ts_[:, 1]); ty = np.interp(t, ts_[:, 0], ts_[:, 2])
    gap = (ey - ty) - L; inlane = np.abs(tx - ex) <= HALF; dsafe = D0 + TAU * v
    tc = row.get("t_cross"); ttrig = row.get("t_trigger"); anchor = tc if tc is not None else ttrig; ref = t0 + anchor
    p = np.zeros(t.size)
    for i, s in enumerate(st):
        try: p[i] = max((float((x.get("acc_mode_prob") or {}).get("cutin") or 0.0) for x in an._targets(s)), default=0.0)
        except Exception: p[i] = 0.0
    hold = int(round(0.5 / dt)); onset = np.nan
    for k in range(cf.size - hold):
        if np.all(cf[k:k + hold] < -0.3): onset = t_a[k] - ref; break
    return dict(t=t - ref, t_a=t_a - ref, a=cf, t_j=t_j - ref, j=jf, v=v, gap=gap, inlane=inlane, dsafe=dsafe,
                dx=np.abs(tx - ex), trig=(ttrig - anchor) if ttrig is not None else np.nan, p=p, onset=onset, row=row)


def save(fig, name, d=OUT):
    fig.savefig(d / (name + ".png"), dpi=200, bbox_inches="tight"); fig.savefig(d / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", d.name + "/" + name)


# ---------------------------------------------------------------- panel drawers (return legend handles)
def draw_accel(ax, S, xlim):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t_a"], S[lab]["a"], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylim(-3.4, 3.2); ax.set_ylabel("Acceleration (m/s$^2$)")
    h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], color=RED, lw=1.0, ls="--")); l.append("$t_{cross}$")
    return h, l


def draw_jerk(ax, S, xlim):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t_j"], S[lab]["j"], color=c, lw=lw, ls=ls, label=lab)
    ax.axhline(5, color="k", lw=0.7, ls="--"); ax.axhline(-5, color="k", lw=0.7, ls="--")
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylabel("Jerk (m/s$^3$)")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=RED, lw=1.0, ls="--"), Line2D([], [], color="k", lw=0.7, ls="--")]; l += ["$t_{cross}$", "jerk limit"]
    return h, l


def draw_gap(ax, S, xlim):
    for lab, _, c, lw, ls in POL:
        s = S[lab]; m = s["gap"] - s["dsafe"]
        ax.plot(s["t"], m, color=c, lw=lw, ls=ls, label=lab)
        ax.fill_between(s["t"], np.minimum(m, 0), 0, where=s["inlane"] & (m < 0), color=c, alpha=0.28, lw=0)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color=RED, lw=1.0, ls="--")
    ax.set_xlim(*xlim); ax.set_ylabel("Gap $-$ required distance (m)"); ax.set_xlabel("Time from lane crossing (s)")
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=RED, lw=1.0, ls="--"), Patch(facecolor="grey", alpha=0.35, lw=0)]
    l += ["$t_{cross}$", "safety-distance violation"]
    return h, l


def legend_in(ax, h, l, loc, ncol=1, fs=6.2):
    ax.legend(h, l, loc=loc, ncol=ncol, fontsize=fs, framealpha=0.85, edgecolor="none", handlelength=1.6, handletextpad=0.5, labelspacing=0.25, borderpad=0.35, borderaxespad=0.3)
LOC = {"accel": "upper left", "jerk": "upper left", "gap": "upper left"}


def fig_A(cell, tag, xlim=(-6, 9), d=OUT):
    g, run = cell
    S = {lab: load(pol, g, run) for lab, pol, _, _, _ in POL}
    fig, ax = plt.subplots(3, 1, figsize=(7.16, 7.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    for a, fn, key in zip(ax, (draw_accel, draw_jerk, draw_gap), ("accel", "jerk", "gap")):
        h, l = fn(a, S, xlim); legend_in(a, h, l, LOC[key])
    ax[0].tick_params(labelbottom=False); ax[1].tick_params(labelbottom=False)
    r = S["Proposed"]["row"]
    fig.suptitle("%s cut-in — ego %.0f m/s, TV %.0f m/s, lane change %.2g s" % (
        "Normal" if g.startswith("01") else "Aggressive", r["ego_speed"], r["target_speed"], r["lane_change_time_s"]), y=0.995, fontsize=10)
    save(fig, "cutin_A_timeseries_%s" % tag, d)
    for fn, nm, key in ((draw_accel, "A1_accel", "accel"), (draw_jerk, "A2_jerk", "jerk"), (draw_gap, "A3_gap", "gap")):
        f1, a1 = plt.subplots(figsize=(3.5, 2.6))
        h, l = fn(a1, S, xlim); a1.set_xlabel("Time from lane crossing (s)"); legend_in(a1, h, l, LOC[key], fs=5.4)
        save(f1, "cutin_%s_%s" % (nm, tag), d)
    return S


# ---------------------------------------------------------------- alternatives -> alt/
def fig_mechanism(S, g, tag, xlim=(-6, 6)):
    fig, ax = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True, gridspec_kw=dict(hspace=0.1))
    s = S["Proposed"]
    ax[0].plot(s["t"], s["dx"], color="k", lw=1.3, label="TV lateral offset $|\\Delta x|$ (m)")
    ax[0].axhline(HALF, color="k", lw=0.7, ls=":", label="lane half-width"); ax[0].set_ylabel("Lateral offset (m)")
    ax0b = ax[0].twinx(); ax0b.plot(s["t"], s["p"], color="#2ca02c", lw=1.6, label="$p_{\\rm cutin}$ (prediction)")
    ax0b.axhline(0.6, color="#2ca02c", lw=0.7, ls="--", label="$\\beta_{ref}$ = 0.6"); ax0b.set_ylim(-0.02, 1.05)
    ax0b.set_ylabel("Cut-in probability", color="#2ca02c"); ax0b.grid(False)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax0b.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.09, 1.0), frameon=False)
    for lab, _, c, lw, ls in POL:
        q = S[lab]; ax[1].plot(q["t_a"], q["a"], color=c, lw=lw, ls=ls, label=lab)
        if np.isfinite(q["onset"]): ax[1].plot(q["onset"], np.interp(q["onset"], q["t_a"], q["a"]), "v", color=c, ms=7, mec="k", mew=0.4)
    h, l = ax[1].get_legend_handles_labels()
    h += [Line2D([], [], color=RED, lw=1.0, ls="--"), Line2D([], [], marker="v", color="w", mec="k", mfc="grey", ms=7, lw=0)]
    l += ["$t_{cross}$", "braking onset"]
    ax[1].legend(h, l, loc="upper left", bbox_to_anchor=(1.09, 1.0), frameon=False)
    ax[1].set_ylabel("Acceleration (m/s$^2$)"); ax[1].set_xlabel("Time from lane crossing (s)")
    for a in ax: a.axvline(0, color=RED, lw=1.0, ls="--"); a.set_xlim(*xlim)
    save(fig, "cutin_B_mechanism_%s" % tag, ALT)


def fig_small_multiples(S, g, tag, xlim=(-6, 9)):
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.0), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.22, wspace=0.08))
    for a, (lab, _, c, lw, ls) in zip(axs.ravel(), POL):
        s = S[lab]
        a.axvspan(xlim[0], 0, color=PRE, lw=0, zorder=0)
        a.plot(s["t"], s["gap"], color=c, lw=1.8, label="bumper gap to TV")
        a.plot(s["t"], s["dsafe"], color="k", lw=1.0, ls="--", label="required distance")
        a.fill_between(s["t"], s["gap"], s["dsafe"], where=s["inlane"] & (s["gap"] < s["dsafe"]), color=RED, alpha=0.35, lw=0, label="violation")
        a.axvline(0, color=RED, lw=0.9, ls="--"); r = s["row"]
        a.set_title("%s — $\\delta_{max}$ %.1f %%, $T_{rec}$ %s s" % (lab, r["delta_max"] or 0.0, "-" if r.get("T_rec") is None else "%.2f" % r["T_rec"]), fontsize=9)
        a.set_xlim(*xlim); a.set_ylim(0, 45)
    h, l = axs[0, 0].get_legend_handles_labels(); h.append(Patch(facecolor=PRE, lw=0)); l.append("before lane crossing")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    for a in axs[1]: a.set_xlabel("Time from lane crossing (s)")
    for a in axs[:, 0]: a.set_ylabel("Distance (m)")
    save(fig, "cutin_C_gap_panels_%s" % tag, ALT)


def fig_bars():
    st = collections.defaultdict(dict)
    for r in csv.DictReader(open(R / "table_cutin_summary.csv")):
        try: st[(r["group"], r["policy"])][r["metric"]] = (float(r["mean"]), float(r["std"]))
        except ValueError: pass
    metrics = [("delta_max", "$\\delta_{max}$ (%)"), ("delta_avg", "$\\delta_{avg}$ (%)"), ("T_rec", "$T_{rec}$ (s)"), ("j_max_cmd_filt", "$j_{max}$ (m/s$^3$)")]
    groups = [("01_cutin_normal", "normal"), ("02_cutin_aggressive", "aggressive")]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.7), gridspec_kw=dict(wspace=0.5)); x = np.arange(len(POL)); wb = 0.38
    for a, (m, lbl) in zip(axs, metrics):
        for gi, (g, gl) in enumerate(groups):
            means = [st[(g, pol)].get(m, (np.nan, 0))[0] for _, pol, _, _, _ in POL]; stds = [st[(g, pol)].get(m, (np.nan, 0))[1] for _, pol, _, _, _ in POL]
            a.bar(x + (gi - 0.5) * wb, means, wb, yerr=stds, capsize=2, color=[c for _, _, c, _, _ in POL], alpha=1.0 if gi == 0 else 0.55,
                  hatch="" if gi == 0 else "//", edgecolor="k", lw=0.5, error_kw=dict(lw=0.6))
        a.set_xticks(x); a.set_xticklabels(["SCC", "LSTM", "MM-SMPC", "Prop."], fontsize=7, rotation=35, ha="right")
        a.set_title(lbl, fontsize=9); a.grid(axis="x", visible=False); a.set_ylim(bottom=0)
    fig.legend([Patch(facecolor="grey", edgecolor="k"), Patch(facecolor="grey", alpha=0.55, hatch="//", edgecolor="k")],
               ["normal cut-in (n=18)", "aggressive cut-in (n=17-18)"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    save(fig, "cutin_D_bars", ALT)


S1 = fig_A(("01_cutin_normal", "cutin_0018"), "01_0018")
fig_mechanism(S1, "01", "01_0018"); fig_small_multiples(S1, "01", "01_0018")
S2 = {lab: load(pol, "02_cutin_aggressive", "aggressive_cutin_0018") for lab, pol, _, _, _ in POL}
fig_A(("02_cutin_aggressive", "aggressive_cutin_0018"), "02_0018", d=ALT)
fig_mechanism(S2, "02", "02_0018"); fig_small_multiples(S2, "02", "02_0018"); fig_bars()
