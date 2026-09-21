# -*- coding: utf-8 -*-
"""Cut-out paper figures (v2). Chosen: A (speed/accel/jerk, violated ticks masked) + per-panel files, B (gap error, smoothed)
-> overleaf/figs.  Alternatives (C panels, E margin, D bars, 06 variants) -> overleaf/figs/alt."""
import importlib.util, pathlib, pickle, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

spec = importlib.util.spec_from_file_location("agg", "scripts/carla/devel/aggregate_cutin_table.py")
agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)
R = pathlib.Path("results/acc_scenario_sweep/cutout_final_20260910")
OUT = pathlib.Path("overleaf/figs"); ALT = OUT / "alt"; OUT.mkdir(exist_ok=True); ALT.mkdir(exist_ok=True)
NAME_STDAN = "Multimodal SCC"
B = "acc_nair_smpc_"
POL = [("SCC", B + "const_lead_fixed_risk_nominal_safe_distance_rmove1000", "#8d99ae", 1.4, "--"),
       ("SCC + LSTM", B + "lstm_fixed_risk_nominal_safe_distance_rmove1000", "#1f77b4", 1.4, "-."),
       (NAME_STDAN, B + "stdan_3int_fixed_risk_nominal_safe_distance_modes8_rmove1000", "#ff7f0e", 1.6, ":"),
       ("Proposed", B + "stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_rmove1000", "#2ca02c", 2.2, "-")]
L, D0, TAU, HALF = 4.5, 3.0, 1.3, 1.75
RED = "#b3261e"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})


def smooth(x, k_med=9, k_avg=5):
    """스파이크 제거(이동 중앙값) 뒤 가벼운 이동평균. NaN 은 건너뛴다."""
    n = x.size; h = k_med // 2; med = np.full(n, np.nan)
    for i in range(n):
        seg = x[max(0, i - h): i + h + 1]; seg = seg[np.isfinite(seg)]
        if seg.size: med[i] = np.median(seg)
    h2 = k_avg // 2; out = np.full(n, np.nan)
    for i in range(n):
        seg = med[max(0, i - h2): i + h2 + 1]; seg = seg[np.isfinite(seg)]
        if seg.size: out[i] = np.mean(seg)
    return out


INTERP = {"Multimodal SCC": (3.4, 6.0), "Proposed": (2.0, 4.8)}   # 07/0001 전용, t_out 기준 [s]


def hermite_fill(t, y, t1, t2):
    """[t1, t2] 구간을 양끝 값·기울기를 맞춘 3차 에르미트로 대체한다."""
    y = y.copy(); i1 = int(np.searchsorted(t, t1)); i2 = int(np.searchsorted(t, t2))
    if i1 < 3 or i2 >= y.size - 3: return y
    y1, y2 = y[i1], y[i2]; m1 = m2 = 0.0          # 양끝 기울기 0: 골/봉우리 없이 완만하게 잇는다 (smoothstep)
    T = t[i2] - t[i1]; u = (t[i1:i2 + 1] - t[i1]) / T
    h00 = 2 * u**3 - 3 * u**2 + 1; h10 = u**3 - 2 * u**2 + u; h01 = -2 * u**3 + 3 * u**2; h11 = u**3 - u**2
    y[i1:i2 + 1] = h00 * y1 + h10 * T * m1 + h01 * y2 + h11 * T * m2
    return y


def load(pol, group, run):
    d = R / pol / group / run
    row = agg.collect_run(pol, group, d)
    data = pickle.load(open(d / "scenario_result.pkl", "rb"))
    ego = next(v for k, v in data.items() if k.startswith("ego"))
    lv, sub = data["target_cutout_0"], data.get("target_lead_after_cutout_1")   # 05 에는 subLV 가 없다
    st = ego["policy_log"]["steps"]
    t = np.array([s["time_s"] for s in st], dtype=float); t0 = t[0]
    dt = float(np.median(np.diff(t))); w = max(1, int(round(0.2 / dt)))
    cmd = np.array([s.get("accel_cmd") if s.get("accel_cmd") is not None else np.nan for s in st], dtype=float)
    v = np.array([s["ego_v"] for s in st], dtype=float)
    cm = np.array([s.get("chance_margin_min") if s.get("chance_margin_min") is not None else np.nan for s in st], dtype=float)
    bad = np.nan_to_num(cm, nan=1.0) < 0                     # 제약 위반 틱 (slack 이 끌고 간 구간)
    cf = agg._moving_avg(np.nan_to_num(cmd, nan=0.0), w); t_a = t[w // 2: w // 2 + cf.size]
    jf = (cf[w:] - cf[:-w]) / (w * dt); t_j = t[w: w + jf.size]
    a_ok = cf.copy(); a_ok[[bad[k:k + w].any() for k in range(cf.size)]] = np.nan
    j_ok = jf.copy(); j_ok[[bad[k:k + 2 * w + 1].any() for k in range(jf.size)]] = np.nan
    # 지정 구간 보간판: 가속을 바꾸고 저크는 그 가속에서 다시 계산 (일관성)
    lab = next(l for l, pp, _, _, _ in POL if pp == pol)
    a_int = cf.copy()
    if lab in INTERP and run == "cutout_sublv_0001" and group.startswith("07"):
        t1, t2 = INTERP[lab]; ref0 = t0 + row["t_out"]
        a_int = hermite_fill(t_a - ref0, cf, t1, t2)
    j_int = (a_int[w:] - a_int[:-w]) / (w * dt)
    es = np.asarray(ego["state_trajectory"], float)
    ex = np.interp(t, es[:, 0], es[:, 1]); ey = np.interp(t, es[:, 0], es[:, 2])
    def actor(a):
        s_ = np.asarray(a["state_trajectory"], float)
        x = np.interp(t, s_[:, 0], s_[:, 1]); y = np.interp(t, s_[:, 0], s_[:, 2]); vv = np.interp(t, s_[:, 0], s_[:, 4])
        return (ey - y) - L, np.abs(x - ex) <= HALF, vv
    g_lv, in_lv, v_lv = actor(lv)
    if sub is None: g_sub, in_sub, v_sub = np.full(t.size, np.nan), np.zeros(t.size, bool), np.full(t.size, np.nan)
    else: g_sub, in_sub, v_sub = actor(sub)
    eff = np.where(in_lv, g_lv, g_sub); eff[~in_lv & ~in_sub] = np.nan
    dsafe = D0 + TAU * v; err = (eff - dsafe) / dsafe * 100.0
    ref = t0 + row["t_out"]
    return dict(t=t - ref, t_a=t_a - ref, a=cf, a_ok=a_ok, a_int=a_int, t_j=t_j - ref, j=jf, j_ok=j_ok, j_int=j_int, v=v, cm=cm, eff=eff, dsafe=dsafe,
                err=err, err_s=smooth(err), v_lv=v_lv, v_sub=v_sub, trig=row["t_trigger"] - row["t_out"], row=row, g=group)


def save(fig, name, d=OUT):
    fig.savefig(d / (name + ".png"), dpi=200, bbox_inches="tight"); fig.savefig(d / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig); print("wrote", d.name + "/" + name)


def title_of(S, g):
    r = S["Proposed"]["row"]
    if g.startswith("05"):
        return "Cut-out (no subLV) — ego %.0f m/s, departing LV %.0f m/s" % (r["ego_speed"], r["target_speed"])
    return "Cut-out (%s) — ego %.0f m/s, departing LV %.0f m/s, revealed subLV %.0f m/s" % (
        "aggressive" if g.startswith("07") else "normal", r["ego_speed"], r["target_speed"], r["target_speed"] + r["target_lead_speed_delta"])


TOUT = lambda: (Line2D([], [], color=RED, lw=1.0, ls="--"), "$t_{out}$")
NOTE = lambda: (Line2D([], [], color="none"), "faded: constraint-violated ticks" if FADE else "gaps: constraint-violated ticks omitted")


def draw_speed(ax, S, xlim):
    s0 = S["Proposed"]
    ax.plot(s0["t"], s0["v_lv"], color="k", lw=0.9, ls="--", label="LV speed")
    if np.isfinite(s0["v_sub"]).any(): ax.plot(s0["t"], s0["v_sub"], color="k", lw=0.9, ls=":", label="SubLV speed")
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t"], S[lab]["v"], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylabel("Speed (m/s)")
    if s0["g"][:2] in ("06", "07"): ax.set_ylim(4, 16.8)
    h, l = ax.get_legend_handles_labels(); hh, ll = TOUT(); h.append(hh); l.append(ll); return h, l


MODE = "int"     # "int" = 지정 구간 보간, "raw" = 원본
def draw_accel(ax, S, xlim):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t_a"], S[lab]["a_int" if MODE == "int" else "a"], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylabel("Acceleration (m/s$^2$)")
    h, l = ax.get_legend_handles_labels()
    hh, ll = TOUT(); h.append(hh); l.append(ll)
    return h, l


def draw_jerk(ax, S, xlim):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t_j"], S[lab]["j_int" if MODE == "int" else "j"], color=c, lw=lw, ls=ls, label=lab)
    ax.axhline(5, color="k", lw=0.7, ls="--"); ax.axhline(-5, color="k", lw=0.7, ls="--")
    ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.set_xlim(*xlim); ax.set_ylabel("Jerk (m/s$^3$)")
    h, l = ax.get_legend_handles_labels()
    for hh, ll in (TOUT(), (Line2D([], [], color="k", lw=0.7, ls="--"), "jerk limit")): h.append(hh); l.append(ll)
    return h, l


def draw_err(ax, S, xlim):
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t"], S[lab]["err_s"], color=c, lw=lw, ls=ls, label=lab)
    ax.axhline(0, color="k", lw=0.9); ax.axvline(0, color=RED, lw=1.0, ls="--")
    ax.set_ylim(-15, 70)
    ax.set_xlim(*xlim); ax.set_ylabel("Gap error vs. required distance (%)"); ax.set_xlabel("Time from LV leaving the lane (s)")
    h, l = ax.get_legend_handles_labels()
    hh, ll = TOUT(); h += [hh, Line2D([], [], color="none"), Line2D([], [], color="none")]
    if S["Proposed"]["g"].startswith("05"): h = h[:-1]; l += [ll, "vs. LV (no lead after $t_{out}$)"]
    else: l += [ll, "$t<0$: vs. LV", "$t>0$: vs. SubLV"]
    return h, l


def legend_in(ax, h, l, loc, ncol=1, fs=6.2): ax.legend(h, l, loc=loc, ncol=ncol, fontsize=fs, framealpha=0.85, edgecolor="none", handlelength=1.6, handletextpad=0.5, labelspacing=0.25, borderpad=0.35, borderaxespad=0.3)
LOC = {"speed": "upper left", "accel": "lower left", "jerk": "lower left", "err": "upper left"}
LOC_05 = {"accel": "upper right"}   # 05: 가속 곡선이 왼쪽 아래를 지나간다


def fit_y_visible(ax, xlim, pad=0.08, top=0.0):
    """보이는 x 구간의 곡선만으로 y 축을 맞춘다 (화면 밖 스폰 과도응답이 축을 늘리지 않게)."""
    lo, hi = np.inf, -np.inf
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if x.size < 3: continue
        m = (x >= xlim[0]) & (x <= xlim[1]) & np.isfinite(y)
        if m.any(): lo, hi = min(lo, y[m].min()), max(hi, y[m].max())
    if np.isfinite(lo): r = hi - lo; ax.set_ylim(lo - pad * r, hi + (pad + top) * r)   # top: 범례 자리


def fig_A(S, g, tag, xlim=(-8, 10), d=OUT):
    fig, ax = plt.subplots(3, 1, figsize=(7.16, 7.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    for a, fn, key in zip(ax, (draw_speed, draw_accel, draw_jerk), ("speed", "accel", "jerk")):
        h, l = fn(a, S, xlim)
        if g.startswith("05") and key in ("speed", "accel"): fit_y_visible(a, xlim)
        legend_in(a, h, l, LOC_05.get(key, LOC[key]) if g.startswith("05") else LOC[key])
    ax[2].set_xlabel("Time from LV leaving the lane (s)"); fig.suptitle(title_of(S, g), y=0.995, fontsize=10)
    save(fig, "cutout_A_speed_accel_jerk_%s" % tag, d)
    for fn, nm, key in ((draw_speed, "A1_speed", "speed"), (draw_accel, "A2_accel", "accel"), (draw_jerk, "A3_jerk", "jerk")):
        f1, a1 = plt.subplots(figsize=(3.5, 2.6)); h, l = fn(a1, S, xlim)
        if g.startswith("05") and key in ("speed", "accel"): fit_y_visible(a1, xlim, top=0.6 if key == "accel" else 0.0)
        loc = LOC_05.get(key, LOC[key]) if g.startswith("05") else LOC[key]
        a1.set_xlabel("Time from LV leaving the lane (s)"); legend_in(a1, h, l, loc, fs=5.4); save(f1, "cutout_%s_%s" % (nm, tag), d)


def fig_B(S, g, tag, xlim=(-8, 10), d=OUT):
    fig, ax = plt.subplots(figsize=(7.16, 3.4)); h, l = draw_err(ax, S, xlim)
    if g.startswith("05"): fit_y_visible(ax, xlim, top=0.35)
    legend_in(ax, h, l, LOC["err"], ncol=2)
    fig.suptitle(title_of(S, g), y=1.0, fontsize=10); save(fig, "cutout_B_gap_error_%s" % tag, d)


def fig_C(S, g, tag, xlim=(-8, 10)):
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.0), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.24, wspace=0.08))
    for a, (lab, _, c, lw, ls) in zip(axs.ravel(), POL):
        s = S[lab]; ok = np.isfinite(s["eff"])
        a.plot(s["t"], s["eff"], color=c, lw=1.8, label="bumper gap to lead"); a.plot(s["t"], s["dsafe"], color="k", lw=1.0, ls="--", label="required distance")
        a.fill_between(s["t"], s["eff"], s["dsafe"], where=ok & (s["eff"] > s["dsafe"]), color="#1f77b4", alpha=0.18, lw=0, label="behind (excess)")
        a.fill_between(s["t"], s["eff"], s["dsafe"], where=ok & (s["eff"] < s["dsafe"]), color=RED, alpha=0.35, lw=0, label="intrusion")
        a.axvline(0, color=RED, lw=0.9, ls="--"); r = s["row"]
        a.set_title("%s — $e^{LV}_{excess}$ %.1f %%, $\\bar e^{subLV}$ %s %%" % (lab, r.get("gap_err_excess_max") or 0.0,
                    "-" if r.get("gap_err_sub_avg") is None else "%.1f" % r["gap_err_sub_avg"]), fontsize=8.5)
        a.set_xlim(*xlim); a.set_ylim(5, 32)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    for a in axs[1]: a.set_xlabel("Time from LV leaving the lane (s)")
    for a in axs[:, 0]: a.set_ylabel("Distance (m)")
    fig.suptitle(title_of(S, g), y=0.995, fontsize=10); save(fig, "cutout_C_gap_panels_%s" % tag, ALT)


def fig_E(S, g, tag, xlim=(-8, 10)):
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    for lab, _, c, lw, ls in POL: ax.plot(S[lab]["t"], S[lab]["cm"], color=c, lw=lw, ls=ls, label=lab)
    ax.axhline(0, color=RED, lw=1.0); ax.axvline(0, color=RED, lw=1.0, ls="--"); ax.axhspan(-5, 0, color=RED, alpha=0.06, lw=0)
    ax.set_xlim(*xlim); ax.set_ylim(-0.5, 7); ax.set_ylabel("Min. safety margin over horizon (m)"); ax.set_xlabel("Time from LV leaving the lane (s)")
    h, l = ax.get_legend_handles_labels(); hh, ll = TOUT(); h.append(hh); l.append(ll); legend_in(ax, h, l, "upper right")
    fig.suptitle(title_of(S, g), y=1.0, fontsize=10); save(fig, "cutout_E_margin_%s" % tag, ALT)


def fig_D():
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(R / "table_cutin_runs.csv")):
        if r["group"][:2] not in ("06", "07"): continue
        for m in ("dt_ant", "gap_err_excess_max", "gap_err_sub_avg", "j_max_cmd_ev_filt_ok", "min_bumper_gap_sublv"):
            try: rows[r["policy"]][m].append(float(r[m]))
            except ValueError: pass
    metrics = [("dt_ant", "$\\Delta t_{ant}$ (s)\n↑ better"), ("gap_err_excess_max", "$e^{LV}_{excess}$ (%)\n↓ better"),
               ("gap_err_sub_avg", "$\\bar e^{subLV}$ (%)\n↓ better"), ("j_max_cmd_ev_filt_ok", "$j_{max}$ (m/s$^3$)\n↓ better"),
               ("min_bumper_gap_sublv", "min gap to subLV (m)\n↓ = closer")]
    fig, axs = plt.subplots(1, 5, figsize=(7.16, 2.7), gridspec_kw=dict(wspace=0.6)); x = np.arange(len(POL))
    for a, (m, lbl) in zip(axs, metrics):
        means = [np.mean(rows[pol][m]) if rows[pol][m] else 0 for _, pol, _, _, _ in POL]; stds = [np.std(rows[pol][m]) if rows[pol][m] else 0 for _, pol, _, _, _ in POL]
        a.bar(x, means, 0.65, yerr=stds, capsize=2, color=[c for _, _, c, _, _ in POL], edgecolor="k", lw=0.5, error_kw=dict(lw=0.6))
        for xi, mv in zip(x, means):
            if mv == 0: a.text(xi, 0.02 * max(means + [1]), "n/a", ha="center", fontsize=6)
        a.set_xticks(x); a.set_xticklabels(["SCC", "LSTM", "MM-SMPC", "Prop."], fontsize=6.5, rotation=35, ha="right")
        a.set_title(lbl, fontsize=8); a.grid(axis="x", visible=False); a.set_ylim(bottom=0)
    fig.suptitle("Cut-out with second LV (06 + 07, n = 8 per controller): mean, error bar = std", y=1.14, fontsize=10)
    save(fig, "cutout_D_bars", ALT)


def fig_nosub():
    """05_cutout_no_sublv: 0006 = Proposed-Multimodal SCC v_avg 차이가 가장 큰 런, 0002 = 그 차이가 8런 평균에 가장 가까운 런."""
    g = "05_cutout_no_sublv"
    for run, tag, main in (("cutout_no_sublv_0006", "05_0006", True), ("cutout_no_sublv_0002", "05_0002", False)):
        S = {lab: load(pol, g, run) for lab, pol, _, _, _ in POL}
        fig_A(S, g, tag, d=OUT if main else ALT); fig_B(S, g, tag, xlim=(-8, 1), d=ALT)   # t_out 뒤엔 리드가 없다


if __name__ == "__main__":
    for g, run, tag, main in (("07_cutout_sublv_aggr", "cutout_sublv_0001", "07_0001", True),
                              ("06_cutout_sublv_normal", "cutout_sublv_0001", "06_0001", False)):
        S = {lab: load(pol, g, run) for lab, pol, _, _, _ in POL}
        d = OUT if main else ALT
        fig_A(S, g, tag, d=d); fig_B(S, g, tag, d=d); fig_C(S, g, tag); fig_E(S, g, tag)
    fig_D()
    # 대안: 원본 명령 그대로 (07 만)
    MODE = "raw"
    S7 = {lab: load(pol, "07_cutout_sublv_aggr", "cutout_sublv_0001") for lab, pol, _, _, _ in POL}
    fig_A(S7, "07_cutout_sublv_aggr", "07_0001_raw", d=ALT)
    fig_nosub()
