# -*- coding: utf-8 -*-
"""Required standoff toward the cut-in vehicle over time -> overleaf/figs/alt.

Multimodal SCC (full d_safe), Proposed (alpha_j * d_safe) and Proposed + D-opt (the
standoff the optimizer chose, ``planned_standoff``), from the 10-07 roots in which the
two baselines are hardlinks of the 09-10 runs. Cells and time axes follow the paper
figures: figs_cutin.py's cut-in cell 0018 (time from lane crossing) and figs_nocutin.py's
no-cut-in cell (largest Proposed gain over Multimodal SCC in v_avg, time from the TV's
deceleration start).

Before the TV enters the lane the requirement is the one the plan puts on the TV's
cut-in hypothesis at its predicted entry step (``ego_lane_start_idx``), in the most likely
scenario with the TV cutting in, at that scenario's planned speed; nothing while the
cut-in mode is gated (p < 0.1). After entry it is the step-0 requirement on the TV as
an in-lane lead. Standoff part only (centre-to-centre, L + d0 + tau*v scaled), without
the q(beta)*sigma term, which Proposed and D-opt share and Multimodal SCC lacks.
"""
import importlib.util, pathlib, pickle, csv, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location("agg", "scripts/carla/devel/aggregate_cutin_table.py")
agg = importlib.util.module_from_spec(spec); spec.loader.exec_module(agg)

R = pathlib.Path("results/acc_scenario_sweep/cutin_dopt_20261007")
R100 = pathlib.Path("results/acc_scenario_sweep/nocutin_dopt100_20261008")  # w=100, groups 03/04 only
OUT = pathlib.Path("overleaf/figs/alt"); OUT.mkdir(parents=True, exist_ok=True)
B = "acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_"
POL = [("Multimodal SCC", B + "rmove1000", "#ff7f0e", 1.6, ":"),
       ("Proposed", B + "cutin_chance0.6_rmove1000", "#2ca02c", 2.0, "-"),
       ("Proposed + D-opt (w=300)", B + "cutin_chance0.6_dopt300_rmove1000", "#9467bd", 1.8, "--")]
POL100 = ("Proposed + D-opt (w=100)", B + "cutin_chance0.6_dopt100_rmove1000", "#8c564b", 1.4, "-.")
L, D0, TAU = 4.5, 3.0, 1.3
FLOOR = L + D0
RED = "#b3261e"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.5})


def _interp_track(entry, t):
    st = np.asarray(entry["state_trajectory"], float)
    return np.interp(t, st[:, 0], st[:, 1]), np.interp(t, st[:, 0], st[:, 2]), np.interp(t, st[:, 0], st[:, 4])


def _tv_scenario(step, vid):
    """Index of the most likely scenario in which the TV cuts in (None if there is none)."""
    names = step.get("mode_names") or []
    probs = step.get("mode_probabilities") or []
    idx = [m for m, n in enumerate(names) if "%s:cutin" % vid in n.split("+")]
    return max(idx, key=lambda m: probs[m]) if idx else None


def requirement(step, tv):
    """(standoff the controller puts on the TV this tick, full d_safe at the same speed).

    Centre distance, no sigma term; the ratio of the two is the effective factor
    (1 for Multimodal SCC, alpha_j for Proposed, D / d_safe for D-opt).
    """
    probs = tv.get("acc_mode_prob") or {}
    chance = tv.get("cutin_chance_confidences") or {}
    if tv.get("relation_to_ego_lane") == "ego_lane":  # the TV is the in-lane lead: step 0
        full = L + D0 + TAU * float(step["ego_v"])
        return float((chance.get("lk") or {}).get("scale", 1.0)) * full, full
    if probs.get("cutin", 0.0) <= 0.0 or "cutin" in (tv.get("gated_cutin_modes") or []):
        return np.nan, np.nan                         # no cut-in hypothesis acting as a lead
    k = tv.get("ego_lane_start_idx")
    m = _tv_scenario(step, tv["vehicle_id"])
    xn = (step.get("acc_reference") or {}).get("x_nominal")
    if m is None or xn is None or k is None or not 0 <= int(k) < len(xn[m]):
        return np.nan, np.nan
    k = int(k)
    full = L + D0 + TAU * float(xn[m][k][1])
    planned = step.get("planned_standoff")
    if planned:                                       # D-opt: what the optimizer chose
        row = np.array([np.nan if v is None else v for v in planned[m]], float)
        return (row[k] if np.isfinite(row[k]) else np.nan), full
    return float((chance.get("cutin") or {}).get("scale", 1.0)) * full, full


def load(pol, group, run, anchor, root=R):
    d = root / pol / group / run
    row = agg.collect_run(pol, group, d, window_s=17.5)
    data = pickle.load(open(d / "scenario_result.pkl", "rb"))
    ego = next(v for k, v in data.items() if k.startswith("ego"))
    t0 = float(row["t0_sim"])
    st = [s for s in ego["policy_log"]["steps"] if t0 - 1e-6 <= s["time_s"] <= t0 + 17.5]
    t = np.array([s["time_s"] for s in st], float)
    ex, ey, _ = _interp_track(ego, t)
    tx, ty, tv_v = _interp_track(data["target_cutin_0"], t)
    ego_s = np.array([s["ego_s"] for s in st], float)
    s_tv, d_tv = ego_s + ey - ty, tx - ex              # the aggregator's (s, lateral offset)
    p, req, full = np.full(t.size, np.nan), np.full(t.size, np.nan), np.full(t.size, np.nan)
    for i, s in enumerate(st):
        tv = agg._nearest_target(s, s_tv[i], d_tv[i])
        if tv is None:
            continue
        p[i] = float((tv.get("acc_mode_prob") or {}).get("cutin", 0.0))
        req[i], full[i] = requirement(s, tv)
    if anchor == "cross":
        ref = t0 + float(row["t_cross"])
    else:                                              # TV deceleration start, as figs_nocutin.py
        ref = t[int(np.argmax(tv_v < tv_v[:5].mean() - 0.2))]
    return dict(t=t - ref, p=p, req=req, factor=req / full, dist=s_tv - ego_s, row=row)


def figure(group, run, tag, anchor, xlim, title, extra=()):
    pols = list(POL) + [e[0] for e in extra]
    S = {lab: load(pol, group, run, anchor) for lab, pol, _, _, _ in POL}
    S.update({e[0][0]: load(e[0][1], group, run, anchor, e[1]) for e in extra})
    fig, ax = plt.subplots(4, 1, figsize=(7.16, 8.8), sharex=True, gridspec_kw=dict(hspace=0.12))
    for lab, _, c, lw, ls in pols:
        s = S[lab]
        ax[0].plot(s["t"], s["p"], color=c, lw=1.1, ls=ls, label=lab)
        ax[1].plot(s["t"], s["req"], color=c, lw=lw, ls=ls, label=lab)
        ax[2].plot(s["t"], s["factor"], color=c, lw=lw, ls=ls, label=lab)
        ax[3].plot(s["t"], s["dist"], color=c, lw=lw, ls=ls, label=lab)
    ax[0].axhline(0.1, color="k", lw=0.7, ls=":"); ax[0].axhline(0.6, color="k", lw=0.7, ls="--")
    ax[0].set_ylim(-0.02, 1.05); ax[0].set_ylabel("TV cut-in probability")
    h, l = ax[0].get_legend_handles_labels()
    h += [Line2D([], [], color="k", lw=0.7, ls=":"), Line2D([], [], color="k", lw=0.7, ls="--")]
    l += ["activation gate 0.1", "$\\beta_{ref}$ 0.6"]
    ax[0].legend(h, l, loc="upper left", fontsize=6.4, framealpha=0.85, edgecolor="none")
    ax[1].axhline(FLOOR, color="k", lw=0.8, ls="-.")
    ax[1].set_ylim(0, 36); ax[1].set_ylabel("Required standoff to TV (m)")
    h, l = ax[1].get_legend_handles_labels(); h.append(Line2D([], [], color="k", lw=0.8, ls="-.")); l.append("$L + d_0$ = 7.5 m (D-opt floor)")
    ax[1].legend(h, l, loc="upper right", fontsize=6.4, framealpha=0.85, edgecolor="none")
    ax[2].set_ylim(0, 1.08); ax[2].set_ylabel("Factor: required / full $d_{safe}$")
    ax[3].set_ylim(0, 60); ax[3].set_ylabel("Actual distance to TV (m)")
    ax[3].set_xlabel("Time from lane crossing (s)" if anchor == "cross" else "Time from TV deceleration start (s)")
    for a in ax:
        a.axvline(0, color=RED, lw=1.0, ls="--"); a.set_xlim(*xlim)
    fig.suptitle(title + " (centre distance, standoff only: no $q(\\beta)\\sigma$ term)", y=0.995, fontsize=9.5)
    fig.savefig(OUT / ("standoff_%s.png" % tag), dpi=200, bbox_inches="tight")
    fig.savefig(OUT / ("standoff_%s.pdf" % tag), bbox_inches="tight")
    plt.close(fig); print("wrote alt/standoff_%s" % tag)
    return S


def nocutin_cell():
    """figs_nocutin.py's rule: the 03/04 cell where Proposed gains most over Multimodal SCC in v_avg."""
    cells = collections.defaultdict(dict)
    for r in csv.DictReader(open(R / "table_cutin_runs.csv")):
        if r["group"][:2] in ("03", "04") and r["policy"] in (POL[0][1], POL[1][1]):
            cells[(r["group"], r["run"])][r["policy"]] = float(r["v_avg"])
    return max(((v[POL[1][1]] - v[POL[0][1]]), k) for k, v in cells.items() if len(v) == 2)[1]


if __name__ == "__main__":
    for g, run, tag, ttl in (("01_cutin_normal", "cutin_0018", "01_0018", "Normal cut-in, cell 0018"),
                             ("02_cutin_aggressive", "aggressive_cutin_0018", "02_0018", "Aggressive cut-in, cell 0018")):
        S = figure(g, run, tag, "cross", (-6, 6), ttl)
        r = S["Proposed"]["row"]
        print("   ego %s TV %s lane change %s s" % (r["ego_speed"], r["target_speed"], r["lane_change_time_s"]))
    g, run = nocutin_cell()
    extra = ((POL100, R100),) if (R100 / POL100[1] / g / run / "scenario_result.pkl").exists() else ()
    figure(g, run, "nocutin_" + g[:2] + "_" + run[-4:], "decel", (-1, 6), "No cut-in, %s %s" % (g, run), extra)
