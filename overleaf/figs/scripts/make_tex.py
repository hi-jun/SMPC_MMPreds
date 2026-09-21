# -*- coding: utf-8 -*-
"""유령 컷아웃 아티팩트 16 절 표 데이터로 논문용 LaTeX 표를 만든다.
컷인: cutin_final_20260910 (--window-s 17.5).  컷아웃: cutout_final_20260910 (--window-start-s 3),
저크 두 열은 제약 만족 구간(j_*_cmd_ev_filt_ok).  기존 overleaf/table_*.tex 는 건드리지 않는다."""
import csv, collections, pathlib

CI = pathlib.Path("results/acc_scenario_sweep/cutin_final_20260910/table_cutin_summary.csv")
CO = pathlib.Path("results/acc_scenario_sweep/cutout_final_20260910/table_cutin_summary.csv")
POL = [("acc_nair_smpc_const_lead_fixed_risk_nominal_safe_distance_rmove1000", "SCC"),
       ("acc_nair_smpc_lstm_fixed_risk_nominal_safe_distance_rmove1000", "SCC + LSTM"),
       ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_rmove1000", "Multimodal SCC"),
       ("acc_nair_smpc_stdan_3int_fixed_risk_nominal_safe_distance_modes8_cutin_chance0.6_rmove1000", r"\textbf{Proposed}")]


def load(path):
    st = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        try:
            st[(r["group"], r["policy"])][r["metric"]] = float(r["mean"])
        except ValueError:
            pass
        st[(r["group"], r["policy"])]["_n"] = int(r["n_runs"])
        st[(r["group"], r["policy"])]["_coll"] = int(r["ego_collision_runs"])
    return st


def fmt(v, nd=2):
    if v is None: return "-"
    if abs(v) < 0.5 * 10 ** (-nd): v = 0.0          # -0.00 방지
    return ("%%.%df" % nd) % v


def block(st, group, cols, label):
    """cols: list of (metric, higher_better, nd).  Best per column in bold."""
    vals = {}
    for pol, _ in POL:
        vals[pol] = [st[(group, pol)].get(m) for m, _, _ in cols]
    best = []
    for j, (m, hb, nd) in enumerate(cols):
        cand = [(vals[p][j], p) for p, _ in POL if vals[p][j] is not None]
        if not cand or hb is None:                      # hb None = 판정하지 않는 열
            best.append(None); continue
        best.append((max if hb else min)(cand)[1])
    n = st[(group, POL[0][0])]["_n"]
    lines = [r"\multirow{4}{*}{\shortstack[l]{%s}}" % label.replace("|", r" \\ ")]
    for pol, name in POL:
        cells = []
        for j, (m, hb, nd) in enumerate(cols):
            s = fmt(vals[pol][j], nd)
            if m == "_coll":
                s = "%d/%d" % (st[(group, pol)]["_coll"], st[(group, pol)]["_n"])
                cells.append(s); continue
            cells.append(r"\textbf{%s}" % s if best[j] == pol and vals[pol][j] is not None else s)
        lines.append(" & %s & %s \\\\" % (name, " & ".join(cells)))
    return "\n".join(lines), n


HEAD = r"""\documentclass[letterpaper, 10 pt, journal]{IEEEtran}

\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{tabularx}
\usepackage{float}

\begin{document}
"""

# ------------------------------------------------------------------ cut-in
ci = load(CI)
CUTIN_COLS = [("a_avg", False, 2), ("j_avg_cmd_filt", False, 2), ("j_max_cmd_filt", False, 2),
              ("delta_max", False, 2), ("delta_avg", False, 2), ("_coll", False, 0),
              ("dt_ant", True, 2), ("T_rec", False, 2)]
b1, n1 = block(ci, "01_cutin_normal", CUTIN_COLS, "Normal|cut-in")
b2, n2 = block(ci, "02_cutin_aggressive", CUTIN_COLS, "Aggressive|cut-in")
FALSE_COLS = [("a_min_cmd", True, 2), ("j_avg_cmd_filt", False, 2), ("j_max_cmd_filt", False, 2), ("v_avg", True, 2)]
# 03 + 04 합침: 런 단위 평균 (n = 14)
import numpy as _np
_runs = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(CI.parent / "table_cutin_runs.csv")):
    if r["group"] not in ("03_no_cutin_decel", "04_no_cutin_decel_onset"): continue
    for m, _, _ in FALSE_COLS:
        try: _runs[r["policy"]][m].append(float(r[m]))
        except (ValueError, KeyError): pass
    _runs[r["policy"]]["_n"].append(1)
    _runs[r["policy"]]["_coll"].append(1 if str(r.get("ego_collision")).lower() == "true" else 0)
fp = collections.defaultdict(dict)
for pol, _ in POL:
    for m, _, _ in FALSE_COLS:
        fp[("no_cutin", pol)][m] = float(_np.mean(_runs[pol][m])) if _runs[pol][m] else None
    fp[("no_cutin", pol)]["_n"] = len(_runs[pol]["_n"]); fp[("no_cutin", pol)]["_coll"] = sum(_runs[pol]["_coll"])
b34, n34 = block(fp, "no_cutin", FALSE_COLS, "No cut-in|(adjacent vehicle|decelerates)")
b3 = b34; b4 = ""; n3 = n34; n4 = n34

tex_ci = HEAD + r"""
% --- Table I: Cut-in Performance (final sweep cutin_final_20260910, window 17.5 s) ---
\begin{table*}[t]
\centering
\caption{Performance comparison in cut-in scenarios ($n$ runs per controller: normal @@n1@@, aggressive @@n2@@).
Ride comfort uses the 0.2\,s band-limited acceleration command. Best value per scenario in bold.}
\label{tab:cutin_final}
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llcccccccc@{}}
\toprule
\multirow{2}{*}{\textbf{Scenario}}
& \multirow{2}{*}{\textbf{Controller}}
& \multicolumn{3}{c}{\textbf{Ride Comfort}}
& \multicolumn{3}{c}{\textbf{Safety}}
& \multicolumn{2}{c}{\textbf{Cut-in Response}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}
 &
 & $a_{\text{avg}}$ (m/s$^2$)
 & $j_{\text{avg}}$ (m/s$^3$)
 & $j_{\text{max}}$ (m/s$^3$)
 & $\delta_{\text{max}}$ (\%)
 & $\delta_{\text{avg}}$ (\%)
 & Coll.
 & $\Delta t_{\text{ant}}$ (s)
 & $T_{\text{rec}}$ (s) \\
\midrule
@@b1@@
\midrule
@@b2@@
\bottomrule
\end{tabular}
\end{table*}

% --- Table II: False-positive (no cut-in) scenarios ---
\begin{table*}[t]
\centering
\caption{Unnecessary braking when the adjacent vehicle decelerates but does not cut in
(lead-deceleration and deceleration-onset variants pooled, $n$ = @@n3@@ runs per controller). No controller
intrudes on the safety distance or collides here ($\delta_{\text{max}}=0$); the columns measure only the
cost of reacting to a prediction that does not materialise. $a_{\text{min}}$ is the strongest braking
command and $v_{\text{avg}}$ the mean ego speed over the run (cruise 15.5\,m/s). Best value in bold.}
\label{tab:false_positive}
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
\multirow{2}{*}{\textbf{Scenario}}
& \multirow{2}{*}{\textbf{Controller}}
& \multicolumn{3}{c}{\textbf{Ride Comfort}}
& \textbf{Speed} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-6}
 &
 & $a_{\text{min}}$ (m/s$^2$)
 & $j_{\text{avg}}$ (m/s$^3$)
 & $j_{\text{max}}$ (m/s$^3$)
 & $v_{\text{avg}}$ (m/s) \\
\midrule
@@b3@@
\bottomrule
\end{tabular}
\end{table*}

\end{document}
"""
for k, v in (("@@n1@@", n1), ("@@n2@@", n2), ("@@b1@@", b1), ("@@b2@@", b2),
             ("@@n3@@", n3), ("@@n4@@", n4), ("@@b3@@", b3), ("@@b4@@", b4)):
    tex_ci = tex_ci.replace(k, str(v))
pathlib.Path("overleaf/table_cutin_final.tex").write_text(tex_ci, encoding="utf-8")

# ------------------------------------------------------------------ cut-out
co = load(CO)
CUTOUT_COLS = [("a_avg", False, 2), ("j_avg_cmd_ev_filt_ok", False, 2), ("j_max_cmd_ev_filt_ok", False, 2),
               ("dt_ant", True, 2), ("v_avg", None, 2), ("gap_err_sub_avg", False, 2)]
c5, n5 = block(co, "05_cutout_no_sublv", CUTOUT_COLS, "Cut-out|w/o second LV")
c6, n6 = block(co, "06_cutout_sublv_normal", CUTOUT_COLS, "Cut-out with|second LV (normal)")
c7, n7 = block(co, "07_cutout_sublv_aggr", CUTOUT_COLS, "Cut-out with|second LV (aggr.)")

# ---- 06+07 풀링 (런 단위 평균, n=8)
import numpy as _np
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(CO.parent / "table_cutin_runs.csv")):
    for m, _, _ in CUTOUT_COLS:
        try: runs[(r["group"], r["policy"])][m].append(float(r[m]))
        except (ValueError, KeyError): pass
    runs[(r["group"], r["policy"])]["_n"].append(1)
    runs[(r["group"], r["policy"])]["_coll"].append(1 if str(r.get("ego_collision")).lower() == "true" else 0)
pooled = collections.defaultdict(dict)
for pol, _ in POL:
    for m, _, _ in CUTOUT_COLS:
        vals = runs[("06_cutout_sublv_normal", pol)][m] + runs[("07_cutout_sublv_aggr", pol)][m]
        pooled[("with_sublv", pol)][m] = float(_np.mean(vals)) if vals else None
    pooled[("with_sublv", pol)]["_n"] = len(runs[("06_cutout_sublv_normal", pol)]["_n"]) + len(runs[("07_cutout_sublv_aggr", pol)]["_n"])
    pooled[("with_sublv", pol)]["_coll"] = sum(runs[("06_cutout_sublv_normal", pol)]["_coll"]) + sum(runs[("07_cutout_sublv_aggr", pol)]["_coll"])
    pooled[("05_cutout_no_sublv", pol)] = co[("05_cutout_no_sublv", pol)]
c5p, n5p = block(pooled, "05_cutout_no_sublv", CUTOUT_COLS, "Cut-out|w/o second LV")
c67, n67 = block(pooled, "with_sublv", CUTOUT_COLS, "Cut-out with|second LV")

tex_co = HEAD + r"""
% --- Table III: Cut-out Performance (final sweep cutout_final_20260910) ---
\begin{table*}[t]
\centering
\caption{Performance comparison in cut-out scenarios ($n$ per controller: w/o second LV @@n5@@,
with second LV normal @@n6@@, aggressive @@n7@@). $a_{\text{avg}}$ excludes the first 3\,s of spawn
alignment. $j_{\text{avg}}$ and $j_{\text{max}}$ are the 0.2\,s band-limited command jerk over the
event window, taken on ticks where the safety constraint is satisfied (violated ticks are recovered
by the slack term, not by the tracking policy). $\Delta t_{\text{ant}}$ is the time from the first
sustained cut-out prediction to the lead actually leaving the lane, $v_{\text{avg}}$ the mean ego speed after
the trigger, and $\bar e^{\text{subLV}}$ the mean absolute gap error against the revealed second LV after
$t_{\text{out}}$, relative to the nominal safety distance (not defined w/o second LV). Best value per
scenario in bold.}
\label{tab:cutout_final}
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llcccccc@{}}
\toprule
\multirow{2}{*}{\textbf{Scenario}}
& \multirow{2}{*}{\textbf{Controller}}
& \multicolumn{3}{c}{\textbf{Ride Comfort}}
& \multicolumn{3}{c}{\textbf{Cut-out Response}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
 &
 & $a_{\text{avg}}$ (m/s$^2$)
 & $j_{\text{avg}}$ (m/s$^3$)
 & $j_{\text{max}}$ (m/s$^3$)
 & $\Delta t_{\text{ant}}$ (s)
 & $v_{\text{avg}}$ (m/s)
 & $\bar e^{\text{subLV}}$ (\%) \\
\midrule
@@c5@@
\midrule
@@c6@@
\midrule
@@c7@@
\bottomrule
\end{tabular}
\end{table*}

% --- Table IV: same, with the two second-LV scenarios pooled (normal + aggressive, n = 8) ---
\begin{table*}[t]
\centering
\caption{Cut-out performance with the two second-LV scenarios pooled (normal and aggressive lane change,
$n$ = @@n67@@ runs per controller); w/o second LV as in Table~\ref{tab:cutout_final} ($n$ = @@n5p@@).
Columns as in Table~\ref{tab:cutout_final}. Best value per scenario in bold.}
\label{tab:cutout_pooled}
\renewcommand{\arraystretch}{1.3}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}llcccccc@{}}
\toprule
\multirow{2}{*}{\textbf{Scenario}}
& \multirow{2}{*}{\textbf{Controller}}
& \multicolumn{3}{c}{\textbf{Ride Comfort}}
& \multicolumn{3}{c}{\textbf{Cut-out Response}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
 &
 & $a_{\text{avg}}$ (m/s$^2$)
 & $j_{\text{avg}}$ (m/s$^3$)
 & $j_{\text{max}}$ (m/s$^3$)
 & $\Delta t_{\text{ant}}$ (s)
 & $v_{\text{avg}}$ (m/s)
 & $\bar e^{\text{subLV}}$ (\%) \\
\midrule
@@c5p@@
\midrule
@@c67@@
\bottomrule
\end{tabular}
\end{table*}

\end{document}
"""
for k, v in (("@@n5@@", n5), ("@@n6@@", n6), ("@@n7@@", n7), ("@@c5@@", c5), ("@@c6@@", c6), ("@@c7@@", c7),
             ("@@n67@@", n67), ("@@n5p@@", n5p), ("@@c5p@@", c5p), ("@@c67@@", c67)):
    tex_co = tex_co.replace(k, str(v))
pathlib.Path("overleaf/table_cutout_final.tex").write_text(tex_co, encoding="utf-8")
print("wrote overleaf/table_cutin_final.tex, overleaf/table_cutout_final.tex")
print("a_min_cmd 존재?", "a_min_cmd" in ci[("03_no_cutin_decel", POL[0][0])])
