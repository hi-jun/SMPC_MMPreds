"""Appendix figure: calibration metrics of STDAN's lane-change probabilities before and after bias-corrected temperature
scaling (token `bcts`, 2026-11-06).

Natural NGSIM Val+Test frames whose +-28-frame label window lies inside the track, held-out predictions (each recording
scored by the calibration fitted on the other five). Event per side: STDAN's own intention label (within +-28 frames of a
Lane_ID transition toward that side). Skill = 1 - score / score of the constant forecast at the training recordings'
base rate (0 = no better than it); O/E = observed / expected events (1 = calibrated in the large); calibration slope =
logistic regression of the event on logit q (1 = ideal). Error bars: 95 % vehicle-cluster bootstrap over fixed
predictions (Poisson weights, 300 draws); none for the slope. Input: results/ngsim_lambda_cal_20261019/replay_out/
bcts_20261106/metrics_bcts_20261106.csv (bcts_figdata.py).

  python overleaf/figs/scripts/make_bcts_metrics.py [--lanes all|main] [--pred stdan|iaimm]
  -> overleaf/figs/bcts_metrics.pdf / .png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / 'results/ngsim_lambda_cal_20261019/replay_out/bcts_20261106/metrics_bcts_20261106.csv'
OUT = ROOT / 'overleaf/figs/bcts_metrics'
C_RAW, C_CAL, C_TXT, C_REF = '#6b6a66', '#2a78d6', '#0b0b0b', '#8b8a85'
NAME = {'stdan': 'STDAN', 'iaimm': 'IAIMM-KF'}
# column, title, reference value, number format, bootstrap columns, bars (else dots), fixed y limits
PANELS = [('ll_skill', 'Log-loss skill\n(higher is better)', 0.0, '%.2f', ('ll_skill_lo', 'll_skill_hi'), True, None),
          ('brier_skill', 'Brier skill\n(higher is better)', 0.0, '%.2f', ('brier_skill_lo', 'brier_skill_hi'), True, None),
          ('OE', 'Observed / expected\n(1 is calibrated)', 1.0, '%.2f', ('OE_lo', 'OE_hi'), False, (0.0, 1.3)),
          ('slope', 'Calibration slope\n(1 is calibrated)', 1.0, '%.2f', None, False, (0.5, 1.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lanes', default='all', choices=('all', 'main'))
    ap.add_argument('--pred', default='stdan', choices=('stdan', 'iaimm'))
    a = ap.parse_args()
    fm.fontManager.addfont('/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf')
    plt.rcParams.update({'font.family': 'Nimbus Roman', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
                         'ytick.labelsize': 7, 'legend.fontsize': 7.5, 'axes.edgecolor': '#8b8a85', 'axes.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    R = pd.read_csv(CSV)
    R = R[(R.pred == a.pred) & (R.subset == a.lanes + ' lanes') & R['map'].isin(['raw', 'BCTS'])]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
    x = np.arange(2); wd = 0.36
    for ax, (col, title, ref, fmt, ci, bars, ylim) in zip(axes, PANELS):
        ax.axhline(ref, color=C_REF, lw=0.6, ls='--', zorder=0)
        for j, (mp, c, mk, lab) in enumerate((('raw', C_RAW, 'o', '%s (raw)' % NAME[a.pred]), ('BCTS', C_CAL, 's', 'BCTS'))):
            d = R[R['map'] == mp].set_index('side').loc[['left', 'right']]
            v = d[col].values
            xs = x + (j - 0.5) * wd
            lo, hi = (d[ci[0]].values, d[ci[1]].values) if ci else (v, v)
            if bars:
                ax.bar(xs, v - ref, bottom=ref, width=wd * 0.92, color=c, label=lab,
                       yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)],
                       error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor='#3a3936'))
                edge, up = np.where(v >= ref, hi, lo), v >= ref
            else:                                 # a dot: no natural baseline for a ratio or a slope
                ax.errorbar(xs, v, yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)], ls='none', marker=mk, ms=4.5,
                            color=c, elinewidth=0.6, capsize=1.5)
                edge, up = hi, np.ones(v.size, bool)
            for xi, ev, vi, u in zip(xs, edge, v, up):
                ax.annotate(fmt % vi, (xi, ev), xytext=(0, 3 if u else -2), textcoords='offset points', ha='center',
                            va='bottom' if u else 'top', fontsize=6.3, color='#3a3936')
        ax.set_xticks(x); ax.set_xticklabels(['Left', 'Right']); ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, fontsize=8, color=C_TXT)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            lo_y, hi_y = ax.get_ylim()
            span = hi_y - lo_y
            ax.set_ylim(lo_y - 0.08 * span, hi_y + 0.12 * span)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=1.2)
    out = str(OUT) + ('' if a.pred == 'stdan' else '_' + a.pred) + ('' if a.lanes == 'all' else '_' + a.lanes)
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    print('wrote', out + '.pdf / .png')


if __name__ == '__main__':
    main()
