"""Appendix figure: reliability of STDAN's lane-change probabilities before and after bias-corrected temperature scaling
(token `bcts`, 2026-11-06).

Natural NGSIM Val+Test frames whose +-28-frame label window lies inside the track (2,378,400 frames, 2,954 vehicles,
all lanes), held-out predictions (each recording scored by the calibration fitted on the other five). Event per side:
STDAN's own intention label, i.e. the frame lies within +-28 frames (2.8 s) of a Lane_ID transition toward that side.
x = mean predicted probability in each bin, y = observed frequency, error bars = 95 % vehicle-cluster bootstrap
(Poisson weights, 500 draws; fixed predictions). Bins: 10 equal-width (default) or, with --log, half-decade bins from
1e-4 on log axes with the frames per bin printed; a bin without an event has no point on log axes.
Input: results/ngsim_lambda_cal_20261019/replay_out/bcts_20261106/reliability_bcts_20261106.pkl (bcts_figdata.py).

  python overleaf/figs/scripts/make_bcts_reliability.py [--log] [--lanes all|main] [--pred stdan|iaimm]
  -> overleaf/figs/bcts_reliability[_log].pdf / .png
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
AGG = ROOT / 'results/ngsim_lambda_cal_20261019/replay_out/bcts_20261106/reliability_bcts_20261106.pkl'
OUT = ROOT / 'overleaf/figs/bcts_reliability'
C_RAW, C_CAL, C_GRID, C_TXT, C_MUTED = '#6b6a66', '#2a78d6', '#c9c8c2', '#0b0b0b', '#5f5e5a'
NAME = {'stdan': 'STDAN', 'iaimm': 'IAIMM-KF'}


def bins(A, rng, draws=500):
    vehs = np.unique(A.veh.values)
    w = rng.poisson(1.0, size=(draws, vehs.size))
    rows = []
    for b, g in A.groupby('bin'):
        iv = np.searchsorted(vehs, g.veh.values)
        n, e = g.n.values.astype(float), g.events.values.astype(float)
        boot = (w[:, iv] @ e) / np.maximum(w[:, iv] @ n, 1)
        rows.append(dict(bin=b, n=n.sum(), events=e.sum(), pred=g.sum_q.sum() / n.sum(), obs=e.sum() / n.sum(),
                         lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5)))
    return pd.DataFrame(rows)


def count_label(n):
    if n >= 1e6:
        return '%.1fM' % (n / 1e6)
    return '%.0fk' % (n / 1e3) if n >= 1e4 else ('%.1fk' % (n / 1e3) if n >= 1e3 else '%d' % n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', action='store_true', help='half-decade bins on log axes, frames per bin printed')
    ap.add_argument('--lanes', default='all', choices=('all', 'main'))
    ap.add_argument('--pred', default='stdan', choices=('stdan', 'iaimm'))
    a = ap.parse_args()
    for f in ('NimbusRoman-Regular.otf', 'NimbusRoman-Italic.otf'):
        fm.fontManager.addfont('/usr/share/fonts/opentype/urw-base35/' + f)
    plt.rcParams.update({'font.family': 'Nimbus Roman', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': '#8b8a85', 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    A = pd.read_pickle(AGG)
    A = A[(A.pred == a.pred) & (A.lanes == a.lanes) & (A.binning == ('log' if a.log else 'lin10'))]
    rng = np.random.default_rng(20261106)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    lo_lim = 1e-4
    for ax, (side, title) in zip(axes, (('left', '(a) Left lane change'), ('right', '(b) Right lane change'))):
        if a.log:
            ax.plot([lo_lim, 1], [lo_lim, 1], color=C_GRID, lw=0.8, ls='--', zorder=0)
        else:
            ax.plot([0, 1], [0, 1], color=C_GRID, lw=0.8, ls='--', zorder=0)
        for mp, col, mk, lab in (('raw', C_RAW, 'o', '%s (raw)' % NAME[a.pred]), ('BCTS', C_CAL, 's', 'BCTS')):
            B = bins(A[(A.side == side) & (A['map'] == mp)], rng)
            if a.log:
                B = B[B.events > 0]
            ax.errorbar(B.pred, B.obs, yerr=[B.obs - B.lo.clip(lower=1e-9) if a.log else B.obs - B.lo, B.hi - B.obs],
                        color=col, marker=mk, ms=3.5, lw=1.0, elinewidth=0.7, capsize=1.5, label=lab)
            if a.log and mp == 'BCTS':
                for _, r in B.iterrows():
                    edge = r.pred < 4 * lo_lim          # next to the y axis: label right, below the line
                    ax.annotate(count_label(r.n), (r.pred, r.obs), xytext=(5, -2) if edge else (-4, 3),
                                textcoords='offset points', ha='left' if edge else 'right', va='top' if edge else 'baseline', fontsize=5.5, color=C_MUTED)
        if a.log:
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(lo_lim, 1); ax.set_ylim(lo_lim, 1)
            ax.set_aspect('equal')
        else:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_title(title, fontsize=8.5, loc='left', color=C_TXT)
        ax.set_xlabel('Predicted probability')
        if side == 'left':
            ax.set_ylabel('Observed frequency')
            ax.legend(loc='upper left', frameon=False)
    fig.tight_layout(w_pad=2.0)
    out = str(OUT) + ('_log' if a.log else '') + ('' if a.pred == 'stdan' else '_' + a.pred) + \
        ('' if a.lanes == 'all' else '_' + a.lanes)
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    print('wrote', out + '.pdf / .png')


if __name__ == '__main__':
    main()
