"""Task 2 aggregation: low polynomial order sweep P in {1,2,3,4,6} on 2D Helmholtz.

Records solution error and residual-derivative norms per P.
Outputs to <repo>/results/low_order/: runs.csv, summary.csv, error_vs_P.png,
residual_norms_vs_P.png, report.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_common import RESULTS, add_success_rate, fmt_sci, load_runs, mean_std
from plot_style import apply_style, log_minor_labels, log_safe_yerr, savefig
import matplotlib.pyplot as plt

OUT = os.path.join(RESULTS, 'low_order')
os.makedirs(OUT, exist_ok=True)

RES_COLS = ['rms_r', 'rms_r_x', 'rms_r_y', 'rms_r_xx', 'rms_r_yy']


def main():
    apply_style()
    lo = load_runs('helmholtz2d', 'low_order')
    ff = load_runs('helmholtz2d', 'ff_baseline')
    if not ff.empty:
        p6 = ff[(ff.key_id == 'LPA_P6_N30') & (ff.nh == 2) & (ff.nn == 10)]
        lo = pd.concat([lo, p6], ignore_index=True)
    if lo.empty:
        print('[task2] no runs yet')
        return
    lo = lo[(lo.nh == 2) & (lo.nn == 10)].copy()
    lo['P'] = lo['lpa_order'].astype(int)
    lo.to_csv(os.path.join(OUT, 'runs.csv'), index=False)

    vcols = ['l2_relative', 'l1_absolute'] + RES_COLS
    summ = mean_std(lo, ['P'], vcols).sort_values('P')
    summ = summ.merge(add_success_rate(lo, ['P'], 'l2_relative'), on='P')
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)

    # error vs P: median line + individual seeds (seed outcomes are bimodal, so
    # mean +/- std whiskers on a log axis mislead; per-seed scatter is honest)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for col, color, marker, label in [
            ('l2_relative', '#d62728', 'o', r'Relative $L_2$'),
            ('l1_absolute', '#4292c6', 's', r'Absolute $L_1$')]:
        ax.plot(summ['P'], summ['%s_median' % col], marker=marker, ms=5.5,
                color=color, lw=1.4, label=label + ' (median)')
        jit = (lo['trial'] - lo['trial'].mean()) * 0.045
        ax.scatter(lo['P'] + jit, lo[col], s=14, facecolors='none',
                   edgecolors=color, linewidths=0.7, alpha=0.75)
    ax.axvline(2, color='gray', lw=0.8, ls='--')
    ax.set_yscale('log')
    ax.text(2.08, ax.get_ylim()[1] * 0.55, 'PDE order', fontsize=10, color='gray')
    ax.set_xlabel(r'Legendre order $P$')
    ax.set_ylabel('Error (markers: individual seeds)')
    ax.set_xticks(sorted(lo['P'].unique()))
    log_minor_labels(ax)
    ax.legend(fontsize=10)
    savefig(fig, os.path.join(OUT, 'error_vs_P.png'))

    # residual derivative norms vs P
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    labels = {'rms_r': r'$\Vert r \Vert$', 'rms_r_x': r'$\Vert \partial_x r \Vert$',
              'rms_r_y': r'$\Vert \partial_y r \Vert$',
              'rms_r_xx': r'$\Vert \partial_{xx} r \Vert$',
              'rms_r_yy': r'$\Vert \partial_{yy} r \Vert$'}
    colors = {'rms_r': '#000000', 'rms_r_x': '#9ecae1', 'rms_r_y': '#4292c6',
              'rms_r_xx': '#fdae6b', 'rms_r_yy': '#e6550d'}
    markers = {'rms_r': 'o', 'rms_r_x': 's', 'rms_r_y': '^', 'rms_r_xx': 'D', 'rms_r_yy': 'v'}
    for c in RES_COLS:
        ax.errorbar(summ['P'], summ['%s_mean' % c],
                    yerr=log_safe_yerr(summ['%s_mean' % c], summ['%s_std' % c]),
                    marker=markers[c], ms=4.5, lw=1.2, capsize=2.5,
                    color=colors[c], label=labels[c])
    ax.set_yscale('log')
    ax.set_xlabel(r'Legendre order $P$')
    ax.set_ylabel('RMS of residual / residual derivatives')
    ax.set_xticks(sorted(lo['P'].unique()))
    ax.legend(fontsize=9, ncol=2)
    savefig(fig, os.path.join(OUT, 'residual_norms_vs_P.png'))

    # report
    lines = []
    lines.append('# Task 2 — Low polynomial order sweep on the 2D Helmholtz equation\n')
    lines.append('**Reviewer mapping:** Referee 2, Comment 2 (P below the PDE order on the 2D '
                 'Helmholtz problem, as explicitly suggested); residual-derivative norms also '
                 'feed the Referee 2, Comment 3 response.\n')
    lines.append('## Setup\n')
    lines.append('- LPA order P in {1, 2, 3, 4, 6}, N_panel = 30, network 2x10, protocol '
                 'identical to the paper (Adam 200 @ 1e-2 + L-BFGS-B 40000), 5 trials.')
    lines.append('- The Helmholtz operator is 2nd order, so P=1 is the sub-PDE-order case.')
    lines.append('- Residual derivative norms are RMS values of r, dr/dx, dr/dy, d2r/dx2, '
                 'd2r/dy2 on 4096 fixed random interior points (u derivatives up to 4th order).\n')
    lines.append('| P | rel. L2 | median | success (<0.1) | abs. L1 | RMS r | RMS r_x | RMS r_xx |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for _, r in summ.iterrows():
        lines.append('| %d | %s | %.2e | %d/%d | %s | %s | %s | %s |' % (
            r['P'], fmt_sci(r['l2_relative_mean'], r['l2_relative_std']),
            r['l2_relative_median'],
            round(r['success_rate_0.1'] * r['n_trials']), r['n_trials'],
            fmt_sci(r['l1_absolute_mean'], r['l1_absolute_std']),
            fmt_sci(r['rms_r_mean'], r['rms_r_std']),
            fmt_sci(r['rms_r_x_mean'], r['rms_r_x_std']),
            fmt_sci(r['rms_r_xx_mean'], r['rms_r_xx_std'])))
    lines.append('\nFull columns (incl. r_y, r_yy, max/mean-abs variants) in `summary.csv`.\n')
    lines.append('## Figures\n')
    lines.append('- `error_vs_P.png` — solution error vs P (dashed line marks P = PDE order 2)')
    lines.append('- `residual_norms_vs_P.png` — residual and residual-derivative RMS vs P')
    with open(os.path.join(OUT, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[task2] wrote %s' % OUT)


if __name__ == '__main__':
    main()
