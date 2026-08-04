"""Task 4 aggregation: sensitivity analysis (P x N_panel grid, lr sweep, 5 seeds).

Outputs to <repo>/results/sensitivity/: runs.csv, grid_summary.csv, lr_summary.csv,
heatmap_mean.png, heatmap_std.png, lr_sweep.png, report.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_common import RESULTS, add_success_rate, fmt_sci, load_runs, mean_std
from plot_style import apply_style, log_minor_labels, log_safe_yerr, savefig
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator

OUT = os.path.join(RESULTS, 'sensitivity')
os.makedirs(OUT, exist_ok=True)

ORDERS = [2, 3, 4, 5, 6]
PANELS = [10, 20, 30, 40]
LRS = [1e-3, 3e-3, 1e-2, 3e-2]


def main():
    apply_style()
    df = load_runs('helmholtz2d', 'sensitivity')
    ff = load_runs('helmholtz2d', 'ff_baseline')
    if not ff.empty:
        p6 = ff[(ff.key_id == 'LPA_P6_N30') & (ff.nh == 2) & (ff.nn == 10)]
        df = pd.concat([df, p6], ignore_index=True)
    if df.empty:
        print('[task4] no runs yet')
        return
    df = df[(df.nh == 2) & (df.nn == 10)].copy()
    df['P'] = df['lpa_order'].astype(int)
    df['N_panel'] = df['lpa_panels'].astype(int)
    df.to_csv(os.path.join(OUT, 'runs.csv'), index=False)

    grid = df[np.isclose(df['lr'], 1e-2)]
    gsumm = mean_std(grid, ['P', 'N_panel'], ['l2_relative', 'l1_absolute'])
    gsumm = gsumm.merge(add_success_rate(grid, ['P', 'N_panel'], 'l2_relative'),
                        on=['P', 'N_panel'])
    gsumm.to_csv(os.path.join(OUT, 'grid_summary.csv'), index=False)

    # heatmaps (sequential single-hue ramp, log scale)
    for stat, fname, norm in [('l2_relative_mean', 'heatmap_mean.png', 'log'),
                              ('l2_relative_std', 'heatmap_std.png', 'log'),
                              ('l2_relative_median', 'heatmap_median.png', 'log')]:
        M = np.full((len(ORDERS), len(PANELS)), np.nan)
        for i, P in enumerate(ORDERS):
            for j, N in enumerate(PANELS):
                row = gsumm[(gsumm.P == P) & (gsumm.N_panel == N)]
                if not row.empty:
                    M[i, j] = row[stat].iloc[0]
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        valid = M[np.isfinite(M) & (M > 0)]
        if valid.size == 0:
            plt.close(fig)
            continue
        im = ax.imshow(M, cmap='Blues', origin='lower', aspect='auto',
                       norm=LogNorm(vmin=valid.min(), vmax=valid.max()))
        ax.set_xticks(range(len(PANELS)))
        ax.set_xticklabels(PANELS)
        ax.set_yticks(range(len(ORDERS)))
        ax.set_yticklabels(ORDERS)
        ax.set_xlabel(r'Number of panels $N$')
        ax.set_ylabel(r'Legendre order $P$')
        ax.tick_params(top=False, right=False)
        for i in range(len(ORDERS)):
            for j in range(len(PANELS)):
                if np.isfinite(M[i, j]):
                    lum = (np.log(M[i, j]) - np.log(valid.min())) / \
                          max(np.log(valid.max()) - np.log(valid.min()), 1e-9)
                    ax.text(j, i, '%.1e' % M[i, j], ha='center', va='center',
                            fontsize=8.5, color='white' if lum > 0.6 else 'black')
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        stat_name = {'l2_relative_mean': '(mean)', 'l2_relative_std': '(std)',
                     'l2_relative_median': '(median)'}[stat]
        cb.set_label(r'Relative $L_2$ error %s' % stat_name, fontsize=12)
        cb.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0), numticks=20))
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: '%g' % v))
        cb.ax.tick_params(which='minor', labelsize=9)
        ax.set_title(r'Adam lr $10^{-2}$', fontsize=12)
        savefig(fig, os.path.join(OUT, fname))

    # lr sweep at (P=6, N=30)
    lrdf = df[(df.P == 6) & (df.N_panel == 30)]
    lsumm = mean_std(lrdf, ['lr'], ['l2_relative', 'l1_absolute']).sort_values('lr')
    lsumm.to_csv(os.path.join(OUT, 'lr_summary.csv'), index=False)
    # median line + individual seeds (bimodal seed outcomes)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(lsumm['lr'], lsumm['l2_relative_median'], marker='o', ms=5.5, lw=1.4,
            color='#d62728', label='median')
    jit_f = 1.0 + (lrdf['trial'] - lrdf['trial'].mean()) * 0.035
    ax.scatter(lrdf['lr'] * jit_f, lrdf['l2_relative'], s=14, facecolors='none',
               edgecolors='#d62728', linewidths=0.7, alpha=0.75)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(list(lsumm['lr']))
    ax.set_xticklabels(['%g' % v for v in lsumm['lr']])
    ax.set_xticks([], minor=True)
    ax.set_xlabel('Adam learning rate')
    ax.set_ylabel(r'Relative $L_2$ error' + '\n(markers: individual seeds)')
    log_minor_labels(ax)
    ax.legend(fontsize=10)
    savefig(fig, os.path.join(OUT, 'lr_sweep.png'))

    # report
    lines = []
    lines.append('# Task 4 — Sensitivity analysis: P x N_panel grid and learning-rate sweep\n')
    lines.append('**Reviewer mapping:** Referee 1, Major 4 (sensitivity to polynomial order, '
                 'panel number, training configuration) and Referee 1, Minor 1 (practical '
                 'selection guidelines).\n')
    lines.append('## Setup\n')
    lines.append('- 2D Helmholtz, network 2x10, paper protocol, 5 seeds per cell '
                 '(seeds 1234-1238).')
    lines.append('- Grid: P in {2,3,4,5,6} x N_panel in {10,20,30,40} at Adam lr 1e-2.')
    lines.append('- lr sweep at (P=6, N=30): lr in {1e-3, 3e-3, 1e-2, 3e-2}. Note: lr only '
                 'affects the 200-step Adam warm-up; L-BFGS-B has no learning rate.\n')
    lines.append('## P x N grid (relative L2, mean ± std over 5 seeds)\n')
    hdr = '| P \\ N | ' + ' | '.join(str(n) for n in PANELS) + ' |'
    lines.append(hdr)
    lines.append('|---' * (len(PANELS) + 1) + '|')
    for P in ORDERS:
        cells = []
        for N in PANELS:
            row = gsumm[(gsumm.P == P) & (gsumm.N_panel == N)]
            cells.append(fmt_sci(row['l2_relative_mean'].iloc[0],
                                 row['l2_relative_std'].iloc[0]) if not row.empty else '-')
        lines.append('| %d | %s |' % (P, ' | '.join(cells)))
    lines.append('\n## Learning-rate sweep at (P=6, N=30)\n')
    lines.append('| lr | rel. L2 | median | abs. L1 | n |')
    lines.append('|---|---|---|---|---|')
    for _, r in lsumm.iterrows():
        lines.append('| %g | %s | %.2e | %s | %d |' % (r['lr'],
                     fmt_sci(r['l2_relative_mean'], r['l2_relative_std']),
                     r['l2_relative_median'],
                     fmt_sci(r['l1_absolute_mean'], r['l1_absolute_std']),
                     r['n_trials']))
    lines.append('\n## Notes on seed variability\n')
    lines.append('- At the compact 2x10 setting, individual seeds either converge to ~1e-2 or '
                 'stall at O(1); means over seeds are therefore bimodal-dominated. '
                 '`grid_summary.csv` includes medians and success rates (rel. L2 < 0.1), and '
                 '`heatmap_median.png` shows the median view.')
    lines.append('\n## Figures\n')
    lines.append('- `heatmap_mean.png` / `heatmap_std.png` / `heatmap_median.png` — P x N grid '
                 'of relative L2 (mean / std / median over seeds)')
    lines.append('- `lr_sweep.png` — error vs Adam warm-up lr at (P=6, N=30)')
    with open(os.path.join(OUT, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[task4] wrote %s' % OUT)


if __name__ == '__main__':
    main()
