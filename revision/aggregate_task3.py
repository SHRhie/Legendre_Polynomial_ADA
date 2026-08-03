"""Task 3 aggregation: fixed total DOF (P+1)*N_panel = 210 on 2D Helmholtz.

p-refinement vs h-refinement: expressivity (solution error) vs residual
suppression (residual-derivative norms) at constant activation DOF.
Outputs to <repo>/results/dof_fixed/: runs.csv, summary.csv,
tradeoff.png, report.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_common import RESULTS, add_success_rate, fmt_sci, load_runs, mean_std
from plot_style import apply_style, log_minor_labels, log_safe_yerr, savefig
import matplotlib.pyplot as plt

OUT = os.path.join(RESULTS, 'dof_fixed')
os.makedirs(OUT, exist_ok=True)


def main():
    apply_style()
    df = load_runs('helmholtz2d', 'dof_fixed')
    ff = load_runs('helmholtz2d', 'ff_baseline')
    if not ff.empty:
        p6 = ff[(ff.key_id == 'LPA_P6_N30') & (ff.nh == 2) & (ff.nn == 10)]
        df = pd.concat([df, p6], ignore_index=True)
    if df.empty:
        print('[task3] no runs yet')
        return
    df = df[(df.nh == 2) & (df.nn == 10)].copy()
    df['P'] = df['lpa_order'].astype(int)
    df['N_panel'] = df['lpa_panels'].astype(int)
    df['dof'] = (df['P'] + 1) * df['N_panel']
    df.to_csv(os.path.join(OUT, 'runs.csv'), index=False)

    vcols = ['l2_relative', 'l1_absolute', 'rms_r', 'rms_r_x', 'rms_r_xx', 'n_params']
    summ = mean_std(df, ['P', 'N_panel', 'dof'], vcols).sort_values('P')
    summ = summ.merge(add_success_rate(df, ['P', 'N_panel', 'dof'], 'l2_relative'),
                      on=['P', 'N_panel', 'dof'])
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)

    # trade-off figure: error and residual-derivative norm vs P at fixed DOF
    # (two separate y-quantities -> two stacked panels, shared x; never dual-axis)
    # median lines + individual seeds (bimodal seed outcomes)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 6.4), sharex=True)
    jit = (df['trial'] - df['trial'].mean()) * 0.045
    ax1.plot(summ['P'], summ['l2_relative_median'], marker='o', ms=5.5, lw=1.4,
             color='#d62728', label='median')
    ax1.scatter(df['P'] + jit, df['l2_relative'], s=14, facecolors='none',
                edgecolors='#d62728', linewidths=0.7, alpha=0.75)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'Relative $L_2$ error' + '\n(markers: individual seeds)')
    log_minor_labels(ax1)
    ax2.plot(summ['P'], summ['rms_r_xx_median'], marker='D', ms=5.5, lw=1.4,
             color='#e6550d', label=r'$\Vert \partial_{xx} r \Vert$ (median)')
    ax2.scatter(df['P'] + jit, df['rms_r_xx'], s=14, facecolors='none',
                edgecolors='#e6550d', linewidths=0.7, alpha=0.75)
    ax2.plot(summ['P'], summ['rms_r_median'], marker='o', ms=5.5, lw=1.4,
             color='#000000', label=r'$\Vert r \Vert$ (median)')
    ax2.scatter(df['P'] + jit, df['rms_r'], s=14, facecolors='none',
                edgecolors='#000000', linewidths=0.7, alpha=0.75)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Legendre order $P$  (panels $N = 210/(P+1)$)')
    ax2.set_ylabel('RMS')
    ax2.legend(fontsize=10)
    ax2.set_xticks(summ['P'])
    labels = ['%d\n(N=%d)' % (p, n) for p, n in zip(summ['P'], summ['N_panel'])]
    ax2.set_xticklabels(labels, fontsize=11)
    savefig(fig, os.path.join(OUT, 'tradeoff.png'))

    # report
    lines = []
    lines.append('# Task 3 — Fixed-DOF experiment: p- vs h-refinement of the LPA layer\n')
    lines.append('**Reviewer mapping:** Referee 2, Comment 3 (why higher P despite the '
                 'residual-suppression argument favoring P=2): expressivity vs residual '
                 'suppression at fixed DOF. Also supports Referee 1, Minor 1 (P/N selection '
                 'guidelines).\n')
    lines.append('## Setup\n')
    lines.append('- Total activation DOF fixed: (P+1) x N_panel = 210 (P=3 uses N=52, DOF=208).')
    lines.append('- Combos: ' + ', '.join('(P=%d, N=%d)' % (p, n) for p, n in
                 zip(summ['P'], summ['N_panel'])) + '.')
    lines.append('- Network 2x10, paper protocol, 5 trials each. Trainable-parameter count only '
                 'changes through N_panel (LPA has N_panel trainable weights).\n')
    lines.append('| P | N_panel | DOF | rel. L2 | median | success (<0.1) | abs. L1 | RMS r | RMS r_xx | params |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for _, r in summ.iterrows():
        lines.append('| %d | %d | %d | %s | %.2e | %d/%d | %s | %s | %s | %d |' % (
            r['P'], r['N_panel'], r['dof'],
            fmt_sci(r['l2_relative_mean'], r['l2_relative_std']),
            r['l2_relative_median'],
            round(r['success_rate_0.1'] * r['n_trials']), r['n_trials'],
            fmt_sci(r['l1_absolute_mean'], r['l1_absolute_std']),
            fmt_sci(r['rms_r_mean'], r['rms_r_std']),
            fmt_sci(r['rms_r_xx_mean'], r['rms_r_xx_std']),
            r['n_params_mean']))
    lines.append('\n## Interpretation guide\n')
    lines.append('- Moving right along P = p-refinement (higher-order basis, coarser panels); '
                 'moving left = h-refinement (finer panels, lower order).')
    lines.append('- The comparison shows whether accuracy gains stem from added spectral '
                 'expressivity (P) rather than from panel resolution (N) alone, at matched DOF.')
    lines.append('\n## Figures\n')
    lines.append('- `tradeoff.png` — solution error (top) and residual/residual-curvature RMS '
                 '(bottom) vs P at fixed DOF')
    with open(os.path.join(OUT, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[task3] wrote %s' % OUT)


if __name__ == '__main__':
    main()
