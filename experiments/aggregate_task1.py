"""Task 1 aggregation: vanilla PINN vs FF-PINN (sigma sweep) vs LPA, all benchmarks.

Outputs to <repo>/results/ff_baseline/:
  <bench>_runs.csv        raw per-run records
  <bench>_summary.csv     mean/std per (nh,nn,model)
  ff_sigma_sweep.csv      FF sigma comparison + best sigma per benchmark
  conv_<bench>.png        convergence curves at (2,10)
  bars_<bench>.png        final L2 relative error bars per condition
  report.md
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agg_common import (BENCH_LABELS, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER,
                        RESULTS, fmt_sci, geo_mean_curves, load_acc_hist,
                        load_runs, mean_std)
from plot_style import apply_style, log_minor_labels, savefig
import matplotlib.pyplot as plt

OUT = os.path.join(RESULTS, 'ff_baseline')
os.makedirs(OUT, exist_ok=True)

BENCHES = ['helmholtz2d', 'diffusion_reaction', 'kovasznay']
ERR_COL = {'helmholtz2d': 'l2_relative', 'diffusion_reaction': 'l2_relative',
           'kovasznay': 'l2_relative_u'}
L1_COL = {'helmholtz2d': 'l1_absolute', 'diffusion_reaction': 'l1_absolute',
          'kovasznay': 'l1_absolute_u'}
ACC_L2_IDX = {'helmholtz2d': 1, 'diffusion_reaction': 1, 'kovasznay': 3}
CONDITIONS = [(2, 10), (3, 10), (4, 10)]


def model_of(key_id):
    if key_id.startswith('LPA'):
        return 'LPA'
    return key_id


def main():
    apply_style()
    all_summaries = {}
    for bench in BENCHES:
        df = load_runs(bench, 'ff_baseline')
        if df.empty:
            print('[task1] %s: no runs yet' % bench)
            continue
        df['model'] = df['key_id'].map(model_of)
        df.to_csv(os.path.join(OUT, '%s_runs.csv' % bench), index=False)

        vcols = [ERR_COL[bench], L1_COL[bench], 'n_params', 'time_lbfgs']
        if bench == 'kovasznay':
            vcols += ['l2_relative_v', 'l2_relative_p', 'l1_absolute_v', 'l1_absolute_p']
        summ = mean_std(df, ['nh', 'nn', 'model'], vcols)
        summ['model'] = pd.Categorical(summ['model'], MODEL_ORDER, ordered=True)
        summ = summ.sort_values(['nh', 'nn', 'model'])
        summ.to_csv(os.path.join(OUT, '%s_summary.csv' % bench), index=False)
        all_summaries[bench] = summ

        # convergence plot at (2,10)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        sub = df[(df.nh == 2) & (df.nn == 10)]
        for model in MODEL_ORDER:
            trials = sub[sub.model == model]
            curves = []
            for _, row in trials.iterrows():
                acc = load_acc_hist(bench, 'ff_baseline', 2, 10, row['key_id'], row['trial'])
                if acc is None or acc.ndim != 2:
                    continue
                curves.append(acc[:, ACC_L2_IDX[bench]])
            if not curves:
                continue
            mcurve = geo_mean_curves(curves)
            x = np.arange(len(mcurve)) * 50
            ax.semilogy(x, mcurve, color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model], lw=1.4,
                        ls='--' if model == 'FF_s10' else '-')
        ax.set_xlabel('Iterations')
        ax.set_ylabel(r'Relative $L_2$ error' + (' (u)' if bench == 'kovasznay' else ''))
        log_minor_labels(ax)
        ax.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5))
        savefig(fig, os.path.join(OUT, 'conv_%s.png' % bench))

        # bar plot per condition
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        width = 0.15
        xs = np.arange(len(CONDITIONS))
        for j, model in enumerate(MODEL_ORDER):
            means, stds = [], []
            for nh, nn in CONDITIONS:
                row = summ[(summ.nh == nh) & (summ.nn == nn) & (summ.model == model)]
                if row.empty:
                    means.append(np.nan); stds.append(0.0)
                else:
                    means.append(row['%s_mean' % ERR_COL[bench]].iloc[0])
                    stds.append(row['%s_std' % ERR_COL[bench]].iloc[0])
            ax.bar(xs + (j - 2) * width, means, width * 0.92, yerr=stds, capsize=2,
                   color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                   error_kw={'lw': 0.8})
        ax.set_yscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels(['%d x %d' % c for c in CONDITIONS])
        ax.set_xlabel('Hidden layers x neurons')
        ax.set_ylabel(r'Relative $L_2$ error' + (' (u)' if bench == 'kovasznay' else ''))
        log_minor_labels(ax)
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
        savefig(fig, os.path.join(OUT, 'bars_%s.png' % bench))

    # sigma sweep summary + best sigma
    sweep_rows = []
    for bench, summ in all_summaries.items():
        for _, r in summ[summ.model.astype(str).str.startswith('FF')].iterrows():
            sweep_rows.append({
                'benchmark': bench, 'nh': r['nh'], 'nn': r['nn'],
                'sigma': float(str(r['model']).split('_s')[1]),
                'l2_mean': r['%s_mean' % ERR_COL[bench]],
                'l2_std': r['%s_std' % ERR_COL[bench]],
                'n_trials': r['n_trials'],
            })
    sweep = pd.DataFrame(sweep_rows)
    best = None
    if not sweep.empty:
        best = (sweep[(sweep.nh == 2) & (sweep.nn == 10)]
                .sort_values(['benchmark', 'l2_mean']).groupby('benchmark').head(1))
        sweep.to_csv(os.path.join(OUT, 'ff_sigma_sweep.csv'), index=False)

    # ---- supplementary: FF with Adam warm-up lr 1e-3 --------------------
    supp_rows = []
    for bench in BENCHES:
        supp = load_runs(bench, 'ff_supp')
        if supp.empty:
            continue
        supp['sigma'] = supp['ff_sigma']
        ssumm = mean_std(supp[(supp.nh == 2) & (supp.nn == 10)], ['sigma'],
                         [ERR_COL[bench], L1_COL[bench]])
        for _, r in ssumm.iterrows():
            supp_rows.append({'benchmark': bench, 'sigma': r['sigma'], 'lr': 1e-3,
                              'l2_mean': r['%s_mean' % ERR_COL[bench]],
                              'l2_std': r['%s_std' % ERR_COL[bench]],
                              'l1_mean': r['%s_mean' % L1_COL[bench]],
                              'l1_std': r['%s_std' % L1_COL[bench]],
                              'n_trials': r['n_trials']})
    supp_df = pd.DataFrame(supp_rows)
    if not supp_df.empty:
        # main-protocol FF rows at (2,10) for side-by-side comparison
        main_rows = []
        for bench, summ in all_summaries.items():
            sub = summ[(summ.nh == 2) & (summ.nn == 10) &
                       summ.model.astype(str).str.startswith('FF')]
            for _, r in sub.iterrows():
                main_rows.append({'benchmark': bench,
                                  'sigma': float(str(r['model']).split('_s')[1]),
                                  'lr': 1e-2,
                                  'l2_mean': r['%s_mean' % ERR_COL[bench]],
                                  'l2_std': r['%s_std' % ERR_COL[bench]],
                                  'l1_mean': r['%s_mean' % L1_COL[bench]],
                                  'l1_std': r['%s_std' % L1_COL[bench]],
                                  'n_trials': r['n_trials']})
        comp = pd.concat([pd.DataFrame(main_rows), supp_df], ignore_index=True)
        comp = comp.sort_values(['benchmark', 'sigma', 'lr'])
        comp.to_csv(os.path.join(OUT, 'ff_supp_lr_comparison.csv'), index=False)

        benches_here = [b for b in BENCHES if b in set(comp.benchmark)]
        fig, axes = plt.subplots(1, len(benches_here),
                                 figsize=(3.4 * len(benches_here), 3.6), squeeze=False)
        fig.subplots_adjust(wspace=0.45)
        lr_colors = {1e-2: '#4292c6', 1e-3: '#fd8d3c'}
        lr_labels = {1e-2: 'Adam lr $10^{-2}$', 1e-3: 'Adam lr $10^{-3}$'}
        for k, bench in enumerate(benches_here):
            ax = axes[0][k]
            cb = comp[comp.benchmark == bench]
            sigmas = sorted(cb.sigma.unique())
            xs = np.arange(len(sigmas))
            for j, lr in enumerate([1e-2, 1e-3]):
                means, stds = [], []
                for s in sigmas:
                    row = cb[(cb.sigma == s) & (np.isclose(cb.lr, lr))]
                    means.append(row['l2_mean'].iloc[0] if not row.empty else np.nan)
                    stds.append(row['l2_std'].iloc[0] if not row.empty else 0.0)
                ax.bar(xs + (j - 0.5) * 0.32, means, 0.30, yerr=stds, capsize=2,
                       color=lr_colors[lr], label=lr_labels[lr] if k == 0 else None,
                       error_kw={'lw': 0.8})
            ax.set_yscale('log')
            ax.set_xticks(xs)
            ax.set_xticklabels(['%g' % s for s in sigmas])
            ax.set_xlabel(r'$\sigma$')
            if k == 0:
                ax.set_ylabel(r'Relative $L_2$ error')
                ax.legend(fontsize=9)
            ax.set_title(BENCH_LABELS[bench], fontsize=12)
        savefig(fig, os.path.join(OUT, 'ff_supp_lr.png'))

    # ---- supplementary: Kovasznay N_r robustness (5000 vs 10000) --------
    nr10k = load_runs('kovasznay', 'ff_nr10k')
    nr_comp = None
    if not nr10k.empty:
        nr10k['model'] = nr10k['key_id'].map(model_of)
        s10k = mean_std(nr10k[(nr10k.nh == 2) & (nr10k.nn == 10)], ['model'],
                        ['l2_relative_u', 'l2_relative_v', 'l2_relative_p'])
        s10k['N_r'] = 10000
        if 'kovasznay' in all_summaries:
            s5k = all_summaries['kovasznay']
            s5k = s5k[(s5k.nh == 2) & (s5k.nn == 10)][
                ['model', 'l2_relative_u_mean', 'l2_relative_u_std',
                 'l2_relative_u_median', 'n_trials']].copy()
            s5k['N_r'] = 5000
            nr_comp = pd.concat([s5k, s10k[['model', 'l2_relative_u_mean',
                                            'l2_relative_u_std', 'l2_relative_u_median',
                                            'n_trials', 'N_r']]], ignore_index=True)
            nr_comp['model'] = pd.Categorical(nr_comp['model'], MODEL_ORDER, ordered=True)
            nr_comp = nr_comp.sort_values(['model', 'N_r'])
            nr_comp.to_csv(os.path.join(OUT, 'kovasznay_nr_comparison.csv'), index=False)

    # ---- report ---------------------------------------------------------
    lines = []
    lines.append('# Task 1 — Fourier-feature baseline (FF-PINN) vs vanilla PINN vs LPA\n')
    lines.append('**Reviewer mapping:** Referee 2, Comment 4 (comparison against a Fourier '
                 'feature layer); also supports Referee 1, Major 1 (positioning vs Fourier '
                 'feature mappings) and Referee 1, Minor 3 (cases where LPA does not outperform '
                 '— see Kovasznay parity).\n')
    lines.append('## Setup\n')
    lines.append('- FF mapping gamma(x) = [cos(2 pi B x), sin(2 pi B x)], B ~ N(0, sigma^2) fixed '
                 '(Tancik et al. 2020), applied after the [-1,1] input scaling of the same MLP as '
                 'the vanilla PINN; sigma in {1, 5, 10}.')
    lines.append('- Number of Fourier frequencies m chosen so trainable parameters match the '
                 'LPA model at the same width/depth (m=3 for all benchmarks; params within ~6%).')
    lines.append('- Identical training protocol for all models: Adam 200 steps (lr 1e-2) + '
                 'L-BFGS-B (maxiter 40000, float32), 5 trials (seeds 1234-1238).')
    lines.append('- LPA reference config: Helmholtz/Kovasznay P=6, N_panel=30; diffusion-reaction '
                 'P=3, N_panel=30 (paper defaults).\n')
    for bench, summ in all_summaries.items():
        lines.append('\n## %s\n' % BENCH_LABELS[bench])
        ec = ERR_COL[bench]
        lc = L1_COL[bench]
        hdr = '| layers x neurons | model | rel. L2 %s | median | abs. L1 %s | params | n |' % (
            '(u)' if bench == 'kovasznay' else '', '(u)' if bench == 'kovasznay' else '')
        lines.append(hdr)
        lines.append('|---|---|---|---|---|---|---|')
        for _, r in summ.iterrows():
            lines.append('| %dx%d | %s | %s | %.2e | %s | %d | %d |' % (
                r['nh'], r['nn'], MODEL_LABELS[str(r['model'])],
                fmt_sci(r['%s_mean' % ec], r['%s_std' % ec]),
                r['%s_median' % ec],
                fmt_sci(r['%s_mean' % lc], r['%s_std' % lc]),
                r['n_params_mean'], r['n_trials']))
        if bench == 'kovasznay':
            lines.append('\n(u, v, p 별 상세 값은 `%s_summary.csv` 참고)' % bench)
    if best is not None and not best.empty:
        lines.append('\n## Best FF sigma per benchmark (at 2x10)\n')
        lines.append('| benchmark | best sigma | rel. L2 (mean ± std) |')
        lines.append('|---|---|---|')
        for _, r in best.iterrows():
            lines.append('| %s | %g | %s |' % (BENCH_LABELS[r['benchmark']], r['sigma'],
                                               fmt_sci(r['l2_mean'], r['l2_std'])))
    lines.append('\n## Figures\n')
    for bench in all_summaries:
        lines.append('- `conv_%s.png` — convergence (rel. L2 vs iterations, geometric mean over '
                     'trials, 2x10)' % bench)
        lines.append('- `bars_%s.png` — final rel. L2 per condition (mean ± std)' % bench)
    if not supp_df.empty:
        lines.append('\n## Supplementary — FF-PINN with Adam warm-up lr 1e-3 (2x10)\n')
        lines.append('Best-shot check that the FF behavior under the default protocol is not an '
                     'artifact of the aggressive Adam warm-up (L-BFGS-B stage is identical).\n')
        lines.append('| benchmark | sigma | Adam lr | rel. L2 | abs. L1 |')
        lines.append('|---|---|---|---|---|')
        for _, r in comp.iterrows():
            lines.append('| %s | %g | %g | %s | %s |' % (
                BENCH_LABELS[r['benchmark']], r['sigma'], r['lr'],
                fmt_sci(r['l2_mean'], r['l2_std']), fmt_sci(r['l1_mean'], r['l1_std'])))
        lines.append('\n- `ff_supp_lr.png` — side-by-side comparison; raw table in '
                     '`ff_supp_lr_comparison.csv`.')
    if nr_comp is not None:
        lines.append('\n## Supplementary — Kovasznay collocation-count robustness (2x10)\n')
        lines.append('The repository scripts differ: main_run_R.py uses N_r=5000 while '
                     'main_run_LPA.py uses N_r=10000. The primary table above uses a unified '
                     'N_r=5000 for all models; this check re-runs all models at N_r=10000.\n')
        lines.append('| model | N_r | rel. L2 (u) | median | n |')
        lines.append('|---|---|---|---|---|')
        for _, r in nr_comp.iterrows():
            lines.append('| %s | %d | %s | %.2e | %d |' % (
                MODEL_LABELS[str(r['model'])], r['N_r'],
                fmt_sci(r['l2_relative_u_mean'], r['l2_relative_u_std']),
                r['l2_relative_u_median'], r['n_trials']))
        lines.append('\n- Full u/v/p columns in `kovasznay_nr_comparison.csv`.')
    lines.append('\n## Notes\n')
    lines.append('- The published vanilla-PINN (R) model for Helmholtz uses a 3-output head '
                 '(scalar problem); metrics reported here follow the published convention. '
                 'Channel-0 / channel-sum variants are stored in the runs CSV '
                 '(`l2_relative_c0`, `l2_relative_sum`).')
    lines.append('- FF runs that plateau are terminated by the same ftol criterion as all other '
                 'models (identical protocol).')
    lines.append('- Diffusion-reaction runs use x in [-pi, pi] (paper text / main_run_LPA.py). '
                 'NOTE: main_run_R.py in the repository uses L=1, under which the exact solution '
                 'violates the enforced BCs and every model floors at ~5.7e-1 — the published '
                 'Table 2 baseline value. Provenance of the published Table 2 baseline should be '
                 'checked before the rebuttal (see results/README.md).')
    with open(os.path.join(OUT, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('[task1] wrote %s' % OUT)


if __name__ == '__main__':
    main()
