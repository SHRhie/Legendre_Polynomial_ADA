"""Shared helpers for aggregating revision experiment runs."""
import glob
import json
import os

import numpy as np
import pandas as pd

from paths import ROOT, RESULTS, bench_dir, runs_dir

BENCH_DIRS = {k: bench_dir(k) for k in
              ('helmholtz2d', 'diffusion_reaction', 'kovasznay')}
BENCH_LABELS = {
    'helmholtz2d': '2D Helmholtz',
    'diffusion_reaction': 'Diffusion-reaction',
    'kovasznay': 'Kovasznay flow',
}


# fixed model order and colors (R black like the paper; FF = blue steps by sigma; LPA red)
# R1 = vanilla with single-output head (like-for-like metric with LPA/FF on Helmholtz)
MODEL_ORDER = ['R', 'R1', 'FF_s1', 'FF_s5', 'FF_s10', 'LPA']
MODEL_LABELS = {
    'R': 'PINN (tanh)',
    'R1': 'PINN (tanh, 1-out)',
    'FF_s1': r'FF-PINN ($\sigma$=1)',
    'FF_s5': r'FF-PINN ($\sigma$=5)',
    'FF_s10': r'FF-PINN ($\sigma$=10)',
    'LPA': 'LPA-PINN',
}
MODEL_COLORS = {
    'R': '#000000',
    'R1': '#7f7f7f',
    'FF_s1': '#9ecae1',
    'FF_s5': '#4292c6',
    'FF_s10': '#08519c',
    'LPA': '#d62728',
}


def load_runs(bench, exp):
    """Load all run_*.json for a benchmark/experiment into a DataFrame."""
    d = runs_dir(bench, exp)
    rows = []
    for p in sorted(glob.glob(os.path.join(d, 'run_*.json'))):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def load_acc_hist(bench, exp, nh, nn, key_id, trial):
    p = os.path.join(runs_dir(bench, exp),
                     'acc_hist_%s_%s_%s_%s.txt' % (nh, nn, key_id, trial))
    if not os.path.exists(p):
        return None
    return np.loadtxt(p, delimiter=',')


def mean_std(df, group_cols, value_cols):
    g = df.groupby(group_cols)[value_cols]
    m = g.mean().add_suffix('_mean')
    s = g.std(ddof=1).add_suffix('_std')
    med = g.median().add_suffix('_median')
    n = g.size().rename('n_trials')
    return pd.concat([m, s, med, n], axis=1).reset_index()


def add_success_rate(df, group_cols, err_col, thresh=0.1):
    """Fraction of trials with err_col below thresh (useful for bimodal seeds)."""
    ok = df.assign(_ok=(df[err_col] < thresh).astype(float))
    return (ok.groupby(group_cols)['_ok'].mean()
            .rename('success_rate_%g' % thresh).reset_index())


def geo_mean_curves(curves):
    """Geometric mean over trials of error-vs-checkpoint curves.
    Curves may have different lengths; each is padded by holding its last value."""
    L = max(len(c) for c in curves)
    padded = np.stack([np.concatenate([c, np.full(L - len(c), c[-1])]) for c in curves])
    return np.exp(np.mean(np.log(np.maximum(padded, 1e-16)), axis=0))


def fmt_sci(mean, std):
    if np.isnan(std):
        return '%.2e' % mean
    return '%.2e ± %.1e' % (mean, std)
