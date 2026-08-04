"""Shared matplotlib style for all figures.

Conventions (fixed by the paper):
  Times New Roman, mathtext.stix, font.size=14, figure.dpi=400,
  4-direction inward ticks, frameless legend, bbox_inches='tight' on save.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'font.size': 14,
        'figure.dpi': 400,
        'savefig.dpi': 400,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,
        'axes.linewidth': 0.8,
    })


def savefig(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def log_safe_yerr(mean, std):
    """Asymmetric yerr for log-scale errorbar plots: the lower whisker is clipped
    so mean - err_low stays positive (mean-std <= 0 would run off the axis)."""
    import numpy as np
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    low = np.minimum(std, mean * 0.8)
    return np.vstack([low, std])


def log_minor_labels(ax, axis='y'):
    """Label 2x/5x minor ticks on a log axis so sub-decade values can be read."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0), numticks=20))
    a.set_minor_formatter(FuncFormatter(lambda v, _: '%g' % v))
    for t in a.get_minorticklabels():
        t.set_fontsize(10)
