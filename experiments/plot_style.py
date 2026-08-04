"""Shared matplotlib style for all revision figures.

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

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

_LOC = {
    'upper right': (1.0, 1.0, 'right', 'top'),
    'upper left': (0.0, 1.0, 'left', 'top'),
    'lower right': (1.0, 0.0, 'right', 'bottom'),
    'lower left': (0.0, 0.0, 'left', 'bottom'),
}


def panel_label(ax, letter, loc='upper right', color='black', fontsize=16,
                pad=0.04, stroke='white'):
    """Paper-style panel label "(a)" placed inside the axes.

    Matches the original manuscript figures: serif, plain weight, inside the
    frame (top-right for line plots, top-left in white over field images).
    `stroke` outlines the glyphs so the label stays legible where a curve or
    a colormap passes underneath.
    """
    import matplotlib.patheffects as pe
    x, y, ha, va = _LOC[loc]
    dx = pad if ha == 'left' else -pad
    dy = -pad if va == 'top' else pad
    t = ax.text(x + dx, y + dy, '(%s)' % letter, transform=ax.transAxes,
                ha=ha, va=va, fontsize=fontsize, color=color, zorder=5)
    if stroke:
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground=stroke)])
    return t


def label_panels(axes, loc='upper right', start=0, **kw):
    """Label a flat sequence of axes (a), (b), (c), ... in order."""
    for k, ax in enumerate(axes):
        panel_label(ax, LETTERS[start + k], loc=loc, **kw)
