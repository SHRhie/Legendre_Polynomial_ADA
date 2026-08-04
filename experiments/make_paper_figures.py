"""Multi-panel replacement figures 2-4 for the revised manuscript.

Data: unified-protocol runs (results/ff_baseline aggregates for convergence
medians) + representative-seed field reruns (exp `fields_rep`; seed rule:
final rel-L2 closest to the cell median, recorded in
results/manuscript_revisions/representative_seeds.json).

Outputs -> results/manuscript_revisions/figures/
  fig2_replacement.png  Helmholtz: (a-c) convergence 2/3/4x10 + (d-f) fields
  fig3_replacement.png  Diffusion-reaction: (a-c) convergence + (d-f) profiles
  fig4_replacement.png  Kovasznay: (a) u-profiles + (b,c) convergence (rel-L2 of u)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig, panel_label
from agg_common import geo_mean_curves
import matplotlib.pyplot as plt

from paths import ROOT, RESULTS, runs_dir

HELM = runs_dir("helmholtz2d")
DIFF = runs_dir("diffusion_reaction")
KOVA = runs_dir("kovasznay")
OUT = f"{RESULTS}/manuscript_revisions/figures"
os.makedirs(OUT, exist_ok=True)

REPS = {(r["dir"], r["model"], r["nh"]): r["trial"]
        for r in json.load(open(f"{RESULTS}/manuscript_revisions/representative_seeds.json"))}

BLACK, RED = "#000000", "#d62728"


def acc(base, exp, stem):
    return np.loadtxt(f"{base}/{exp}/acc_hist_{stem}.txt", delimiter=",")


def median_curve(base, key_id, nh, trials, col):
    curves = []
    for t in trials:
        try:
            a = acc(base, "ff_baseline", f"{nh}_10_{key_id}_{t}")
            curves.append(a[:, col])
        except OSError:
            pass
    return geo_mean_curves(curves)


def conv_panel(ax, base, keys, nh, trials_map, col, labels):
    for key_id, color, label in keys:
        c = median_curve(base, key_id, nh, trials_map[key_id], col)
        ax.semilogy(np.arange(len(c)) * 50, c, color=color, lw=1.4, label=label)
    ax.set_xlabel("Iterations")
    ax.set_title(f"{nh} hidden layers", fontsize=13)


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    trials_map = {"R1": range(10), "LPA_P6_N30": range(10)}
    for j, nh in enumerate((2, 3, 4)):
        tm = {k: (range(10) if nh == 2 else range(5)) for k in trials_map}
        conv_panel(axes[0][j], HELM, [("R1", BLACK, "PINN (tanh)"),
                                      ("LPA_P6_N30", RED, "LPA-PINN")],
                   nh, tm, 1, None)
        panel_label(axes[0][j], "abc"[j], loc="upper right")
        if j == 0:
            axes[0][j].set_ylabel(r"Relative $L_2$ error")
    axes[0][2].legend(fontsize=11, loc="center left", bbox_to_anchor=(1.03, 0.5))
    # fields at 3x10 representative seed
    nh = 3
    t = REPS[("2D Helmholtz v2", "LPA", nh)]
    F = np.load(f"{HELM}/fields_rep/fields_{nh}_10_LPA_P6_N30_{t}.npz")
    xy, pred, ex = F["xy"], F["prediction"].reshape(-1), F["exact"].reshape(-1)
    n = int(np.sqrt(len(ex)))
    X = xy[:, 0].reshape(n, n); Y = xy[:, 1].reshape(n, n)
    panels = [(ex.reshape(n, n), "Exact $u$"),
              (pred.reshape(n, n), "LPA-PINN prediction"),
              (np.abs(pred - ex).reshape(n, n), "Absolute error")]
    for j, (Z, title) in enumerate(panels):
        ax = axes[1][j]
        cmap = "RdBu_r" if j < 2 else "magma"
        vmax = np.abs(panels[0][0]).max()
        kw = dict(vmin=-vmax, vmax=vmax) if j < 2 else {}
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", rasterized=True, **kw)
        fig.colorbar(im, ax=ax, shrink=0.9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("$x$")
        if j == 0:
            ax.set_ylabel("$y$")
        ax.set_aspect("equal")
        panel_label(ax, "def"[j], loc="upper left", color="white",
                    stroke="#333333")
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    savefig(fig, f"{OUT}/fig2_replacement.png")


# ---------------------------------------------------------------- Fig 3
def fig3():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for j, nh in enumerate((2, 3, 4)):
        tm = {"R": range(5), "LPA_P3_N30": range(5)}
        conv_panel(axes[0][j], DIFF, [("R", BLACK, "PINN (tanh)"),
                                      ("LPA_P3_N30", RED, "LPA-PINN")],
                   nh, tm, 1, None)
        panel_label(axes[0][j], "abc"[j], loc="upper right")
        if j == 0:
            axes[0][j].set_ylabel(r"Relative $L_2$ error")
    axes[0][2].legend(fontsize=11, loc="center left", bbox_to_anchor=(1.03, 0.5))
    # profiles u(x) at t in {0, 0.5, 1} for the 3x10 representative seed
    nh = 3
    t = REPS[("Diffusion-Reaction", "LPA", nh)]
    F = np.load(f"{DIFF}/fields_rep/fields_{nh}_10_LPA_P3_N30_{t}.npz")
    tx, pred, ex = F["tx"], F["prediction"].reshape(-1), F["exact"].reshape(-1)
    n = 100
    T = tx[:, 0].reshape(n, n); Xx = tx[:, 1].reshape(n, n)
    P = pred.reshape(n, n); E = ex.reshape(n, n)
    for j, tv in enumerate((0.0, 0.5, 1.0)):
        ax = axes[1][j]
        i = int(np.argmin(np.abs(T[:, 0] - tv)))
        t_actual = float(T[i, 0])  # label the actual grid coordinate
        ax.plot(Xx[i], E[i], "-", color=BLACK, lw=1.4, label="Exact")
        ax.plot(Xx[i], P[i], "--", color=RED, lw=1.4, label="LPA-PINN")
        ax.set_title(f"$t = {t_actual:.2f}$", fontsize=13)
        ax.set_xlabel("$x$")
        panel_label(ax, "def"[j], loc="upper right")
        if j == 0:
            ax.set_ylabel("$u(t,x)$")
            ax.legend(fontsize=10, loc="upper left")
    fig.subplots_adjust(hspace=0.35, wspace=0.28)
    savefig(fig, f"{OUT}/fig3_replacement.png")


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    # (a) u(y) profiles at several x for the 3x10 representative LPA seed
    nh = 3
    t = REPS[("Kovasznay flow", "LPA", nh)]
    F = np.load(f"{KOVA}/fields_rep/fields_{nh}_10_LPA_P6_N30_{t}.npz")
    xy, pred, ex = F["xy"], F["prediction"], F["exact"]
    xs = np.unique(xy[:, 0]); ys = np.unique(xy[:, 1])
    nx, ny = len(xs), len(ys)
    U_p = pred[:, 0].reshape(ny, nx)   # meshgrid(x,y) -> rows over y
    U_e = ex[:, 0].reshape(ny, nx)
    ax = axes[0]
    shades = ["#9ecae1", "#4292c6", "#08519c"]
    for k, xv in enumerate((-0.25, 0.25, 0.75)):
        i = int(np.argmin(np.abs(xs - xv)))
        x_actual = float(xs[i])  # label the actual grid coordinate
        ax.plot(ys, U_e[:, i], "-", color=shades[k], lw=1.4,
                label=f"Exact, $x={x_actual:.2f}$")
        ax.plot(ys, U_p[:, i], "--", color=shades[k], lw=1.4)
    ax.set_xlabel("$y$"); ax.set_ylabel("$u$")
    ax.set_title("Profiles (dashed: LPA-PINN)", fontsize=13)
    ax.legend(fontsize=9)
    panel_label(ax, "a", loc="upper right")
    # (b,c) convergence of the u-velocity relative L2 (column 3!)
    for j, nh in enumerate((3, 4)):
        tm = {"R": range(5), "LPA_P6_N30": range(5)}
        conv_panel(axes[j + 1], KOVA, [("R", BLACK, "PINN (tanh)"),
                                       ("LPA_P6_N30", RED, "LPA-PINN")],
                   nh, tm, 3, None)
        axes[j + 1].set_ylabel(r"Relative $L_2$ error ($u$)")
        panel_label(axes[j + 1], "bc"[j], loc="upper right")
    axes[2].legend(fontsize=11, loc="center left", bbox_to_anchor=(1.03, 0.5))
    fig.subplots_adjust(wspace=0.32)
    savefig(fig, f"{OUT}/fig4_replacement.png")


if __name__ == "__main__":
    apply_style()
    fig2(); fig3(); fig4()
    print("wrote", OUT)
