"""Build results/supplementary/ — the data files promised in the response.

Every file gets a header block with a title and a description, because IOP
requires a title + description for each supplementary item. The headers are
comment lines prefixed with '#', so the files still read as CSV.

Usage:  python build_supplementary.py
"""
import os

import numpy as np
import pandas as pd

from paths import ROOT, RESULTS as RES

OUT = f"{RES}/supplementary"
os.makedirs(OUT, exist_ok=True)

INDEX = []


def write(name, df, title, description, note=None):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        f.write(f"# Title: {title}\n")
        f.write(f"# Description: {description}\n")
        if note:
            f.write(f"# Note: {note}\n")
        f.write(f"# Rows: {len(df)}\n")
        df.to_csv(f, index=False)
    INDEX.append((name, title, description, len(df)))
    print(f"  {name:<44} {len(df):>5} rows")


# ------------------------------------------------------------------ S1 per-seed
def s1_per_seed():
    frames = []
    files = {
        "helmholtz2d": f"{RES}/ff_baseline/helmholtz2d_runs.csv",
        "diffusion_reaction": f"{RES}/ff_baseline/diffusion_reaction_runs.csv",
        "kovasznay": f"{RES}/ff_baseline/kovasznay_runs.csv",
    }
    keep = ["benchmark", "key_id", "nh", "nn", "trial", "seed", "lpa_order",
            "lpa_panels", "ff_sigma", "ff_features", "lr", "N_r", "N_b",
            "n_params", "l2_relative", "l1_absolute", "l2_relative_u",
            "l2_relative_v", "l2_relative_p", "l1_absolute_u", "l1_absolute_v",
            "l1_absolute_p", "lbfgs_steps", "final_loss"]
    for bench, p in files.items():
        df = pd.read_csv(p)
        cols = [c for c in keep if c in df.columns]
        frames.append(df[cols])
    for extra, p in (("low_order", f"{RES}/low_order/runs.csv"),
                     ("dof_fixed", f"{RES}/dof_fixed/runs.csv"),
                     ("sensitivity", f"{RES}/sensitivity/runs.csv")):
        df = pd.read_csv(p)
        df = df.assign(experiment=extra)
        cols = [c for c in keep + ["experiment", "P", "N_panel", "dof",
                                   "rms_r", "rms_r_xx"] if c in df.columns]
        frames.append(df[cols])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out.insert(0, "experiment", out.pop("experiment") if "experiment" in out else "ff_baseline")
    out["experiment"] = out["experiment"].fillna("ff_baseline")
    write("S1_per_seed_values.csv", out,
          "Per-seed results for every reported PINN configuration",
          "One row per training run behind Tables 1-3 and the order, "
          "panel-count and sensitivity studies: benchmark, architecture, model "
          "key, seed, hyperparameters, relative L2 and absolute L1 errors "
          "(component-wise for Kovasznay), L-BFGS iteration count and final "
          "loss. No run is excluded; table statistics are computed over exactly "
          "these rows.",
          "seed = 1234 + trial; key_id encodes the model and its hyperparameters")


# ------------------------------------------------ S2 optimizer termination
def s2_termination():
    df = pd.read_csv(f"{RES}/major1_compare/confirm_runs.csv")
    dev = pd.read_csv(f"{RES}/major1_compare/dev_runs.csv")
    cols = ["experiment", "model", "key_id", "nh", "nn", "trial", "init_seed",
            "data_seed", "feature_seed", "collocation_sha256", "n_params",
            "l2_relative", "lbfgs_success", "lbfgs_status", "lbfgs_message",
            "lbfgs_nit", "lbfgs_nfev", "time_lbfgs", "scipy", "tensorflow"]
    both = pd.concat([dev[[c for c in cols if c in dev.columns]],
                      df[[c for c in cols if c in df.columns]]], ignore_index=True)
    write("S2_optimizer_termination.csv", both,
          "SciPy L-BFGS-B termination status for the paired comparison campaign",
          "Development and held-out confirmation runs of the same-protocol "
          "comparison, with the optimizer success flag, status code, termination "
          "message, iteration and function-evaluation counts, the three "
          "independent seed streams and the SHA-256 checksum of the collocation "
          "point set used by every model within a trial.",
          "ftol = gtol = machine epsilon; the legacy 'factr' option is not a valid "
          "scipy.optimize.minimize argument and was never applied")


# ---------------------------------------------------- S3 paired statistics
def s3_paired_statistics():
    st = pd.read_csv(f"{RES}/major1_compare/paired_statistics.csv")
    summ = pd.read_csv(f"{RES}/major1_compare/confirm_summary.csv")
    keep = ["model", "key_id", "n", "n_params", "l2_median", "l2_q1", "l2_q3",
            "l2_mean", "l2_sd", "l2_geomean", "success_rate_0.1", "nit_median"]
    write("S3_paired_statistics.csv", st,
          "Paired statistics for the same-protocol comparison (Holm-adjusted)",
          "Per-comparison mean and median difference of log10 relative L2 over "
          "the ten held-out paired seeds, paired bootstrap 95% interval, exact "
          "Wilcoxon signed-rank p value and the Holm-adjusted p value across the "
          "five comparisons.",
          "n = 10 paired seeds; the proposed layer is the reference of each pair")
    write("S3b_comparison_summary.csv", summ[[c for c in keep if c in summ.columns]],
          "Per-model summary of the held-out comparison runs",
          "Median, quartiles, mean, standard deviation and geometric mean of the "
          "relative L2 error over the ten held-out seeds for each compared model, "
          "with parameter counts, the fraction of runs below the descriptive "
          "threshold 0.1 and the median L-BFGS iteration count.",
          "the 0.1 threshold is a descriptive reporting choice, not a "
          "pre-specified success criterion")


# --------------------------------------------------- S4 DeepONet intervals
def s4_deeponet_intervals():
    ci = pd.read_csv(f"{RES}/deeponet_rev/paired_ci.csv")
    write("S4_deeponet_paired_intervals.csv", ci,
          "Paired interval estimates for the operator-learning comparison",
          "Baseline-minus-proposed differences at each width over the five paired "
          "seeds, with both a paired bootstrap 95% interval and a Student-t 95% "
          "interval, for the reference Reynolds number and for the sweep median.",
          "the two interval methods disagree at n = 5, so the manuscript reports "
          "these comparisons descriptively rather than as significance claims")
    runs = pd.read_csv(f"{RES}/deeponet_rev/runs.csv")
    cols = ["benchmark", "key_id", "nh", "nn", "use_lpa", "lpa_order", "lpa_panels",
            "latent_dim", "head_width", "epochs", "lr", "n_int", "n_b", "seed",
            "trial", "n_params", "l2_u_ref", "l2_v_ref", "l2_p_ref",
            "sweep_median_l2_u", "sweep_median_l2_v", "sweep_median_l2_p",
            "sweep_mean_l2_u", "sweep_mean_l2_v", "sweep_mean_l2_p", "time_train"]
    write("S4b_deeponet_per_seed.csv", runs[[c for c in cols if c in runs.columns]],
          "Per-seed results of the PI-DeepONet re-run",
          "One row per operator-network training run: architecture, latent and "
          "head widths, Legendre order and panel count, seed, parameter count, "
          "errors at the reference Reynolds number and mean/median errors over "
          "the Re = 1-199 sweep, and training time.",
          "five seeds per width per model; all rows are included in the reported "
          "statistics")


# ------------------------------------------------- S5 Fourier warm-up sweep
def s5_ff_lr():
    comp = pd.read_csv(f"{RES}/ff_baseline/ff_supp_lr_comparison.csv")
    sig = pd.read_csv(f"{RES}/ff_baseline/ff_sigma_sweep.csv")
    write("S5_fourier_warmup_rate.csv", comp,
          "Fourier-feature baseline at two Adam warm-up learning rates",
          "Relative L2 and absolute L1 statistics for the parameter-matched "
          "random Fourier feature baseline at warm-up rates 1e-2 and 1e-3 on all "
          "three benchmarks, showing that its failure on the oscillatory problem "
          "is not an artefact of the warm-up rate.",
          "corresponds to results/ff_baseline/ff_supp_lr.png")
    write("S5b_fourier_sigma_sweep.csv", sig,
          "Fourier-feature baseline across frequency scales",
          "Relative L2 and absolute L1 statistics for the fixed random Fourier "
          "feature baseline at sigma in {1, 5, 10} on all three benchmarks and "
          "all tested depths.",
          "m = 3 frequency pairs, parameter-matched to the proposed layer")


def s6_kovasznay_nr():
    p = f"{RES}/ff_baseline/kovasznay_nr_comparison.csv"
    write("S6_kovasznay_collocation_check.csv", pd.read_csv(p),
          "Kovasznay collocation-budget robustness check",
          "Component-wise errors for the baseline and the proposed layer at the "
          "canonical budget N_r = 5000 and at N_r = 10000, confirming that the "
          "conclusion does not depend on the budget.",
          "the original submission trained the baseline at 5000 and the proposed "
          "layer at 10000; both are unified here")


def index_file():
    lines = ["# Supplementary data — MLST-105534 (revised)",
             "",
             "| File | Title | Rows |",
             "|---|---|---:|"]
    for name, title, desc, n in INDEX:
        lines.append(f"| `{name}` | {title} | {n} |")
    lines += ["", "Each CSV carries its own title, description and row count in "
                  "comment lines (`#`) at the top.", "",
              "Generated by `revision/build_supplementary.py` from the CSVs under "
              "`results/`; the same numbers appear in `results/final_numbers.json`.",
              ""]
    for name, title, desc, n in INDEX:
        lines += [f"### {name}", f"**{title}**", "", desc, ""]
    open(f"{OUT}/README.md", "w").write("\n".join(lines))
    print(f"  README.md written with {len(INDEX)} entries")


if __name__ == "__main__":
    print("building supplementary files:")
    s1_per_seed(); s2_termination(); s3_paired_statistics()
    s4_deeponet_intervals(); s5_ff_lr(); s6_kovasznay_nr()
    index_file()
    print(f"-> {OUT}")
