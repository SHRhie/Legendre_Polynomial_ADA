"""Aggregation, frozen-rule selection, statistics and verification for the
Referee 1 / Major 1 paired-baseline campaign.

Usage:
  python aggregate_major1.py dev       # Phase A: dev_runs.csv + dev_selection.csv
  python aggregate_major1.py confirm   # Phase B: summaries, stats, plots, checks

All rules implement results/major1_compare/preregistration.md verbatim.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_style, savefig
import matplotlib.pyplot as plt

import os as _os
HELM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "2D_Helmholtz")
OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results", "major1_compare")
CLIP_LO, CLIP_HI = 1e-12, 1e3
BOOT_SEED = 20260803
DEV_TRIALS = {0, 1, 2}
CONFIRM_TRIALS = set(range(10, 20))
GENERAL_MODELS = ["TANH", "LPA", "FF", "TFF", "SIREN", "NLAAF"]
MODEL_LABELS = {
    "TANH": "tanh PINN", "LPA": "LPA (P=6, N=30)", "FF": "fixed random FF",
    "TFF": "trainable FF", "SIREN": "SIREN", "NLAAF": "N-LAAF",
    "FF_ORACLE": "oracle FF (ceiling)",
}
MODEL_COLORS = {
    "TANH": "#000000", "LPA": "#d62728", "FF": "#9ecae1", "TFF": "#08519c",
    "SIREN": "#2ca02c", "NLAAF": "#9467bd", "FF_ORACLE": "#7f7f7f",
}
EXPECTED_SHA = {
    0: "ae1801d2a9f18a1b9981ce1c55acdeb6b170567b1f7de9ba4ea01fb4f9925a6d",
    1: "be5301dba3d39d76d823a45f67800d51fdae62a838a4820c5d8ac789073c0834",
    2: "7b0a65d90daf8e9405bb75d2b6ad038ab16e09d7961d840747a440941bf50fc3",
    10: "0fb74df0bd2f88ad4f28fa6044d6367e61832782da47f85ebf92624beafc27a2",
    11: "6c909b202f01334e613d04b74d41a114be0217af8040016bae2c0f3de9acc3d6",
    12: "7b83f0057d18869ea06d4504b6fd4bc4aaf7ac20e82e926a191ce9e4f079eab9",
    13: "e073d1b3f95374617ba10f0de47aee81d7469b5bc9e8fbc3d39738dec55f96ff",
    14: "e8d15ee310a129e8292139a604196cba36aa295bee8a52e3a38c89f79bfa94be",
    15: "88dcfbd6a03e399ca9655f79269b8582235f702d946f2aa4b9d5e6c2a6fb0831",
    16: "4e288d32d0769cd961971cda4eeb15aa361c9e9f094bde2ff924238e22aaf387",
    17: "4436e43d0db65656c1980747481e84488f5e71c8f0961d65afdd705d5f7fda4a",
    18: "9b631f7ae65ead03483252b5468db4041312e681781777fe4cdbab46b853c036",
    19: "2e00166dc02649b405f475b0e68228b61ca1a56e8ff55eb5f4499e6ddaca7978",
}


def load_runs(exp):
    rows = []
    for p in sorted(glob.glob(os.path.join(HELM, "results", "runs", exp, "run_*.json"))):
        with open(p) as f:
            r = json.load(f)
        r["_path"] = p
        rows.append(r)
    return pd.DataFrame(rows)


def clipped_log10(err):
    if err is None or not np.isfinite(err):
        err = CLIP_HI
    return float(np.log10(np.clip(err, CLIP_LO, CLIP_HI)))


def verify_common(df, allowed_trials, exp):
    problems = []
    # 1) checksums per trial: single unique value AND matches expectation
    for t, g in df.groupby("trial"):
        s = set(g["collocation_sha256"])
        if len(s) != 1:
            problems.append(f"trial {t}: {len(s)} distinct checksums")
        elif EXPECTED_SHA.get(int(t)) and s.pop() != EXPECTED_SHA[int(t)]:
            problems.append(f"trial {t}: checksum != preregistered value")
    # 2) trial set
    bad = set(df["trial"]) - allowed_trials
    if bad:
        problems.append(f"unexpected trials {sorted(bad)} in {exp}")
    # 3) duplicates
    dup = df.duplicated(subset=["key_id", "trial"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (key_id, trial) rows")
    # 4) finite metrics
    nf = df[~np.isfinite(df["l2_relative"].astype(float))]
    if len(nf):
        problems.append(f"{len(nf)} non-finite l2_relative rows")
    # 5) acc_hist last row consistency (tolerance: metrics recomputed identically)
    for _, r in df.iterrows():
        acc_path = r["_path"].replace("run_", "acc_hist_").replace(".json", ".txt")
        try:
            acc = np.loadtxt(acc_path, delimiter=",")
            last_l2 = acc[-1, 1] if acc.ndim == 2 else acc[1]
            if not np.isclose(last_l2, r["l2_relative"], rtol=1e-6, atol=1e-12):
                problems.append(f"{os.path.basename(r['_path'])}: acc_hist last l2 "
                                f"{last_l2:.3e} != json {r['l2_relative']:.3e}")
        except Exception as e:
            problems.append(f"{os.path.basename(acc_path)}: unreadable ({e})")
    return problems


def phase_dev():
    os.makedirs(OUT, exist_ok=True)
    df = load_runs("major1_dev")
    if df.empty:
        print("no dev runs")
        return
    problems = verify_common(df, DEV_TRIALS, "major1_dev")
    df.drop(columns=["_path"]).to_csv(os.path.join(OUT, "dev_runs.csv"), index=False)

    rows = []
    for (model, key), g in df.groupby(["model", "key_id"]):
        logs = [clipped_log10(v) for v in g["l2_relative"]]
        # missing trials count as CLIP_HI per the frozen rule
        for _ in range(3 - len(logs)):
            logs.append(clipped_log10(None))
        logs = np.array(sorted(logs))
        rows.append({
            "model": model, "key_id": key, "n_runs": len(g),
            "median_log10": float(np.median(logs)),
            "iqr_log10": float(np.percentile(logs, 75) - np.percentile(logs, 25)),
            "n_params": int(g["n_params"].iloc[0]),
            "l2_values": ";".join(f"{v:.6e}" for v in g.sort_values("trial")["l2_relative"]),
        })
    sel = pd.DataFrame(rows)
    sel["selected"] = False
    for model in GENERAL_MODELS:
        sub = sel[sel.model == model].sort_values(
            ["median_log10", "iqr_log10", "n_params"]).head(1)
        sel.loc[sub.index, "selected"] = True
    sel = sel.sort_values(["model", "median_log10"])
    sel.to_csv(os.path.join(OUT, "dev_selection.csv"), index=False)
    print(sel[sel.selected][["model", "key_id", "median_log10", "iqr_log10", "n_params"]]
          .to_string(index=False))
    if problems:
        print("\nVERIFICATION PROBLEMS:")
        for p in problems:
            print(" -", p)
    else:
        print("\nverification: OK (checksums, trials, duplicates, finiteness, acc_hist)")


def bootstrap_ci_median(x, n_boot=10000, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    med = [np.median(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def bootstrap_ci_mean(x, n_boot=10000, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    m = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def phase_confirm():
    from scipy.stats import wilcoxon
    apply_style()
    os.makedirs(OUT, exist_ok=True)
    df = load_runs("major1_confirm")
    if df.empty:
        print("no confirm runs")
        return
    problems = verify_common(df, CONFIRM_TRIALS, "major1_confirm")
    # no dev/confirm mixing
    dev_overlap = set(df["trial"]) & DEV_TRIALS
    if dev_overlap:
        problems.append(f"dev trials {dev_overlap} present in confirm data")
    df = df.sort_values(["model", "trial"])
    df.drop(columns=["_path"]).to_csv(os.path.join(OUT, "confirm_runs.csv"), index=False)

    # ---- per-model summary ------------------------------------------------
    rows = []
    for model, g in df.groupby("model"):
        errs = g["l2_relative"].astype(float).values
        logs = np.array([clipped_log10(v) for v in errs])
        q1, q3 = np.percentile(errs, [25, 75])
        lo, hi = bootstrap_ci_median(errs)
        status_counts = g["lbfgs_status"].value_counts().to_dict()
        rows.append({
            "model": model, "key_id": g["key_id"].iloc[0], "n": len(g),
            "n_params": int(g["n_params"].iloc[0]),
            "l2_median": float(np.median(errs)), "l2_q1": float(q1), "l2_q3": float(q3),
            "l2_iqr": float(q3 - q1),
            "l2_mean": float(np.mean(errs)), "l2_sd": float(np.std(errs, ddof=1)),
            "l2_geomean": float(10 ** np.mean(logs)),
            "l2_median_ci95_lo": lo, "l2_median_ci95_hi": hi,
            "success_rate_0.1": float(np.mean(errs < 0.1)),
            "common_loss_median": float(np.median(g["common_pde_boundary_loss"])),
            "opt_success_rate": float(np.mean(g["lbfgs_success"])),
            "opt_status_counts": json.dumps(status_counts),
            "nit_median": float(np.median(g["lbfgs_nit"])),
            "nfev_median": float(np.median(g["lbfgs_nfev"])),
            "raw_l2": ";".join(f"{v:.6e}" for v in errs),
        })
    summ = pd.DataFrame(rows).sort_values("l2_median")
    summ.to_csv(os.path.join(OUT, "confirm_summary.csv"), index=False)

    # ---- paired statistics: LPA vs each competitor ------------------------
    piv = df[df.model.isin(GENERAL_MODELS)].pivot_table(
        index="trial", columns="model", values="l2_relative")
    stats_rows = []
    competitors = [m for m in GENERAL_MODELS if m != "LPA" and m in piv.columns]
    pvals = []
    for comp in competitors:
        d = np.log10(np.clip(piv["LPA"], CLIP_LO, CLIP_HI)) - \
            np.log10(np.clip(piv[comp], CLIP_LO, CLIP_HI))
        d = d.dropna().values
        lo, hi = bootstrap_ci_mean(d)
        try:
            w = wilcoxon(d, alternative="two-sided", mode="exact")
            p = float(w.pvalue)
        except Exception:
            p = float("nan")
        pvals.append(p)
        stats_rows.append({
            "comparison": f"LPA - {comp}", "n_pairs": len(d),
            "mean_log10_diff": float(np.mean(d)),
            "median_log10_diff": float(np.median(d)),
            "paired_bootstrap_ci95_lo": lo, "paired_bootstrap_ci95_hi": hi,
            "wilcoxon_exact_p": p,
            "direction": "LPA better" if np.median(d) < 0 else "LPA worse",
        })
    # Holm correction over the 5 comparisons
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    for row, a in zip(stats_rows, adj):
        row["holm_adjusted_p"] = float(a)
    pstats = pd.DataFrame(stats_rows)
    pstats.to_csv(os.path.join(OUT, "paired_statistics.csv"), index=False)

    # ---- plots -------------------------------------------------------------
    order_models = [m for m in GENERAL_MODELS if m in set(df.model)]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, model in enumerate(order_models + (["FF_ORACLE"] if "FF_ORACLE" in set(df.model) else [])):
        g = df[df.model == model]
        errs = g["l2_relative"].astype(float).values
        x = np.full(len(errs), i, dtype=float)
        x += (g["trial"].values - 14.5) * 0.018
        ax.scatter(x, errs, s=18, facecolors="none",
                   edgecolors=MODEL_COLORS.get(model, "#333333"), linewidths=0.9, zorder=3)
        q1, med, q3 = np.percentile(errs, [25, 50, 75])
        ax.plot([i - 0.22, i + 0.22], [med, med], color=MODEL_COLORS.get(model, "#333"),
                lw=2.0, zorder=4)
        ax.add_patch(plt.Rectangle((i - 0.22, q1), 0.44, q3 - q1, fill=False,
                                   edgecolor=MODEL_COLORS.get(model, "#333"), lw=0.9, zorder=2))
    ax.set_yscale("log")
    labels = [MODEL_LABELS[m] for m in order_models]
    if "FF_ORACLE" in set(df.model):
        labels.append(MODEL_LABELS["FF_ORACLE"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(r"Relative $L_2$ error (10 paired seeds)")
    savefig(fig, os.path.join(OUT, "confirm_box_strip.png"))

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for model in order_models + (["FF_ORACLE"] if "FF_ORACLE" in set(df.model) else []):
        g = df[df.model == model]
        med = g["l2_relative"].median()
        ax.scatter(g["n_params"].iloc[0], med, s=45,
                   color=MODEL_COLORS.get(model, "#333"), zorder=3)
        ax.annotate(MODEL_LABELS[model], (g["n_params"].iloc[0], med),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.scatter(np.full(len(g), g["n_params"].iloc[0]), g["l2_relative"], s=10,
                   facecolors="none", edgecolors=MODEL_COLORS.get(model, "#333"),
                   linewidths=0.6, zorder=2, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel(r"Relative $L_2$ error (median + seeds)")
    savefig(fig, os.path.join(OUT, "confirm_error_vs_params.png"))

    print(summ[["model", "key_id", "n", "l2_median", "l2_mean", "l2_geomean",
                "success_rate_0.1"]].to_string(index=False))
    print()
    print(pstats.to_string(index=False))
    if problems:
        print("\nVERIFICATION PROBLEMS:")
        for p in problems:
            print(" -", p)
    else:
        print("\nverification: OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        phase_confirm()
    else:
        phase_dev()
