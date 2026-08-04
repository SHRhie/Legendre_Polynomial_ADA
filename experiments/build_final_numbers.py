"""Build results/final_numbers.json — every number that may appear in the
revised manuscript or the author response, in one machine-checkable file.

Two protocols are kept side by side so corrections can be diffed:
  "unified-v3"          : the revision campaigns (this repository's CSVs)
  "original-submission" : the numbers printed in manuscript_mlst.docx tables

Values are stored unrounded; rounding belongs to the documents.

Usage:  python build_final_numbers.py [--out PATH]
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from paths import ROOT, RESULTS as RES, bench_dir, runs_dir, manuscript_docx

BENCH_DIR = {k: bench_dir(k) for k in ("helmholtz2d", "diffusion_reaction", "kovasznay")}
DOCX = manuscript_docx()

# key_id prefix -> canonical model name required by the schema
MODEL_OF_KEY = {
    "R1": "baseline_tanh_1out",
    "R": "baseline_tanh",
    "LPA": "lpa",
    "FF": "ff_fixed",
    "TFF": "ff_trainable",
    "SIREN": "siren",
    "NLAAF": "adaptive_tanh",
    "FF_ORACLE": "ff_oracle",
    "TANH": "baseline_tanh",
    "VAN": "baseline_deeponet",
}
records = []


def model_of(key_id):
    for pre in ("FF_ORACLE", "TFF", "SIREN", "NLAAF", "TANH", "LPA", "R1", "FF", "R", "VAN"):
        if str(key_id).startswith(pre):
            return MODEL_OF_KEY[pre]
    return str(key_id)


def arch(layers, neurons):
    return {"layers": int(layers), "neurons": int(neurons)}


def stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {
        "n_seeds": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else None,
        "median": float(np.median(v)),
        "per_seed": [float(x) for x in v],
        "frac_below_0p1": float((v < 0.1).mean()),
    }


def add(rec_id, benchmark, protocol, model, metric, values=None, *, arch_=None,
        hyperparams=None, component=None, n_params=None, source_file=None,
        script=None, appears_in=None, reviewer_ref=None, scalar=None, notes=None,
        extra=None, allow_empty=False):
    rec = {
        "id": rec_id,
        "benchmark": benchmark,
        "protocol": protocol,
        "arch": arch_,
        "model": model,
        "hyperparams": hyperparams or {},
        "metric": metric,
        "component": component,
        "n_params": None if n_params is None else int(n_params),
        "source_file": source_file,
        "script": script,
        "appears_in": appears_in or [],
        "reviewer_ref": reviewer_ref or [],
    }
    if values is not None:
        s = stats(values)
        if s is None:
            if not allow_empty:
                return
            s = {"n_seeds": 0, "mean": None, "sd": None, "median": None,
                 "per_seed": [], "frac_below_0p1": None}
        rec.update(s)
        if metric not in ("rel_l2", "abs_l1"):
            # the fraction below 0.1 only means something for an error metric
            rec["frac_below_0p1"] = None
    if scalar is not None:
        rec.update({"n_seeds": None, "mean": None, "sd": None, "median": None,
                    "per_seed": [], "frac_below_0p1": None, "value": scalar})
    if extra:
        rec.update(extra)
    if notes:
        rec["notes"] = notes
    records.append(rec)


# ----------------------------------------------------------------- campaigns
def ff_baseline():
    """Task 1: unified-protocol re-runs of the three PINN benchmarks."""
    specs = {
        "helmholtz2d": ("Table 1", "Figure 2", [("l2_relative", "rel_l2", None),
                                                ("l1_absolute", "abs_l1", None)]),
        "diffusion_reaction": ("Table 2", "Figure 3", [("l2_relative", "rel_l2", None),
                                                       ("l1_absolute", "abs_l1", None)]),
        "kovasznay": ("Table 3", "Figure 4",
                      [(f"{m}_{c}", k, c) for c in ("u", "v", "p")
                       for m, k in (("l2_relative", "rel_l2"), ("l1_absolute", "abs_l1"))]),
    }
    for bench, (tbl, figref, cols) in specs.items():
        src = f"{RES}/ff_baseline/{bench}_runs.csv"
        df = pd.read_csv(src)
        for (nh, nn, key_id), g in df.groupby(["nh", "nn", "key_id"], sort=True):
            for col, metric, comp in cols:
                if col not in g:
                    continue
                hp = {}
                row = g.iloc[0]
                for src_col, name in (("lpa_order", "P"), ("lpa_panels", "N_panel"),
                                      ("ff_sigma", "sigma"), ("ff_features", "m"),
                                      ("lr", "lr_warmup"), ("N_r", "N_r")):
                    if src_col in g and pd.notna(row[src_col]):
                        hp[name] = row[src_col].item() if hasattr(row[src_col], "item") else row[src_col]
                rid = f"{bench}|L{nh}N{nn}|{key_id}|{metric}" + (f"|{comp}" if comp else "")
                add(rid, bench, "unified-v3", model_of(key_id), metric,
                    values=g[col].tolist(), arch_=arch(nh, nn), hyperparams=hp,
                    component=comp, n_params=row.get("n_params"),
                    source_file=os.path.relpath(src, ROOT),
                    script="revision/aggregate_task1.py",
                    appears_in=[tbl, figref],
                    reviewer_ref=["R2-Comment4", "R1-Major1", "R1-Minor3"])


def low_order():
    """Task 2 (R2 Comment 2/3): polynomial order below the PDE order."""
    src = f"{RES}/low_order/runs.csv"
    df = pd.read_csv(src)
    for (P,), g in df.groupby(["P"]):
        row = g.iloc[0]
        for col, metric in (("l2_relative", "rel_l2"), ("l1_absolute", "abs_l1"),
                            ("rms_r", "rms_residual"), ("rms_r_xx", "rms_dxx_residual")):
            add(f"helmholtz2d|L2N10|lpa_P{int(P)}|{metric}|low_order", "helmholtz2d",
                "unified-v3", "lpa", metric, values=g[col].tolist(),
                arch_=arch(2, 10),
                hyperparams={"P": int(P), "N_panel": int(row["lpa_panels"]),
                             "lr_warmup": float(row["lr"])},
                n_params=row.get("n_params"),
                source_file=os.path.relpath(src, ROOT),
                script="revision/aggregate_task2.py",
                appears_in=["low-order section"],
                reviewer_ref=["R2-Comment2", "R2-Comment3", "R1-Major3"])


def low_order_converged():
    """Residual norms restricted to the seeds that converged (rel_l2 < 0.1).

    Pure post-hoc selection over the runs already loaded by low_order(): the
    per-seed arrays of rel_l2, rms_residual and rms_dxx_residual come from the
    same grouped frame in the same row order, so selecting on the error index
    selects the matching residual entries. No re-runs are involved.

    Cross-checked against the published per-seed arrays (identical means), and
    against P = 1, which has no converged seed and is still emitted with
    n_conv = 0 so the absence is explicit rather than a missing record.
    """
    src = f"{RES}/low_order/runs.csv"
    df = pd.read_csv(src)
    cols = ["l2_relative", "rms_r", "rms_r_xx"]
    assert not df[cols].isna().any().any(), (
        "NaN in the low-order metrics would break index alignment between the "
        "error array and the residual arrays")
    for (P,), g in df.groupby(["P"]):
        row = g.iloc[0]
        conv = g[g["l2_relative"] < 0.1]
        n_conv = int(len(conv))
        for col, metric in (("rms_r", "rms_residual"), ("rms_r_xx", "rms_dxx_residual")):
            add(f"helmholtz2d|L2N10|lpa_P{int(P)}|{metric}|low_order_converged",
                "helmholtz2d", "unified-v3", "lpa", metric,
                values=conv[col].tolist(), arch_=arch(2, 10),
                hyperparams={"P": int(P), "N_panel": int(row["lpa_panels"]),
                             "lr_warmup": float(row["lr"])},
                n_params=row.get("n_params"),
                source_file=os.path.relpath(src, ROOT),
                script="revision/build_final_numbers.py",
                appears_in=["low-order section"],
                reviewer_ref=["R1-Major3", "R2-Comment3"],
                extra={"n_conv": n_conv, "n_total": int(len(g)),
                       "condition": "rel_l2 < 0.1"},
                notes="conditioned on rel_l2 < 0.1; post-hoc subset, n_conv stated",
                allow_empty=True)


def dof_fixed():
    """Task 3 (R2 Comment 3): (P+1)xN held at a matched nominal budget."""
    src = f"{RES}/dof_fixed/runs.csv"
    df = pd.read_csv(src)
    for (P, N), g in df.groupby(["P", "N_panel"]):
        row = g.iloc[0]
        add(f"helmholtz2d|L2N10|lpa_P{int(P)}_N{int(N)}|rel_l2|dof_fixed", "helmholtz2d",
            "unified-v3", "lpa", "rel_l2", values=g["l2_relative"].tolist(),
            arch_=arch(2, 10),
            hyperparams={"P": int(P), "N_panel": int(N), "dof": int(row["dof"]),
                         "lr_warmup": float(row["lr"])},
            n_params=row.get("n_params"),
            source_file=os.path.relpath(src, ROOT),
            script="revision/aggregate_task3.py",
            appears_in=["panel-count reparameterization section"],
            reviewer_ref=["R2-Comment3", "R1-Minor1"],
            notes="panel-count reparameterization; the p- vs h-refinement reading is retracted")


def sensitivity():
    """Task 4 (R1 Major 4): order x panel grid and warm-up rate sweep."""
    src = f"{RES}/sensitivity/runs.csv"
    df = pd.read_csv(src)
    base_lr = 1e-2
    grid = df[np.isclose(df["lr"], base_lr)]
    for (P, N), g in grid.groupby(["P", "N_panel"]):
        add(f"helmholtz2d|L2N10|lpa_P{int(P)}_N{int(N)}|rel_l2|sensitivity", "helmholtz2d",
            "unified-v3", "lpa", "rel_l2", values=g["l2_relative"].tolist(),
            arch_=arch(2, 10),
            hyperparams={"P": int(P), "N_panel": int(N), "lr_warmup": base_lr},
            n_params=g.iloc[0].get("n_params"),
            source_file=os.path.relpath(src, ROOT),
            script="revision/aggregate_task4.py",
            appears_in=["sensitivity section"], reviewer_ref=["R1-Major4", "R1-Minor1"])
    cell = df[(df["P"] == 6) & (df["N_panel"] == 30)]
    for lr, g in cell.groupby("lr"):
        add(f"helmholtz2d|L2N10|lpa_P6_N30|rel_l2|lr{lr:g}", "helmholtz2d",
            "unified-v3", "lpa", "rel_l2", values=g["l2_relative"].tolist(),
            arch_=arch(2, 10),
            hyperparams={"P": 6, "N_panel": 30, "lr_warmup": float(lr)},
            source_file=os.path.relpath(src, ROOT),
            script="revision/aggregate_task4.py",
            appears_in=["sensitivity section"], reviewer_ref=["R1-Major4"])


def major1():
    """Paired same-protocol comparison (R1 Major 1 / R2 Comment 4)."""
    src = f"{RES}/major1_compare/confirm_runs.csv"
    df = pd.read_csv(src)
    for (model_key,), g in df.groupby(["model"]):
        row = g.iloc[0]
        hp = {k: (None if pd.isna(row[k]) else row[k].item() if hasattr(row[k], "item") else row[k])
              for k in ("order", "panels", "ff_features", "sigma", "omega0",
                        "hidden_omega0", "adaptive_scale", "adaptive_init",
                        "slope_recovery_weight", "lr")
              if k in g}
        add(f"helmholtz2d|L3N10|{model_key}|rel_l2|major1_confirm", "helmholtz2d",
            "unified-v3", model_of(model_key), "rel_l2",
            values=g["l2_relative"].tolist(), arch_=arch(3, 10), hyperparams=hp,
            n_params=row["n_params"], source_file=os.path.relpath(src, ROOT),
            script="revision/aggregate_major1.py",
            appears_in=["comparison table", "comparison figure"],
            reviewer_ref=["R1-Major1", "R2-Comment4"],
            notes="held-out confirmation seeds; frac_below_0p1 uses the descriptive "
                  "threshold 0.1, which was not a pre-specified criterion")
        add(f"helmholtz2d|L3N10|{model_key}|lbfgs_nit|major1_confirm", "helmholtz2d",
            "unified-v3", model_of(model_key), "lbfgs_nit",
            values=g["lbfgs_nit"].tolist(), arch_=arch(3, 10),
            source_file=os.path.relpath(src, ROOT),
            script="revision/aggregate_major1.py", reviewer_ref=["R1-Major5"])

    stat = pd.read_csv(f"{RES}/major1_compare/paired_statistics.csv")
    for _, r in stat.iterrows():
        cmp_id = r["comparison"].replace(" ", "")
        add(f"helmholtz2d|L3N10|{cmp_id}|paired_log10_diff|major1", "helmholtz2d",
            "unified-v3", "paired_comparison", "paired_log10_diff",
            scalar={"mean_log10_diff": float(r["mean_log10_diff"]),
                    "median_log10_diff": float(r["median_log10_diff"]),
                    "bootstrap_ci95": [float(r["paired_bootstrap_ci95_lo"]),
                                       float(r["paired_bootstrap_ci95_hi"])],
                    "wilcoxon_exact_p": float(r["wilcoxon_exact_p"]),
                    "holm_adjusted_p": float(r["holm_adjusted_p"]),
                    "direction": r["direction"], "n_pairs": int(r["n_pairs"])},
            arch_=arch(3, 10),
            source_file="results/major1_compare/paired_statistics.csv",
            script="revision/aggregate_major1.py",
            appears_in=["comparison table"],
            reviewer_ref=["R1-Major1", "R2-Comment4"])


def deeponet():
    """PI-DeepONet re-run (R2 Comment 7 + Table 4 correction)."""
    src = f"{RES}/deeponet_rev/runs.csv"
    df = pd.read_csv(src)
    for (nn, key_id), g in df.groupby(["nn", "key_id"], sort=True):
        row = g.iloc[0]
        hp = {"latent_dim": int(row["latent_dim"]), "head_width": int(row["head_width"]),
              "lpa_order": None if pd.isna(row["lpa_order"]) else int(row["lpa_order"]),
              "lpa_panels": None if pd.isna(row["lpa_panels"]) else int(row["lpa_panels"]),
              "epochs": int(row["epochs"]), "lr": float(row["lr"]),
              "n_int": int(row["n_int"]), "n_b": int(row["n_b"])}
        for comp in ("u", "v", "p"):
            add(f"deeponet_kovasznay|L{row['nh']}N{nn}|{key_id}|rel_l2|{comp}|re40",
                "deeponet_kovasznay", "unified-v3", model_of(key_id), "rel_l2",
                values=g[f"l2_{comp}_ref"].tolist(), arch_=arch(row["nh"], nn),
                hyperparams=hp, component=comp, n_params=row["n_params"],
                source_file=os.path.relpath(src, ROOT),
                script="revision/aggregate_deeponet.py", appears_in=["Table 4"],
                reviewer_ref=["R2-Comment7"],
                notes="single reference Reynolds number Re = 40 (training monitor)")
            add(f"deeponet_kovasznay|L{row['nh']}N{nn}|{key_id}|rel_l2|{comp}|sweep_median",
                "deeponet_kovasznay", "unified-v3", model_of(key_id), "rel_l2",
                values=g[f"sweep_median_l2_{comp}"].tolist(), arch_=arch(row["nh"], nn),
                hyperparams=hp, component=comp, n_params=row["n_params"],
                source_file=os.path.relpath(src, ROOT),
                script="revision/aggregate_deeponet.py",
                appears_in=["Table 4", "Figure 6"], reviewer_ref=["R2-Comment7"],
                notes="median over the Re = 1..199 sweep")
    ci = pd.read_csv(f"{RES}/deeponet_rev/paired_ci.csv")
    for _, r in ci.iterrows():
        add(f"deeponet_kovasznay|N{int(r['width'])}|paired|{r['metric']}|interval",
            "deeponet_kovasznay", "unified-v3", "paired_comparison", "paired_interval",
            scalar={"n": int(r["n"]),
                    "mean_diff_baseline_minus_lpa": float(r["mean_diff_van_minus_lpa"]),
                    "bootstrap_ci95": [float(r["boot_ci95_lo"]), float(r["boot_ci95_hi"])],
                    "bootstrap_includes_zero": bool(r["boot_includes_zero"]),
                    "t_ci95": [float(r["t_ci95_lo"]), float(r["t_ci95_hi"])],
                    "t_includes_zero": bool(r["t_includes_zero"])},
            arch_={"layers": 3, "neurons": int(r["width"])},
            source_file="results/deeponet_rev/paired_ci.csv",
            script="revision/aggregate_deeponet.py",
            appears_in=["supplementary paired-interval table"],
            reviewer_ref=["R2-Comment7"],
            notes="interval estimates are method-sensitive at n = 5; reported descriptively")


def timing_single():
    """Single-worker cost measurements (R1 Minor 2 / R2 Comment 5)."""
    d = runs_dir("helmholtz2d", "timing_single")
    per_key = {}
    for p in sorted(glob.glob(f"{d}/run_*.json")):
        j = json.load(open(p))
        nh, nn, key_id = j["nh"], j["nn"], j["key_id"]
        it = j.get("lbfgs_steps") or j.get("lbfgs_nit")
        if not it:
            continue
        ms = 1000.0 * float(j["time_lbfgs"]) / float(it)
        per_key.setdefault((nh, nn, key_id), []).append(ms)
        per_key.setdefault((nh, nn, key_id, "e2e"), []).append(float(j["time_lbfgs"]))
    for k, v in per_key.items():
        if len(k) == 4:
            continue
        nh, nn, key_id = k
        add(f"helmholtz2d|L{nh}N{nn}|{key_id}|per_iteration_ms|timing", "helmholtz2d",
            "unified-v3", model_of(key_id), "per_iteration_ms", values=v,
            arch_=arch(nh, nn), source_file="v4/2D Helmholtz v2/results/revision/timing_single",
            script="run_experiment.py (Helmholtz)",
            appears_in=["cost paragraph"], reviewer_ref=["R1-Minor2", "R2-Comment5"],
            notes="single dedicated worker; includes periodic accuracy evaluations; "
                  "descriptive only, n = 1-2 per depth")
        add(f"helmholtz2d|L{nh}N{nn}|{key_id}|optimizer_stage_seconds|timing", "helmholtz2d",
            "unified-v3", model_of(key_id), "optimizer_stage_seconds",
            values=per_key[(nh, nn, key_id, "e2e")], arch_=arch(nh, nn),
            source_file=os.path.relpath(runs_dir("helmholtz2d", "timing_single"), ROOT),
            script="run_experiment.py (Helmholtz)",
            appears_in=["cost paragraph"], reviewer_ref=["R1-Minor2", "R2-Comment5"])


# --------------------------------------------------------- original submission
def original_submission():
    """Numbers as printed in manuscript_mlst.docx (for correction diffs).

    Skipped when the manuscript file is absent, as in the public repository.

    Each numbered table in the original manuscript consists of two blocks
    (accuracy, then timing) sharing one caption.
    """
    if DOCX is None:
        print("  [skip] manuscript_mlst.docx not present: "
              "original-submission records omitted")
        return
    try:
        from docx import Document
        from docx.table import Table as DocxTable
    except ImportError:
        print("  [skip] python-docx not installed: original-submission records "
              "omitted (run this script in an environment with python-docx)")
        return
    doc = Document(DOCX)
    tables = [DocxTable(c, doc) for c in doc.element.body.iterchildren()
              if c.tag.endswith('}tbl')]

    def cells(t):
        return [[c.text.strip() for c in r.cells] for r in t.rows]

    def num(s):
        s = s.replace("​", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    # Tables 1-3: (accuracy block, timing block); columns L1/L2 for R and LPA
    plan = [("Table 1", "helmholtz2d", 5, 6), ("Table 2", "diffusion_reaction", 7, 8)]
    for tbl, bench, acc_i, time_i in plan:
        rows = cells(tables[acc_i])[2:]
        for r in rows:
            layers, neurons = int(num(r[0])), int(num(r[1]))
            for model, l1, l2 in (("baseline_tanh", num(r[2]), num(r[3])),
                                  ("lpa", num(r[4]), num(r[5]))):
                for metric, val in (("abs_l1", l1), ("rel_l2", l2)):
                    add(f"{bench}|L{layers}N{neurons}|{model}|{metric}|original", bench,
                        "original-submission", model, metric,
                        scalar={"reported": val}, arch_=arch(layers, neurons),
                        source_file="manuscript_mlst.docx", script=None,
                        appears_in=[tbl],
                        notes="single retrospectively selected run; no aggregation rule stated")
        for r in cells(tables[time_i])[2:]:
            layers, neurons = int(num(r[0])), int(num(r[1]))
            for model, off in (("baseline_tanh", 2), ("lpa", 5)):
                add(f"{bench}|L{layers}N{neurons}|{model}|time_per_epoch_s|original", bench,
                    "original-submission", model, "time_per_epoch_s",
                    scalar={"epochs": num(r[off]), "total_seconds": num(r[off + 1]),
                            "seconds_per_epoch": num(r[off + 2])},
                    arch_=arch(layers, neurons), source_file="manuscript_mlst.docx",
                    appears_in=[tbl], notes="timing of the selected run only")

    # Table 3 (Kovasznay): variable x layers
    rows = cells(tables[9])[2:]
    for r in rows:
        comp = r[0].split("-")[0].strip().lower() or None
        comp = {"u": "u", "v": "v", "pressure": "p", "p": "p"}.get(comp, comp)
        layers = int(num(r[1]))
        for model, l1, l2 in (("baseline_tanh", num(r[2]), num(r[3])),
                              ("lpa", num(r[4]), num(r[5]))):
            for metric, val in (("abs_l1", l1), ("rel_l2", l2)):
                add(f"kovasznay|L{layers}N10|{model}|{metric}|{comp}|original", "kovasznay",
                    "original-submission", model, metric, scalar={"reported": val},
                    arch_=arch(layers, 10), component=comp,
                    source_file="manuscript_mlst.docx", appears_in=["Table 3"],
                    notes="component-wise minimum over runs; entries in one row may come "
                          "from different runs")
    for r in cells(tables[10])[2:]:
        layers, neurons = int(num(r[0])), int(num(r[1]))
        for model, off in (("baseline_tanh", 2), ("lpa", 5)):
            add(f"kovasznay|L{layers}N{neurons}|{model}|time_per_epoch_s|original", "kovasznay",
                "original-submission", model, "time_per_epoch_s",
                scalar={"epochs": num(r[off]), "total_seconds": num(r[off + 1]),
                        "seconds_per_epoch": num(r[off + 2])},
                arch_=arch(layers, neurons), source_file="manuscript_mlst.docx",
                appears_in=["Table 3"],
                notes="baseline and LPA used different collocation budgets (5000 vs 10000)")

    # Table 4 (PI-DeepONet): the reported columns were mis-mapped in the original
    for r in cells(tables[11])[2:]:
        layers, neurons = int(num(r[0])), int(num(r[1]))
        for comp, (base, lpa) in zip(("u", "v", "p"),
                                     ((num(r[2]), num(r[3])), (num(r[4]), num(r[5])),
                                      (num(r[6]), num(r[7])))):
            for model, val in (("baseline_deeponet", base), ("lpa", lpa)):
                add(f"deeponet_kovasznay|L{layers}N{neurons}|{model}|rel_l2|{comp}|original",
                    "deeponet_kovasznay", "original-submission", model, "rel_l2",
                    scalar={"reported": val}, arch_=arch(layers, neurons), component=comp,
                    source_file="manuscript_mlst.docx", appears_in=["Table 4"],
                    notes="column-mapping defect: the printed u/v/p values are other logged "
                          "metrics; single run at Re = 40 presented as generalization")
    for r in cells(tables[12])[2:]:
        layers, neurons = int(num(r[0])), int(num(r[1]))
        add(f"deeponet_kovasznay|L{layers}N{neurons}|timing|time_ratio|original",
            "deeponet_kovasznay", "original-submission", "paired_comparison", "time_ratio",
            scalar={"baseline_seconds": num(r[2]), "lpa_seconds": num(r[3]),
                    "ratio_text": r[4]}, arch_=arch(layers, neurons),
            source_file="manuscript_mlst.docx", appears_in=["Table 4"])


# ------------------------------------------------------------------------ main
def git_commit():
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{RES}/final_numbers.json")
    ap.add_argument("--generated-at", default=None,
                    help="ISO timestamp; defaults to the current time")
    args = ap.parse_args()

    ff_baseline(); low_order(); low_order_converged(); dof_fixed(); sensitivity()
    major1(); deeponet(); timing_single(); original_submission()

    if args.generated_at:
        ts = args.generated_at
    else:
        import datetime
        ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    payload = {
        "meta": {
            "generated_at": ts,
            "git_commit": git_commit(),
            "protocol": "unified-v3",
            "n_records": len(records),
            "notes": (
                "Two protocols: 'unified-v3' = revision campaigns (seed statistics from "
                "the CSVs under results/); 'original-submission' = the single values printed "
                "in manuscript_mlst.docx, kept for correction diffs. Aggregation rule for "
                "unified-v3 records: mean, sample SD (ddof=1), median and the full per-seed "
                "list over all seeded runs of the cell; no run is excluded and no best run is "
                "selected. frac_below_0p1 is a descriptive fraction, not a pre-specified "
                "success criterion. Figure representative seeds follow a separate "
                "pre-specified rule (final rel-L2 closest to the cell median; recorded in "
                "results/manuscript_revisions/representative_seeds.json). Values are stored "
                "unrounded. Timing records are descriptive (n = 1-2 per depth, include "
                "periodic accuracy evaluations)."
            ),
        },
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, allow_nan=False)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    print(f"wrote {args.out}: {len(records)} records, sha256:{h}")
    by = {}
    for r in records:
        by[(r["protocol"], r["benchmark"])] = by.get((r["protocol"], r["benchmark"]), 0) + 1
    for k in sorted(by):
        print(f"  {k[0]:<20} {k[1]:<22} {by[k]:>4}")


if __name__ == "__main__":
    main()
