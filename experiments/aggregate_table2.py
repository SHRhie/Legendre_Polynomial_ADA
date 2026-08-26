"""Recompute Table 2 of the manuscript from the archived run records.

Table 2 reports, for each configured trunk-readout combination on the 2D
Helmholtz problem at three hidden layers of ten neurons, the median and range
of the relative L2 error over the ten held-out seeds (trials 10-19), the number
of paired runs in which the LPA readout is more accurate than the linear
readout of the same trunk, and the number of runs whose error exceeds one half.

Usage:
    python revision/aggregate_table2.py            # print the table
    python revision/aggregate_table2.py --csv out.csv
"""
import argparse
import json
import os

from paths import runs_dir

TRIALS = tuple(range(10, 20))

# (row label, readout label, experiment directory, key_id)
ROWS = [
    ("tanh", "linear", "major1_confirm", "TANH_lr0.003"),
    ("tanh", "LPA, M = 6", "major1_confirm", "LPA_P6_N30_lr0.001"),
    ("Fourier features (m = 3, sigma = 0.5)", "linear", "major1_confirm",
     "FF_m3_s0.5_lr0.003"),
    ("Fourier features (m = 3, sigma = 0.5)", "LPA, M = 3", "ff_lpa_confirm_m3",
     "FF_LPA_m3_s0.5_P3_N30_lr0.003"),
    ("Fourier features (m = 3, sigma = 0.5)", "LPA, M = 6", "ff_lpa_confirm",
     "FF_LPA_m3_s0.5_P6_N30_lr0.003"),
    ("SIREN (omega0 = 30/1)", "linear", "siren_lpa_mismatch", "SIREN_w30_hw1_lr0.001"),
    ("SIREN (omega0 = 30/1)", "LPA, M = 3", "siren_lpa_mismatch",
     "SIREN_LPA_w30_hw1_P3_N30_lr0.001"),
    ("SIREN (omega0 = 30/5)", "linear", "siren_lpa_confirm2", "SIREN_w30_hw5_lr0.001"),
    ("SIREN (omega0 = 30/5)", "LPA, M = 3", "siren_lpa_confirm2",
     "SIREN_LPA_w30_hw5_P3_N30_lr0.001"),
    ("SIREN (omega0 = 30/30)", "linear", "major1_confirm", "SIREN_w30_hw30_lr0.001"),
    ("SIREN (omega0 = 30/30)", "LPA, M = 3", "siren_lpa_hw30_m3",
     "SIREN_LPA_w30_hw30_P3_N30_lr0.001"),
]

COLLAPSE_THRESHOLD = 0.5


def load_cell(experiment, key_id):
    """Return {trial: record} for one configuration, from its archived runs."""
    directory = runs_dir("helmholtz2d", experiment)
    found = {}
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("run_") and name.endswith(".json")):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("key_id") == key_id:
            found[record["trial"]] = record
    missing = [t for t in TRIALS if t not in found]
    if missing:
        raise RuntimeError(f"{experiment}/{key_id}: missing trials {missing}")
    return found


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="also write the table to this path")
    args = parser.parse_args()

    cells = {}
    for trunk, readout, experiment, key_id in ROWS:
        records = load_cell(experiment, key_id)
        cells[(trunk, readout)] = [records[t]["l2_relative"] for t in TRIALS]
        # every configuration in a row shares the collocation set of its trial
        hashes = {records[t]["collocation_sha256"] for t in TRIALS}
        if len(hashes) != len(TRIALS):
            raise RuntimeError(f"{experiment}/{key_id}: collocation hashes not distinct per trial")

    rows = []
    for trunk, readout, _, _ in ROWS:
        errors = cells[(trunk, readout)]
        if readout == "linear":
            wins = "-"
        else:
            linear = cells[(trunk, "linear")]
            wins = f"{sum(1 for a, b in zip(errors, linear) if a < b)}/{len(TRIALS)}"
        rows.append({
            "trunk": trunk, "readout": readout,
            "median": median(errors), "min": min(errors), "max": max(errors),
            "paired_wins": wins,
            "collapsed": sum(1 for e in errors if e > COLLAPSE_THRESHOLD),
        })

    header = f"{'Trunk':<38}{'Readout':<13}{'Median (Min-Max)':<34}{'Wins':>6}{'Coll.':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        span = f"{r['median']:.2E} ({r['min']:.2E}-{r['max']:.2E})"
        print(f"{r['trunk']:<38}{r['readout']:<13}{span:<34}{r['paired_wins']:>6}{r['collapsed']:>7}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
