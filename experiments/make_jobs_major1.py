"""Job generators for the Referee 1 / Major 1 paired-baseline campaign.

Phase A (development search):
  trials 0,1,2 x exactly 6 frozen candidate configurations per general model
  (TANH, LPA, FF, TFF, SIREN, NLAAF) = 108 runs, exp tag ``major1_dev``.
  FF_ORACLE is a physics-informed representation ceiling and takes no part in
  the search budget.

Phase B (confirmatory):
  one frozen configuration per model (chosen by the pre-registered rule in
  results/major1_compare/preregistration.md), paired trials 10..19,
  exp tag ``major1_confirm``.  Generated via
  ``python make_jobs_major1.py confirm '<json dict model->args>'``.

The configuration tables below are FROZEN before Phase A execution; do not
edit after results are seen.
"""
import json
import os
import sys

import os as _os
HELM = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "2D_Helmholtz")
DEV_TRIALS = [0, 1, 2]
CONFIRM_TRIALS = list(range(10, 20))
LR6 = ["0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03"]

# ---- frozen Phase A candidate table (6 per general model) -----------------
DEV_CONFIGS = []
for lr in LR6:
    DEV_CONFIGS.append(("TANH", ["--model", "TANH", "--lr", lr]))
for lr in LR6:
    DEV_CONFIGS.append(("LPA", ["--model", "LPA", "--order", "6", "--panels", "30", "--lr", lr]))
for m in ("2", "3"):
    for s in ("0.5", "1", "2"):
        DEV_CONFIGS.append(("FF", ["--model", "FF", "--ff-features", m, "--sigma", s, "--lr", "0.003"]))
for m in ("2", "3"):
    for s in ("0.5", "1", "2"):
        DEV_CONFIGS.append(("TFF", ["--model", "TFF", "--ff-features", m, "--sigma", s, "--lr", "0.001"]))
for w in ("5", "10", "30"):
    for lr in ("0.0001", "0.001"):
        DEV_CONFIGS.append(("SIREN", ["--model", "SIREN", "--omega0", w, "--hidden-omega0", w, "--lr", lr]))
for sr in ("0", "0.1", "1"):
    for lr in ("0.001", "0.01"):
        DEV_CONFIGS.append(("NLAAF", ["--model", "NLAAF", "--adaptive-scale", "10",
                                      "--adaptive-init", "0.1", "--slope-recovery", sr, "--lr", lr]))


def key_id_from_args(a):
    d = {a[i]: a[i + 1] for i in range(0, len(a) - 1, 2) if a[i].startswith("--")}
    model = d["--model"]
    lr = float(d.get("--lr", "0.01"))
    if model == "LPA":
        base = f"LPA_P{d['--order']}_N{d['--panels']}"
    elif model in ("FF", "TFF"):
        base = f"{model}_m{d['--ff-features']}_s{float(d['--sigma']):g}"
    elif model == "FF_ORACLE":
        base = "FF_ORACLE_m2"
    elif model == "SIREN":
        base = f"SIREN_w{float(d['--omega0']):g}_hw{float(d['--hidden-omega0']):g}"
    elif model == "NLAAF":
        base = (f"NLAAF_n{float(d['--adaptive-scale']):g}_a{float(d['--adaptive-init']):g}"
                f"_sr{float(d['--slope-recovery']):g}")
    else:
        base = model
    return f"{base}_lr{lr:g}"


def job(exp, extra_args, trial):
    args = extra_args + ["--nh", "3", "--nn", "10", "--trial", str(trial),
                         "--exp", exp, "--acc-every", "500"]
    stem = f"3_10_{key_id_from_args(args)}_{trial}"
    return {
        "dir": HELM,
        "script": "major1_compare_run.py",
        "args": args,
        "log": f"results/runs/{exp}/log_{stem}.txt",
        "done_check": f"results/runs/{exp}/run_{stem}.json",
    }


def dev_jobs():
    return [job("major1_dev", a, t) for _, a in DEV_CONFIGS for t in DEV_TRIALS]


def confirm_jobs(selected):
    """selected: dict model_name -> list of extra CLI args (frozen winners),
    plus FF_ORACLE appended automatically as the supplementary ceiling."""
    jobs = []
    for _, a in sorted(selected.items()):
        for t in CONFIRM_TRIALS:
            jobs.append(job("major1_confirm", a, t))
    for t in CONFIRM_TRIALS:
        jobs.append(job("major1_confirm",
                        ["--model", "FF_ORACLE", "--ff-features", "2", "--lr", "0.01"], t))
    return jobs


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        selected = json.loads(sys.argv[2])
        out = confirm_jobs(selected)
        path = os.path.join(here, "jobs_major1_confirm.json")
    else:
        out = dev_jobs()
        path = os.path.join(here, "jobs_major1_dev.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{path}: {len(out)} jobs")
