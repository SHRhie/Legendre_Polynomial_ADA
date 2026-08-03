# Unified-protocol experiment framework (2026-08)

Seed-controlled runners, a resumable job queue, and aggregation scripts used to
regenerate all benchmark results under one protocol, plus new baselines and
sweeps. All statistics in the regenerated tables are mean ± sample std with
medians, over ≥5 seeded runs (seed = 1234 + trial index).

## What was added

- **`<benchmark>/revision_run.py`** — one training run per invocation
  (`--key {R|R1|LPA|FF} --order --panels --sigma --lr --nh --nn --trial --exp`).
  Writes `results/revision/<exp>/` with `run_*.json` (full config + final
  metrics), `acc_hist_*` (error history, every 50 optimizer steps), and
  `cal_time_*`. The Helmholtz runner also records post-training RMS norms of
  the PDE residual and its derivatives up to second order.
- **`DeepONet/revision_run_deeponet.py`** — PI-DeepONet runs with an explicit
  column header on `acc_hist` (`Re_ref, l1_u, l1_v, l1_p, l2_u, l2_v, l2_p`),
  a full Reynolds sweep (`re_sweep_*.csv`, Re = 1…199), per-run config
  snapshots, and archived weights (`DeepONet/checkpoints/revision/`).
- **New model variants** in `pinn_utils.py`:
  - `FourierFeatures` + key `FF` — random Fourier feature baseline
    (Tancik et al., 2020), parameter-matched to the LPA models (m = 3).
  - key `R1` (Helmholtz) — vanilla baseline with a single-output head,
    like-for-like with the LPA/FF models.
- **`revision/`** — `run_queue.py` (resumable multi-process queue; jobs from
  `make_jobs*.py`), `aggregate_*.py` (tables/figures from the run JSONs;
  paths assume the local experiment tree), `plot_style.py` (figure
  conventions).
- **`results/`** (top level) — aggregated CSVs and figures per experiment:
  `ff_baseline` (vanilla vs Fourier-feature vs LPA, three benchmarks),
  `low_order` (P ∈ {1,2,3,4,6} incl. below the PDE order),
  `dof_fixed` ((P+1)×N ≈ 210), `sensitivity` (P × N grid + learning-rate
  sweep), `deeponet_rev` (PI-DeepONet, four widths, Re sweep).

## Code fixes

- `ScipyOptimizer` in all `pinn_utils.py` now iterates
  `model.trainable_variables` (previously `model.variables`), which silently
  mis-mapped gradients for models with non-trainable weights.
- `Diffusion-Reaction/main_run_R.py`: domain unified to x ∈ [-π, π]
  (matching `main_run_LPA.py`; with L = 1 the manufactured solution violates
  the boundary conditions and every model floors at a relative L2 of ≈0.575).
- `Kovasznay_flow/main_run_R.py`: interior collocation count unified to
  N_r = 10000 (matching `main_run_LPA.py`).

## Environment

TensorFlow 2.10 (Keras 2), float32, CPU. Python ≥3.10 with numpy, scipy,
sympy, pandas, tqdm, matplotlib.

```bash
cd <benchmark>
python revision_run.py --key LPA --order 6 --panels 30 --nh 2 --nn 10 --trial 0 --exp demo
```
