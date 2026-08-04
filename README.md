# Legendre_Polynomial-based_ADA

## Update (2026-08): unified-protocol re-runs and new baselines

All benchmark results were regenerated under a single seed-controlled protocol
(≥5 seeds per configuration, mean ± std with medians), with new experiments:
a parameter-matched random Fourier-feature baseline, a polynomial-order study
including orders below the PDE order, a panel-count reparameterization study at matched nominal budget, a hyperparameter sensitivity grid, and a
PI-DeepONet re-run with a full Reynolds sweep. See `revision/README.md` for
the framework and `results/` for aggregated CSVs and figures.

## Repository structure

```
Legendre_Polynomial_ADA/
├── 2D_Helmholtz/            # 2D Helmholtz benchmark (k0 = 4π, unit square)
│   ├── pinn_utils.py        #   models (vanilla R / single-output R1 / LPA / FourierFeatures) + solver
│   ├── main_run_R|LPA|ADAF.py   # legacy entry scripts (one model each)
│   ├── revision_run.py      #   seed-controlled single-run CLI (unified protocol)
│   ├── results/revision/    #   per-run error histories + config JSONs, grouped by experiment
│   └── checkpoints/         #   legacy trained weights
├── Diffusion-Reaction/      # 1D diffusion-reaction benchmark (x ∈ [-π, π], multi-frequency source)
│   └── (same layout as 2D_Helmholtz)
├── Kovasznay_flow/          # Kovasznay flow benchmark (steady Navier–Stokes, Re = 40)
│   └── (same layout as 2D_Helmholtz)
├── Burgers/                 # viscous Burgers benchmark (legacy)
├── DeepONet/                # PI-DeepONet operator learning (Kovasznay, branch = Re)
│   ├── core/                #   model builders, physics residuals, sampling, metrics
│   ├── train_deeponet.py    #   legacy training entry (config.py driven)
│   ├── revision_run_deeponet.py  # seed-controlled CLI + full Re = 1…199 sweep
│   ├── results/revision/    #   error histories, Re-sweep CSVs, run configs
│   └── checkpoints/revision/#   archived weights for all unified-protocol runs
├── revision/                # experiment framework
│   ├── run_queue.py         #   resumable multi-process job queue
│   ├── make_jobs*.py        #   job-list generators for the campaigns
│   ├── aggregate_*.py       #   build tables/figures from run JSONs
│   └── plot_style.py        #   shared figure conventions (serif, 400 dpi, inward ticks)
└── results/                 # aggregated outputs (CSV + PNG), one folder per experiment
    ├── ff_baseline/         #   vanilla vs Fourier-feature vs LPA, three benchmarks
    ├── low_order/           #   polynomial order P ∈ {1,2,3,4,6} incl. below PDE order
    ├── dof_fixed/           #   panel-count reparameterization, matched nominal budget
    ├── sensitivity/         #   P × N_panel grid + learning-rate sweep heatmaps
    └── deeponet_rev/        #   PI-DeepONet, four widths, Reynolds sweep
```

Per-benchmark file roles: `pinn_utils.py` holds all model definitions and the
two-stage solver (Adam → L-BFGS-B); `main_run_<KEY>.py` are the original
one-model entry scripts; `revision_run.py` is the parameterized CLI used for
all regenerated results (`--key {R|R1|LPA|FF} --order --panels --sigma --lr
--nh --nn --trial --exp`). Raw per-run outputs land in
`<benchmark>/results/revision/<experiment>/`; cross-run tables and figures are
built by `revision/aggregate_*.py` into the top-level `results/`.


## Status notes and corrections (2026-08)

- **`results/dof_fixed/` interpretation retracted.** The runs are valid, but
  the original "fixed degrees-of-freedom (p- vs h-refinement)" reading is
  not: the LPA panel weights enter the forward pass only through their
  Legendre-projection coefficients, so the activation's effective
  coefficient space has dimension at most min(N, P+1) — P+1 for all tested
  configurations. The data should be read as a panel-count
  reparameterization study.
- **Paired same-protocol comparison added** (`results/major1_compare/`,
  runner `2D_Helmholtz/major1_compare_run.py`, scripts
  `revision/make_jobs_major1.py`, `revision/aggregate_major1.py`): tanh
  PINN, LPA, fixed/trainable random Fourier features, SIREN, and neuron-wise
  adaptive tanh (learned neuron-wise slopes; the slope-recovery coefficient
  was part of the search and was selected as zero) under identical
  collocation points, loss, optimizer budget, and paired seeds, plus a
  target-frequency-informed oracle Fourier reference. Per-run records are in
  `2D_Helmholtz/results/revision/major1_{dev,confirm}/`; see
  `results/major1_compare/MAJOR1.md` for the frozen configurations and
  selection rule.
- **Availability.** The implementation code, per-run records, aggregate
  tables, trained operator-network weights, and representative figures
  supporting the revision are available in this repository; a versioned
  release will be tagged for citation in the revised manuscript.
