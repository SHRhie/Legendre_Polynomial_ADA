# Legendre_Polynomial-based_ADA

## Update (2026-08): unified-protocol re-runs and new baselines

All benchmark results were regenerated under a single seed-controlled protocol
(≥5 seeds per configuration, mean ± std with medians), with new experiments:
a parameter-matched random Fourier-feature baseline, a polynomial-order study
including orders below the PDE order, a panel-count reparameterization study at matched nominal budget, a hyperparameter sensitivity grid, and a
PI-DeepONet re-run with a full Reynolds sweep. See `experiments/README.md` for
the framework and `results/` for aggregated CSVs and figures.

## Repository structure

```
Legendre_Polynomial_ADA/
├── 2D_Helmholtz/                  # 2D Helmholtz benchmark (k0 = 4π, unit square)
│   ├── pinn_utils.py              #   models (vanilla R / single-output R1 / LPA / FourierFeatures) + solver
│   ├── main_run_R|LPA|ADAF.py     # legacy entry scripts (one model each)
│   ├── run_experiment.py          #   seed-controlled single-run CLI (unified protocol)
│   ├── results/runs/              #   per-run error histories + config JSONs, grouped by experiment
│   └── checkpoints/               #   legacy trained weights
├── Diffusion-Reaction/            # 1D diffusion-reaction benchmark (x ∈ [-π, π], multi-frequency source)
│   └── (same layout as 2D_Helmholtz)
├── Kovasznay_flow/                # Kovasznay flow benchmark (steady Navier–Stokes, Re = 40)
│   └── (same layout as 2D_Helmholtz)
├── Burgers/                       # viscous Burgers benchmark (legacy)
├── DeepONet/                      # PI-DeepONet operator learning (Kovasznay, branch = Re)
│   ├── core/                      #   model builders, physics residuals, sampling, metrics
│   ├── train_deeponet.py          #   legacy training entry (config.py driven)
│   ├── run_experiment_deeponet.py # seed-controlled CLI + full Re = 1…199 sweep
│   ├── results/runs/              #   error histories, Re-sweep CSVs, run configs
│   └── checkpoints/runs/          #   archived weights for all unified-protocol runs
├── experiments/                   # experiment framework
│   ├── run_queue.py               #   resumable multi-process job queue
│   ├── make_jobs*.py              #   job-list generators for the campaigns
│   ├── aggregate_*.py             #   build tables/figures from run JSONs
│   ├── make_paper_figures.py      #   manuscript replacement figures (Fig 2-4)
│   ├── build_final_numbers.py     #   collect every reported number into one JSON
│   ├── build_supplementary.py     #   supplementary data files with titles
│   ├── build_figure_manifest.py   #   figure inventory + placement proposal
│   ├── paths.py                   #   layout resolution (runs from a plain checkout)
│   └── plot_style.py              #   shared figure conventions (serif, 400 dpi, inward ticks)
└── results/                       # aggregated outputs (CSV + PNG), one folder per experiment
    ├── final_numbers.json         #   every reported number, unrounded, with per-seed values
    ├── figure_manifest.json       #   figure inventory: sha256, generator, proposed placement
    ├── supplementary/             #   supplementary data files (title + description in each)
    ├── manuscript_revisions/      #   replacement figures + representative-seed record
    ├── ff_baseline/               #   vanilla vs Fourier-feature vs LPA, three benchmarks
    ├── low_order/                 #   polynomial order P ∈ {1,2,3,4,6} incl. below PDE order
    ├── dof_fixed/                 #   panel-count reparameterization, matched nominal budget
    ├── sensitivity/               #   P × N_panel grid + learning-rate sweep heatmaps
    └── deeponet_rev/              #   PI-DeepONet, four widths, Reynolds sweep
```

Per-benchmark file roles: `pinn_utils.py` holds all model definitions and the
two-stage solver (Adam → L-BFGS-B); `main_run_<KEY>.py` are the original
one-model entry scripts; `run_experiment.py` is the parameterized CLI used for
all regenerated results (`--key {R|R1|LPA|FF} --order --panels --sigma --lr
--nh --nn --trial --exp`). Raw per-run outputs land in
`<benchmark>/results/runs/<experiment>/`; cross-run tables and figures are
built by `experiments/aggregate_*.py` into the top-level `results/`.


## Reproducing a specific number

Every number in the revised manuscript is also stored, unrounded and with its
per-seed values, in **`results/final_numbers.json`** (one record per
benchmark x architecture x model x metric, with the source CSV, the generating
script, the reviewer comment it answers, and the table or figure it appears in).
Records tagged `"protocol": "original-submission"` are the values printed in the
original submission, kept so the corrections can be diffed; records tagged
`"protocol": "unified-v3"` are the revision campaigns. To look one up:

```bash
python -c "import json;[print(r['id'], r['median'], r['per_seed']) for r in json.load(open('results/final_numbers.json'))['records'] if r['id'].startswith('helmholtz2d|L3N10')]"
```

| Revised item | Numbers in | Regenerate with | Seeds behind each cell |
|---|---|---|---|
| Table 1, Figure 2 (Helmholtz) | `results/ff_baseline/helmholtz2d_{runs,summary}.csv` | `python experiments/make_jobs.py` → `python experiments/run_queue.py jobs_all.json --workers 6` → `python experiments/aggregate_task1.py` → `python experiments/make_paper_figures.py` | 10 at 2x10, 5 at 3x10 and 4x10 (seed = 1234 + trial) |
| Table 2, Figure 3 (diffusion-reaction) | `results/ff_baseline/diffusion_reaction_{runs,summary}.csv` | same as above | 5 per cell |
| Table 3, Figure 4 (Kovasznay) | `results/ff_baseline/kovasznay_{runs,summary}.csv`, `kovasznay_nr_comparison.csv` | same as above | 5 per cell; canonical N_r = 5000, robustness check at 10000 |
| Table 4, Figure 6 (PI-DeepONet) | `results/deeponet_rev/{runs,table4_replacement,paired_ci}.csv` | `python DeepONet/run_experiment_deeponet.py --nn {8,16,32,64} [--lpa] --trial {0..4}` → `python experiments/aggregate_deeponet.py` | 5 per width per model |
| Order study (orders below the PDE order) | `results/low_order/{runs,summary}.csv` | `python experiments/aggregate_task2.py` | 5 per order (10 at P = 6, reused from the baseline campaign) |
| Panel-count reparameterization | `results/dof_fixed/{runs,summary}.csv` | `python experiments/aggregate_task3.py` | 5 per cell |
| Sensitivity grid and warm-up sweep | `results/sensitivity/{runs,grid_summary,lr_summary}.csv` | `python experiments/aggregate_task4.py` | 5 per grid cell; 10 at (P = 6, N = 30, lr = 1e-2) |
| Same-protocol comparison | `results/major1_compare/{dev_runs,dev_selection,confirm_runs,confirm_summary,paired_statistics}.csv` | `python experiments/make_jobs_major1.py` → `run_queue.py` → `python experiments/aggregate_major1.py` | 3 development seeds (trials 0-2), 10 held-out confirmation seeds (trials 10-19); independent init/data/feature streams 1234 / 101234 / 201234 + trial |
| Cost paragraph | `2D_Helmholtz/results/runs/timing_single/` | `python 2D_Helmholtz/run_experiment.py --exp timing_single ...` on an otherwise idle machine | 2 runs at 2x10, 1 at 3x10 and 4x10 — descriptive only |
| Figures 2-5 (manuscript replacements) | `results/manuscript_revisions/figures/` | `python experiments/make_paper_figures.py`, `python experiments/make_fig5.py` | representative seed per panel = the seed whose final relative L2 is closest to the cell median, recorded in `results/manuscript_revisions/representative_seeds.json` |
| Supplementary data files | `results/supplementary/` (each file carries its own title and description) | `python experiments/build_supplementary.py` | as in the source campaigns |
| Figure inventory and placement | `results/figure_manifest.json` (sha256, generator, reviewer comment, proposed placement) | `python experiments/build_figure_manifest.py` | — |

`experiments/paths.py` resolves the benchmark directories, so the scripts run
from a checkout of this repository without editing paths.
`experiments/patch_manuscript_images.py` documents the two pixel-level
corrections applied to the original Figure 1 and Figure 5 rasters (a spelling
error and the latent dimension); it needs the manuscript file, which is not part
of this repository.


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
  `experiments/make_jobs_major1.py`, `experiments/aggregate_major1.py`): tanh
  PINN, LPA, fixed/trainable random Fourier features, SIREN, and neuron-wise
  adaptive tanh (learned neuron-wise slopes; the slope-recovery coefficient
  was part of the search and was selected as zero) under identical
  collocation points, loss, optimizer budget, and paired seeds, plus a
  target-frequency-informed oracle Fourier reference. Per-run records are in
  `2D_Helmholtz/results/runs/major1_{dev,confirm}/`; see
  `results/major1_compare/MAJOR1.md` for the frozen configurations and
  selection rule.
- **Code corrections in this release.** `DeepONet/config.py` is marked legacy:
  it drives the original single-run entry point and its `HEAD_WIDTH = 32` /
  `ARCH_LIST = [(3, 32)]` are *not* the settings of the reported
  operator-learning results, which come from `run_experiment_deeponet.py`
  (latent 64, head width 16, order 3, 16 panels, widths 8/16/32/64) and are
  recorded per run in `results/deeponet_rev/runs.csv`. The legacy
  `main_run_{R,LPA,ADAF}.py` scripts of the two 2-D benchmarks built their
  `properties` dictionary with `'xmin'`/`'xmax'` repeated in place of
  `'ymin'`/`'ymax'`; the keys are corrected here. The defect was inert — those
  scripts pass the domain to the solver through separate `lb`/`ub` tensors and
  nothing reads the affected keys — so no result changes.
- **Availability.** The implementation code, per-run records, aggregate
  tables, trained operator-network weights, representative figures, the
  supplementary data files, and `results/final_numbers.json` (every reported
  number with its per-seed values) are available in this repository; a
  versioned release is tagged for citation in the revised manuscript.
