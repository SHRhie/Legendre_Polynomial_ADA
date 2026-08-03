# Legendre_Polynomial-based_ADA

## Update (2026-08): unified-protocol re-runs and new baselines

All benchmark results were regenerated under a single seed-controlled protocol
(≥5 seeds per configuration, mean ± std with medians), with new experiments:
a parameter-matched random Fourier-feature baseline, a polynomial-order study
including orders below the PDE order, a fixed-degrees-of-freedom
(p- vs h-refinement) study, a hyperparameter sensitivity grid, and a
PI-DeepONet re-run with a full Reynolds sweep. See `revision/README.md` for
the framework and `results/` for aggregated CSVs and figures.
