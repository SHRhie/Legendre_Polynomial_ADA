# config.py
#
# LEGACY CONFIGURATION — not the source of the results reported in the revised
# manuscript.  This file drives the original single-run training entry point
# (train_deeponet.py).  Every operator-learning number in the revision comes
# from run_experiment_deeponet.py, whose command line carries the settings
# explicitly and records them in each run's JSON:
#
#     latent_dim   = 64      (--latent 64;        same as LATENT_DIM below)
#     head_width   = 16      (--head-width 16;    differs from HEAD_WIDTH below)
#     lpa_order    = 3       (--order 3)
#     lpa_panels   = 16      (--panels 16)
#     branch/trunk width sweep 8 / 16 / 32 / 64   (differs from ARCH_LIST below)
#     Adam only, 5000 steps, lr 2e-3, Re sweep 1..199, 5 seeds per cell
#
# The values kept here are therefore historical.  Do not cite them as the
# configuration of the reported experiments; see experiments/README.md and
# results/deeponet_rev/runs.csv (one row per reported run).
import os

# =========================
# Global runtime (CPU + float32)
# =========================
DTYPE = "float32"
CPU_ONLY = True  # True면 GPU 비활성화

# =========================
# Experiment naming (PINN-style)
# =========================
MODEL_VARIANT = "B"     # "A" or "B"
USE_LPA = True          # True: LPA, False: vanilla

TRIAL = 0

# PINN-style naming fields
NUM_HIDDEN_LAYERS = 3
NUM_NEURONS_PER_LAYER = 32

ARCH_LIST = [
    #(3, 64),
    (3, 32),
    #(2, 64),
]

# Auto key
KEY = f"DeepONet_PINN_{MODEL_VARIANT}_" + ("LPA" if USE_LPA else "VAN")

# =========================
# Domain
# =========================
# (xmin, xmax, ymin, ymax)
DOMAIN = (0.0, 1.0, -0.5, 1.5)

# =========================
# Operator parameter set (Re)
# =========================
RE_TRAIN_LIST = [50.0, 100.0, 150.0]
RE_TEST_LIST  = list(range(1, 200))#[35.0, 47.0, 143.0]
RE_REF = 40.0

# =========================
# Model hyperparameters
# =========================
LATENT_DIM = 64
HEAD_WIDTH = 32  # B-variant only

# LPA hyperparameters
LPA_ORDER = 3
LPA_PANELS = 16
LPA_SOFTMAX = False

# =========================
# Training hyperparameters
# =========================
N_INT = 10000      # residual points
N_B   = 200      # boundary points per edge
EPOCHS = 5000
LR = 2e-3

PDE_WEIGHT = .1
BC_WEIGHT  = 1.0

PRINT_EVERY = 200

# =========================
# Saving paths (PINN-style)
# =========================
RESULTS_ROOT = "./results"
CHECKPOINTS_ROOT = "./checkpoints"

def results_subdir():
    return os.path.join(RESULTS_ROOT, f"{NUM_HIDDEN_LAYERS}_{NUM_NEURONS_PER_LAYER}", KEY)

def ckpt_dir():
    return os.path.join(CHECKPOINTS_ROOT, f"{NUM_HIDDEN_LAYERS}_{NUM_NEURONS_PER_LAYER}", KEY)

def ckpt_path():
    # 이름은 PINN 호환 위해 ckpt_lbfgs 유지 (실제론 Adam only)
    return os.path.join(ckpt_dir(), f"ckpt_lbfgs_{TRIAL}")
