# config.py
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
