import os

# =========================
# Global runtime (CPU + float32)
# =========================
DTYPE = "float32"
CPU_ONLY = True  # True면 GPU 비활성화

# =========================
# Experiment naming
# =========================
MODEL_VARIANT = "A"     # "A" or "B"
USE_LPA = False         # True: LPA, False: vanilla (basic DeepONet은 보통 False 권장)

TRIAL = 0

NUM_HIDDEN_LAYERS = 3
NUM_NEURONS_PER_LAYER = 64

# Supervised DeepONet key
KEY = f"DeepONet_SUP_{MODEL_VARIANT}_" + ("LPA" if USE_LPA else "VAN")

# =========================
# Domain (Kovasznay)
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
HEAD_WIDTH = 16  # B-variant only

# LPA hyperparameters (USE_LPA=True일 때만 사용)
LPA_ORDER = 3
LPA_PANELS = 30
LPA_SOFTMAX = False

# =========================
# Training hyperparameters (supervised)
# =========================
N_DATA = 10000        # supervised collocation points per step
N_B    = 0         # boundary points per step (optional, but helps)
EPOCHS = 5000
LR = 5e-3

DATA_WEIGHT = 1.0
BC_WEIGHT   = 0.0

PRINT_EVERY = 200

# =========================
# Saving paths
# =========================
RESULTS_ROOT = "./results"
CHECKPOINTS_ROOT = "./checkpoints"

def results_subdir():
    return os.path.join(RESULTS_ROOT, f"{NUM_HIDDEN_LAYERS}_{NUM_NEURONS_PER_LAYER}", KEY)

def ckpt_dir():
    return os.path.join(CHECKPOINTS_ROOT, f"{NUM_HIDDEN_LAYERS}_{NUM_NEURONS_PER_LAYER}", KEY)

def ckpt_path():
    return os.path.join(ckpt_dir(), f"ckpt_{TRIAL}")
