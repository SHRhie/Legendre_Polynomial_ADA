import os
import numpy as np
import tensorflow as tf

import config_supervised as C
from core.deeponet import build_model_variant_A, build_model_variant_B
from core.sampling import make_eval_grid
from core.utils import compute_errors_on_grid, plot_exact_pred_error

def setup_runtime():
    tf.keras.backend.set_floatx(C.DTYPE)
    if C.CPU_ONLY:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

def build_same_model():
    branch_width = C.NUM_NEURONS_PER_LAYER
    trunk_width  = C.NUM_NEURONS_PER_LAYER
    branch_depth = C.NUM_HIDDEN_LAYERS
    trunk_depth  = C.NUM_HIDDEN_LAYERS

    if C.MODEL_VARIANT == "A":
        return build_model_variant_A(
            latent_dim=C.LATENT_DIM,
            branch_width=branch_width, branch_depth=branch_depth,
            trunk_width=trunk_width,   trunk_depth=trunk_depth,
            output_dim=3,
            use_lpa=C.USE_LPA,
            lpa_order=C.LPA_ORDER, lpa_panels=C.LPA_PANELS, lpa_softmax=C.LPA_SOFTMAX,
            dtype=C.DTYPE,
        )
    return build_model_variant_B(
        latent_dim=C.LATENT_DIM,
        branch_width=branch_width, branch_depth=branch_depth,
        trunk_width=trunk_width,   trunk_depth=trunk_depth,
        head_width=C.HEAD_WIDTH,
        output_dim=3,
        use_lpa=C.USE_LPA,
        lpa_order=C.LPA_ORDER, lpa_panels=C.LPA_PANELS, lpa_softmax=C.LPA_SOFTMAX,
        dtype=C.DTYPE,
    )

def infer_single(Re_new, do_plot=True, save_txt=True):
    setup_runtime()
    os.makedirs(C.results_subdir(), exist_ok=True)

    model = build_same_model()
    model.load_weights(C.ckpt_path())
    print(f"[INFO] Loaded weights: {C.ckpt_path()}")

    Xg, Yg, xy_grid = make_eval_grid(C.DOMAIN, Nx=100, Ny=100)

    Re_vec = np.full((xy_grid.shape[0], 1), float(Re_new), dtype=np.float32)
    pred = model.predict([Re_vec, xy_grid], batch_size=8192, verbose=0).astype(np.float32)

    (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), exact = compute_errors_on_grid(float(Re_new), Xg, Yg, pred)

    print(
        f"[INFER Re={Re_new}] "
        f"L1(u,v,p)=({l1_u:.3e},{l1_v:.3e},{l1_p:.3e}) | "
        f"RelL2(u,v,p)=({l2_u:.3e},{l2_v:.3e},{l2_p:.3e})"
    )

    if save_txt:
        pred_path = os.path.join(C.results_subdir(), f"prediction_{C.KEY}_Re{Re_new}.txt")
        exact_path = os.path.join(C.results_subdir(), f"exact_{C.KEY}_Re{Re_new}.txt")
        np.savetxt(pred_path, pred, delimiter=",")
        np.savetxt(exact_path, exact.astype(np.float32), delimiter=",")
        print(f"[INFO] prediction saved: {pred_path}")
        print(f"[INFO] exact saved:      {exact_path}")

    if do_plot:
        plot_exact_pred_error(float(Re_new), C.DOMAIN, Xg, Yg, pred)

def main():
    for Re_new in C.RE_TEST_LIST:
        infer_single(Re_new, do_plot=False, save_txt=True)

if __name__ == "__main__":
    main()
