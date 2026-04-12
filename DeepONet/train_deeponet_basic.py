import os
import time
import numpy as np
import tensorflow as tf

import config_supervised as C
from core.deeponet import build_model_variant_A, build_model_variant_B
from core.sampling import sample_interior, sample_boundary, make_eval_grid
from core.physics import kovasznay_solution_tf
from core.utils import compute_errors_on_grid

def setup_runtime():
    tf.keras.backend.set_floatx(C.DTYPE)
    if C.CPU_ONLY:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

def build_model():
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

@tf.function
def supervised_step(model, opt, Re_scalar):
    # Sample points
    xy_int = sample_interior(C.DOMAIN, C.N_DATA, dtype=tf.float32)
    X_b0, X_b1, Y_b0, Y_b1 = sample_boundary(C.DOMAIN, C.N_B, dtype=tf.float32)

    Re_int = tf.ones((C.N_DATA, 1), dtype=tf.float32) * Re_scalar
    Re_b   = tf.ones((C.N_B,   1), dtype=tf.float32) * Re_scalar

    # Labels from analytic solution
    uvp_int_true = kovasznay_solution_tf(Re_int, xy_int)
    uvp_b_true_0 = kovasznay_solution_tf(Re_b, X_b0)
    uvp_b_true_1 = kovasznay_solution_tf(Re_b, X_b1)
    uvp_b_true_2 = kovasznay_solution_tf(Re_b, Y_b0)
    uvp_b_true_3 = kovasznay_solution_tf(Re_b, Y_b1)

    with tf.GradientTape() as tape:
        uvp_int_pred = model([Re_int, xy_int], training=True)
        data_loss = tf.reduce_mean(tf.square(uvp_int_pred - uvp_int_true))

        # boundary supervised loss (optional)
        bc = (
            tf.reduce_mean(tf.square(model([Re_b, X_b0], training=True) - uvp_b_true_0)) +
            tf.reduce_mean(tf.square(model([Re_b, X_b1], training=True) - uvp_b_true_1)) +
            tf.reduce_mean(tf.square(model([Re_b, Y_b0], training=True) - uvp_b_true_2)) +
            tf.reduce_mean(tf.square(model([Re_b, Y_b1], training=True) - uvp_b_true_3))
        )

        total = C.DATA_WEIGHT * data_loss + C.BC_WEIGHT * bc

    grads = tape.gradient(total, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return total, data_loss, bc

def main():
    setup_runtime()

    os.makedirs(C.results_subdir(), exist_ok=True)
    os.makedirs(C.ckpt_dir(), exist_ok=True)
    os.makedirs(C.RESULTS_ROOT, exist_ok=True)

    model = build_model()
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=C.LR)

    # Fixed eval grid (for reporting only)
    Xg, Yg, xy_grid = make_eval_grid(C.DOMAIN, Nx=100, Ny=100)

    loss_hist = []
    acc_hist  = []

    t0 = time.time()
    for ep in range(1, C.EPOCHS + 1):
        Re_choice = float(np.random.choice(C.RE_TRAIN_LIST))
        total, data_loss, bc = supervised_step(model, opt, tf.constant(Re_choice, dtype=tf.float32))

        total_val = float(total.numpy())
        loss_hist.append(total_val)

        if (ep % C.PRINT_EVERY == 0) or (ep == 1):
            # evaluation at RE_REF
            Re_vec = np.full((xy_grid.shape[0], 1), C.RE_REF, dtype=np.float32)
            pred = model.predict([Re_vec, xy_grid], batch_size=8192, verbose=0).astype(np.float32)
            (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), _ = compute_errors_on_grid(C.RE_REF, Xg, Yg, pred)
            acc_hist.append([C.RE_REF, l1_u, l1_v, l1_p, l2_u, l2_v, l2_p])

            print(
                f"[Ep {ep:05d}] total={total_val:.3e} | data={float(data_loss.numpy()):.3e} | bc={float(bc.numpy()):.3e} "
                f"|| RefRe={C.RE_REF}: RelL2(u,v,p)=({l2_u:.2e},{l2_v:.2e},{l2_p:.2e})"
            )

    wall = time.time() - t0

    # Save histories
    np.savetxt(
        os.path.join(C.RESULTS_ROOT, f"loss_hist_{C.NUM_HIDDEN_LAYERS}_{C.NUM_NEURONS_PER_LAYER}_{C.KEY}_{C.TRIAL}.txt"),
        np.array(loss_hist, dtype=np.float32),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(C.RESULTS_ROOT, f"acc_hist_{C.NUM_HIDDEN_LAYERS}_{C.NUM_NEURONS_PER_LAYER}_{C.KEY}_{C.TRIAL}.txt"),
        np.array(acc_hist, dtype=np.float32),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(C.RESULTS_ROOT, f"cal_time_{C.NUM_HIDDEN_LAYERS}_{C.NUM_NEURONS_PER_LAYER}_{C.KEY}_{C.TRIAL}.txt"),
        np.array([wall], dtype=np.float32),
        delimiter=",",
    )

    # Save checkpoint (weights only)
    model.save_weights(C.ckpt_path())
    print(f"[INFO] Weights saved to: {C.ckpt_path()}")

if __name__ == "__main__":
    main()
