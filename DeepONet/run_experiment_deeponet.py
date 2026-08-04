"""Unified-protocol runner for the PI-DeepONet Kovasznay experiments (Section 4).

One (architecture, model, trial) training per process, following the published
protocol (Variant B, Adam-only, EPOCHS steps at LR, PDE_WEIGHT*pde + bc), with:
  - full seed control (seed = 1234 + trial),
  - acc_hist saved with an explicit header (7 columns:
    Re_ref, l1_u, l1_v, l1_p, l2_u, l2_v, l2_p) so column semantics are
    unambiguous (the published Table 4 misread these columns),
  - a full Reynolds sweep (Re = 1..199) evaluated on the 100x100 grid and
    saved per run,
  - a JSON config snapshot + final metrics per run,
  - weights archived per run.

Usage:
  python run_experiment_deeponet.py --nn 16 --lpa --trial 0 --exp deeponet_rev
"""
import os
import argparse
import json
import platform
import time

import numpy as np
import tensorflow as tf

import config as C
from core.deeponet import build_model_variant_B
from core.sampling import sample_interior, sample_boundary, make_eval_grid
from core.physics import ns_residual, boundary_loss
from core.utils import compute_errors_on_grid

SEED = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nh', type=int, default=3)
    ap.add_argument('--nn', type=int, default=16)
    ap.add_argument('--lpa', action='store_true')
    ap.add_argument('--order', type=int, default=3)
    ap.add_argument('--panels', type=int, default=16)
    ap.add_argument('--latent', type=int, default=64)
    ap.add_argument('--head-width', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--exp', default='deeponet_rev')
    ap.add_argument('--re-sweep-max', type=int, default=200, help='sweep Re = 1..max-1')
    args = ap.parse_args()

    tf.keras.backend.set_floatx(C.DTYPE)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass

    trial_seed = SEED + args.trial
    np.random.seed(trial_seed)
    tf.random.set_seed(trial_seed)

    key_id = ('LPA_K%d_N%d' % (args.order, args.panels)) if args.lpa else 'VAN'
    stem = '%s_%s_%s_%s' % (args.nh, args.nn, key_id, args.trial)
    out_dir = os.path.join('./results/runs', args.exp)
    ckpt_dir = os.path.join('./checkpoints/runs', args.exp, stem)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    cfg = {
        'variant': 'B', 'nh': args.nh, 'nn': args.nn, 'use_lpa': bool(args.lpa),
        'lpa_order': args.order if args.lpa else None,
        'lpa_panels': args.panels if args.lpa else None,
        'lpa_softmax': False,
        'latent_dim': args.latent, 'head_width': args.head_width,
        'domain': list(C.DOMAIN), 're_train': list(C.RE_TRAIN_LIST), 're_ref': C.RE_REF,
        'n_int': C.N_INT, 'n_b': C.N_B,
        'epochs': args.epochs, 'lr': args.lr,
        'pde_weight': C.PDE_WEIGHT, 'bc_weight': C.BC_WEIGHT,
        'seed': trial_seed, 'trial': args.trial, 'dtype': C.DTYPE,
        'machine': platform.machine(), 'tf': tf.__version__,
    }
    print('[cfg]', json.dumps(cfg))

    model = build_model_variant_B(
        latent_dim=args.latent,
        branch_width=args.nn, branch_depth=args.nh,
        trunk_width=args.nn, trunk_depth=args.nh,
        head_width=args.head_width,
        output_dim=3,
        use_lpa=bool(args.lpa),
        lpa_order=args.order, lpa_panels=args.panels, lpa_softmax=False,
        dtype=C.DTYPE,
    )
    model.summary()
    n_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    Xg, Yg, xy_grid = make_eval_grid(C.DOMAIN, Nx=100, Ny=100)

    loss_hist = []
    acc_hist = []

    @tf.function
    def train_step(Re_scalar):
        xy_int = sample_interior(C.DOMAIN, C.N_INT, dtype=tf.float32)
        X_b0, X_b1, Y_b0, Y_b1 = sample_boundary(C.DOMAIN, C.N_B, dtype=tf.float32)
        Re_int = tf.ones((C.N_INT, 1), dtype=tf.float32) * Re_scalar
        Re_b = tf.ones((C.N_B, 1), dtype=tf.float32) * Re_scalar
        with tf.GradientTape() as tape:
            x_mom, y_mom, cont = ns_residual(model, Re_int, xy_int)
            pde = (tf.reduce_mean(tf.square(x_mom)) +
                   tf.reduce_mean(tf.square(y_mom)) +
                   tf.reduce_mean(tf.square(cont)))
            bc = (boundary_loss(model, Re_b, X_b0) +
                  boundary_loss(model, Re_b, X_b1) +
                  boundary_loss(model, Re_b, Y_b0) +
                  boundary_loss(model, Re_b, Y_b1))
            total = C.PDE_WEIGHT * pde + C.BC_WEIGHT * bc
        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return total, pde, bc

    from tqdm import tqdm
    t0 = time.time()
    for ep in tqdm(range(1, args.epochs + 1), desc='Adam', unit='steps'):
        Re_choice = float(np.random.choice(C.RE_TRAIN_LIST))
        total, pde, bc = train_step(tf.constant(Re_choice, dtype=tf.float32))
        loss_hist.append(float(total.numpy()))
        if (ep % C.PRINT_EVERY == 0) or (ep == 1):
            Re_vec = np.full((xy_grid.shape[0], 1), C.RE_REF, dtype=np.float32)
            pred = model.predict([Re_vec, xy_grid], batch_size=8192, verbose=0).astype(np.float32)
            (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), _ = compute_errors_on_grid(C.RE_REF, Xg, Yg, pred)
            acc_hist.append([C.RE_REF, l1_u, l1_v, l1_p, l2_u, l2_v, l2_p])
    wall = time.time() - t0
    print('[time] %.1f s' % wall)

    # save histories (explicit header documents column semantics)
    np.savetxt(os.path.join(out_dir, 'loss_hist_%s.txt' % stem),
               np.array(loss_hist, dtype=np.float32), delimiter=',')
    np.savetxt(os.path.join(out_dir, 'acc_hist_%s.txt' % stem),
               np.array(acc_hist, dtype=np.float32), delimiter=',',
               header='Re_ref,l1_u,l1_v,l1_p,l2_u,l2_v,l2_p')
    np.savetxt(os.path.join(out_dir, 'cal_time_%s.txt' % stem),
               np.array([wall], dtype=np.float32), delimiter=',')

    # Reynolds sweep Re = 1..re_sweep_max-1 on the eval grid
    sweep_rows = []
    for Re in tqdm(range(1, args.re_sweep_max), desc='Re sweep', unit='Re'):
        Re_vec = np.full((xy_grid.shape[0], 1), float(Re), dtype=np.float32)
        pred = model.predict([Re_vec, xy_grid], batch_size=8192, verbose=0).astype(np.float32)
        (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), _ = compute_errors_on_grid(float(Re), Xg, Yg, pred)
        sweep_rows.append([Re, l1_u, l1_v, l1_p, l2_u, l2_v, l2_p])
    sweep = np.array(sweep_rows, dtype=np.float32)
    np.savetxt(os.path.join(out_dir, 're_sweep_%s.csv' % stem), sweep, delimiter=',',
               header='Re,l1_u,l1_v,l1_p,l2_u,l2_v,l2_p', comments='')

    model.save_weights(os.path.join(ckpt_dir, 'ckpt'))

    # final metrics at Re_ref + sweep aggregates
    Re_vec = np.full((xy_grid.shape[0], 1), C.RE_REF, dtype=np.float32)
    pred = model.predict([Re_vec, xy_grid], batch_size=8192, verbose=0).astype(np.float32)
    (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), _ = compute_errors_on_grid(C.RE_REF, Xg, Yg, pred)

    record = dict(cfg)
    record.update({
        'benchmark': 'kovasznay_pideeponet', 'key_id': key_id, 'n_params': n_params,
        'time_train': wall,
        'l1_u_ref': float(l1_u), 'l1_v_ref': float(l1_v), 'l1_p_ref': float(l1_p),
        'l2_u_ref': float(l2_u), 'l2_v_ref': float(l2_v), 'l2_p_ref': float(l2_p),
        'sweep_mean_l2_u': float(sweep[:, 4].mean()),
        'sweep_mean_l2_v': float(sweep[:, 5].mean()),
        'sweep_mean_l2_p': float(sweep[:, 6].mean()),
        'sweep_median_l2_u': float(np.median(sweep[:, 4])),
        'sweep_median_l2_v': float(np.median(sweep[:, 5])),
        'sweep_median_l2_p': float(np.median(sweep[:, 6])),
        'final_loss': float(loss_hist[-1]),
    })
    with open(os.path.join(out_dir, 'run_%s.json' % stem), 'w') as f:
        json.dump(record, f, indent=2)
    print('Saved %s/run_%s.json' % (out_dir, stem))
    print('FINAL Re=%g rel-L2 (u,v,p)=(%.3e, %.3e, %.3e); sweep median u=%.3e'
          % (C.RE_REF, l2_u, l2_v, l2_p, record['sweep_median_l2_u']))


if __name__ == '__main__':
    main()
