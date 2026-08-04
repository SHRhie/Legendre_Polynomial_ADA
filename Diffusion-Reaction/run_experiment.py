"""Unified-protocol experiment runner for the 1D diffusion-reaction benchmark.

Runs a single (model config, trial) training following the paper protocol
(Adam 200 steps @ lr, then L-BFGS-B) and stores per-run artifacts under
./results/runs/<exp>/ (same layout as the Helmholtz unified-protocol runner).

Usage example:
  python run_experiment.py --key FF --sigma 5 --nh 2 --nn 10 --trial 0 --exp ff_baseline
"""
import os
CPU_ONLY = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if CPU_ONLY else "100"

import argparse
import json
import platform

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from time import time
from pinn_utils import *

SEED = 1234

# paper-default LPA configuration for this benchmark (used for FF param matching)
LPA_ORDER_DEFAULT = 3
LPA_PANELS_DEFAULT = 30


def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def ff_params_analytic(m, nh, nn, n_out=1, in_dim=2):
    p = (2*m)*nn + nn
    p += (nh-1)*(nn*nn + nn)
    p += nn*n_out + n_out
    return p


def match_ff_features(lpa_params, nh, nn, n_out=1):
    # ties go to the larger m (slightly favors the FF baseline)
    best_m, best_diff = 1, float('inf')
    for m in range(1, 65):
        diff = abs(ff_params_analytic(m, nh, nn, n_out) - lpa_params)
        if diff <= best_diff:
            best_m, best_diff = m, diff
    return best_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default='LPA', choices=['R', 'LPA', 'FF'])
    ap.add_argument('--sigma', type=float, default=1.0, help='FF sigma')
    ap.add_argument('--ff-features', type=int, default=0, help='FF frequencies m; 0 = match LPA params')
    ap.add_argument('--order', type=int, default=LPA_ORDER_DEFAULT)
    ap.add_argument('--panels', type=int, default=LPA_PANELS_DEFAULT)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--nh', type=int, default=2)
    ap.add_argument('--nn', type=int, default=10)
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--adam-steps', type=int, default=200)
    ap.add_argument('--lbfgs-maxiter', type=int, default=40000)
    ap.add_argument('--exp', default='default')
    args = ap.parse_args()

    tf.keras.utils.disable_interactive_logging()

    trial_seed = SEED + args.trial
    set_global_seed(trial_seed)

    DTYPE = 'float32'
    loss_dict = {
        'loss_BC_coeff': tf.constant([1e0, 1e0]),
        'loss_PDE_coeff': tf.constant(1e0),
        'loss_IC_coeff': tf.constant(1e0),
        'loss_GRAD_coeff': 0             #1e-5
        }

    # Material Properties (identical to main_run_LPA.py: x in [-pi, pi] as in the
    # paper; note main_run_R.py has L=1, under which the exact solution violates
    # the enforced boundary conditions and no model can drop below ~5.7e-1)
    viscosity = .01/np.pi
    time_concern = 1.
    L = np.pi
    tmin = 0.
    tmax = time_concern
    xmin = -L
    xmax = L
    time_stepping_number = 1
    time_marching_constant = 1
    properties = {
        'viscosity': viscosity,
        'L': L,
        'time_concern': time_concern,
        'time_stepping_number': time_stepping_number,
        'time_marching_constant': time_marching_constant,
        'tmin': tmin,
        'tmax': time_concern,
        'xmin': xmin,
        'xmax': L,
        }

    N_0 = 200
    N_b = 200
    N_r = 10000
    lb = tf.constant([tmin, xmin], dtype=DTYPE)
    ub = tf.constant([tmax/time_marching_constant, xmax], dtype=DTYPE)

    if args.key == 'R':
        key_id = 'R'
    elif args.key == 'LPA':
        key_id = 'LPA_P%d_N%d' % (args.order, args.panels)
        if abs(args.lr - 1e-2) > 1e-12:
            key_id += '_lr%g' % args.lr
    else:
        key_id = 'FF_s%g' % args.sigma
        if abs(args.lr - 1e-2) > 1e-12:
            key_id += '_lr%g' % args.lr

    ff_features = args.ff_features
    lpa_ref_params = None
    if args.key == 'FF':
        ref = Build_PINN(lb, ub, args.nh, args.nn, 'LPA',
                         lpa_order=LPA_ORDER_DEFAULT, lpa_panels=LPA_PANELS_DEFAULT)
        lpa_ref_params = count_params(ref.model)
        if ff_features == 0:
            ff_features = match_ff_features(lpa_ref_params, args.nh, args.nn, n_out=1)
        else:
            key_id += '_m%d' % ff_features
        del ref

    print_runtime_info(seed=trial_seed, extra_config={
        'key': args.key, 'key_id': key_id,
        'num_hidden_layers': args.nh, 'num_neurons_per_layer': args.nn,
        'lpa_order': args.order, 'lpa_panels': args.panels,
        'ff_sigma': args.sigma, 'ff_features': ff_features,
        'N_0': N_0, 'N_b': N_b, 'N_r': N_r,
        'adam_steps': args.adam_steps, 'adam_lr': args.lr,
        'lbfgs_maxiter': args.lbfgs_maxiter,
    })

    pinn = Build_PINN(lb, ub, args.nh, args.nn, args.key,
                      lpa_order=args.order, lpa_panels=args.panels,
                      ff_sigma=args.sigma, ff_features=ff_features, ff_seed=trial_seed)
    pinn.model.summary()
    n_params = count_params(pinn.model)
    print('Trainable parameters: %d (LPA reference: %s)' % (n_params, lpa_ref_params))

    solver = Solver_PINN(pinn, properties, loss_dict, N_0=N_0, N_b=N_b, N_r=N_r, lr=args.lr)

    ref_time = time()
    solver.train_adam(args.adam_steps)
    time1 = time() - ref_time
    print('\nComputation time (Adam): {} seconds'.format(time1))

    ref_time = time()
    solver.ScipyOptimizer(method='L-BFGS-B',
        options={'maxiter': args.lbfgs_maxiter,
            'maxfun': 50000,
            'maxcor': 50,
            'maxls': 50,
            'ftol': np.finfo(float).eps,
            'gtol': np.finfo(float).eps,
            'factr': np.finfo(float).eps,
            'iprint': 50})
    time2 = time() - ref_time
    print('\nComputation time (L-BFGS-B): {} seconds'.format(time2))

    # ---- final metrics -------------------------------------------------
    solver.accuracy_update()
    prediction = solver.cur_pinn.model.predict(solver.X_exam)
    exact = solver.call_exact(solver.X_exam).numpy()
    l1_legacy = float(np.mean(np.abs(prediction - exact)))
    l2_legacy = float(np.linalg.norm(prediction - exact, 2) / np.linalg.norm(exact, 2))

    # ---- save ----------------------------------------------------------
    out_dir = './results/runs/%s/' % args.exp
    os.makedirs(out_dir, exist_ok=True)
    stem = '%s_%s_%s_%s' % (args.nh, args.nn, key_id, args.trial)
    np.savetxt(out_dir + 'loss_hist_%s.txt' % stem, np.array(solver.loss_history), delimiter=',')
    np.savetxt(out_dir + 'acc_hist_%s.txt' % stem, np.array(solver.accuracy_history), delimiter=',')
    np.savetxt(out_dir + 'cal_time_%s.txt' % stem, np.array((time1, time2)), delimiter=',')

    record = {
        'benchmark': 'diffusion_reaction',
        'key': args.key, 'key_id': key_id,
        'nh': args.nh, 'nn': args.nn, 'trial': args.trial, 'seed': trial_seed,
        'lpa_order': args.order if args.key == 'LPA' else None,
        'lpa_panels': args.panels if args.key == 'LPA' else None,
        'ff_sigma': args.sigma if args.key == 'FF' else None,
        'ff_features': ff_features if args.key == 'FF' else None,
        'lr': args.lr,
        'adam_steps': args.adam_steps, 'lbfgs_maxiter': args.lbfgs_maxiter,
        'N_0': N_0, 'N_b': N_b, 'N_r': N_r,
        'L': float(L), 'domain': [[tmin, tmax], [float(xmin), float(xmax)]],
        'n_params': n_params, 'lpa_ref_params': lpa_ref_params,
        'l1_absolute': l1_legacy, 'l2_relative': l2_legacy,
        'final_loss': float(solver.loss),
        'time_adam': time1, 'time_lbfgs': time2,
        'lbfgs_steps': int(solver.lbfgs_step),
        'machine': platform.machine(),
    }
    with open(out_dir + 'run_%s.json' % stem, 'w') as f:
        json.dump(record, f, indent=2)
    print('Saved %srun_%s.json' % (out_dir, stem))
    print('FINAL l2_relative=%.6e l1_absolute=%.6e' % (l2_legacy, l1_legacy))


if __name__ == '__main__':
    main()
