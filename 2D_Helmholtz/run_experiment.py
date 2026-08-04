"""Unified-protocol experiment runner for the 2D Helmholtz benchmark.

Runs a single (model config, trial) training following the paper protocol
(Adam 200 steps @ lr, then L-BFGS-B) and stores per-run artifacts under
./results/runs/<exp>/ :
  - acc_hist / loss_hist / cal_time txt files (same format as main_run_*.py)
  - run_*.json with full config + final metrics (+ residual derivative norms)

Usage example:
  python run_experiment.py --key FF --sigma 5 --nh 2 --nn 10 --trial 0 --exp ff_baseline
  python run_experiment.py --key LPA --order 6 --panels 30 --nh 2 --nn 10 --trial 3 --exp ff_baseline
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
SANITY_POINTS = 2048
RESID_POINTS = 4096
RESID_SEED = 777

# paper-default LPA configuration for this benchmark (used for FF param matching)
LPA_ORDER_DEFAULT = 6
LPA_PANELS_DEFAULT = 30


def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def ff_params_analytic(m, nh, nn, n_out=1, in_dim=2):
    # gamma(x) has 2m features -> Dense(nn) x nh -> Dense(n_out)
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
    ap.add_argument('--key', default='LPA', choices=['R', 'R1', 'LPA', 'FF'])
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
    ap.add_argument('--acc-every', type=int, default=50)
    args = ap.parse_args()

    tf.keras.utils.disable_interactive_logging()

    trial_seed = SEED + args.trial
    set_global_seed(trial_seed)

    # Material Properties
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    properties = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
    DTYPE = 'float32'
    N_b = 200
    N_r = 10000
    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    ub = tf.constant([xmax, ymax], dtype=DTYPE)

    # key_id used in file names distinguishes hyperparameter variants
    if args.key in ('R', 'R1'):
        key_id = args.key
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
        ref = Build_PINN(lb, ub, properties, args.nh, args.nn, 'LPA',
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
        'N_b': N_b, 'N_r': N_r,
        'adam_steps': args.adam_steps, 'adam_lr': args.lr,
        'lbfgs_maxiter': args.lbfgs_maxiter,
    })
    residual_sanity_check(lb.numpy(), ub.numpy(), num_points=SANITY_POINTS, dtype=DTYPE, seed=trial_seed)

    pinn = Build_PINN(lb, ub, properties, args.nh, args.nn, args.key,
                      lpa_order=args.order, lpa_panels=args.panels,
                      ff_sigma=args.sigma, ff_features=ff_features, ff_seed=trial_seed)
    pinn.model.summary()
    n_params = count_params(pinn.model)
    print('Trainable parameters: %d (LPA reference: %s)' % (n_params, lpa_ref_params))

    solver = Solver_PINN(pinn, properties, N_b=N_b, N_r=N_r, lr=args.lr)
    solver.plot_every = args.acc_every

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
    prediction = solver.cur_pinn.model.predict(solver.XY_test)
    exact = solution(solver.XY_test).numpy()
    l1_legacy = float(np.mean(np.abs(prediction - exact)))
    l2_legacy = float(np.linalg.norm(prediction - exact, 2) / np.linalg.norm(exact, 2))
    # unified metrics: channel 0 (and channel-sum for multi-output heads)
    pred_c0 = prediction[:, 0:1]
    l1_c0 = float(np.mean(np.abs(pred_c0 - exact)))
    l2_c0 = float(np.linalg.norm((pred_c0 - exact).ravel()) / np.linalg.norm(exact.ravel()))
    pred_sum = prediction.sum(axis=1, keepdims=True)
    l2_sum = float(np.linalg.norm((pred_sum - exact).ravel()) / np.linalg.norm(exact.ravel()))

    res_norms = residual_derivative_norms(solver.cur_pinn.model, lb.numpy(), ub.numpy(),
                                          num_points=RESID_POINTS, seed=RESID_SEED)
    print('Residual derivative norms:', json.dumps(res_norms, indent=2))

    # ---- save ----------------------------------------------------------
    out_dir = './results/runs/%s/' % args.exp
    os.makedirs(out_dir, exist_ok=True)
    stem = '%s_%s_%s_%s' % (args.nh, args.nn, key_id, args.trial)
    np.savetxt(out_dir + 'loss_hist_%s.txt' % stem, np.array(solver.loss_history), delimiter=',')
    np.savetxt(out_dir + 'acc_hist_%s.txt' % stem, np.array(solver.accuracy_history), delimiter=',')
    np.savetxt(out_dir + 'cal_time_%s.txt' % stem, np.array((time1, time2)), delimiter=',')

    record = {
        'benchmark': 'helmholtz2d',
        'key': args.key, 'key_id': key_id,
        'nh': args.nh, 'nn': args.nn, 'trial': args.trial, 'seed': trial_seed,
        'lpa_order': args.order if args.key == 'LPA' else None,
        'lpa_panels': args.panels if args.key == 'LPA' else None,
        'ff_sigma': args.sigma if args.key == 'FF' else None,
        'ff_features': ff_features if args.key == 'FF' else None,
        'lr': args.lr,
        'adam_steps': args.adam_steps, 'lbfgs_maxiter': args.lbfgs_maxiter,
        'N_b': N_b, 'N_r': N_r,
        'domain': [[xmin, xmax], [ymin, ymax]],
        'n_params': n_params, 'lpa_ref_params': lpa_ref_params,
        'l1_absolute': l1_legacy, 'l2_relative': l2_legacy,
        'l1_absolute_c0': l1_c0, 'l2_relative_c0': l2_c0, 'l2_relative_sum': l2_sum,
        'final_loss': float(solver.loss),
        'time_adam': time1, 'time_lbfgs': time2,
        'lbfgs_steps': int(solver.lbfgs_step),
        'machine': platform.machine(),
    }
    record.update(res_norms)
    with open(out_dir + 'run_%s.json' % stem, 'w') as f:
        json.dump(record, f, indent=2)
    print('Saved %srun_%s.json' % (out_dir, stem))
    print('FINAL l2_relative=%.6e l1_absolute=%.6e' % (l2_legacy, l1_legacy))


if __name__ == '__main__':
    main()
