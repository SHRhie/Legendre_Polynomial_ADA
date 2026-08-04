"""Unified-protocol experiment runner for the Kovasznay flow benchmark (Re=40).

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
LPA_ORDER_DEFAULT = 6
LPA_PANELS_DEFAULT = 30


def count_params(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def ff_params_analytic(m, nh, nn, n_out=3, in_dim=2):
    p = (2*m)*nn + nn
    p += (nh-1)*(nn*nn + nn)
    p += nn*n_out + n_out
    return p


def match_ff_features(lpa_params, nh, nn, n_out=3):
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
    ap.add_argument('--nr', type=int, default=5000,
                    help='collocation points (main_run_R.py: 5000, main_run_LPA.py: 10000)')
    ap.add_argument('--adam-steps', type=int, default=200)
    ap.add_argument('--lbfgs-maxiter', type=int, default=40000)
    ap.add_argument('--exp', default='default')
    args = ap.parse_args()

    tf.keras.utils.disable_interactive_logging()

    trial_seed = SEED + args.trial
    set_global_seed(trial_seed)

    # Material Properties (identical to main_run_R.py)
    xmin, xmax = -0.5, 1.0
    ymin, ymax = -0.5, 1.5
    properties = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
    DTYPE = 'float32'
    N_b = 200
    N_r = args.nr
    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    ub = tf.constant([xmax, ymax], dtype=DTYPE)

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
        ref = Build_PINN(lb, ub, properties, args.nh, args.nn, 'LPA',
                         lpa_order=LPA_ORDER_DEFAULT, lpa_panels=LPA_PANELS_DEFAULT)
        lpa_ref_params = count_params(ref.model)
        if ff_features == 0:
            ff_features = match_ff_features(lpa_ref_params, args.nh, args.nn, n_out=3)
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

    pinn = Build_PINN(lb, ub, properties, args.nh, args.nn, args.key,
                      lpa_order=args.order, lpa_panels=args.panels,
                      ff_sigma=args.sigma, ff_features=ff_features, ff_seed=trial_seed)
    pinn.model.summary()
    n_params = count_params(pinn.model)
    print('Trainable parameters: %d (LPA reference: %s)' % (n_params, lpa_ref_params))

    solver = Solver_PINN(pinn, properties, N_b=N_b, N_r=N_r, lr=args.lr)

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
    u_pred, v_pred, p_pred = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    u, v, p = list(map(lambda x: x.numpy().reshape(-1), solution(solver.XY_test)))
    metrics = {
        'l1_absolute_u': float(np.mean(np.abs(u_pred-u))),
        'l2_relative_u': float(np.linalg.norm(u_pred-u, 2)/np.linalg.norm(u, 2)),
        'l1_absolute_v': float(np.mean(np.abs(v_pred-v))),
        'l2_relative_v': float(np.linalg.norm(v_pred-v, 2)/np.linalg.norm(v, 2)),
        'l1_absolute_p': float(np.mean(np.abs(p_pred-p))),
        'l2_relative_p': float(np.linalg.norm(p_pred-p, 2)/np.linalg.norm(p, 2)),
    }

    # ---- save ----------------------------------------------------------
    out_dir = './results/runs/%s/' % args.exp
    os.makedirs(out_dir, exist_ok=True)
    stem = '%s_%s_%s_%s' % (args.nh, args.nn, key_id, args.trial)
    np.savetxt(out_dir + 'loss_hist_%s.txt' % stem, np.array(solver.loss_history), delimiter=',')
    np.savetxt(out_dir + 'acc_hist_%s.txt' % stem, np.array(solver.accuracy_history), delimiter=',')
    np.savetxt(out_dir + 'cal_time_%s.txt' % stem, np.array((time1, time2)), delimiter=',')

    record = {
        'benchmark': 'kovasznay',
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
        'final_loss': float(solver.loss),
        'time_adam': time1, 'time_lbfgs': time2,
        'lbfgs_steps': int(solver.lbfgs_step),
        'machine': platform.machine(),
    }
    record.update(metrics)
    with open(out_dir + 'run_%s.json' % stem, 'w') as f:
        json.dump(record, f, indent=2)
    print('Saved %srun_%s.json' % (out_dir, stem))
    print('FINAL l2_relative_u=%.6e l2_relative_v=%.6e l2_relative_p=%.6e' %
          (metrics['l2_relative_u'], metrics['l2_relative_v'], metrics['l2_relative_p']))


if __name__ == '__main__':
    main()
