import numpy as np
from time import time
from pinn_utils_mlx import *
from multiprocessing import Pool

ADAM_STEPS = 200
ADAM_LR = 1e-2
RESAMPLE_EVERY = None
BOUNDARY_WEIGHT = 150.0
LPA_ORDER = 3
LPA_PANELS = 30
SEED = 1234
SANITY_POINTS = 2048


def run(trial, condition):
    key = 'LPA'
    num_hidden_layers = condition[0]
    num_neurons_per_layer = condition[1]
    trial_seed = SEED + trial

    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    properties = {
        'xmin': xmin, 'xmax': xmax,
        'ymin': ymin, 'ymax': ymax,
    }

    N_b = 200
    N_r = 10000

    lb = [xmin, ymin]
    ub = [xmax, ymax]
    set_global_seed(trial_seed)
    print_runtime_info(
        seed=trial_seed,
        extra_config={
            'key': key,
            'num_hidden_layers': num_hidden_layers,
            'num_neurons_per_layer': num_neurons_per_layer,
            'lpa_order': LPA_ORDER,
            'lpa_panels': LPA_PANELS,
            'N_b': N_b,
            'N_r': N_r,
            'adam_steps': ADAM_STEPS,
            'adam_lr': ADAM_LR,
            'boundary_weight': BOUNDARY_WEIGHT,
            'resample_every': RESAMPLE_EVERY,
        },
    )
    residual_sanity_check(lb, ub, num_points=SANITY_POINTS, seed=trial_seed)
    pinn = Build_PINN(
        lb,
        ub,
        properties,
        num_hidden_layers,
        num_neurons_per_layer,
        key,
        lpa_order=LPA_ORDER,
        lpa_panels=LPA_PANELS,
    )

    solver = Solver_PINN(
        pinn,
        properties,
        N_b=N_b,
        N_r=N_r,
        adam_learning_rate=ADAM_LR,
        boundary_weight=BOUNDARY_WEIGHT,
    )

    # Train Adam
    ref_time = time()
    solver.train_adam(ADAM_STEPS, log_every=200, resample_every=RESAMPLE_EVERY)
    time1 = time() - ref_time
    print('\nComputation time: {} seconds'.format(time1))

    # Train L-BFGS
    ref_time = time()
    lbfgs_result = solver.ScipyOptimizer(
        method='L-BFGS-B',
        options=get_lbfgsb_options()
    )
    time2 = time() - ref_time
    print('\nComputation time: {} seconds'.format(time2))
    print_lbfgsb_result(lbfgs_result, label=f'L-BFGS-B ({key}, trial={trial})')

    solver.save_results(trial, (time1, time2))
    solver.save_error()


def run_upper(trial):
    condition_set = [(2, 10)]
    for condition in condition_set:
        run(trial, condition)


if __name__ == '__main__':
    p = Pool(processes=1)
    p.map(run_upper, range(0, 1))
