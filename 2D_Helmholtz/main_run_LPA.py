import os
CPU_ONLY = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if CPU_ONLY else "100"
if CPU_ONLY:
    os.environ["TF_METAL_DEVICE"] = "0"

import pickle
import tensorflow as tf
#tf.keras.utils.disable_interactive_logging()

import numpy as np
import matplotlib.pyplot as plt

from time import time
from pinn_utils import *
from multiprocessing import Pool

LPA_ORDER = 3
LPA_PANELS = 30
SEED = 1234
SANITY_POINTS = 2048



def run(trial, condition):
    # Model keyword
    key = 'LPA'
    num_hidden_layers=condition[0]
    num_neurons_per_layer=condition[1]
    trial_seed = SEED + trial

    # Material Properties
    xmin = 0.0
    xmax = 1.0

    ymin = 0.0
    ymax = 1.0


    # Properties_dict
    properties = {
        'xmin':xmin,
        'xmax':xmax,
        'xmin':ymin,
        'xmax':ymax,    
        }

    DTYPE = 'float32'
    # Set number of data points
    N_b = 200
    N_r = 10000

    #Model construction
    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    ub = tf.constant([xmax, ymax], dtype=DTYPE)
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
            'adam_steps': 200,
            'adam_lr': 1e-2,
        },
    )
    residual_sanity_check(lb.numpy(), ub.numpy(), num_points=SANITY_POINTS, dtype=DTYPE, seed=trial_seed)
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
    pinn.model.summary()
    #tf.keras.utils.disable_interactive_logging()

    #Solver
    solver = Solver_PINN(pinn, properties, N_b=N_b, N_r=N_r)
    #Train Adam
    ref_time = time()
    solver.train_adam(200)
    time1 = time()-ref_time
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    #Train lbfgs
    ref_time = time()
    solver.ScipyOptimizer(method='L-BFGS-B', 
        options={'maxiter': 40000, 
            'maxfun': 50000, 
            'maxcor': 50, 
            'maxls': 50, 
            'ftol': np.finfo(float).eps,
            'gtol': np.finfo(float).eps,            
            'factr':np.finfo(float).eps,
            'iprint':50})
    time2 = time()-ref_time
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    solver.save_results(trial, (time1,time2))   
    solver.save_error()


def run_upper(trial):
    #condition_set = [(2,10), (3,10), (4,10)]
    condition_set = [(2,10)]
    for condition in condition_set:
        run(trial, condition)   
    

if __name__== '__main__':
    p = Pool(processes=1)
    p.map(run_upper, range(0,1))  # 0~7 trial → order 2,2,3,3,4,4,5,5
    
