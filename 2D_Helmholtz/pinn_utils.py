import numpy as np
import tensorflow as tf
import scipy.optimize
try:
    from drawnow import drawnow
except ImportError:
    def drawnow(draw_func, *args, **kwargs):
        draw_func()
from tqdm import tqdm
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import sympy
import os


import sympy as sp
import os
import sys
import platform
import multiprocessing as mp


K0 = 2*np.pi*2


def set_global_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_runtime_info(seed=None, extra_config=None):
    print('\nRuntime info (TensorFlow):')
    print(f'  python: {sys.version.split()[0]}')
    print(f'  platform: {platform.platform()}')
    print(f'  machine: {platform.machine()}')
    print(f'  multiprocessing_start_method: {mp.get_start_method()}')
    print(f'  numpy: {np.__version__}')
    print(f'  scipy: {scipy.__version__}')
    print(f'  tensorflow: {tf.__version__}')
    if seed is not None:
        print(f'  seed: {seed}')
    if extra_config:
        for key in sorted(extra_config):
            print(f'  {key}: {extra_config[key]}')

def solution(X_r):
    x, y = tf.split(X_r, 2, axis=1)
    return tf.math.sin(K0*x)*tf.math.sin(K0*y)


def residual_sanity_check(lb, ub, num_points=2048, dtype='float32', seed=0):
    rng = np.random.default_rng(seed)
    XY = rng.uniform(low=np.array(lb, dtype=np.float32), high=np.array(ub, dtype=np.float32), size=(num_points, 2))
    X_r = tf.convert_to_tensor(XY, dtype=dtype)

    with tf.GradientTape(persistent=True) as tape:
        x, y = tf.split(X_r, 2, axis=1)
        tape.watch(x)
        tape.watch(y)
        u = tf.math.sin(K0 * x) * tf.math.sin(K0 * y)
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    del tape

    residual = -u_xx - u_yy - K0 * K0 * u - K0 * K0 * tf.math.sin(K0 * x) * tf.math.sin(K0 * y)
    residual_np = residual.numpy()
    mean_abs = float(np.mean(np.abs(residual_np)))
    max_abs = float(np.max(np.abs(residual_np)))
    print(f'Residual sanity check (TF exact solution): mean_abs={mean_abs:.6e}, max_abs={max_abs:.6e}')
    return {'mean_abs': mean_abs, 'max_abs': max_abs}

def residual_derivative_norms(model, lb, ub, num_points=4096, seed=0, DTYPE='float32', chunk=512):
    """Norms of the PDE residual r and its derivatives r_x, r_y, r_xx, r_yy on random points.
    r_xx/r_yy require 4th-order derivatives of u, hence the 4-level tape nesting."""
    rng = np.random.default_rng(seed)
    XY = rng.uniform(low=np.array(lb, dtype='float32'), high=np.array(ub, dtype='float32'),
                     size=(num_points, 2)).astype(DTYPE)
    keys = ['r', 'r_x', 'r_y', 'r_xx', 'r_yy']
    sq_sums = dict.fromkeys(keys, 0.0)
    abs_sums = dict.fromkeys(keys, 0.0)
    abs_maxs = dict.fromkeys(keys, 0.0)
    n_total = 0
    for i0 in range(0, num_points, chunk):
        Xc = tf.convert_to_tensor(XY[i0:i0+chunk])
        x = Xc[:, 0:1]
        y = Xc[:, 1:2]
        with tf.GradientTape(persistent=True) as t4:
            t4.watch([x, y])
            with tf.GradientTape(persistent=True) as t3:
                t3.watch([x, y])
                with tf.GradientTape(persistent=True) as t2:
                    t2.watch([x, y])
                    with tf.GradientTape(persistent=True) as t1:
                        t1.watch([x, y])
                        u = model(tf.concat([x, y], axis=1))
                    u_x = t1.gradient(u, x)
                    u_y = t1.gradient(u, y)
                u_xx = t2.gradient(u_x, x)
                u_yy = t2.gradient(u_y, y)
                r = -u_xx - u_yy - K0*K0*u - K0*K0*tf.math.sin(K0*x)*tf.math.sin(K0*y)
            r_x = t3.gradient(r, x)
            r_y = t3.gradient(r, y)
        r_xx = t4.gradient(r_x, x)
        r_yy = t4.gradient(r_y, y)
        del t1, t2, t3, t4
        vals = {'r': r, 'r_x': r_x, 'r_y': r_y, 'r_xx': r_xx, 'r_yy': r_yy}
        n_total += int(Xc.shape[0])
        for k, v in vals.items():
            v_np = v.numpy()
            sq_sums[k] += float(np.sum(np.square(v_np)))
            abs_sums[k] += float(np.sum(np.abs(v_np)))
            abs_maxs[k] = max(abs_maxs[k], float(np.max(np.abs(v_np))))
    out = {}
    for k in keys:
        out['rms_%s' % k] = float(np.sqrt(sq_sums[k]/n_total))
        out['mean_abs_%s' % k] = float(abs_sums[k]/n_total)
        out['max_abs_%s' % k] = float(abs_maxs[k])
    return out


def get_Legendre_coefs(order=0, n_panel=10):
    x = sp.symbols('x')
    P = sp.legendre(order,x)
    P_int = sp.integrate(P,x)

    inds = np.linspace(-1,1,n_panel+1)
    coefs = np.array([P_int.subs(x,ind) for ind in inds],dtype='float')
    coefs = coefs[1:]-coefs[:-1]
    coefs *= (2.*order+1.)/2.
    return coefs

def Leg_Poly(x, order):
    if order == 1:
        return x
    elif order == 2:
        return 0.5*(3.*tf.math.square(x)-1.)
    elif order == 3:
        return 0.5*(5.*tf.math.pow(x,3)-3.*x)
    elif order == 4:
        return (1./8.)*(35.*tf.math.pow(x,4)-30.*tf.math.square(x)+3.)
    elif order == 5:
        return (1./8.)*(63*tf.math.pow(x,5)-70*tf.math.pow(x,3)+15.*x)
    elif order == 6:
        return (1./16.)*(231*tf.math.pow(x,6)-315*tf.math.pow(x,4)+105*tf.math.pow(x,2)-5.)
    
class LPA(tf.keras.layers.Layer):
    def __init__(self,  order = 3, N_p = 10, DTYPE='float32', kernel_regularizer=None):
        super(LPA, self).__init__()        
        self.N_p = N_p
        self.coefs = np.array([get_Legendre_coefs(i, N_p) for i in range(1,order+1)],dtype=DTYPE)
        self.order = order
        self.DTYPE = DTYPE
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    def build(self, input_shape):        
        self.shape = input_shape[-1]
        self.W_i = self.add_weight( 'W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               
    def call(self, inputs):
        if inputs.dtype == self.DTYPE:
            pass
        else:        
            inputs = tf.cast(inputs,self.DTYPE)
        Am = tf.tensordot(self.coefs, self.W_i,1)
        sum_ = tf.reduce_mean(self.W_i)
        for i in range(self.order):
            sum_ += Leg_Poly(inputs, i+1)*Am[i]
        return sum_


class GPA(tf.keras.layers.Layer):
    """Gegenbauer Polynomial-based Adaptive Activation.
    Generalizes LPA: C_n^(lambda)(x) with learnable lambda.
    lambda=0.5 recovers Legendre (LPA).
    """
    _GAUSS_PTS = np.array([
        -0.9061798459386640, -0.5384693101056831, 0.0,
         0.5384693101056831,  0.9061798459386640], dtype='float32')
    _GAUSS_WTS = np.array([
        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
        0.4786286704993665, 0.2369268850561891], dtype='float32')

    def __init__(self, order=3, N_p=10, init_lambda=0.5, DTYPE='float32', kernel_regularizer=None):
        super(GPA, self).__init__()
        self.N_p = N_p
        self.order = order
        self.DTYPE = DTYPE
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        init_raw = float(np.log(np.exp(init_lambda) - 1.0))
        self._lambda_raw = tf.Variable(
            init_raw, trainable=True, name='gegenbauer_lambda', dtype=DTYPE)
        self._gauss_pts = tf.constant(self._GAUSS_PTS, dtype=DTYPE)
        self._gauss_wts = tf.constant(self._GAUSS_WTS, dtype=DTYPE)
        self._panel_edges = tf.constant(
            np.linspace(-1.0, 1.0, N_p + 1).astype('float32'), dtype=DTYPE)

    @property
    def lam(self):
        return tf.nn.softplus(self._lambda_raw) + 1e-4

    def build(self, input_shape):
        self.shape = input_shape[-1]
        self.W_i = self.add_weight(
            'W_i', shape=(self.N_p,), initializer='random_normal',
            regularizer=self.kernel_regularizer, trainable=True, dtype=self.DTYPE)

    def _gegenbauer_all(self, n_max, x, lam):
        """Compute C_0^(lam) through C_{n_max}^(lam) at points x via recurrence."""
        C = [tf.ones_like(x)]
        if n_max >= 1:
            C.append(2.0 * lam * x)
        for k in range(2, n_max + 1):
            k_f = tf.constant(k, dtype=self.DTYPE)
            c_k = (1.0 / k_f) * (
                2.0 * x * (k_f + lam - 1.0) * C[k - 1]
                - (k_f + 2.0 * lam - 2.0) * C[k - 2])
            C.append(c_k)
        return C

    def _compute_coefs(self, lam):
        """Panel integrals of C_n^(lam)(x) via 5-pt Gauss quadrature."""
        a = self._panel_edges[:-1]
        b = self._panel_edges[1:]
        mid = (a + b) / 2.0
        half = (b - a) / 2.0
        mapped = mid[:, None] + half[:, None] * self._gauss_pts[None, :]
        mapped_flat = tf.reshape(mapped, [-1])
        C_all = self._gegenbauer_all(self.order, mapped_flat, lam)
        coefs = []
        for n in range(1, self.order + 1):
            C_vals = tf.reshape(C_all[n], [self.N_p, 5])
            integrals = half * tf.reduce_sum(
                self._gauss_wts[None, :] * C_vals, axis=1)
            norm = (2.0 * n + 1.0) / 2.0
            coefs.append(integrals * norm)
        return tf.stack(coefs)

    def call(self, inputs):
        if inputs.dtype != self.DTYPE:
            inputs = tf.cast(inputs, self.DTYPE)
        lam = self.lam
        coefs = self._compute_coefs(lam)
        Am = tf.tensordot(coefs, self.W_i, 1)
        C_input = self._gegenbauer_all(self.order, inputs, lam)
        sum_ = tf.reduce_mean(self.W_i)
        for i in range(self.order):
            sum_ = sum_ + C_input[i + 1] * Am[i]
        return sum_


class FourierFeatures(tf.keras.layers.Layer):
    """Random Fourier feature mapping (Tancik et al. 2020).
    gamma(x) = [cos(2*pi*B*x), sin(2*pi*B*x)], B ~ N(0, sigma^2), B fixed (non-trainable).
    Applied after the [-1,1] input scaling layer.
    """
    def __init__(self, num_features=10, sigma=1.0, DTYPE='float32', seed=None):
        super(FourierFeatures, self).__init__()
        self.num_features = num_features
        self.sigma = sigma
        self.DTYPE = DTYPE
        self.seed = seed
    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        rng = np.random.default_rng(self.seed)
        B0 = rng.normal(0.0, self.sigma, size=(in_dim, self.num_features)).astype(self.DTYPE)
        self.B = self.add_weight('B', shape=(in_dim, self.num_features),
            initializer=tf.keras.initializers.Constant(B0), trainable=False, dtype=self.DTYPE)
    def call(self, inputs):
        if inputs.dtype != self.DTYPE:
            inputs = tf.cast(inputs, self.DTYPE)
        proj = 2.0*np.pi*tf.matmul(inputs, self.B)
        return tf.concat([tf.math.cos(proj), tf.math.sin(proj)], axis=1)


def get_XB(lb, ub, N_b, DTYPE='float32'):
    x_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    y_b = tf.random.uniform((N_b,1), lb[1], ub[1], dtype=DTYPE)
    
    x_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[0]
    x_L = tf.ones((N_b,1),dtype=DTYPE)*ub[0]
    y_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[1]
    y_L = tf.ones((N_b,1),dtype=DTYPE)*ub[1]

    X_b_0 = tf.concat([x_0, y_b], axis=1)
    X_b_L = tf.concat([x_L, y_b], axis=1)
    Y_b_0 = tf.concat([x_b, y_0], axis=1)
    Y_b_L = tf.concat([x_b, y_L], axis=1)    
    return X_b_0, X_b_L, Y_b_0, Y_b_L

def get_Xr(lb, ub, N_r, DTYPE='float32'):    
    x_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    y_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
    XY_r = tf.concat([x_r, y_r], axis=1)
    return XY_r

class Custom_Normal(tf.keras.layers.Layer):
    def __init__(self):
        super(Custom_Normal, self).__init__() 
    def call(self, inputs):  
        max_ = tf.math.reduce_max(inputs)
        min_ = tf.math.reduce_min(inputs)
        return (inputs - min_)/(max_ - min_)
        
        
class ADAF(tf.keras.layers.Layer):
    def __init__(self,  N_p = 5, N_m = 5, L=1.,  DTYPE='float32', kernel_regularizer=None):
        super(ADAF, self).__init__()        
        self.N_p = N_p
        self.N_m = N_m
        self.L = L
        self.x_i = tf.cast(tf.linspace(0., L, N_p+1),dtype=DTYPE)
        self.DTYPE = DTYPE
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    def build(self, input_shape):        
        self.W_i = self.add_weight('W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               
        self.w = self.add_weight('w', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)        

    def out_an(self, n, x_1, x_2, W_i):
        if n == 0:
            a_n = tf.reduce_sum(W_i)
            a_n = a_n/self.N_p
        else:
            sum_1 = tf.math.sin(n*np.pi/self.L* x_1)
            sum_2 = -tf.math.sin(n*np.pi/self.L* x_2)
            a_n = W_i * (sum_1 + sum_2)
            a_n = tf.reduce_sum(a_n)
            a_n = (2./(n*np.pi)) * a_n
        return a_n 
    def out_bn(self, n, x_1, x_2, W_i):                
        sum_1 = -tf.math.cos(n*np.pi/self.L* x_1)
        sum_2 = tf.math.cos(n*np.pi/self.L* x_2)        
        b_n = W_i *(sum_1 + sum_2)
        b_n = tf.reduce_sum(b_n)
        b_n = (2./ (n*np.pi))*b_n
        return b_n
    def out_g_x_1(self, x):           
        x_1 = self.x_i[1:]
        x_2 = self.x_i[:-1]                
                
        g_x = tf.cast(0., self.DTYPE)
        g_x += self.out_an(0, x_1, x_2, self.W_i)/2. * tf.math.square(x)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(1.-tf.math.cos(x/factor))
        return g_x
    def call(self, inputs):
        return self.w*self.out_g_x_1(inputs)


                  
class Build_PINN():
    def __init__(self, lb, ub, properties,
        num_hidden_layers=2,
        num_neurons_per_layer=10,
        key = 'R',
        lpa_order=6,
        lpa_panels=30,
        init_lambda=0.5,
        ff_sigma=1.0,
        ff_features=3,
        ff_seed=None):
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = lb
        self.ub = ub
        self.key = key
        self.properties = properties
        self.lpa_order = lpa_order
        self.lpa_panels = lpa_panels
        self.init_lambda = init_lambda
        self.ff_sigma = ff_sigma
        self.ff_features = ff_features
        self.ff_seed = ff_seed
        if key == 'ADAF':
            self.model = self.init_model_ADAF()
        elif key == 'R':
            self.model = self.init_model_VAN()
        elif key == 'R1':
            self.model = self.init_model_VAN1()
        elif key == 'LPA':
            self.model = self.init_model_LPA()
        elif key == 'GPA':
            self.model = self.init_model_GPA()
        elif key.startswith('FF'):
            self.model = self.init_model_FF()
        else:
            pass
    def init_model_VAN1(self):
        # vanilla MLP with a single-output head (like-for-like with LPA/FF;
        # the legacy 'R' model keeps its published 3-output head)
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)
        for _ in range(self.num_hidden_layers):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    def init_model_VAN(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)        
        for _ in range(self.num_hidden_layers):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(hiddens)
        prediction = tf.keras.layers.Dense(3)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    def init_model_ADAF(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)               
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal', 
                activation='tanh',
                )(hiddens)
        for _ in range(self.num_hidden_layers-2):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                    kernel_initializer='glorot_normal', 
                    activation='tanh',
                    )(hiddens)
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal', 
                )(hiddens)
        hiddens = ADAF(3,3)(hiddens)
        hiddens = tf.math.tanh(hiddens)
        prediction = tf.keras.layers.Dense(3)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    # Build_PINN.init_model_LPA 내부만 변경
    def init_model_LPA(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)               
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal', 
                activation='tanh',
                )(hiddens)
        for _ in range(self.num_hidden_layers-2):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                    kernel_initializer='glorot_normal', 
                    activation='tanh',
                    )(hiddens)
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation='tanh',
                kernel_initializer='glorot_normal', 
                )(hiddens)
        hiddens = LPA(self.lpa_order, self.lpa_panels)(hiddens)
        #hiddens = tf.math.tanh(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    def init_model_FF(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)
        hiddens = FourierFeatures(self.ff_features, self.ff_sigma, seed=self.ff_seed)(hiddens)
        for _ in range(self.num_hidden_layers):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
    def init_model_GPA(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal',
                activation='tanh',
                )(hiddens)
        for _ in range(self.num_hidden_layers-2):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                    kernel_initializer='glorot_normal',
                    activation='tanh',
                    )(hiddens)
        hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation='tanh',
                kernel_initializer='glorot_normal',
                )(hiddens)
        hiddens = GPA(self.lpa_order, self.lpa_panels, self.init_lambda)(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model

class Solver_PINN():
    def __init__(self, pinn, properties, N_b=150, N_r=2500, show=False, DTYPE='float32', lr=1e-2):
        self.ref_pinn = None
        self.loss_element = None

        self.lbfgs_step = 0
        self.loss_history = []
        self.cur_pinn = pinn
        self.properties = properties
        self.show = show
        self.DTYPE = DTYPE
        self.lr_init = lr
        self.N_b = N_b
        self.N_r = N_r
        self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r = self.data_sampling()        
        
        self.lr = None
        self.optim = None
        self.optim_lambda = None
        self.build_optimizer()
        self.call_examset()

        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)

        self.accuracy_history =[]
        self.lambda_history = []
    def data_sampling(self):    
        X_b_0, X_b_L, Y_b_0, Y_b_L = get_XB(self.cur_pinn.lb, self.cur_pinn.ub, self.N_b)
        XY_r = get_Xr(self.cur_pinn.lb, self.cur_pinn.ub, self.N_r)
        return X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r
    def call_examset(self):
        x = np.linspace(self.cur_pinn.lb[0],self.cur_pinn.ub[0],100)
        y = np.linspace(self.cur_pinn.lb[1],self.cur_pinn.ub[1],100)
        xx, yy = np.meshgrid(x,y)
        self.XY_test = np.stack((xx.flatten(), yy.flatten()), axis=1)    
    def save_results(self, trial, times, num_hidden_layers=2, num_neurons_per_layer=10):
        self.accuracy_update()
        self.loss_history.append(self.loss)
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_lbfgs_%s' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial))        
        np.savetxt('./results/loss_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.loss_history), delimiter=',')
        np.savetxt('./results/acc_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.accuracy_history), delimiter=',') 
        np.savetxt('./results/cal_time_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(times), delimiter=',')
        if self.cur_pinn.key == 'GPA' and self.lambda_history:
            np.savetxt('./results/lambda_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.lambda_history), delimiter=',') 
    def plot_iteration(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # ── 1) one-time defaults (속성 없으면 기본값 세팅) ─────────────────
        if not hasattr(self, 'live_axis'):
            self.live_axis   = 'y'      # 'y': u(x,y0), 'x': u(y,x0)
        if not hasattr(self, 'live_index'):
            self.live_index  = None     # 정수 인덱스 우선
        if not hasattr(self, 'live_value'):
            self.live_value  = None     # 좌표값으로 가장 가까운 인덱스 선택
        if not hasattr(self, 'live_out_idx'):
            self.live_out_idx = 0       # 출력이 여러 개일 때 사용할 채널
        if not hasattr(self, '_live_enabled'):
            self._live_enabled = True   # 'p' 키로 토글
        if not hasattr(self, '_keybound'):
            self._keybound = False

        if not self._live_enabled:
            return  # 토글 OFF면 그리지 않음

        # ── 2) 예측/정답 계산 ───────────────────────────────────────────────
        pred  = self.cur_pinn.model.predict(self.XY_test, verbose=0)
        exact = solution(self.XY_test).numpy()

        pred  = np.atleast_2d(pred)
        exact = np.atleast_2d(exact)
        if pred.ndim  == 1: pred  = pred[:, None]
        if exact.ndim == 1: exact = exact[:, None]
        k = int(self.live_out_idx)
        predk  = pred[:,  k].ravel()
        exactk = exact[:, k].ravel()

        # ── 3) 격자 해석(100×100) & 단면 선택 ───────────────────────────────
        xs = self.XY_test[:, 0]; ys = self.XY_test[:, 1]
        ux = np.unique(xs);       uy = np.unique(ys)
        NX, NY = ux.size, uy.size

        if self.live_axis == 'y':      # 고정 y: u(x, y0)
            if self.live_index is not None:
                j = int(self.live_index)
            else:
                j = NY//2 if self.live_value is None else int(np.argmin(np.abs(uy - self.live_value)))
            start, end = j*NX, (j+1)*NX
            xx = ux
            e  = exactk[start:end]
            p  = predk[start:end]
            xlabel = 'x'
            title  = f'Fit @ y ≈ {uy[j]:.3g} (idx {j})'
        else:                          # 고정 x: u(y, x0)
            if self.live_index is not None:
                i = int(self.live_index)
            else:
                i = NX//2 if self.live_value is None else int(np.argmin(np.abs(ux - self.live_value)))
            idx = np.arange(i, NX*NY, NX)
            xx = uy
            e  = exactk[idx]
            p  = predk[idx]
            xlabel = 'y'
            title  = f'Fit @ x ≈ {ux[i]:.3g} (idx {i})'

        rel = np.linalg.norm(p - e) / (np.linalg.norm(e) + 1e-12)

        # ── 4) 그리기 ───────────────────────────────────────────────────────
        plt.clf()
        ax = plt.gca()
        ax.plot(xx, e, '-',  lw=1.2, color='black', label='Exact')
        ax.plot(xx, p, '--', lw=1.2, color='red',   label=f'PINN (rel L2={rel:.2e})')
        ax.set_xlabel(xlabel); ax.set_ylabel('u')
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
        ax.legend(frameon=False, fontsize=10, loc='best')
        ax.set_title(title)

        # 첫 호출 때만 키 이벤트 바인딩: 'p'로 ON/OFF
        if not self._keybound:
            def _on_key(ev):
                if ev.key == 'p':
                    self._live_enabled = not self._live_enabled
                    print(f'[live plot] {"ON" if self._live_enabled else "OFF"}')
            plt.gcf().canvas.mpl_connect('key_press_event', _on_key)
            self._keybound = True
            
    def build_optimizer(self):
        del self.lr
        del self.optim
        if self.optim_lambda is not None:
            del self.optim_lambda
        self.lr = self.lr_init
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if self.cur_pinn.key == 'GPA':
            self.optim_lambda = tf.keras.optimizers.Adam(learning_rate=self.lr * 0.1)
        else:
            self.optim_lambda = None
    
    def get_B(self, X):
        pred = self.cur_pinn.model(X)
        return tf.reduce_mean(tf.square(pred))
    def source(self, x, y):
        return K0*K0*tf.math.sin(K0*x)*tf.math.sin(K0*y)
    def get_r(self, X_r):
        # 기존 residual 계산만 하던 부분 그대로 둡니다.
        with tf.GradientTape(persistent=True) as tape:
            x, y = tf.split(X_r, 2, axis=1)
            tape.watch(x); tape.watch(y)
            u = self.cur_pinn.model(tf.stack([x[:,0], y[:,0]], axis=1))
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        del tape
        return -u_xx - u_yy - K0*K0*u - self.source(x, y)

    def get_r_and_grads(self, X_r):
        # 1) inner tape로 residual r 계산
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(X_r)
            with tf.GradientTape(persistent=True) as tape:
                x, y = tf.split(X_r, 2, axis=1)
                tape.watch(x); tape.watch(y)
                u = self.cur_pinn.model(tf.stack([x[:,0], y[:,0]], axis=1))
                u_x = tape.gradient(u, x)
                u_y = tape.gradient(u, y)
            u_xx = tape.gradient(u_x, x)
            u_yy = tape.gradient(u_y, y)
            del tape
            r = -u_xx - u_yy - K0*K0*u - self.source(x, y)

        # 2) outer tape로 ∂r/∂x, ∂r/∂y 계산
        r_x = tape2.gradient(r, X_r)[:,0:1]
        r_y = tape2.gradient(r, X_r)[:,1:2]
        del tape2

        return r, r_x, r_y

    def compute_loss(self):
        # PDE residual + gradient-enhanced 항
        r, r_x, r_y = self.get_r_and_grads(self.XY_r)

        # 기본 residual loss
        Phi_r  = tf.reduce_mean(tf.square(r))
        # gradient loss
        Phi_rx = tf.reduce_mean(tf.square(r_x))
        Phi_ry = tf.reduce_mean(tf.square(r_y))

        # 가중치: 필요에 따라 조정
        lambda_g = 1e-2

        # 경계 손실 (기존)
        BX0 = self.get_B(self.X_b_0)
        BXL = self.get_B(self.X_b_L)
        BY0 = self.get_B(self.Y_b_0)
        BYL = self.get_B(self.Y_b_L)

        # 총 손실
        total_loss = Phi_r + 150*(BX0 + BXL + BY0 + BYL) #+ lambda_g*(Phi_rx + Phi_ry)
        return total_loss
    @tf.function    
    def get_grad(self):
        with tf.GradientTape() as tape:
            tape.watch(self.cur_pinn.model.trainable_weights)
            total_loss = self.compute_loss()
        g = tape.gradient(total_loss, self.cur_pinn.model.trainable_weights)
        del tape
        return g, total_loss
    def train_step(self):
        grad_theta, loss = self.get_grad()
        self.loss = loss
        self.loss_history.append(self.loss)
        if self.cur_pinn.key == 'GPA':
            all_vars = self.cur_pinn.model.trainable_weights
            main_grads, main_vars = [], []
            lam_grads, lam_vars = [], []
            for g, v in zip(grad_theta, all_vars):
                if 'gegenbauer_lambda' in v.name:
                    lam_grads.append(g)
                    lam_vars.append(v)
                else:
                    main_grads.append(g)
                    main_vars.append(v)
            self.optim.apply_gradients(zip(main_grads, main_vars))
            if lam_grads:
                self.optim_lambda.apply_gradients(zip(lam_grads, lam_vars))
                for v in lam_vars:
                    raw_min = float(np.log(np.exp(0.01) - 1.0))
                    raw_max = float(np.log(np.exp(5.0) - 1.0))
                    v.assign(tf.clip_by_value(v, raw_min, raw_max))
        else:
            self.optim.apply_gradients(zip(grad_theta, self.cur_pinn.model.trainable_weights))
        return
    def train_adam(self, N=5000):
        for num_step in tqdm(range(N), desc='Adam', unit='steps'):
            self.train_step()
            # 기존: if num_step % 50 == 0:
            if num_step % getattr(self, 'plot_every', 10) == 0:
                self.accuracy_update()
                if self.show:
                    drawnow(self.plot_iteration)

    def _get_current_lambda(self):
        for layer in self.cur_pinn.model.layers:
            if isinstance(layer, GPA):
                return float(layer.lam.numpy())
        return None

    def accuracy_update(self):
        prediction = self.cur_pinn.model.predict(self.XY_test)
        exact = solution(self.XY_test)
        l1_absolute = np.mean(np.abs(prediction-exact))
        l2_relative = np.linalg.norm(prediction-exact,2)/np.linalg.norm(exact,2)
        print('     l1_absolute_error:   ', l1_absolute)
        print('     l2_relative_error:   ', l2_relative)
        if self.cur_pinn.key == 'GPA':
            lam_val = self._get_current_lambda()
            print(f'     gegenbauer_lambda:   {lam_val:.6f}')
            self.lambda_history.append(lam_val)
        self.accuracy_element = np.array([l1_absolute, l2_relative])
        self.accuracy_history.append(self.accuracy_element)
    def callback(self, xr=None):
        self.loss_history.append(self.loss)
        if getattr(self, 'pbar', None) is not None:
            self.pbar.update(1)
        plot_every = getattr(self, 'plot_every', 50)
        if self.lbfgs_step % plot_every == 0:
            self.accuracy_update()
            if self.show:
                drawnow(self.plot_iteration)
        self.lbfgs_step += 1

    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        self.pbar = tqdm(total=kwargs.get('options', {}).get('maxiter', None), desc='L-BFGS-B', unit='steps')
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in self.cur_pinn.model.trainable_variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            for v in self.cur_pinn.model.trainable_variables:
                vs = v.shape
                
                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1]))
                    idx += sw
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx+=vs[0]
                elif len(vs) ==0:
                    new_val = weight_list[idx]
                    idx+=1
                elif len(vs) ==3:
                    sw = vs[0]*vs[1]*vs[2]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2]))                    
                    idx += sw
                elif len(vs) == 4:
                    sw = vs[0]*vs[1]*vs[2]*vs[3]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2],vs[3]))                    
                    idx += sw                    
                v.assign(tf.cast(new_val, self.DTYPE))   
        
        def get_loss_and_grad(w):
            set_weight_tensor(w)
            grad, loss = self.get_grad()
            loss = loss.numpy().astype(np.float64)
            grad_flat=[]
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            grad_flat = np.array(grad_flat, dtype=np.float64)
            self.loss = loss
            return loss, grad_flat

        result = scipy.optimize.minimize(fun=get_loss_and_grad,
                                    x0 = x0,
                                    jac = True,
                                    callback=self.callback,
                                    method=method,
                                    **kwargs)
        if getattr(self, 'pbar', None) is not None:
            self.pbar.close()
            self.pbar = None
        return result

    def save_error(self):
        self.prediction = self.cur_pinn.model.predict(self.XY_test)        
        self.exact = solution(self.XY_test)
        l1_absolute = np.mean(np.abs(self.prediction-self.exact))
        l2_relative = np.linalg.norm(self.prediction-self.exact,2)/np.linalg.norm(self.exact,2)
        print('l2_absolute_error:   ', l1_absolute)   
        print('l2_relative_error:   ', l2_relative)
        np.savetxt(self.path+'prediction_%s.txt' % self.cur_pinn.key, self.prediction, delimiter=',')
        np.savetxt(self.path+'exact_%s.txt' % self.cur_pinn.key, self.exact, delimiter=',')
        f = open(self.path2+'Error_%s_%s_%s.txt'% (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key), 'w')
        f.write('l1_absolute_error:  %s\n' % l1_absolute)
        f.write('l2_relative_error:   %s\n' % l2_relative)
        f.close()
