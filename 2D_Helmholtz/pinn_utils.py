import numpy as np
import tensorflow as tf
import scipy.optimize
from drawnow import drawnow
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
        lpa_panels=30):        
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = lb
        self.ub = ub
        self.key = key
        self.properties = properties
        self.lpa_order = lpa_order
        self.lpa_panels = lpa_panels
        if key == 'ADAF':
            self.model = self.init_model_ADAF()      
        elif key == 'R':
            self.model = self.init_model_VAN()  
        elif key == 'LPA':
            self.model = self.init_model_LPA()
        else:
            pass
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
      

class Solver_PINN():
    def __init__(self, pinn, properties, N_b=150, N_r=2500, show=False, DTYPE='float32'):
        self.ref_pinn = None
        self.loss_element = None                
                
        self.lbfgs_step = 0        
        self.loss_history = []
        self.cur_pinn = pinn
        self.properties = properties
        self.show = show
        self.DTYPE = DTYPE
        self.N_b = N_b
        self.N_r = N_r                
        self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r = self.data_sampling()        
        
        self.lr = None
        self.optim = None
        self.build_optimizer()                
        self.call_examset()
                
        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)

        self.accuracy_history =[]
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
        self.lr = 1e-2
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr) 
    
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
        self.optim.apply_gradients(zip(grad_theta, self.cur_pinn.model.trainable_weights))
        return 
    def train_adam(self, N=5000):
        for num_step in range(N):
            self.train_step()            
            # 기존: if num_step % 50 == 0:
            if num_step % getattr(self, 'plot_every', 10) == 0:
                self.accuracy_update()
                if self.show:
                    drawnow(self.plot_iteration)

    def accuracy_update(self):
        prediction = self.cur_pinn.model.predict(self.XY_test)
        exact = solution(self.XY_test)
        l1_absolute = np.mean(np.abs(prediction-exact))
        l2_relative = np.linalg.norm(prediction-exact,2)/np.linalg.norm(exact,2)
        print('     l1_absolute_error:   ', l1_absolute)   
        print('     l2_relative_error:   ', l2_relative)               
        self.accuracy_element = np.array([l1_absolute, l2_relative])
        self.accuracy_history.append(self.accuracy_element)    
    def callback(self, xr=None):
        self.loss_history.append(self.loss)
        plot_every = getattr(self, 'plot_every', 50)
        if self.lbfgs_step % plot_every == 0:
            self.accuracy_update()
            if self.show:
                drawnow(self.plot_iteration)
        self.lbfgs_step += 1
            
    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):    
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in self.cur_pinn.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)
            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            for v in self.cur_pinn.model.variables:
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

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                    x0 = x0,
                                    jac = True,
                                    callback=self.callback,
                                    method=method,
                                    **kwargs)
    
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
