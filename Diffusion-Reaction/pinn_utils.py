import numpy as np
import tensorflow as tf
import scipy.optimize
import os
import matplotlib.pyplot as plt
import scipy.io

from drawnow import drawnow
from matplotlib.pyplot import cm

import sympy as sp
import os

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
        self.W_i = self.add_weight('W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               
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
        self.w = self.add_weight('w', shape=(), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)        
        self.W_i = self.add_weight('W_i', shape=(self.N_p,), initializer='random_normal', regularizer = self.kernel_regularizer, trainable=True, dtype=self.DTYPE)               
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
                
        g_x = self.out_an(0, x_1, x_2, self.W_i)/2. * tf.math.square(x)
        for n in range(1,self.N_m+1):
            factor = self.L/(n*np.pi)
            factor = tf.constant(factor,self.DTYPE)            
            g_x += tf.math.square(factor)*self.out_an(n, x_1, x_2, self.W_i)*(1.-tf.math.cos(x/factor))
        return g_x
    def call(self, inputs):
        return self.w*self.out_g_x_1(inputs)



def get_X0(lb, ub, N_0, DTYPE='float32'):
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
    x_0 = tf.cast(np.random.uniform(lb[1], ub[1], (N_0, 1)), dtype=DTYPE)
    X_0 = tf.concat([t_0, x_0], axis=1)
    return X_0

def get_XB(lb, ub, N_b, DTYPE='float32'):    
    t_b = tf.cast(np.random.uniform(lb[0], ub[0], (N_b,1)), dtype=DTYPE)
    x_b_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[1]
    x_b_L = tf.ones((N_b,1),dtype=DTYPE)*ub[1]
    X_b_0 = tf.concat([t_b, x_b_0], axis=1)
    X_b_L = tf.concat([t_b, x_b_L], axis=1)
    return X_b_0, X_b_L

def get_Xr(lb, ub, N_r, DTYPE='float32'):    
    t_r = tf.cast(np.random.uniform(lb[0], ub[0], (N_r,1)), dtype=DTYPE)
    x_r = tf.cast(np.random.uniform(lb[1], ub[1], (N_r,1)), dtype=DTYPE)
    X_r = tf.concat([t_r, x_r], axis=1)
    return X_r
                  
class Build_PINN():
    def __init__(self, lb, ub, 
        num_hidden_layers=2, 
        num_neurons_per_layer=10, 
        key = 'R'):        
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = lb
        self.ub = ub
        self.key =key
        if key == 'R':
            self.model = self.init_model_VAN()        
        elif key == 'ADAF':
            self.model = self.init_model_ADAF()
        elif key == 'LPA':
            self.model = self.init_model_LPA()            
        else:
            pass
    def init_model_VAN(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(2))
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)
        model.add(scaling_layer)
        for _ in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal'))    
        model.add(tf.keras.layers.Dense(1))
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
                activation='tanh',
                )(hiddens)
        hiddens = ADAF(3,3)(hiddens)
        hiddens = tf.math.tanh(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model
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
                kernel_initializer='glorot_normal', 
                activation='tanh',
                )(hiddens)
        hiddens = LPA(3,30)(hiddens)
        #hiddens = tf.math.tanh(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        model = tf.keras.Model(X_in, prediction)
        return model

class Solver_PINN():
    def __init__(self, pinn, properties, loss_dict, N_0=150, N_b=150, N_r=2500, show=False, DTYPE='float32'):
        self.ref_pinn = None
        self.loss_element = None                
        self.ref_index = 0
        self.lbfgs_step = 0        
        self.loss_history = []
        self.accuracy_history = []
        self.cur_pinn = pinn
        self.properties = properties
        self.loss_dict = loss_dict
        self.show = show
        self.DTYPE = DTYPE
        self.N_0 = N_0
        self.N_b = N_b
        self.N_r = N_r
        self.X_0, self.X_b_0, self.X_b_L, self.X_r = self.data_sampling()        
        self.lr = None
        self.optim = None
        self.build_optimizer()        
        self.call_examset()
        self.u_I = self.fun_u_I(self.X_0)
        self.initial_Y_I = self.fun_u_I(tf.stack([self.x_exam[:,0], self.x_exam[:,0]], axis=1))                
        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)

        self.u_I = self.fun_u_I(self.X_0)
        self.initial_Y_I = self.fun_u_I(tf.stack([self.x_exam[:,0], self.x_exam[:,0]], axis=1))
        
        self.loss = self.compute_loss()
        self.accuracy_update()
        self.loss_history.append(self.loss)
    '''def call_examset(self):
        t_exam_set = np.arange(self.cur_pinn.lb[0],self.cur_pinn.ub[0],(self.cur_pinn.ub[0]-self.cur_pinn.lb[0])/100)
        t_exam = np.ones((100,len(t_exam_set)))
        self.t_exam = np.multiply(t_exam, t_exam_set)
        self.x_exam = np.linspace(self.cur_pinn.lb[1],self.cur_pinn.ub[1],100).reshape(100,1).astype('float32')
        self.X_exam_set=[]
        for i in range(100):
            self.X_exam_set.append(np.concatenate( (self.t_exam[:,i:i+1], self.x_exam), axis=1))
        self.X_exam = np.concatenate( (self.t_exam[:,0:1], self.x_exam), axis=1)'''
    def call_examset(self):
        # 시간과 공간 축을 각각 100개 포인트로 균등 분할
        t_exam_set = np.linspace(self.cur_pinn.lb[0], self.cur_pinn.ub[0], 100)
        x_exam_set = np.linspace(self.cur_pinn.lb[1], self.cur_pinn.ub[1], 100)

        # meshgrid 생성 (t, x) 짝을 모두 만들기 위해 indexing='ij' 사용
        T_exam, X_exam = np.meshgrid(t_exam_set, x_exam_set, indexing='ij')  # shape: (100, 100)

        # 검증용 좌표: (10000, 2)
        self.X_exam = np.column_stack((T_exam.flatten(), X_exam.flatten()))

        # 시간별로 잘라 저장: 100개 리스트, 각 리스트는 shape (100, 2)
        self.X_exam_set = [
            np.column_stack((T_exam[i, :], X_exam[i, :])) for i in range(100)
        ]

        # 각각 시간/공간 벡터도 저장 (optional)
        self.t_exam = t_exam_set.astype('float32')
        self.x_exam = x_exam_set.astype('float32').reshape(-1, 1)               
    def time_stepping(self, num_hidden_layers=2, num_neurons_per_layer=10, key='STD'):
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_%s_lbfgs' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key,self.ref_index))        
        np.savetxt(self.path + 'loss_hist_%s_%s.txt' % (self.cur_pinn.key, self.ref_index), np.array(self.loss_history), delimiter=',')
        np.savetxt(self.path + 'acc_hist_%s_%s.txt' % (self.cur_pinn.key, self.ref_index), np.array(self.accuracy_history), delimiter=',')
        
        self.loss_history = []
        self.ref_index += 1
        del self.ref_pinn
        self.ref_pinn = self.cur_pinn
        self.ref_pinn.model.trainable = False
        del self.cur_pinn        
        lb = tf.constant([(self.properties['tmax']/self.properties['time_marching_constant'])*(self.ref_index), self.properties['xmin']], dtype=self.DTYPE)
        ub = tf.constant([(self.properties['tmax']/self.properties['time_marching_constant'])*(self.ref_index+1), self.properties['xmax']], dtype=self.DTYPE) 
        self.cur_pinn = Build_PINN(lb, ub, num_hidden_layers, num_neurons_per_layer, key)
        self.build_optimizer()
        self.X_0, self.X_b_0, self.X_b_L, self.X_r = self.data_sampling()
        self.call_examset()
    def plot_iteration(self):
        color = cm.Reds(np.linspace(0.1,1,10))
        for i in range(10):
            plt.plot(self.x_exam, self.cur_pinn.model.predict(self.X_exam_set[i]),c=color[len(color)-1-i])
        plt.plot(self.x_exam, self.initial_Y_I, 'b--') 
        plt.plot(self.X_0[:,1], self.fun_u_I(self.X_0), 'k.')                    
    def build_optimizer(self):
        del self.lr
        del self.optim
        self.lr = 1e-2
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr) 
    def call_exact(self, X_0):
        t, x = tf.split(X_0, 2, axis=1)
        return tf.math.exp(-t)*(tf.math.sin(x) + tf.math.sin(2.*x)/2. + tf.math.sin(3.*x)/3. + tf.math.sin(4.*x)/4. + tf.math.sin(8.*x)/8.)
    def fun_u_I(self, X_0):
        if self.ref_pinn:
            return self.ref_pinn.model(X_0)        
        else:
            t, x = tf.split(X_0, 2, axis=1)
            return tf.math.sin(x) + tf.math.sin(2.*x)/2. + tf.math.sin(3.*x)/3. + tf.math.sin(4.*x)/4. + tf.math.sin(8.*x)/8.
    def get_u_I(self, X_0):
        return self.cur_pinn.model(X_0)  
    def fun_r(self, t, x, u, u_t, u_x, u_xx):
        return u_t - u_xx - self.source(t,x)
    def source(self, t, x):
        return tf.math.exp(-t)* (3*tf.math.sin(2.*x)/2. + 8.*tf.math.sin(3.*x)/3. + 15.*tf.math.sin(4.*x)/4. + 63.*tf.math.sin(8.*x)/8.)
    def get_r(self, X_r):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(X_r, 2, axis=1)
            tape.watch(t)
            tape.watch(x)
            u = self.cur_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
            u_x = tape.gradient(u,x)
        u_t = tape.gradient(u,t)
        u_xx = tape.gradient(u_x,x)
        del tape
        return self.fun_r(t, x, u, u_t, u_x, u_xx)
    def data_sampling(self):    
        X_0 = get_X0(self.cur_pinn.lb, self.cur_pinn.ub, self.N_0)
        X_b_0, X_b_L = get_XB(self.cur_pinn.lb, self.cur_pinn.ub, self.N_b)
        X_r = get_Xr(self.cur_pinn.lb, self.cur_pinn.ub, self.N_r)
        return X_0, X_b_0, X_b_L, X_r

    def compute_loss(self):
        # 1. PDE Residual (r)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.X_r)
            r = self.get_r(self.X_r)  # shape: (Nr, 1)

        # 2. Gradient of Residual w.r.t inputs (∇r)
        grad_r = tape.gradient(r, self.X_r)  # shape: (Nr, D) — D = input dim (e.g., 2 for (t,x))
        del tape

        # 3. Loss Terms
        Phi_r  = self.loss_dict['loss_PDE_coeff'] * tf.reduce_mean(tf.square(r))
        Phi_gr = self.loss_dict.get('loss_GRAD_coeff', 1.0) * tf.reduce_mean(tf.square(grad_r))  # gPINN term

        # 4. Initial Condition Loss
        r_I = self.get_u_I(self.X_0) - self.u_I
        R_I = self.loss_dict['loss_IC_coeff'] * tf.reduce_mean(tf.square(r_I))

        # 5. Boundary Condition Losses
        b0 = self.cur_pinn.model(self.X_b_0)
        bL = self.cur_pinn.model(self.X_b_L)
        B0 = self.loss_dict['loss_BC_coeff'][0] * tf.reduce_mean(tf.square(b0))
        BL = self.loss_dict['loss_BC_coeff'][1] * tf.reduce_mean(tf.square(bL))

        # 6. Total Loss
        total_loss = Phi_r + Phi_gr + R_I + B0 + BL
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
            if num_step%50 == 0:
                self.accuracy_update()            
                print('Iter {:05d}: loss = {:10.8e}'.format(num_step, self.loss))                
                if self.show:
                    drawnow(self.plot_iteration)
    def accuracy_update(self):
        prediction = self.cur_pinn.model.predict(self.X_exam)                                    
        exact = self.call_exact(self.X_exam) 
        l1_absolute = np.mean(np.abs(prediction-exact))
        l2_relative = np.linalg.norm(prediction-exact,2)/np.linalg.norm(exact,2)
        print('     l1_absolute_error:   ', l1_absolute)   
        print('     l2_relative_error:   ', l2_relative)
        self.accuracy_element = np.array([l1_absolute, l2_relative])
        self.accuracy_history.append(self.accuracy_element)    
    def callback(self, xr=None):
        self.loss_history.append(self.loss)
        if self.lbfgs_step % 50 == 0:        
            self.accuracy_update()
            if self.show:
                drawnow(self.plot_iteration)
        self.lbfgs_step+=1            
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
    def save_results(self, trial, times, num_hidden_layers=2, num_neurons_per_layer=10):
        self.accuracy_update()
        self.loss_history.append(self.loss)
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_lbfgs_%s' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial))        
        np.savetxt('./results/loss_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.loss_history), delimiter=',')
        np.savetxt('./results/acc_hist_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(self.accuracy_history), delimiter=',') 
        np.savetxt('./results/cal_time_%s_%s_%s_%s.txt' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key, trial), np.array(times), delimiter=',') 
    def save_error(self):         
        self.prediction = self.cur_pinn.model.predict(self.X_exam)        
        exact = self.call_exact(self.X_exam)                 
        l1_absolute = np.mean(np.abs(self.prediction-exact))
        l2_relative = np.linalg.norm(self.prediction-exact,2)/np.linalg.norm(exact,2)
        print('l2_absolute_error:   ', l1_absolute)   
        print('l2_relative_error:   ', l2_relative)
        np.savetxt(self.path+'prediction_%s.txt' % self.cur_pinn.key, self.prediction, delimiter=',')
        np.savetxt(self.path+'exact_%s.txt' % self.cur_pinn.key, exact, delimiter=',')
        f = open(self.path2+'Error_%s_%s_%s.txt'% (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key), 'w')
        f.write('l1_absolute_error:  %s\n' % l1_absolute)
        f.write('l2_relative_error:   %s\n' % l2_relative)
        f.close()