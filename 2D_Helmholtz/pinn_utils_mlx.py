import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
import scipy.optimize
import matplotlib.pyplot as plt
import sympy as sp
import os
import sys
import platform
import multiprocessing as mp
from importlib import metadata as importlib_metadata


K0 = 2 * np.pi * 2

DEFAULT_LBFGSB_OPTIONS = {
    'maxiter': 40000,
    'maxfun': 50000,
    'maxcor': 50,
    'maxls': 50,
    'ftol': np.finfo(float).eps,
    'gtol': np.finfo(float).eps,
    'factr': np.finfo(float).eps,
    'iprint': 50,
}


def get_lbfgsb_options(**overrides):
    options = dict(DEFAULT_LBFGSB_OPTIONS)
    options.update(overrides)
    return options


def print_lbfgsb_result(result, label='L-BFGS-B'):
    final_loss = getattr(result, 'fun', np.nan)
    final_loss_str = f'{final_loss:.6e}' if np.isfinite(final_loss) else str(final_loss)
    print(f'\n{label} summary:')
    print(f'  success: {getattr(result, "success", False)}')
    print(f'  status: {getattr(result, "status", "n/a")}')
    print(f'  message: {getattr(result, "message", "")}')
    print(f'  nit: {getattr(result, "nit", "n/a")}')
    print(f'  nfev: {getattr(result, "nfev", "n/a")}')
    print(f'  final_loss: {final_loss_str}')


def set_global_seed(seed):
    np.random.seed(seed)
    if hasattr(mx.random, 'seed'):
        mx.random.seed(seed)


def _get_distribution_version(dist_name):
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return 'unknown'


def print_runtime_info(seed=None, extra_config=None):
    print('\nRuntime info (MLX):')
    print(f'  python: {sys.version.split()[0]}')
    print(f'  platform: {platform.platform()}')
    print(f'  machine: {platform.machine()}')
    print(f'  multiprocessing_start_method: {mp.get_start_method()}')
    print(f'  numpy: {np.__version__}')
    print(f'  scipy: {scipy.__version__}')
    print(f'  mlx: {_get_distribution_version("mlx")}')
    if seed is not None:
        print(f'  seed: {seed}')
    if extra_config:
        for key in sorted(extra_config):
            print(f'  {key}: {extra_config[key]}')


def init_linear_layers_like_tf(model):
    """Match TF Dense initialization used in the TF baseline."""
    glorot_normal = nn.init.glorot_normal()
    glorot_uniform = nn.init.glorot_uniform()
    flat_params = tree_flatten(model.parameters())
    reinitialized = []

    for key, value in flat_params:
        if key.startswith('layers.') and key.endswith('.weight'):
            reinitialized.append((key, glorot_normal(value)))
        elif key == 'last_hidden.weight':
            reinitialized.append((key, glorot_normal(value)))
        elif key == 'out_layer.weight':
            # TF final Dense(1) uses the default Glorot uniform initializer.
            reinitialized.append((key, glorot_uniform(value)))
        elif key.endswith('.bias'):
            reinitialized.append((key, mx.zeros(value.shape, dtype=value.dtype)))
        else:
            reinitialized.append((key, value))

    model.update(tree_unflatten(reinitialized))
    mx.eval(model.parameters())


def solution(X_r):
    """Analytic solution (numpy)."""
    x = X_r[:, 0:1]
    y = X_r[:, 1:2]
    return np.sin(K0 * x) * np.sin(K0 * y)


def get_Legendre_coefs(order=0, n_panel=10):
    x = sp.symbols('x')
    P = sp.legendre(order, x)
    P_int = sp.integrate(P, x)
    inds = np.linspace(-1, 1, n_panel + 1)
    coefs = np.array([P_int.subs(x, ind) for ind in inds], dtype='float32')
    coefs = coefs[1:] - coefs[:-1]
    coefs *= (2.0 * order + 1.0) / 2.0
    return coefs


def Leg_Poly(x, order):
    if order == 1:
        return x
    elif order == 2:
        return 0.5 * (3.0 * mx.square(x) - 1.0)
    elif order == 3:
        return 0.5 * (5.0 * mx.power(x, 3) - 3.0 * x)
    elif order == 4:
        return (1.0 / 8.0) * (35.0 * mx.power(x, 4) - 30.0 * mx.square(x) + 3.0)
    elif order == 5:
        return (1.0 / 8.0) * (63.0 * mx.power(x, 5) - 70.0 * mx.power(x, 3) + 15.0 * x)
    elif order == 6:
        return (1.0 / 16.0) * (231.0 * mx.power(x, 6) - 315.0 * mx.power(x, 4) + 105.0 * mx.square(x) - 5.0)


# ─── Custom Layers ───────────────────────────────────────────────

class LPA(nn.Module):
    def __init__(self, order=3, N_p=10):
        super().__init__()
        self.order = order
        self.N_p = N_p
        # Store as numpy so it is NOT an MLX parameter
        self._coefs = np.array(
            [get_Legendre_coefs(i, N_p) for i in range(1, order + 1)], dtype='float32'
        )
        random_normal = nn.init.normal(std=0.05)
        self.W_i = random_normal(mx.zeros((N_p,), dtype=mx.float32))

    def __call__(self, inputs):
        coefs = mx.array(self._coefs)
        Am = coefs @ self.W_i  # (order,)
        sum_ = mx.mean(self.W_i)
        for i in range(self.order):
            sum_ = sum_ + Leg_Poly(inputs, i + 1) * Am[i]
        return sum_


class ADAF(nn.Module):
    def __init__(self, N_p=5, N_m=5, L=1.0):
        super().__init__()
        self.N_p = N_p
        self.N_m = N_m
        self.L = L
        # Store grid as numpy (not a trainable parameter)
        self._x_i = np.linspace(0.0, L, N_p + 1).astype(np.float32)
        random_normal = nn.init.normal(std=0.05)
        self.W_i = random_normal(mx.zeros((N_p,), dtype=mx.float32))
        self.w = random_normal(mx.zeros((), dtype=mx.float32))

    def __call__(self, inputs):
        x_i = mx.array(self._x_i)
        x_1 = x_i[1:]
        x_2 = x_i[:-1]

        # a_0 / 2 * x^2
        a_0 = mx.sum(self.W_i) / self.N_p
        g_x = a_0 / 2.0 * mx.square(inputs)

        for n in range(1, self.N_m + 1):
            factor = self.L / (n * np.pi)
            sum_1 = mx.sin(n * np.pi / self.L * x_1)
            sum_2 = -mx.sin(n * np.pi / self.L * x_2)
            a_n = (2.0 / (n * np.pi)) * mx.sum(self.W_i * (sum_1 + sum_2))
            g_x = g_x + factor * factor * a_n * (1.0 - mx.cos(inputs / factor))

        return self.w * g_x


# ─── Model Architectures ────────────────────────────────────────

class PINN_Vanilla(nn.Module):
    """R (vanilla) model: normalize -> [Dense+tanh]*L -> Dense(1)"""
    def __init__(self, lb, ub, num_hidden_layers=2, num_neurons_per_layer=10, output_dim=1):
        super().__init__()
        self._lb = np.array(lb, dtype=np.float32)
        self._ub = np.array(ub, dtype=np.float32)
        layers = []
        in_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, num_neurons_per_layer))
            in_dim = num_neurons_per_layer
        self.layers = layers
        self.out_layer = nn.Linear(num_neurons_per_layer, output_dim)

    def __call__(self, x):
        lb = mx.array(self._lb)
        ub = mx.array(self._ub)
        h = 2.0 * (x - lb) / (ub - lb) - 1.0
        for layer in self.layers:
            h = mx.tanh(layer(h))
        return self.out_layer(h)


class PINN_ADAF(nn.Module):
    """ADAF model: normalize -> Dense+tanh -> [Dense+tanh]*(L-2) -> Dense -> ADAF -> tanh -> Dense(1)"""
    def __init__(self, lb, ub, num_hidden_layers=2, num_neurons_per_layer=10, output_dim=1):
        super().__init__()
        self._lb = np.array(lb, dtype=np.float32)
        self._ub = np.array(ub, dtype=np.float32)
        # First + middle layers (with tanh activation)
        layers = []
        in_dim = 2
        for _ in range(max(num_hidden_layers - 1, 1)):
            layers.append(nn.Linear(in_dim, num_neurons_per_layer))
            in_dim = num_neurons_per_layer
        self.layers = layers
        # Last hidden layer (no activation — applied after ADAF)
        self.last_hidden = nn.Linear(num_neurons_per_layer, num_neurons_per_layer)
        self.adaf = ADAF(3, 3)
        self.out_layer = nn.Linear(num_neurons_per_layer, output_dim)

    def __call__(self, x):
        lb = mx.array(self._lb)
        ub = mx.array(self._ub)
        h = 2.0 * (x - lb) / (ub - lb) - 1.0
        for layer in self.layers:
            h = mx.tanh(layer(h))
        h = self.last_hidden(h)  # no activation
        h = self.adaf(h)
        h = mx.tanh(h)
        return self.out_layer(h)


class PINN_LPA(nn.Module):
    """LPA model: normalize -> [Dense+tanh]*L -> LPA -> Dense(1)"""
    def __init__(self, lb, ub, num_hidden_layers=2, num_neurons_per_layer=10,
                 lpa_order=6, lpa_panels=30):
        super().__init__()
        self._lb = np.array(lb, dtype=np.float32)
        self._ub = np.array(ub, dtype=np.float32)
        layers = []
        in_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, num_neurons_per_layer))
            in_dim = num_neurons_per_layer
        self.layers = layers
        self.lpa = LPA(order=lpa_order, N_p=lpa_panels)
        self.out_layer = nn.Linear(num_neurons_per_layer, 1)

    def __call__(self, x):
        lb = mx.array(self._lb)
        ub = mx.array(self._ub)
        h = 2.0 * (x - lb) / (ub - lb) - 1.0
        for layer in self.layers:
            h = mx.tanh(layer(h))
        h = self.lpa(h)
        return self.out_layer(h)


# ─── Factory ────────────────────────────────────────────────────

class Build_PINN:
    def __init__(self, lb, ub, properties,
                 num_hidden_layers=2, num_neurons_per_layer=10, key='R',
                 lpa_order=6, lpa_panels=30):
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = list(lb)
        self.ub = list(ub)
        self.key = key
        self.properties = properties
        self.lpa_order = lpa_order
        self.lpa_panels = lpa_panels

        if key == 'R':
            self.model = PINN_Vanilla(lb, ub, num_hidden_layers, num_neurons_per_layer, output_dim=1)
        elif key == 'ADAF':
            self.model = PINN_ADAF(lb, ub, num_hidden_layers, num_neurons_per_layer, output_dim=1)
        elif key == 'LPA':
            self.model = PINN_LPA(
                lb,
                ub,
                num_hidden_layers,
                num_neurons_per_layer,
                lpa_order=lpa_order,
                lpa_panels=lpa_panels,
            )

        # Trigger lazy initialization
        dummy = mx.zeros((1, 2))
        self.model(dummy)
        init_linear_layers_like_tf(self.model)
        mx.eval(self.model.parameters())

        total_params = sum(p.size for _, p in tree_flatten(self.model.trainable_parameters()))
        print(f'Model: {key}, Hidden layers: {num_hidden_layers}, Neurons: {num_neurons_per_layer}')
        if key == 'LPA':
            print(f'LPA config: order={lpa_order}, panels={lpa_panels}')
        print(f'Total trainable parameters: {total_params}')


# ─── Data Generation ────────────────────────────────────────────

def get_XB(lb, ub, N_b):
    x_b = mx.random.uniform(low=lb[0], high=ub[0], shape=(N_b, 1))
    y_b = mx.random.uniform(low=lb[1], high=ub[1], shape=(N_b, 1))
    x_0 = mx.full((N_b, 1), lb[0])
    x_L = mx.full((N_b, 1), ub[0])
    y_0 = mx.full((N_b, 1), lb[1])
    y_L = mx.full((N_b, 1), ub[1])
    X_b_0 = mx.concatenate([x_0, y_b], axis=1)
    X_b_L = mx.concatenate([x_L, y_b], axis=1)
    Y_b_0 = mx.concatenate([x_b, y_0], axis=1)
    Y_b_L = mx.concatenate([x_b, y_L], axis=1)
    return X_b_0, X_b_L, Y_b_0, Y_b_L


def get_Xr(lb, ub, N_r):
    x_r = mx.random.uniform(low=lb[0], high=ub[0], shape=(N_r, 1))
    y_r = mx.random.uniform(low=lb[1], high=ub[1], shape=(N_r, 1))
    return mx.concatenate([x_r, y_r], axis=1)


# ─── PDE Functions ──────────────────────────────────────────────

def source(x, y):
    return K0 * K0 * mx.sin(K0 * x) * mx.sin(K0 * y)


class ExactSolutionModel:
    def __call__(self, X_r):
        x = X_r[:, 0:1]
        y = X_r[:, 1:2]
        return mx.sin(K0 * x) * mx.sin(K0 * y)


def residual_sanity_check(lb, ub, num_points=2048, seed=0):
    rng = np.random.default_rng(seed)
    XY = rng.uniform(low=np.array(lb), high=np.array(ub), size=(num_points, 2)).astype(np.float32)
    XY_mx = mx.array(XY)
    residual = compute_residual(ExactSolutionModel(), XY_mx)
    mx.eval(residual)
    residual_np = np.array(residual)
    mean_abs = float(np.mean(np.abs(residual_np)))
    max_abs = float(np.max(np.abs(residual_np)))
    print(f'Residual sanity check (MLX exact solution): mean_abs={mean_abs:.6e}, max_abs={max_abs:.6e}')
    return {'mean_abs': mean_abs, 'max_abs': max_abs}


def compute_residual(model, X_r):
    """PDE residual: -u_xx - u_yy - k^2*u - f = 0
    Uses sum trick for batched second derivatives via mx.grad.
    """
    def u_sum(x):
        return model(x).sum()

    def ux_sum(x):
        return mx.grad(u_sum)(x)[:, 0].sum()

    def uy_sum(x):
        return mx.grad(u_sum)(x)[:, 1].sum()

    u = model(X_r)
    u_xx = mx.grad(ux_sum)(X_r)[:, 0:1]
    u_yy = mx.grad(uy_sum)(X_r)[:, 1:2]
    x = X_r[:, 0:1]
    y = X_r[:, 1:2]
    f = source(x, y)
    return -u_xx - u_yy - K0 * K0 * u - f


def compute_loss_terms(model, X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r, boundary_weight=150.0):
    r = compute_residual(model, XY_r)
    Phi_r = mx.mean(mx.square(r))

    BX0 = mx.mean(mx.square(model(X_b_0)))
    BXL = mx.mean(mx.square(model(X_b_L)))
    BY0 = mx.mean(mx.square(model(Y_b_0)))
    BYL = mx.mean(mx.square(model(Y_b_L)))

    Phi_b = boundary_weight * (BX0 + BXL + BY0 + BYL)
    total = Phi_r + Phi_b
    return total, Phi_r, Phi_b, BX0, BXL, BY0, BYL


def compute_loss(model, X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r, boundary_weight=150.0):
    total, *_ = compute_loss_terms(
        model, X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r, boundary_weight=boundary_weight
    )
    return total


# ─── Solver ─────────────────────────────────────────────────────

class Solver_PINN:
    def __init__(self, pinn, properties, N_b=200, N_r=10000, DTYPE='float32',
                 adam_learning_rate=1e-3, boundary_weight=150.0):
        self.cur_pinn = pinn
        self.model = pinn.model
        self.properties = properties
        self.DTYPE = DTYPE
        self.N_b = N_b
        self.N_r = N_r
        self.adam_learning_rate = adam_learning_rate
        self.boundary_weight = boundary_weight

        self.loss = 0.0
        self.lbfgs_step = 0
        self.lbfgs_nfev = 0
        self.lbfgs_result = None
        self.loss_history = []
        self.accuracy_history = []

        self.data_sampling()
        self.call_examset()

        self.path = './results/%s_%s/%s/' % (
            pinn.num_hidden_layers, pinn.num_neurons_per_layer, pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)

        self.optimizer = optim.Adam(learning_rate=self.adam_learning_rate)
        self.loss_and_grad_fn = nn.value_and_grad(
            self.model,
            lambda model, X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r: compute_loss(
                model, X_b_0, X_b_L, Y_b_0, Y_b_L, XY_r,
                boundary_weight=self.boundary_weight
            )
        )

    def data_sampling(self):
        lb = self.cur_pinn.lb
        ub = self.cur_pinn.ub
        self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L = get_XB(lb, ub, self.N_b)
        self.XY_r = get_Xr(lb, ub, self.N_r)
        mx.eval(self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r)

    def call_examset(self):
        lb = self.cur_pinn.lb
        ub = self.cur_pinn.ub
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        xx, yy = np.meshgrid(x, y)
        self.XY_test = np.stack((xx.flatten(), yy.flatten()), axis=1).astype(np.float32)

    def train_step(self):
        loss, grads = self.loss_and_grad_fn(
            self.model, self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r
        )
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.loss = loss.item()
        self.loss_history.append(self.loss)

    def loss_breakdown(self, prefix='loss'):
        total, phi_r, phi_b, bx0, bxl, by0, byl = compute_loss_terms(
            self.model, self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r,
            boundary_weight=self.boundary_weight
        )
        mx.eval(total, phi_r, phi_b, bx0, bxl, by0, byl)
        print(
            f'{prefix}: total={float(total.item()):.6e}, '
            f'pde={float(phi_r.item()):.6e}, bc={float(phi_b.item()):.6e}, '
            f'bx0={float(bx0.item()):.6e}, bxl={float(bxl.item()):.6e}, '
            f'by0={float(by0.item()):.6e}, byl={float(byl.item()):.6e}'
        )

    def train_adam(self, N=5000, log_every=200, resample_every=100):
        for step in range(N):
            if step > 0 and resample_every is not None and step % resample_every == 0:
                self.data_sampling()
            self.train_step()
            if step % log_every == 0:
                self.loss_breakdown(prefix=f'Adam step {step}')
                self.accuracy_update()

    def accuracy_update(self):
        XY_mx = mx.array(self.XY_test)
        prediction = np.array(self.model(XY_mx))
        exact = solution(self.XY_test)
        l1_absolute = np.mean(np.abs(prediction - exact))
        l2_relative = np.linalg.norm(prediction - exact, 2) / np.linalg.norm(exact, 2)
        print(f'     l1_absolute_error: {l1_absolute:.6e}')
        print(f'     l2_relative_error: {l2_relative:.6e}')
        self.accuracy_element = np.array([l1_absolute, l2_relative])
        self.accuracy_history.append(self.accuracy_element)

    def callback(self, xr=None):
        self.loss_history.append(self.loss)
        if self.lbfgs_step % 50 == 0:
            self.accuracy_update()
        self.lbfgs_step += 1

    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        options = kwargs.pop('options', None)
        if options is None:
            options = get_lbfgsb_options()
        else:
            options = get_lbfgsb_options(**options)

        self.lbfgs_step = 0
        self.lbfgs_nfev = 0

        # Template for flatten / unflatten
        param_template = tree_flatten(self.model.trainable_parameters())

        # Initial flat vector
        x0 = np.concatenate(
            [np.array(v).flatten() for _, v in param_template]
        ).astype(np.float64)
        best_state = {'flat': np.array(x0, copy=True), 'loss': np.inf}

        def set_weights(flat_np):
            nonlocal param_template
            idx = 0
            new_flat = []
            for key, ref in param_template:
                size = ref.size
                new_val = mx.array(
                    flat_np[idx:idx + size].reshape(ref.shape).astype(np.float32)
                )
                new_flat.append((key, new_val))
                idx += size
            self.model.update(tree_unflatten(new_flat))
            mx.eval(self.model.parameters())
            param_template = tree_flatten(self.model.trainable_parameters())

        def get_loss_and_grad(w):
            self.lbfgs_nfev += 1
            set_weights(w)
            loss, grads = self.loss_and_grad_fn(
                self.model, self.X_b_0, self.X_b_L, self.Y_b_0, self.Y_b_L, self.XY_r
            )
            mx.eval(loss, grads)
            loss_val = float(loss.item())
            grad_flat = tree_flatten(grads)
            grad_np = np.concatenate(
                [np.array(v).flatten() for _, v in grad_flat]
            ).astype(np.float64)

            if not np.isfinite(loss_val):
                raise FloatingPointError(
                    f'Encountered non-finite loss during L-BFGS-B at evaluation {self.lbfgs_nfev}.'
                )
            if not np.all(np.isfinite(grad_np)):
                raise FloatingPointError(
                    f'Encountered non-finite gradient during L-BFGS-B at evaluation {self.lbfgs_nfev}.'
                )

            if loss_val < best_state['loss']:
                best_state['loss'] = loss_val
                best_state['flat'] = np.array(w, copy=True)

            self.loss = loss_val
            return np.float64(loss_val), grad_np

        try:
            if method == 'L-BFGS-B':
                x_opt, f_opt, info = scipy.optimize.fmin_l_bfgs_b(
                    func=get_loss_and_grad,
                    x0=x0,
                    callback=self.callback,
                    maxiter=options.get('maxiter', 15000),
                    maxfun=options.get('maxfun', 15000),
                    m=options.get('maxcor', 10),
                    maxls=options.get('maxls', 20),
                    factr=options.get('factr', 1e7),
                    pgtol=options.get('gtol', 1e-5),
                    iprint=options.get('iprint', -1),
                )
                task = info.get('task', '')
                if isinstance(task, bytes):
                    task = task.decode()
                result = scipy.optimize.OptimizeResult(
                    success=info.get('warnflag', 1) == 0,
                    status=info.get('warnflag', 1),
                    message=task,
                    nit=info.get('nit', self.lbfgs_step),
                    nfev=info.get('funcalls', self.lbfgs_nfev),
                    fun=np.float64(f_opt),
                    x=np.array(x_opt, copy=True),
                )
            else:
                result = scipy.optimize.minimize(
                    fun=get_loss_and_grad,
                    x0=x0,
                    jac=True,
                    callback=self.callback,
                    method=method,
                    options=options,
                    **kwargs
                )
        except FloatingPointError as exc:
            result = scipy.optimize.OptimizeResult(
                success=False,
                status=3,
                message=str(exc),
                nit=self.lbfgs_step,
                nfev=self.lbfgs_nfev,
                fun=np.float64(best_state['loss']) if np.isfinite(best_state['loss']) else np.float64(np.nan),
                x=np.array(best_state['flat'], copy=True),
            )

        if np.isfinite(best_state['loss']):
            set_weights(best_state['flat'])
            self.loss = float(best_state['loss'])
            if not getattr(result, 'success', False):
                result.message = f'{result.message} Restored best finite iterate.'
            result.fun = np.float64(best_state['loss'])
            result.x = np.array(best_state['flat'], copy=True)

        self.lbfgs_result = result
        return result

    def save_results(self, trial, times):
        self.accuracy_update()
        self.loss_history.append(self.loss)
        ckpt_dir = './checkpoints/%s_%s/%s/' % (
            self.cur_pinn.num_hidden_layers,
            self.cur_pinn.num_neurons_per_layer,
            self.cur_pinn.key
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_weights(ckpt_dir + 'ckpt_lbfgs_%s.npz' % trial)
        np.savetxt('./results/loss_hist_%s_%s_%s_%s.txt' % (
            self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer,
            self.cur_pinn.key, trial
        ), np.array(self.loss_history), delimiter=',')
        np.savetxt('./results/acc_hist_%s_%s_%s_%s.txt' % (
            self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer,
            self.cur_pinn.key, trial
        ), np.array(self.accuracy_history), delimiter=',')
        np.savetxt('./results/cal_time_%s_%s_%s_%s.txt' % (
            self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer,
            self.cur_pinn.key, trial
        ), np.array(times), delimiter=',')

    def save_error(self):
        XY_mx = mx.array(self.XY_test)
        self.prediction = np.array(self.model(XY_mx))
        self.exact = solution(self.XY_test)
        l1_absolute = np.mean(np.abs(self.prediction - self.exact))
        l2_relative = np.linalg.norm(self.prediction - self.exact, 2) / np.linalg.norm(self.exact, 2)
        print('l1_absolute_error:   ', l1_absolute)
        print('l2_relative_error:   ', l2_relative)
        np.savetxt(self.path + 'prediction_%s.txt' % self.cur_pinn.key,
                   self.prediction, delimiter=',')
        np.savetxt(self.path + 'exact_%s.txt' % self.cur_pinn.key,
                   self.exact, delimiter=',')
        f = open(self.path2 + 'Error_%s_%s_%s.txt' % (
            self.cur_pinn.num_hidden_layers,
            self.cur_pinn.num_neurons_per_layer,
            self.cur_pinn.key
        ), 'w')
        f.write('l1_absolute_error:  %s\n' % l1_absolute)
        f.write('l2_relative_error:   %s\n' % l2_relative)
        f.close()
