"""Paired coordinate-PINN baselines for Referee 1, Major Comment 1.

This runner is intentionally separate from ``run_experiment.py``.  It keeps the
Helmholtz PDE, automatic-differentiation residual, loss weights, and training
budget common while varying only the coordinate representation/activation.

Key reproducibility property: model initialization, Fourier-feature sampling,
and collocation-point sampling use independent seeds.  Consequently, every
model with the same ``trial`` sees exactly the same boundary and residual
points even though the architectures consume different random streams.

Examples
--------
python major1_compare_run.py --model TANH --trial 0 --exp major1_pilot
python major1_compare_run.py --model SIREN --omega0 10 --trial 0 --exp major1_pilot
python major1_compare_run.py --model NLAAF --trial 0 --exp major1_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from time import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import scipy
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from pinn_utils import (  # noqa: E402
    Solver_PINN,
    get_Legendre_coefs,
    print_runtime_info,
    residual_derivative_norms,
    solution,
)


BASE_SEED = 1234
DATA_SEED_OFFSET = 100_000
FEATURE_SEED_OFFSET = 200_000
DEFAULT_ORDER = 6
DEFAULT_PANELS = 30


def count_params(model: tf.keras.Model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_weights))


def seeded_glorot(seed: int) -> tf.keras.initializers.Initializer:
    return tf.keras.initializers.GlorotNormal(seed=seed)


class ExplicitLPA(tf.keras.layers.Layer):
    """The paper's LPA map with an explicit, reproducible weight seed."""

    def __init__(self, order: int, panels: int, seed: int, dtype: str = "float32"):
        super().__init__(dtype=dtype, name="lpa")
        self.order = int(order)
        self.panels = int(panels)
        self.seed = int(seed)
        coefs = [get_Legendre_coefs(i, self.panels) for i in range(1, self.order + 1)]
        self.coefs = tf.constant(np.asarray(coefs, dtype=np.float32), dtype=dtype)

    def build(self, input_shape):
        self.panel_weights = self.add_weight(
            name="panel_weights",
            shape=(self.panels,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=self.seed),
            trainable=True,
            dtype=self.dtype,
        )

    @staticmethod
    def _legendre_all(x: tf.Tensor, order: int) -> list[tf.Tensor]:
        values = [tf.ones_like(x)]
        if order >= 1:
            values.append(x)
        for n in range(2, order + 1):
            n_float = tf.cast(n, x.dtype)
            values.append(
                ((2.0 * n_float - 1.0) * x * values[-1]
                 - (n_float - 1.0) * values[-2]) / n_float
            )
        return values

    def call(self, inputs):
        inputs = tf.cast(inputs, self.dtype)
        amplitudes = tf.tensordot(self.coefs, self.panel_weights, axes=1)
        polys = self._legendre_all(inputs, self.order)
        output = tf.reduce_mean(self.panel_weights)
        for idx in range(self.order):
            output = output + polys[idx + 1] * amplitudes[idx]
        return output


class DirectLegendre(tf.keras.layers.Layer):
    """Direct coefficients for the same global polynomial function family."""

    def __init__(self, order: int, seed: int, dtype: str = "float32"):
        super().__init__(dtype=dtype, name="direct_legendre")
        self.order = int(order)
        self.seed = int(seed)

    def build(self, input_shape):
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(self.order + 1,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=self.seed),
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, inputs):
        inputs = tf.cast(inputs, self.dtype)
        polys = ExplicitLPA._legendre_all(inputs, self.order)
        output = self.coefficients[0]
        for idx in range(self.order):
            output = output + polys[idx + 1] * self.coefficients[idx + 1]
        return output


class FourierFeatures(tf.keras.layers.Layer):
    """Fixed or trainable random Fourier features with an explicit seed."""

    def __init__(
        self,
        features: int,
        sigma: float,
        seed: int,
        trainable_frequencies: bool = False,
        oracle: bool = False,
        dtype: str = "float32",
    ):
        super().__init__(dtype=dtype, name="fourier_features")
        self.features = int(features)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.trainable_frequencies = bool(trainable_frequencies)
        self.oracle = bool(oracle)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        if self.oracle:
            if input_dim != 2 or self.features != 2:
                raise ValueError("Oracle Fourier features require input_dim=2 and features=2")
            # For xi=2*x-1, the exact field is
            # 0.5*(cos(2*pi*(xi_x-xi_y))-cos(2*pi*(xi_x+xi_y))).
            initial = np.asarray([[1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        else:
            rng = np.random.default_rng(self.seed)
            initial = rng.normal(0.0, self.sigma, (input_dim, self.features)).astype(np.float32)
        self.frequencies = self.add_weight(
            name="frequencies",
            shape=initial.shape,
            initializer=tf.keras.initializers.Constant(initial),
            trainable=self.trainable_frequencies,
            dtype=self.dtype,
        )

    def call(self, inputs):
        projection = 2.0 * np.pi * tf.matmul(tf.cast(inputs, self.dtype), self.frequencies)
        return tf.concat([tf.cos(projection), tf.sin(projection)], axis=-1)


class SineDense(tf.keras.layers.Layer):
    """A canonical SIREN layer with the paper's frequency-aware initialization."""

    def __init__(
        self,
        units: int,
        omega0: float,
        first: bool,
        seed: int,
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.units = int(units)
        self.omega0 = float(omega0)
        self.first = bool(first)
        self.seed = int(seed)

    def build(self, input_shape):
        fan_in = int(input_shape[-1])
        kernel_limit = 1.0 / fan_in if self.first else np.sqrt(6.0 / fan_in) / self.omega0
        bias_limit = 1.0 / np.sqrt(fan_in)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(fan_in, self.units),
            initializer=tf.keras.initializers.RandomUniform(
                -kernel_limit, kernel_limit, seed=self.seed
            ),
            trainable=True,
            dtype=self.dtype,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(
                -bias_limit, bias_limit, seed=self.seed + 10_000
            ),
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, inputs):
        return tf.sin(self.omega0 * (tf.matmul(tf.cast(inputs, self.dtype), self.kernel) + self.bias))


class NeuronAdaptiveTanh(tf.keras.layers.Layer):
    """Neuron-wise locally adaptive tanh, tanh(n*a_i*z_i)."""

    def __init__(
        self,
        units: int,
        scale: float,
        initial_slope: float,
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.units = int(units)
        self.scale = float(scale)
        self.initial_slope = float(initial_slope)

    def build(self, input_shape):
        self.slopes = self.add_weight(
            name="slopes",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.initial_slope),
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, inputs):
        return tf.tanh(self.scale * self.slopes * tf.cast(inputs, self.dtype))


@dataclass
class ModelBundle:
    model: tf.keras.Model
    key: str
    num_hidden_layers: int
    num_neurons_per_layer: int
    lb: tf.Tensor
    ub: tf.Tensor
    properties: dict


def scaled_input(x: tf.Tensor, lb: tf.Tensor, ub: tf.Tensor) -> tf.Tensor:
    return 2.0 * (x - lb) / (ub - lb) - 1.0


def dense_tanh_stack(x, hidden_layers: int, width: int, seed: int):
    for layer_idx in range(hidden_layers):
        x = tf.keras.layers.Dense(
            width,
            activation="tanh",
            kernel_initializer=seeded_glorot(seed + layer_idx),
            bias_initializer="zeros",
            name=f"tanh_dense_{layer_idx + 1}",
        )(x)
    return x


def build_model(args, lb: tf.Tensor, ub: tf.Tensor, init_seed: int, feature_seed: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(2,), dtype="float32", name="coordinates")
    x = tf.keras.layers.Lambda(lambda value: scaled_input(value, lb, ub), name="scale_to_unit")(inputs)
    model_key = args.model.upper()

    if model_key == "FF" or model_key == "TFF" or model_key == "FF_ORACLE":
        x = FourierFeatures(
            args.ff_features,
            args.sigma,
            feature_seed,
            trainable_frequencies=(model_key == "TFF"),
            oracle=(model_key == "FF_ORACLE"),
        )(x)
        x = dense_tanh_stack(x, args.nh, args.nn, init_seed)
    elif model_key == "SIREN":
        for layer_idx in range(args.nh):
            omega = args.omega0 if layer_idx == 0 else args.hidden_omega0
            x = SineDense(
                args.nn,
                omega0=omega,
                first=(layer_idx == 0),
                seed=init_seed + layer_idx,
                name=f"sine_dense_{layer_idx + 1}",
            )(x)
    elif model_key == "NLAAF":
        for layer_idx in range(args.nh):
            x = tf.keras.layers.Dense(
                args.nn,
                activation=None,
                kernel_initializer=seeded_glorot(init_seed + layer_idx),
                bias_initializer="zeros",
                name=f"adaptive_dense_{layer_idx + 1}",
            )(x)
            x = NeuronAdaptiveTanh(
                args.nn,
                scale=args.adaptive_scale,
                initial_slope=args.adaptive_init,
                name=f"adaptive_tanh_{layer_idx + 1}",
            )(x)
    else:
        x = dense_tanh_stack(x, args.nh, args.nn, init_seed)
        if model_key == "LPA":
            x = ExplicitLPA(args.order, args.panels, seed=init_seed + 5_000)(x)
        elif model_key == "LPA_MIN":
            x = ExplicitLPA(args.order, args.order + 1, seed=init_seed + 5_000)(x)
        elif model_key == "DIRECT_LEGENDRE":
            x = DirectLegendre(args.order, seed=init_seed + 5_000)(x)
        elif model_key != "TANH":
            raise ValueError(f"Unknown model: {args.model}")

    if model_key == "SIREN":
        final_limit = np.sqrt(6.0 / args.nn) / args.hidden_omega0
        output_init = tf.keras.initializers.RandomUniform(
            -final_limit, final_limit, seed=init_seed + args.nh + 1
        )
    else:
        output_init = seeded_glorot(init_seed + args.nh + 1)
    outputs = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=output_init,
        bias_initializer="zeros",
        name="solution",
    )(x)
    return tf.keras.Model(inputs, outputs, name=model_key.lower())


class PairedSolver(Solver_PINN):
    """Existing AD solver plus optional canonical N-LAAF slope recovery."""

    def __init__(self, *args, slope_recovery_weight: float = 0.0, **kwargs):
        self.slope_recovery_weight = float(slope_recovery_weight)
        super().__init__(*args, **kwargs)

    def slope_recovery(self) -> tf.Tensor:
        layer_means = []
        for layer in self.cur_pinn.model.layers:
            if isinstance(layer, NeuronAdaptiveTanh):
                layer_means.append(tf.reduce_mean(layer.slopes))
        if not layer_means:
            return tf.constant(0.0, dtype=self.DTYPE)
        return 1.0 / tf.reduce_mean(tf.exp(tf.stack(layer_means)))

    def compute_loss(self):
        common_loss = super().compute_loss()
        if self.slope_recovery_weight == 0.0:
            return common_loss
        return common_loss + self.slope_recovery_weight * self.slope_recovery()


def paired_samples(lb: np.ndarray, ub: np.ndarray, n_b: int, n_r: int, seed: int):
    rng = np.random.default_rng(seed)
    x_b = rng.uniform(lb[0], ub[0], size=(n_b, 1)).astype(np.float32)
    y_b = rng.uniform(lb[1], ub[1], size=(n_b, 1)).astype(np.float32)
    x0 = np.full((n_b, 1), lb[0], dtype=np.float32)
    x1 = np.full((n_b, 1), ub[0], dtype=np.float32)
    y0 = np.full((n_b, 1), lb[1], dtype=np.float32)
    y1 = np.full((n_b, 1), ub[1], dtype=np.float32)
    residual = rng.uniform(lb, ub, size=(n_r, 2)).astype(np.float32)
    arrays = (
        np.concatenate([x0, y_b], axis=1),
        np.concatenate([x1, y_b], axis=1),
        np.concatenate([x_b, y0], axis=1),
        np.concatenate([x_b, y1], axis=1),
        residual,
    )
    return tuple(tf.constant(value, dtype=tf.float32) for value in arrays), arrays


def sample_checksum(arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def key_id(args) -> str:
    # NOTE (2026-08-03, phase A/B campaigns): the learning rate is part of the
    # searched configuration, so it is embedded in every key to prevent output
    # collisions between configs that differ only in lr.  The earlier pilot
    # (exp=major1_pilot_t0) used keys without the lr suffix; its files are
    # untouched and it is not mixed with phase A/B data.
    model = args.model.upper()
    if model == "LPA":
        base = f"LPA_P{args.order}_N{args.panels}"
    elif model == "LPA_MIN":
        base = f"LPA_MIN_P{args.order}_N{args.order + 1}"
    elif model == "DIRECT_LEGENDRE":
        base = f"DIRECT_LEGENDRE_P{args.order}"
    elif model in {"FF", "TFF"}:
        base = f"{model}_m{args.ff_features}_s{args.sigma:g}"
    elif model == "FF_ORACLE":
        base = "FF_ORACLE_m2"
    elif model == "SIREN":
        base = f"SIREN_w{args.omega0:g}_hw{args.hidden_omega0:g}"
    elif model == "NLAAF":
        base = f"NLAAF_n{args.adaptive_scale:g}_a{args.adaptive_init:g}_sr{args.slope_recovery:g}"
    else:
        base = model
    return f"{base}_lr{args.lr:g}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["TANH", "LPA", "LPA_MIN", "DIRECT_LEGENDRE", "FF", "TFF", "FF_ORACLE", "SIREN", "NLAAF"],
    )
    parser.add_argument("--nh", type=int, default=3)
    parser.add_argument("--nn", type=int, default=10)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--panels", type=int, default=DEFAULT_PANELS)
    parser.add_argument("--ff-features", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--omega0", type=float, default=10.0)
    parser.add_argument("--hidden-omega0", type=float, default=10.0)
    parser.add_argument("--adaptive-scale", type=float, default=10.0)
    parser.add_argument("--adaptive-init", type=float, default=0.1)
    parser.add_argument("--slope-recovery", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--adam-steps", type=int, default=200)
    parser.add_argument("--lbfgs-maxiter", type=int, default=40000)
    parser.add_argument("--n-b", type=int, default=200)
    parser.add_argument("--n-r", type=int, default=10000)
    parser.add_argument("--acc-every", type=int, default=50)
    parser.add_argument("--resid-points", type=int, default=4096)
    parser.add_argument("--skip-residual-metrics", action="store_true")
    parser.add_argument("--exp", default="major1_compare")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.disable_interactive_logging()

    init_seed = BASE_SEED + args.trial
    data_seed = DATA_SEED_OFFSET + BASE_SEED + args.trial
    feature_seed = FEATURE_SEED_OFFSET + BASE_SEED + args.trial
    tf.keras.utils.set_random_seed(init_seed)
    np.random.seed(init_seed)

    lb_np = np.asarray([0.0, 0.0], dtype=np.float32)
    ub_np = np.asarray([1.0, 1.0], dtype=np.float32)
    lb = tf.constant(lb_np)
    ub = tf.constant(ub_np)
    properties = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}

    model = build_model(args, lb, ub, init_seed, feature_seed)
    bundle = ModelBundle(model, args.model.upper(), args.nh, args.nn, lb, ub, properties)
    samples, sample_arrays = paired_samples(lb_np, ub_np, args.n_b, args.n_r, data_seed)
    checksum = sample_checksum(sample_arrays)

    print_runtime_info(seed=init_seed, extra_config={
        "model": args.model,
        "key_id": key_id(args),
        "init_seed": init_seed,
        "data_seed": data_seed,
        "feature_seed": feature_seed,
        "collocation_sha256": checksum,
        "nh": args.nh,
        "nn": args.nn,
        "n_b": args.n_b,
        "n_r": args.n_r,
        "adam_steps": args.adam_steps,
        "adam_lr": args.lr,
        "lbfgs_maxiter": args.lbfgs_maxiter,
    })
    model.summary()
    n_params = count_params(model)
    print(f"Trainable parameters: {n_params}")

    slope_weight = args.slope_recovery if args.model.upper() == "NLAAF" else 0.0
    solver = PairedSolver(
        bundle,
        properties,
        N_b=args.n_b,
        N_r=args.n_r,
        lr=args.lr,
        slope_recovery_weight=slope_weight,
    )
    solver.X_b_0, solver.X_b_L, solver.Y_b_0, solver.Y_b_L, solver.XY_r = samples
    solver.plot_every = args.acc_every

    start = time()
    solver.train_adam(args.adam_steps)
    time_adam = time() - start

    start = time()
    lbfgs_result = solver.ScipyOptimizer(
        method="L-BFGS-B",
        options={
            "maxiter": args.lbfgs_maxiter,
            "maxfun": 50000,
            "maxcor": 50,
            "maxls": 50,
            "ftol": np.finfo(float).eps,
            "gtol": np.finfo(float).eps,
            "iprint": -1,
        },
    )
    time_lbfgs = time() - start

    solver.accuracy_update()
    prediction = model.predict(solver.XY_test, verbose=0).reshape(-1)
    exact = solution(solver.XY_test).numpy().reshape(-1)
    l1_absolute = float(np.mean(np.abs(prediction - exact)))
    l2_relative = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))
    common_loss = float(Solver_PINN.compute_loss(solver).numpy())
    slope_recovery = float(solver.slope_recovery().numpy())

    residual_metrics = {}
    if not args.skip_residual_metrics:
        residual_metrics = residual_derivative_norms(
            model, lb_np, ub_np, num_points=args.resid_points, seed=777
        )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs", args.exp)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{args.nh}_{args.nn}_{key_id(args)}_{args.trial}"
    np.savetxt(os.path.join(output_dir, f"loss_hist_{stem}.txt"), np.asarray(solver.loss_history), delimiter=",")
    np.savetxt(os.path.join(output_dir, f"acc_hist_{stem}.txt"), np.asarray(solver.accuracy_history), delimiter=",")
    np.savetxt(os.path.join(output_dir, f"cal_time_{stem}.txt"), np.asarray([time_adam, time_lbfgs]), delimiter=",")

    record = {
        "benchmark": "helmholtz2d",
        "experiment": args.exp,
        "model": args.model.upper(),
        "key_id": key_id(args),
        "nh": args.nh,
        "nn": args.nn,
        "trial": args.trial,
        "init_seed": init_seed,
        "data_seed": data_seed,
        "feature_seed": feature_seed,
        "collocation_sha256": checksum,
        "n_params": n_params,
        "order": args.order if args.model.upper() in {"LPA", "LPA_MIN", "DIRECT_LEGENDRE"} else None,
        "panels": args.panels if args.model.upper() == "LPA" else (args.order + 1 if args.model.upper() == "LPA_MIN" else None),
        "ff_features": args.ff_features if args.model.upper() in {"FF", "TFF", "FF_ORACLE"} else None,
        "sigma": args.sigma if args.model.upper() in {"FF", "TFF"} else None,
        "omega0": args.omega0 if args.model.upper() == "SIREN" else None,
        "hidden_omega0": args.hidden_omega0 if args.model.upper() == "SIREN" else None,
        "adaptive_scale": args.adaptive_scale if args.model.upper() == "NLAAF" else None,
        "adaptive_init": args.adaptive_init if args.model.upper() == "NLAAF" else None,
        "slope_recovery_weight": slope_weight,
        "slope_recovery_final": slope_recovery,
        "lr": args.lr,
        "adam_steps": args.adam_steps,
        "lbfgs_maxiter": args.lbfgs_maxiter,
        "n_b": args.n_b,
        "n_r": args.n_r,
        "l1_absolute": l1_absolute,
        "l2_relative": l2_relative,
        "common_pde_boundary_loss": common_loss,
        "optimized_loss": float(solver.loss),
        "time_adam": time_adam,
        "time_lbfgs": time_lbfgs,
        "lbfgs_steps": int(solver.lbfgs_step),
        "lbfgs_success": bool(lbfgs_result.success),
        "lbfgs_status": int(lbfgs_result.status),
        "lbfgs_message": str(lbfgs_result.message),
        "lbfgs_nit": int(getattr(lbfgs_result, "nit", -1)),
        "lbfgs_nfev": int(getattr(lbfgs_result, "nfev", -1)),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
    }
    record.update(residual_metrics)
    output_path = os.path.join(output_dir, f"run_{stem}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print(f"Saved {output_path}")
    print(f"FINAL l2_relative={l2_relative:.6e} l1_absolute={l1_absolute:.6e}")
    print(
        "L-BFGS success=%s status=%s message=%s"
        % (lbfgs_result.success, lbfgs_result.status, lbfgs_result.message)
    )


if __name__ == "__main__":
    main()
