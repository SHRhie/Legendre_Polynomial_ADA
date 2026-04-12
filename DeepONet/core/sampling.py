# core/sampling.py
import numpy as np
import tensorflow as tf

def sample_interior(domain, N_int, dtype=tf.float32):
    xmin, xmax, ymin, ymax = domain
    x = tf.random.uniform((N_int, 1), xmin, xmax, dtype=dtype)
    y = tf.random.uniform((N_int, 1), ymin, ymax, dtype=dtype)
    return tf.concat([x, y], axis=1)

def sample_boundary(domain, N_b, dtype=tf.float32):
    xmin, xmax, ymin, ymax = domain
    xb = tf.random.uniform((N_b, 1), xmin, xmax, dtype=dtype)
    yb = tf.random.uniform((N_b, 1), ymin, ymax, dtype=dtype)

    x0 = tf.ones((N_b, 1), dtype=dtype) * xmin
    x1 = tf.ones((N_b, 1), dtype=dtype) * xmax
    y0 = tf.ones((N_b, 1), dtype=dtype) * ymin
    y1 = tf.ones((N_b, 1), dtype=dtype) * ymax

    X_b0 = tf.concat([x0, yb], axis=1)
    X_b1 = tf.concat([x1, yb], axis=1)
    Y_b0 = tf.concat([xb, y0], axis=1)
    Y_b1 = tf.concat([xb, y1], axis=1)
    return X_b0, X_b1, Y_b0, Y_b1

def make_eval_grid(domain, Nx=128, Ny=128):
    xmin, xmax, ymin, ymax = domain
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    return X, Y, xy
