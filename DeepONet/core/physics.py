# core/physics.py
import numpy as np
import tensorflow as tf

def kovasznay_lambda(Re):
    return Re / 2.0 - tf.sqrt(0.25 * Re**2 + 4.0 * (np.pi**2))

def kovasznay_solution_tf(Re, xy):
    # Re: [N,1] or scalar
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    Re = tf.cast(Re, xy.dtype)
    lam = kovasznay_lambda(Re)
    exp_lx = tf.exp(lam * x)
    u = 1.0 - exp_lx * tf.cos(2.0 * np.pi * y)
    v = lam / (2.0 * np.pi) * exp_lx * tf.sin(2.0 * np.pi * y)
    p = 0.5 * (1.0 - tf.exp(2.0 * lam * x))
    return tf.concat([u, v, p], axis=1)

def kovasznay_solution_np(Re, X, Y):
    lam = float(Re) / 2.0 - np.sqrt(0.25 * float(Re)**2 + 4.0 * np.pi**2)
    exp_lx = np.exp(lam * X)
    u = 1.0 - exp_lx * np.cos(2.0 * np.pi * Y)
    v = lam / (2.0 * np.pi) * exp_lx * np.sin(2.0 * np.pi * Y)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * X))
    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32)

def ns_residual(model, Re_batch, xy_int):
    """
    Steady incompressible NS:
      x-mom: u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
      y-mom: u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
      cont : u_x + v_y
    """
    Re_batch = tf.cast(Re_batch, xy_int.dtype)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(xy_int)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(xy_int)
            pred = model([Re_batch, xy_int], training=True)
            u, v, p = tf.split(pred, 3, axis=1)

        du = tape1.batch_jacobian(u, xy_int)  # [N,1,2]
        dv = tape1.batch_jacobian(v, xy_int)
        dp = tape1.batch_jacobian(p, xy_int)
        del tape1

        u_x = du[:, 0, 0:1]; u_y = du[:, 0, 1:2]
        v_x = dv[:, 0, 0:1]; v_y = dv[:, 0, 1:2]
        p_x = dp[:, 0, 0:1]; p_y = dp[:, 0, 1:2]

        u_xx = tape2.gradient(u_x, xy_int)[:, 0:1]
        u_yy = tape2.gradient(u_y, xy_int)[:, 1:2]
        v_xx = tape2.gradient(v_x, xy_int)[:, 0:1]
        v_yy = tape2.gradient(v_y, xy_int)[:, 1:2]
    del tape2

    invRe = 1.0 / Re_batch
    x_mom = u * u_x + v * u_y + p_x - invRe * (u_xx + u_yy)
    y_mom = u * v_x + v * v_y + p_y - invRe * (v_xx + v_yy)
    cont  = u_x + v_y
    return x_mom, y_mom, cont

def boundary_loss(model, Re_b, xy_b):
    uvp_true = kovasznay_solution_tf(Re_b, xy_b)
    uvp_pred = model([Re_b, xy_b], training=True)
    return tf.reduce_mean(tf.square(uvp_pred - uvp_true))
