# core/utils.py
import numpy as np
import matplotlib.pyplot as plt
from .physics import kovasznay_solution_np

def rel_l2(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

def compute_errors_on_grid(Re, Xg, Yg, pred_flat):
    Nx, Ny = Xg.shape
    u_pred = pred_flat[:, 0].reshape(Nx, Ny)
    v_pred = pred_flat[:, 1].reshape(Nx, Ny)
    p_pred = pred_flat[:, 2].reshape(Nx, Ny)

    u_true, v_true, p_true = kovasznay_solution_np(Re, Xg, Yg)

    l1_u = np.mean(np.abs(u_pred - u_true))
    l1_v = np.mean(np.abs(v_pred - v_true))
    l1_p = np.mean(np.abs(p_pred - p_true))

    l2_u = rel_l2(u_pred, u_true)
    l2_v = rel_l2(v_pred, v_true)
    l2_p = rel_l2(p_pred, p_true)

    exact_flat = np.stack([u_true.ravel(), v_true.ravel(), p_true.ravel()], axis=1)
    return (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), exact_flat

def plot_exact_pred_error(Re, domain, Xg, Yg, pred_flat, cmap_main="plasma", cmap_error="RdBu_r"):
    xmin, xmax, ymin, ymax = domain
    Nx, Ny = Xg.shape

    u_pred = pred_flat[:, 0].reshape(Nx, Ny)
    v_pred = pred_flat[:, 1].reshape(Nx, Ny)
    p_pred = pred_flat[:, 2].reshape(Nx, Ny)

    u_true, v_true, p_true = kovasznay_solution_np(Re, Xg, Yg)

    # 요청하신 fixed colormap range (필요시 여기만 조정)
    u_vmin, u_vmax = -1.0, 1.0
    v_vmin, v_vmax = -0.2, 0.2
    p_vmin, p_vmax = -0.8, 0.4

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    # u
    im = axes[0,0].imshow(u_true.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_main, vmin=u_vmin, vmax=u_vmax)
    axes[0,0].set_title(r"$u_{\mathrm{exact}}$"); plt.colorbar(im, ax=axes[0,0])
    im = axes[0,1].imshow(u_pred.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_main, vmin=u_vmin, vmax=u_vmax)
    axes[0,1].set_title(r"$u_{\mathrm{DeepONet}}$"); plt.colorbar(im, ax=axes[0,1])
    im = axes[0,2].imshow((u_pred-u_true).T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_error)
    axes[0,2].set_title(r"$u_{\mathrm{error}}$"); plt.colorbar(im, ax=axes[0,2])

    # v
    im = axes[1,0].imshow(v_true.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_main, vmin=v_vmin, vmax=v_vmax)
    axes[1,0].set_title(r"$v_{\mathrm{exact}}$"); plt.colorbar(im, ax=axes[1,0])
    im = axes[1,1].imshow(v_pred.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_main, vmin=v_vmin, vmax=v_vmax)
    axes[1,1].set_title(r"$v_{\mathrm{DeepONet}}$"); plt.colorbar(im, ax=axes[1,1])
    im = axes[1,2].imshow((v_pred-v_true).T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_error)
    axes[1,2].set_title(r"$v_{\mathrm{error}}$"); plt.colorbar(im, ax=axes[1,2])

    # p
    im = axes[2,0].imshow(p_true.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_main, vmin=p_vmin, vmax=p_vmax)
    axes[2,0].set_title(r"$p_{\mathrm{exact}}$"); plt.colorbar(im, ax=axes[2,0])
    im = axes[2,1].imshow(p_pred.T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_main, vmin=p_vmin, vmax=p_vmax)
    axes[2,1].set_title(r"$p_{\mathrm{DeepONet}}$"); plt.colorbar(im, ax=axes[2,1])
    im = axes[2,2].imshow((p_pred-p_true).T, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
                          cmap=cmap_error)
    axes[2,2].set_title(r"$p_{\mathrm{error}}$"); plt.colorbar(im, ax=axes[2,2])

    plt.suptitle(f"DeepONet-PINN (Re={Re})")
    plt.tight_layout()
    plt.show()
