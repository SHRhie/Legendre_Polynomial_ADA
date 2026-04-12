import os
import re
import glob
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from core.sampling import make_eval_grid
from core.utils import compute_errors_on_grid

# -------------------------
# Plot Style Settings
# -------------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.dpi'] = 400

# -------------------------
# Robust file discovery
# -------------------------
_RE_PAT = re.compile(r"_Re([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.txt$")

def _parse_Re_from_fname(fname: str):
    m = _RE_PAT.search(fname.replace("\\", "/"))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _find_file_recursive(root_dir, prefix, key, Re, tol=1e-9):
    pattern = os.path.join(root_dir, "**", f"{prefix}_{key}_Re*.txt")
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        return None

    best = None
    best_diff = 1e30
    for h in hits:
        r = _parse_Re_from_fname(os.path.basename(h))
        if r is None:
            continue
        diff = abs(r - float(Re))
        if diff < best_diff:
            best_diff = diff
            best = h

    if best is None:
        return None
    if best_diff > tol:
        return None
    return best

def _load_txt(path):
    arr = np.loadtxt(path, delimiter=",").astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

# -------------------------
# Plot Function (1x3 Layout)
# -------------------------
def plot_comparison_1x3(metrics_by_model, outpath=None):
    """
    u, v, p 에러를 1x3 서브플롯으로 그려서 하나의 파일로 저장
    """
    # 데이터가 하나도 없으면 종료
    if not any(metrics_by_model.values()):
        return

    # 1행 3열 그래프 생성 (가로로 길게)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    # 반복문으로 처리할 설정들 (Metric Key, Label Suffix, Subplot Axis)
    plot_targets = [
        ("l2_u", "(u)", axes[0]),
        ("l2_v", "(v)", axes[1]),
        ("l2_p", "(p)", axes[2])
    ]

    for metric_key, suffix, ax in plot_targets:
        # 모델별로 플롯
        for idx, (label, rows) in enumerate(metrics_by_model.items()):
            if not rows: continue
            
            # Re 순서로 정렬
            rows = sorted(rows, key=lambda d: d["Re"])
            Re_list = [d["Re"] for d in rows]
            val_list = [d[metric_key] for d in rows]
    
            
            ax.plot(Re_list, val_list, label=label, 
                    linestyle='-', lw=2)

        # 축 설정
        ax.set_yscale("log")
        ax.set_xlabel("$Re$")
        ax.set_ylabel(f"Relative $L^2$ Error ${suffix}$")
        ax.tick_params(axis='both', which='major', direction='in', labelsize=14, top=True, right=True)
        ax.tick_params(axis='both', which='minor', labelsize=14, left=False, right=False)

        # 범위 설정 (필요하면 주석 해제하여 고정)
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_ylim(5e-3, 2.0)
        
        ax.grid(False)
        
        # 범례 (첫 번째 그래프에만 넣거나, 모두 넣거나 선택)
        # 여기서는 모두 넣되 작게 설정
        ax.legend(fontsize=10, frameon=False, loc='best')

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"[SAVE PLOT] {outpath}")
    
    plt.close(fig)

# -------------------------
# Arch discovery / config loading
# -------------------------
def _discover_arch_list(results_root):
    arch_pat = re.compile(r"^\d+_\d+$")
    arch_list = []
    if not os.path.isdir(results_root):
        return arch_list
    for name in os.listdir(results_root):
        if arch_pat.match(name) and os.path.isdir(os.path.join(results_root, name)):
            a, b = name.split("_")
            arch_list.append((int(a), int(b)))
    arch_list.sort(key=lambda t: (t[0], t[1]))
    return arch_list

def _load_config_module(name):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="./results")
    ap.add_argument("--save_dir", type=str, default="./results_compare")
    ap.add_argument("--config_module", type=str, default="config")
    ap.add_argument("--Re_list", type=str, default="")
    ap.add_argument("--models", type=str,
                    default="SUP=DeepONet_SUP_A_VAN,PID=DeepONet_PINN_B_VAN,LPA=DeepONet_PINN_B_LPA")
    ap.add_argument("--Nx", type=int, default=100)
    ap.add_argument("--Ny", type=int, default=100)
    ap.add_argument("--domain", type=str, default="")
    args = ap.parse_args()

    C = _load_config_module(args.config_module)

    # Domain
    if args.domain.strip():
        domain_vals = [float(s.strip()) for s in args.domain.split(",")]
        domain = tuple(domain_vals)
    else:
        domain = getattr(C, "DOMAIN", (0.0, 1.0, -0.5, 1.5)) if C else (0.0, 1.0, -0.5, 1.5)

    # Re_list
    if args.Re_list.strip():
        Re_list = [float(s.strip()) for s in args.Re_list.split(",") if s.strip()]
    else:
        Re_list = list(getattr(C, "RE_TEST_LIST", [])) if C else []
        if len(Re_list) == 0:
            Re_list = [40.0]

    # Models map
    pairs = [p.strip() for p in args.models.split(",") if p.strip()]
    model_map = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--models formatting error: {p}")
        label, key = p.split("=", 1)
        model_map[label.strip()] = key.strip()

    # Arch list
    arch_list = []
    if C is not None and hasattr(C, "ARCH_LIST"):
        arch_list = [(int(a), int(b)) for (a, b) in getattr(C, "ARCH_LIST")]
    if not arch_list:
        arch_list = _discover_arch_list(args.results_root)

    if not arch_list:
        print("[WARN] No architecture folders found.")

    # Grid generation
    Xg, Yg, xy_grid = make_eval_grid(domain, Nx=args.Nx, Ny=args.Ny)

    # Main Loop per Architecture
    for (L, W) in arch_list:
        arch_dir = os.path.join(args.results_root, f"{L}_{W}")
        if not os.path.isdir(arch_dir):
            continue

        print(f"\n===== Processing ARCH: {L}_{W} =====")
        metrics_by_model = {label: [] for label in model_map.keys()}
        save_dir_arch = os.path.join(args.save_dir, f"{L}_{W}")
        os.makedirs(save_dir_arch, exist_ok=True)

        # Collect Data
        for Re in Re_list:
            for label, key in model_map.items():
                model_root = os.path.join(arch_dir, key)
                pred_path = _find_file_recursive(model_root, "prediction", key, Re, tol=1e-6)
                
                if pred_path is None:
                    continue

                pred_flat = _load_txt(pred_path)
                (l1_u, l1_v, l1_p, l2_u, l2_v, l2_p), _ = compute_errors_on_grid(Re, Xg, Yg, pred_flat)
                
                metrics_by_model[label].append(
                    {"Re": float(Re), 
                     "l1_u": l1_u, "l1_v": l1_v, "l1_p": l1_p,
                     "l2_u": l2_u, "l2_v": l2_v, "l2_p": l2_p}
                )
                print(f"  [LOAD] {label} Re={Re}")

        # 1. Plot Combined (1x3)
        out_png = os.path.join(save_dir_arch, "relL2_combined_1x3.png")
        plot_comparison_1x3(metrics_by_model, outpath=out_png)

        # 2. Save CSV
        csv_path = os.path.join(save_dir_arch, "metrics.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("label,Re,l1_u,l1_v,l1_p,l2_u,l2_v,l2_p\n")
            for label, rows in metrics_by_model.items():
                rows = sorted(rows, key=lambda d: d["Re"])
                for d in rows:
                    f.write(f"{label},{d['Re']},{d['l1_u']},{d['l1_v']},{d['l1_p']},{d['l2_u']},{d['l2_v']},{d['l2_p']}\n")
        print(f"[SAVE CSV] {csv_path}")

if __name__ == "__main__":
    main()