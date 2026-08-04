"""Path resolution that works in both trees.

The public repository and the author's working tree hold the same content under
different names:

    working tree                      repository
    v4/2D Helmholtz v2/               2D_Helmholtz/
    v4/Diffusion-Reaction/            Diffusion-Reaction/
    v4/Kovasznay flow/                Kovasznay_flow/
    DeepONet/                         DeepONet/
    <bench>/results/revision/<exp>/   <bench>/results/runs/<exp>/

Every script imports from here instead of hard-coding a path, so one copy of
each script runs in either tree.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

_BENCH = {
    "helmholtz2d": ("2D_Helmholtz", os.path.join("v4", "2D Helmholtz v2")),
    "diffusion_reaction": ("Diffusion-Reaction", os.path.join("v4", "Diffusion-Reaction")),
    "kovasznay": ("Kovasznay_flow", os.path.join("v4", "Kovasznay flow")),
    "deeponet": ("DeepONet", "DeepONet"),
}


def bench_dir(key):
    """Benchmark directory for `key`, whichever layout is present."""
    for cand in _BENCH[key]:
        p = os.path.join(ROOT, cand)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"no directory for benchmark {key!r} under {ROOT}")


def runs_dir(key, exp=""):
    """Per-run output directory: <bench>/results/{runs|revision}/<exp>."""
    base = os.path.join(bench_dir(key), "results")
    for name in ("runs", "revision"):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return os.path.join(p, exp) if exp else p
    raise FileNotFoundError(f"no runs directory under {base}")


def checkpoints_dir(exp=""):
    base = os.path.join(bench_dir("deeponet"), "checkpoints")
    for name in ("runs", "revision"):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return os.path.join(p, exp) if exp else p
    return os.path.join(base, exp) if exp else base


def manuscript_docx():
    """The submitted manuscript, if it is available (it is not in the repo)."""
    p = os.path.join(ROOT, "manuscript_mlst.docx")
    return p if os.path.exists(p) else None
