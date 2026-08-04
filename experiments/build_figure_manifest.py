"""Build results/figure_manifest.json — one entry per candidate figure.

Placement is proposed on a single criterion: does the figure directly answer a
reviewer comment that no other figure answers? Everything else goes to the
supplementary material. The proposal is advisory; the final selection is made
by the authors.

Usage:  python build_figure_manifest.py
"""
import datetime
import hashlib
import json
import os

from paths import ROOT, RESULTS as RES

# path (relative to results/) -> metadata
FIGURES = [
    # ---------------- replacements for existing manuscript figures
    ("manuscript_revisions/figures/fig1_corrected.png",
     "manual pixel edit of manuscript_mlst.docx word/media/image1.png "
     "(revision/README_fig1_edit.md documents the edit)",
     ["R2-Comment6"], "main:Figure 1", 1,
     "Original schematic with the spelling of 'Expansion' corrected; no other "
     "change. The 'Fourier Coefficient' label is deliberately kept, matching the "
     "Section 2 title."),
    ("manuscript_revisions/figures/fig2_replacement.png",
     "revision/make_paper_figures.py", ["R1-Minor4", "R2-Comment4"],
     "main:Figure 2", 2,
     "Replaces the original: like-for-like single-output baseline, geometric-mean "
     "convergence over seeds, and field panels drawn on the true unit-square "
     "coordinates (the original axes were labelled [-1,1])."),
    ("manuscript_revisions/figures/fig3_replacement.png",
     "revision/make_paper_figures.py", ["R1-Minor4"], "main:Figure 3", 3,
     "Replaces the original: the baseline curve is re-run on the correct domain "
     "x in [-pi, pi] and the profile panels use that domain."),
    ("manuscript_revisions/figures/fig4_replacement.png",
     "revision/make_paper_figures.py", ["R1-Minor4"], "main:Figure 4", 4,
     "Replaces the original, which plotted the absolute L1 error of the "
     "v-velocity under a relative-L2 axis label; regenerated with the correct "
     "metric."),
    ("manuscript_revisions/figures/fig5_replacement.png",
     "revision/make_fig5.py", ["R1-Major5"], "main:Figure 5", 5,
     "Redrawn operator-network schematic with the corrected latent dimension "
     "(R^64) and the channel-wise layer stated explicitly (K = 3, N_p = 16 per "
     "channel). Editable source, unlike the original raster."),
    ("manuscript_revisions/figures/fig5_original_corrected.png",
     "manual pixel edit of word/media/image5.png", ["R1-Major5"],
     "alternative to main:Figure 5", 5,
     "Minimal alternative: the original raster with only the latent-dimension "
     "superscript changed from 32 to 64, for authors who prefer to keep the "
     "original drawing."),
    ("deeponet_rev/fig6_replacement.png",
     "revision/aggregate_deeponet.py", ["R2-Comment7"], "main:Figure 6", 6,
     "Replaces the original: five seeds per cell with interquartile bands, all "
     "four widths and the full Re = 1-199 sweep; shows that the two error "
     "profiles share their shape, which is the referee's point."),

    # ---------------- new figures proposed for the main text
    ("major1_compare/confirm_box_strip.png",
     "revision/aggregate_major1.py", ["R1-Major1", "R2-Comment4"],
     "main:new", None,
     "Per-seed distributions of all seven compared models under one protocol. "
     "This is the only figure that shows the requested comparison against "
     "Fourier features, SIREN and adaptive tanh, including that SIREN attains "
     "lower typical errors."),
    ("low_order/error_vs_P.png",
     "revision/aggregate_task2.py", ["R2-Comment2", "R2-Comment3"],
     "main:new", None,
     "Error versus polynomial order including orders below the differential "
     "order of the PDE - the experiment the referee asked for."),
    ("sensitivity/heatmap_median.png",
     "revision/aggregate_task4.py", ["R1-Major4", "R1-Minor1"],
     "main:new", None,
     "Median error over the order x panel-count grid; the requested sensitivity "
     "analysis and the basis for the practical-selection discussion."),
    ("major1_compare/confirm_error_vs_params.png",
     "revision/aggregate_major1.py", ["R1-Major1"], "main:new", None,
     "Error versus trainable-parameter count for every compared model, which is "
     "where the compactness claim has to be judged."),

    # ---------------- supplementary
    ("low_order/residual_norms_vs_P.png",
     "revision/aggregate_task2.py", ["R2-Comment3", "R1-Major3"],
     "supplementary", None,
     "RMS norms of the residual and its second derivatives versus polynomial "
     "order; supports the mechanism discussion but duplicates the ordering "
     "already visible in the error-versus-order figure."),
    ("dof_fixed/tradeoff.png",
     "revision/aggregate_task3.py", ["R2-Comment3", "R1-Minor1"],
     "supplementary", None,
     "Panel-count reparameterization at a matched nominal budget. Kept out of "
     "the main text because the differences are not significant "
     "(Kruskal-Wallis p = 0.322) and the original p- versus h-refinement "
     "reading is retracted."),
    ("sensitivity/heatmap_mean.png", "revision/aggregate_task4.py",
     ["R1-Major4"], "supplementary", None,
     "Mean view of the same grid; seed-bimodal cells make the mean less "
     "informative than the median."),
    ("sensitivity/heatmap_std.png", "revision/aggregate_task4.py",
     ["R1-Major4"], "supplementary", None,
     "Standard deviation across seeds over the grid; documents where the "
     "bimodality lives."),
    ("sensitivity/lr_sweep.png", "revision/aggregate_task4.py",
     ["R1-Major4"], "supplementary", None,
     "Adam warm-up rate sweep at the default cell only; reported as a "
     "cell-specific observation, not a general recommendation."),
    ("ff_baseline/conv_helmholtz2d.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "Convergence of every model on the Helmholtz benchmark including all "
     "Fourier sigma values; the main-text Figure 2 shows the two headline "
     "curves only."),
    ("ff_baseline/conv_diffusion_reaction.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "As above for the diffusion-reaction benchmark."),
    ("ff_baseline/conv_kovasznay.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "As above for the Kovasznay benchmark."),
    ("ff_baseline/bars_helmholtz2d.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "Final-error bars with seed spread per model and depth, Helmholtz."),
    ("ff_baseline/bars_diffusion_reaction.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "As above for the diffusion-reaction benchmark."),
    ("ff_baseline/bars_kovasznay.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "As above for the Kovasznay benchmark."),
    ("ff_baseline/ff_supp_lr.png", "revision/aggregate_task1.py",
     ["R2-Comment4"], "supplementary", None,
     "Fourier-feature baseline at Adam warm-up rates 1e-2 and 1e-3, showing its "
     "failure is not a warm-up-rate artefact."),

    # ---------------- evidence images (not manuscript figures)
    ("manuscript_revisions/figures/fig1_typo_evidence.png",
     "5x crop of the original image1.png", ["R2-Comment6"], "unused", None,
     "Internal evidence for the spelling error; not for publication."),
    ("manuscript_revisions/figures/fig1_typo_before_after.png",
     "revision/README_fig1_edit.md", ["R2-Comment6"], "unused", None,
     "Before/after comparison of the Figure 1 edit; internal record."),
    ("manuscript_revisions/figures/fig2_axis_mislabel_evidence.png",
     "3x crop of the original image2.png", ["R1-Minor4"], "unused", None,
     "Internal evidence that the original Figure 2 field axes were labelled "
     "[-1,1]; not for publication."),
    ("manuscript_revisions/figures/fig5_latent_before_after.png",
     "revision/build_figure_manifest.py record", ["R1-Major5"], "unused", None,
     "Before/after comparison of the Figure 5 latent-dimension edit; internal "
     "record."),
]

STYLE = {
    "font": "Times New Roman, mathtext.stix, font.size 14",
    "dpi": 400,
    "ticks": "4-direction inward",
    "legend": "frameless",
    "savefig": "bbox_inches='tight'",
    "metrics": "relative L2, absolute L1",
    "panel_labels": "(a)(b)(c)... inside the axes, serif upright; upper right on "
                    "line plots, upper left in white on field images "
                    "(revision/plot_style.py:panel_label)",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries, missing = [], []
    for rel, script, refs, placement, replaces, why in FIGURES:
        p = os.path.join(RES, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        st = os.stat(p)
        entries.append({
            "path": f"results/{rel}",
            "sha256": sha256(p),
            "bytes": st.st_size,
            "generated_at": datetime.datetime.fromtimestamp(st.st_mtime)
                            .astimezone().isoformat(timespec="seconds"),
            "generator": script,
            "reviewer_ref": refs,
            "proposed_placement": placement,
            "replaces_original_figure": replaces,
            "rationale": why,
        })
    payload = {
        "meta": {
            "n_figures": len(entries),
            "style": STYLE,
            "placement_rule": (
                "A figure is proposed for the main text only if it answers a "
                "reviewer comment that no other figure answers. Replacements "
                "inherit the slot of the figure they replace. Everything else is "
                "supplementary. 'unused' marks internal evidence images. The "
                "final selection is the authors' decision."
            ),
            "main_text_new_figures": [e["path"] for e in entries
                                      if e["proposed_placement"] == "main:new"],
            "missing": missing,
        },
        "figures": entries,
    }
    out = f"{RES}/figure_manifest.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"wrote {out}: {len(entries)} figures"
          + (f", MISSING {missing}" if missing else ""))
    for e in entries:
        print(f"  {e['proposed_placement']:<28} {e['path']}")


if __name__ == "__main__":
    main()
