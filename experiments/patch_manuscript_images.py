"""Reproduce the two pixel-level corrections to the original figure rasters.

The original Figures 1 and 5 exist only as rasters: LPA_figures.pptx holds the
same PNGs (image1/2/3.png on slide 1) with the text already burned in, and no
vector source (.svg/.ai/.drawio/.eps) exists in the project. Two factual errors
therefore have to be corrected in pixels:

  Figure 1  "Fourier-Legendre Exapansion" -> "Fourier-Legendre Expansion"
            (spelling; Referee 2, Comment 6)
  Figure 5  latent dimension "R^32" -> "R^64"
            (the runs use latent_dim = 64; see results/deeponet_rev/runs.csv)

Nothing else is touched. In particular the label "Fourier Coefficient A_n" is
kept: naming the coefficients of an orthogonal expansion generalized Fourier
coefficients is standard, and Section 2 is titled "... based on Fourier-Legendre
expansion", so changing the figure alone would introduce an inconsistency.

Measured from the source rasters (do not re-derive by eye):
  image1.png (4400x1474): the misspelled word occupies x 3871-4153,
      y 1236-1289; the cap height of 'E' is 40 px (rows 1236-1275), which is
      Times New Roman at 60 px; glyph cores are near-black on white.
  image5.png (3356x1454): the superscript digits occupy x 1423-1451,
      y 701-722 (the R glyph ends at x 1418); digit height 22 px, which is
      Times New Roman Bold at 33 px.

Usage:  python patch_manuscript_images.py [--docx PATH] [--out DIR]
"""
import argparse
import os
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from paths import ROOT, RESULTS
TIMES = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
TIMES_BOLD = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"


def extract(docx, member, dest):
    with zipfile.ZipFile(docx) as z:
        with z.open(member) as src, open(dest, "wb") as out:
            out.write(src.read())
    return dest


def _ink(text, font_path, size):
    f = ImageFont.truetype(font_path, size)
    img = Image.new("L", (1200, 300), 0)
    ImageDraw.Draw(img).text((60, 80), text, fill=255, font=f)
    a = np.asarray(img)
    ys, xs = np.where(a > 40)
    return img, (xs.min(), ys.min(), xs.max(), ys.max())


def fit_cap(cap_ref, font_path, target_cap_h):
    """Font size whose `cap_ref` glyph ink height equals target_cap_h."""
    best, err = None, None
    for size in range(10, 110):
        _, (_, y0, _, y1) = _ink(cap_ref, font_path, size)
        e = abs((y1 - y0 + 1) - target_cap_h)
        if err is None or e < err:
            best, err = size, e
    return best


def paste_text(im, text, font_path, box, blur, cap_ref="E", cap_bottom=None):
    """Erase `box`, then draw `text` with its cap height and cap top matched.

    box        ink bounding box of the text being replaced (x0, y0, x1, y1)
    cap_bottom baseline row of the capital letters; cap height = cap_bottom - y0 + 1
               (defaults to the box bottom, i.e. text without descenders)
    """
    x0, y0, x1, y1 = box
    cap_h = (cap_bottom or y1) - y0 + 1
    size = fit_cap(cap_ref, font_path, cap_h)
    _, (_, capTop, _, _) = _ink(cap_ref, font_path, size)
    img, (tx0, ty0, tx1, ty1) = _ink(text, font_path, size)
    crop = img.crop((tx0, ty0, tx1 + 1, ty1 + 1))
    ImageDraw.Draw(im).rectangle([x0 - 6, y0 - 8, x1 + 8, y1 + 6], fill=(255, 255, 255))
    ink = Image.new("RGB", crop.size, (10, 10, 14))
    im.paste(ink, (x0, y0 - (capTop - ty0)), crop.filter(ImageFilter.GaussianBlur(blur)))
    return size, crop.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", default=f"{ROOT}/manuscript_mlst.docx")
    ap.add_argument("--out", default=f"{RESULTS}/manuscript_revisions/figures")
    ap.add_argument("--work", default="/tmp/lpa_media")
    args = ap.parse_args()
    if not os.path.exists(args.docx):
        raise SystemExit(f"manuscript not found: {args.docx}\n"
                         "This script edits rasters extracted from the submitted "
                         "manuscript and cannot run without it.")
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    # ---- Figure 1: spelling
    src1 = extract(args.docx, "word/media/image1.png", f"{args.work}/image1.png")
    im1 = Image.open(src1).convert("RGB")
    assert im1.size == (4400, 1474), f"unexpected image1 size {im1.size}"
    size, glyphs = paste_text(im1, "Expansion", TIMES,
                              (3871, 1236, 4153, 1289), 0.45, cap_bottom=1275)
    im1.save(f"{args.out}/fig1_corrected.png")
    print(f"fig1_corrected.png: 'Expansion' at Times {size}px, ink {glyphs}")

    # ---- Figure 5: latent dimension
    src5 = extract(args.docx, "word/media/image5.png", f"{args.work}/image5.png")
    im5 = Image.open(src5).convert("RGB")
    assert im5.size == (3356, 1454), f"unexpected image5 size {im5.size}"
    size, glyphs = paste_text(im5, "64", TIMES_BOLD, (1423, 701, 1451, 722), 0.35,
                              cap_ref="6")
    im5.save(f"{args.out}/fig5_original_corrected.png")
    print(f"fig5_original_corrected.png: '64' at Times Bold {size}px, ink {glyphs}")

    # ---- before/after records
    for tag, before, after, box, scale in (
            ("fig1_typo", src1, f"{args.out}/fig1_corrected.png",
             (3380, 1215, 4200, 1310), 2),
            ("fig5_latent", src5, f"{args.out}/fig5_original_corrected.png",
             (1330, 660, 1520, 770), 5)):
        a = Image.open(before).convert("RGB").crop(box)
        b = Image.open(after).convert("RGB").crop(box)
        w, h = a.size
        canvas = Image.new("RGB", (w * scale, (h * scale + 26) * 2), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for i, (img, label, colour) in enumerate(
                ((a, "BEFORE (original manuscript raster)", (150, 0, 0)),
                 (b, "AFTER (corrected)", (0, 110, 0)))):
            y = i * (h * scale + 26)
            d.text((6, y + 6), label, fill=colour)
            canvas.paste(img.resize((w * scale, h * scale), Image.LANCZOS), (0, y + 24))
        canvas.save(f"{args.out}/{tag}_before_after.png")
        print(f"{tag}_before_after.png written")

    # ---- report the extent of the change
    for src, out in ((src1, f"{args.out}/fig1_corrected.png"),
                     (src5, f"{args.out}/fig5_original_corrected.png")):
        A = np.asarray(Image.open(src).convert("RGB")).astype(int)
        B = np.asarray(Image.open(out).convert("RGB")).astype(int)
        diff = np.abs(A - B).sum(axis=2)
        ys, xs = np.where(diff > 20)
        print(f"{os.path.basename(out)}: changed x {xs.min()}-{xs.max()}, "
              f"y {ys.min()}-{ys.max()} ({len(xs)} px, "
              f"{100 * len(xs) / diff.size:.4f}% of the image)")


if __name__ == "__main__":
    main()
