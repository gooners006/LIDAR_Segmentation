"""Crop one figure (without its caption) out of the submitted thesis PDF as a high-res PNG.

Finds the page whose caption block starts with "Figure <num>:", takes the region between the
last full-width body paragraph above the caption and the caption itself, tightens it to the
drawings and text inside, and renders it. Vector TikZ figures come out sharp at any zoom.

Usage: .venv/Scripts/python.exe docs/defense/scripts/crop_thesis_figure.py 3.4 geo_filter
       -> docs/defense/assets/thesis_geo_filter.png
"""
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[3]
PDF = ROOT / "docs/writing/thesis/main.pdf"
num, name = sys.argv[1], sys.argv[2]
OUT = ROOT / f"docs/defense/assets/thesis_{name}.png"
ZOOM, PAD = 4, 4  # 4x = 288 dpi; PAD in PDF points

doc = pymupdf.open(PDF)
hits = [(i, b) for i, page in enumerate(doc) for b in page.get_text("blocks")
        if b[4].startswith(f"Figure {num}:")]
assert len(hits) == 1, f"caption 'Figure {num}:' found {len(hits)} times"
pno, cap = hits[0]
page = doc[pno]
cap_top = cap[1]

# top of the figure = bottom of the last full-width body paragraph (or running header) above the caption
blocks = [b for b in page.get_text("blocks") if b[3] <= cap_top]
wide = [b for b in blocks if b[2] - b[0] > 0.6 * page.rect.width or b[1] < 50]
top = max((b[3] for b in wide), default=page.rect.y0)

region = pymupdf.Rect()
for b in blocks:
    if b[1] >= top:
        region |= pymupdf.Rect(b[:4])
for g in page.get_drawings():
    r = g["rect"]
    if r.y0 >= top - 1 and r.y1 <= cap_top + 1:
        region |= r
assert not region.is_empty, "nothing found between paragraph and caption"
clip = pymupdf.Rect(region.x0 - PAD, region.y0 - PAD, region.x1 + PAD, region.y1 + PAD)
page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=clip, alpha=False).save(OUT)
print(f"page {pno + 1}: clip {[round(v) for v in clip]} -> {OUT}")
