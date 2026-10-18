"""D10: crop the frame-250 row (split car) and the legend from the thesis fig:failure-zooms.

Source: docs/writing/thesis/figures/seq08_failure_zooms.png (thesis figure, unchanged pixels).
Output: docs/defense/assets/d10_split_car_frame250.png
Run:    .venv\Scripts\python.exe docs/defense/scripts/make_d10_crop.py
"""
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "docs" / "writing" / "thesis" / "figures" / "seq08_failure_zooms.png"
OUT = REPO / "docs" / "defense" / "assets" / "d10_split_car_frame250.png"
LEGEND = (470, 45, 1210, 85)      # TP / FP / FN / GT footprint key (source pixels)
ROW = (0, 1760, 1676, 2242)       # frame-250 row, both panels

im = Image.open(SRC).convert("RGB")
leg, row = im.crop(LEGEND), im.crop(ROW)
out = Image.new("RGB", (row.width, leg.height + 10 + row.height), "white")
out.paste(leg, ((row.width - leg.width) // 2, 0))
out.paste(row, (0, leg.height + 10))
out.save(OUT)
print(f"wrote {OUT} ({out.width}x{out.height})")
