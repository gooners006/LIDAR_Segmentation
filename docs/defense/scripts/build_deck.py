"""Build the defense deck skeleton (Deck v2 in docs/defense/outline.md) as a PPTX.

Layout mirrors the peer FPT deck (docs/defense/reference/LVTN.pptx): I-IV sections, sub-dividers in III,
hierarchical numbering, footer bar with section name + page number, THE END, backups.
Colours are ours (#1F2933 bars), not theirs. Coordinates are written in px on a
1920x1080 canvas (= Canva's canvas); font sizes are in pt on the 1440x810 pt page, which
is exactly how Canva exports, so a size here equals the size Canva shows after import.

Speaker notes carry the talk-path marker: "★ TALK m:ss" or "○ SUPPORT", plus the thesis source.

Usage: .venv/Scripts/python.exe docs/defense/scripts/build_deck.py
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ROOT / "deck_v2.pptx"

FONT = "Be Vietnam Pro"
DARK = RGBColor(0x1F, 0x29, 0x33)     # bars + main text
MUTED = RGBColor(0x6B, 0x72, 0x80)    # placeholder labels
PH_FILL = RGBColor(0xF7, 0xF7, 0xF7)
PH_LINE = RGBColor(0xD1, 0xD5, 0xDB)
CARD_FILL = RGBColor(0xEE, 0xF0, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

X0, X1 = 120, 1800          # content left/right
Y0, Y1 = 220, 960           # content top/bottom
GAP = 40


def px(v):
    return Emu(int(round(v * 9525)))


# ---------------------------------------------------------------- primitives
def rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, px(x), px(y), px(w), px(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def text(slide, x, y, w, h, lines, size, bold=False, italic=False, color=DARK,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, upper=False, space_after=0):
    """lines: str or list of str / (str, dict-overrides)."""
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        s, ov = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(ov.get("space_after", space_after))
        r = p.add_run()
        r.text = s.upper() if upper else s
        f = r.font
        f.name = FONT
        f.size = Pt(ov.get("size", size))
        f.bold = ov.get("bold", bold)
        f.italic = ov.get("italic", italic)
        f.color.rgb = ov.get("color", color)
    return tb


def placeholder(slide, x, y, w, h, label):
    rect(slide, x, y, w, h, PH_FILL, PH_LINE)
    text(slide, x + 20, y, w - 40, h, f"[ {label} ]", 20, italic=True, color=MUTED,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def card(slide, x, y, w, h, head, body):
    rect(slide, x, y, w, h, CARD_FILL)
    text(slide, x + 30, y + 40, w - 60, h - 80,
         [(head, {"bold": True, "size": 32, "space_after": 14}), (body, {"size": 28})],
         28, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def image_fit(slide, path, x, y, w, h):
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    slide.shapes.add_picture(str(path), px(x + (w - dw) / 2), px(y + (h - dh) / 2), px(dw), px(dh))
    return x + (w - dw) / 2, y + (h - dh) / 2, dw, dh


def footer(slide, section, number):
    rect(slide, 0, 1000, 1920, 80, DARK)
    if section:
        text(slide, 40, 1000, 1200, 80, section, 24, bold=True, italic=True, color=WHITE,
             anchor=MSO_ANCHOR.MIDDLE, upper=True)
    text(slide, 1600, 1000, 280, 80, str(number), 24, color=WHITE,
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)


def top_bar(slide):
    rect(slide, 1240, 50, 600, 45, DARK)


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


# ---------------------------------------------------------------- layouts
def split(n, x=X0, w=X1 - X0):
    cw = (w - GAP * (n - 1)) / n
    return [(x + i * (cw + GAP), cw) for i in range(n)]


def lay_full(sl, labels):
    placeholder(sl, X0, Y0, X1 - X0, Y1 - Y0, labels[0])


def lay_two(sl, labels, ratio=0.5):
    w = X1 - X0 - GAP
    wl = w * ratio
    placeholder(sl, X0, Y0, wl, Y1 - Y0, labels[0])
    placeholder(sl, X0 + wl + GAP, Y0, w - wl, Y1 - Y0, labels[1])


def lay_cards_ph(sl, labels):
    for (x, w), lab in zip(split(len(labels)), labels):
        placeholder(sl, x, Y0, w, Y1 - Y0, lab)


def lay_cards_text(sl, items):
    for (x, w), (h, b) in zip(split(len(items)), items):
        card(sl, x, Y0, w, Y1 - Y0, h, b)


def lay_columns(sl, cols, item_size=32):
    for (x, w), (head, items) in zip(split(len(cols)), cols):
        text(sl, x, Y0, w, 60, head, 36, bold=True)
        text(sl, x, Y0 + 80, w, Y1 - Y0 - 80, items, item_size, space_after=18)


def lay_image(sl, name):
    image_fit(sl, ASSETS / name, X0, Y0, X1 - X0, Y1 - Y0)


def lay_image_or_ph(sl, arg):
    """(asset name, placeholder label): the user-drawn export once it exists, else a placeholder."""
    name, label = arg
    if (ASSETS / name).exists():
        lay_image(sl, name)
    else:
        lay_full(sl, [label])


def lay_rows_image_or_ph(sl, items):
    """Stacked full-width rows; each row is the user's draw.io export once it exists, else a
    placeholder. items: [(asset name, placeholder label), ...]"""
    h = (Y1 - Y0 - GAP * (len(items) - 1)) / len(items)
    for i, (name, label) in enumerate(items):
        y = Y0 + i * (h + GAP)
        if (ASSETS / name).exists():
            image_fit(sl, ASSETS / name, X0, y, X1 - X0, h)
        else:
            placeholder(sl, X0, y, X1 - X0, h, label)


def table(slide, x, y, w, rows, col_w, size=22, row_h=64, bold=(), align_left=1):
    """Native table: dark header row, light zebra body. rows[0] = header; bold = {(r, c)};
    the first `align_left` columns are left-aligned, the rest centred."""
    shp = slide.shapes.add_table(len(rows), len(rows[0]), px(x), px(y), px(w), px(row_h * len(rows)))
    tbl = shp.table
    for c, cw in enumerate(col_w):
        tbl.columns[c].width = px(w * cw / sum(col_w))
    for r, row in enumerate(rows):
        tbl.rows[r].height = px(row_h)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK if r == 0 else (CARD_FILL if r % 2 == 0 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = px(12)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c < align_left else PP_ALIGN.CENTER
            run = p.add_run()
            run.text = val
            f = run.font
            f.name, f.size = FONT, Pt(size)
            f.bold = r == 0 or (r, c) in bold
            f.color.rgb = WHITE if r == 0 else DARK
    return y + row_h * len(rows)


def lay_clustering_tables(sl, _):
    """3.6.2: tab:clustering (Mean IoU column left out) + tab:splitrate, numbers verbatim."""
    w = X1 - X0 - GAP
    wl = w * 0.56
    xr = X0 + wl + GAP
    text(sl, X0, Y0, wl, 50, "Clustering method (0.10 m grid, per frame, track filter off)", 24,
         bold=True)
    rows = [["Seq", "Method", "Precision", "Recall", "F1", "Time (ms)"],
            ["00", "HDBSCAN", "0.972", "0.731", "0.835", "306"],
            ["00", "DBSCAN", "0.975", "0.637", "0.771", "54"],
            ["00", "Euclidean", "0.979", "0.636", "0.771", "76"],
            ["08", "HDBSCAN", "0.840", "0.671", "0.746", "320"],
            ["08", "DBSCAN", "0.870", "0.565", "0.685", "57"],
            ["08", "Euclidean", "0.869", "0.576", "0.693", "75"]]
    yb = table(sl, X0, Y0 + 70, wl, rows, [0.7, 1.5, 1.3, 1.1, 1.0, 1.3], align_left=2,
               bold={(1, 1), (1, 3), (1, 4), (4, 1), (4, 3), (4, 4)})
    text(sl, X0, yb + 24, wl, 120,
         "Precision within 0.03; HDBSCAN gains 9–11 recall points for about 6× DBSCAN's clustering time.", 24)
    text(sl, xr, Y0, w - wl, 50, "Clustering grid (seq 08, 34,057 cars)", 24, bold=True)
    rows = [["", "0.05 m", "0.10 m"],
            ["Car in one cluster", "55.1%", "66.0%"],
            ["Car split", "44.8%", "33.7%"],
            ["Car merged", "0.2% (56)", "0.8% (284)"],
            ["Clusters per split car", "4.7", "3.0"]]
    yb = table(sl, xr, Y0 + 70, w - wl, rows, [1.9, 1.1, 1.2], bold={(2, 2), (3, 2)})
    text(sl, xr, yb + 24, w - wl, 160,
         ["Recall 0.710 → 0.732 (+756 true positives),", "F1 0.798 → 0.812."], 24)


def lay_results(sl, _):
    """3.3.1: D9 stage chart (tab:ablation-stage) + headline table (tab:headline, tab:loso), verbatim."""
    w = X1 - X0 - GAP
    wl = w * 0.58
    image_fit(sl, ASSETS / "d9_stage_funnel.png", X0, Y0, wl, Y1 - Y0)
    xr = X0 + wl + GAP
    text(sl, xr, Y0 + 40, w - wl, 50, "Detection results", 24, bold=True)
    rows = [["", "Precision", "Recall", "F1"],
            ["Seq 08", "0.911", "0.732", "0.812"],
            ["All 11 sequences", "0.888", "0.717", "0.793"]]
    yb = table(sl, xr, Y0 + 110, w - wl, rows, [2.4, 1.2, 1.0, 0.9], size=21, row_h=78,
               bold={(1, 0), (1, 1), (1, 2), (1, 3)})
    text(sl, xr, yb + 24, w - wl, 200,
         ["All 11 sequences are cross-validated: each is scored by a classifier trained without it.",
          "Within about 2 points of seq 08 on every metric."], 21, space_after=10)


def lay_geo_filter(sl, _):
    """2.2.3: thesis fig:geo-filter crop + all gates (ch3 §Geometric Filtering list), verbatim."""
    w = X1 - X0 - GAP
    wl = w * 0.54
    image_fit(sl, ASSETS / "thesis_geo_filter.png", X0, Y0 + 60, wl, Y1 - Y0 - 120)
    xr = X0 + wl + GAP
    rows = [["Gate", "Keep the cluster if"],
            ["Points", "≥ 15"],
            ["Box volume", "0.5–50 m³"],
            ["Longest side", "≤ 6.0 m"],
            ["Size floors", "longest ≥ 0.5 m, middle ≥ 0.4 m"],
            ["Centre height", "≤ 1.5 m above the ground plane"],
            ["Vertical span", "≤ 1.8 m"],
            ["Aspect ratio", "longest / shortest ≤ 6.0"]]
    yb = table(sl, xr, Y0, w - wl, rows, [1.0, 1.9], size=21, row_h=62, align_left=2)
    text(sl, xr, yb + 20, w - wl, 120,
         ["After this filter alone: precision 0.199, recall 0.773.",
          "Loose by design: the classifier makes the fine decision."], 21, space_after=8)


def lay_stage_ablation(sl, _):
    """3.6.1: tab:ablation-stage, verbatim (full seq 08, frozen configuration)."""
    rows = [["Configuration", "Precision", "Recall", "F1", "Mean IoU", "TP / FP / FN"],
            ["Geometric only", "0.199", "0.773", "0.316", "0.907", "27,002 / 108,831 / 7,920"],
            ["+ classifier", "0.846", "0.676", "0.751", "0.922", "23,600 / 4,303 / 11,322"],
            ["+ classifier, track filter", "0.911", "0.732", "0.812", "0.913", "25,568 / 2,513 / 9,354"]]
    yb = table(sl, X0, Y0 + 40, X1 - X0, rows, [2.3, 1.1, 1.0, 0.9, 1.1, 2.6], size=24, row_h=80,
               bold={(3, 0), (3, 1), (3, 2), (3, 3)})
    text(sl, X0, yb + 30, X1 - X0, 200,
         ["The classifier is the precision step: false positives 108,831 → 4,303, at a recall cost.",
          "The track filter wins recall back: true positives 23,600 → 25,568.",
          "First two rows are per frame; the last row uses the offline track filter. "
          "Mean IoU is over matched detections only."], 24, space_after=10)


def lay_pretraining(sl, _):
    """3.6.4: ch4 sec:ablation-stagea numbers + tab:cross-domain, verbatim."""
    w = X1 - X0 - GAP
    wl = w * 0.5
    xr = X0 + wl + GAP
    text(sl, X0, Y0, wl, 50, "Classifier level (seq 08 validation clusters)", 24, bold=True)
    rows = [["Training recipe", "Macro-F1"],
            ["Real clusters only, from scratch (final)", "0.9285"],
            ["Synthetic pretraining, then real", "0.9225"]]
    yb = table(sl, X0, Y0 + 70, wl, rows, [2.6, 1.0], size=22, row_h=72, bold={(1, 0), (1, 1)})
    text(sl, X0, yb + 24, wl, 160,
         ["Within run-to-run noise. At pipeline level the two give identical F1 to three decimals."], 22)
    text(sl, xr, Y0, w - wl, 50, "Cross-domain transfer (car-class F1)", 24, bold=True)
    rows = [["Trained on ↓ / tested on →", "Synthetic", "Real"],
            ["Synthetic (Stage A)", "0.999", "0.000"],
            ["Real (Stage B)", "0.000", "0.885"]]
    yb = table(sl, xr, Y0 + 70, w - wl, rows, [2.2, 1.1, 1.0], size=22, row_h=72,
               bold={(1, 2), (2, 1)})
    text(sl, xr, yb + 24, w - wl, 160,
         ["A model from one domain finds no cars in the other: the gap is total and symmetric."], 22)


def lay_literature(sl, _):
    """1.3: three lines of prior work, wording from ch2 §2.2 (bg-deep, bg-classical, bg-completion)."""
    rows = [["Approach", "Representative works", "Strengths", "Remaining challenges"],
            ["End-to-end deep segmentation", "RangeNet++, RandLA-Net, Cylinder3D; panoptic: Panoptic-PolarNet, DS-Net",
             "Benchmark leaders (Cylinder3D 67.8 mIoU on the SemanticKITTI test set)",
             "Need dense per-point (and per-instance) labels; a missed object can't be traced to one step"],
            ["Classical geometric segmentation", "RANSAC ground removal + DBSCAN / HDBSCAN clustering",
             "No training data for segmentation; every stage inspectable",
             "No notion of objectness: clustering splits cars, which costs recall"],
            ["Object-level completion", "PCN, PoinTr, SnowflakeNet, SeedFormer",
             "Learns to map a partial view to a complete shape",
             "Trained on synthetic CAD shapes (domain gap); Chamfer scoring needs a complete true shape that real LiDAR lacks"]]
    table(sl, X0, Y0 + 20, X1 - X0, rows, [1.25, 1.75, 1.6, 2.1], size=21, row_h=125, align_left=4,
          bold={(1, 0), (2, 0), (3, 0)})


def lay_datasets(sl, _):
    """3.1.1: tab:dataset-roles, tab:sensor-spec, tab:dataset, verbatim."""
    w = X1 - X0 - GAP
    wl = w * 0.6
    xr = X0 + wl + GAP
    rows = [["", "SemanticKITTI", "ShapeNetCore v2"],
            ["Data", "Real LiDAR scans", "Synthetic CAD meshes"],
            ["Role", "Detection, mining, evaluation", "Completion network training"],
            ["Volume", "23,201 frames (11 sequences)", "~3,150 car models"],
            ["Annotations used", "Point-wise semantic + instance", "None (synthetic partials)"]]
    table(sl, X0, Y0, wl, rows, [1.1, 1.6, 1.5], size=20, row_h=58)
    rows = [["Sensor", "Velodyne HDL-64E"],
            ["Beams", "64"],
            ["Frame rate", "10 Hz"],
            ["Horizontal resolution", "~0.09°"],
            ["Max range", "120 m"],
            ["Points per frame", "~100,000–120,000"]]
    table(sl, xr, Y0, w - wl, rows, [1.4, 1.3], size=20, row_h=48)
    seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    frames = ["4541", "1101", "4661", "801", "271", "2761", "1101", "1101", "4071", "1591", "1201"]
    rows = [["Seq"] + seqs, ["Frames"] + frames, ["Classifier"] + ["val" if q == "08" else "train" for q in seqs]]
    yb = table(sl, X0, Y0 + 370, X1 - X0, rows, [1.6] + [1.0] * 11, size=20, row_h=56,
               bold={(1, 9), (2, 9)})
    text(sl, X0, yb + 18, X1 - X0, 100,
         ["Sequences 11–21 have no public labels, so no labelled sequence is left as a separate test set. "
          "For completion, every sequence is evaluation only: PCN trains on synthetic ShapeNet."], 20)


def lay_problem(sl, _):
    w = X1 - X0 - GAP
    wl = w * 0.6
    image_fit(sl, ASSETS / "d1_occlusion.png", X0, Y0, wl, Y1 - Y0)
    xr = X0 + wl + GAP
    ix, iy, iw, ih = image_fit(sl, ASSETS / "scan_panel_b.png", xr, Y0, w - wl, 680)
    text(sl, xr, iy + ih + 12, w - wl, 40, "Real car, 6 m away (Thesis Fig. 2.3)", 24,
         align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------- slide builders
def content(prs, n, section, title, key, layout, arg, note):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    text(sl, X0, 30, X1 - X0, 70, title, 44, bold=True, align=PP_ALIGN.CENTER, upper=True)
    if key:
        text(sl, X0, 105, X1 - X0, 100, key, 30, align=PP_ALIGN.CENTER)
    layout(sl, arg)
    footer(sl, section, n)
    notes(sl, note)


def divider(prs, n, line1, line2=None, note="divider: click through", sub=False):
    """Section divider (line1 at 56) or, with sub=True, a mid-section sub-divider whose only
    line uses the same style as a section divider's second line (44)."""
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    top_bar(sl)
    lines = [(line1, {"size": 44 if sub else 56})] + ([(line2, {"size": 44})] if line2 else [])
    text(sl, X0, 300, X1 - X0, 400, lines, 56, bold=True, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, upper=True, space_after=12)
    footer(sl, None, n)
    notes(sl, note)


def cover(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    image_fit(sl, ASSETS / "fpt_logo.png", 120, 40, 300, 110)
    top_bar(sl)
    text(sl, 160, 280, 1600, 330,
         "A Modular Detection and Completion Framework for Occluded Vehicles in Automotive LiDAR",
         48, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, upper=True)
    text(sl, 1100, 720, 760, 220,
         [("Supervisor: Dr. Doan Nhat Quang", {"italic": True}),
          "Presenter: Ngo Vi Viet Anh", "ID: 24MSE13205"],
         30, bold=True, space_after=6)
    footer(sl, None, 1)
    notes(sl, "★ TALK 0:15 — title, name, supervisor.")


def toc(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    top_bar(sl)
    text(sl, X0, 150, X1 - X0, 90, "Table of Contents", 56, bold=True,
         align=PP_ALIGN.CENTER, upper=True)
    text(sl, 300, 400, 640, 300, ["I. Introduction", "II. Proposed Methodology"], 36,
         bold=True, space_after=60)
    text(sl, 1040, 400, 700, 300, ["III. Experimental Evaluation", "IV. Conclusion & Future Work"],
         36, bold=True, space_after=60)
    footer(sl, None, 2)
    notes(sl, "★ TALK 0:10 — four parts.")


def end(prs, n):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    image_fit(sl, ASSETS / "fpt_logo.png", 810, 60, 300, 110)
    text(sl, X0, 200, X1 - X0, 140,
         [("Thank you.", {"size": 56}), ("Questions are welcome.", {"size": 36})],
         56, bold=True, align=PP_ALIGN.CENTER, upper=True, space_after=8)
    image_fit(sl, ASSETS / "d3_master_pipeline.drawio.png", X0, 400, X1 - X0, 560)  # D3: Q&A anchor
    footer(sl, None, n)
    notes(sl, "★ END — leave up during Q&A; jump to slides by number.")


# ---------------------------------------------------------------- deck content
I, II, III, IV, BK = ("I. Introduction", "II. Proposed Methodology",
                      "III. Experimental Evaluation", "IV. Conclusion & Future Work", "Backup")
T = lambda t, src: f"★ TALK {t} — source: {src}"     # noqa: E731
S = lambda src: f"○ SUPPORT (skip in talk) — source: {src}"  # noqa: E731


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = px(1920), px(1080)
    nxt = lambda: len(prs.slides) + 1  # noqa: E731  (auto page numbers)
    C = lambda sec, title, key, lay, arg, note: content(prs, nxt(), sec, title, key, lay, arg, note)  # noqa: E731
    D = lambda l1, l2=None: divider(prs, nxt(), l1, l2)  # noqa: E731
    SUB = lambda l1: divider(prs, nxt(), l1, sub=True)  # noqa: E731

    cover(prs)
    toc(prs)

    # ---------------- I. INTRODUCTION (friend: 1 Background ... 5 Objectives & Contributions)
    D("I. Introduction")
    C(I, "1.1 Background", "The laser only reaches the near side, so half of every car is missing.",
      lay_problem, None, T("1:00", "D1; fig:bg-lidar-scan (b); say the 3 point-cloud properties aloud (backup slide)"))
    C(I, "1.2 Current Approaches",
      "End-to-end networks are accurate but need dense labels and act as black boxes; we take the modular route.",
      lay_image, "d2_end_to_end_vs_modular.png",
      T("1:00", "D2; ch1 §1.1 (examples); ch2 §2.2.2–2.2.3. If asked how we compare: different metrics "
               "(their all-class mIoU vs our car-only point F1), not one scale; see 1.3 / tab:positioning"))
    C(I, "1.3 Literature Review", "Three lines of prior work, and what each leaves open.",
      lay_literature, None,
      S("ch2 §2.2 (bg-deep, bg-classical, bg-completion); tab:positioning for the per-method table. "
        "No accuracy column on purpose: their mIoU / PQ over all classes and our car-only point F1 are "
        "different measurements (ch2 §positioning)."))
    C(I, "1.4 Research Gap",
      "Accurate detectors need dense labels and are hard to inspect, and real-world completion has no ground truth to train or test against.",
      lay_cards_text, [("Dense labels", "Accurate networks need a class per point or a box per object"),
                       ("Opacity", "A missed car can't be traced to one step"),
                       ("No paired real data", "No real occluded car comes with its full shape"),
                       ("No ground truth", "Real completions can't be scored against a true shape")],
      T("0:40", "ch1 §1.1–1.2"))
    C(I, "1.5 Objectives & Contributions", "Three objectives, two contributions.",
      lay_columns, [("Objectives", [
          "1. Test whether a modular, geometry-first pipeline can detect vehicles as a transparent alternative to end-to-end networks.",
          "2. Evaluate point cloud completion on real LiDAR, where no complete ground truth exists.",
          "3. Test whether synthetic-only training and auto-mined supervision transfer to real LiDAR without new manual labels."]),
          ("Contributions", [
          "1. A modular detection and completion framework for occluded vehicles, inspectable and reproducible at every stage.",
          "2. Zero additional manual annotation: classifier labels are mined automatically; the completion network trains only on synthetic shapes."])],
      T("0:45", "ch1 §Objectives, §Contributions"))

    # ---------------- II. PROPOSED METHODOLOGY (friend: 1 -> 2 -> 2.1..2.4 -> 3 -> 4 -> 5)
    D("II. Proposed Methodology")
    C(II, "2.1 System Architecture", "Eight steps from a raw scan to completed cars.",
      lay_image, "d3_master_pipeline.drawio.png", T("1:00", "D3; fig:architecture, fig:pipeline, fig:completion-pipeline"))
    C(II, "2.2 Detection Pipeline", "Six steps turn a raw scan into tracked cars.",
      lay_image, "thesis_pipeline.png",
      S("ch3 §detection pipeline; fig:pipeline (crop: scripts/crop_thesis_figure.py 3.2 pipeline)"))
    C(II, "2.2.1 Preprocessing and Ground Removal",
      "Clean the scan, then remove the ground plane with RANSAC.",
      lay_image, "d4_ransac_real.png",
      T("0:45", "D4 (seq 08 frame 2500, one-frame illustration: counts are NOT thesis results); ch2 §RANSAC; ch3 §preprocessing, §RANSAC."))
    C(II, "2.2.2 Clustering (HDBSCAN)", "HDBSCAN groups the remaining points into object candidates.",
      lay_image, "d5_ground_then_clusters.png",
      T("0:45", "D5 = same seq 08 frame 2500 as D4 (scripts/make_d5_clusters.py, pipeline's own "
               "preprocess, remove_ground, cluster_objects); one-frame illustration: the 205 clusters are NOT "
               "a thesis result. Concept: fig:clustering; fig:bg-pipeline-stages."))
    C(II, "2.2.3 Geometric Filter",
      "Size and shape rules on each cluster's oriented box drop obvious non-cars.",
      lay_geo_filter, None,
      S("fig:geo-filter (crop: scripts/crop_thesis_figure.py 3.4 geo_filter); gates = ch3 §Geometric Filtering; "
        "P 0.199 / R 0.773 = tab:ablation-stage geometric-only row. If asked which gates cost cars: 27,718 rejected "
        "clusters overlap a GT car, 73.9% by the minimum-volume and 12.7% by the minimum-points gate, i.e. small "
        "splinters of already-split cars (the thesis states this as an inference)."))
    C(II, "2.2.4 Car / Not-Car Classifier",
      "A small PointNet decides car or not-car, trained on labels mined automatically from SemanticKITTI.",
      lay_image_or_ph, ("d67_classifier_training.drawio.png",
                        "D6+D7 merged: training lane trains the classifier (draw.io: assets/d67_classifier_training.drawio)"),
      T("1:00", "D6+D7 merged. Top lane (training) = ch3 §training data (mined, not hand labelled); bottom lane "
               "(inference) = fig:classifier, simplified. Story: the top lane is the answer key, looked up from "
               "SemanticKITTI labels; the classifier learns to give the same answer from the points alone, so on "
               "a new scan, with no labels, only the bottom lane runs. "
               "Say 'zero additional manual annotation', not 'annotation-free'. "
               "Counts: 420,333 training clusters (seq 00-07, 09, 10), 130,394 validation (seq 08), "
               "about 21% car / 79% not-car."))
    C(II, "2.2.5 Tracking and Track Filter",
      "Detections are linked into tracks across frames; a track is kept only if at least half of its frames vote car.",
      lay_full, ["D8 tracking timeline"], T("0:30", "fig:tracker"))
    C(II, "2.3 Completion Pipeline",
      "Each detected car is gated, normalised, given a length, and completed by PCN.",
      lay_full, ["thesis fig:completion-pipeline (crop from main.pdf)"],
      S("ch3 §object completion; fig:completion-pipeline"))
    C(II, "2.3.1 Completion Network (PCN)",
      "The completion network trains only on synthetic cars and must work on real scans it has never seen.",
      lay_full, ["D11 PCN training and inference"], T("1:00", "fig:pcn"))
    C(II, "2.3.2 Synthetic Training Partials",
      "Training inputs are simulated LiDAR-style partial views of synthetic ShapeNet cars.",
      lay_full, ["thesis fig:kitti-like-partial (fig61_partials PNG)"], S("fig:kitti-like-partial"))
    C(II, "2.3.3 L-Shape Gate, Normalisation and Length",
      "Each real cluster passes an L-shape gate, then is rotated and scaled into the network's training frame using an estimated car length.",
      lay_two, ["thesis fig:lshape-gate (crop from main.pdf)", "table: tab:completion-config"],
      S("fig:lshape-gate, tab:completion-config"))

    # ---------------- III. EXPERIMENTAL EVALUATION (friend: sub-dividers 1..5; results 3.1 -> 3.1.x, 3.2 -> 3.2.x)
    D("III. Experimental Evaluation", "3.1 Dataset and Protocol")
    C(III, "3.1.1 Datasets",
      "Real SemanticKITTI scans for detection; synthetic ShapeNet cars for completion.",
      lay_datasets, None,
      S("tab:dataset-roles, tab:sensor-spec, tab:dataset, verbatim. Seq 08 is the classifier's validation "
        "sequence, not a test set; LOSO (3.2.3) is the leakage control."))
    C(III, "3.1.2 Target Eligibility and Ground Truth",
      "A car is scored only if enough of its points survive preprocessing.",
      lay_full, ["eligibility rule + example"],
      S("ch4 §eligibility — ≥10 surviving points; 25.5% of annotated cars excluded; "
        "recall vs ALL annotated cars ≈ 0.55 (ch4, ch5) vs 0.732 eligible"))
    SUB("3.2 Experimental Setup")
    C(III, "3.2.1 Baselines",
      "Detection stages are compared by ablation, not against supervised detectors; "
      "completion is compared with the raw partial, a dense reference, and PoinTr.",
      lay_full, ["table: tab:positioning (why no head-to-head) + completion references"],
      S("ch4 §baselines; tab:positioning"))
    C(III, "3.2.2 Metrics",
      "Detections are matched to real cars by point overlap (IoU); completions are scored by an independent car classifier.",
      lay_two, ["detection metrics: IoU matching", "completion metric: plausibility"],
      T("0:30", "fig:bg-iou; ch4 §metrics; ch4 §plausibility"))
    C(III, "3.2.3 Data Split: Leave-One-Sequence-Out",
      "Train on ten sequences, test on the eleventh, and repeat for every sequence.",
      lay_full, ["LOSO diagram"], S("ch4 §splits; tab:loso"))
    SUB("3.3 Detection Results")
    C(III, "3.3.1 Overall Detection Results", "F1 0.812 on seq 08 (0.793 cross-validated), and each stage has a measurable job.",
      lay_results, None,
      T("1:00", "D9 = tab:ablation-stage, numbers verbatim (scripts/make_d9_funnel.py); table = tab:headline + "
               "tab:loso pooled row. Story, left to right: the geometric filter keeps most cars (recall 0.773) but 4 "
               "of 5 detections are false positives (precision 0.199); the classifier removes about 104,000 false "
               "positives (precision 0.846) and pays some recall (0.676); the track filter wins back 1,968 true "
               "positives and drops more false positives (0.911 / 0.732 / F1 0.812). Counts are per-frame matches "
               "over the sequence, not distinct cars. Table row 2 = tab:loso pooled (each fold's classifier never "
               "saw its test sequence). If asked whether we selected and report on seq 08: the seq-08 fold of "
               "LOSO (classifier blind to seq 08) gets 0.888 / 0.736 / 0.805, so selection bought about 2 "
               "precision points and no recall (tab:headline)."))
    C(III, "3.3.2 Detections on a Real Frame",
      "One seq-08 frame: found cars, false detections and missed cars.",
      lay_full, ["thesis seq08_bev_detections.png"], S("fig:bev-detections"))
    C(III, "3.3.3 What Limits Recall",
      "Clustering splits cars into fragments; the classifier is not the limit.",
      lay_full, ["D10 = thesis seq08_failure_zooms.png, frame-250 row (split car)"],
      T("1:00", "tab:split, fig:failure-zooms"))
    C(III, "3.3.4 Accuracy vs Range", "Recall falls with distance as cars get sparser.",
      lay_full, ["table: accuracy by distance"], S("tab:distance"))
    C(III, "3.3.5 Detection Result Analysis", "What each detection stage contributes, and where it fails.",
      lay_cards_ph, ["geometric filter", "classifier", "track filter"], S("ch4 §result analysis"))
    SUB("3.4 Completion Results")
    C(III, "3.4.1 Judging Completion Without Ground Truth",
      "Distance metrics reward doing nothing, so an independent judge decides whether each output looks like a car.",
      lay_two, ["D12 judge schematic", "acceptance bar chart"],
      T("1:30", "D12; tab:plausibility, fig:plausibility"))
    C(III, "3.4.2 Completion Examples: Raw vs Completed", "The same real cars before and after completion.",
      lay_full, ["thesis completion_shape_08.png"], S("fig:completion-shape, fig:plausibility"))
    C(III, "3.4.3 Synthetic Fidelity",
      "On held-out synthetic cars, the network reproduces the complete shape.",
      lay_two, ["synthetic examples", "CD and F-score"], S("ch4 §synthetic fidelity"))
    C(III, "3.4.4 Plausibility Robustness",
      "The plausibility gain holds on every sequence, for moving cars, and under a heading flip.",
      lay_two, ["table: per-sequence plausibility", "movers / flip / McNemar"],
      S("tab:plausibility-seq; ch4 §plausibility"))
    C(III, "3.4.5 Completion Result Analysis",
      "Where completion helps, and where it stays flat.",
      lay_cards_ph, ["strengths", "limitations", "vs dense reference"], S("ch4 §scope of the completion claim; ch5"))
    SUB("3.5 Runtime Evaluation")
    C(III, "3.5.1 Runtime by Stage",
      "The pipeline runs offline; the classical CPU stages dominate the time per scan.",
      lay_full, ["table: runtime per stage"], S("tab:runtime"))
    C(III, "3.5.2 Runtime Optimisation",
      "Fewer RANSAC iterations and a coarser clustering voxel cut the time per scan by about a third.",
      lay_full, ["before vs after optimisation"], S("ch4 §runtime evaluation (921.3 → 623.5 ms)"))
    SUB("3.6 Ablation Study")
    C(III, "3.6.1 Stage Ablation", "Adding one filtering stage at a time shows what each contributes.",
      lay_stage_ablation, None,
      S("tab:ablation-stage, verbatim. Same numbers as the slide 29 chart, plus mean IoU and the raw counts. "
        "Counts are per-frame matches over the full sequence, not distinct cars."))
    C(III, "3.6.2 Clustering Alternatives and Resolution",
      "No faster clusterer matches HDBSCAN's recall; a coarser resolution reduces fragmentation.",
      lay_clustering_tables, None,
      S("tab:clustering, tab:splitrate. Mean IoU column of tab:clustering left out. "
        "If asked 'why not an even coarser grid, e.g. 0.20 m?': merges grow as the grid coarsens "
        "(56 → 284), so we expect more fused cars; 0.20 m was NOT tested, say 'expect', not 'found'. "
        "If asked 'did you tune DBSCAN's radius?': yes, an epsilon sweep; the best fixed radius "
        "(Euclidean, 0.4 m) reaches F1 0.800 on seq 00, still below HDBSCAN's 0.835 (ch4 §clustering)."))
    C(III, "3.6.3 IoU-Threshold Sensitivity", "How the results change with the matching threshold.",
      lay_full, ["table: IoU sensitivity"], S("tab:iou-sensitivity"))
    C(III, "3.6.4 Synthetic Pretraining",
      "Synthetic pretraining adds nothing once enough real clusters are mined.",
      lay_pretraining, None,
      S("ch4 sec:ablation-stagea (macro-F1 0.9285 vs 0.9225; pipeline F1 identical to three decimals); "
        "tab:cross-domain. Don't mix the metrics: 0.9285 is macro-F1 (mean of car and not-car F1), 0.885 is "
        "car-class F1. Why keep it in the thesis: it changes the recipe, so it is reported as a negative result."))
    C(III, "3.6.5 Completion Ablations",
      "Input quality and normalisation matter more than the network: PoinTr improves synthetic Chamfer distance only from 0.161 to 0.153 m.",
      lay_full, ["five ablations: normalisation · sim-to-real fine-tuning (negative) · L-shape gate · length · PCN vs PoinTr"],
      S("ch4 §completion ablations (sec:abl-completion)"))

    # ---------------- IV. CONCLUSION & FUTURE WORK (friend: 1 summary, 2 limitations, 3 discussion & future)
    D("IV. Conclusion & Future Work")
    C(IV, "4.1 Research Summary",
      "Four findings from a modular framework that needs zero additional manual annotation.",
      lay_cards_text, [("Detection on real LiDAR", "F1 0.812 on seq 08; 0.793 cross-validated over all 11 sequences"),
                       ("Plausible completions", "Independent judge's car acceptance: 0.253 raw → 0.785 completed"),
                       ("Clustering-bound recall", "HDBSCAN splits 33.7% of cars; the classifier is not the limit"),
                       ("Redundant synthetic pretraining", "Enough auto-mined real clusters make it unnecessary")],
      T("0:45", "ch5 §principal outcomes, §diagnostic findings, Final Remarks"))
    C(IV, "4.2 Limitations", "Four limits on what the results show.",
      lay_cards_text, [("Dense scenes only", "Highway seq 01 recall 0.336 vs 0.758 on seq 00 (precision still 0.951)"),
                       ("Recall ceiling", "0.773 before any learned stage"),
                       ("Plausibility only", "Fidelity metrics do not corroborate the completion gain"),
                       ("Offline", "623.5 ms per frame; real time is out of scope")],
      T("0:30", "ch5 §limitations (disc-architectural, disc-completion-scope, disc-eval-validity)"))
    C(IV, "4.3 Discussion & Future Work", "Three next steps, each scoped by a finding.",
      lay_cards_text, [("Real-time variant", "A native or GPU HDBSCAN implementation"),
                       ("Real-data fine-tuning", "Close the synthetic-to-real gap for completion"),
                       ("Downstream detection", "Test whether completed cars improve 3D detection")],
      T("0:30", "ch5 §future work"))
    end(prs, nxt())

    # ---------------- Backups (after THE END, like the friend's appendix)
    C(BK, "LiDAR Point Cloud Properties",
      "A LiDAR scan is an unordered set of 3D points that gets sparser with range and sees each object from one side.",
      lay_cards_text, [("Unordered", "No fixed order, unlike the pixels of an image"),
                       ("Sparser with range", "Far objects get only a handful of points"),
                       ("One-sided", "Each object is seen from one side only")],
      S("ch2 §LiDAR point cloud; tab:sensor-spec; S1 notes A1"))
    C(BK, "How HDBSCAN Works",
      "It tries every radius at once, and keeps a group whole if it holds together longer than its pieces would.",
      lay_image, "hdbscan_explainer.png",
      S("backup for 'how does HDBSCAN work?'. TOY 2D example (scripts/make_hdbscan_explainer.py), "
        "real hdbscan library with our settings (min_samples 5, min_cluster_size 10); NOT a thesis result. "
        "Say: (1) each point measures how crowded its neighbourhood is: the distance to its 5th nearest "
        "point (itself included); (2) HDBSCAN then looks at every radius at once: at a small radius only "
        "the dense near car forms, at a medium one both cars form, at a large one they fuse; (3) shrinking "
        "the radius, it keeps a group whole if it holds together longer than its pieces would. Each car is "
        "judged only against its own pieces, never against the other car, so the dense near car and the "
        "sparse far car both survive; a fixed-radius DBSCAN would have to pick one column. "
        "Toy numbers: near car 335.7 vs its pieces 19.2 -> whole; far car 24.1, no pieces -> kept "
        "(S2 notes A4). "
        "Precision if pressed: the 'radius' is the mutual reachability distance max(core a, core b, "
        "distance a-b), and 'longest' is stability, lifetime measured in 1/radius and summed over points; "
        "groups under 10 points are noise. ch2 §density-based clustering; S2 notes A4."))
    for t in ["Repair Strategies and Why They Fail", "The Normalisation Bug",
              "Is the Judge Independent?", "Fidelity Metrics: UCD and UHD",
              "Worked Example: IoU Matching", "Pipeline Configuration"]:
        C(BK, t, None, lay_full, [t], S("backup"))

    out = OUT
    try:
        prs.save(out)
    except PermissionError:  # deck open in PowerPoint locks the file
        out = OUT.with_name(OUT.stem + "_new.pptx")
        prs.save(out)
        print(f"NOTE: {OUT.name} is open/locked; saved to {out.name} instead")
    print(f"wrote {out} ({len(prs.slides)} slides)")
    for i, sl in enumerate(prs.slides, 1):
        txt = [sh.text_frame.text for sh in sl.shapes if sh.has_text_frame and sh.text_frame.text.strip()]
        print(i, (txt[0] if txt else "").replace("\n", " / ")[:70])


if __name__ == "__main__":
    build()
