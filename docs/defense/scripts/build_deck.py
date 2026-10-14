"""Build the defense deck skeleton (Deck v2 in docs/defense/outline.md) as a PPTX.

Layout mirrors the peer FPT deck (docs/LVTN.pptx): I-IV sections, sub-dividers in III,
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
    text(sl, 1040, 400, 700, 300, ["III. Experiment Evaluation", "IV. Conclusion & Future Work"],
         36, bold=True, space_after=60)
    footer(sl, None, 2)
    notes(sl, "★ TALK 0:10 — four parts.")


def end(prs, n):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    image_fit(sl, ASSETS / "fpt_logo.png", 810, 60, 300, 110)
    text(sl, X0, 200, X1 - X0, 140,
         [("The End.", {"size": 56}), ("Thanks for attending and listening.", {"size": 36})],
         56, bold=True, align=PP_ALIGN.CENTER, upper=True, space_after=8)
    placeholder(sl, X0, 400, X1 - X0, 520, "D3 master pipeline — anchor for Q&A")
    footer(sl, None, n)
    notes(sl, "★ END — leave up during Q&A; jump to slides by number.")


# ---------------------------------------------------------------- deck content
I, II, III, IV, BK = ("I. Introduction", "II. Proposed Methodology",
                      "III. Experiment Evaluation", "IV. Conclusion & Future Work", "Backup")
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
      lay_full, ["D2 end-to-end vs modular"], T("1:00", "ch1 §1.1; ch2 §2.2.2–2.2.3"))
    C(I, "1.3 Literature Review", "Three lines of prior work, and what each leaves open.",
      lay_full, ["table: approach / representative works / strengths / remaining challenges"],
      S("ch2 §2.2; tab:positioning"))
    C(I, "1.4 Research Gap",
      "Accurate detectors need dense labels and are hard to inspect, and real-world completion has no ground truth to train or test against.",
      lay_cards_text, [("Dense labels", "Accurate networks need a label on every point"),
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
      T("0:45", "ch1 §Objectives, §Contributions (Objective-2 weak spot: project_state.md)"))

    # ---------------- II. PROPOSED METHODOLOGY (friend: 1 -> 2 -> 2.1..2.4 -> 3 -> 4 -> 5)
    D("II. Proposed Methodology")
    C(II, "2.1 System Architecture", "Eight small steps from a raw scan to completed cars.",
      lay_full, ["D3 master pipeline"], T("1:00", "fig:architecture"))
    C(II, "2.2 Detection Pipeline", "Six steps turn a raw scan into tracked cars.",
      lay_full, ["fig: detection pipeline (points → clusters → cars)"], S("ch3 §detection pipeline; fig:pipeline"))
    C(II, "2.2.1 Preprocessing and Ground Removal",
      "Clean the scan, then remove the ground plane with RANSAC.",
      lay_full, ["D4 RANSAC plane fit"], T("0:45", "ch3 §preprocessing, §RANSAC"))
    C(II, "2.2.2 Clustering (HDBSCAN)", "HDBSCAN groups the remaining points into object candidates.",
      lay_full, ["D5 clusters before/after"], T("0:45", "fig:clustering"))
    C(II, "2.2.3 Geometric Filter",
      "Size and shape rules on each cluster's oriented box drop obvious non-cars.",
      lay_full, ["fig: geometric filter"], S("fig:geo-filter"))
    C(II, "2.2.4 Car / Not-Car Classifier",
      "A small PointNet decides car or not-car, trained on labels mined automatically from SemanticKITTI.",
      lay_two, ["D6 label mining flow", "D7 simplified PointNet"], T("1:00", "fig:classifier"))
    C(II, "2.2.5 Tracking and Track Filter",
      "A car seen across many frames is kept; short-lived detections are dropped.",
      lay_full, ["D8 tracking timeline"], T("0:30", "fig:tracker"))
    C(II, "2.3 Object Completion",
      "Each detected car is gated, normalised, given a length, and completed by PCN.",
      lay_full, ["fig: completion path (car cluster → gate → normalise → PCN → full car)"],
      S("ch3 §object completion; fig:completion-pipeline"))
    C(II, "2.3.1 Completion Network (PCN)",
      "The completion network trains only on synthetic cars and must work on real scans it has never seen.",
      lay_full, ["D11 PCN training and inference"], T("1:00", "fig:pcn"))
    C(II, "2.3.2 Synthetic Training Partials",
      "Training inputs are simulated LiDAR-style partial views of synthetic ShapeNet cars.",
      lay_full, ["fig: KITTI-like partials"], S("fig:kitti-like-partial"))
    C(II, "2.3.3 Normalisation, L-Shape Gate and Length",
      "Each real cluster is placed in the network's training frame, gated by its L-shape, and given an estimated length.",
      lay_two, ["fig: L-shape gate", "completion settings"], S("fig:lshape-gate, tab:completion-config"))

    # ---------------- III. EXPERIMENT EVALUATION (friend: sub-dividers 1..5; results 3.1 -> 3.1.x, 3.2 -> 3.2.x)
    D("III. Experiment Evaluation", "3.1 Dataset and Protocol")
    C(III, "3.1.1 Datasets",
      "Real SemanticKITTI scans for detection; synthetic ShapeNet cars for completion.",
      lay_two, ["table: dataset roles", "table: sequences and sensor"],
      S("tab:dataset-roles, tab:dataset, tab:sensor-spec"))
    C(III, "3.1.2 Target Eligibility and Ground Truth",
      "A car is scored only if enough of its points survive preprocessing.",
      lay_full, ["eligibility rule + example"], S("ch4 §eligibility (recall-denominator weak spot)"))
    SUB("3.2 Experiment Setup")
    C(III, "3.2.1 Baselines",
      "Stages are compared by ablation; completion is compared with the raw partial, a dense reference, and PoinTr.",
      lay_full, ["table: baselines"], S("ch4 §baselines"))
    C(III, "3.2.2 Metrics",
      "Detections are matched to real cars by point overlap (IoU); completions are scored by an independent car classifier.",
      lay_two, ["detection metrics: IoU matching", "completion metric: plausibility"],
      T("0:30", "fig:bg-iou; ch4 §metrics; ch4 §plausibility"))
    C(III, "3.2.3 Data Split: Leave-One-Sequence-Out",
      "Train on ten sequences, test on the eleventh, and repeat for every sequence.",
      lay_full, ["LOSO diagram"], S("ch4 §splits; tab:loso"))
    SUB("3.3 Detection Results")
    C(III, "3.3.1 Overall Detection Results", "Each stage has a measurable job.",
      lambda sl, a: lay_two(sl, a, 0.6), ["D9 false-positive funnel chart", "results table"],
      T("1:00", "D9; tab:headline, tab:loso, tab:ablation-stage"))
    C(III, "3.3.2 What Limits Recall",
      "Clustering splits cars into fragments; the classifier is not the limit.",
      lay_full, ["D10 fragmented car example"], T("1:00", "tab:split, fig:failure-zooms"))
    C(III, "3.3.3 Accuracy vs Range", "Recall falls with distance as cars get sparser.",
      lay_full, ["table: accuracy by distance"], S("tab:distance"))
    C(III, "3.3.4 Detection Result Analysis", "What each detection stage contributes, and where it fails.",
      lay_cards_ph, ["geometric filter", "classifier", "track filter"], S("ch4 §result analysis"))
    SUB("3.4 Completion Results")
    C(III, "3.4.1 Judging Completion Without Ground Truth",
      "Distance metrics reward doing nothing, so an independent judge decides whether each output looks like a car.",
      lay_two, ["D12 judge schematic", "acceptance bar chart"],
      T("1:30", "D12; tab:plausibility, fig:plausibility"))
    C(III, "3.4.2 Completion Examples: Raw vs Completed", "The same real cars before and after completion.",
      lay_full, ["fig: raw vs completed cars"], S("fig:completion-shape, fig:plausibility"))
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
    C(III, "3.6.1 Stage Ablation", "Removing each stage in turn shows what it contributes.",
      lay_full, ["table: stage ablation"], S("tab:ablation-stage"))
    C(III, "3.6.2 Clustering Alternatives and Resolution",
      "No faster clusterer matches HDBSCAN's recall; a coarser resolution reduces fragmentation.",
      lay_two, ["table: clustering comparison", "table: split rate by resolution"],
      S("tab:clustering, tab:splitrate"))
    C(III, "3.6.3 IoU-Threshold Sensitivity", "How the results change with the matching threshold.",
      lay_full, ["table: IoU sensitivity"], S("tab:iou-sensitivity"))
    C(III, "3.6.4 Synthetic Pretraining",
      "Synthetic pretraining adds nothing once enough real clusters are mined.",
      lay_full, ["table: Stage A vs Stage B"], S("tab:cross-domain"))
    C(III, "3.6.5 Completion Ablations",
      "Comparing PCN with PoinTr, and the effect of the completion gates.",
      lay_full, ["table: completion ablations"], S("ch4 §completion ablations"))

    # ---------------- IV. CONCLUSION & FUTURE WORK (friend: 1 summary, 2 limitations, 3 discussion & future)
    D("IV. Conclusion & Future Work")
    C(IV, "4.1 Research Summary", "What we built and what we found.",
      lay_columns, [("What we built", ["• A modular detection and completion framework",
                                       "• Zero additional manual annotation"]),
                    ("What we found", ["• Recall is limited by clustering, not by the classifier",
                                       "• Completed cars are judged car-like far more often than raw scans"])],
      T("0:45", "ch5 §principal outcomes, §diagnostic findings"))
    C(IV, "4.2 Limitations", "Where the results stop.",
      lambda sl, a: text(sl, 360, Y0 + 20, 1200, Y1 - Y0, a, 32, space_after=24),
      ["• Cars only", "• One dataset and one sensor", "• Offline, not real-time",
       "• Recall capped by clustering fragmentation"],
      T("0:30", "ch5 §limitations"))
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
