# Canva AI — skeleton v2: friend's layout/structure, our colours (2026-10-14)

Reference: peer FPT defense deck (Canva DAHKdf4GLzM, local `docs/LVTN.pptx`). Copied: layout and
structure (logo, TOC, Roman-numeral dividers, centered UPPERCASE numbered titles restarting per
section, full-width footer bar with section name + page number, end slide then backups).
**Not** copied: their green/Roboto, their density (55 main slides).

Change vs v1: slide "Completing the Car" moves from after the detection results into section II
(methodology), so sections mirror thesis ch3 (methods) / ch4 (evaluation).

Numbers are still left out on purpose; type them by hand from the thesis (checklist in v1 file).

---

```
Create a 19-slide thesis-defense presentation, 16:9, canvas 1920 x 1080 px. Use my Brand Kit fonts and colours only. White background. No stock photos, icons, illustrations, gradients or decorative shapes beyond what is listed. Use all text VERBATIM; do not rephrase, add or remove text, numbers or bullets. Wherever I write [PLACEHOLDER: ...], insert an empty light-grey rectangle (#F7F7F7) with that exact label centred inside it, size 20.

SHARED ELEMENTS
- Footer bar (every slide): full-width rectangle x = 0, y = 1000, width 1920, height 80, fill #1F2933. Inside it: the section name on the left (x = 40, white, Bold Italic, UPPERCASE, size 24) and the slide number on the right (right edge x = 1880, white, size 24). The section name is given per slide below; if none is given, leave it empty.
- Top-right bar (only on slides 1, 2 and the section dividers): rectangle x = 1240, y = 50, width 600, height 45, fill #1F2933.
- Content slide layout (every slide not marked otherwise): title centred at top, y = 30, Bold, UPPERCASE, size 44, one line. Key message centred below it, top at y = 105, Medium, size 30, max width 1680, may wrap to 2 lines. Content area from y = 220 to y = 880, x = 120 to 1800.
- Pipeline strip (only where stated): [PLACEHOLDER: pipeline progress strip] at x = 120, y = 905, width 1680, height 50.
- Section divider layout: only the section title, centred both ways, Bold, UPPERCASE, size 56, plus the top-right bar and the footer bar (footer section name empty).

SLIDE 1 (title)
Top-left: [PLACEHOLDER: university logo] at x = 120, y = 40, width 300, height 110.
Centred title, Bold, UPPERCASE, size 48, max 3 lines: A Modular Detection and Completion Framework for Occluded Vehicles in Automotive LiDAR
Bottom-right block, left-aligned at x = 1100, starting y = 700, size 30:
  line 1 (Bold Italic): Supervisor: Dr. Doan Nhat Quang
  line 2 (Bold): Presenter: Ngo Vi Viet Anh
  line 3 (Bold): ID: 24MSE13205
Footer section name: empty.

SLIDE 2 (table of contents)
Title centred, Bold, UPPERCASE, size 56, y = 120: Table of Contents
Two columns of section names, size 36, Bold:
  left column: I. Introduction / II. Proposed Methodology
  right column: III. Experiment Evaluation / IV. Conclusion & Future Work
Footer section name: empty.

SLIDE 3 (section divider): I. INTRODUCTION

SLIDE 4. Footer: I. INTRODUCTION
Title: 1. The Problem: A One-Sided View
Key message: The laser only reaches the near side, so half of every car is missing.
Content: [PLACEHOLDER: D1 occlusion diagram] from y = 220 to y = 860, left 60%; [PLACEHOLDER: real scan, panel b] from y = 220 to y = 860, right 40%. Caption "Real car, 6 m away (Thesis Fig. 2.3)", size 24, centred under the right box with its top at y = 875; it must end above y = 920.

SLIDE 5. Footer: I. INTRODUCTION
Title: 2. Two Ways to Solve It
Key message: End-to-end networks are accurate but need dense labels and act as black boxes; we take the modular route.
Content: [PLACEHOLDER: D2 end-to-end vs modular] full width.

SLIDE 6. Footer: I. INTRODUCTION
Title: 3. Objectives and Contributions
Key message: Three objectives, two contributions.
Content: two columns. Column headings SemiBold size 36; numbered items Regular size 32, line spacing 1.25, 20 px between items.
Left heading "Objectives":
1. Test whether a modular, geometry-first pipeline can detect vehicles as a transparent alternative to end-to-end networks.
2. Evaluate point cloud completion on real LiDAR, where no complete ground truth exists.
3. Test whether synthetic-only training and auto-mined supervision transfer to real LiDAR without new manual labels.
Right heading "Contributions":
1. A modular detection and completion framework for occluded vehicles, inspectable and reproducible at every stage.
2. Zero additional manual annotation: classifier labels are mined automatically; the completion network trains only on synthetic shapes.

SLIDE 7 (section divider): II. PROPOSED METHODOLOGY

SLIDE 8. Footer: II. PROPOSED METHODOLOGY
Title: 1. Framework Overview
Key message: Eight small steps from a raw scan to completed cars.
Content: [PLACEHOLDER: D3 master pipeline] full width.

SLIDE 9. Footer: II. PROPOSED METHODOLOGY. With pipeline strip.
Title: 2. Finding Object Candidates
Key message: Remove the ground plane (RANSAC), then group the remaining points into clusters (HDBSCAN).
Content: [PLACEHOLDER: D4 RANSAC plane fit] left half; [PLACEHOLDER: D5 clusters before/after] right half; 40 px gap.

SLIDE 10. Footer: II. PROPOSED METHODOLOGY. With pipeline strip.
Title: 3. Keeping Only Cars
Key message: Size and shape rules drop obvious non-cars; a small PointNet decides car or not-car, trained on labels mined automatically from SemanticKITTI.
Content: [PLACEHOLDER: D6 label mining flow] left half; [PLACEHOLDER: D7 simplified PointNet] right half; 40 px gap.

SLIDE 11. Footer: II. PROPOSED METHODOLOGY. With pipeline strip.
Title: 4. Tracking Removes Flicker
Key message: A car seen across many frames is kept; short-lived detections are dropped.
Content: [PLACEHOLDER: D8 tracking timeline] full width.

SLIDE 12. Footer: II. PROPOSED METHODOLOGY. With pipeline strip.
Title: 5. Completing the Car
Key message: The completion network trains only on synthetic cars and must work on real scans it has never seen.
Content: [PLACEHOLDER: D11 PCN training and inference] full width.

SLIDE 13 (section divider): III. EXPERIMENT EVALUATION

SLIDE 14. Footer: III. EXPERIMENT EVALUATION. With pipeline strip.
Title: 1. Detection Results
Key message: Each stage has a measurable job.
Content: [PLACEHOLDER: D9 false-positive funnel chart] left 60%; [PLACEHOLDER: results table] right 40%; 40 px gap.

SLIDE 15. Footer: III. EXPERIMENT EVALUATION. With pipeline strip.
Title: 2. What Limits Recall
Key message: Clustering splits cars into fragments; the classifier is not the limit.
Content: [PLACEHOLDER: D10 fragmented car example] full width.

SLIDE 16. Footer: III. EXPERIMENT EVALUATION. With pipeline strip.
Title: 3. Judging Completion Without Ground Truth
Key message: Distance metrics reward doing nothing, so an independent judge decides whether each output looks like a car.
Content: [PLACEHOLDER: D12 judge schematic] left half; [PLACEHOLDER: acceptance bar chart] right half; 40 px gap.

SLIDE 17 (section divider): IV. CONCLUSION & FUTURE WORK

SLIDE 18. Footer: IV. CONCLUSION & FUTURE WORK
Title: 1. Conclusions, Limitations and Future Work
Key message: What we built, what we found, and where it stops.
Content: three equal columns, all top-aligned at the same y. Column headings SemiBold size 36, one line each; bullets Regular size 32, line spacing 1.25, 20 px between items.
Column "What we built":
- A modular detection and completion framework
- Zero additional manual annotation
Column "What we found":
- Recall is limited by clustering, not by the classifier
- Completed cars are judged car-like far more often than raw scans
Column "Limits and next steps":
- Cars only; one dataset and sensor
- Offline, not real-time
- Next: faster clustering, real-data fine-tuning, downstream detection test

SLIDE 19 (end). Footer section name: empty.
Centred, Bold, UPPERCASE, size 56, y = 120: Thank You
Below it, centred, size 36: Questions?
[PLACEHOLDER: D3 master pipeline] from y = 330 to y = 900, x = 120 to 1800.

Nothing may cross into the footer bar (y >= 1000). No text may be hidden behind any rectangle.
```

---

## After generation — checklist

- [ ] 19 slides; footer bar on every slide with the right section name.
- [ ] Titles numbered and restarting per section (I.1–3, II.1–5, III.1–3, IV.1).
- [ ] No invented text or numbers; nothing orange or blue added by the AI.
- [ ] No stray rectangles over placeholders (v1 had one on slides 6–12).
- [ ] Upload the FPT logo into the slide-1 placeholder yourself.
- [ ] Slide 4 (The Problem): the AI can't place images. Re-upload `figures/d1_occlusion.svg`
      (labels enlarged 2026-10-14) into the D1 box, and copy the cropped panel-(b) scan from the
      old deck into the right box. Then delete each grey placeholder rectangle behind them (or
      set it to white) so the white image edges don't show as a box-in-a-box.
- [ ] Export PNG (all pages) + PPTX into `docs/defense/canva_check/` for review.
