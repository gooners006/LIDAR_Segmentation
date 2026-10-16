# Canva AI — whole-deck skeleton prompt (2026-10-14)

Source: `outline.md` (main deck, slides 1–14) + thesis ch1 §Objectives/§Contributions and
ch5 §Conclusion/§Future Work. Numbers are deliberately **left out** of the prompt and typed
by hand afterwards (AI rewrites numbers). If Canva rejects the length, split after slide 7 and
repeat the STYLE block in both halves.

---

```
Create a 14-slide academic thesis-defense presentation, 16:9. Use my Brand Kit colours and fonts only.

STYLE (apply to every slide):
- White background. Minimal, clean, academic. No stock photos, no icons, no illustrations, no decorative shapes, no gradients.
- Never use orange or blue as decoration: in this deck those colours have a fixed meaning that I will apply myself.
- Content slides (2-13) share ONE layout: slide title top-left, a one-line key message directly under the title in a smaller size, a large content area in the middle, and an EMPTY footer band about 10% of the slide height along the bottom. Keep the title, key message and footer band in exactly the same position on every content slide.
- Slide number bottom-right inside the footer band.
- Wherever I write [PLACEHOLDER: ...], insert an empty light-grey rectangle with that exact label inside it. Do not fill it with any image or content.
- Use all text below VERBATIM. Do not rephrase, shorten, translate, or add any text, numbers, statistics or bullet points of your own.

SLIDE 1 (title slide, no footer band)
Title: A Modular Detection and Completion Framework for Occluded Vehicles in Automotive LiDAR
Subtitle: Master's Thesis Defense
Lines: [Your name] · Advisor: [Advisor name] · [University] · 25 October 2026
[PLACEHOLDER: point-cloud render] on the right half.

SLIDE 2
Title: The Problem: A One-Sided View
Key message: The laser only reaches the near side, so half of every car is missing.
Content: [PLACEHOLDER: D1 occlusion diagram] on the left 60%; [PLACEHOLDER: real scan, panel b] on the right 40% with the caption "Real car, 6 m away (Thesis Fig. 2.3)".

SLIDE 3
Title: Two Ways to Solve It
Key message: End-to-end networks are accurate but need dense labels and act as black boxes; we take the modular route.
Content: [PLACEHOLDER: D2 end-to-end vs modular] across the full width.

SLIDE 4
Title: Objectives and Contributions
Key message: Three objectives, two contributions.
Content: two columns with a gap between them for arrows I will draw.
Left column heading "Objectives":
1. Test whether a modular, geometry-first pipeline can detect vehicles as a transparent alternative to end-to-end networks.
2. Evaluate point cloud completion on real LiDAR, where no complete ground truth exists.
3. Test whether synthetic-only training and auto-mined supervision transfer to real LiDAR without new manual labels.
Right column heading "Contributions":
1. A modular detection and completion framework for occluded vehicles, inspectable and reproducible at every stage.
2. Zero additional manual annotation: classifier labels are mined automatically; the completion network trains only on synthetic shapes.

SLIDE 5
Title: Framework Overview
Key message: Eight small steps from a raw scan to completed cars.
Content: [PLACEHOLDER: D3 master pipeline] across the full width.

SLIDE 6
Title: Finding Object Candidates
Key message: Remove the ground plane (RANSAC), then group the remaining points into clusters (HDBSCAN).
Content: [PLACEHOLDER: D4 RANSAC plane fit] left half; [PLACEHOLDER: D5 clusters before/after] right half.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 7
Title: Keeping Only Cars
Key message: Size and shape rules drop obvious non-cars; a small PointNet decides car or not-car, trained on labels mined automatically from SemanticKITTI.
Content: [PLACEHOLDER: D6 label mining flow] left half; [PLACEHOLDER: D7 simplified PointNet] right half.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 8
Title: Tracking Removes Flicker
Key message: A car seen across many frames is kept; short-lived detections are dropped.
Content: [PLACEHOLDER: D8 tracking timeline] across the full width.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 9
Title: Detection Results
Key message: Each stage has a measurable job.
Content: [PLACEHOLDER: D9 false-positive funnel chart] left 60%; [PLACEHOLDER: results table] right 40%.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 10
Title: What Limits Recall
Key message: Clustering splits cars into fragments; the classifier is not the limit.
Content: [PLACEHOLDER: D10 fragmented car example] across the full width.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 11
Title: Completing the Car
Key message: The completion network trains only on synthetic cars and must work on real scans it has never seen.
Content: [PLACEHOLDER: D11 PCN training and inference] across the full width.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 12
Title: Judging Completion Without Ground Truth
Key message: Distance metrics reward doing nothing, so an independent judge decides whether each output looks like a car.
Content: [PLACEHOLDER: D12 judge schematic] left half; [PLACEHOLDER: acceptance bar chart] right half.
Footer band: [PLACEHOLDER: pipeline progress strip].

SLIDE 13
Title: Conclusions, Limitations and Future Work
Key message: What we built, what we found, and where it stops.
Content: three equal columns of short bullets.
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

SLIDE 14 (no footer band)
Title: Thank You
Subtitle: Questions?
Content: [PLACEHOLDER: D3 master pipeline] across the full width.
```

---

## After generation — checklist

- [ ] 14 slides returned (if fewer: split the prompt, see top).
- [ ] Title, key message and footer band sit in the same place on slides 2–13.
- [ ] No text the AI invented; no numbers anywhere except "6 m", "Fig. 2.3" and the date.
- [ ] Nothing coloured orange or blue by the AI.
- [ ] Then type the numbers by hand, from the thesis only:
  - Slide 9 table: P 0.911 / R 0.732 / F1 0.812 (seq 08, `tab:headline`); LOSO pooled F1 0.793
    (`tab:loso`). Funnel FP: 108,831 → 4,303 → 2,513 (`tab:ablation-stage`).
  - Slide 10: 33.7% of seq-08 cars fragmented.
  - Slide 12 bars: raw 0.253 / completed 0.785 / dense-reference ceiling 0.807 (`tab:plausibility`).
  - Slide 2 caption: add "1971 points" if wanted (Fig. 2.3).
- Backup slides (14, see `outline.md`) are a separate prompt later.
