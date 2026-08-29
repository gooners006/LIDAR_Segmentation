# Defense Deck — Slide-by-Slide Storyboard

**Status:** draft for sign-off (Section B, Step 1 of `docs/defense/plan.md`).
Written 2026-08-30. **Locked decisions:** A2 = 25–30 min / ~45 slides;
A3 = completion-forward. A1 (build tool) still open — this storyboard is
tool-agnostic.

**Conventions.** Per slide: **N. Title** — one-line message · _source_ (§ / Finding) ·
_visual_. Numbers are copied from `docs/project_state.md` / `docs/findings.md` — do not
re-derive. Every claim ⊆ the C1–C11 claims map. Hedges that must survive onto the
slide are marked **⚠HEDGE**. Running footer = current section (I–IV) + slide number,
mirroring `LVTN.pptx`. Target ~40 content slides + backups.

---

## Front matter

**1. Title** — thesis title, presenter, advisor · _—_ · full-bleed title layout.
Title: *A Donor-Frame Coverage Metric for Reference-Free Evaluation of
Occluded-Vehicle Completion in Automotive LiDAR*. Presenter: Ngo Vi Viet Anh
(MSE13205). Supervisor: Dr. Doan Nhat Quang.

**2. Table of contents** — I. Introduction · II. Proposed Methodology ·
III. Experiment Evaluation · IV. Conclusion & Future Work · _—_ · 4-item TOC.

---

## I. Introduction  (~6 content)

**3. [Divider] I. Introduction**

**4. Background & motivation** — LiDAR returns 100k+ points several times a second;
turning that into an object list is the first perception step. The dominant route —
end-to-end deep semantic segmentation — needs densely labelled scans (SemanticKITTI:
billions of point labels), an annotation cost out of reach for most, and its failures
are distributed across millions of weights (uninspectable). · _§1.1; Behley 2019_ ·
one KITTI scan visual + the annotation-cost point.

**5. This thesis: the opposite starting point** — how far can an *interpretable,
near-annotation-free* pipeline go? Classical geometry where geometry is reliable,
small learned models only where discrimination needs them. And a *second thread*:
completing occluded cars — where the hard part turns out to be **how to measure
completion on real data at all**. · _§1.1, §1_ · two-thread framing (detection ‖
completion); foreshadow completion as the headline.

**6. Approach at a glance** — modular hybrid: ground removal → HDBSCAN clustering →
geometric filter → binary classifier → tracker → PCN completion. ⚠HEDGE tracker is a
temporal *filter*, not a contribution; completion trained on synthetic renders only
(never a labelled real car). · _§1.2_ · teaser pipeline strip (full diagram on slide 10).

**7. Research questions** — RQ1: how to evaluate completion on real LiDAR with no clean
reference? RQ2: does completion recover genuinely *unseen* surface and improve amodal
boxes — and does it hold out? RQ3: what limits detector recall, and is synthetic
pretraining necessary? · _§1.3_ · three-RQ list.

**8. Contributions** — (1) a leakage-free real-data completion metric *(headline)*;
(2) a root-cause account of the recall bottleneck + a chain of negative repairs;
(3) evidence synthetic pretraining is redundant at scale. ⚠HEDGE methodological /
diagnostic, **not** architectural; pipeline not claimed novel. · _§1.4, §2.5_ ·
three-card layout, donor metric largest.

---

## II. Proposed Methodology  (~10 content)

**9. [Divider] II. Proposed Methodology**

**10. System architecture** — the full pipeline with measured per-stage cost. · _§3,
Fig 3.1_ · **Fig 3.1** (TikZ → export PNG for the deck).

**11. Detection front-end** — ground removal (relative-height RANSAC) + HDBSCAN density
clustering proposes candidate objects with no training and no fixed radius. · _§3_ ·
before/after ground-removal + coloured clusters.

**12. Filter + classifier** — geometric size filter drops non-car-sized clusters; a
dual-branch PointNet-style **binary** car/not-car classifier labels the rest. ⚠HEDGE
two-stage synthetic→real training appears later only as an *ablation*, not a claim. ·
_§3_ · classifier schematic.

**13. Tracker = temporal filter** — centroid tracking links detections across frames so
one-frame flicker is voted away; it does **not** interpolate or recover missed cars. ·
_§3, §1.2_ · track-vs-flicker illustration.

**14. From detection to completion — the measurement problem** — each detected car is a
single-view partial; PCN can fill the occluded side. But on *real* LiDAR there is no
clean complete shape to score against. How do we know a completion is good? · _§6→§7_ ·
partial car → "?" → completed car.

**15. Why pseudo-ground-truth Chamfer is INVALID** ⚠KEY — accumulating the same static
car over many frames looks like a reference, but it is itself **one-sided** (occluded on
the same side as the input). Consequence: the **raw partial input scores best**; a lower
Chamfer distance rewards *under*-completion, not reconstruction. · _§7.1, Finding #26_ ·
diagram: input vs accumulated-pseudo-GT both missing the far side; CD-ranking table with
raw-partial on top. *Speaker note: this is the motivating failure — spend time here.*

**16. The donor-frame occluded-side coverage metric** ⚠KEY — complete the car from *one*
frame's cluster, then score how much of the surface seen only in **other** frames
(donors) of the same static car the completion recovers. Leakage-free: donor frames are
never shown to the model. · _§7.2, Finding #32_ · donor-frame schematic (query frame +
donor frames → occluded-side coverage).

**17. Hallucination guard + validation gate** — a per-band guard penalises surface
invented *outside* the car's amodal box; the metric passes a four-item validation gate.
⚠HEDGE the per-band form later exposes a band-blind failure — reported as a metric-design
lesson (#37), not hidden. · _§7.2, Finding #37_ · guard illustration (in-box vs out-of-box).

**18. Why completion produces car shapes at all** — the fix that mattered: a KITTI-like
single-view partial generator + an inference-normalisation correction turned diffuse
blobs into car-footprint shapes (#26). ⚠ hold this to the one load-bearing idea
(blob→car); the L-shape input gate is backup **B7**. · _§6, Fig 6.1_ · before/after
blob → car.

---

## III. Experiment Evaluation  (~15 content)

**19. [Divider] III. Experiment Evaluation**

**20. Setup & protocol** — SemanticKITTI; **seq 08 = reported operating point** (4,071
scans). Match rule: point-IoU ≥ 0.3, greedy 1-to-1, micro-averaged. ⚠HEDGE recall is
measured against the **car-only, ≥10-surviving-point** eligibility denominator —
establish this ground rule here; the against-all-annotated-cars figure lands with the
headline (slide 21). · _§4.1_ · protocol box; define the eligibility rule (no number yet).

**21. Detection headline** — P **0.905** / R **0.730** / F1 **0.808** / mIoU **0.912**
(TP 25478 / FP 2676 / FN 9444). Precision-saturated, recall-limited, FP ≈ 1/frame.
⚠HEDGE reveal the penalty *with* the result: 0.730 is on eligible cars; against **all**
annotated cars ≈ **0.54** (#48). · _§4.2, #34/#48_ · results table + BEV detections; the
~0.54 caveat sits beside the headline.

**22. LOSO cross-validation — validation-as-test defense** ⚠KEY — seq 08 is also the
classifier model-selection split, so we cross-validate: leave-one-sequence-out over all
11 labelled sequences, each fold blind to its test sequence. Leakage is **precision-only,
~2 pts**: leakage-free fold-08 recall **0.737** vs shipped 0.730 (recall not inflated);
pooled **P 0.874 / R 0.719 / F1 0.789**; per-sequence recall spread shown. · _Finding #51_ ·
per-sequence recall bar chart + the fold-08-vs-shipped reconciliation. *Speaker note:
pre-empt the leakage question before the committee raises it.*

**23. Recall by distance** — near-range over-segmentation dip (0–10 m R 0.787 < 10–20 m
0.862) then far-range sparsity decline. · _§4.4, Finding #50_ · distance-binned recall plot.

**24. What the classifier buys: the precision–recall trade** — one decomposition, both
axes. Recall: geometric-only 0.775 → classifier 0.677 → track filter 0.730. Precision,
same stages: **0.149 → 0.820 → 0.905**. The classifier is a precision mechanism paid for
in recall; the track filter then recovers part of the recall. · _§5.1/§4.3, #47/#49_ ·
two-line waterfall (recall + precision). *(merged former slide 34.)*

**25. Root cause: HDBSCAN splits cars** — 31–37% of GT cars are fragmented across
clusters (66% split at 0–10 m, 4% at 30–50 m); merging is negligible. · _§5.2, Finding
#23_ · **Fig 5.1** failure zooms.

**26. Repairs all fail to generalise** — six strategies (larger MCS, fragment merge,
adaptive & BEV clustering, temporal accumulation, threshold lowering) each fail for a
structural reason; the *only* lever that helped was clustering **resolution** (recall
0.699→0.730, #34), so the earlier "~0.74 hard limit" was partly a resolution artifact. ·
_§5.3–5.4, #21/#24/#34_ · Tab 5.1 (compact).

**27. Completion recovers unseen surface** ⚠KEY — occluded-side coverage @0.1 m (per-car
medians): raw partial **0.000**, symmetry-mirror baseline **0.043**, completion **0.304**
— genuinely unseen surface, not redistributed points. · _§7.3, #29/#32_ · three-bar coverage.

**28. Completion walkthrough — baseline vs ours (one car)** — partial → mirror baseline →
completion, with donor-observed surface overlaid, BEV + side. · _§7, Fig 6.3/7.x_ ·
peer-template-style walkthrough panel.

**29. Downstream amodal boxes improve** — BEV IoU vs the amodal GT box rises **0.725 →
0.771** on static, gate-passed cars (compact + normal bands); per-car length estimate cut
box |ΔL| 0.354→0.304 (#36). ⚠HEDGE static, gate-passed only. · _§7.4, #35/#36_ · **Fig
6.2** box overlay (old/corrected/GT).

**30. Moving cars** — ⚠HEDGE plausibility check only, never an accuracy claim. · _§7.5,
Finding #44_ · one mover example.

**31. Held-out replication (seq 00) — PARTIALLY HOLDS** ⚠KEY — pre-registered. Two primary
metrics replicate (BEV IoU 0.739→0.766, p=1.6e-3; donor coverage 0.000→0.413), **but** the
long length band was empty on seq 00, so long-band behaviour was untestable → not a full
HOLDS. · _§7.6, Finding #42_ · **PARTIALLY HOLDS** badge + the two replicated metrics.

**32. Runtime** — 623.5 ms/frame, 1.6 fps; classical CPU stages dominate (HDBSCAN ~59%).
⚠HEDGE offline is a *scope* boundary, not a walk-back. · _§4.5, #33/#34_ · per-stage
timing bar.

**33. Ablation: synthetic pretraining is redundant** ⚠KEY (contribution 3) — from-scratch
classifier matches pretrained (macro F1 0.9285 vs 0.9225); cross-domain matrix shows the
sim-to-real gap is **total and symmetric** (car F1 0.000 in every off-diagonal cell). ·
_§4.3, #25/#30_ · cross-domain matrix.

---

## IV. Conclusion & Future Work  (~5 content)

**34. [Divider] IV. Conclusion & Future Work**

**35. Research summary** — an interpretable, near-annotation-free pipeline: detection
root-caused rather than tuned; completion measured before claimed. · _§9.1_ · the arc, one visual.

**36. Answers to the three RQs** — RQ1 valid real-data metric (donor-frame) · RQ2
recovers unseen surface + improves boxes, **partially holds** held-out · RQ3 recall is
clustering-bound + synthetic pretraining redundant. · _§9.2_ · three-row RQ→answer table.

**37. Limitations (own it)** ⚠KEY — lead with the three most probable committee targets:
(1) 23% length-fallback to the fixed 4.14 m prior → reintroduces compact-car
hallucination (#37/#41); (2) architectural recall ceiling — density clustering has no
objectness (#34/#43); (3) operational-domain brittleness on sparse highway scenes — LOSO
seq-01 recall 0.336 (#51). _Ch 8 names six; the other three (amodal-GT unvalidatable #48,
movers plausibility-only #44, completion sim-to-real gap #26) in speaker notes._ · _Ch 8/§9_ ·
three-limit slide.

**38. Future work** — real-time variant via a native/GPU HDBSCAN (⚠ **not** cuML DBSCAN —
no fixed-radius clusterer reaches HDBSCAN recall, #43); completion real-data fine-tuning
(a documented negative line, #16/#17/#19); long-band amodal GT on a second held-out
sequence. · _§9.4_ · three future items.

**39. Contributions restated** — the same three, compactly; ⊆ claims map C1–C11. · _§9.3_ ·
three-card recap.

**40. Thanks / Q&A.**

---

## Backup / appendix (after "thanks" — pull up on demand)

- **B1.** Full seq-08 headline + per-sequence LOSO table (all 11 folds). · _#51_
- **B2.** Tab 5.1 — full six repair strategies with outcomes. · _#21/#24_
- **B3.** Donor-metric four-item validation gate detail. · _#32/#37_
- **B4.** Full cross-domain classifier matrix + Stage-A/B numbers. · _#25/#30_
- **B5.** Fig 6.2 box-IoU triples (sparse/mid/dense 0.32→0.73 / 0.25→0.74 / 0.47→0.85). · _§6_
- **B6.** Completion length geometry (far_end cov 0.13→0.32, #35). · _§7.4_
- **B7.** L-shape input gate — rejects non-L partials, raising completion precision
  38%→69% (#27); demoted from the main deck (slide 18). · _§6_

---

## Slide-count check

Front 2 · I 1+5 · II 1+9 · III 1+14 · IV 1+5 · thanks 1 = **~40** + 7 backups.
Within the ~44–48 target for 25–30 min at ~1.5 min/substantive slide.
