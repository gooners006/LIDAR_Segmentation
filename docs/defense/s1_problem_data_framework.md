# S1 — The Problem, the Data, the Framework

Study notes for slides 2–5 and diagrams D1–D3. All facts come from the submitted thesis
(`docs/writing/thesis/`); thesis locations are given in brackets. Anything not from the thesis
is marked **[not in thesis]**.

---

## Part A — Concepts

### A1. What a LiDAR scan is

**Core meaning.** A LiDAR is a spinning laser rangefinder. It fires laser pulses, measures how
long each one takes to bounce back, and turns that time into a distance. One full turn gives a
*scan*: a list of 3D points (x, y, z) on whatever surfaces the lasers hit. That list is the
**point cloud** [ch2 §2.1.1].

**How it's produced.** In SemanticKITTI the sensor is a Velodyne HDL-64E [ch4 Table
`tab:sensor-spec`]:

| Parameter | Value |
|---|---|
| Laser beams (stacked vertically) | 64 |
| Spin rate | 10 Hz (10 scans per second) |
| Horizontal angle between neighbouring points | ~0.09° |
| Max range | 120 m |
| Points per scan | ~100,000–120,000 |

**Three properties drive every difficulty in the thesis** [ch2 §2.2.1]:
1. **Unordered.** The points have no fixed order, unlike pixels in an image. This is why a
   special network (PointNet) is needed later.
2. **Sparse, with density that drops with range.** Near objects get many points; far objects
   get only a handful.
3. **Single-view partial.** Each object is seen from one side only.

**Worked example (density vs range).** Neighbouring laser spots are ~0.09° apart horizontally.
Convert to radians: 0.09 × π/180 ≈ 0.00157 rad. Spot spacing ≈ range × angle:
- at **6 m**: 6 × 0.00157 ≈ **0.9 cm** apart
- at **40 m**: 40 × 0.00157 ≈ **6.3 cm** apart, about 7× sparser *in each direction*

The thesis figure `bg_lidar_scan.png`(b) shows a real car at 6 m with **1971 points**. If
spacing grows in both horizontal and vertical directions, point count falls roughly with
1/range². So the same car at 30 m would get roughly 1971/25 ≈ 80 points. *(This 1/r² estimate
is my back-of-envelope inference and is not measured in the thesis. The thesis does measure the
effect indirectly: recall falls to 0.196 beyond 40 m in `tab:distance`.)*

**Why it matters.** Sparsity at range is why distant cars are missed (S4). Single-view
partiality is why completion exists at all (S5).

### A2. The problem: half of every car is missing

**Core meaning.** A car hides its own far side from the sensor (self-occlusion). Seen from a
diagonal, the laser hits two faces, for example the rear and the near side, and the points form
an **"L"**. The other two faces are never observed [ch2 Fig `bg-lidar-scan`(b)]. Seen exactly
side-on or rear-on, only one face is hit and the points form an **"I"**.

The thesis therefore has two tasks:
1. **Detection**: which points in the scan belong to a car?
2. **Completion**: what does the hidden side of each detected car look like?

**Why completion is hard on real data** [ch1 §1.2]:
- **No paired training data.** No dataset has a real occluded car next to its true complete
  shape, so the network trains on synthetic CAD cars and must transfer to real scans.
- **No ground truth for evaluation.** On real LiDAR the complete shape is unknown. Stacking the
  same parked car over many frames doesn't fix this, because every frame sees it from roughly
  the same side. Metrics that compare the output with the observed points reward "doing
  nothing". (Details in S6.)

### A3. Two ways to solve it, and why the thesis picks the modular one

**The dominant approach: end-to-end deep networks** [ch1 §1.1]. One big network maps the raw
scan straight to labels or boxes (RangeNet++, Cylinder3D, PointPillars, …). It is the most
accurate, but it has two costs:
1. **Supervision.** It needs dense per-point labels: a class on every one of billions of points.
   That is too expensive for most groups and most new datasets.
2. **Opacity.** Each decision is spread over millions of weights, so when a car is missed you
   can't point to the step that failed.

**The thesis approach: modular** [ch1 §1.1, ch3 §3.1]. A chain of small steps. Classical
geometry does what geometry does well (finding the ground, grouping points). Small learned
models are used only where a judgement is needed (car or not; what the hidden side looks like).

**What is traded.** Peak accuracy is traded for inspectability and zero additional manual
annotation. The thesis says so openly [ch2 §2.2.5]: published networks reach 52.2–67.8 mIoU
(semantic) and 37.1–57.4 PQ (panoptic), while this work reports car-only point-level F1 0.812 on
seq 08. **These are different measurements**, so the thesis deliberately doesn't put them in one
column. "The lower accuracy is a difference of paradigm, not a lower score on a shared axis."

**Why the trade pays off, with evidence.** Because the steps are separate, the evaluation can
pin the recall limit on **one stage** (clustering fragments 33.7% of cars), not on an opaque
network [ch3 §3.1, ch4 §4.4]. That diagnosis is the payoff of modularity (S4).

### A4. The two datasets

| | SemanticKITTI | ShapeNetCore v2 |
|---|---|---|
| What it is | Real Velodyne scans from a car driving in Karlsruhe **[not in thesis: city name]** | Synthetic 3D CAD models |
| Role | Detection, classifier label mining, evaluation (ground truth) | Training the completion network (PCN) |
| Size used | 11 labelled sequences (00–10), **23,201 frames** | ~3,150 car models |
| Labels used | Per-point semantic + instance labels | None (partial views rendered from the meshes) |

[ch4 Tables `tab:dataset-roles`, `tab:dataset`]

**Split** [ch4 `tab:dataset`]. The classifier trains on all sequences except **08**, which is
validation. That's why seq 08 is the "headline" sequence. PCN never sees any real sequence, so
all 11 sequences are evaluation-only for completion.

**Why there is no normal held-out test set.** Sequences 11–21 have no public labels (they're
scored only on SemanticKITTI's server) [ch4 §4.1.1]. Every labelled sequence is needed for
something, so instead of a test set the thesis uses **leave-one-sequence-out (LOSO)
cross-validation**: train on 10 sequences, test on the 11th, repeat 11 times (S4).

**"Zero additional manual annotation", stated precisely.**
- The classifier's labels **do** come from SemanticKITTI's existing ground truth. The thesis's
  clusters are matched to existing labels automatically ("auto-mined").
- The claim is that **no new human labelling** was done, not that no labels were used. Never say
  "annotation-free".
- The completion network uses no real labels at all; it trains on synthetic shapes only.

### A5. The framework

**Two stages, eight steps** [ch3 Figs `fig:architecture`, `fig:pipeline`;
ch4 `tab:sup-unsup`]:

```
Raw scan
 └─ SEGMENTATION (points → clusters)        [classical]
     1. Preprocess: height filter → denoise + voxel downsample
     2. Ground removal: RANSAC plane
     3. Clustering: HDBSCAN @ 0.10 m
 └─ DETECTION (clusters → cars)
     4. Geometric filter: oriented-box size/shape gates  [classical]
     5. Classifier: PointNet, car / not-car              [LEARNED: auto-mined labels]
     6. Tracking: centroid tracker + track vote          [classical]
 └─ COMPLETION (car → full car)
     7. L-shape gate + canonical normalisation + length  [classical]
     8. PCN network                                      [LEARNED: synthetic ShapeNet]
Completed cars
```

- **6 classical (training-free) steps, 2 learned ones.** Learned = orange; classical = blue. Use
  the thesis colours in the slides.
- **Offline by design.** The tracker uses the whole sequence, so it isn't real-time; real-time
  operation is explicitly out of scope [ch3 §3.2.1].
- **Deterministic.** Fixed seeds mean every number is exactly reproducible [ch4 §4.1.1].
- **Frozen configuration.** All parameters were fixed before the write-up; nothing was tuned to
  produce the reported numbers [ch4 §4.1.1, ch3 `tab:pipeline-config`].

---

## Part B — Q&A traps from this session

1. **"You say no annotation, but you use SemanticKITTI labels."**
   → "We say zero *additional* manual annotation. The classifier's labels are mined
   automatically from existing SemanticKITTI ground truth; no new labelling was done. The
   completion network uses no real labels at all, only synthetic shapes."

2. **"Your accuracy is far below RangeNet++ or Cylinder3D."**
   → "Those are all-class mIoU/PQ on the test server; ours is car-only point-level F1 on seq 08.
   They aren't on the same axis, and we don't contest the gap. What we gain is a pipeline with
   no segmentation labels, a single consumer GPU, and failures we can trace to a specific stage.
   That's how we found that recall is limited by clustering."

3. **"Why call it a detector if you don't output boxes / report box AP?"** ⚠️ **The thesis is
   inconsistent here.** ch3 §3.1 says detection "emits an oriented bounding box". ch2 §2.2.5
   says the pipeline "never fits an oriented box; box AP would require a box fitting stage the
   pipeline deliberately does not have".
   - Reality, from ch3 §3.2.5: each cluster **does** get an oriented bounding box (OBB) fitted
     tightly around its *observed* points. The geometric filter uses it, and the classifier uses
     its dimensions as features.
   - What ch2 means is that there is no box *estimation* stage producing full-extent car boxes
     that are scored like PointPillars output. The thing evaluated is the **point cluster**,
     scored by point-level IoU.
   - Prepared answer: "Every cluster gets a tight oriented box around its observed points; we use
     it for filtering and as a classifier feature. We don't score boxes, because our output is
     the point cluster, which is also what completion needs. So point-level IoU is the metric
     that matches what we produce. The wording in §2.2.5 should have said we have no box
     *estimation* stage." If pressed, admit the wording slip calmly. That's better than
     defending it.

4. **"Why not just use PointPillars?"**
   → It needs 3D box labels for training and outputs boxes, not point clusters. The thesis wants
   (a) no new labels and (b) per-point clusters to feed completion.

5. **"Can it run in real time?"**
   → "It's offline by design: the tracker looks at the whole sequence. Real-time was out of
   scope. Per-stage runtime is measured in §4.2.3." (Runtime numbers come in S4.)

6. **"You said stacking frames can't give ground truth, but you stack frames for the
   'static-accumulation ceiling'."** (added 2026-10-16)
   → "Stacking a parked car gives a denser shape, but still from roughly the same side, so it
   isn't a complete ground truth. We use it only as a reference ceiling for the judge (0.807,
   N=140); no accumulated pseudo-ground truth enters the completion claim [ch5 §Ground Truth and
   Dataset Assumptions]."

7. **"Why is PointNet in your pipeline if end-to-end networks are the problem?"** (added
   2026-10-16; slide 1.2 deliberately names RangeNet++, Cylinder3D, PointPillars, CenterPoint,
   not the PointNet family)
   → "The problem isn't learning; it's one opaque network trained on dense labels. Our PointNet is
   a small car/not-car classifier on one cluster at a time, trained on auto-mined labels, so
   its decision is one inspectable step."

**Precision reminders from the 2026-10-16 retry:**
- ShapeNet = synthetic **CAD meshes**, not scans. PCN trains on simulated LiDAR-style partial
  views **rendered from** the meshes.
- Seq 08 = the classifier's **validation** sequence, not an untouched test set (the final
  classifier was selected on it), which is why LOSO exists (fold-08: 0.888 / 0.736).
- No held-out test set because **sequences 11–21 have no public labels**.

---

## Part C — Diagrams D1–D3

### Deck-wide conventions (D1 and D2 were drawn by Claude as SVG; their Canva steps below are kept only for reference)

- **Colours:** blue `#DCE8F5` fill / `#2B5C8A` outline = classical; orange `#FDE6CC` /
  `#D9822B` = learned; red `#D7263D` = sensor and things to notice; grey `#9AA0A6` = unseen or
  missing. Same meaning on every slide, so the colour itself tells part of the story.
- **Font:** one sans-serif for everything. Body text ≥ 18 pt; diagram labels ≥ 14 pt.
- **Canva setup:** Create a design → Presentation (16:9). In Brand Kit (included with Business),
  save the four colours so they're one click away.

### D1 — "The sensor only sees one side" (Canva, animated) — slide 2

**What it must show.** Rays from the sensor hit the near faces of a car (the L); the far faces
are never reached; there is a shadow behind the car.

**Layout (top-down view):**
```
                              ┌─────────────┐
   ▲ sensor ·····ray····· ▶ ██│             │░░░░  occlusion
   (ego car)  ··ray··  ▶ ████ │  ? unseen ? │░░░░░  shadow
             ····ray···▶ ███████████████████ ░░░░
                                (L = observed faces, thick red)
                         far faces = dashed grey
```

**Steps:**
1. **Ego car.** Elements → Shapes → rounded rectangle, small, left side. Put a red triangle on
   it for the sensor.
2. **Target car.** A larger rounded rectangle, right side, rotated about 20° so the sensor sees
   it diagonally (that gives the L).
3. **Observed faces.** Elements → Lines. Draw two **thick red lines** over the two faces nearest
   the sensor. That's the L.
4. **Unseen faces.** Two **dashed grey lines** over the other two faces. Line style → dashed.
5. **Rays.** Draw 5–6 thin lines from the sensor tip to points on the red faces. Draw one, then
   duplicate (Ctrl+D) and drag its far endpoint. Keep the near endpoint on the sensor.
6. **Shadow.** A grey triangle (or freeform) behind the car, transparency ~70%, labelled
   "occlusion shadow".
7. **Animate.** Animate → element animations. Order: rays → red L → grey dashed faces + "?"
   label. In the talk: "the laser hits here… and here… but this side, it never reaches."
8. **Real evidence.** Place `figures/bg_lidar_scan.png` panel (b) beside it (crop panel (b) in
   Canva), captioned "real car, 6 m, 1971 points".

**Self-check while drawing.** Rotate the target car to 0° (side-on). The visible part becomes an
"I", not an "L". You should be able to explain why.

### D2 — "End-to-end vs modular" (Canva) — slide 3

**Layout: two horizontal lanes.**
```
  LANE 1   [scan] ──▶ ███ ONE DEEP NETWORK ███ ──▶ [labels]
                           ▲ dense per-point labels (billions of points)
                      + highest accuracy   − opaque   − label-hungry

  LANE 2   [scan] ──▶ [□]→[□]→[□]→[□]→[■]→[□]→[□]→[■] ──▶ [completed cars]
                                       ▲              ▲
                              auto-mined labels  synthetic shapes
                      + inspectable   + zero additional annotation   − lower accuracy
```

**Steps:**
1. **Lane 1.** One wide, dark rectangle (black or navy, white text) between two small icons
   (Elements → search "lidar" or "point cloud" for an icon, or use a crop of a scan). A label
   above with an arrow down into the box: "dense per-point labels".
2. **Lane 2.** Eight small rectangles, one per step in A5: 6 blue (classical) and 2 orange
   (learned, steps 5 and 8) in pipeline order. Draw one, Ctrl+D, then Position → Tidy up for even spacing. Small arrows between them.
3. **Inputs to the learned boxes.** Labels under the two orange boxes: "auto-mined labels" and
   "synthetic shapes".
4. **Pros and cons** under each lane, using ✓ and ✗. **Keep "highest accuracy" as a ✓ on lane
   1.** Being honest on the slide disarms the accuracy question before it's asked.

### D3 — Master pipeline (draw.io, drawn by you) — slides 2.1 and THE END

*(Rewritten 2026-10-16 for the local deck. Canva and the "you are here" strip are gone.)*

**What D3 must add over D2.** Slide 1.2 (D2) already shows the eight steps as a plain chain,
so D3 shouldn't repeat it. D3 adds three things:
1. **The two stages** from the thesis title and `fig:architecture`: *Detection*, then
   *Completion*, with a data hand-off between them.
2. **What flows along each arrow** (the I/O labels from `fig:pipeline` and
   `fig:completion-pipeline`), so a committee member can point anywhere and ask "what's here?".
3. **The method inside each box** (RANSAC, HDBSCAN, PointNet, PCN), so it works as the Q&A
   anchor on THE END slide, which stays up for all 30 minutes of questions.

Keep it at **eight steps**, the same as D2 and slide 2.1's key message ("Eight small steps from
a raw scan to completed cars"). The thesis completion figure has six boxes; here they collapse
into two, with "rotate back" folded into PCN's output label.

**Layout: two rows, both read left to right.**
```
 ROW 1 ─ DETECTION ────────────────────────────────────────────────────────────────────────
 (Raw scan) → ┌ Segmentation: points → clusters ┐   ┌ Detection: clusters → cars ┐
              │ [Preprocess] [Ground   ] [Cluster-]│ → │ [Geometric] [Classifier] [Tracking] │ → (Car detections)
              │ z·voxel·noise  removal    ing      │   │  filter     PointNet      centroid  │    boxes + track ids
              └──────────────────────────────────┘   └───────────────────────────────┘         │
               ↓cleaned      ↓non-ground ↓clusters      ↓candidates ↓car clusters ↓tracked cars    │
                                                                                                │
 ROW 2 ─ COMPLETION ───────────────────────────────────────────────────────────────── ┌───────┘
   ┌ Canonical-frame prep ─────────────────────────┐   ┌ Completion ┐
 → │ [Gate + normalise                             ]│ → │ [PCN       ]│ → (Completed cars)
   │  L-shape gate · canonical pose · centre · scale│   │  256 → 4096 │    dense surface + box
   └───────────────────────────────────────────────┘   └────────────┘
     ↓256-point input                                    ↓car in sensor frame
```
The hand-off arrow runs from *Car detections* (end of row 1) down and back to the start of
row 2. Draw.io's elbow connector does this for you.

**Box text** (name in bold; method underneath in a smaller font). Colour: blue = classical, orange = learned.

| # | Box | Method line | Colour | Output label (grey italic, under the box) |
|---|---|---|---|---|
| 1 | **Preprocess** | height filter · voxel · denoise | blue | → cleaned points |
| 2 | **Ground removal** | RANSAC plane | blue | → non-ground points |
| 3 | **Clustering** | HDBSCAN @ 0.10 m | blue | → object clusters |
| 4 | **Geometric filter** | oriented-box size gates | blue | → car candidates |
| 5 | **Classifier** | PointNet, car / not-car | orange | → car clusters |
| 6 | **Tracking** | centroid tracker + track vote | blue | → tracked cars |
| 7 | **Gate + normalise** | L-shape gate · canonical pose · centre · scale | blue | → 256-point input |
| 8 | **PCN** | 256 → 4096 points | orange | → car in sensor frame |

Data nodes (grey ellipses): **Raw LiDAR scan**, **Car detections** (boxes + track ids), and
**Completed cars** (dense surface + box). Small orange labels feeding the two learned boxes,
as in D2: *auto-mined labels* → Classifier and *synthetic CAD shapes* → PCN.

All of this is from ch3 (`fig:architecture`, `fig:pipeline`, `fig:completion-pipeline`). If you
want to change a label, check it against those figures first.

**Size and font (so it drops into the deck without rescaling).**
- Canvas about **1680 × 560 px** (≈ 3:1). It fills THE END slot (1680 × 520) and sits with a
  little margin on slide 2.1 (1680 × 740).
- Font **Segoe UI**, the font D1/D2 actually rendered in (Be Vietnam Pro isn't installed).
  Box names **24 px bold**, method lines **20 px**, output labels **20 px italic grey**, phase
  labels **22 px bold**. On the slide, 1 px = 0.75 pt, so 20 px ≈ 15 pt, readable from the back
  of the room.
- Colours (same as D1/D2): blue `#DCE8F5` fill / `#2B5C8A` line; orange `#FDE6CC` / `#D9822B`;
  data ellipses `#F3F4F6` / `#9AA0A6`; text `#1F2933`; arrows `#3E4C59`.

**Steps in draw.io** (app.diagrams.net → Device → Blank diagram):
1. **Page and grid.** File → Page setup → Custom, 1680 × 560. View → Grid on, so shapes snap.
2. **One box, then copies.** Drag a rounded rectangle, about 190 × 100. Style: fill
   `#DCE8F5`, line `#2B5C8A`, width 2. Double-click and type the name (bold, 24), Shift+Enter,
   then the method line (20). Ctrl+D to duplicate for the other boxes. Recolour boxes 5 and 8
   orange.
3. **Equal sizes.** Select all eight boxes → Arrange → set the same width and height (box 7 is
   the exception: make it about twice as wide, since it carries four operations).
4. **Rows.** Row 1 = boxes 1–6; row 2 = boxes 7–8. Select a row → Arrange → Align → Middle, then
   Distribute → Horizontally.
5. **Data ellipses.** Raw scan at the far left of row 1, Car detections at the far right of
   row 1, Completed cars at the far right of row 2.
6. **Arrows.** Hover over a box and drag its blue arrow onto the next box, so connectors stay
   attached when you move things. For the hand-off, connect *Car detections* → box 7 and set the
   connector to **orthogonal** (Edit style → `edgeStyle=orthogonalEdgeStyle`). It will route
   down and back.
7. **Phase groups.** A dashed rectangle with no fill around boxes 1–3 (*Segmentation: points →
   clusters*), 4–6 (*Detection: clusters → cars*), 7 (*Canonical-frame prep*) and 8
   (*Completion*). Arrange → To Back. Put the label on the top edge.
8. **Stage labels.** Left margin, bold: **DETECTION** for row 1 and **COMPLETION** for row 2.
   These are the two stages of the thesis title.
9. **Output labels.** Grey italic text under each box (table above).
10. **Training-source labels.** Small orange text with a short arrow up into boxes 5 and 8.
11. **Legend.** Bottom right: blue square "classical (training-free)", orange square "learned".
12. **Save and export.**
    - File → Save as → Device → `docs/defense/figures/d3_master_pipeline.drawio` (the editable
      source).
    - File → Export as → PNG: **Zoom 200%**, border 10, **transparent background off**. Save to
      `docs/defense/assets/d3_master_pipeline.png`.
    - Tell me when it's there: I'll wire it into slides 2.1 and THE END and check it at slide
      size.

**Self-check while drawing.** You should be able to answer each of these by pointing at D3:
1. Why does the geometric filter come *before* the classifier, and not the other way round?
   (Hint: it's cheap and removes most of the junk first. Geometric-only leaves 108,831 FP; the
   classifier cuts that to 4,303 [`tab:ablation-stage`].)
2. What's on the hand-off arrow, and why is the completion input a **single-frame** cluster,
   not a stack of frames? [ch3 §Completion Targets]
3. Why must box 7 come before PCN? Point to the two things the L-shape gate rejects
   (fragments and merges) and say what would happen to PCN's output without it.

---

## Part D — Glossary (EN → VI, starter set; correct to your department's usage)

| English | Tiếng Việt |
|---|---|
| point cloud | đám mây điểm |
| (self-)occlusion | (tự) che khuất |
| scan / frame | lượt quét / khung |
| ground plane | mặt phẳng mặt đất |
| clustering | phân cụm |
| classifier | bộ phân loại |
| tracking | theo vết (theo dõi) đối tượng |
| point cloud completion | hoàn thiện đám mây điểm |
| (manual) annotation | gán nhãn (thủ công) |
| ground truth | nhãn thực / dữ liệu chuẩn |
| end-to-end | đầu-cuối |
| modular pipeline | quy trình dạng mô-đun |
| cross-validation | kiểm định chéo |
| precision / recall / F1 | usually kept in English |
