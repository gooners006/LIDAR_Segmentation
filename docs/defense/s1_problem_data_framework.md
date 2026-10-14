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

---

## Part C — Diagrams D1–D3

### Deck-wide conventions (set once in Canva and reuse)

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

### D3 — Master pipeline (draw.io → SVG → Canva) — slides 5, 14, plus "you are here"

The thesis version is vertical. For a 16:9 slide, make it **horizontal**.

**Layout:**
```
 (Raw scan) → ┌─ SEGMENTATION: points → clusters ─┐  ┌─ DETECTION: clusters → cars ─┐  ┌ COMPLETION ┐ → (Completed cars)
              │ [Preprocess][Ground][Clustering]  │→ │ [Geo filter][Classifier][Track]│→ │ [Gate+norm][PCN] │
              └───────────────────────────────────┘  └────────────────────────────────┘  └────────────┘
               cleaned → non-ground → clusters        candidates → cars → tracked cars    partial → full car
```

**Steps in draw.io** (app.diagrams.net → "Device" storage → Blank diagram):
1. **Grid.** View → tick Grid. Shapes snap to it, which keeps everything aligned.
2. **First stage box.** From the left panel, drag a rounded rectangle. In the right panel
   (Style): fill `#DCE8F5`, line `#2B5C8A`, line width 2. Double-click to type
   "**Ground removal**", press Shift+Enter, then type "RANSAC plane" in a smaller font.
3. **The other boxes.** Ctrl+D to duplicate and edit the text. Make the Classifier and PCN boxes
   orange (`#FDE6CC` / `#D9822B`). Keep all boxes the **same size**: select them all, then in
   Arrange set equal width and height.
4. **Alignment.** Select a row → Arrange → Align → Middle, then Arrange → Distribute →
   Horizontally.
5. **Arrows.** Hover over a box, drag the small blue arrow onto the next box. Draw.io makes a
   connector that stays attached when you move boxes.
6. **Phase groups.** Drag a rectangle around each group, set fill to none and line style to
   dashed, then Arrange → To Back. Put the phase label on its top edge.
7. **Outputs.** Small grey italic text under each box: "→ non-ground points", etc. These are the
   I/O contracts, copied from ch3 Fig `fig:pipeline`.
8. **Ellipses for data.** Use ellipses for "Raw scan" and "Completed cars", with grey fill. The
   convention is ellipse = data, box = processing.
9. **Legend.** Bottom right: a blue square "classical (training-free)" and an orange square
   "learned".
10. **Export.** File → Export as → SVG, with "Transparent background" ticked. In Canva:
    Uploads → upload the SVG. It stays sharp at any size.
11. **"You are here" reuse.** On slides 6–12, shrink the SVG to a strip along the top and place
    a semi-transparent red rounded rectangle over the current stage. Draw one diagram and use it
    seven times. It also keeps a non-specialist audience oriented.

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
