# S5 — Completion Method: PCN, Synthetic Partials, Normalisation, L-Shape Gate, Length

Study notes for slides 2.3 (completion pipeline), 2.3.1 (PCN), 2.3.2 (synthetic training
partials) and 2.3.3 (L-shape gate, normalisation, length); diagram D11. All facts come from the
submitted thesis (`docs/writing/thesis/` ch3 §Object Completion, ch4 §Completion Ablations);
thesis locations are in brackets. **[code]** = checked in `src/`, not stated in the thesis.
**[measured, not in thesis]** = counted from saved outputs this session; **[illustration]** = my
own example. Don't quote those as thesis results.

---

## Part A — Concepts

### A1. What gets completed, and why one frame

**Core meaning.** Completion takes **one detected car, as seen in one scan** (one-sided, partial)
and predicts the whole car surface [ch3 §Completion Targets].

**Which frame [code].** For each car track, `src/main.py` picks the track's **densest frame**
(most points), moves it back into that frame's sensor coordinates (ego at the origin, gravity
up), completes it there, then maps the result back to the world frame for saving.

**Why not stack all frames of the track?** "Accumulating a moving car over many frames using a
simple centroid tracker introduces significant motion smear": small centroid shifts stretch the
car beyond its true size [ch3]. One frame is "one clean, if partial, observation". (Stacking
*static* cars is exactly what the thesis uses as the **upper reference** in the plausibility test,
the static-accumulation ceiling, S6.)

### A2. PCN: the completion network

**Core meaning.** PCN (Yuan et al. 2018) is an **encoder–decoder for point clouds**: it squeezes
the partial into one shape summary, then grows a full car from that summary [ch3 §The Completion
Network, `fig:pcn`]. Used **off the shelf, unmodified**; the architecture is not a contribution.

**How it's computed** [ch3; `src/pcn.py` **[code]**]:
1. **Input:** the partial, resampled to **256 points**.
2. **Encoder (stacked PointNet):** a shared per-point MLP (3 → 128 → 256), max pool → a global
   vector; that vector is **copied back onto every point** (each point now knows its local
   feature *and* the whole shape), second MLP (512 → 1024), max pool → one **1024-number shape
   code**. Same max-pool idea as our classifier (S3 A2).
3. **Decoder, coarse-to-fine:** a fully connected layer predicts **1,024 coarse seed points** (a
   rough skeleton). Then **grid folding**: each seed gets a tiny 2 × 2 grid patch, and an MLP
   "folds" it into local surface → 1,024 × 4 = **4,096 points** (`PCN_GRID_SIZE = 2`).
4. **Loss:** Chamfer distance on the coarse output + 0.5 × Chamfer on the fine output. The fine
   term is down-weighted so the skeleton is learned first.

**Chamfer distance, worked [illustration].** For each point in cloud A, the distance to its
nearest point in B, averaged; then the same from B to A; add the two.
- A = {(0, 0), (1, 0)}, B = {(0, 0.1), (1, 0.3), (2, 0)}
- A → B: 0.1 and 0.3 → mean **0.2**
- B → A: 0.1, 0.3 and 1.0 (the point (2, 0) has nothing near it) → mean **0.467**
- CD = 0.2 + 0.467 = **0.667**. The extra far point in B is punished by the B → A direction.
  That's why both directions are needed: one direction alone can be fooled by a cloud that covers
  only part of the target. **[code]** evaluation CD = sum of the two mean distances
  (`chamfer_distance`, `src/completion.py:596`).

**Why PCN and not something newer?** Controlled swap with PoinTr: synthetic Chamfer only
**0.161 → 0.153 m**, "too small to justify the heavier model", and the real-data gap is a centre
estimation limit that no architecture fixes [ch4 §Completion architecture].

### A3. Training data: partials that look like real scans

**Core meaning.** PCN learns from **synthetic** ShapeNet meshes: input = a partial view, target =
the complete shape. The partials must look like what the real pipeline delivers, or the network
sees something different at test time [ch3 §Training Data: KITTI-Like Partials,
`fig:kitti-like-partial`].

**What a naive generator got wrong** (derisk study vs real seq-08 clusters):
- a dense pinhole depth render instead of LiDAR scan lines;
- no 0.05 m voxelisation and no ground removal (the real input has both);
- noise growing with range (up to 0.15 m at 30 m) that washed out the scan-ring banding.

**The KITTI-like generator** [ch3]:
- orient the mesh to gravity; cast **one HDL-64E viewpoint** (64 beams, 0.09° horizontal
  resolution) from a sensor **1.73 m** high, at a range of **8–30 m**;
- small absolute noise **0.015 m**; voxelise at **0.05 m** (as the pipeline); cut points within
  **0.30 m** of the ground (as ground removal does);
- target = a dense sample of the complete mesh.

**Training** [ch3; checkpoint metadata **[code]**]: 100 epochs, Adam (lr 1e-4, weight decay 1e-6),
lr halved every 40 epochs, batch 8, 256-point input, 4,096-point target. **No real cars in
training.** Kept: best validation loss (**0.1246**; the checkpoint stores epoch index 70
**[code]**). The loss is in
unit-sphere units, not metres.

**⚠ What "ShapeNet" contains here: see trap #9** (cars, buses and motorcycles).

### A4. Inference normalisation: putting a real scan into the training frame

**Core meaning.** In training every shape sat in a fixed **canonical car frame**: X = width,
Y = up, Z = length, centred on the **true car centre**, scaled by the **true car radius**. A real
partial must be put into that same frame, or the network gets an input it never saw. The hard
part: a one-sided partial doesn't show the true centre or the true size [ch3 §Inference-Time
Partial Normalisation].

**How it's computed** (`estimate_canonical_frame`, `src/completion.py:438` **[code]**):
1. **Rotate:** gravity → up; the L-shape heading (A5) → length. **No PCA.**
2. **Centre**, starting from the middle of the partial's bounding box:
   - **down 0.25 m**: ground removal cut the car's lowest ~0.3 m, so the box sits too high
     (the largest correction);
   - **width:** if the observed width is under the **1.9 m** prior, push the centre *away from
     the sensor* by the missing half: 0.95 m − observed width / 2;
   - **length:** the same push along length, toward the far end, up to the per-car length (A6).
3. **Scale:** divide by (largest point-to-centre distance ÷ **1.137**). 1.137 is the median ratio
   of the partial's radius to the true radius, calibrated on synthetic cars.
4. Run PCN on the 256-point normalised input; undo the scale and centre; rotate back.

**Worked example [illustration]:** a car parked alongside us, its near side 10 m to our left.
The scan sees one long side: observed length 4.2 m, observed width 0.2 m.
- Box centre across: 10.1 m (the middle of 10.0–10.2 m). Missing width: 0.95 − 0.1 = **0.85 m**,
  pushed away from us → **10.95 m**. A real 1.9 m-wide car with its near side at 10.0 m has its
  centre at 10.95 m. ✓
- Height: the box middle is 0.25 m too high → **lowered by 0.25 m**.
- Scale: if the partial's largest distance from that centre is 2.79 m, the radius used is
  2.79 / 1.137 = **2.45 m**, about half the diagonal of a 4.3 × 1.9 × 1.5 m car.

**Why it matters (the key lesson of the completion work)** [ch4 §Inference normalisation,
§Sim-to-real fine-tuning]:
- With a naive frame (3D PCA, the partial's own centroid and radius) the **same checkpoint** gives
  mis-oriented, mis-scaled **blobs**. 3D PCA picks an arbitrary axis instead of gravity.
- The blobs were first blamed on a sim-to-real domain gap, but noise augmentation and fine-tuning
  on real data **didn't fix them** (a reported negative result). The cause was the input
  distribution and this normalisation, not the weights (Finding #26: a train/inference
  normalisation mismatch that cost several sessions; backup slide 57).

### A5. The L-shape gate: only whole, single cars get completed

**Core meaning.** Some detected "cars" are not one whole car: an HDBSCAN **fragment** (part of a
car) or a **merge** (two cars, or a 90° heading error). Completing those gives junk, so a gate
keeps them out [ch3 §L-Shape Gating, `fig:lshape-gate`].

**How it's computed** (`_lshape_axes`, `src/completion.py:402` **[code]**; Zhang et al. 2017):
1. Look at the cluster from above (its ground footprint).
2. Try rectangle orientations from 0° to 89°, 1° apart. For each, measure every point's
   distance to the nearest rectangle edge; score = sum of 1 / distance (points hugging the edges
   score high).
3. The best-scoring rectangle gives the heading; its **longer side = length**, shorter = width.
4. **Reject as a fragment** if length < **2.7 m**; **reject as a merge** if width > **2.3 m**.

**Worked example** [thesis `fig:lshape-gate` numbers]:

| Footprint | Fitted L × W | Decision |
|---|---|---|
| single car | 4.3 × 1.8 m | accept |
| fragment | 2.0 m long | reject: fragment (2.0 < 2.7) |
| two cars side by side | width 3.2 m | reject: merge (3.2 > 2.3) |

**Why L-shape and not PCA?** On a near-square or two-faced footprint PCA's axes are ambiguous;
the edge-hugging score stays well defined. Honest secondary finding: on real data the L-shape
*heading* is **neutral** against PCA; the fit "earns its place by detecting bad inputs, not by
giving a better heading" [ch3].

**Evidence** [ch3, ch4]: on seq 08 "every implausible completion originated from a fragment or
merge input"; the gate "carries a measured cost, dropping a fraction of otherwise usable cars".
**[code comment, not thesis]** The thresholds come from a seq-08 A/B on 47 tracks (0/15 plausible
below 2.7 m, 0/7 above 2.3 m). The thesis gives no count of false rejections.

**How many cars actually get completed? [measured, not in thesis]** From the saved plausibility
run (`output/experiments/plausibility_gen`, all 11 seqs, 5,496 car tracks):
- **2,822 completed (51%)**: this is the N of the plausibility test;
- **2,022 rejected** with ≥ 64 input points (the gate, or a degenerate fit) **(37%)**;
- **652 too few points** (< 64) **(12%)**.

### A6. Per-car length: reaching the hidden far end

**Core meaning.** A one-sided partial is cut off at the far end, so it is normalised around its
near part, and PCN's output stops short of the far end. The centre is pushed along the length
toward the far end, up to an estimated full length [ch3 §Per-Car Length Estimation].

**How it's computed** (`track_length_estimate` **[code]**):
- Fit the L-shape in **every frame of the track** that passes the gate; take the **90th
  percentile** of those fitted lengths **+ 0.12 m** bias correction.
- If the track has **fewer than 5** gate-passed frames, fall back to a fixed **4.14 m** (the
  median car length).
- 90th percentile, not the maximum, because one bad fit would set the maximum. 4.5 m was tried
  and rejected: it over-stretched compact cars.

**Worked example [illustration]:** gate-passed fits 3.6, 3.9, 4.0, 4.1, 4.2, 4.3 m → 90th
percentile ≈ 4.25 → + 0.12 = **4.37 m**. A car seen in only 3 gate-passed frames gets 4.14 m.
If the observed length is 3.0 m: push = (4.37 − 3.0) / 2 ≈ **0.69 m** toward the far end
(extend only: never pushed backwards).

**The fallback is not rare** (stated as a limitation [ch3, ch4]; the thesis says "non-trivial
fraction" but gives no number). **[measured, not in thesis]** In the plausibility run: **119 of
518 (23%)** completed cars on seq 08, **843 of 2,822 (30%)** over all 11 seqs used the 4.14 m
fallback.

### A7. What comes out

4,096 points, rotated back to the scan (then to the world frame for saving). **Points only**:
no box is output or scored (weak spot in `project_state.md`: ch3 figures say "dense surface +
box"). The thesis `fig:completion-pipeline` is no longer on the deck.

---

## Part B — Q&A traps for this session

1. **"PCN is from 2018. Why not a modern network?"**
   → "We tested PoinTr in a controlled swap: synthetic Chamfer 0.161 → 0.153 m, a small edge for
   a much heavier model, and on real data the limit is estimating the car's centre, which no
   network fixes. The architecture isn't our contribution; the training data and the
   normalisation are." [ch4]

2. **"It's trained only on synthetic data. Isn't there a domain gap?"**
   → "Two bridges: the training partials are generated to look like real scans (one HDL-64E view,
   5 cm voxels, ground cut), and a real scan is put into the same frame the network trained in.
   Retraining on noisier, more LiDAR-like simulated scans didn't help: the early blobs came from the
   normalisation, not the domain." [ch3, ch4; findings #16/#17/#19. Not "fine-tuning on real data":
   that was never run, see S6 notes A8 #2.]

3. **"What is 1.137?"**
   → "The typical ratio between a one-sided partial's radius and the true car radius, measured
   on synthetic cars. Dividing by it puts a real partial at the scale the network trained on."
   (Calibrated on synthetic val cars, Finding #26 **[code comment]**.)

4. **"Your width prior is 1.9 m. What about vans or small cars?"**
   → "It only sets where the centre goes when the far side is hidden; the network then shapes the
   car. It's a prior, so very wide or narrow cars get a slightly shifted centre." Not measured per
   car width; don't claim robustness numbers.

5. **"Why not use all frames of the track?"**
   → "With a centroid tracker, stacking frames of a moving car smears it. One frame is clean. We
   do use stacked parked cars as the upper reference in the evaluation." [ch3]

6. **"Which frame do you complete?"** **[code]** "The densest frame of the track." Follow-up trap:
   it is chosen *before* the gate, with no fallback to another frame, so a car followed from
   directly behind (only ~1.8 m visible) is rejected as a fragment (weak spot in
   `project_state.md`, L-shape gate vs I-shaped scans).

7. **"How many detected cars get completed?"** **[measured, not in thesis]** "About half: 2,822 of
   5,496 car tracks over the eleven sequences. The rest are rejected by the gate as fragments or
   merges, or have too few points." Use only if asked.

8. **"How often does the length fall back to the fixed prior?"** The thesis promises the number
   (ch3: "the exact fallback rate on sequence 08 is reported with the completion results") but
   ch4 only says "a non-trivial fraction". **[measured]** 23% on seq 08 (119 / 518), 30% overall.
   → "About a quarter of completed cars on seq 08 had fewer than five good frames, so they used
   the median length, 4.14 m."

9. **"Was PCN trained on cars?"** ⚠ **Weak spot [code + checkpoint metadata, not in thesis].**
   The thesis says ShapeNet cars (ch3 "the ShapeNet car"; `tab:dataset-roles` "~3,150 models (Car
   category)"). `src/train_pcn.py` `TRAIN_CONFIG["categories"]` holds **car, bus and
   motorcycle**, added 2026-05-06 and never removed; nothing on the command line overrides it, and
   the config snapshot stored in `pcn_kitti_best.pth` (written by the training run, 2026-06-26)
   lists all three. Rebuilding the dataset with that config gives **train 3,835 = 2,813 cars
   (73%) + 752 buses + 270 motorcycles; val 955 = 701 + 187 + 67** (measured). So the production
   PCN was almost certainly trained on all three; the synthetic CD 0.16 m / F 0.76 is likely
   over the mixed val set too (inferred: `evaluate` uses the full val loader). The "~3,150 car
   models" figure doesn't match any count here (3,533 car folders, 3,514 usable). Effect on car
   completion: **unknown**, untested (every shape is scaled into the unit sphere, and completion
   is only ever run on cars).
   → If asked: "PCN trained on ShapeNet vehicle meshes, about three quarters cars, plus buses and
   motorcycles left from the early multi-class design. Every shape is normalised the same way,
   and completion is only applied to detected cars; the plausibility test then checks the output
   on real cars directly."

10. **"Isn't the judge biased if PCN and the judge both learned from CAD models?"** (S6 detail)
    → "The judge is trained on ModelNet40, a different repository from ShapeNet." [ch4]

11. **"Your figure says the output is a surface plus a box. Where is the box?"** Known weak spot
    (`project_state.md`). → "The completion estimates a full-car frame that implies a box, but the
    evaluated output is the point cloud; boxes aren't scored."

13. **"How was PCN trained: any augmentation?"** ⚠ **Weak spot [code, measured; not in
    thesis].** `_augment_rotation` (`src/train_pcn.py`, in the code since 2026-05-06, so in the
    2026-06-26 production run) rotates every training sample about its **Z** axis, written as if Z
    were up. But ShapeNet cars have **Y = up, Z = length** (measured on 6 meshes: Z longest, Y
    smallest), so each training car was **rolled about its length axis by a random angle**. ch3
    says PCN "is trained on shapes expressed in a fixed canonical car frame (X = width, Y = up,
    Z = length)": only the length axis was fixed. Inference feeds upright cars (roll 0°, inside the
    training range), consistent with completion working; whether the random roll cost quality is
    **untested** (would need a retrain without it).
    → If asked: "Random rotation about the car's length axis; the length axis is the one the
    normalisation fixes, and real cars arrive upright, which the network has seen." Don't claim
    "a fixed up axis in training".

12. **"Were these constants tuned on the test data?"**
    → "The completion constants were developed on seq 08, so the plausibility test runs on all
    eleven sequences: the gain holds on every one, and seq 08 is unremarkable among them." [ch4
    §plausibility-seq] The gate thresholds came from 47 seq-08 tracks **[code comment]**; don't
    volunteer that.

---

## Part C — Diagrams and slides

### D11 — PCN training vs inference (user, draw.io) — slide 17 (2.3, ★ talk)
Scaffold: `assets/d11_pcn.drawio` (from `scripts/make_d11_scaffold.py`).
**Story:** top lane = TRAINING on synthetic shapes: ShapeNet mesh → simulated LiDAR view → PCN →
Chamfer loss against the full mesh sample. Bottom lane = INFERENCE on a real scan: real car
cluster → L-shape gate → canonical frame → the same PCN (frozen) → rotate back → completed car.
Orange dashed "same weights" arrow between the two PCN boxes. Orange line: "The bridge: training
inputs look like real scans, and real scans are put into the training frame."
**Fill the "?":** range **8–30 m**, voxels **0.05 m**; PCN **256** in → **1,024** coarse →
**4,096** out; gate length ≥ **2.7 m**, width ≤ **2.3 m**; width prior **1.9 m**, scale ÷
**1.137**; completed car **4,096** points.
**Wording kept deliberately:** "ShapeNet mesh", "synthetic shapes", not "car" (trap #9).
Export: `assets/d11_pcn.drawio.png` (Zoom 200%, Transparent off, Appearance Light, diagram
embedded); exported 2026-10-18, on slide 17.

### Support slides (order follows D11: training lane, then inference lane)
- Slide 19 (2.3.2 Real Scan Normalisation): `assets/d15_check_turn_shift_size.png` from
  `scripts/make_d15_normalisation.py`: the production geometry code on car C's exact production
  scan (checked: reproduces the saved fit 3.684 × 1.518 m). **Check** (fit 3.7 × 1.5 m, passes) →
  **Turn** → **Shift** (0.19 m away from the sensor, 0.23 m toward the unseen end, 0.25 m down) →
  **Size** (÷ 1.87 m = largest distance ÷ 1.137; points may poke past the circle). Length shown
  with the 4.14 m prior; production used this car's per-car estimate (not stored), a few cm
  different at most. Memory hook: check, turn, shift, size. (The Fig. 3.7 crop,
  `thesis_completion_pipeline.png`, is no longer on the deck.)
- Slide 18 (2.3.1 Synthetic Training Partials): `assets/d14_training_vs_real.png` from
  `scripts/make_d14_partials.py` [illustration]: simulated scan (the training code's own
  generator, thesis mesh) → full 3D model (the training answer; ShapeNet meshes include the
  interior, hence the seats) | real scan (car C, slide 35). Point: the training input looks like
  the real scan. The old "naive partial" panel was dropped (Q&A only, trap #2).
- Slide 20 (2.3.3 Gate and Normalisation Constants, close-up of slide 19): `assets/thesis_lshape_gate.png` = Fig. 3.10, plus a native table of
  `tab:completion-config`.

---

## Part E — Spoken answer: "How was PCN trained?" (2026-10-18)

Every sentence is true and none raises traps #9 or #13. Don't say "only cars" or "a fixed
upright frame".

- EN (≈ 35 s): "PCN was trained entirely on synthetic data, because a real scan never shows the
  complete car, so there is nothing to learn the hidden side from. We took 3D vehicle models from
  ShapeNet, and for each one we simulated what our own sensor would see: a single Velodyne
  viewpoint, eight to thirty metres away, with the same five-centimetre grid and the same ground
  cut as our real pipeline. That one-sided scan is the input. The answer is the complete model, as
  4,096 points spread over its surface. The network learns to turn one into the other by
  minimising the Chamfer distance, which measures how far the predicted points are from the true
  surface, in both directions. We trained for 100 epochs and kept the epoch with the best
  validation loss. No real scan is used in training: afterwards the weights are frozen, and each
  real car is put into the same frame before it is completed."
- VI: "PCN được huấn luyện hoàn toàn trên dữ liệu tổng hợp, vì một bản quét thực không bao giờ
  cho thấy toàn bộ chiếc xe, nên không có gì để học phần bị che khuất. Chúng tôi lấy các mô hình 3D
  phương tiện từ ShapeNet, và với mỗi mô hình, chúng tôi mô phỏng những gì cảm biến của mình sẽ
  thấy: một góc nhìn Velodyne duy nhất, cách từ tám đến ba mươi mét, cùng lưới năm xăng-ti-mét và
  cùng bước cắt mặt đất như pipeline thực. Bản quét một phía đó là đầu vào. Đáp án là mô hình hoàn
  chỉnh, gồm 4.096 điểm phân bố trên bề mặt. Mạng học cách biến cái này thành cái kia bằng cách
  cực tiểu hoá khoảng cách Chamfer, đo độ lệch giữa các điểm dự đoán và bề mặt thật theo cả hai
  chiều. Chúng tôi huấn luyện 100 epoch và giữ epoch có loss kiểm định tốt nhất. Không có bản quét
  thực nào được dùng để huấn luyện: sau đó trọng số được cố định, và mỗi chiếc xe thực được đưa về
  cùng hệ toạ độ trước khi hoàn thiện."

Follow-ups (one or two sentences each):
- **Why synthetic?** Real LiDAR only sees the near side, so there's no complete answer to train
  against; a 3D model gives the complete shape for free.
- **How do you know it learned?** Held-out validation models: Chamfer 0.16 m, F-score 0.76 at
  10 cm [ch4]. Real cars: the plausibility test (S6).
- **Why simulate the sensor?** A simpler dense render didn't look like LiDAR (no scan lines, grid
  or ground cut); matching those makes the training input look like the detector's output.
- **Augmentation?** "Random rotations of the training shapes; at test time each real car is lined
  up along its length the same way the training models are." (True; see trap #13.)
- **How many models / split?** "A few thousand ShapeNet models, split 80/20 by model."
- **Why 256 points?** Fixed input size; in training the input is randomly thinned to as few as 32
  points, so it copes with sparse cars **[code: `partial_min_points`]**.
- **Coarse/fine loss?** A rough 1,024-point skeleton first, refined to 4,096; the rough stage is
  weighted higher (fine × 0.5), so the overall shape is learned first.

---

## Part D — Glossary additions (EN → VI; correct to your department's usage)

| English | Vietnamese |
|---|---|
| point cloud completion | hoàn thiện đám mây điểm |
| encoder / decoder | bộ mã hoá / bộ giải mã |
| shape code (global feature) | mã hình dạng (đặc trưng toàn cục) |
| coarse-to-fine | từ thô đến mịn |
| grid folding | gấp lưới |
| Chamfer distance | khoảng cách Chamfer |
| synthetic partial (view) | góc nhìn một phần tổng hợp |
| domain gap / sim-to-real | khoảng cách miền / từ mô phỏng sang thực |
| canonical frame | hệ toạ độ chuẩn |
| normalisation (centre, scale) | chuẩn hoá (tâm, tỉ lệ) |
| prior (width / length prior) | giá trị tiên nghiệm (bề rộng / chiều dài) |
| L-shape fitting | khớp hình chữ L |
| fragment / merge | mảnh vỡ / gộp |
| input-quality gate | cổng kiểm tra chất lượng đầu vào |
| fallback | giá trị dự phòng |
