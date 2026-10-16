# S3 — Geometric Filter, Car / Not-Car Classifier, Auto-Mined Labels

Study notes for slides 2.2.3–2.2.4 (steps 4–5 of D3: Filter → Classify), diagrams D6 and D7,
and the classifier ablations on 3.6.1 / 3.6.4. All facts come from the submitted thesis
(`docs/writing/thesis/`); thesis locations are in brackets. **[code]** = checked in `src/`, not
stated in the thesis. **[not in thesis]** and **[illustration]** mark my own additions; don't
quote those as results.

---

## Part A — Concepts

### A1. Geometric filter: cheap rules that drop obvious non-cars

**Core meaning.** Most clusters are not cars: trees, poles, facades, fragments. Before any
learning, each cluster's **oriented bounding box (OBB)** is checked against generous car-sized
limits. Anything clearly too big, too small, too tall or too thin is dropped [ch3
§Geometric Filtering, `fig:geo-filter`].

**How it's computed.** A cluster survives only if it passes **every** gate (frozen config):

| Gate | Rule | Rejects |
|---|---|---|
| Points | ≥ 15 | specks |
| Volume | 0.5–50 m³ | splinters, buildings |
| Longest side | ≤ 6.0 m | walls, long hedges |
| Size floors | longest ≥ 0.5 m, middle ≥ 0.4 m | slivers |
| Centre height | ≤ 1.5 m above the **fitted ground plane** | tall structures |
| Vertical span | ≤ 1.8 m | poles, trees |
| Aspect | longest / shortest ≤ 6.0 | thin walls, poles |

The OBB is Open3D's `get_oriented_bounding_box()` on the cluster **[code]**,
`_passes_geometric_filter` in `src/pipeline.py`. The height gate is measured from the RANSAC
plane, not a fixed z, so it tolerates a tilted sensor. When RANSAC returned no plane (the ~5%
wall frames, S2), it falls back to raw sensor z **[code]**.

**Worked example** [thesis `fig:geo-filter` numbers]:
- Car: longest 4.3 m, aspect 2.9, span 1.5 m → inside every gate → **accept**
- Pole or tree: span 4.0 m (> 1.8), aspect 10 (> 6) → **reject**
- Wall: longest 9 m (> 6), aspect 30 (> 6) → **reject**

**Why it's designed loose.** These are "generous car priors, not a tight car model". The
filter is **high recall, low precision by design**, and the fine decision is left to the
classifier [ch3]. The evidence [ch4 §recall diagnosis, `tab:ablation-stage`]:
- Seq 08: of **903,901** HDBSCAN clusters, **127,654 (14.1%)** pass.
- After the filter alone, recall is already **0.773** but precision only **0.199**: about 4
  of 5 survivors are still not cars.
- It costs some cars: **27,718** rejected clusters overlap a GT car. 73.9% of those were
  dropped by the minimum-volume gate and 12.7% by the minimum-points gate, i.e. **small
  clusters**. The thesis reads these as the splinters of cars HDBSCAN already split, and says
  that's an **inference**, not a measurement.
- Runtime: 36.2 ms per frame (5.8%) [`tab:runtime`].

### A2. PointNet: a network that doesn't care about point order

**Core meaning.** A point cloud is a **set**: the same car can arrive with its points listed in
any order (S1 A1, "unordered"). A classifier must give the same answer for every ordering.
PointNet does this with one trick [ch2 §Permutation Invariant Point Classification]:

f({x₁, …, xₙ}) = g( max over i of h(xᵢ) )

1. **h**, the *same* small network, turns each point into a feature vector, one point at a time.
2. **max**: for each feature, keep the largest value over all points. Max ignores order.
3. **g** turns that pooled vector into the answer.

**Worked example [illustration].** Three points, two features each:

| Point | h(x) |
|---|---|
| p₁ | (0.2, **0.9**) |
| p₂ | (**0.7**, 0.1) |
| p₃ | (0.4, 0.5) |
| **max per feature** | **(0.7, 0.9)** |

List them as p₃, p₁, p₂ and the max is still (0.7, 0.9). Gluing the vectors end to end instead
would give (0.2, 0.9, 0.7, 0.1, 0.4, 0.5), which changes when the order changes.

**Why it matters.** No sorting, no voxel grid, no image projection: the network reads the
cluster's points directly.

### A3. Our classifier: two branches, about 80,000 parameters

**How it's built** [ch3 §Car/Not-Car Classifier, `fig:classifier`]:
- **Point branch (shape):** 512 points (sampled or padded), normalised into the unit sphere →
  shared MLP 3 → 64 → 128 → 256 → max pool → a **256-number shape summary**.
- **Box branch (size):** 8 numbers from the OBB: the 3 sorted extents, volume, 2 aspect ratios,
  log(1 + point count), vertical span. They are z-score normalised, then a small MLP → **32
  numbers**.
- **Head:** concatenate (288) → 128 → 2 (car / not-car), with dropout.
- No T-Nets (PointNet's learned alignment), because clusters already sit in a consistent
  sensor frame.

**Why two branches. Worked example [illustration].** Unit-sphere normalisation means:
subtract the centroid, then divide by the largest distance from it.
- A real car, 4.3 × 1.8 × 1.5 m: largest radius ≈ 2.4 m (centre to corner), so divide by 2.4.
- A 1:10 toy car, 0.43 × 0.18 × 0.15 m: largest radius ≈ 0.24 m, so divide by 0.24.
- **After normalisation the two are identical.** The point branch alone can't tell them apart.
- The box branch still sees extents 4.3 / 1.8 / 1.5 versus 0.43 / 0.18 / 0.15. That's how real
  size gets back in: "a distant small car and a nearby one have the same normalised shape but
  different real extents" [ch3].

**Cost.** 33.7 ms per frame for ~45 clusters, ~0.75 ms each: **5.4%** of runtime, the only
learned stage in detection [`tab:runtime`].

**Why binary.** The first design was multi-class (car / bus / motorcycle). It was dropped
"once the data showed near zero real bus and motorcycle instances" [ch3].

### A4. Auto-mined training labels: zero additional manual annotation

**Core meaning.** Nobody labelled clusters by hand. The pipeline is run on SemanticKITTI
frames that **already have** per-point labels, and each cluster it produces inherits a label
from them [ch3 §Training data (mined, not hand labelled)]. Say "**zero additional manual
annotation**" or "auto-mined supervision", never "annotation-free". The labels come from
SemanticKITTI's existing ground truth.

**How it's computed:**
1. Replay the detection pipeline up to the geometric filter (preprocess, ground, cluster,
   filter) on each training frame. The thesis calls this "stages 1–5" but never numbers the
   stages, and on D3 the 5th box is Classify. **Don't say "stages 1–5" on the slide.** It
   sounds circular, as if the classifier labelled its own training data.
2. Copy the ground-truth label onto each surviving point by nearest neighbour.
3. Majority vote per cluster: **car** = SemanticKITTI `car` (10) or `moving-car` (252);
   **everything else is not-car**, including truck, bus, other-vehicle and unlabelled
   **[code]** (`SEMKITTI_TO_CLASS` in `src/mine_stage_b.py`).
4. Keep the cluster only if its majority has **purity ≥ 0.75**; mixed clusters are dropped.

**Worked example [illustration].**
- 200 points: 170 car, 30 road → purity 0.85 → **car**
- 100 points: 60 car, 40 vegetation → purity 0.60 → **discarded** (too mixed to trust)
- 80 points: 75 building, 5 car → not-car purity 0.94 → **not-car**

**Result:** **420,333** training clusters (sequences 00–07, 09, 10) and **130,394** validation
clusters (sequence 08), about **21% car / 79% not-car** [ch3]. (Raw counts 88,600 car /
331,733 not-car, 2,866 discarded **[not in thesis]**, session history 2026-06-02.)

**Imbalance.** Cross-entropy is weighted by inverse square-root class frequency, normalised to
mean 1; no resampling [ch3]. Worked out **[illustration from the thesis's 21/79]**: weights ≈
1.32 (car) vs 0.68 (not-car), so a car mistake costs ~1.9× a not-car mistake. Full inverse
frequency would make it 3.7×; the square root softens it.

**Training** [ch3]: 15 epochs, Adam (lr 1e-4, weight decay 1e-5), lr halves every 10 epochs,
batch 64. Augmentation: random rotation about the vertical axis and 5 mm Gaussian jitter. The
saved checkpoint is the epoch with the best **seq-08 validation macro-F1**. That selection is
why LOSO exists (S4).

**The thesis's key claim** [ch3]: the training clusters "are produced by the same clustering and
filtering the classifier meets at inference, including HDBSCAN splits", so it learns on the
distribution it must classify. **⚠ This is only approximately true; see Part B trap #8.**

### A5. Stage A (synthetic) vs Stage B (real): synthetic pretraining is redundant

**Core meaning.** The original recipe had two stages. **Stage A** pretrained on synthetic
ShapeNet partial renders; **Stage B** fine-tuned on the mined real clusters. The final
classifier skips Stage A and trains **from scratch on the real clusters only**. Stage A is kept
as an ablation and reported as a **negative result** [ch3; ch4 §Redundancy of synthetic
pretraining].

**Evidence** [ch4 `sec:ablation-stagea`, `tab:cross-domain`]:
- Classifier level (seq-08 validation clusters): **macro-F1 0.9285 from scratch vs 0.9225
  pretrained**, within run-to-run noise, scratch marginally ahead.
- Pipeline level: the two checkpoints are **identical in F1 to three decimals** on full seq 08.
- Cross-domain (car-class F1):

| Trained on ↓ / tested on → | Synthetic | Real |
|---|---|---|
| Synthetic (Stage A) | 0.999 | **0.000** |
| Real (Stage B) | **0.000** | 0.885 |

**Reading it:** the gap is **total and symmetric**. A model from one domain finds *no* cars in
the other, and fine-tuning on real data erases whatever Stage A learned. With 420,000 real
clusters, the real data does all the work.

**Don't mix the two metrics:** 0.9285 is **macro-F1** (the average of car-F1 and not-car-F1);
0.885 is **car-class F1**. Both are right; they measure different things.

### A6. What the evaluation says about filter + classifier

[ch4 `tab:ablation-stage`, full seq 08]:

| Configuration | Precision | Recall | F1 | FP |
|---|---|---|---|---|
| Geometric only | 0.199 | **0.773** | 0.316 | 108,831 |
| + classifier | **0.846** | 0.676 | 0.751 | 4,303 |
| + classifier, track filter | 0.911 | 0.732 | 0.812 | 2,513 |

- **The classifier is the precision mechanism.** It removes ~104,000 false positives, lifting
  precision 0.199 → 0.846.
- It pays with recall (0.773 → 0.676): being conservative, it drops borderline real cars.
  The track filter wins most of that back (S4).
- **The classifier isn't the recall limit.** Nothing after clustering can add a car, so
  recall is capped at 0.773 before the classifier ever runs (S2 A5, S4).

---

## Part B — Q&A traps for this session

1. **"Why PointNet and not PointNet++ or a bigger network?"**
   → "The job is small: one already-segmented cluster, car or not. The classifier isn't the
   bottleneck: recall is capped at 0.773 by clustering before it runs, and it costs 5% of
   runtime." **We did not test PointNet++**; don't imply we did. If pressed: "A stronger
   classifier can't recover cars that clustering already split; that's where the gain is."

2. **"So your labels aren't free. You used SemanticKITTI's ground truth."**
   → "Correct, and we say so. It's zero *additional* manual annotation: the labels are mined
   from existing SemanticKITTI annotations, with no new labelling. The point is that no one
   labelled clusters for this task." Never "annotation-free".

3. **"Why does it need the box branch?"**
   → "Normalising into the unit sphere removes size, so a toy car and a real car look the same
   to the point branch. The box branch puts real metres back in." [ch3]

4. **"Is a van or a truck a car?"**
   → "We use SemanticKITTI's own classes: car and moving-car are positives; truck, bus,
   other-vehicle are not-car." **[code]** Don't define "car" beyond SemanticKITTI's label.

5. **"Why 0.75 purity?"**
   → "It keeps clusters whose label is clear and drops mixed ones. It's a fixed choice; we
   didn't ablate it." (Not ablated in the thesis; don't claim a sweep.)

6. **"If synthetic pretraining didn't help, why is it in the thesis?"**
   → "Because it changes the recipe, and that's a result. It was the planned design and the
   advisor asked whether synthetic data was too clean **[not in thesis; Finding #25]**. We tested it: no gain at either level,
   and a model trained on one domain scores zero on the other." [ch4]

7. **"Does the geometric filter throw away real cars?"**
   → "Some: 27,718 rejected clusters overlap a GT car, mostly caught by the minimum-volume and
   minimum-points gates, so small clusters. Those look like splinters of cars HDBSCAN had
   already split; the thesis states that as an inference. Without any classifier, recall is
   already 0.773, so the filter is not what limits recall." [ch4]

8. **"Were the training clusters really mined with your final pipeline?"** ⚠ **Weak spot
   [not in thesis; verified from file dates and session history].**
   - The 420,333 clusters were mined **2026-06-02/03** (`dataset/stage_b/`, `metadata_*.json`).
     The production checkpoint `stage_b_scratch_best.pth` is dated **2026-06-25**.
   - The final config (0.10 m clustering grid, voxel-before-denoise, 300 RANSAC tries) was
     promoted **2026-07-23** (Finding #34). It was **never re-mined or retrained**.
   - `src/mine_stage_b.py` still denoises before downsampling, the old order **[code]**.
   - So the classifier trained on **pre-promotion clusters** (finer clustering, more split
     cars) and runs on **final-config clusters**. The thesis sentence "same clustering and
     filtering the classifier meets at inference" is true of the method, not literally of the
     shipped checkpoint.
   - **What protects the results:** every reported number (0.911 / 0.732 / 0.812, LOSO) was
     measured with this checkpoint *on the final pipeline*, so the numbers are honest
     measurements of what ships. Whether the mismatch **cost** anything is unknown: untested.
   - **Answer:** "The classifier was trained on clusters mined before we adopted the coarser
     clustering grid, so its training clusters are slightly more fragmented than what it sees
     now. All reported results use that classifier on the final pipeline, so the mismatch is
     already priced in. Re-mining at the final settings is a cheap improvement we haven't
     run."
   - Whether re-mining would help is **untested**. Settling it would take re-running
     `mine_stage_b.py` with the promoted order, retraining, and a full seq-08 evaluation; no
     re-runs during defense prep, so it stays a stated limitation.

9. **"Your validation set is seq 08 and you report on seq 08. Isn't that cheating?"**
   → "The checkpoint was picked on seq-08 validation clusters, so we ran leave-one-sequence-out
   to check. A classifier that never saw seq 08 gets recall 0.736 vs our 0.732; selection bought
   about 2 precision points and no recall." [ch4 `tab:headline`, `tab:loso`] (Details: S4.)

---

## Part C — Diagrams

### D6 + D7 merged (2026-10-16) — slide 2.2.4, one figure `assets/d67_classifier_training.drawio`
Top lane = TRAINING (D6), bottom lane = INFERENCE (D7), joined by an orange **trains** arrow from
"labelled clusters" into the classifier frame. **How they relate:** both take a pipeline cluster and
end in car / not-car. The top lane *looks the answer up* (SemanticKITTI labels, majority vote): the
answer key. The classifier *learns* to give the same answer from the points alone. On a new scan
there are no labels, so only the bottom lane runs.

### D6 — Auto-mined labels flow (user, draw.io) — slide 2.2.4, bottom
**Story (left → right):** SemanticKITTI scan *with* its existing per-point labels → run our
detection pipeline up to the geometric filter → clusters → label transfer by nearest neighbour
→ majority vote per cluster → purity ≥ 75%? → car / not-car clusters, or discarded → train.
**Must show:** the labels come **from SemanticKITTI** (not drawn by us), and the clusters come
**from our own pipeline**. Those two facts are the whole idea. Counts (420,333 train / 130,394
val, 21/79) are spoken, not drawn (user's choice).

### D7 — Simplified classifier (user, draw.io) — slide 2.2.4, top
**Story:** cluster → (point branch) normalise to 512 points in the unit sphere → shared MLP →
max pooling → 256-D shape feature; (box branch) 8 OBB features → MLP → 32-D size feature; →
classification head → car / not-car.
**Must show:** shared MLP + max pooling (why order doesn't matter), and the box branch (why
size survives normalisation). Reference: thesis `fig:classifier`. Use the standard terms as box
titles, with a plain-language detail line under each.

---

## Part E — Spoken answers (from the S3 quiz; VI drafts, correct to your department's usage)

**Why order doesn't matter (PointNet).**
- EN: "A point cloud is a set, not a list, so the same car can arrive in any order. PointNet
  handles that in two steps. First, every point goes through the same small network, so each
  point is described the same way wherever it sits in the list. Second, for each feature it
  keeps only the largest value over all points. A maximum doesn't depend on order: the tallest
  person in a room is the same however people line up. Shuffle the points and the answer stays
  the same."
- VI: "Đám mây điểm là một tập hợp, không phải một danh sách, nên cùng một chiếc xe có thể đến
  với các điểm theo bất kỳ thứ tự nào. PointNet xử lý bằng hai bước. Thứ nhất, mọi điểm đi qua
  cùng một mạng nhỏ, nên điểm nào cũng được mô tả theo cùng một cách, dù nằm ở vị trí nào. Thứ
  hai, với mỗi đặc trưng, mạng chỉ giữ giá trị lớn nhất trên tất cả các điểm. Phép lấy giá trị
  lớn nhất không phụ thuộc thứ tự, giống như người cao nhất trong phòng vẫn là người đó dù mọi
  người xếp hàng thế nào. Xáo trộn các điểm thì kết quả vẫn không đổi."

**Why the box branch.**
- EN: "Before the point branch sees a cluster, we scale it to fit a ball of fixed size. That
  makes shapes comparable, but it erases size: a real car 4 metres long and a toy car 40
  centimetres long look identical. The box branch gives the real dimensions back, so the
  classifier knows both what the object looks like and how big it really is."
- VI: "Trước khi vào nhánh điểm, mỗi cụm được co giãn cho vừa một hình cầu có kích thước cố
  định. Việc này giúp so sánh hình dạng, nhưng xoá mất kích thước: một chiếc xe thật dài 4 mét
  và một chiếc xe đồ chơi dài 40 cm trông giống hệt nhau. Nhánh hộp bao đưa kích thước thật trở
  lại, nên bộ phân loại biết cả hình dạng lẫn độ lớn thật của vật thể."

**Why a car splits (S2 A4).**
- EN: "HDBSCAN tries every radius, from large to small. If a car has a sparse band through it,
  at a small enough radius the car breaks in two along that band. Each half is still dense, so
  it holds together for a long time. HDBSCAN then asks: did the whole car hold together longer
  than its two halves did combined? If the halves win, it keeps the halves, and one car becomes
  two clusters. It only sees density; it has no idea what an object is."
- VI: "HDBSCAN thử mọi bán kính, từ lớn đến nhỏ. Nếu bên trong một chiếc xe có một dải điểm
  thưa, khi bán kính đủ nhỏ thì chiếc xe vỡ đôi tại dải đó. Mỗi nửa vẫn dày đặc nên giữ được
  rất lâu. HDBSCAN so sánh: cả chiếc xe giữ được lâu hơn, hay hai nửa cộng lại giữ được lâu hơn?
  Nếu hai nửa thắng, nó giữ hai nửa, và một chiếc xe thành hai cụm. HDBSCAN chỉ thấy mật độ,
  không biết thế nào là một vật thể."

---

## Part D — Glossary additions (EN → VI; correct to your department's usage)

| English | Vietnamese |
|---|---|
| oriented bounding box (OBB) | hộp bao có hướng |
| geometric filter / gate | bộ lọc hình học / ngưỡng |
| aspect ratio | tỉ lệ cạnh |
| classifier | bộ phân loại |
| permutation invariant | bất biến với hoán vị (thứ tự điểm) |
| shared MLP | MLP dùng chung (cho mọi điểm) |
| max pooling | gộp cực đại |
| unit sphere normalisation | chuẩn hoá vào hình cầu đơn vị |
| auto-mined labels | nhãn được khai thác tự động |
| majority vote / purity | bỏ phiếu đa số / độ thuần |
| class imbalance / class-weighted loss | mất cân bằng lớp / hàm mất mát có trọng số theo lớp |
| pretraining / fine-tuning | tiền huấn luyện / tinh chỉnh |
| domain gap (sim-to-real) | khoảng cách miền (mô phỏng – thực) |
| negative result | kết quả âm tính (kết quả không có cải thiện) |
| macro-F1 | F1 trung bình macro |
