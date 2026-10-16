# S2 — Preprocessing, Ground Removal, Clustering

Study notes for slides 2.2.1–2.2.2 (steps 1–3 of D3: Preprocess → Remove ground → Cluster),
diagrams D4 and D5, and the clustering half of the recall story (S4 finishes it). All facts come
from the submitted thesis (`docs/writing/thesis/`); thesis locations are in brackets.
**[code]** = checked in `src/pipeline.py`, not stated in the thesis. **[not in thesis]** and
**[illustration]** mark my own additions; don't quote those as results.

---

## Part A — Concepts

### A1. Preprocessing: three cleaning steps

**Core meaning.** Before looking for objects, make the scan smaller and cleaner: drop points
that can't matter, even out the density, and remove isolated specks [ch3 §Preprocessing].

**How it's computed** (frozen configuration, in this order):

| Step | Rule | Why |
|---|---|---|
| 1. Height filter | keep points with z > −2.0 m (sensor frame) | cheaply drops the bulk of sub-road noise before the expensive steps |
| 2. Voxel downsample | one point per 0.05 m cube | evens out range-dependent density; ~10× fewer points |
| 3. Statistical outlier removal | drop a point if its mean distance to its 20 nearest neighbours is > mean + 2.0 × std (over all points) | removes isolated speckle |

The order of steps 2 and 3 **changes the result**: downsampling first means the outlier test
runs on ~10× fewer, regularised points, so different points survive. This order was adopted
because it improved detection, and the thesis flags it as behaviour-changing rather than
pretending the orders are equivalent [ch3 §Preprocessing]. The rule in step 3 is Open3D's
`remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)` **[code]**; the thesis words it as
"2.0 standard deviations of the local distribution".

**Worked example: voxel downsampling** [ch2 §Voxel Grid Downsampling, formula
vox(x) = ⌊(x − x_min)/v⌋]. Take v = 0.05 m and x_min = (0, 0, 0) **[illustration]**:

| Point (m) | (x−x_min)/v | Cell |
|---|---|---|
| (1.02, 0.31, 0.12) | (20.4, 6.2, 2.4) | (20, 6, 2) |
| (1.04, 0.34, 0.14) | (20.8, 6.8, 2.8) | (20, 6, 2) ← same cell |
| (1.07, 0.31, 0.12) | (21.4, 6.2, 2.4) | (21, 6, 2) ← next cell |

The two points in cell (20, 6, 2) are replaced by their centroid (1.03, 0.325, 0.13). A near car
that had 50 laser hits inside one 5 cm cube now has one point there; a far car with one hit per
cube keeps it. That's how the grid **evens out density**.

**Worked example: outlier removal [illustration].** Suppose the mean neighbour distances over
the scan average 0.08 m with std 0.03 m. The cut is 0.08 + 2.0 × 0.03 = 0.14 m. A point on a car
body (mean neighbour distance 0.06 m) stays; a lone speck in the air (0.50 m) goes.

**Why it matters.** Every later step is cheaper on a smaller cloud, and HDBSCAN is superlinear
in point count. The 0.05 m grid here is *not* the clustering grid: clustering re-voxelises to
0.10 m (A4).

### A2. Ground removal with RANSAC

**Core meaning.** After preprocessing, the biggest structure left is the road. If it stays, it
connects every car, pole and wall into one giant cluster. So find the road plane and delete its
points [ch3 §Ground Plane Removal].

**How it's computed** [ch2 §Ground Plane Removal by RANSAC; ch3]:
1. Pick **3 random points**. Three points define a plane.
2. Compute the plane: normal **n** = (p₁−p₀) × (p₂−p₀), scaled to length 1; offset
   d = −**n**·p₀.
3. Score it: count **inliers**, the points with |**n**·x + d| < 0.2 m.
4. Repeat **300 times**; keep the plane with the **most inliers**.
5. **Safety check:** the kept plane's normal must be near vertical (z-component > 0.5). If it
   isn't (RANSAC found a wall or a truck side), **nothing is removed** and the whole cloud goes on
   as objects. Rare, but it stops a bad fit from deleting real cars.
6. The kept plane's inliers are removed as ground.

Implementation details **[code]**: fixed seed 42, so every run is identical; the normal is
flipped to point up before the check, because a cross product can point either way.

**Worked example: one RANSAC try [illustration].** Sample p₀ = (0, 0, 0), p₁ = (4, 0, 0.1),
p₂ = (0, 4, 0).
- u = p₁ − p₀ = (4, 0, 0.1), v = p₂ − p₀ = (0, 4, 0)
- **n** = u × v = (0·0 − 0.1·4, 0.1·0 − 4·0, 4·4 − 0·0) = (−0.4, 0, 16); length ≈ 16.005;
  unit **n** ≈ (−0.025, 0, 0.9997); d = 0
- Point (2, 1, 0.10): distance = |−0.025·2 + 0.9997·0.10| = |−0.05 + 0.09997| ≈ **0.05 m → inlier**
- Point (2, 1, 0.30): |−0.05 + 0.2999| ≈ **0.25 m → not an inlier** (25 cm above: the bottom of
  a car)
- Check: n_z = 0.9997 > 0.5 → **accepted as ground**
- A wall at x = 5 has **n** = (1, 0, 0), n_z = 0 → rejected by the check

**Why 300 iterations is enough** [ch2 gives the formula; the numbers are my illustration].
T = ⌈log(1−p) / log(1−w³)⌉ is how many tries you need to draw one all-ground sample with
probability p, when a fraction w of points are ground (s = 3):
- w = 0.5, p = 0.99 → T = ⌈log 0.01 / log 0.875⌉ = ⌈34.5⌉ = **35**
- w = 0.3 → **169**
- Solving for 300 tries at p = 0.99 gives w ≈ 0.25: 300 iterations are 99%-safe only when at
  least ~25% of the cleaned points are road **[inference]**. **19% of seq-08 frames have less
  road than that**, and in ~1 frame in 5, 1000 tries would still noticeably improve the plane.
  See `qa_backup/ransac_seq08.md` (private; not thesis numbers).
- The thesis's evidence is empirical: the iteration count was cut from 1000 to 300 for speed
  "after confirming the plane fit converges well before 1000 iterations on this data" [ch3].

**Why a single plane, and its limit.** It assumes the local ground is roughly flat. That holds
on the urban SemanticKITTI sequences but would degrade on strong slopes. A grid-based ground model
(Patchwork++-style) is named as future work, not claimed as solved [ch3].

### A3. DBSCAN: why a fixed radius fails on LiDAR

**Core meaning.** DBSCAN calls a point a **core point** if at least m points lie within radius
ε of it. Clusters are chains of core points [ch2 §Density-Based Clustering].

**The problem.** LiDAR density falls with range (S1 A1: neighbouring spots ~0.9 cm apart at
6 m, ~6.3 cm at 40 m, horizontally). Pick ε small enough to keep two parked cars apart at 5 m,
and a car at 40 m crumbles into pieces. Pick it large enough to hold the far car together, and
near cars fuse. "A single Euclidean radius that merges nothing at 5 m will over-segment at
40 m" [ch3 §Clustering].

### A4. HDBSCAN: clustering without a fixed radius

**Core meaning.** HDBSCAN tries *every* radius at once, builds the whole hierarchy of how points
merge as the radius grows, and keeps the clusters that **survive the longest** (the most
*stable*) [ch2 §Density-Based Clustering].

**How it's computed:**
1. **Core distance** of a point = distance to its m-th nearest neighbour (here m =
   min_samples = 5). Small in dense regions, large in sparse ones.
2. **Mutual reachability distance** between a and b = max(core(a), core(b), ‖a−b‖). Sparse
   points get pushed apart; dense points keep their true distance.
3. Build a **minimum spanning tree** on that distance; removing edges from longest to shortest
   gives a **hierarchy** of clusters, indexed by λ = 1/distance.
4. **Stability** of a cluster C: S(C) = Σ over its points of (λ_p − λ_min(C)), i.e. how long each
   point stays in C after C is born.
5. **Extraction:** pick the non-overlapping clusters with the **largest total stability**.
   Groups smaller than **min_cluster_size = 10** are noise.

**Worked example: mutual reachability [illustration].** m = 5.
- Two points on a near car: core distances 0.10 and 0.11 m, 0.10 m apart → d = max(0.10, 0.11,
  0.10) = **0.11 m** (barely changed).
- A point on a near car (core 0.12) and a sparse speck (core 0.35), 0.20 m apart → d = max(0.12,
  0.35, 0.20) = **0.35 m**. The speck is pushed away, so it can't bridge the car to something
  else.

**Worked example: "holds together longest", with the slide-55 toy [illustration; measured from
the toy's condensed tree, not a thesis result].** Shrink the radius from huge to tiny:
- At **3.04 m** the one blob splits: near car (85 points) and far car (25 points) are born.
- The **far car** never breaks into pieces of 10+ points; its points just drop off one by one
  (median at 0.72 m).
- The **near car** holds until **0.23 m**, then breaks into pieces of 16 and 50 (the 50 then
  into 20 + 16) plus loose points.
- Lifetime is counted in 1/radius, per point: stability = Σ (1/r_leave − 1/r_born).
  Near car ≈ 85 × (1/0.23 − 1/3.04) ≈ 85 × (4.37 − 0.33) → exactly **335.7**. Its pieces:
  5.4 + max(6.8, 9.8 + 4.0) = **19.2**. 335.7 > 19.2 → **kept whole**.
- Far car: **24.1**, no pieces → kept. The whole scan is never picked (library default).
- **The comparison is only whole vs. its own pieces, never car vs. car.** The far car scores 14×
  less than the near car and is still kept. That's why sparse far cars survive.
- Why 1/radius: holding together at small radii (tight, dense) earns a lot; existing only at
  large radii (loose) earns little. 0.5 → 0.25 m adds 2.0; 3.0 → 2.5 m adds only 0.07.

**Worked example: why stability can split a car [illustration; toy numbers].** A car of 100
points with a thin density gap through the middle (two dense halves of 50):
- The whole car is born at λ = 2 (distance 0.5 m) and splits at λ = 3.3 (0.3 m):
  S = 100 × (3.3 − 2) = **130**.
- Each half lives from λ = 3.3 until it dissolves at λ = 8 (0.125 m): S = 50 × (8 − 3.3) = 235,
  so both halves together = **470**.
- 470 > 130, so HDBSCAN keeps the **two halves**: one car becomes two clusters. This is exactly
  the thesis's mechanism: "an internal density gap makes two sub-clusters individually more
  stable than the whole" [ch2].

**The frozen configuration** [ch3 §Clustering, `fig:clustering`]:
1. Re-voxelise the object points to a **coarser 0.10 m** grid.
2. Run HDBSCAN there (min_cluster_size 10, min_samples 5).
3. Copy each coarse point's label back to the full-resolution points by **nearest neighbour**.

The coarse grid closes small gaps inside a car, so fewer cars split, and it's faster too.

### A5. What the evaluation says about these three steps

**HDBSCAN vs fixed-radius methods** [ch4 `tab:clustering`; 0.10 m grid, per-frame, track
filter off]:

| Seq | Method | Precision | Recall | F1 | Clustering time |
|---|---|---|---|---|---|
| 00 | **HDBSCAN** | 0.972 | **0.731** | **0.835** | 306 ms |
| 00 | DBSCAN | 0.975 | 0.637 | 0.771 | 54 ms |
| 00 | Euclidean | 0.979 | 0.636 | 0.771 | 76 ms |
| 08 | **HDBSCAN** | 0.840 | **0.671** | **0.746** | 320 ms |
| 08 | DBSCAN | 0.870 | 0.565 | 0.685 | 57 ms |
| 08 | Euclidean | 0.869 | 0.576 | 0.693 | 75 ms |

Read it as: HDBSCAN buys about **+10 recall points** for about **6× the clustering time**.

**Fragmentation, and what the 0.10 m grid fixed** [ch4 `tab:splitrate`, seq 08, 34,057
eligible cars]:

| | 0.05 m grid | 0.10 m grid (frozen) |
|---|---|---|
| Car in one clean cluster | 55.1% | 66.0% |
| Car **split** into 2+ clusters | **44.8%** | **33.7%** |
| Car **merged** with a neighbour | 0.2% (56) | 0.8% (284) |
| Clusters per split car | 4.7 | 3.0 |

- Coarsening is a **trade**: about 3,700 cars stop splitting, about 230 newly merge. The rising
  merge count is why you can't keep coarsening.
- Pipeline effect: recall **0.710 → 0.732** (+756 TP), precision 0.910 → 0.911, F1 0.798 →
  0.812 [ch4 §Clustering Resolution].
- Geometric clustering alone already reaches recall **0.773**. Everything after it can only keep
  or lose recall, so **the recall ceiling is set here, in clustering**, not by the classifier
  (S4 goes deeper).

**Runtime** [ch4 `tab:runtime`, 623.5 ms per frame total]: preprocessing 125.9 ms (20.2%),
RANSAC 56.2 ms (9.0%), **HDBSCAN 370.4 ms (59.4%)**. Clustering is the single biggest cost.

---

## Part B — Q&A traps for this session

1. **"Your road isn't flat everywhere. Why one plane?"**
   → "On these urban sequences the local ground is close to planar, and one plane is cheap. On
   strong slopes it would degrade. A grid-based model like Patchwork++ is the future-work fix; we
   don't claim it's solved." [ch3]

2. **"What if RANSAC fits a wall instead of the road?"**
   → "The kept plane is accepted only if its normal is near vertical (z-component above 0.5). A
   wall fails that check, and then nothing is removed, so a bad fit can't delete real cars."
   **If pressed on how often:** in ~5% of seq-08 frames a wall wins and no ground is removed
   (private check, not in the thesis, which says "rare"). The better design checks every
   candidate, not just the winner. See `qa_backup/ransac_seq08.md`.

3. **"RANSAC is random. Are your results reproducible?"**
   → "Yes. The sampler uses a fixed seed, so ground removal, and the whole pipeline, gives
   identical output on every run." [ch3]

4. **"HDBSCAN is 59% of your runtime. Why not DBSCAN?"**
   → "Because it costs about ten recall points: 0.731 vs 0.637 on seq 00, 0.671 vs 0.565 on seq
   08 at the same grid. No fixed-radius method we tested reaches HDBSCAN's recall. The speed fix is
   a faster HDBSCAN implementation, not a different algorithm." [ch4 `tab:clustering`; ch5 future
   work]

5. **"If coarsening to 0.10 m helped, why not 0.20 m?"**
   → "It's a trade. Going from 0.05 to 0.10 m un-split about 3,700 cars but merged about 230 new
   ones (56 → 284). Coarser still would fuse more neighbouring cars, which costs both detection
   and completion." [ch4 for the numbers]
   **Careful:** 0.20 m was **not tested**; the thesis compares only 0.05 and 0.10 m. "Coarser
   fuses more" is an extrapolation from the merge trend, so say "we expect", not "we found". If
   pressed: "That's the next point on the curve, and it's untested."

6. **"Why downsample before removing outliers? Isn't the order arbitrary?"**
   → "It isn't arbitrary, and we say so in the thesis. The order changes which points survive.
   Downsampling first runs the outlier test on ~10× fewer, more regular points, and that order
   gave better detection." [ch3]
   **Careful:** `tab:runtime`'s row label reads "Preprocessing (z filter, denoise, voxel)", listing
   the steps in the other order. ch3 and the code (`voxel_before_denoise: True`) both say voxel
   first; the table label is just loose wording. If someone points at it: "the label lists the
   steps, not their order; ch3 gives the order."

7. **"Why does a car split at all?"**
   → "HDBSCAN keeps the most stable clusters. If a car has an internal density gap, its two halves
   can each be more stable than the whole car, so it keeps two clusters. It has no idea what an
   object is; it only sees density." [ch2; ch4 §recall analysis]

---

## Part C — Diagrams

### D4 — RANSAC on a real scan (generated from our data) — slide 2.2.1

*(Changed 2026-10-16: the draw.io scaffold was dropped as unclear. Online, RANSAC is taught
with a 2D line-fit toy (scikit-learn, MathWorks) and LiDAR papers show ground removal as a
before/after point cloud. D5 already covers the "after", so D4 shows the mechanism on real
data instead.)*

**File:** `docs/defense/assets/d4_ransac_real.png`, made by
`docs/defense/scripts/make_d4_ransac.py`. The script replays `remove_ground()`'s own seeded loop
on seq 08 frame 2500 and asserts that it picks the same plane as the pipeline.

**How to read it** (bird's-eye view, sensor at the black triangle):
- **Left, one random try (#2).** The 3 red points are scattered (two 18–27 m behind on the
  left, one 8 m ahead). The plane through them is tilted 43°, so only a thin stripe of **1,334 points** lies
  within 0.2 m of it.
- **Right, best of 300 tries (#3).** All 3 red points are on the road, so the whole road surface
  agrees: **31,707 points** within 0.2 m. RANSAC keeps this plane, and those points are removed
  as ground.

**What to say (~45 s):** "RANSAC guesses a plane from three random points and counts how many
points agree with it, meaning they lie within 20 cm. A bad guess, left, gets about a thousand
votes. A guess on the road, right, gets thirty thousand. After 300 guesses it keeps the winner
and deletes those points as ground."

**Questions this figure invites:**
- *"The best was found at try 3. Why run 300?"* → This frame is lucky: across seq 08 the best
  plane typically turns up around try 159. 300 was cut from 1000 for runtime; on a typical
  frame the plane at 300 is within ~1–2% of the plane at 1000, though about 1 frame in 5 would
  still improve. (Private check, `qa_backup/ransac_seq08.md`; not thesis numbers.)
- *"Some road at the lower right is still grey."* → That road is more than 0.2 m off the single
  fitted plane (a slope or camber), so it isn't removed and goes on to clustering. This is the
  one-plane limitation of trap #1, visible on real data.
- The counts are an **illustration on one frame**, not a thesis result. If pressed: "this is one
  scan, chosen to show the mechanism".

### D5 — Clusters before/after (thesis crop, done) — slide 2.2.2

`docs/defense/assets/d5_ground_then_clusters.png` = panels (b) and (c) of the thesis figure
`fig:bg-pipeline-stages` (`bg_pipeline_stages.png`): (b) the same real scan after RANSAC, ground
tan vs objects blue; (c) the objects after HDBSCAN, one colour per cluster. It's real data from the
thesis, so it needs no re-run. **Self-check:** in panel (c), find a long wall or hedge that
HDBSCAN cut into several colours. That's the same density mechanism that splits cars (A4).

### Q&A slides for this session (done 2026-10-16)

- **3.6.2 (slide 45):** native tables of `tab:clustering` (Mean IoU column left out) and
  `tab:splitrate`. Jump here for "why HDBSCAN, not DBSCAN?" and "why 0.10 m?".
- **How HDBSCAN Works (backup, slide 55):** `assets/hdbscan_explainer.png` from
  `scripts/make_hdbscan_explainer.py`. Toy 2D scene (dense near car, sparse far car), run
  through the real `hdbscan` library with our settings; the script asserts each panel's story.
  Step 1: core distance (circle to the 5th nearest point, itself included). Step 2: clusters at
  three radii cut from HDBSCAN's own hierarchy (0.3 m: near car only; 1.0 m: both; 3.2 m: fused).
  Step 3: the final answer keeps both cars. A toy, not a thesis result; it answers only "how
  does it work", not why cars split (that's A4's stability example, spoken).
- The RANSAC seq-08 check stays **off** the slides: private, not a thesis number. Answer it out
  loud, with its caveats (`qa_backup/ransac_seq08.md`).

---

## Part D — Glossary additions (EN → VI; correct to your department's usage)

| English | Vietnamese |
|---|---|
| preprocessing | tiền xử lý |
| voxel / voxel grid downsampling | voxel / giảm mẫu theo lưới voxel |
| statistical outlier removal | loại bỏ điểm ngoại lai thống kê |
| ground plane removal | loại bỏ mặt phẳng mặt đất |
| inlier / outlier | điểm nội / điểm ngoại lai |
| plane normal | vectơ pháp tuyến của mặt phẳng |
| iteration | vòng lặp |
| density-based clustering | phân cụm dựa trên mật độ |
| core distance | khoảng cách lõi |
| mutual reachability distance | khoảng cách khả đạt tương hỗ |
| minimum spanning tree | cây khung nhỏ nhất |
| cluster stability | độ ổn định của cụm |
| fragmentation (split) / merge | phân mảnh (tách cụm) / gộp cụm |
| nearest-neighbour label propagation | lan truyền nhãn theo láng giềng gần nhất |

