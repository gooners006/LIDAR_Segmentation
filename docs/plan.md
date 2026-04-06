# LiDAR Object Reconstruction — Project Plan

## Current State (2026-04-06)

### Pipeline
1. Load `.bin` → z-filter (local frame) → denoise → downsample
2. RANSAC ground segmentation (local frame)
3. DBSCAN clustering → bbox filtering (local frame)
4. Transform to global frame (poses.txt)
5. Centroid tracking (Hungarian algorithm)
6. Visualization (Open3D, camera follows ego vehicle)

### Evaluation Results (sequence 00, 100 frames)
| Metric | Value |
|---|---|
| Precision | 0.245 |
| Recall | 0.803 |
| F1 | 0.375 |
| Mean IoU | 0.927 |

**Diagnosis:** High recall + high IoU = detection is good when it matches. Low precision = ~3 false positives per true positive. FPs are non-object clusters (vegetation, walls, poles) that pass geometric filters but aren't "thing" classes in GT.

---

## Step 1: Integrate Semantic Labels ← NEXT

**Goal:** Use `.label` files to filter detections to "thing" classes only, eliminating vegetation/building FPs.

**Approach:**
- Load `.label` alongside `.bin` each frame
- Propagate semantic labels through the pipeline (z-filter → denoise → downsample → ground removal) using index mapping + nearest-neighbor (same as `evaluate.py`)
- After DBSCAN clustering, check majority semantic class of each cluster
- Keep only clusters where majority class is a "thing" class (car=10, person=30, cyclist=31, etc.)
- Expected impact: precision jumps from ~0.25 to ~0.7+ with minimal recall loss

**Files to modify:**
- `src/main.py` — add label loading + propagation + semantic filtering
- `src/evaluate.py` — update to reflect new filtering step

---

## Step 2: Semantic-Aware Evaluation

**Goal:** Break down detection metrics by object class.

**Approach:**
- Report per-class precision/recall/F1 (car vs pedestrian vs cyclist)
- Identify which classes are well-detected and which are missed
- Use this to guide parameter tuning per class (e.g., different volume thresholds for cars vs pedestrians)

**Files to modify:**
- `src/evaluate.py` — add per-class breakdown

---

## Step 3: Tracking Evaluation (MOTA/MOTP)

**Goal:** Quantify tracking consistency using standard MOT metrics.

**Approach:**
- Use GT instance IDs from `.label` files as ground truth tracks
- Compare against our tracker's assigned IDs frame-by-frame
- Compute MOTA (missed + FP + ID switches), MOTP (localization accuracy), ID switch count
- Can use `py-motmetrics` library or implement manually

**Files to create:**
- `src/evaluate_tracking.py`

---

## Step 4: Point Accumulation & Object Export

**Goal:** Stack per-object point clouds across frames and export `.ply` files.

**Approach:**
- For each tracked object, collect global-frame points from all frames where it was detected
- Concatenate and optionally denoise the accumulated cloud
- Export to `output/00/objects/<track_id>.ply`
- Export metadata to `output/00/tracks.json` (track ID, class, frame range, point count)

**Files to create:**
- `src/accumulate.py` or integrate into `main.py`
- Output: `output/<seq>/objects/*.ply`, `output/<seq>/tracks.json`

---

## Step 5: Reconstruction Quality Evaluation

**Goal:** Measure how well accumulated point clouds match GT object shapes.

**Approach:**
- For each tracked object, gather GT points (from `.label` instance IDs) across all frames in global frame
- Compute Chamfer distance between our accumulated cloud and GT cloud
- Report per-object and aggregate reconstruction quality

**Files to create:**
- `src/evaluate_reconstruction.py`

---

## Step 6: Parameter Tuning & Optimization

**Goal:** Improve metrics based on evaluation feedback.

**Candidates:**
- DBSCAN `eps` — may need per-distance-range tuning (objects far from sensor are sparser)
- Volume/dimension thresholds — per-class values (car vs pedestrian)
- Tracker `max_distance` — may need adjustment after semantic filtering reduces FPs
- Voxel size — tradeoff between detail and noise

---

## Out of Scope
- Deep learning detection/segmentation
- Real-time execution
- SLAM / map optimization
- Camera–LiDAR fusion
