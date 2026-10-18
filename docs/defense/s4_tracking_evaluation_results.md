# S4 — Tracking, Evaluation Protocol, Detection Results, Recall Analysis

Study notes for slides 2.2.5 (tracking), 3.1.2 (eligibility), 3.2.2–3.2.3 (metrics, LOSO),
3.3.1–3.3.5 (detection results), 3.5.1–3.5.2 (runtime) and 3.6.3 (IoU sensitivity); diagrams D8
and D10. All facts come from the submitted thesis (`docs/writing/thesis/`); thesis locations are
in brackets. **[code]** = checked in `src/`, not stated in the thesis. **[not in thesis]** and
**[illustration]** mark my own additions; don't quote those as results.

---

## Part A — Concepts

### A1. The centroid tracker: linking one object across frames

**Core meaning.** Detection runs one frame at a time. The tracker says "this car in frame 7 is
the same car as in frame 6", so each object gets one **track id** for as long as it is seen
[ch3 §Multi-Frame Tracking, `fig:tracker`].

**How it's computed** [ch3; `tab:pipeline-config` tracker rows]:
1. Each detection is reduced to one point, the **centroid** of its box.
2. Every frame, the current centroids are matched to the existing tracks by the **Hungarian
   algorithm** on the table of centroid-to-centroid distances. It picks the pairing with the
   smallest *total* distance, one detection per track.
3. A pair further apart than **2.0 m** is refused.
4. An unmatched detection **starts a new track**. A track that goes unmatched for **more than
   5 consecutive frames** is retired.
5. No motion model, no appearance: position only.

**[code]** The centroids are first moved into one **world frame** with the per-frame poses
(`poses[i] @ Tr`, `src/evaluate.py` pass 1), so a parked car stays put while the ego car drives
past it. The thesis does not mention the poses; see trap #9. **Every** geometric-filter survivor is
tracked, including the ones the classifier called not-car (`keep_unknown=True`), because the
track filter (A2) needs their votes.

**Worked example [illustration].** Two tracks: A at (0, 0), B at (3, 0). Two new detections:
d1 at (0.5, 0), d2 at (2.6, 0.2).

| | d1 | d2 |
|---|---|---|
| A | 0.50 m | 2.61 m |
| B | 2.50 m | 0.45 m |

- Pairing A–d1, B–d2: total 0.95 m. Pairing A–d2, B–d1: total 5.11 m. Hungarian takes the first.
- Both distances are ≤ 2 m, so both are accepted. If d2 had been 2.5 m from B, B would get no
  match this frame, d2 would start a new track, and B's missed-frame counter would go to 1.

**Why so simple.** "Detection, not association, is the bottleneck" [ch3]. **[illustration]** The
scan rate is 10 Hz, so between frames a car moves its speed × 0.1 s: 1.4 m at 50 km/h, 2.0 m at
72 km/h. Because the centroids are in a world frame, that's the car's *own* speed, not the speed
relative to us. A parked or town-speed car stays well inside the 2 m gate. Above ~72 km/h it
wouldn't (trap #8). An IoU-based (SORT-style) tracker is listed as future work. Cost: **0.3 ms per frame**
(0.1%) [`tab:runtime`]. The tracker is **never scored as a tracker** (no MOTA or IDF1) [ch5].

### A2. The track-level filter: decide once per object, not once per frame

**Core meaning.** The classifier sometimes flickers: car in one frame, not-car in the next. The
track filter takes **one decision per track** using all its frames, and applies it to every
frame of that track [ch3; `fig:tracker`].

**How it's computed** [ch3; `resolve_track_class`, `src/tracker.py` **[code]**]. A track is
accepted as a car only if:
1. at least **2** of its votes are car, **and**
2. car votes are at least **50%** of all its votes.

(The thesis adds "and they agree on a single class". With a binary classifier the only class is
car, so that condition is always true. It is left over from the multi-class design.)

Then the evaluation keeps **every detection of an accepted track, in every frame**, including
the frames where the classifier said not-car; it drops every detection of a rejected track
**[code]**, `src/evaluate.py` pass 2.

**Worked example [illustration]** (these are the D8 lanes):

| Track | Votes per frame | Car votes | Ratio | Decision | Effect |
|---|---|---|---|---|---|
| parked car | car, car, **not-car**, car, car | 4 | 0.80 | accept | frame 3 is kept: an FN becomes a TP |
| bush | not-car, **car**, not-car, not-car | 1 | 0.25 | reject | frame 2 is dropped: one FP removed |
| brief car | car, not-car | 1 | 0.50 | reject | a car seen for 2 frames is lost |
| borderline | car, car, not-car, not-car | 2 | 0.50 | accept | exactly at both limits |

**Evidence** [`tab:ablation-stage`, full seq 08]:

| | Precision | Recall | F1 | TP | FP |
|---|---|---|---|---|---|
| + classifier (per frame) | 0.846 | 0.676 | 0.751 | 23,600 | 4,303 |
| + track filter | **0.911** | **0.732** | **0.812** | 25,568 | 2,513 |

- **Both** go up: +1,968 TP (the parked-car case) and −1,790 FP (the bush case).
- It is **offline**: it needs the whole track, so it uses future frames. That's in scope
  (offline system) and stated in the thesis [ch3, ch4 §protocol].

### A3. Who counts as a car: eligibility and the recall denominator

**Core meaning.** Ground truth is SemanticKITTI's own instance labels for **car (10)** and
**moving-car (252)**, the "supported vehicles" [ch4 §Target Eligibility]. A car instance is
**counted** only if **at least 10 of its points survive** preprocessing and ground removal
(z-filter, voxel, denoise, RANSAC). Recall is measured against these **eligible** cars, not
against every annotated car.

**Why.** A car reduced to a few points by occlusion or range can't be segmented by any
clustering method; counting it would "measure sensor sparsity rather than the pipeline" [ch4].

**How much it narrows the denominator** [ch4, stride-20 sample of seq 08, 204 frames]:
- 2,332 annotated car instances, 1,737 eligible → **25.5% excluded** (1 − 1737/2332).
- Against cars with ≥ 10 points in the *raw* scan (2,033), 14.6% are excluded by preprocessing.
  Two different "≥ 10" rules; don't mix them.

**Worked example** [ch4's own arithmetic]: recall 0.732 is against eligible cars. Eligible
cars are 74.5% of all annotated cars, and an excluded car can never be found, so against all
annotated cars recall ≈ 0.732 × 0.745 = **0.545 ≈ 0.55**. The thesis calls this **inferred**
(the ratio comes from the stride-20 sample). **Quote 0.55**, not 0.54.

**[code]** The ≥ 10 count is taken on the object points after ground removal
(`gt_masks`, `m.sum() >= 10`, `src/evaluate.py`), so a car whose lowest points were removed as
ground counts only its remaining points.

### A4. Point-level IoU, greedy matching, micro-averaging

**Core meaning.** A detection and a ground-truth car are compared **as sets of points**, not as
boxes [ch4 §Evaluation Metrics, `fig:bg-iou`]:

IoU(D, G) = |D ∩ G| / |D ∪ G|

**How it's computed:**
1. Every (detection, GT car) pair with IoU ≥ **0.3** is a candidate.
2. Sort the candidates by IoU, highest first. Take the best; once a detection or a car is used it
   can't be used again (**greedy, one-to-one**).
3. Matched pair = **TP**. Unmatched detection = **FP**. Unmatched eligible car = **FN**.
4. Sum TP, FP, FN over **all frames**, then compute precision, recall, F1 once
   (**micro-average**: busy frames count more).
5. **Mean IoU** = the average IoU of the matched pairs only. It says how good the matches are,
   **not** how many cars were missed.

**Worked example [illustration]:** a GT car of 400 points.

| Case | Detection(s) | IoU | Outcome |
|---|---|---|---|
| Clean | one cluster of 360 points, 340 on the car | 340 / (400 + 360 − 340) = 0.81 | 1 TP |
| Split in two | 250-point and 150-point fragments | 0.625 and 0.375 | 1 TP (the 250) + **1 FP** (the 150: car already used) |
| Split in four | four 100-point fragments | 0.25 each, below 0.3 | **1 FN**, plus 1 FP for each fragment that survives filter + classifier |

So **a split costs twice**: a lost car (FN) and often a false detection (FP). D10 (frame 250)
shows exactly this: one parked car, one thin slice of it detected (FP), the car missed (FN).

**Headline arithmetic** [`tab:headline`]: TP 25,568 / FP 2,513 / FN 9,354.
- Precision = 25,568 / (25,568 + 2,513) = **0.911**
- Recall = 25,568 / (25,568 + 9,354) = **0.732**
- F1 = 2 × 0.911 × 0.732 / (0.911 + 0.732) = **0.812**

**Counts are per frame**: a car visible in 100 frames contributes up to 100 TPs. "25,568 TPs"
is not 25,568 distinct cars.

**Why point level, not boxes.** The pipeline outputs a point cluster, not a scored box, so the
point set is what gets judged. It's also stricter on segmentation errors: a box can look fine
around a car that's missing half its points.

### A5. Is 0.3 too lenient? The IoU-threshold sensitivity

[ch4 §Choice of IoU threshold, `tab:iou-sensitivity`, full seq 08]:

| τ | Precision | Recall | F1 | Mean IoU |
|---|---|---|---|---|
| 0.25 | 0.914 | 0.735 | 0.815 | 0.910 |
| **0.30** | **0.911** | **0.732** | **0.812** | **0.913** |
| 0.50 | 0.884 | 0.711 | 0.788 | 0.927 |

- From 0.30 to the strict 0.50, F1 drops only **2.4 points**. Across the whole range, 2.7.
- The reason: the matched detections have mean IoU **0.913**, far above either threshold. Matches
  aren't marginal overlaps scraping past 0.3.
- Pre-registered: a threshold-sensitive result would have been reported as a finding, not tuned
  away. It wasn't sensitive.

### A6. Leave-one-sequence-out (LOSO): the leakage control

**The problem it solves.** The final classifier's checkpoint was chosen on **seq 08**
validation, and seq 08 is where the headline is reported. That could flatter the numbers.

**How it's computed** [ch4 §Dataset Splits, §LOSO, `tab:loso`]:
1. 11 folds, one per labelled sequence (00–10).
2. Fold K: train a classifier on the mined clusters of the **other ten** sequences, then run the
   full pipeline on **all of sequence K**.
3. **No checkpoint selection:** a fixed 15 epochs, last epoch kept (historically the best epoch
   was 13/15, so last ≈ best). Sequence K is untouched in training *and* selection.
4. The mined clusters carry a sequence prefix, so folds are a re-partition: **no re-mining**.

**Results** [`tab:loso`]:

| | Precision | Recall | F1 |
|---|---|---|---|
| Final classifier on seq 08 (headline) | 0.911 | 0.732 | 0.812 |
| LOSO fold-08 (blind to seq 08) | 0.888 | **0.736** | 0.805 |
| **Pooled, all 11 folds** | **0.888** | **0.717** | **0.793** |
| Pooled, without seq 00 | 0.871 | 0.695 | 0.773 |
| Macro mean ± std | 0.861 ± 0.059 | 0.632 ± 0.119 | 0.721 ± 0.095 |

**Reading it:**
- **Selection bought ~2 precision points and no recall.** Fold-08 and the final model share
  architecture, the same ten training sequences and the same 15-epoch regime; they differ only in
  which epoch is kept. Recall 0.736 vs 0.732; precision 0.888 vs 0.911 (715 extra FPs over 4,071
  frames).
- The one confound (fold-08 trains on 95% of the clusters, 5% held back to monitor loss) removes
  data, which would *lower* its recall, so it only strengthens the reading.
- **Pooled is within ~2 points of the headline** on every metric: seq 08 is representative.
- **Spread:** recall 0.336 (seq 01, sparse highway) to 0.758 (seq 00, dense urban). Precision is
  much steadier.
- **Seq 00 exposure is small:** the classical stages were historically hand-tuned on seq 00.
  Dropping that fold moves pooled recall 0.717 → 0.695 (0.022).

**Worked example: why pooled 0.717 ≠ macro 0.632.** Pooled sums the counts: seq 00 has
56,190 eligible car-frames (42,583 + 13,607), seq 01 only 1,606 (540 + 1,066). Macro gives each
sequence one equal vote, so seq 01's 0.336 and seq 04's 0.510 (only 271 frames) pull the mean
down. Both are reported "so neither hides the spread" [ch4].

**For seq 00, quote the fold-00 row** (P 0.921 / R 0.758 / F1 0.831). The full seq-00 F1 0.838
is not in the thesis.

### A7. The detection result in one picture

[ch4 §Reported Operating Point; `tab:headline`, `fig:bev-detections`, `fig:timeseries`]:
- **Precision-saturated:** about **0.6 FP per frame** (2,513 / 4,071), flat over time while the
  number of cars per frame swings.
- **Recall-limited:** about one eligible car in four is missed (R 0.732), and the cause is
  structural (A8).
- Stage story (S3 A6 + A2 above): geometric filter keeps most cars but 4 of 5 detections are junk
  (P 0.199 / R 0.773) → classifier removes ~104,000 FPs (0.846 / 0.676) → track filter wins back
  1,968 TPs and drops 1,790 more FPs (0.911 / 0.732).

### A8. What limits recall: clustering splits cars

**Core meaning.** The misses are made **before** the classifier runs. HDBSCAN breaks large, close
cars into fragments; no later stage can rejoin them [ch4 §Result Analysis].

**Step 1: it's not the classifier.** Geometric filter alone: recall **0.773**, already higher than
the full pipeline's 0.732. The classifier can only drop clusters; the tracker only keeps or drops
clusters that exist in that frame and **never fills a gap**. Nothing after clustering can add a
car. (The net trade: ~1,400 TPs given up for ~106,000 FPs removed.)

**Step 2: it's not the geometric filter either.** 27,718 rejected clusters overlap a GT car, but
mostly via the min-volume (73.9%) and min-points (12.7%) gates, i.e. small splinters (an
**inference** in the thesis; S3 A1).

**Step 3: the root cause** [`tab:split`, `tab:splitrate`, full seq 08, 34,057 GT cars]:

| | 0.05 m grid (before) | **0.10 m grid (final)** |
|---|---|---|
| ok (one dominant cluster) | 55.1% | **66.0%** |
| split (2+ clusters) | 44.8% | **33.7%** |
| merged with another car | 0.2% (56) | 0.8% (284) |
| mean clusters per split car | 4.7 | 3.0 |
| recall (full pipeline) | 0.710 | **0.732** |

- **Splitting, not merging**, is the failure: a third of cars split, under 1% merge.
- **Worst where cars are nearest** (0.10 m grid): split rate **53.5%** at 0–10 m, 40.2% at
  10–20 m, 22.3% at 20–40 m, 3.5% past 40 m. A close car returns many points and HDBSCAN finds
  density gaps inside it (roof vs body, near side vs far side).
- **The fix that worked** was coarsening the clustering grid 0.05 → 0.10 m: ~3,700 cars move from
  split to clean, ~230 newly merge, recall +0.022 (+756 TP). Six repair strategies (merging
  fragments, other clusterers, adaptive parameters, accumulation…) all failed to generalise
  (slide 56 backup; S2).
- The thesis had first called ~0.74 a "hard ceiling"; the resolution fix showed that was a limit
  of the repairs, not of the approach, and the thesis corrects itself [ch4 §Clustering
  Resolution and the Corrected Ceiling].

**Step 4: the range table shows it** [`tab:distance`, stride-20, per frame, **track filter
off**, so pooled recall is 0.671, not 0.732]:

| Range | Recall | Split rate (0.10 m) |
|---|---|---|
| 0–10 m | 0.787 | 53.5% |
| 10–20 m | **0.862** | 40.2% |
| 20–40 m | 0.550 | 22.3% |
| 40 m+ | 0.196 | 3.5% |

- **Near-range dip:** the nearest cars carry the most points yet have *lower* recall than 10–20 m.
  That's over-segmentation, not sparsity.
- **Beyond 20 m:** ordinary sparsity; cars thin out below what clustering and the 10-point rule
  can handle. Mean IoU of matched cars *rises* with range (0.902 → 0.943): the far cars found are
  compact single clusters.

### A9. Runtime

[ch4 §Runtime, `tab:runtime`, full seq 08, 4,068 timed frames, RTX 3070 Ti / Ryzen 7 7800X3D]:

| Stage | ms | Share |
|---|---|---|
| Preprocessing | 125.9 | 20.2% |
| Ground (RANSAC) | 56.2 | 9.0% |
| **Clustering (HDBSCAN)** | **370.4** | **59.4%** |
| Geometric filter | 36.2 | 5.8% |
| Classifier | 33.7 | 5.4% |
| Tracking | 0.3 | 0.1% |
| **Total** | **623.5** | ≈ **1.6 fps** |

- Classical CPU stages ≈ 94%; the only learned detection stage is 5.4%.
- **Optimisation** (fewer RANSAC tries, 0.10 m clustering grid): **921.3 → 623.5 ms**, about a
  third faster, and the coarser grid *also* improved recall.
- Below 400 ms was not reachable: the post-ground cloud is already sparse, and Python HDBSCAN has a
  ~185 ms floor. Real time is out of scope (offline system). Completion adds ~13 ms per completed
  car, outside the detection budget.

---

## Part B — Q&A traps for this session

1. **"Isn't IoU 0.3 too lenient? Benchmarks use 0.5."**
   → "0.5 is the usual *box* threshold; ours is point-level. We re-ran at 0.5: F1 drops only 2.4
   points, 0.812 → 0.788, because the matched detections already overlap at 0.913 on average."
   [`tab:iou-sensitivity`]

2. **"You excluded a quarter of the cars. Isn't recall inflated?"**
   → "Recall is against cars with at least 10 surviving points; a car with fewer can't be
   segmented by any method. We disclose the effect: 25.5% are excluded, so against every
   annotated car recall is about 0.55." [ch4 §eligibility] Say "about 0.55", and that it's
   inferred from a sample.

3. **"The track filter uses future frames. Isn't that cheating?"**
   → "It's offline by design and stated as such; the system processes recorded drives. Its effect
   is isolated as its own ablation row: +1,968 TPs and −1,790 FPs." A causal version (only past
   frames) was **not tested**; don't claim it would work equally well.

4. **"How can a filter increase recall?"**
   → "It re-admits frames the classifier rejected: if a track is mostly car, the frames where the
   classifier said not-car are kept too. The tracker never creates a detection where no cluster
   exists, so it can't go above the 0.773 the geometric stage allows." **[code]** for the
   re-admission detail; the thesis says "a car seen across many frames is kept even where a single
   frame classified it as unknown" [ch4 §Stage decomposition].

5. **"What does 'agree on a single class' mean for a binary classifier?"**
   → "Nothing extra: with one positive class it's always satisfied. It's left from the earlier
   multi-class design; the effective rule is at least 2 car votes and at least half the frames."
   **[code]** Don't defend it as a separate check.

6. **"Why should we trust seq 08 if you selected the classifier on it?"**
   → "Leave-one-sequence-out: a classifier that never saw seq 08 gets recall 0.736 against our
   0.732, precision 0.888 against 0.911. Selection bought about two precision points and no recall.
   Pooled over all eleven held-out folds: 0.888 / 0.717 / 0.793." [`tab:headline`, `tab:loso`]

7. **"Why is seq 01 so bad (recall 0.336)?"**
   → "It's a highway: cars are far, and recall collapses with range (0.196 past 40 m). The
   failure is conservative: precision stays high." Quote seq-01 precision as **0.957**
   (`tab:loso`). ch5 says 0.951; don't quote that one.

8. **"Why does the tracker work at all with no motion model?"**
   → "At 10 Hz a car in town moves about one to one and a half metres between scans, so nearest
   centroid within 2 m links it. Detection, not association, is the bottleneck; a SORT-style
   tracker is future work." The tracker is **not scored** (no MOTA / IDF1) [ch5 limitations].
   Don't claim tracking quality.
   ⚠ **Follow-up "and on a highway?" [not in thesis; speculative, from the code, not
   measured].** Above ~72 km/h a car moves more than 2 m per frame, so every frame would start a
   new one-frame track, and the track filter (≥ 2 car votes) would reject them all. Fast movers
   would then be lost **by the tracker**, not the detector. That could add to seq 01's low recall
   (0.336, highway), but the thesis attributes seq 01 to range, and I have **not** measured how
   much is tracking. Settle with a private check: run seq 01 with and without the track filter
   (`evaluate.py --seq 01 --frames 5000`, once as is, once with `--tracker-max-distance 4.0`, each
   with `--json-out` to a new file; ~25 min for both). Decided in advance: if the 4 m gate lifts
   seq-01 recall by more than 0.03, the 2 m gate is costing highway cars; if not, it isn't. Until then answer: "Above about 70 km/h a car moves
   further than the 2-metre gate between scans, so a motion model would be needed there; that's
   part of the tracker future work."

9. **"Your tracker runs in a world frame. Where do the poses come from?"** ⚠ **[code; not in
   thesis]** The centroids are mapped with the per-frame poses shipped with the dataset
   (`poses.txt`, `poses[i] @ Tr`). The thesis never mentions this. *Inferred, not checked:*
   SemanticKITTI ships poses for the unlabelled test sequences too (our `docs/datasets.md` lists
   poses for 11, 12), which fits SemanticKITTI's SLAM-estimated poses rather than KITTI's GPS
   ground truth (00–10 only). Settle by checking the SemanticKITTI dataset page for its pose
   source before quoting it.
   → "The tracker links centroids in a world frame using the per-frame vehicle poses that come
   with the dataset; on a car they'd come from its odometry. It only needs them to be locally
   consistent over a few frames."

10. **"Your FP rate: 'roughly one per frame' or 0.6?"** ch4 says both. The number is
    2,513 / 4,071 = **0.6**. Use 0.6.

11. **"25,568 TPs: is that 25,568 cars?"**
    → "No, per-frame matches: a car seen in 100 frames counts up to 100 times." Never say
    "detected 25,568 cars".

12. **"Why is near-range recall lower than mid-range?"**
    → "Over-segmentation: half of the cars within 10 m get split by HDBSCAN (53.5%), so no single
    fragment matches the whole car. Beyond 20 m it's ordinary sparsity." Note the range table is
    **per-frame with the track filter off** (pooled 0.671), so it shows a trend, not the headline.

13. **"Split counts: 34,057 cars here, but TP + FN = 34,922 in the headline. Which is it?"**
    **[not in thesis; not reconciled]** The split analysis and the evaluation are separate scripts
    and count eligible car-frames separately; the gap (2.5%) is not explained in the thesis and I
    have not traced it. Answer if asked: "The split analysis is a separate diagnostic run with its
    own counting; the percentages are what matter, and they don't move with a 2.5% count
    difference." Unlikely to be asked.

---

## Part C — Diagrams

### D8 — Tracking timeline (user, draw.io) — slide 2.2.5
Scaffold: `assets/d8_tracking_timeline.drawio` (from `scripts/make_d8_scaffold.py`).
**Story (two lanes):** top = a parked car the classifier misses in one frame → the track vote
accepts the whole track → that frame counts too (**recall ↑**). Bottom = a bush the classifier
calls a car once → the track vote rejects it → that FP is gone (**precision ↑**). Orange line:
"Decide once per object over its whole track, not once per frame".
**Fill the "?":** link gate **≤ 2 m**; a track survives **5** missed frames; top vote **4 of 5**,
bottom **1 of 4**; rule **≥ 2 votes and ≥ 50%** (✓ top, ✗ bottom).
Export as usual: `assets/d8_tracking_timeline.drawio.png` (Zoom 200%, Transparent off,
Appearance Light, diagram embedded).

### D10 — Split car, frame 250 (thesis crop) — slide 3.3.3
`assets/d10_split_car_frame250.png` (from `scripts/make_d10_crop.py`: thesis
`seq08_failure_zooms.png`, frame-250 row + its legend, pixels unchanged).
**Story:** left panel: one parked car ~16 m behind (orange = missed) with a thin red slice (a
false positive). Right panel: the dashed GT footprint with only that slice detected (light blue).
**One split car = one FN + one FP.** The car on the right (green / dark blue) is clustered in one
piece and matched. Say: "this is what a third of cars look like to the clusterer".

---

## Part E — Spoken answers (from the S4 quiz, 2026-10-18; VI drafts, correct to your department's usage)

Quiz result: Q1 ✓, Q2 ✓ (read from the notes table), **Q3 ✗**, **Q4 incomplete**, **Q5 half**;
Q6–Q7 skipped.

**"So you find three out of four cars?" (Q3, answered wrongly: "yes").**
- EN: "Not quite. 0.732 is against cars that keep at least 10 points after preprocessing; with
  fewer, no clustering method can segment them. That rule excludes about a quarter of annotated
  cars. Against every annotated car, recall is about 0.55, and we report both."
- VI: "Không hẳn. 0,732 là độ phủ tính trên những xe còn ít nhất 10 điểm sau tiền xử lý; ít hơn
  thì không phương pháp phân cụm nào tách được. Quy tắc này loại khoảng một phần tư số xe được gán
  nhãn. Tính trên toàn bộ xe được gán nhãn, độ phủ khoảng 0,55, và luận văn báo cáo cả hai con số."

**"Isn't choosing the checkpoint on seq 08 leakage?" (Q4, incomplete).**
- EN: "We checked with leave-one-sequence-out. A classifier that never saw seq 08 gets the same
  recall, 0.736 against 0.732, and two points less precision. The two models differ only in which
  epoch was kept, so choosing on seq 08 bought a little precision and no recall. Across all eleven
  held-out sequences the result stays within about two points."
- VI: "Chúng tôi kiểm tra bằng kiểm định chéo loại trừ từng chuỗi. Một bộ phân loại chưa từng thấy
  chuỗi 08 đạt cùng độ phủ, 0,736 so với 0,732, và độ chính xác thấp hơn hai điểm. Hai mô hình chỉ
  khác nhau ở epoch được giữ lại, nên việc chọn trên chuỗi 08 chỉ thêm một chút độ chính xác, không
  thêm độ phủ. Trên cả mười một chuỗi kiểm tra, kết quả chênh không quá khoảng hai điểm."

**"Why is near-range recall lower?" (Q5, missed the caveat).**
- EN: "Half of the cars within 10 metres are split by the clustering, so no single piece matches
  the whole car. Beyond 20 metres it's ordinary sparsity. This table is a per-frame sample with the
  tracker off, so it shows the trend, not the headline number."
- VI: "Một nửa số xe trong phạm vi 10 mét bị thuật toán phân cụm tách ra, nên không mảnh nào khớp
  với cả chiếc xe. Xa hơn 20 mét là do điểm thưa. Bảng này là mẫu theo từng khung, không dùng bộ
  theo vết, nên nó cho thấy xu hướng chứ không phải con số chính."

**What the 0.10 m grid is (side question).**
- EN: "HDBSCAN adapts to how dense each object is, but it compares gaps to the normal point
  spacing. The 10-centimetre grid makes small gaps inside a car look like normal spacing, while
  gaps between cars still stand out."
- VI: "HDBSCAN tự thích ứng với mật độ của từng vật thể, nhưng nó so khoảng trống với khoảng cách
  điểm thông thường. Lưới 10 cm làm các khe nhỏ bên trong một chiếc xe trông như khoảng cách bình
  thường, còn khoảng trống giữa hai xe vẫn nổi bật."

---

## Part D — Glossary additions (EN → VI; correct to your department's usage)

| English | Vietnamese |
|---|---|
| tracking / track id | theo vết / mã định danh vết |
| centroid tracker | bộ theo vết dựa trên tâm |
| Hungarian algorithm (assignment) | thuật toán Hungary (bài toán gán) |
| track-level filter / majority vote | bộ lọc mức vết / bỏ phiếu đa số |
| offline / causal | ngoại tuyến / nhân quả (chỉ dùng khung quá khứ) |
| ground truth instance | thể hiện (đối tượng) nhãn chuẩn |
| eligibility rule / recall denominator | quy tắc đủ điều kiện / mẫu số của độ phủ |
| intersection over union (IoU) | tỉ lệ giao trên hợp (IoU) |
| greedy one-to-one matching | ghép cặp tham lam một-một |
| true / false positive, false negative | dương tính đúng / dương tính giả, âm tính giả |
| precision / recall / F1 | độ chính xác / độ phủ / F1 |
| micro- vs macro-average | trung bình vi mô / trung bình vĩ mô |
| leave-one-sequence-out cross-validation | kiểm định chéo loại trừ từng chuỗi |
| data leakage / model selection | rò rỉ dữ liệu / lựa chọn mô hình |
| over-segmentation (split) / under-segmentation (merge) | phân đoạn quá mức (tách) / phân đoạn thiếu (gộp) |
| runtime / frames per second | thời gian chạy / số khung hình mỗi giây |
