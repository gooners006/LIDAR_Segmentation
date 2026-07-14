# LiDAR Vehicle Segmentation — Project Walkthrough

_Synthesis of `project_state.md` and the 27 logged findings. Last updated: 2026-06-29._

## 1. What the project is

A master's thesis on **car detection / segmentation from SemanticKITTI LiDAR sequences**.
The core is a classical, interpretable pipeline (clustering → geometric filters → a
learned classifier → tracking), with **point-cloud completion** explored as a potential
headline contribution. It is research code: the priority is reproducibility and clean
experiments over abstraction or packaging.

**Evaluation discipline** (hard-won): every change follows a fixed protocol — state the
hypothesis, run `evaluate.py` before/after, compare Precision / Recall / F1 / meanIoU.
The primary eval sequence is **seq 08 (held-out)**; seq 00 numbers exist but are flagged
as leaked (see §3C).

## 2. The pipeline architecture

| Stage | What it does | Status |
|---|---|---|
| 1–2 | Z-filter → statistical denoise → 0.05 m voxel downsample | Working |
| 3 | RANSAC ground-plane removal (seeded, deterministic) | Tuned |
| 4 | HDBSCAN clustering | Working (recall ceiling confirmed) |
| 5 | Geometric filtering (ground-plane-relative) | Tuned |
| 6 | Binary car/not-car PointNet classifier (dual-branch: points + bbox features) | Trained |
| — | Centroid tracker + track-level filtering | Working |
| 7 | Point completion (PCN; PoinTr in progress) | Active research |

Current headline metrics (seq 08, held-out, classifier + track filter):
**P 0.956 / R 0.731 / F1 0.829 / mIoU 0.888**.

## 3. The four storylines

### A. Clustering and the recall ceiling — *characterized and closed*

Recall plateaus at **~0.74** and will not budge. This was chased hard:

- **Root cause (Findings #22–23):** HDBSCAN **splits** 31–37% of GT cars into multiple
  clusters (large/close cars worst: 66% split rate at 0–10 m). Merging is negligible
  (<0.5%). The "recoverable ceiling" of single-cluster cars (63–68%) almost exactly
  matches observed recall — the pipeline already recovers nearly every clean car.
- **Everything tried failed (Findings #21, #24):** BEV clustering (F1 0.779, merges
  overlapping objects), higher min-cluster-size (overfits seq 00, hurts seq 08),
  post-clustering fragment merge (precision loss outweighs recall), distance-adaptive
  HDBSCAN (ring artifacts), temporal aggregation (F1 collapsed to 0.073).
- **Verdict:** ~0.74 is a hard limit of density clustering on voxelized LiDAR without
  learned object priors. Accepted; alternatives kept CLI-toggleable for documentation.

### B. Geometric filtering — *the cheap precision win*

Pre-tuning, the pipeline ran at P 0.24 / R 0.87 — drowning in false positives
(vegetation 38%, buildings 33%, trunks). Finding #4 added three filters
(`max_center_height_above_ground=1.5`, `max_height_span=1.8`, `max_aspect_max_min=6.0`),
tuned against a per-cluster FP-analysis CSV. Result: **FPs cut 85%** (4489→676),
precision 0.24→0.65. The residual FPs are compact vegetation that geometrically mimics
cars — exactly what the classifier is for.

### C. The classifier — *two-stage, then a binary pivot, then a leak fix*

- **Two-stage strategy:** Stage A on synthetic ShapeNet (balanced), Stage B fine-tuned on
  real mined SemanticKITTI clusters. Stage A alone is **useless on real data**
  (Finding #7: classifies *everything* as unknown — total synthetic-to-real domain gap),
  which validated the need for Stage B.
- **Binary pivot (Finding #20):** the original 4-class (car/bus/motorcycle/unknown)
  produced phantom motorcycle detections and starved rare classes. Collapsed to
  **car / not-car**. Stage B mined ~420k train / 130k val clusters; best macro F1 ~0.92,
  car P 0.84 / R 0.90.
- **The data leak (Finding #18):** all the great early numbers (F1 0.834) were on
  **seq 00 — which was in Stage B's training set.** Re-evaluating on held-out **seq 08
  dropped F1 to 0.730**. This reset the project's honesty baseline; seq 08 is now
  mandatory.
- **Stage A ablation (Finding #25):** the advisor's "synthetic data too perfect" concern
  was tested directly. From-scratch Stage B *matches* pretrained (macro F1 0.9285 vs
  0.9225) — the 420k real clusters do all the work; the synthetic prior is redundant but
  harmless (keeps a weak pipeline precision edge).

### D. Completion — *the long saga, and the current frontier*

The most involved thread and where active work lives:

1. **PCN domain gap (Findings #15–19):** PCN trained on ShapeNet produced **blobs** on
   real LiDAR. Tried noise augmentation, virtual Velodyne ray-casting, sparse-input
   training — **all failed.** Concluded the mismatch was structural (single-view depth
   render vs accumulated LiDAR partiality).
2. **The plot twist (Finding #26):** built a **KITTI-like single-view partial generator**
   and retrained PCN (val 0.1246). Careful verification revealed the blobs were **mostly
   an inference bug in `completion.py`**, not a data/model failure — it applied 3D PCA +
   partial-radius/centroid normalization the model never saw in training (wrecked *every*
   checkpoint, ~3.5× worse CD even in-distribution). The model is actually good
   in-distribution (CD 0.16 m, F@0.1 0.76). Fix: drop PCA, reorient gravity→Y /
   length→Z, scale ×1.137, estimate full-car center. **Key caveat:** the static-car
   pseudo-GT metric is *invalid* — accumulated LiDAR is itself one-sided, so CD rewards
   under-completion. Real-data assessment must stay qualitative.
3. **Input gating (Finding #27):** with inference fixed, some dense cars still completed
   poorly. It was not heading estimation (PCA vs L-shape fitting was a wash, 18/47 either
   way) — it was **bad inputs**: fragments and merges leaked by HDBSCAN splitting. Added
   an **L-shape input gate** (reject fit-length < 2.7 m fragments, fit-width > 2.3 m
   merges). Lifted completion precision **38% → 69%**, retaining all 18 plausible cars.
   The residual 8/26 implausible *clean* completions are genuine model error.
4. **PoinTr — done, verdict keep PCN (Finding #28):** a faithful self-contained
   **PoinTr** (transformer completer, 8.9M params) was implemented and trained on PCN's
   exact dataset for a clean one-variable comparison. **Synthetic: small PoinTr edge**
   (matched eval, corrected 2026-07-06: CD 0.153 vs 0.161 m, F@0.1m 0.782 vs 0.755;
   the originally recorded "decisive win" mixed metrics from different protocols).
   **Real seq-08: a wash** — same 26/62 clean-gated tracks, plausible-car rate 16/26
   vs PCN's 18/26 (the gap is height-threshold noise), near-identical BEV footprints.
   Both models hit the same partiality-gap and centroid-estimation ceiling, not a
   capacity limit. Kept PCN as production.

## 4. Where things stand now

- **Pipeline:** mature and frozen. Detection at P 0.96 / R 0.73 / F1 0.83 on held-out
  seq 08; recall ceiling understood and accepted.
- **Classifier:** binary Stage B is production (`stage_b_best.pth`).
- **Completion:** PCN inference bug fixed, input gate live; **PoinTr training in the
  background**, then the PoinTr-vs-PCN comparison decides whether to swap.

## 5. Open threads

1. **Finish PoinTr training → run the comparison** (synthetic val CD/F-score, then real
   seq-08 plausibility) → record verdict, flip project_state Step 5 to DONE.
2. **Thesis writing** — pipeline description, recall-ceiling discussion, completion
   narrative, pipeline diagram.
3. **Backlog** (low priority): Patchwork++ ground removal, IOU/SORT tracker, clustering
   benchmarks.

---

## 6. Possible storylines to tell

Five framings for the thesis narrative. They are not mutually exclusive — most reports
pick a primary spine and fold one or two others into the discussion.

### Storyline 1 — "A rigorous, honest pipeline whose limits are understood" (safest)
The contribution is **a fully characterized classical pipeline**, not a leaderboard number.
The spine: each stage tuned with before/after evidence, the recall ceiling traced to a
**specific, named cause** (HDBSCAN splitting, #22–23), and a battery of mitigations shown
to fail on held-out data (#21, #24). Strength: defensible, complete, and the negative
results *are* the science. Best if the completion work doesn't land a clean win.

### Storyline 2 — "Detection is solved by geometry; the residual error is a clustering ceiling"
A tight quantitative argument: geometric filters + classifier push precision to ~0.96, and
the **only** remaining error is recall, which is provably **not** a filter/classifier
problem but a density-clustering limitation. Frame the thesis around *decomposing* where
error comes from and proving each component is at its achievable limit. Strength: crisp,
falsifiable, every claim backed by an ablation. The recall ceiling becomes a feature of the
analysis, not an embarrassment.

### Storyline 3 — "The completion détour: when a 'domain gap' was really an inference bug" (most interesting)
A methods-and-rigor story built on Findings #15–27. The arc: a model that "obviously"
failed (blobs), four expensive domain-adaptation attempts that all failed, and then the
discovery via controlled ablation that **the model was never the problem — the inference
normalization was** (#26). Then the second correction: completion quality is gated by
*input cleanliness*, not heading (#27), tying completion failures back to the same HDBSCAN
splitting from Storyline 2. Strength: genuinely instructive, shows scientific maturity, and
the "invalid pseudo-GT metric" insight is publishable-grade. Risk: it's a story about
*process*, so it needs at least one concrete positive completion result to anchor it.

### Storyline 4 — "Transformer completion tested, and capacity is not the bottleneck" (resolved)
PoinTr was implemented and compared against PCN (Finding #28, corrected 2026-07-06). It
holds a **small synthetic edge** (matched eval: CD 0.153 vs 0.161 m, F 0.782 vs 0.755;
the original "decisive win" was a metric mix-up) and is a **wash on real seq-08** (plausible
16/26 vs 18/26, threshold noise). The lesson — a stronger decoder does not help when the
limit is the synthetic↔real partiality gap and centroid estimation, not capacity — folds
naturally into Storyline 3. This is now a *resolved* sub-result, not a headline: it
strengthens the transfer-gap argument rather than providing a "transformer beats PCN" win.

### Storyline 5 — "Synthetic priors are redundant at scale" (a sharp side-claim)
A focused secondary thread from Finding #25: with 420k real mined clusters, ShapeNet
pretraining contributes essentially nothing (from-scratch matches it). A clean,
counter-intuitive, well-controlled result about when synthetic pretraining stops mattering.
Too small to be the whole thesis, but an excellent discussion-section or ablation-chapter
highlight.

### Recommended spine
Lead with **Storyline 2** (decomposed error analysis — the most rigorous and already fully
supported by evidence), use **Storyline 3** as the completion chapter (the détour is the most
memorable part of the work), fold **Storyline 4** into it as the closing experiment (the
PoinTr comparison resolved that decoder capacity is not the real-data bottleneck — a clean
reinforcement of the transfer-gap thesis), and add **#5** as a sharp ablation. The completion
chapter now has a definite arc: blobs → "domain gap" → inference bug → input gating →
transformer test → *the bottleneck is the synthetic↔real partiality gap, not the model*.
