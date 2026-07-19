# Pipeline Runtime Optimization Plan

Created 2026-07-19 (advisor asked for inference time; user wants real improvement,
not restricted to the current pipeline).

**Hypothesis:** the detection pipeline's 934 ms/frame is dominated by
implementation overheads (per-cluster GPU calls, single-threaded core-distance
computation, oversized RANSAC budget, denoising at full resolution), not by
anything the method needs; it can reach ≤ 400 ms/frame with detection metrics
unchanged.

**Metric to improve:** mean ms/frame from `scratchpad/timing_benchmark.py`
(iteration benchmark: `--stride 20 --frames 204`, baseline **934 ms**; final
verdict vs the full-sequence baseline `timing_seq08_full_n4068.json` =
**921 ms/frame**, completed 2026-07-19).

**Guard metrics (must not regress):** seq 08 full P/R/F1/mIoU =
0.903 / 0.699 / 0.788 / 0.895. Tier-1 changes must reproduce TP/FP/FN
*exactly* (eval is deterministic, Finding-independent); behavior-changing
tiers follow the Experiment Protocol.

---

## Section A — Decisions to tweak

### A1. Regression budget — what may an accepted change do to the numbers?
Biggest blast radius: every published number (§2 tables, and the completion
experiments #29/#32 whose inputs are the current pipeline's clusters) is
downstream of this.

- **(a) Two-class rule (recommended).** Implementation-only changes must
  reproduce TP/FP/FN bit-for-bit before landing. Algorithm swaps live behind
  `PIPELINE_CONFIG` flags + `--out-tag` outputs and are only promoted to
  production if F1 improves, or speedup ≥ 3× at ΔF1 ≥ −0.005 — and promotion
  triggers regenerating `output/08` and re-running #29/#32 scripts.
  *Why:* deterministic eval (RANSAC `rng(42)`) makes exact verification free;
  keeps the thesis story "same method, faster" for everything already written.
- **(b) Flat tolerance (ΔF1 ≥ −0.01) for all changes.** Simpler, but silently
  invalidates the provenance of #29/#32 and forces a re-write of §5 numbers.

Downstream if (b): every tier can land directly, but all report tables must be
re-generated at the end.

### A2. HDBSCAN (511 ms, 55% — the decision that matters most)
- **(i) Library tuning (recommended first).** `core_dist_n_jobs=-1` on
  `hdbscan.HDBSCAN` (`src/pipeline.py:191`; library default is 4 threads,
  7800X3D has 16). Parallel core-distances do not change values → Tier 1.
  Expect 1.2–1.5×.
- **(ii) Coarse-voxel clustering (recommended experiment).** Cluster on a
  0.10 m voxel grid, propagate labels back to the 0.05 m points by nearest
  neighbor. HDBSCAN is superlinear in N → expect 3–5× on this stage. Changes
  clusters → full protocol. *Quality side-bet:* #23's split cause is internal
  density gaps in close cars; coarser voxels close small gaps, so recall may
  move (either way) — measure split rate with `src/analyze_clustering.py`.
- **(iii) cuML HDBSCAN (GPU).** ~10× expected, but RAPIDS has no native
  Windows build → WSL2 + duplicate venv + dataset path plumbing. Only if
  (i)+(ii) miss the ≤ 400 ms target.
- **(iv) Open3D `cluster_dbscan` (C++, parallel) — backlog #5.** Fixed-eps
  DBSCAN fights range-varying density; #21/#24 precedent says clustering swaps
  regress on held-out data. Run once for the thesis comparison table, not as a
  production candidate.

### A3. Ground removal (167 ms, 18%)
- **(a) Cut RANSAC budget (recommended now).** `ransac_iterations` 1000 → ~300
  (`PIPELINE_CONFIG`). The road plane is dominant; expect identical plane on
  nearly all frames. Changes the RNG stream → behavior-changing tier, cheap
  re-eval.
- **(b) Patchwork++ — backlog #6.** ~30 ms C++ and better on sloped roads,
  but: external integration → **mandatory `/semantics-map` + sign-off first**,
  and it returns per-patch planes while `filter_clusters` /
  `_center_height_above_ground` (`src/pipeline.py:349`) assume one global
  plane → interface redesign. Treat as a separate quality-driven direction,
  not a quick win.
- **(c) Keep as-is.**

### A4. Preprocessing (134 ms, 14%)
- **(a) Voxel-downsample before denoise (recommended).** Statistical outlier
  removal is KNN over ~110k points; running it after the 0.05 m voxel cut
  (~10× fewer) should save ~50–80 ms. Changes the surviving point set →
  behavior-changing tier.
- **(b) Drop denoise entirely.** Hypothesis: voxelization already suppresses
  isolated returns. Free 56 ms if metrics hold.
- **(c) Keep order.** Note `z_filter`'s 56 ms is mostly the numpy→Open3D
  `Vector3dVector` copy — trim independently of this decision (Tier 1).

---

## Section B — Build sequence (after A is locked)

0. ~~Wait for the running full-sequence benchmark~~ — done 2026-07-19: the
   official "before" is `timing_seq08_full_n4068.json` (921 ms/frame).
1. **Baseline snapshot:** `evaluate.py --seq 08 --frames 300` output saved as
   the exact-match reference (TP/FP/FN per frame).
2. **Tier 1 (exact-output):**
   - Batch classifier inference: new `classify_clusters_batch()` in
     `src/classifier.py` (per-cluster preprocessing unchanged — `sample_or_pad`
     seeds `default_rng(0)` per cluster, so results are order-independent);
     call sites `src/evaluate.py:222`, `src/main.py`. Expect 75 → ~10 ms.
   - `core_dist_n_jobs=-1` (A2-i), config key in `PIPELINE_CONFIG`.
   - Trim `z_filter`/conversion copies (keep numpy until Open3D needs it).
   - Verify: TP/FP/FN identical to step-1 reference; then
     `timing_benchmark.py --stride 20 --frames 204`.
3. **Tier 2 (cheap behavior-changing, one at a time, Experiment Protocol each):**
   RANSAC iterations (A3-a), preprocessing reorder / drop-denoise (A4).
   Judge on `--frames 300`, confirm any keeper on full seq 08.
4. **Tier 3:** coarse-voxel clustering (A2-ii) behind
   `PIPELINE_CONFIG["cluster_voxel_size"]` (off by default) — full protocol +
   split-rate measurement (#23 comparison).
5. **Tier 4 (conditional):** cuML/WSL2 (A2-iii) or Patchwork++ (A3-b, starts
   with `/semantics-map`).
6. **Finalize:** re-run full-sequence timing; update
   `results_overview_2026_07_19.docx` inference-time section,
   `docs/findings.md`, `docs/project_state.md`. If an algorithm swap was
   promoted under A1(a): regenerate `output/08`, re-run
   `scratchpad/completion_box_eval*.py` + `donor_metric_*.py`.

Rough budget if recommendations hold: 934 → ~780 (Tier 1) → ~620 (Tier 2) →
~350–420 ms/frame (Tier 3), i.e. ~2.5 fps without touching GPU clustering.

## Section C — Mechanical (skippable)

- Config keys + argparse overrides in `evaluate.py` for every new knob.
- Timing JSONs keep distinct names (`timing_seq08_*.json`); never overwrite.
- Findings entries per experiment; project_state update at the end.
- Old code paths stay CLI-toggleable (convention set by the clustering
  alternatives).
- VRAM cap for the classifier batch (chunk at 256 clusters; far above the
  observed ~41/frame).
