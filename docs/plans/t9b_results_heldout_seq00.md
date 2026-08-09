# T9b results — seq-00 held-out replication (tables for T9c)

Date: 2026-08-09. Executor: this session. **This document reports numbers only.
It deliberately does NOT apply the pre-registration's verdict** — that is T9c, a
separate fresh session (executor/judge separation, per the delegate brief). Read
alongside `docs/plans/preregistration_heldout.md`.

## Provenance (frozen, no tuning)

- Held-out sequence: **seq 00**; amodal GT `output/00/amodal_gt.json` (T8, 46
  well-observed static cars).
- Config: production `PIPELINE_CONFIG`; checkpoints `stage_b_scratch_best.pth` +
  `pcn_kitti_best.pth`; shipped completion with the per-car length estimate ON
  (`donor_metric_recompute.py --length-mode track-q90off`, i.e. q90+0.12,
  fallback 4.14, min_frames 5 — verified to match `completion.py` production
  constants). `L_est` median 3.93 m, range [3.43, 4.90].
- Pipeline: `donor_metric_step1.py` → `donor_metric_recompute.py` →
  `donor_metric_step2.py` → `donor_metric_step3.py` (+ d2) → `length_1b_box_eval.py`
  → `t9b_box_all_wilcoxon.py` (pooled ALL-cars Wilcoxon; length_1b only does
  per-band). Dirs: `output/experiments/donor_metric_00{,_lenon}` (gitignored).
- Seq-08 reference columns below are the production-config numbers from Findings
  #36/#37 (n=40).

## Sample sizes

- **n_pairs = 2588** TP pairs on well-observed cars; **1592 gate-passed**
  (963 fragment_input + 33 merge_suspected rejected by the #27 L-shape gate).
- **n_cars = 45** (of 46 well-observed; one car had all pairs gate-rejected).
- Length-band split (amodal-GT length, bands fixed at 3.6/4.6 m):
  **compact <3.6 = 17 cars, normal = 28, long ≥4.6 = 0 cars.**

## Donor metric (#32), per-car medians, n=45 cars

| tau | method | cov@0.1 | med novel-dist (m) | completed_vs_raw p | completed_vs_mirrored p |
|-----|--------|---------|--------------------|--------------------|-------------------------|
| 0.10 | raw | 0.000 | 0.322 | — | — |
| 0.10 | mirrored | 0.070 | 0.213 | — | — |
| 0.10 | completed | 0.447 | 0.110 | ~0 (p=0.0) | ~0 (p=0.0) |
| **0.15 (primary)** | raw | 0.000 | 0.417 | — | — |
| **0.15** | mirrored | 0.050 | 0.280 | — | — |
| **0.15** | **completed** | **0.413** | **0.123** | **~0 (p=0.0)** | **~0 (p=0.0)** |
| 0.20 | raw | 0.000 | 0.488 | — | — |
| 0.20 | mirrored | 0.041 | 0.329 | — | — |
| 0.20 | completed | 0.401 | 0.125 | ~0 (p=0.0) | ~0 (p=0.0) |

Ordering completed ≫ mirrored ≫ raw at every tau; raw ranks last (cov 0.000 by
construction). Seq-08 primary-tau reference: cov@0.1 raw 0.000 / mirrored ~0.043
/ completed 0.364.

**Validation gate:** a_raw_last = True; c_ranking_stable_across_tau = True;
d_completed_not_worse_than_mirrored = True (pooled out-of-box: raw 0.000,
mirrored 0.0128, completed 0.0010); median per-car IQR completed cov = 0.176.

**Region breakdown (per-car medians, primary tau, completed vs mirrored):**
far_side 0.431 vs 0.082 (p=0.0); far_end 0.366 vs 0.007 (p=0.0); top 0.237 vs
0.083 (p=1.5e-5). Symmetry self-CD (completed) 0.124 m.

### d2 per-band hallucination guard (#37) — completed / per-band mirrored

| band | n_cars | completed | mirrored | pass-bit (completed ≤ mirrored) | seq-08 completed/mirrored |
|------|--------|-----------|----------|---------------------------------|---------------------------|
| compact <3.6 | 17 | 0.0090 | 0.0050 | **False** | 0.0065 / 0.0004 (~16×) |
| normal | 28 | 0.0003 | 0.0144 | True | 0.0015 / 0.0129 |
| long ≥4.6 | 0 | — | — | N/A (no cars) | 0.0000 / 0.0292 |

d2_all_bands_pass = False (compact fails the pass-bit). **Note for T9c (neutral):**
the compact **ratio** here is ~1.8× (0.0090/0.0050), vs seq-08's compact ~16×
(0.0065/0.0004) — the seq-00 compact mirrored baseline is itself much higher
(0.0050 vs 0.0004) because seq-00 compacts are less box-tight. Pre-registration
R2 is worded against "the seq-08 completed level (compact ≈16×) … or a band that
passed on 08 fails on 00"; compact did not pass on 08 either. T9c applies R2.

## Box metric (#29), per-car medians, raw vs completed (=on)

| band (n) | metric | raw | completed | Wilcoxon p |
|----------|--------|-----|-----------|------------|
| compact <3.6 (17) | BEV IoU | 0.783 | 0.771 | 0.85 (n.s.) |
| | signed_dL | +0.012 | +0.130 | 5.1e-2 |
| | \|ΔL\| | 0.141 | 0.185 | 1.0 (n.s.) |
| | \|ΔW\| | 0.174 | 0.165 | 0.26 (n.s.) |
| | \|ΔH\| | 0.276 | 0.111 | 3.1e-5 |
| | center_err | 0.168 | 0.213 | 0.35 (n.s.) |
| normal (28) | BEV IoU | 0.731 | 0.765 | 3.2e-5 |
| | signed_dL | −0.391 | −0.289 | 5.3e-4 |
| | \|ΔL\| | 0.392 | 0.294 | 4.2e-5 |
| | \|ΔW\| | 0.208 | 0.163 | 2.9e-4 |
| | \|ΔH\| | 0.280 | 0.123 | 7.5e-9 |
| | center_err | 0.267 | 0.188 | 8.9e-3 |
| long ≥4.6 (0) | — | — | — | no cars |
| **ALL (45)** | **BEV IoU** | **0.739** | **0.766** | **1.6e-3** |
| | signed_dL | −0.232 | −0.067 | 6.1e-5 |
| | \|ΔL\| | 0.359 | 0.227 | 2.0e-3 |
| | \|ΔW\| | 0.203 | 0.165 | 4.0e-4 |
| | \|ΔH\| | 0.278 | 0.120 | 7.8e-9 |
| | yaw_err | 3.000 | 3.250 | 0.40 (n.s.) |
| | center_err | 0.241 | 0.201 | 8.2e-2 (n.s.) |

Seq-08 ALL reference (#36): BEV IoU 0.725→0.771, |ΔL| 0.428→0.304,
|ΔW| 0.270→0.170, |ΔH| 0.255→0.131, center_err 0.271→0.184.

## Neutral flags for T9c (not verdicts)

1. **Long band is empty (0 cars ≥ 4.6 m).** Seq-00's well-observed set is all
   compact/normal. The pre-registration's long-band predictions and the long-band
   d2 check are **untestable** on seq 00 — potential R3 ("band Ns too small").
2. **BEV IoU splits by band:** pooled win (0.739→0.766, p=1.6e-3) driven by the
   28 normal cars (p=3.2e-5); the 17 compact cars are neutral/slightly negative
   (0.783→0.771, p=0.85). Compacts start high (raw 0.783) — little headroom.
3. **Donor cov@0.1** is an unambiguous, large, significant win across all bands,
   taus, and regions.
4. **d2 compact pass-bit fails** (see table) — R2-relevant; ratio context given.
