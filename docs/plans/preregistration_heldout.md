# Held-out replication pre-registration (seq 00)

Date: 2026-08-09. Status: registered before any seq-00 completion eval was run.
Held-out sequence: **seq 00** (T8; default seq 05 failed the well-observed
guard, 11 < 15). Confound (logged, accepted): seq 00 is in the classifier's
Stage-B training split, so detection recall reads optimistic; the completion
claims are unaffected because the comparison is paired (raw vs completed on the
same TP inputs) and PCN trained only on synthetic. No labeled sequence is both
classifier-held-out and completion-held-out.

**Fixed before running (no tuning of any kind):** production `PIPELINE_CONFIG`;
production checkpoints (`stage_b_scratch_best.pth`, `pcn_kitti_best.pth`);
shipped completion with the per-car length estimate ON
(`track_length_estimate`, q90+0.12, 4.14 m fallback). Metrics: #29 box eval and
#32 donor metric (steps 1–3 + the #37 per-band d2 guard). Unit: per-car
medians over static well-observed cars; significance by Wilcoxon signed-rank,
α = .05. Seq-08 reference values below are the production-config numbers from
Findings #29/#36/#37 (n=40).

## Expected outcomes (directional predictions)

1. **Box (#29): completed beats raw partial on BEV IoU** (seq-08 ref
   0.725 → 0.771), median, Wilcoxon p < .05. Also expected better (secondary,
   reported not refutation-bearing): |ΔW| (0.270 → 0.170), |ΔH| (0.255 → 0.131),
   |ΔL| (0.428 → 0.304), center err (0.271 → 0.184). Yaw expected neutral.
2. **Donor (#32): cov@0.1 (τ=0.15) ordering completed ≫ mirrored ≫ raw**
   (seq-08 ref 0.364 ≫ 0.043 ≫ 0.000), completed vs raw Wilcoxon p < .05; median
   novel-distance completed < raw.
3. **Magnitudes may shrink vs seq 08.** Seq 00's well-observed set skews compact
   (median L 3.80 vs 4.14) and all completion constants were tuned on seq 08, so
   smaller deltas are anticipated and do **not** by themselves count as failure.

## Refutation criteria (pre-committed; applied verbatim in T9c)

- **R1 — does not generalize.** If completed **fails to beat raw on BEV IoU**
  (median, Wilcoxon p < .05) **OR fails to beat raw on donor cov@0.1** (median,
  Wilcoxon p < .05), then "completion adds value" does **not** generalize; the
  thesis scopes the claim to seq 08. **Do not retune.**
- **R2 — 08-specific length constants.** If the #37 per-band d2 out-of-box
  violation is **worse than seq-08 levels** — i.e. any band's completed /
  per-band-mirrored ratio exceeds its seq-08 completed level (compact ≈16×
  baseline @ 0.0065/0.0004; normal ≈0.12× @ 0.0015/0.0129; long 0× @
  0.0000/0.0292), or a band that passed on 08 fails on 00 — then the length
  constants are 08-specific; **report as a limitation, do not retune constants.**
- **R3 — uncovered result.** Any result this pre-registration does not cover
  (e.g. box holds but donor does not, or vice versa; band Ns too small to test)
  → **STOP and escalate to the user with the tables** (per T9c). Do not
  improvise a verdict.

## Outcome taxonomy (for T9c and the Tier-3 gate)

- **HOLDS** — completed beats raw on **both** BEV IoU and donor cov@0.1
  (each Wilcoxon p < .05) and no R2 d2 regression.
- **PARTIALLY HOLDS** — completed beats raw on BEV IoU and donor cov@0.1 but
  with an R2 d2 caveat, OR one primary metric is a clean win and the other is
  directionally positive but not significant (n-limited). Reported with the
  caveat named.
- **DOES NOT GENERALIZE** — R1 triggers.

(Tier 3 / T13 proceeds only if the T9c outcome is HOLDS or PARTIALLY HOLDS.)
