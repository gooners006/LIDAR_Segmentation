# T13 / Step 1c — residual completion under-extension: plan + pre-registration

Date: 2026-08-09. Status: **registered before any Step-1c eval was run.** This is
the immutable grading key for T13 (delegate brief, Tier 3). Produced via
`/tweakable-plan`; decisions D1–D3 locked by the user 2026-08-09. Scope: option 3
(length-dependent target) stays OUT. No `PIPELINE_CONFIG`/completion-constant
change ships unless the pre-registered gate below passes.

## Hypothesis

Normal/long-band completions land too short for two coupled reasons: (a) the Z
length-push moves `center` toward the occluded far end, which inflates
`radius = ‖pts_c − center‖.max()/1.137` (`completion.py:499`), so output scale
rides on the length prior instead of a principled estimate; and (b) PCN
under-fills its normalized frame. Decoupling `radius` from the Z push and
applying a fill-factor calibrated on synthetic true GT will lengthen normal/long
completions toward GT **without over-extending compacts**.

**Metric expected to improve:** band-split #29 box `|ΔL|` and BEV IoU on
**normal/long** bands, and #32 far_end coverage. Headroom is measured: #36's
q90+0.45 control gave normal/long `|ΔL|` 0.304→0.210, cov 0.364→0.483, far_end
0.316→0.509 — but through the wrong mechanism (abusing the center push). Compact
band must stay flat (#36's per-car estimate fixed compact overshoot).

## Locked decisions (D1–D3)

- **D1 = 1a (Z-only radius decouple).** Compute `radius` against the center
  *without* the Z length-push (keep X-width, Y-up); use the full pushed center
  only as the reconstruction offset. Preserves the #26 ×1.137 partial-derived
  scale and #35 extend-only semantics; compacts (~zero Z push) barely change.
- **D2 = 2a (length-axis fill, confirm on synthetic).** Default: correct the Z
  extent only, by a constant `fill_Z` = median(GT Z-extent / PCN-output
  Z-extent) over KITTI-like synthetic val (true GT). **Pre-registered widen
  rule:** measure per-axis fill first; adopt a W or H correction ONLY if that
  axis's median fill exceeds **1.10** (i.e. ≥10% under-fill). Otherwise leave
  W/H untouched — #29 `|ΔW|` 0.165 / `|ΔH|` 0.120 are already strong (#36).
  Calibrated on synthetic, applied (not refit) on real.
- **D3 = 3a (seq-08-only long-band evidence).** seq 00's long band is empty
  (T9c R3, 0 cars ≥4.6 m), so long-band improvement is judged on **seq 08**;
  seq 00 gates compact+normal non-regression only.

## Build sequence

1. **Baseline capture** — run the eval harness below on current production
   (shipped per-car length estimate ON), both sequences; save before-tables.
2. **D1 radius decouple** — edit `estimate_canonical_frame`
   (`completion.py:481–499`) so `radius` uses the non-Z-pushed center, behind an
   A/B flag. Re-run eval; verify on synthetic true GT that length recovery moves
   the right direction.
3. **D2 measure** — synthetic per-axis median fill (KITTI-like val, large n);
   apply the widen rule to pick 2a vs a W/H correction.
4. **D2 apply** — thread the constant fill factor into reconstruction; re-run
   eval both sequences.
5. **Ship decision** — apply the gate verbatim. Fail on either sequence →
   negative result (house precedent #16/#17/#19/#40); document, do not ship.

**Eval harness (both seq 08 n=40, seq 00 n=45):** `donor_metric_step1 →
donor_metric_recompute → donor_metric_step2 → donor_metric_step3 (+d2) →
length_1b_box_eval → t9b_box_all_wilcoxon`. Per-car medians, Wilcoxon α=.05.
Metrics: band-split #29 box (BEV IoU, signed ΔL, |ΔL|/|ΔW|/|ΔH|, center err) +
#32 donor (cov@0.1, far_end, regions) + per-band d2.

## Pre-registered ship gate (applied verbatim; no post-hoc thresholds)

**SHIP** iff ALL hold:

- **Primary (seq 08):** normal-band median `|ΔL|` decreases by **≥0.03 m**
  (Wilcoxon p<.05), AND long-band median `|ΔL|` is directionally down (same
  sign; no p requirement, small n).
- **Compact non-regression (both seqs):** compact-band median `|ΔL|` not worse
  by **>0.02 m**, and compact BEV IoU not down by **>0.01**.
- **Normal non-regression on the held-out seq (seq 00):** normal-band `|ΔL|`
  not worse by >0.02 m, BEV IoU not down >0.01.
- **Donor guard (both seqs, every testable band):** per-band cov@0.1 not down
  by **>0.02**; far_end cov not down by >0.02.
- **d2 guard (both seqs, every testable band):** no band's per-band d2 pass-bit
  flips pass→fail, and no passing band's completed/mirrored ratio worsens beyond
  its current level (the #37 guard, seq-08 long band included).

**Otherwise → NEGATIVE RESULT.** Record the tables, keep the A/B flag defaulted
OFF, do not change constants. A primary improvement that trips any guard =
report with the caveat, do NOT ship silently.

## Out of scope (pre-committed)

- Length-dependent fill target (option 3): excluded.
- Heading/center-on-diagonal work (Step 2): separate task.
- Any architecture change (ruled out #28).
