# Thesis Reframe + Restructure Plan

Status: Section A LOCKED 2026-09-04. Created 2026-09-03.
Supersedes the in-conversation 9→5 restructure sketch; folds it under the reframe.

## LOCKED DECISIONS (2026-09-04)

- **A1 = Opt B, ALL 11 labelled sequences (00–10)** — harden the input-quality headline
  across the whole labelled set, not just 2 sequences (breadth is the point of the retitle).
  Freeze-compliant (experiments into `output/experiments/`, no edits to frozen artifacts).

  **Ablation design (checked 2026-09-04):**
  - **One gate-OFF render per sequence** (11 total); gate-ON rate derived post-hoc as the
    footprint-passing subset (gate is deterministic on fit_length≥2.7 / fit_width≤2.3). Halves
    compute vs on+off. `output/08` (frozen, full-seq, gate-on, 518 completed tracks) = free
    cross-check for the post-hoc reconstruction. No other free runs (`output/00` is 20-frame
    no-completion; `output/05` GT-only; 8 sequences unrendered).
  - **Shipped classifier across all 11** (no per-fold retrain): headline is the gate on-vs-off
    delta; identical detections both arms → classifier in-sample advantage cancels in the delta.
    Report per-seq absolute rate (optimistic on in-sample seqs) + the delta + pooled.
  - **Compute:** ~23.2k frames × ~0.62 s (#34) ≈ 4–5 h, one resumable background job.
  - **Build (all freeze-safe):** additive `--no-gate` flag on `main.py` (not frozen; defaults
    gated) + save raw partials; `scripts/run_gate_ablation.ps1` (LOSO-style loop) →
    `output/experiments/gate_ablation_v2/<seq>/`; `scratchpad/gate_ablation_analyze.py`
    (imports frozen `completion.py` read-only for the footprint fit).
  - **Pre-registered read:** replicates if pooled gate-on plausibility rate materially exceeds
    gate-off AND the effect holds in the majority of per-sequence rows (low-n sparse seqs, esp.
    01 highway, get wide error bars / caveats, not exclusion). A null (no gate effect on the
    non-08 sequences) → headline stays scoped to seq 08, fall back to A2=Opt 1 title.
- **A2 = Opt 2** — retitle toward the segmentation-bottleneck finding. **Advisor sign-off
  required (cover change).** Draft candidates below; not final until approved.
- **A3 = Opt 1** — keep detection metrics as front-end characterization; recall chapter = bridge.
- **A4 = Opt 1** — promote #27/#28 to a first-class Ch 4 "Input quality vs model capacity" section.
- **A5 = Opt 1** — donor metric defined in Methodology, recalled in eval-setup, validated in eval.
- **A6 = Opt 1** — Theoretical Background w/ ~12–18 cited equations + global + per-stage diagrams.

**Coupling note:** A2=2 raises the bar on A1 — a cover that promises "the segmentation
bottleneck" must not rest on an n=47 seq-08-only result, which is exactly why A1=B. The two
choices reinforce each other.

### Draft title candidates for advisor (A2=2)
1. "Segmentation Quality as the Bottleneck in LiDAR Vehicle Completion: A Reference-Free
   Evaluation" (leads with the finding, keeps the metric as subtitle).
2. "From Detection to Completion: Why Input Segmentation, Not Model Capacity, Limits
   Occluded-Vehicle Reconstruction in Automotive LiDAR."
3. "Reference-Free Evaluation of Occluded-Vehicle Completion and the Segmentation Bottleneck
   in Automotive LiDAR" (co-headline: metric + finding; closest to the current title).

## The reframe in one sentence

Real-LiDAR vehicle completion is bottlenecked by **input segmentation quality, not
completion-model capacity**; the detection pipeline is the front-end that governs that
quality, and a reference-free donor metric is what makes the whole thing measurable.
Detection stops being a standalone SOTA claim and becomes the object of study's front-end.

## "Hypothesis" / success criterion (prose surgery, no numeric metric)

Reassembled `main.tex` builds clean (exit 0, zero undefined refs/citations, no new
>20pt overfull); **no frozen artifact touched** (`src/`, checkpoints, `output/08`);
**no number/finding-citation lost** at any fold (diff-audit); TOC matches the peer
5-chapter template; every headline claim traces to a frozen finding.

## Measured spine (all frozen, all pre-existing)

- **#27** — completion quality bottlenecked by input cleanliness; geometric input-gate
  lifts plausible-completion rate **38% → 69%** (seq 08, 300 frames, n=47 completed).
- **#28** — PCN ↔ PoinTr is a wash on real data (18/26 vs 16/26) → model choice ≈ irrelevant.
- **#23** — HDBSCAN splits 31–37% of cars; the *same* fragments/merges are #27's bad inputs
  (findings.md:671 draws this link explicitly). One bottleneck, two symptoms (recall + completion).
- **#32/#42** — completion adds value: coverage 0.000→0.304 vs mirror; amodal BEV IoU 0.725→0.771.
- **#26/#32/#37** — donor metric: Chamfer-invalidity, leakage-free construction, hallucination guard.

---

## Section A — Decisions that need the user (ordered by blast radius)

### A1. Headline strength: accept the thinness, or harden it? (BIGGEST — decides defensibility)
Promoting #27 to the *headline* rests it on n=47 completed tracks, one sequence (08),
two completion models (#28). Strong as a supporting finding; thin for a cover claim.
- **Opt A (reframe-only): accept and scope honestly.** Freeze respected, writing-only, days.
  Headline reads "on seq 08, input quality dominates model choice" — true, bounded.
- **Opt B (bounded harden): mini-unfreeze.** Re-run the gate on/off ablation across more
  frames + ≥1 more sequence (00 already mined) to widen n and add a 2nd-sequence replication;
  optionally a 3rd completion model. ~1 week, needs advisor buy-in. Upgrades headline from
  "shown on seq 08" to "replicated."
- Tradeoff: A ships faster and is defensible *if* scoped; B makes the new headline as solid as
  a headline should be. Echoes the earlier Path-A/Path-B call, now specifically about #27's n.
- *Recommendation:* **A unless the advisor signals the headline needs a 2nd sequence** — but
  write the scope sentence so an examiner can't mistake seq-08 n=47 for a general law.

### A2. Title / cover framing (needs advisor — it's on the cover)
Current title is metric-centric and completion-centric; it does NOT mention the
input-quality finding that A-reframe makes the headline.
- **Opt 1: keep the title, lead the abstract/intro with the input-quality spine.** Safest
  (no advisor-facing title change), metric stays a clean titled contribution. *Recommended.*
- Opt 2: retitle toward the bottleneck finding (e.g. "...on the Segmentation Bottleneck in
  LiDAR Vehicle Completion..."). Stronger match to the new headline; requires advisor sign-off
  and re-does the cover/front matter.
- Opt 3: co-headline title (metric + bottleneck). Longer, risk of a mouthful.
- *Blast radius:* title choice sets the intro funnel's endpoint and the abstract's first sentence.

### A3. Fate of detection evaluation + the recall chapter (the bridge)
Under the reframe, detection P/R/F1 (0.905/0.730/0.808) is front-end characterization, and
the recall-bottleneck story (#23/#34) becomes the *bridge* to completion ("splits cap recall
AND poison completion input").
- **Opt 1: keep detection metrics, re-narrate as front-end quality; recall chapter becomes the
  hinge that connects to completion.** *Recommended* — makes the unification visible.
- Opt 2: minimize detection numbers to an appendix. Loses the bridge; wastes measured work.
- *Downstream:* Opt 1 means Ch 4's detection section explicitly forward-refs the completion
  section via the shared split/merge mechanism.

### A4. Where the money experiments (#27 gate, #28 model-wash) live and how prominent
Currently buried in the completion-method chapter (sec_6).
- **Opt 1: promote to a dedicated Ch 4 evaluation section "Input quality vs model capacity"
  with #27's 38→69 table + #28's PCN/PoinTr wash as the companion.** *Recommended* — this is
  the headline result; it must be a first-class table, not a paragraph.
- Opt 2: leave in methodology, reference from eval. Under-sells the headline.

### A5. Donor-metric definition placement (carried from restructure sketch)
- **Opt 1: full construction + validation as a Methodology section (Ch 3.x); recalled briefly
  in eval-setup 4.1.3; validation-battery *results* in the completion-eval section.**
  *Recommended* — keeps the titled contribution a method, not plumbing.
- Opt 2: define literally in 4.1.3 (peer template slot). Template-faithful, demotes it.

### A6. New-math + per-block-diagram scope (carried; effort driver)
Thesis currently has 1 equation total. Peer wants a Theoretical Background w/ math + per-block
diagrams.
- **Opt 1 (recommended): Theoretical Background (2.B) with ~12–18 standard equations (RANSAC,
  voxel grid, HDBSCAN mutual-reachability + stability, PointNet symmetric fn, Chamfer, point-IoU,
  donor coverage), each citing its source paper; global pipeline diagram + one block diagram per
  non-trivial stage (clustering, geo filter, dual-branch classifier, completion). ~4–5 new TikZ.**
- Opt 2 (minimal): only contribution-critical math (~5 eqs), global diagram only. Under-delivers
  on the explicit peer demand.
- Opt 3 (heavy): full derivations. Padding for an MSc.

---

## Section B — Build sequence (once A locks)

1. **Checkpoint commit** current clean-building tree (safety net).
2. **(If A1=Opt B only)** run bounded gate/model ablations; record per Experiment Protocol
   (hypothesis, baseline #27 38→69, compare). Otherwise skip.
3. **Front matter:** add acknowledgments + declaration; wire before abstract.
4. **Abstract + Ch 1 intro rewrite** to the reframe funnel: AD context → occluded-vehicle
   completion problem → completion is input-bottlenecked (the gap) → contributions → RQs
   rewritten to match (RQ1 metric; RQ2 input-quality-vs-model; RQ3 the shared split/merge cause).
5. **Ch 2 split:** 2.A Literature Review (existing + reposition around completion/eval) +
   2.B Theoretical Background (new math, A6).
6. **Ch 3 Methodology merge:** sec_3 detection + sec_6 completion + donor-metric definition (A5);
   notation + per-block diagrams (A6).
7. **Ch 4 Evaluation rebuild** to 4.1 setup / 4.2 detection front-end + runtime (A3) /
   4.3 completion result incl. the **input-quality-vs-model headline table** (A4) + donor
   validation battery / 4.4 ablations (recall strategies #21/#24, Stage-A #25, cross-domain #30) /
   4.5 error analysis (distance recall #50, failure cases, limitations from sec_8).
8. **Ch 5 Conclusion:** sec_9 + consolidated Limitations + "what generalises" from sec_8.
9. **Rewire main.tex**, fix cross-refs, build-verify + fold diff-audit.

## Section C — Mechanical (skippable)

- Heading-level demotion at each merge; stale MERGE/NUMBERING comment cleanup.
- `\label`/`\ref` renaming to avoid collisions.
- Figure-path checks after moves (`\graphicspath` unchanged).
- Old sec_*.tex: keep for git history vs delete (cosmetic).

---

## Honesty boundaries (do not cross)

- No statistical-denoise→completion ablation exists → do NOT headline "denoising helps
  completion." Headline the **geometric gate** (#27) and **segmentation-as-precondition**.
- Segmentation is a *precondition* for PCN (single-object input), not a measured % improvement.
- "Model choice doesn't matter" rests on 2 models (#28) — state that boundary.
- Completion scope unchanged: static, gate-passed, compact/normal cars, seq 08; held-out
  PARTIALLY HOLDS (#42).
- The recall↔completion unification is an **inference** linking #23 and #27 — but one
  findings.md:671 already draws.
