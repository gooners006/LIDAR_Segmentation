# Canva AI — expand the 19-slide deck to Deck v2 (2026-10-14)

Target: `outline.md` → "Deck v2". Run the three prompts **in order, one at a time**, on a
copy of the design (File → Make a copy first). Export PNG after each and check before the
next. Numbers stay out of the prompts; they're typed by hand from the thesis later.

Layout keys used below (all inside the content area y = 220–960; the pipeline strip was dropped 2026-10-14):
- FULL: one full-width grey placeholder.
- TWO: two equal placeholders side by side, 40 px gap.
- CARDS-3 / CARDS-4: 3 or 4 equal placeholder cards in one row, 40 px gaps, each with the given label.
- TABLE: one full-width placeholder labelled as a table.

## Shared preamble (paste at the top of EVERY prompt)

```
Edit this existing deck. Every NEW content slide must copy the exact layout of the existing content slides: centred UPPERCASE Bold title at y = 30 size 44, centred key message at y = 105 size 30, content area y = 220 to 960, the dark footer bar with the section name (white, Bold Italic, UPPERCASE, size 24) and slide number. Every NEW divider copies the existing divider slides. Placeholders are empty light-grey rectangles with the given label centred inside, size 20. Use all text VERBATIM; add no other text, numbers, icons or images. Do not change existing images. Do not leave any stray rectangle on top of a placeholder. Renumber all slide numbers at the end. Every footer section name on every slide must be white, Bold Italic (fix any that are dark). Every title I give, including retitles, is displayed in UPPERCASE even though I type it in Title Case.
```

## Prompt 1 — Section I and II

```
[preamble]

Make slides 1-19 of the deck exactly this sequence. "EXISTING: X" = keep that slide, apply only the changes given. "NEW" = insert a new slide.

1. EXISTING: cover. 2. EXISTING: Table of Contents. 3. EXISTING: divider "I. INTRODUCTION".
4. NEW. Footer: I. INTRODUCTION. Title: 1. Background: LiDAR Point Clouds. Key message: A LiDAR scan is an unordered set of 3D points that gets sparser with range and sees each object from one side. Layout CARDS-3: "[ unordered points ]", "[ sparser with range ]", "[ one-sided view ]".
5. EXISTING: "1. The Problem: A One-Sided View" → retitle "2. The Problem: A One-Sided View".
6. EXISTING: "2. Two Ways to Solve It" → retitle "3. Current Approaches".
7. NEW. Footer: I. INTRODUCTION. Title: 4. Literature Review. Key message: Three lines of prior work, and what each leaves open. Layout TABLE: "[ table: approach / representative works / strengths / remaining challenges ]".
8. NEW. Footer: I. INTRODUCTION. Title: 5. Research Gap. Key message: Accurate detectors need dense labels and are hard to inspect, and real-world completion has no ground truth to train or test against. Layout CARDS-4: "[ dense labels ]", "[ opacity ]", "[ no paired real data ]", "[ no ground truth for real completion ]".
9. EXISTING: "3. Objectives and Contributions" → retitle "6. Objectives and Contributions".
10. EXISTING: divider "II. PROPOSED METHODOLOGY".
11. EXISTING: "1. Framework Overview" → retitle "1. System Architecture".
12. EXISTING: "2. Finding Object Candidates" → retitle "2.1 Preprocessing and Ground Removal"; key message: Clean the scan, then remove the ground plane with RANSAC. Make "[ D4 RANSAC plane fit ]" FULL width and delete the D5 placeholder from this slide.
13. NEW. Footer: II. PROPOSED METHODOLOGY. Title: 2.2 Clustering (HDBSCAN). Key message: HDBSCAN groups the remaining points into object candidates. Layout FULL: "[ D5 clusters before/after ]".
14. NEW. Footer: II. PROPOSED METHODOLOGY. Title: 2.3 Geometric Filter. Key message: Size and shape rules on each cluster's oriented box drop obvious non-cars. Layout FULL: "[ fig: geometric filter ]".
15. EXISTING: "3. Keeping Only Cars" → retitle "2.4 Car / Not-Car Classifier"; key message: A small PointNet decides car or not-car, trained on labels mined automatically from SemanticKITTI.
16. EXISTING: "4. Tracking Removes Flicker" → retitle "2.5 Tracking and Track Filter".
17. EXISTING: "5. Completing the Car" → retitle "3.1 Completion Network (PCN)".
18. NEW. Footer: II. PROPOSED METHODOLOGY. Title: 3.2 Synthetic Training Partials. Key message: Training inputs are simulated LiDAR-style partial views of synthetic ShapeNet cars. Layout FULL: "[ fig: KITTI-like partials ]".
19. NEW. Footer: II. PROPOSED METHODOLOGY. Title: 3.3 Normalisation, L-Shape Gate and Length. Key message: Each real cluster is placed in the network's training frame, gated by its L-shape, and given an estimated length. Layout TWO: "[ fig: L-shape gate ]", "[ completion settings ]".
All remaining existing slides keep their order after slide 19.
```

## Prompt 2 — Section III

Canva AI stopped after the first new slide in prompt 1 and asked whether to continue; if it does
that again, reply: "Continue with ALL remaining numbered items without stopping."

```
[preamble]

FIRST fix slides 1-19:
- Every slide title on every content slide must be in UPPERCASE (several are in Title Case now, e.g. "2.2 Clustering (HDBSCAN)" must read "2.2 CLUSTERING (HDBSCAN)").
- Slide 7 title must be "4. LITERATURE REVIEW" (not 5). Slide 8 title must be "5. RESEARCH GAP" (not 6). Slide 9 stays "6. OBJECTIVES AND CONTRIBUTIONS".
- DELETE the "[ pipeline progress strip ]" bar from EVERY slide. Then stretch every content placeholder, image and text column on every content slide so its bottom edge is at y = 960 (top stays at y = 220; keep horizontal positions and gaps). Exception: on "2. THE PROBLEM: A ONE-SIDED VIEW" do not resize the two images; only move the caption so it stays directly under the right image. Do not stretch or distort any image. Nothing may enter the footer bar (y >= 1000).

THEN, after slide 19, make the deck continue exactly like this:

20. EXISTING: divider "III. EXPERIMENT EVALUATION" → add a second line under it, same style, size 44: "1. Dataset and Protocol".
21. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 1.1 Datasets. Key message: Real SemanticKITTI scans for detection; synthetic ShapeNet cars for completion. Layout TWO: "[ table: dataset roles ]", "[ table: sequences and sensor ]".
22. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 1.2 Target Eligibility and Ground Truth. Key message: A car is scored only if enough of its points survive preprocessing. Layout FULL: "[ eligibility rule + example ]".
23. NEW divider: "III. EXPERIMENT EVALUATION" with second line "2. Experiment Setup".
24. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 2.1 Baselines. Key message: Stages are compared by ablation; completion is compared with the raw partial, a dense reference, and PoinTr. Layout TABLE: "[ table: baselines ]".
25. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 2.2 Metrics. Key message: Detections are matched to real cars by point overlap (IoU); completions are scored by an independent car classifier. Layout TWO: "[ detection metrics: IoU matching ]", "[ completion metric: plausibility ]".
26. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 2.3 Data Split: Leave-One-Sequence-Out. Key message: Train on ten sequences, test on the eleventh, and repeat for every sequence. Layout FULL: "[ LOSO diagram ]".
27. NEW divider: "III. EXPERIMENT EVALUATION" with second line "3. Results".
28. EXISTING: "1. Detection Results" → retitle "3.1 Detection Results".
29. EXISTING: "2. What Limits Recall" → retitle "3.2 What Limits Recall".
30. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 3.3 Accuracy vs Range. Key message: Recall falls with distance as cars get sparser. Layout TABLE: "[ table: accuracy by distance ]".
31. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 3.4 Completion: Synthetic Fidelity. Key message: On held-out synthetic cars, the network reproduces the complete shape. Layout TWO: "[ synthetic examples ]", "[ CD and F-score ]".
32. EXISTING: "3. Judging Completion Without Ground Truth" → retitle "3.5 Judging Completion Without Ground Truth".
33. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 3.6 Plausibility Robustness. Key message: The plausibility gain holds on every sequence, for moving cars, and under a heading flip. Layout TWO: "[ table: per-sequence plausibility ]", "[ movers / flip / McNemar ]".
34. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 3.7 Result Analysis. Key message: What each stage and each completion input contributes. Layout CARDS-3: "[ detection stages ]", "[ completion inputs ]", "[ limits ]".
35. NEW divider: "III. EXPERIMENT EVALUATION" with second line "4. Runtime".
36. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 4.1 Runtime. Key message: The pipeline runs offline; the classical CPU stages dominate the time per scan. Layout TABLE: "[ table: runtime per stage ]".
37. NEW divider: "III. EXPERIMENT EVALUATION" with second line "5. Ablation Study".
38. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 5.1 Stage Ablation. Key message: Removing each stage in turn shows what it contributes. Layout TABLE: "[ table: stage ablation ]".
39. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 5.2 Clustering Alternatives and Resolution. Key message: No faster clusterer matches HDBSCAN's recall; a coarser resolution reduces fragmentation. Layout TWO: "[ table: clustering comparison ]", "[ table: split rate by resolution ]".
40. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 5.3 IoU-Threshold Sensitivity. Key message: How the results change with the matching threshold. Layout TABLE: "[ table: IoU sensitivity ]".
41. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 5.4 Synthetic Pretraining. Key message: Synthetic pretraining adds nothing once enough real clusters are mined. Layout TABLE: "[ table: Stage A vs Stage B ]".
42. NEW. Footer: III. EXPERIMENT EVALUATION. Title: 5.5 Completion Ablations. Key message: Comparing PCN with PoinTr, and the effect of the completion gates. Layout TABLE: "[ table: completion ablations ]".
All remaining existing slides keep their order after slide 42.
```

## Prompt 3 — Section IV, end, backups

```
[preamble]

FIRST fix slides 1-45 (do every item; do not ask whether to continue):
- Slide 23 is wrong: rebuild it as a divider exactly like slide 27 (centred two-line title "III. EXPERIMENT EVALUATION" / "2. EXPERIMENT SETUP") and delete its two grey rectangles.
- All divider slides (3, 10, 20, 23, 27, 35, 37, 43) look identical: same font as slides 3 and 10 (my Brand Kit font), first line Bold size 56, optional second line Bold size 44, both centred; footer bar shows ONLY the slide number, no section name.
- Slides 21-42: every text in my Brand Kit font (the same font as slides 1-19). Titles Bold size 44; key messages Medium size 30, not bold; footer section name size 24 on ONE line.
- Add the missing placeholder labels (size 20, centred in the grey rectangle): slide 22 "[ eligibility rule + example ]", slide 24 "[ table: baselines ]", slide 30 "[ table: accuracy by distance ]", slide 36 "[ table: runtime per stage ]", slide 40 "[ table: IoU sensitivity ]", slide 42 "[ table: completion ablations ]". Slide 38's label must be size 20 like the others.
- Slide 39 key message: fix the typo "redices" to "reduces".

THEN, after slide 42, make the deck continue exactly like this:

43. EXISTING: divider "IV. CONCLUSION & FUTURE WORK".
44. EXISTING: "1. Conclusions, Limitations and Future Work" → retitle "1. Research Summary"; key message: What we built and what we found. Delete the "Limits and next steps" column and spread the other two columns across the width.
45. NEW. Footer: IV. CONCLUSION & FUTURE WORK. Title: 2. Limitations. Key message: Where the results stop. Bullets, Regular size 32:
- Cars only
- One dataset and one sensor
- Offline, not real-time
- Recall capped by clustering fragmentation
46. NEW. Footer: IV. CONCLUSION & FUTURE WORK. Title: 3. Discussion and Future Work. Key message: Three next steps, each scoped by a finding. Layout CARDS-3: "[ real-time: faster HDBSCAN implementation ]", "[ real-data fine-tuning for completion ]", "[ completion for downstream detection ]".
47. EXISTING: "Thank You" slide → change the title to "THE END." and the line under it to "THANKS FOR ATTENDING AND LISTENING.". Keep the D3 placeholder. Add the university logo centred above the title (copy it from the cover).
48-53. NEW backup slides, footer section name "BACKUP", each with one FULL placeholder labelled with its title:
48 "Repair Strategies and Why They Fail" · 49 "The Normalisation Bug" · 50 "Is the Judge Independent?" · 51 "Fidelity Metrics: UCD and UHD" · 52 "Worked Example: IoU Matching" · 53 "Pipeline Configuration".
For backups, the key message line is empty.
```

## After each prompt — check

- Slide count and order match `outline.md` Deck v2 (26 → 45 → 53).
- Footer section names correct; dividers have no footer name.
- No invented text/numbers; nothing hidden behind rectangles; nothing enters the footer bar.
- Export PNG (all pages) into `canva_check/` (clear the folder first) → I check.
