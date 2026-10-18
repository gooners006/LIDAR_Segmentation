# S6 — Completion Evaluation: Synthetic Fidelity, Plausibility Judge, Ceiling, Ablations

Study notes for slides 3.2.1 (baselines), 3.4.2 (judge, ★), 3.4.3 (synthetic fidelity), 3.4.4
(plausibility robustness), 3.4.5 (completion result analysis), 3.6.5 (completion ablations); diagrams
D12a (user) and D12b, D16. All facts come from the submitted thesis (`docs/writing/thesis/` ch4
§Completion Evaluation, §Baseline Methods; ch2 §completion metrics; ch5) unless tagged.
**[code]** = checked in `src/` or `scratchpad/plausibility/`, not stated in the thesis.
**[findings #n]** = `docs/findings.md`, not in the thesis. **[measured, not in thesis]** = counted
this session. **[illustration]** = my own example. Don't quote tagged numbers as thesis results.

---

## Part A — Concepts

### A1. Why completion can't be scored like detection

**Core meaning.** Detection has an answer key: SemanticKITTI labels every car point. Completion
predicts the car's **hidden** side, and on a real scan nobody ever observed it, so there is **no
complete true shape** to measure against [ch1 §1.2, ch2].

**The obvious substitute fails.** Stack the same parked car over many frames? The car is still
seen from the same road, so the stack is **one-sided too** [ch1]. Scoring a completion against it
rewards a network that copies the visible side and adds nothing.

**So the thesis splits the question in two** [ch4 §Completion Results]:
1. **Does the network work?** Test on synthetic cars, where the true model exists (A2).
2. **Do real completions look like cars?** Ask an independent judge (A4), with the standard
   reference-free metrics reported beside it and their caveat stated (A3).

### A2. Synthetic fidelity: the network in distribution (slide 37)

**Numbers** [ch4 §Synthetic Fidelity]: held-out synthetic cars, **Chamfer distance 0.16 m**,
**F-score @ 0.1 m 0.76**. Best validation loss **0.1246**.

**Chamfer distance (CD)**: for every output point, the distance to the nearest true point,
averaged; the same from the true model to the output; add the two (S5 A2 has the worked example).
In metres, because the cars are at real size.

**F-score @ 0.1 m, worked [illustration].** Output 4,096 points, true model 4,096 points.
- **Precision** = share of output points within 0.1 m of the true model. Say 3,200 / 4,096 = 0.78.
- **Recall** = share of true points within 0.1 m of the output. Say 3,000 / 4,096 = 0.73.
- F = 2 · P · R / (P + R) = 2 · 0.78 · 0.73 / 1.51 = **0.76**.

Why both: precision alone is fooled by an output that covers only the easy part; recall alone is
fooled by points sprayed everywhere.

**Don't mix 0.1246 with 0.16.** 0.1246 is the training loss (coarse CD + 0.5 × fine CD) in the
**normalised unit-sphere frame**; 0.16 m is CD in metres on the held-out set. Different units,
different formula [findings #28 corrects exactly this mix-up].

**How it was measured [findings #26, #28; reproduced this session, measured, not in thesis].** The
first **30 car models of PCN's validation split** (never trained on), each given a simulated scan,
placed exactly as in training (**true centre, true radius**), completed, scored in metres:
CD **0.161 ± 0.022 m**, F **0.754** (`scripts/make_d16_synthetic_example.py`; the thesis rounds to
0.16 / 0.76). The D16 car is the median-CD one of the 30 (0.158 m, F 0.771), picked by rule.

**What it shows / doesn't.** The network has learned "one-sided scan → whole car" for cars like
its training data. It says **nothing** about real LiDAR [ch4, verbatim spirit]. That's why A4
exists.

### A3. Reference-free fidelity: UCD and UHD, and why they reward doing nothing (slide 39)

**Core meaning.** The field's standard metrics for real cars compare a completion **with its own
input** [ch2, Chen et al. 2020, P2C 2023]:
- **UCD** (unidirectional Chamfer): for each **scan** point, the distance to the nearest point of
  the completed cloud; average.
- **UHD** (unidirectional Hausdorff): the **largest** of those distances.
Direction: scan → cloud. "Did the completion keep what was seen?"

**Worked example [illustration].** Scan = {A, B}. Completion 1 = {A, B} (adds nothing). Every scan
point finds itself at distance 0 → UCD = UHD = **0, the best possible score**. Completion 2 = {A',
B', + 3,000 new points}, where A' and B' sit 0.05 m from A and B → UCD = 0.05. **The completion
that actually filled the hidden side scores worse.** Any cloud that **contains** the scan scores 0,
however much junk it adds (the "mirrored" row: scan + its mirror copy).

**Thesis numbers** [tab:fidelity; seq 08, 39 cars, per-car medians]: raw 0.000 / mirrored 0.000 /
completed **UCD 0.090 m, UHD max 0.286 m, UHD p95 0.183 m**.

**Reading** [ch4]: not a negative result about completion; it is a property of the metric. "An
input fidelity measure is minimised by the cloud that adds nothing." That's why Chen et al. pair
UHD with a plausibility term, and why the thesis rests its real-data claim on plausibility.

### A4. The plausibility judge (Chen et al. 2020 protocol) (slide 36, ★)

**Core meaning.** Instead of "how close to the truth?", ask "**does it look like a car** to someone
who has no stake in our network?" [ch4 §Real Data Plausibility].

**The judge.** **PointNet++** (single-scale grouping), trained **from scratch on ModelNet40**: 40
everyday object classes (airplane, chair, car, ...), CAD models from a **different repository** than
the ShapeNet shapes PCN learned from. So a car the judge accepts is not an artefact of shared
training data. Chen et al. chose a ModelNet40 judge for the same reason.
- Test accuracy **0.9214** on ModelNet40, above a **pre-registered usability gate of 0.88**.
- Trained **without rotation augmentation** → orientation sensitive → every cloud is first
  harmonised into the judge's frame (width, up, length), resampled to **1,024 points**, scaled into
  the **unit sphere** [ch4]. **[code]** Training augmentation = point dropout, anisotropic scale,
  shift; no rotation (`scratchpad/plausibility/modelnet40.py`). 1.48 M parameters [findings #53].
- **Not the pipeline's own classifier**: grading our completions with our own car/not-car network
  would be circular [findings #53].

**Two scores per cloud.**
- **Acceptance** = 1 if the judge's top class is "car" (Chen's definition); the rate over cars.
- **Car probability** = the judge's softmax for "car"; graded, doesn't need to win the top spot.
  **[findings #53]** ModelNet40 has no truck / bus / van class, so "car" isn't split with
  near-vehicle classes.

**Pre-registration.** The hypotheses, gate and the reading of every outcome were written **before**
any scoring [ch4; findings #53 dated 2026-09-09, results #54 dated 2026-09-12].
- H1: completed acceptance > raw (McNemar, p < 0.05).
- H2: completed car probability > raw (Wilcoxon, p < 0.05).
- H3: completed vs ceiling = the shape read (A6).
- Pre-committed: if completion failed, report the negative; **no** switching to a friendlier
  judge, no tuning the preprocessing.

### A5. The three arms and the paired tests

[tab:plausibility; N = 2,822 completed car tracks, all 11 sequences, each paired with its own raw
single-frame partial]

| Arm | Acceptance | Mean car prob |
|---|---|---|
| Raw partial | 0.253 | 0.219 |
| **Completed** | **0.785** | **0.745** |
| Static-accumulation ceiling (N = 140) | 0.807 | 0.728 |

**McNemar, worked.** Paired yes/no on the same car. Only the cars that **change** carry
information:
- raw rejected → completed accepted: **1,615**
- raw accepted → completed rejected: **115**
- If completion did nothing, the two would be about equal (each change a coin flip). 1,615 vs 115
  out of 1,730 changes is like 1,730 coin flips giving 1,615 heads → p ≈ 0.
- The cars that don't change (both accepted or both rejected) say nothing about the difference.

**Wilcoxon signed rank.** Paired graded version: per car, completed car probability minus raw;
ranks the differences and checks whether the positive ones dominate. p ≈ 0.

**Which frame is "raw"?** **[code]** The same densest single frame that was completed (S5 A1),
so raw and completed are the same car, same frame, same moment.

### A6. The ceiling: dense, correctly shaped real cars

**What it is** [ch4]: dense **multi-frame accumulations of well-observed static cars**, N = 140.
"Both dense and correctly shaped, the ceiling a completion could aspire to."

**How it's built [code, `scratchpad/plausibility/build_ceiling.py` + `scratchpad/amodal_gt.py`]:**
SemanticKITTI **static-car instances** (labels select and stack them; any instance ever labelled
moving is rejected), accumulated over ≥ 5 frames, ≥ 1,500 points, seen from ≥ 3 of 8 directions
and with support at both ends and both sides. **140 of 1,861** candidates pass, from 8 sequences
[findings #54]. Each is turned to its fitted heading before judging.

**Why it matters: the density confound.** A completed cloud is dense and evenly spread, like the
CAD models the judge trained on; a raw scan is sparse. So raw → completed could just mean "the
judge likes dense clouds". The ceiling is dense **and** correctly shaped. If completion only made
dense blobs, it would land well below the ceiling. It lands at it:
- Static subset (N = 2,220): completed car probability **0.730** vs ceiling **0.728**; completed
  acceptance 0.770 [ch4].
- "Recovers about 96% of the raw → ceiling acceptance gap" [ch4]. **Arithmetic check [measured,
  not in thesis]:** pooled (0.785 − 0.253) / (0.807 − 0.253) = 0.532 / 0.554 = **96%**; on the
  static subset alone (0.770 − 0.248) / (0.807 − 0.248) = **93%**. The thesis sentence pairs 0.770
  with 96%. If asked to compute it, give the pooled version; "93 to 96%, depending on the subset"
  is the safe phrase.
- Mann-Whitney p = 6.8e-7 (completed vs ceiling car probability): at N = 2,220 vs 140 a test
  detects any difference in **distribution shape**; the centre of the completed distribution is at
  or above the ceiling [ch4]. Say "at the ceiling", never "identical".

**Why static only?** Moving cars can't be stacked: the centroid tracker smears them (S5 A1).

### A7. Robustness (slide 38)

- **Every sequence** [tab:plausibility-seq]: completed 0.646 – 0.885 (rounded 0.65 – 0.89), several
  times the raw rate on each. The completion constants were developed on seq 08, which is
  unremarkable (0.311 → 0.799). PCN never saw any real sequence → a generalisation result.
- **Moving cars** (N = 602, net centroid displacement > 2 m **[code]**): acceptance **0.837**, car
  probability **0.801**, slightly above the static subset. Expected: one frame has no motion smear.
- **Heading flip**: the footprint fit can't tell front from back, so every completed cloud was
  rescored turned 180°: acceptance changes by **0.009**.

### A8. Completion ablations (slide 48)

[ch4 §Completion Ablation Studies; one component changed, network unchanged]
1. **Normalisation**: car frame vs a naive 3D PCA frame + the scan's own centre and radius → the
   same checkpoint makes mis-oriented, mis-scaled blobs. Pure normalisation effect (S5 trap #2;
   backup slide 57).
2. **Sim-to-real fine-tuning (negative)**: fine-tuning PCN on LiDAR-like simulated scans (noise
   augmentation, a virtual Velodyne; findings #16, #17, #19) did not recover car geometry; the blobs
   were the input frame's fault. Dropped. **[findings, checkpoints]** Despite ch4's wording ("real data
   fine-tuning", "fine-tuning PCN on mined real pairs"), no real-data fine-tuning run exists: no
   checkpoint, no finding, no script. Never say it was tried; real-data fine-tuning is future work.
3. **L-shape gate**: on seq 08 every implausible completion came from a fragment or merge input; its
   cost is some usable cars rejected.
4. **Length**: per-car estimate vs one fixed 4.14 m prior → the prior stretches compacts and
   shortens long cars. A small far-end shortfall on the longest cars remains (ch5).
5. **PCN vs PoinTr**: held-out synthetic CD 0.161 → 0.153 m, too small for the heavier model; the
   real-data gap is centre estimation, architecture independent.

### A9. The claim, exactly as scoped

[ch4 §Scope] "On real automotive LiDAR, learned completion produces shapes an independent
classifier accepts as cars far more often than the raw partials, and as often as dense correctly
shaped references, while established fidelity metrics, which reward reproducing the input, do not
credit it." Not claimed: exact hidden geometry, or a downstream benefit (that's future work, ch5:
feed completed cars to a box estimator / detector).

---

## Part B — Q&A traps for this session

1. **"The judge just likes dense clouds. Completion makes clouds dense. So what?"**
   → "That's exactly the confound we pre-registered. The ceiling is dense real cars with the
   correct shape. If completion only added density, it would sit well below them; it reaches them:
   car probability 0.730 vs 0.728 on parked cars." [ch4]

2. **"ModelNet40 and ShapeNet are both CAD. Isn't the judge biased toward PCN's outputs?"**
   → "Different repositories, and the judge never saw a ShapeNet model; that's why Chen et al. use
   ModelNet40 too. And the judge is not easy: it rejects three in four raw real scans, and it
   accepts the dense real reference cars at 0.807, so it works on real LiDAR once the shape is
   complete." [ch4]

3. **"Raw scans only 0.25? Then your judge can't read real LiDAR."**
   → "A single frame shows one side of the car; most objects look ambiguous from one side. The
   dense real references pass at 0.807, so real LiDAR itself is not the problem; missing shape is.
   The test is paired on the same cars, so a constant domain effect cancels out."

4. **"What about the 21% of completions the judge rejects?"**
   → Not broken down in the thesis; don't invent a split. Say: "They're not analysed one by one in
   the thesis. Chapter 5 names the sparsest, most one-sided inputs as the weakest: those completions
   stay flatter, which traces to synthetic-only training."

5. **"Why top-1 acceptance? A car could lose narrowly to another class."**
   → "That's why we also report the mean car probability, which needs no top spot: 0.219 → 0.745.
   Both were fixed in advance, so the claim doesn't hang on one cut-off." **[findings #53]**
   ModelNet40 has no truck / bus / van class.

6. **"Synthetic 0.16 m: how did you place the input? With the true centre?"**
   → Yes **[findings #26, #28; not in thesis]**: the synthetic test places each scan as in training
   (true centre, true radius), on 30 held-out cars. Answer: "That test isolates the network: can
   it complete a car when the input is where it expects it? Placing a **real** scan is the job of
   the four geometry steps (slides 19–20), and the real-data judge then tests the whole chain end
   to end." Only if pressed for a number with our own placement: about 0.19 m on synthetic cars
   **[findings #35, before the length step; not a thesis number, say so]**.

7. **"Why is the ceiling only parked cars? And only 140?"**
   → "Moving cars can't be stacked without smearing. 140 is what passes a strict 'seen well from
   enough sides' test across all sequences; it's a reference level, not a test set." **[code]**
   140 / 1,861 candidates.

8. **"The ceiling is built with SemanticKITTI labels. Isn't that ground truth sneaking in?"**
   → **[code]** Yes, instance labels select and stack the reference cars. Answer: "Only to build a
   reference arm, which is judged by the same judge like everything else. Nothing is scored
   **against** it, and the main test (raw vs completed) is paired on identical inputs, with no
   labels." ch5: "no accumulated pseudo-ground truth enters the completion claim."

9. **"Mann-Whitney p = 6.8e-7: completed is significantly different from the ceiling!"**
   → "At 2,220 vs 140 cars, the test detects any difference in the shape of the distributions.
   The centre of the completed distribution is at or above the ceiling: 0.730 vs 0.728. So we say
   'at the ceiling', not 'identical'." [ch4]

10. **"How exactly do you put a cloud into the judge's frame?"**
    → "Turned upright, resampled to 1,024 points, scaled into the unit sphere, the same for every
    arm." **Private, don't volunteer [measured 2026-10-18, not in thesis;
    `qa_backup/judge_aligned_rescore.json`, `scripts/check_judge_aligned.py`]:** in the thesis run
    the raw and completed clouds were turned upright but **kept their scene heading** (only 38% had
    the length within 20° of the judge's length axis), although ch4 says length was harmonised too.
    Re-scored with every pair turned so the car's length lies on the judge's axis (pre-registered:
    ≥ 0.75 completed and ≤ 0.35 raw = headline holds): **raw 0.311 → completed 0.852**, McNemar
    1,644 vs 117, every sequence 0.79–0.98. The unaligned rerun reproduced 0.253 / 0.785 exactly.
    So if pressed on heading: "We also checked with every car turned so its length lies along the
    judge's length axis: completed acceptance goes up, to 0.85, and raw to 0.31. The reported 0.785
    is the conservative number." Don't say the reported run was length-aligned.

11. **"96% of the gap: show me."** → pooled (0.785 − 0.253) / (0.807 − 0.253) ≈ 96%. On the parked
    subset alone it's about 93% **[measured, not in thesis]**. Say "93 to 96%".

12. **"Pre-registered where? Any deviations?"** → findings #53, written before scoring (2026-09-09;
    results 2026-09-12). **[findings #54]** Deviations, all immaterial: the min-point gate was coded
    at 30 points instead of 10 (no partial had fewer than 30, so the same set was scored); an
    up-axis flip was needed because our clouds are Y-down and the judge is Y-up.

13. **"UCD and UHD are what the field uses. Why not just report them?"**
    → "We do report them (slide 39). The raw scan scores a perfect zero, because those metrics
    compare the output with its own input. They can't credit the hidden side, which is the whole
    point of completion. Chen et al. pair them with plausibility for the same reason." [ch2, ch4]

14. **"Fidelity on only 39 cars?"** → **[findings #52]** a seq-08 subset reused from an earlier
    analysis. Answer: "The point is structural, not statistical: any cloud that contains the scan
    scores zero, for every car, by definition."

15. **"How many cars get completed at all?"** → Not stated in the thesis; don't volunteer the
    51% **[measured, project_state]**. Answer: "Every tracked car whose densest frame passes the
    L-shape gate with enough points. Fragments and merged pairs are deliberately skipped, because
    they never complete into a plausible car."

16. **"Does completion help anything downstream?"** → "Not measured yet. The plausibility result
    says completed clouds read as cars; feeding them to a box estimator or detector, as Ren et al.
    do, is the next step (ch5 future work)."

17. **"PoinTr is better on synthetic. On real data?"** → "No decisive difference; the real-data gap
    is estimating where the centre of a half-seen car is, and that doesn't depend on the network."
    [ch2, ch4] **[findings #28]**: real seq-08 plausible rate 18/26 PCN vs 16/26 PoinTr, threshold
    noise.

18. **"Your UCD formula squares the distance, but the table is in metres."** **[code]** The reported
    0.090 m is the plain mean distance (`ucd_mean`); the ch2 equation is written with squares. Answer:
    "The table reports the plain average distance, in metres, as Chen et al. and P2C read it; the
    squared form ranks the clouds the same way, and the raw scan is zero either way." Don't volunteer.

---

## Part C — Diagrams and slides

### D12a — The plausibility judge (user, draw.io) — slide 36 (3.4.2, ★ talk), left half
Scaffold: `assets/d12a_judge.drawio` (from `scripts/make_d12a_scaffold.py`).
**Story:** three kinds of cloud on top (raw scan blue, completed orange, dense reference grey) →
the **same preparation** → one **independent judge** (thick dark border: not part of the
pipeline) → "car?" → acceptance = share judged car. Bottom note: the judge's own accuracy.
**Fill the "?":** raw scan N = **2,822**; dense reference N = **140**; preparation **1,024**
points; judge trained on **ModelNet40** (**40** object classes); accuracy **0.9214** (pass mark
**0.88**).
**Optional:** paste small thumbnails into the three ellipses (car C from `d13_completion_example.png`
for raw / completed); not needed.
Export: `assets/d12a_judge.drawio.png` (exported 2026-10-18, on slide 36; Zoom 200%, Transparent off, Appearance Light, diagram
embedded); slide 36 picks it up automatically.

### D12b — Acceptance bars (Claude, native PPTX chart) — slide 36, right half
`build_deck.py lay_judge`: tab:plausibility acceptance, verbatim (0.253 / 0.785 / 0.807), deck
colours (blue scan, orange completed, grey reference), plus the 1,615 vs 115 line.

**How to say slide 36 (1:00)** is in its speaker notes. Hook: *no answer key → ask a stranger*.

### Support slides
- **25 (3.2.1 Baselines):** one table: what each part is compared against (stages by ablation; no
  head-to-head with supervised networks; completion vs raw scan, dense reference, PoinTr).
- **37 (3.4.3 Synthetic Fidelity):** D16 = `assets/d16_synthetic_example.png` from
  `scripts/make_d16_synthetic_example.py` [illustration, fixed pick rule] + CD 0.16 m / F 0.76.
- **38 (3.4.4 Plausibility Robustness):** tab:plausibility-seq + four cards (paired test, movers,
  heading flip, parked vs reference).
- **39 (3.4.5 Completion Result Analysis):** tab:fidelity (UHD p95 column dropped) + three cards
  (plausible, at the ceiling, scoped claim).
- **48 (3.6.5 Completion Ablations):** five rows, one per ablation (A8).

---

## Part E — Spoken answers (2026-10-18)

### "How do you know the completion is right, without ground truth?" (★ slide 36 and Q&A)

- EN (≈ 35 s): "On a real car nobody has ever seen the hidden side, so there is no true shape to
  measure against. And the usual trick, measuring how close the output stays to the scan, rewards
  doing nothing: the raw scan scores a perfect zero. So we follow Chen et al.: we ask an
  independent judge whether each output looks like a car. The judge is a PointNet++ trained on
  ModelNet40, a different collection from the shapes our network learned from, so it has no reason
  to favour us. A raw single-frame scan is judged a car 25% of the time; after completion, 79%;
  dense multi-frame scans of parked cars, the best real shapes we have, 81%. So completion brings
  a single frame almost to the level of the best real shapes."
- VI: "Với một chiếc xe thật, chưa ai từng nhìn thấy phía bị che khuất, nên không có hình dạng
  chuẩn nào để so sánh. Cách đo thông thường, tức là đo đầu ra còn sát với bản quét đến đâu, lại
  thưởng cho việc không làm gì: chính bản quét gốc đạt điểm hoàn hảo bằng không. Vì vậy chúng tôi
  làm theo Chen và cộng sự: dùng một bộ phân loại độc lập để hỏi xem mỗi đầu ra có giống một chiếc
  xe hay không. Bộ phân loại này là PointNet++ huấn luyện trên ModelNet40, một bộ dữ liệu khác hẳn
  với các mô hình mà mạng của chúng tôi đã học, nên nó không có lý do gì để thiên vị chúng tôi. Bản
  quét một khung hình gốc chỉ được nhận là xe 25% số lần; sau khi hoàn thiện là 79%; còn các bản
  quét dày đặc nhiều khung hình của xe đỗ, hình dạng thật tốt nhất mà chúng tôi có, là 81%. Tức là
  bước hoàn thiện đưa một khung hình đơn lẻ lên gần bằng mức của những hình dạng thật tốt nhất."

### "Couldn't the judge just be rewarding density?" (trap #1, 15 s)

- EN: "We expected that question before we ran it. The dense reference is dense too, and correctly
  shaped. If completion only made clouds denser, it would fall well short of it. It doesn't: on
  parked cars the car probability is 0.730 against the reference's 0.728."
- VI: "Chúng tôi đã lường trước câu hỏi này trước khi chạy thí nghiệm. Tập tham chiếu cũng dày đặc,
  và có hình dạng đúng. Nếu bước hoàn thiện chỉ làm đám mây điểm dày hơn, kết quả sẽ thấp hơn hẳn
  tập tham chiếu. Nhưng không phải vậy: với xe đỗ, xác suất là xe đạt 0,730, so với 0,728 của tập
  tham chiếu."

---

## Part D — Glossary additions (EN → VI; correct to your department's usage)

| English | Vietnamese |
|---|---|
| ground truth (complete shape) | dữ liệu chuẩn (hình dạng đầy đủ) |
| plausibility | tính hợp lý (mức độ giống thật) |
| independent classifier / judge | bộ phân loại độc lập / bộ đánh giá độc lập |
| acceptance rate | tỉ lệ chấp nhận |
| car probability (softmax) | xác suất là xe |
| reference-free metric | độ đo không cần tham chiếu |
| unidirectional Chamfer / Hausdorff distance | khoảng cách Chamfer / Hausdorff một chiều |
| F-score at a threshold | điểm F tại ngưỡng |
| held-out set | tập giữ lại (không dùng để huấn luyện) |
| static-accumulation ceiling | mức trần tích luỹ xe đứng yên |
| density confound | yếu tố gây nhiễu do mật độ |
| paired test (McNemar, Wilcoxon) | kiểm định ghép cặp (McNemar, Wilcoxon) |
| pre-registration | đăng ký trước (giả thuyết) |
| ablation study | nghiên cứu loại bỏ thành phần |
| negative result | kết quả âm tính |
