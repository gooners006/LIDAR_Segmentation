# RANSAC on seq 08: private Q&A backup (NOT thesis results, NOT for slides)

Run 2026-10-16. Scripts: `docs/defense/scripts/ransac_convergence.py` (all 4,071 frames) and
`ransac_300_vs_1000.py` (every 20th frame = 204, plus all rejected frames). Raw data: the two
`.json` files in this folder. Both replay `remove_ground()`'s seeded loop; the replay was asserted
against the pipeline every 200th frame and on all rejected frames.

## Is 300 tries enough?
- Pre-registered criterion 1 ("best plane first found at try ≤ ~100 on the worst frame") came
  out **tight**: best found at try median 159, p90 275, max 300. That metric is crude (it also
  counts near-identical road planes edging up by a few points); this was noted after seeing the
  result, so criterion 2 was added before running it.
- Pre-registered criterion 2 (300 vs 1000 tries; "within ~2% and < 1° in most frames → enough"):
  **mixed**. Median gain +1.4%, median tilt 0.37°, so the typical frame is fine. But 41/204 (20%)
  gain > 5% and 40/204 (20%) tilt > 1° (max +42%, max 89.5°). **For about 1 frame in 5, 300 is
  tight.**
- Road fraction (best inliers / points): min 18%, median 36%; 790 frames (19%) have < 25% road,
  below the "~25% road ⇒ 300 tries is 99% safe" bound in the S2 notes.
- Frame 2500 (the D4 figure) found its best at try #3, which is atypical (median 159).

## Frames with no ground removed
- **204 / 4,071 frames (5.0%)**: the winning plane fails the near-vertical check (median normal
  z = 0.07, i.e. a wall/facade), so `remove_ground()` returns no plane and removes nothing;
  confirmed with the pipeline function itself on all 204.
- 192 / 204 are **still** rejected at 1000 tries, so more tries don't help.
- Cause (design detail in `src/pipeline.py:remove_ground`): the check is applied to the winner
  only. Filtering candidates by the normal *inside* the loop would keep the best horizontal
  plane instead. This is not in the thesis; the thesis calls the fail-safe "rare".
- Effect on detection: **not measured.** Hypothesis: lower recall in those frames (the road
  bridges objects into large clusters). A check would need per-frame recall split by these
  frames.

## How to answer if asked
- "Why 300 iterations?" → "It was cut from 1000 for runtime. On a typical frame the plane at 300
  is within about 1–2% of the plane at 1000, though some frames would still improve with more
  tries. It was a runtime-vs-precision trade, not a free choice."
- "Does the wall check ever fire?" → "Yes, in about 5% of frames a wall wins, and then no ground
  is removed in that frame. The better design applies the check to every candidate, not just the
  winner. That's an easy improvement I'd make next."
