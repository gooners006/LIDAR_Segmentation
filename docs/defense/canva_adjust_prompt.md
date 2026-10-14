# Canva AI — adjust the existing deck to the friend's layout (2026-10-14)

Alternative to `canva_skeleton_prompt_v2.md` (which generates a new deck). Same target layout;
keeps everything already placed (D1, cropped scan, names, tuned sizes).

## Before prompting (by hand, ~2 min)

1. **File → Make a copy** of the current design. Prompt the copy; the original is the fallback.
2. In the page strip, **drag "Completing the Car" to sit right after "Tracking Removes Flicker"**.
   Order is then: Cover, Problem, Two Ways, Objectives, Framework, Candidates, Keeping Cars,
   Tracking, Completion, Detection Results, Limits Recall, Judging, Conclusions, Thank You.

## Prompt

```
Restyle this existing 14-slide deck. Keep every text, image and placeholder that is already here; do not rephrase, add or remove text except where listed. Do not move, crop or recolour the two images on "The Problem" slide except to reposition them as stated. Canvas is 1920 x 1080 px. Use my Brand Kit fonts.

A. INSERT 5 NEW SLIDES (white background):
- After the cover: a "Table of Contents" slide. Title centred, Bold, UPPERCASE, size 56, y = 120. Two columns, Bold, size 36: left "I. Introduction" and "II. Proposed Methodology"; right "III. Experiment Evaluation" and "IV. Conclusion & Future Work".
- Section divider slides, each with only its title centred both ways (Bold, UPPERCASE, size 56):
  "I. INTRODUCTION" right after the Table of Contents;
  "II. PROPOSED METHODOLOGY" right after "Objectives and Contributions";
  "III. EXPERIMENT EVALUATION" right after "Completing the Car";
  "IV. CONCLUSION & FUTURE WORK" right after "Judging Completion Without Ground Truth".
The deck then has 19 slides.

B. EVERY SLIDE:
- Delete the thin horizontal rule line at the top and the thin footer line.
- Add a footer bar: rectangle x = 0, y = 1000, width 1920, height 80, fill #1F2933. Inside it, left (x = 40): the section name in white, Bold Italic, UPPERCASE, size 24; right (right edge x = 1880): the slide number in white, size 24. Remove the old slide number. Section names: slides 4-6 "I. INTRODUCTION", slides 8-12 "II. PROPOSED METHODOLOGY", slides 14-16 "III. EXPERIMENT EVALUATION", slide 18 "IV. CONCLUSION & FUTURE WORK"; all other slides leave it empty.
- Cover, Table of Contents and the four dividers also get a top-right bar: x = 1240, y = 50, width 600, height 45, fill #1F2933.

C. CONTENT SLIDES (the 12 original slides between the cover and "Thank You"):
- Title: centred, y = 30, Bold, UPPERCASE, size 44, one line, prefixed with its number within its section:
  I: "1. The Problem: A One-Sided View", "2. Two Ways to Solve It", "3. Objectives and Contributions";
  II: "1. Framework Overview", "2. Finding Object Candidates", "3. Keeping Only Cars", "4. Tracking Removes Flicker", "5. Completing the Car";
  III: "1. Detection Results", "2. What Limits Recall", "3. Judging Completion Without Ground Truth";
  IV: "1. Conclusions, Limitations and Future Work".
- Key message: centred, top at y = 105, Medium, size 30, max width 1680, may wrap to 2 lines.
- Move every placeholder, image and text column up so the content area runs from y = 220 to y = 880. Keep horizontal positions, widths and the gaps between side-by-side items.
- "[ pipeline progress strip ]" bars: move to x = 120, y = 905, width 1680, height 50.
- "The Problem" slide: both images from y = 220 to y = 860; the caption "Real car, 6 m away (Thesis Fig. 2.3)" size 24, centred under the right image, top at y = 875.

D. COVER:
- Add [ university logo ] as an empty light-grey rectangle at x = 120, y = 40, width 300, height 110.
- Centre the thesis title (Bold, UPPERCASE, size 48, max 3 lines).
- Move the Supervisor / Presenter / ID lines to the bottom-right, left-aligned at x = 1100, starting y = 700, size 30. Change "Presentor:" to "Presenter:".

E. "THANK YOU" SLIDE: "Thank You" centred, Bold, UPPERCASE, size 56, y = 120; "Questions?" centred below it, size 36; the D3 placeholder from y = 330 to y = 900, x = 120 to 1800.

Nothing may enter the footer bar (y >= 1000). No text may be hidden behind any rectangle.
```

## If Canva AI fails at part A (inserting slides)

Do A by hand: duplicate the cover (it already has the top-right bar after D), delete its
content and type the divider title; repeat. Then re-run the prompt with part A removed.

## After — checklist

- [ ] 19 slides, order as in step 2 plus TOC + 4 dividers.
- [ ] D1 and the scan still on "The Problem", not cropped or recoloured; caption above y = 920.
- [ ] Footer section names correct; numbering restarts per section.
- [ ] No stray rectangles over placeholders; nothing hidden.
- [ ] Re-upload `figures/d1_occlusion.svg` (enlarged labels) over the old D1 if not yet done.
- [ ] Export PNG (all pages) + PPTX into `canva_check/`.
