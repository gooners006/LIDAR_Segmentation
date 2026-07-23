# Master's Thesis Structure

The starting point is **`thesis-skeleton.tex`** in this same `references/`
directory: a compilable, content-free template (preamble, titlepage, front
matter, empty five-chapter body). It is self-contained — the skill does not
depend on any file outside itself.

The structure, format, and conventions below were distilled from a completed
master's thesis by another of the same advisor's students (Dr. Doan Nhat Quang,
FPT School of Business & Technology), which was approved and then **redacted
before being shared** — results prose, numbers, and several analysis subsections
were removed on purpose. So this skeleton is authoritative for **chapter
structure, section conventions, LaTeX format, and writing register**, never for
content. Fill every result from this project's own findings and evaluation
outputs.

That reference thesis was about VLM/graph traffic-scene retrieval — nothing to do
with the LiDAR work. Borrow the skeleton, not the subject matter.

## Format (this differs from the paper/report LNCS format)

The thesis is a long-form, chaptered document, **not** an LNCS proceedings paper.
It uses its own preamble (already in `thesis-skeleton.tex`):

```latex
\documentclass[a4paper, 11pt]{article}
\usepackage[english]{babel}
\usepackage[utf8]{inputenc}
\usepackage{graphicx, float, caption, booktabs, multirow, tabularx, array, makecell}
\usepackage{amsmath}
\usepackage{algorithm, algorithmicx, algpseudocode}   % formal pseudocode
\usepackage{listings}                                  % JSON / code listings
\usepackage{tikz}                                      % pipeline diagrams, title border
\usepackage{fancyhdr}                                  % running header/footer
\usepackage[top=2.5cm, bottom=2.5cm, left=3.0cm, right=2.5cm]{geometry}
\usepackage{hyperref}
\bibliographystyle{IEEEtran}                           % NOT splncs04
```

Format points that differ from the LNCS paper/report skill:

- **Class:** `article` with a custom `titlepage` (tikz border, FPT logo, "MASTER
  THESIS ON SOFTWARE ENGINEERING", title, Student/Supervisor block). Chapters are
  numbered `\section`s separated by `\newpage`; `\subsection` / `\subsubsection`
  below.
- **Bibliography:** `IEEEtran`, not `splncs04`.
- **Captions:** both figure *and* table captions sit at the **bottom**
  (`\captionsetup[table]{position=bottom}`). The LNCS "table caption above" rule
  applies only to the paper/report types, not the thesis.
- **Running header:** `\lhead{FPT School of Business \& Technology}`, page number
  in the center footer.

When starting a thesis, copy `thesis-skeleton.tex` (in this directory) into
`docs/report/thesis.tex` rather than reusing the LNCS template — it already
contains this preamble and the titlepage.

## Chapter skeleton

Five chapters. Each maps to a phase of the argument, and the pieces line up: the
numbered Research Objectives in Ch1 become the numbered Contributions, which
become the methodology subsections, which become the experiments.

**Ch1 — Introduction**
- Background & Rationale
- Challenges (the specific difficulties this work addresses)
- Research Objectives — a short *numbered* list
- Research Contributions — a *numbered* list, each contribution given a concrete
  name (e.g. the approved thesis names its four: a query framework, a graph
  representation, an alignment algorithm, a reasoning method). The rest of the
  thesis is organized to deliver and validate each one.
- Thesis Organization (one sentence per following chapter)

**Ch2 — Literature Review & Theoretical Background** (deliberately two halves)
- *Literature Review*: prior work grouped by **research direction / competing
  paradigm** (the approved thesis uses four), not one flat list of papers.
- *Theoretical Background*: the concepts a reader needs to follow the method,
  defined before they are used.

**Ch3 — Proposed Methodology**
- Open with a **system-architecture figure** and an N-phase decomposition of the
  approach.
- Then **one subsection per phase**, each with its own figure.
- Formal algorithms in `algpseudocode` (`\begin{algorithm}` + `\Require`/`\Ensure`).
- Definition/ontology tables for any fixed vocabulary the method relies on.
- Numbered `equation`s, with **every symbol defined immediately after** the
  equation ("where $x$ denotes ...").

**Ch4 — Experimental Evaluation**
- **Hardware/software environment** stated up front as an itemized block, with a
  sentence that all methods ran under the same environment.
- Dataset preparation: a *dataset-properties* table and a *statistics* table
  (counts per split / per class).
- Experimental setup, three parts:
  - *Baselines*: a table with a **"Evaluation Purpose" column** — each baseline
    is included to probe one specific claim, not to pad the comparison.
  - *Metrics*: grouped by task, each in its own table with one-line descriptions.
  - *Implementation details / evaluation protocol*: **separate protocols for
    separate capabilities**, each with an explicit **query/test vs. knowledge-base
    (train/test) split table** and stated isolation guarantees ("no contextual
    overlap between test and database").
- Results, grouped by task/capability. **Report losses honestly** and explain the
  tradeoff (e.g. the approved thesis loses one single-frame setting but has
  precision 1.000, and says so, then explains the recall cost).
- Ablation framed as a **progression** (strategy A → B → C) that shows why each
  design decision earns its place, not a grid of disconnected variants.
- **Runtime Evaluation is its own first-class subsection**, not a footnote.

**Ch5 — Conclusion & Future Work**
- Conclusion (what was built and shown)
- Limitations & Discussion — the honest place for negative results and proven
  ceilings (e.g. a target that was demonstrated to be unreachable)
- Future Work

## Rigor conventions worth carrying over

- Objectives → contributions → methodology subsections → experiments align 1:1.
  If a contribution has no matching experiment, the thesis has a hole.
- Every figure and table is referenced by `\ref` from the prose and has a caption
  that stands on its own.
- Every equation symbol is defined right where the equation appears.
- Baselines and metrics justify their own inclusion.
- Claims are hedged or definitive to match the evidence; a lost row is reported
  and explained rather than hidden.

## Do not copy

- **Content and numbers.** The reference is redacted precisely so its results are
  not reused. Fill Ch4 from this project's own `docs/findings.md` and evaluation
  outputs.
- **Blemishes.** The approved thesis expands its own acronym two different ways
  (Interaction-*Centric* vs. Interaction-*Aware* Hungarian Graph Alignment). Keep
  acronym expansions and named terms consistent throughout.
