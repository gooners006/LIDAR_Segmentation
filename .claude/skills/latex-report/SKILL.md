---
name: latex-report
description: Write LaTeX papers or reports following ICTA / Springer LNCS proceedings formatting. Use this skill when the user asks to write a report, draft a paper section, create a LaTeX document, write up results, or prepare a document for their advisor. Also trigger when the user mentions "report", "thesis", "paper", "writeup", "latex", "write up my progress", "draft a section", or wants to document their research formally.
---

# LaTeX Report Writer

Write LaTeX documents (conference papers or progress reports) using the Springer LNCS proceedings format for ICTA submission.

## Step 1: Determine document type

Ask the user if unclear. The three types are:

- **Conference paper** — formal paper for ICTA submission. Follows standard academic structure: abstract, introduction, related work, methodology, experiments, conclusion.
- **Progress report** — periodic update to advisor. Covers a date range, summarizes what was done, shows results, outlines next steps. Uses the same LNCS format but with a report-oriented section structure.
- **Master's thesis** — the full thesis document. A long-form, chaptered document that uses its **own** format (article class, IEEEtran bib, custom FPT titlepage), **not** LNCS. Start from `references/thesis-skeleton.tex` and follow `references/thesis-structure.md` (both self-contained in this skill; distilled from an advisor-approved, redacted thesis).

## Step 2: Gather context automatically

Read these sources in parallel to build a picture of the current research state:

| Source | What to extract |
|---|---|
| `docs/session_summary.md` | Work done, results, decisions, next steps |
| `docs/findings.md` | Technical findings, benchmarks, decisions |
| `CLAUDE.md` | Project overview, pipeline stages, parameters, research status |
| `git log --since=<date>` | Recent commits (scope to relevant date range for progress reports) |
| `checkpoints/training_log.csv` | Training metrics if it exists |
| `src/evaluate.py` output or saved results | Evaluation metrics if available |
| `docs/references.bib` | Available BibTeX references |
| `docs/report/*.tex` | Previous reports for style reference, content continuity, and avoiding duplication |
| `references/thesis-structure.md` + `references/thesis-skeleton.tex` (thesis only) | The skill's own self-contained thesis template and annotated structure. Authoritative for chapter structure, section conventions, format, and register — never for content. Fill results from `docs/findings.md` and evaluation outputs. |

For progress reports, scope the git log and session summaries to the relevant date range. For conference papers, gather everything relevant to the paper topic.

## Step 3: Present summary and ask for input

Before writing anything, present a concise summary to the user:

- What date range / topic you'll cover
- Key points you plan to include (bulleted list)
- Any data or metrics you found
- Any gaps where you need the user's input

Ask: "Anything to add, remove, or change before I write?"

Wait for the user's response before proceeding.

## Step 4: Write the LaTeX document

### Template format

The **conference paper** and **progress report** use the Springer LNCS class
(`llncs.cls`); template files are in `docs/writing/lncs-template/`. The **master's
thesis** does not — it uses its own article-class format; start from
`references/thesis-skeleton.tex` (self-contained in this skill) and see the
"Master's thesis structure" section below.

```latex
\documentclass[runningheads]{llncs}

\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}

\begin{document}

\title{<TITLE>}

\author{Ngo Vi Viet Anh\inst{1} \and
Doan Nhat Quang\inst{1}}

\authorrunning{N. V. V. Anh et al.}

\institute{FPT School of Business \& Technology, FPT University, Vietnam\\
\email{ngovivietanh@gmail.com}}

\maketitle

\begin{abstract}
<ABSTRACT TEXT — 150 to 250 words>

\keywords{<keyword1> \and <keyword2> \and <keyword3>}
\end{abstract}

% --- Body sections here ---

\begin{credits}
\subsubsection{\ackname}
<Acknowledgments if any>

\subsubsection{\discintname}
The authors have no competing interests to declare that are relevant to the content of this article.
\end{credits}

\bibliographystyle{splncs04}
\bibliography{../references}

\end{document}
```

### Bibliography

Use `splncs04` style (the LNCS BibTeX style with alphabetic sorting). The bib file is at `docs/references.bib`:

```latex
\bibliographystyle{splncs04}
\bibliography{../references}
```

Add any new references the document needs to `docs/references.bib` as well.

### Writing conventions

- Default language is English.
- Use `booktabs` for tables (`\toprule`, `\midrule`, `\bottomrule`) — not `\hline`.
- Table captions go **above** the table. Figure captions go **below** the figure.
- Use `\cite{}` for all referenced papers with square brackets (LNCS default). Check `docs/references.bib` for available keys.
- Prefer vector graphics (EPS/PDF) over rasterized images for diagrams.
- Math: use `equation` environment for numbered equations, inline `$...$` for simple expressions.
- Only two levels of headings should be numbered (`\section`, `\subsection`). Use `\subsubsection` for unnumbered run-in headings and `\paragraph` for fourth-level.
- No more than four heading levels total.
- First paragraph after a heading is not indented; subsequent paragraphs are.

### Conference paper structure

1. **Abstract** — 150-250 words summarizing the contribution, with keywords
2. **Introduction** — problem statement, motivation, contribution summary
3. **Related Work** — prior work with citations
4. **Methodology** — technical approach with equations and diagrams
5. **Experiments** — setup, dataset, metrics, results tables/figures
6. **Conclusion** — summary of findings, future work

### Progress report structure

Use the same LNCS format but with these sections:

1. **Overview** — one paragraph situating the report in the thesis timeline
2. **One section per major work item** — with subsections for method, results, analysis as needed
3. **Results** — tables and figures with quantitative metrics. Always include the numbers, not just descriptions.
4. **Next Steps** — numbered list of planned work, ordered by priority

### Master's thesis structure

Uses its own format (not LNCS — see Template format above). Start from
`references/thesis-skeleton.tex`; the full annotated skeleton, format deltas, and
rigor conventions are in **`references/thesis-structure.md`** (read it before
drafting). Five chapters:

1. **Introduction** — Background & Rationale, Challenges, *numbered* Research
   Objectives, *numbered and named* Research Contributions, Thesis Organization.
2. **Literature Review & Theoretical Background** — two halves: prior work grouped
   by research direction, then the concepts needed to follow the method.
3. **Proposed Methodology** — system-architecture figure + phase decomposition,
   then one subsection per phase; formal `algpseudocode` algorithms, ontology
   tables, numbered equations with every symbol defined.
4. **Experimental Evaluation** — hardware/software block, dataset prep (properties
   + statistics tables), experimental setup (baselines with an "Evaluation Purpose"
   column, metrics tables, per-capability protocols each with a query/knowledge-base
   split table), results grouped by task with losses reported honestly, ablation as
   a progression, runtime as a first-class subsection.
5. **Conclusion & Future Work** — Conclusion, Limitations & Discussion (the honest
   home for negative results and proven ceilings), Future Work.

Objectives → contributions → methodology subsections → experiments should align
1:1. Fill results from this project's own `docs/findings.md` and evaluation outputs;
never copy content or numbers from the redacted reference.

## Step 5: Output

Save the LaTeX file to `docs/report/` directory. Use descriptive filenames:
- Progress reports: `progress_report_YYYY_MM_DD.tex`
- Conference papers: `paper_<short_name>.tex`
- Master's thesis: `thesis.tex` (a single living document, not dated per revision)

If the `docs/report/` directory doesn't exist, create it.

For the paper/report types, copy `llncs.cls` and `splncs04.bst` from
`docs/writing/lncs-template/` into `docs/report/` if not already present, so the
document compiles. The **thesis** does not use LNCS: copy
`references/thesis-skeleton.tex` (article class, `IEEEtran` bib) to
`docs/report/thesis.tex` as the starting point, and place figures in
`docs/report/figures/` as usual (including the FPT logo the titlepage expects).

Place any figures into `docs/report/figures/` and reference them with `\includegraphics{figures/<name>}`. Reuse existing figures in that folder when applicable.

After writing, tell the user the file path and remind them to compile with:
```
cd docs/report && pdflatex <filename>.tex && bibtex <filename> && pdflatex <filename>.tex && pdflatex <filename>.tex
```
