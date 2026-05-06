---
name: latex-report
description: Write LaTeX progress reports or thesis chapters following FPT university formatting. Use this skill when the user asks to write a report, draft a thesis section, create a LaTeX document, write up results, or prepare a document for their advisor. Also trigger when the user mentions "report", "thesis", "writeup", "latex", "write up my progress", "draft a chapter", or wants to document their research formally.
---

# LaTeX Report Writer

Write LaTeX documents (progress reports or thesis chapters) for an FPT School of Business & Technology master's thesis, using the university's template format.

## Step 1: Determine document type

Ask the user if unclear. The two types are:

- **Progress report** — periodic update to advisor. Covers a date range, summarizes what was done, shows results, outlines next steps.
- **Thesis chapter** — formal chapter or section for the thesis document itself (Introduction, Methodology, Results, etc.).

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

For progress reports, scope the git log and session summaries to the relevant date range. For thesis chapters, gather everything relevant to the chapter topic.

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

All documents use this FPT university template structure:

```latex
\documentclass[a4paper, 12pt]{article}

% --- Packages ---
\usepackage[english]{babel}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{fancyhdr}
\usepackage{array}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{bm}
\usepackage{float}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{enumitem}
\usepackage{caption}

% --- Header & Footer ---
\setlength{\headheight}{15pt}
\addtolength{\topmargin}{-3pt}
\pagestyle{fancy}
\fancyhf{}
\lhead{FPT School of Business \& Technology}
\cfoot{\thepage}
\rfoot{Ngo Vi Viet Anh - MSE13205}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0.5pt}
```

### Title page

```latex
\begin{titlepage}
    \begin{center}
        \includegraphics[scale=0.3]{Images/fpt.png}
    \end{center}
    \center
    \vspace{0.5in}
    \textbf{\large <DOCUMENT TYPE>}
    \vspace{0.5in}
    \noindent\makebox[\linewidth]{\rule{\linewidth}{1.2pt}}
    \textbf{\large <TITLE>}
    \noindent\makebox[\linewidth]{\rule{\linewidth}{1.2pt}}
    \vspace{0.5in}
    \begin{minipage}{0.65\textwidth}
        \begin{flushleft}
            \textit{Student:} \\
            Ngo Vi Viet Anh - MSE13205 \\
        \end{flushleft}
    \end{minipage}
    \begin{minipage}{0.3\textwidth}
        \begin{flushright}
            \textit{Advisor:} \\
            Dr. Doan Nhat Quang \\
        \end{flushright}
    \end{minipage}
    \vspace{2in}
    \textbf{\large GRI501} \\
    \today
\end{titlepage}
```

For progress reports, use `PROGRESS REPORT` as document type and include the date range in the title. For thesis chapters, use `MASTER THESIS` and the thesis title.

### Page numbering and TOC

```latex
\pagenumbering{arabic}
\setcounter{page}{2}
\tableofcontents
\newpage
```

### Bibliography

Use IEEEtran style. The bib file is at `docs/references.bib`:

```latex
\bibliography{../references}
\bibliographystyle{IEEEtran}
```

Add any new references the document needs to `docs/references.bib` as well.

### Writing conventions

- Default language is English.
- Use `booktabs` for tables (`\toprule`, `\midrule`, `\bottomrule`) — not `\hline`.
- Use `\cite{}` for all referenced papers. Check `docs/references.bib` for available keys.
- Place figures and tables with `[H]` (float) for precise positioning.
- Math: use `equation` environment for numbered equations, inline `$...$` for simple expressions.
- Use `\subsection` and `\subsubsection` for hierarchy — avoid going deeper than 3 levels.

### Progress report structure

For progress reports, use this section structure:

1. **Overview** — one paragraph situating the report in the thesis timeline
2. **One section per major work item** — with subsections for method, results, analysis as needed
3. **Results** — tables and figures with quantitative metrics. Always include the numbers, not just descriptions.
4. **Next Steps** — numbered list of planned work, ordered by priority

### Thesis chapter structure

Follow standard academic conventions for the chapter type. Match the style and depth of `docs/report-template/personal.tex` — formal academic prose, proper citations, mathematical notation where appropriate.

## Step 5: Output

Save the LaTeX file to `docs/report/` directory. Use descriptive filenames:
- Progress reports: `progress_report_YYYY_MM_DD.tex`
- Thesis chapters: `chapter_<name>.tex`

If the `docs/report/` directory doesn't exist, create it. Also copy `docs/report-template/Images/` to `docs/report/Images/` if not already present, so the FPT logo is available for compilation.

After writing, tell the user the file path and remind them to compile with:
```
cd docs/report && pdflatex <filename>.tex && bibtex <filename> && pdflatex <filename>.tex && pdflatex <filename>.tex
```
