---
name: note-finding
description: >
  Append a new finding to docs/findings.md. Use this skill when the user says "note this",
  "record this finding", "log this", "add to findings", "write this down", "save this finding",
  or any variation of wanting to document a technical finding, benchmark result, decision,
  or discovery during the session.
---

# Note Finding Skill

You are appending a structured finding to `docs/findings.md` — the project's running log of technical discoveries, benchmarks, and decisions.

## Steps

1. **Read** `docs/findings.md`. If the file does not exist, confirm the path with the user before creating it with a `# Technical Findings` heading.
2. **Determine the next number** from the highest existing heading matching `## N. ` — do not count entries.
3. **Check for duplicates.** Scan recent entries for the same topic, metric, command, or decision. If a likely duplicate exists, ask whether to append as a follow-up, merge into the existing entry, or skip.
4. **Draft** a new numbered section from conversation context without asking the user to restate already-discussed details. Ask only for confirmation, missing-path approval, duplicate handling, or genuinely unavailable information.
5. **Show** the draft to the user for confirmation before writing.
6. **Re-read** `docs/findings.md` before appending and confirm the next number is still valid.
7. **Append** the finding to the end of `docs/findings.md`.

## What belongs in findings.md

Use `docs/findings.md` for durable technical findings: benchmark results, observed or validated technical behavior (with uncertainty stated when not fully verified), design decisions, failure analyses, dataset discoveries, and reproducible commands tied to a result or decision.

Do not use it for:
- Transient TODOs or next-session plans (→ `project_state.md`)
- Broad project status updates (→ `project_state.md`)
- Implementation history without a durable conclusion (→ `session_history.md`)
- Unresolved brainstorming

If the note doesn't fit findings.md, suggest the right destination.

## Format

Follow the existing structure. Each finding has:

```markdown

## N. Title (YYYY-MM-DD)

**Context:** Why this came up — one or two sentences.

**Finding:** The core result, benchmark, observation, or decision. Include numbers, code snippets, commands, or tables as appropriate.

**Decision:** What was decided and why (if applicable). Omit if the finding is purely observational.
```

## Guidelines

- Use today's date.
- Be specific: include exact numbers, file paths, parameter values, commands.
- Keep it concise — this is a reference log, not a report.
- Include code snippets or tables only when they add clarity.
- If the user gives a vague "note this down", review the recent conversation to extract the relevant details yourself.
- If the user provides multiple unrelated findings, split them into separate numbered entries. If they are parts of one experiment, keep them in one entry.
