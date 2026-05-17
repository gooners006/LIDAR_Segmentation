---
name: session-summary
description: >
  Summarize the current session and update project state. Use this skill whenever the user says
  "update session summary", "summarize session", "save session", "wrap up", "end of session",
  "what did we do", or any variation of wanting to record what happened before ending.
---

# Session Summary Skill

You are recording the current session and updating project state across two files:

- `docs/session_history.md` — append-only chronological diary (for the user's records)
- `docs/project_state.md` — living document of current project state (read at session start)

## Steps

1. Review available conversation context to identify:
   - What was done (code changes, experiments, investigations)
   - Results, metrics, decisions, or unresolved issues
   - Immediate next steps and blockers

   If context is incomplete (e.g., after compaction in a long session), summarize only what is available and mark uncertain details. Do not invent experiment results or decisions.

2. Check repository changes:
   - `git status --short`
   - `git diff --stat`
   - `git diff --cached --stat` if staged changes exist

3. Read:
   - `docs/project_state.md` for current state
   - `docs/session_history.md` only enough to check whether today's session was already summarized before drafting

4. Draft both updates:

   **a. `docs/session_history.md`** — append a new dated section with this structure:
   - Horizontal rule (`---`)
   - `# Session — YYYY-MM-DD`
   - `## What was done` — topic subsections or bullets with file paths, parameters, metrics
   - `## Files changed` — modified, new, deleted
   - `## Results / findings` — optional; include only if meaningful results or metrics were produced
   - `## Next` — carry-over items and immediate follow-ups

   **b. `docs/project_state.md`** — full replacement draft of current project state:
   - Remove completed tasks from next steps
   - Update architecture notes if components changed
   - Update blockers and immediate next steps
   - Keep enough detail for a session-start recap to be accurate without reading session_history.md
   - Keep it focused on the next session, not the full task universe

5. Show both drafts to the user for confirmation before writing.

6. After confirmation, re-read both files to catch changes since the draft was prepared:
   - Re-read `docs/session_history.md`. If today's session now has a new entry, ask whether to append a second entry, update the existing entry, or skip.
   - Re-read `docs/project_state.md`. If it changed since the draft was prepared, show the conflict and regenerate or ask before overwriting.

7. After any duplicate/conflict handling is resolved, write the confirmed updates: append or update `docs/session_history.md` first, then overwrite `docs/project_state.md`.

## Guidelines

- Use today's date in `YYYY-MM-DD` format.
- Use conversation context for decisions and rationale. Use git output for changed files.
- Do not infer decisions from git output alone, and do not report file changes that are not supported by git output.
- If git commands fail or the project is not a git repository, say so and report only file changes explicitly known from the conversation, marked as unverified by git.
- Do not invent details missing from conversation, git output, or existing docs.
- Prefer accurate partial summaries over guessed complete summaries.
- Preserve exact file paths, commands, parameters, metrics, dataset splits, frame ranges, and checkpoint names.
- Append only to `docs/session_history.md`; do not edit prior entries unless the user explicitly asks.
- If today's session already has an entry, ask whether to append a second entry, update the existing entry, or skip. Only update an existing entry if the user explicitly chooses that option.
- Ask before creating either file if it does not exist.
- If there are no repository changes, write "No file changes detected" under Files changed.
- Exclude irrelevant generated files, caches, temporary outputs, and large artifacts unless intentionally part of the session outcome.
- Avoid unnecessary duplication between the two files — history records what happened; state records what matters for resuming work.
- Keep `project_state.md` current-state focused; move chronological details to `session_history.md`.
- Preserve the existing structure and headings of `project_state.md` unless the structure is clearly stale or the user asks to reorganize it.
- Do not repeat `project_state.md` verbatim — distill and update.
