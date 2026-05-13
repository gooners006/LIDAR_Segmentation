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

1. **Review** the full conversation to identify:
   - What was done (code changes, experiments, investigations)
   - What's next (blockers, immediate follow-ups, backlog changes)
   - Files changed (check with `git status` and `git diff --stat`)

2. **Read** `docs/project_state.md` to see current state.

3. **Draft both updates:**

   **a. `docs/session_history.md`** — append a new dated section:
   ```markdown
   ---

   # Session — YYYY-MM-DD

   ## What was done
   ### 1. [Topic]
   Brief description with file paths, parameters, metrics.

   ## Files changed
   ```
   Modified: ...
   New: ...
   ```
   ```

   **b. `docs/project_state.md`** — overwrite with updated state:
   - Remove completed tasks from next steps
   - Update architecture notes if components changed
   - Update blockers and immediate next steps
   - Keep it dense — this goes into prompt context every session

4. **Show the user** drafts of both updates and get confirmation before writing.

## Writing guidelines

- `session_history.md`: Be specific (file paths, commands, metrics). One line per bullet.
- `project_state.md`: Be dense. Remove stale info. This is the only file read at session start.
- Don't duplicate content between the two files — history is for the record, state is for context.
- Skip "Environment notes" in history unless something non-obvious happened.
