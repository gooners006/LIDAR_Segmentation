---
name: session-start
description: >
  Read docs/project_state.md and present a brief current-state recap.
  Use this skill when the user explicitly asks for current project state,
  session-start context, where to resume work, or a short recap before starting.
---

# Session Start Skill

You are reading the project state and giving the user a short recap so they can resume work.

## Steps

1. Read `docs/project_state.md`.
2. Summarize only the current project state from that file.
3. Present a short, scannable recap:
   - Current state of the pipeline and components
   - Active blockers or issues
   - Up to 3 immediate next steps listed or clearly implied by `docs/project_state.md`
4. Ask what the user wants to work on, unless they requested only a recap.

## Output format

**Current state:**
- ...

**Blockers:**
- ...

**Up next:**
1. ...
2. ...
3. ...

What would you like to tackle?

## Guidelines

- Use `docs/project_state.md` as the source of truth for the recap.
- Do not read `docs/session_history.md` unless the user asks for historical detail.
- Do not infer missing status from memory or prior conversation.
- If the file appears stale, incomplete, or internally inconsistent based on its contents or metadata, mention that briefly.
- If there are no explicit blockers, say "No explicit blockers listed."
- If fewer than 3 next steps are available, list only those available.
- Do not repeat the file verbatim — distill it.
- If `docs/project_state.md` does not exist, say so and ask whether the user wants to create it.
- Keep the recap brief: this is a session-start summary, not a report.
