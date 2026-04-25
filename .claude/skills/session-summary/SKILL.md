---
name: session-summary
description: >
  Summarize the current session and append it to docs/session_summary.md so the next session
  has full context. Use this skill whenever the user says "update session summary",
  "summarize session", "save session", "wrap up", "end of session", "what did we do",
  or any variation of wanting to record what happened in this session before ending.
  Also use when the user references docs/session_summary.md and wants it updated.
---

# Session Summary Skill

You are writing a concise, structured summary of the current Claude Code session and appending it to `docs/session_summary.md`. This file is the handoff document between sessions — the next session reads it on startup (per CLAUDE.md) to understand where things left off.

## Steps

1. **Read** `docs/session_summary.md` to see existing entries and follow the established format.
2. **Review** the full conversation to identify:
   - What was done (code changes, new files, experiments, investigations, installs)
   - Environment notes (gotchas, workarounds, auth setup, dependency quirks)
   - What's next (immediate follow-ups, medium-term plans, open questions)
   - Files changed (new, modified, deleted — check with `git status` and `git diff --stat`)
3. **Append** a new dated section (`# Session Summary — YYYY-MM-DD`) separated by `---` from the previous entry.
4. **Show the user** what you're about to append and get confirmation before writing.

## Format

Follow the structure of existing entries. Each session entry has:

```markdown
---

# Session Summary — YYYY-MM-DD

## What was done

### 1. [Topic]
Brief description. Include specifics: file paths, parameter values, metric results, commands.

### 2. [Topic]
...

## Environment notes
- Bullet points for gotchas, workarounds, config changes worth remembering.
- Only include if there's something non-obvious a future session should know.

## What's next

### Immediate
1. ...

### Medium-term
2. ...

## Files changed
\```
Modified:  file1.py, file2.py
New:       file3.py
Deleted:   file4.py
\```
```

## Writing guidelines

- Be specific: include file paths, function names, metric values, exact commands.
- Be concise: one line per bullet, no fluff. This goes into prompt context.
- Prioritize what a fresh session needs to know to continue the work effectively.
- The "What's next" section is the most important part — it tells the next session where to pick up.
- Don't repeat information already in CLAUDE.md; focus on what changed this session.
- Skip the "Environment notes" section if nothing noteworthy happened.
- For "Files changed", use `git status` and `git diff --stat` to get the actual list.
