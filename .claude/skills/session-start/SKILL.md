---
name: session-start
description: >
  Read docs/session_summary.md and present a brief recap at the start of a new session.
  Use this skill when the user says "start session", "session start", "catch me up",
  "what's the context", "where did we leave off", "recap", "what happened last time",
  or any variation of wanting to review previous session context before starting work.
  Also proactively suggested at the beginning of a new conversation.
---

# Session Start Skill

You are reading the session summary and giving the user a quick recap so they can jump into work with full context.

## Steps

1. **Read** `docs/session_summary.md`.
2. **Summarize** the most recent session entry in a few bullet points:
   - What was accomplished last time
   - Any open issues or blockers
   - What's next (the most actionable items)
3. **Present** the recap to the user in a short, scannable format — no walls of text.
4. **Ask** what they'd like to work on this session.

## Output format

Keep it brief. Example:

```
**Last session (YYYY-MM-DD):**
- Did X, Y, Z
- Open issue: ...

**Up next:**
1. First priority
2. Second priority

What would you like to tackle?
```

## Guidelines

- Only summarize the most recent session entry unless the user asks for more history.
- Highlight blockers or gotchas from "Environment notes" if they're relevant to the next steps.
- Don't repeat the full session summary verbatim — distill it.
- If `docs/session_summary.md` doesn't exist, say so and offer to create it.
