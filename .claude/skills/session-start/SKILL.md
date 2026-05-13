---
name: session-start
description: >
  Read docs/project_state.md and present a brief recap at the start of a new session.
  Use this skill when the user says "start session", "session start", "catch me up",
  "what's the context", "where did we leave off", "recap", "what happened last time",
  or any variation of wanting to review previous session context before starting work.
  Also proactively suggested at the beginning of a new conversation.
---

# Session Start Skill

You are reading the project state and giving the user a quick recap so they can jump into work.

## Steps

1. **Read** `docs/project_state.md`.
2. **Summarize** in a few bullet points:
   - Current state of the pipeline and components
   - Active blockers or issues
   - Top 3 immediate next steps
3. **Present** the recap in a short, scannable format.
4. **Ask** what they'd like to work on this session.

## Output format

Keep it brief. Example:

```
**Current state:**
- Pipeline stages 1-6 + tracking working
- Classifier: code done, not trained
- PCN: needs retrain

**Blockers:**
- No classifier checkpoint

**Up next:**
1. Train classifier
2. Retrain PCN
3. Pipeline smoke test

What would you like to tackle?
```

## Guidelines

- Only read `docs/project_state.md`. Do NOT read `docs/session_history.md` unless the user asks for history.
- Don't repeat the file verbatim — distill it.
- If `docs/project_state.md` doesn't exist, say so and offer to create it.
