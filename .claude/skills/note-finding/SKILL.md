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

1. **Read** `docs/findings.md` to get the current entry count and follow the established format.
2. **Draft** a new numbered section based on what the user wants to record. Pull details from the conversation — don't ask the user to re-state things already discussed.
3. **Show** the draft to the user for confirmation before writing.
4. **Append** the finding to the end of `docs/findings.md`.

## Format

Follow the existing structure. Each finding has:

```markdown

## N. Title (YYYY-MM-DD)

**Context:** Why this came up — one or two sentences.

**Finding:** The core result, benchmark, observation, or decision. Include numbers, code snippets, commands, or tables as appropriate.

**Decision:** What was decided and why (if applicable). Omit if the finding is purely observational.
```

## Guidelines

- Number sequentially from the last entry in the file.
- Use today's date.
- Be specific: include exact numbers, file paths, parameter values, commands.
- Keep it concise — this is a reference log, not a report.
- Include code snippets or tables only when they add clarity.
- If the user gives a vague "note this down", review the recent conversation to extract the relevant details yourself.
