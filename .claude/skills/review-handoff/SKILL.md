---
name: review-handoff
description: >
  Prepare work for external LLM review or process incoming review feedback. Use this skill when the
  user asks for a review handoff, second opinion, external LLM review prompt, or to process returned
  review feedback. Trigger phrases include "prepare for review", "send for review", "get second opinion",
  "prepare handoff", "process feedback", "here's the review". Also trigger when the user pastes a block
  of review comments and wants to go through them point by point.
---

# Review Handoff

This skill supports a multi-LLM review workflow where Claude Code does implementation, another LLM provides a second opinion, and the user relays between them.

## Detecting the Mode

- If the user provides review comments or pastes external feedback, use **Process Mode**.
- If the user asks for review/handoff without pasted feedback, use **Prepare Mode**.
- Only ask for clarification if both modes appear equally likely.

## Prepare Mode

Package the session's recent work into a structured, copy-paste-ready review prompt. It must be self-contained — no references to "this conversation" or "as discussed."

### Steps

1. **Identify what was done.** If the user points to specific files or changes, use those directly. Only fall back to repo inspection (`git status --short`, `git diff`, `git diff --staged`, `git log --oneline -5`) when the scope is unclear or the user asks for a broad review.

   Do not include generated files, lockfile churn, build artifacts, or vendored code unless directly relevant. If there is nothing to review and no user-provided summary, ask what should be packaged rather than inventing work.

2. **Self-critique** before generating the handoff. Include at least one of:
   - A real concern about correctness, edge cases, or tradeoffs
   - A known limitation or untested area
   - An assumption that might be wrong
   - Or: "No specific concern found after checking X, Y, Z."

   Do not write generic concerns like "could be improved" unless tied to code, tests, or design evidence.

3. **Generate the handoff block.** Use the smallest format that remains self-contained.

   **Compact handoff** — for small, localized changes (one-file edits, config tweaks, small fixes, doc updates):

```
Review this [code change / design decision / approach].
Specifically flag: correctness issues, unnecessary complexity, and anything I might be missing.

## Context
[One or two sentences]

## What changed
[Summary of the change]

## Code / details
[Relevant snippet or config excerpt]

## Verification performed
[Commands run, results, untested areas]

## Review focus
[1-3 specific questions]
```

**Full handoff** — for multi-file changes, architectural decisions, risky logic, security-sensitive work, or large diffs:

```
Review this [code change / design decision / approach / architecture].
Specifically flag: correctness issues, unnecessary complexity, things that won't scale, and anything I might be missing.

## Context
[One or two sentences: what the project is, what problem is being solved]

## Repo state
- Branch:
- Changed files:
- Untracked files (if relevant):
- Recent commits (if relevant):

## What was done
[Concise summary of the change or decision — what and why]

## Code / details
[Relevant code snippets, config changes, or decision details. For large changes: identify changed subsystems, summarize each, include only the most review-critical snippets, list omitted files with one-line reasons.]

## Verification performed
- Commands/tests run:
- Results:
- Known failures or untested areas:

## Author's concerns
[Self-critique points as a bulleted list]

## What I'd like you to focus on
[1-3 specific questions]
```

4. **Keep it brief** but do not omit context required for correctness evaluation. Prefer focused excerpts over full files. If tests, typechecks, or builds failed, include the failure summary and do not present the work as complete.

## Process Mode

The user has returned with feedback from an external reviewer. Go through it systematically.

### Steps

1. **Extract actionable points.** Parse the feedback into discrete claims or suggestions. Ignore filler — focus on specific critique.

2. **Verify against code.** Before accepting or rejecting any point, inspect the relevant code, config, or docs. Do not push back based only on conversation memory. If a point cannot be verified from the repo, mark it as **Investigate** and state what evidence is missing.

3. **Evaluate each point.** For each feedback point, include:
   - The feedback claim
   - What was checked (file, line, config, test output)
   - Verdict: **Concede** / **Push back** / **Partially agree** / **Investigate**
   - Reason
   - Action (if any)

4. **Summarize the verdict.** End with a clear action list:
   - **Change:** [things to actually change based on the feedback]
   - **Keep as-is:** [things that were challenged but should stay, with brief reasoning]
   - **Investigate:** [things worth looking into but not clear-cut]

5. When rejecting a point, point to the specific file, line, snippet, config, or observed behavior that contradicts it. When conceding, state what to change and why they're right.

## Guardrails

**Do not include in handoff prompts:**

- Secrets, API keys, credentials, tokens, or private keys
- Personal data or private URLs
- Unrelated full-file dumps or excessive logs
- Generated files, lockfile churn, or build artifacts (unless directly relevant)
- Speculative claims not backed by code or observed behavior

When redacting sensitive values, replace them with descriptive placeholders (e.g., `<REDACTED_API_KEY>`, `<PRIVATE_URL>`) and preserve the surrounding config shape so the reviewer can still assess the structure. If a file is mostly sensitive, summarize its role instead of excerpting it. If unsure whether a value is sensitive, redact it.
