---
name: quiz-me
description: >
  Generate a comprehension quiz on recently written or changed code so the user can
  defend every design decision at their thesis defense. Use after a substantial
  Claude-written change lands (new module, new metric, new GT builder, algorithm
  change), or when the user says "quiz me", "test my understanding", "can I defend
  this", or before committing thesis-critical code.
---

# Quiz Me Skill

This is a master's thesis project: the user must be able to explain and defend every
design decision to their examiners, including code Claude wrote. After a substantial
change, convert "Claude did it and the metrics improved" into "I can explain why it
works."

## Steps

1. **Scope the material.** Default to the most recent substantial change (current
   diff, last commit, or the module just built). Confirm scope in one line, don't
   interrogate the user about it.
2. **Write a short context section first** (before the questions):
   - The mental model: what the code does and why this design, in a few sentences.
   - The 2–4 **non-obvious decisions** — the things an examiner would probe (why
     this normalization, why this threshold, why median not mean, why this split).
3. **Ask 5–7 scenario-based questions, one at a time.** Wait for the user's answer
   before revealing the next question.
   - Prefer "what happens if / why does X instead of Y" over trivia ("what is the
     variable called").
   - Target the decisions that matter under examination: metric validity, data
     assumptions, failure modes, statistical choices (e.g., why Wilcoxon, why
     per-car medians), coordinate/normalization semantics.
   - At least one question about a known limitation or failure mode of the change.
4. **Grade each answer honestly.** If the user's answer is wrong or incomplete, say
   so plainly, give the correct reasoning, and point to the exact code/finding
   (`file:line`, finding number) they should re-read. Do not soften wrong into
   "partially right."
5. **Close with a gap list:** the specific concepts the user should review before
   the defense, if any. If they aced it, say so and stop.

## Rules

- Questions must be answerable from the code and findings that exist — no gotchas
  about hypotheticals the project never faced.
- Keep the whole quiz grounded in thesis-defense relevance: "would an examiner ask
  this?" is the filter.
- This skill is read-only: do not modify code while quizzing.
