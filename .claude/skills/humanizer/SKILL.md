---
name: humanizer
description: >
  Revise text to remove telltale signs of AI-generated writing (puffery, "delve"/"boasts"/
  "underscore"-style vocabulary, em-dash overuse, rule-of-three padding, "not just X but Y"
  parallelism, canned "In conclusion/Despite challenges" structure, inline-bold-header bullet
  lists, curly quotes) while preserving all facts and meaning. Grounded in Wikipedia:Signs of
  AI writing (WP:AISIGNS). Use this whenever the user says "humanize this," "make this sound
  less like AI/ChatGPT," "this reads like a robot wrote it," or asks to polish/edit a draft,
  email, report, or thesis section for tone — and proactively as a self-check pass before
  handing back any long-form prose you drafted yourself (reports, docs, summaries, emails),
  even if the user didn't ask for a style pass explicitly.
---

# Humanizer Skill

A checklist-driven style pass, not a rewrite-from-scratch. The source taxonomy is
`Wikipedia:Signs of AI writing` (WP:AISIGNS) — a crowd-maintained catalog of the specific
tics that give away LLM-authored prose. It works because these tells are narrow and
mechanical (specific words, specific sentence shapes, specific formatting habits), not
vibes — so you can check for them the same way each time.

## How to use this skill

1. **Read the target text.** Either text the user pasted, or a draft you just wrote that
   you're about to hand back.
2. **Scan it against the categories below**, one pass per category rather than trying to
   catch everything in one read — the categories are unrelated enough that scanning for
   all of them at once causes misses.
3. **Revise surface style only.** Facts, numbers, citations, technical claims, and the
   actual argument must come out identical. This is a style pass, not a content edit —
   if a sentence is wrong or unclear on the merits, that's a different problem than
   sounding AI-written.
4. **Judge each hit in context before changing it.** Every word/pattern below is flagged
   because LLMs overuse it, not because it's forbidden. "The graph's y-axis shows an
   underscore-separated label" is a legitimate, literal use of "underscore" — leave it.
   The tell is the *rate* and the *context* (vague significance-claims, not literal
   description), not the string match.
5. **Return the revised text by default.** Don't add a change-log, diff, or explanation
   of what you fixed unless the user asks for one — a list of "here's what I changed and
   why" is itself exactly the kind of unrequested meta-commentary this skill exists to
   strip out.

## Categories to check

**A. Vocabulary overuse.** Words LLMs reach for far more than human writers do, in
roughly this order of how strongly they give writing away: *delve, boasts, crucial,
underscore(s)/underscoring, pivotal, intricate/intricacies, tapestry, testament,
vibrant, meticulous/meticulously, foster/fostering, showcase/showcasing,
highlight/highlighting, enhance, align with, garner, bolster, landscape* (used
abstractly, e.g. "the evolving landscape of..."), *interplay, valuable, enduring,
key* (as a filler adjective). One or two of these in a long piece is nothing; three or
more, or any of them doing real work in a sentence that's actually just padding, is a
signal. Full era-by-era word lists (which words peaked with which model generation) are
in `references/ai-tells.md` if you need finer-grained detection.

**B. Copula avoidance.** LLMs dodge plain "is/are" in favor of dressed-up verbs: *serves
as, stands as, functions as, operates as, marks, represents* (as in "represents a
shift"), and open definitions with "**X** refers to..." instead of a natural lead-in.
Human writing uses "is/are" constantly and isn't embarrassed by it — put it back unless
the fancier verb is actually doing semantic work (e.g., "functions as a load-bearing
wall" is fine; "functions as the company's headquarters" should just be "is").

**C. Puffery and vague attribution.** Real-estate-listing adjectives with no content:
*nestled, in the heart of, renowned, groundbreaking, rich* (as in "rich history"),
*diverse array, boasts a, showcasing.* Paired with this: hand-wavy sourcing —
*industry reports, experts argue, observers have cited, several sources* — used when no
specific source is actually named. If you can't name who said it, either name them or
cut the claim.

**D. Superficial participle tack-ons.** A sentence that ends by bolting on a vague
"-ing" clause restating significance: "...**, highlighting its growing importance**,"
"...**, reflecting broader industry trends**." These almost never add information — cut
the clause, or replace it with the actual specific fact it's gesturing at.

**E. Negative parallelism and rule-of-three padding.** Constructions like *"not just X
but also Y,"* *"not X, but Y,"* and *"X rather than Y"* used as a rhetorical crutch
rather than because a real contrast exists. Same with triads — three adjectives or three
short parallel phrases in a row — used to make a thin point sound thorough. If the third
item in a triad isn't pulling distinct weight, cut it; vary sentence shape instead of
defaulting to threes.

**F. Elegant variation (synonym-cycling).** Renaming the same thing every time it's
mentioned instead of just repeating the word — "the constraints of socialist realism" →
"the challenging climate of Soviet artistic constraints" → "the confines of
state-imposed norms," all describing one thing. This is a repetition-penalty artifact.
Human writing repeats plain nouns freely; let a repeated word stay repeated rather than
hunting for a fresh synonym each time.

**G. Canned structure.** Formulaic closers — *"Despite these challenges..." / "Looking
ahead..." / "In summary," "In conclusion," "Overall,"* — and a rigid
significance → challenges → future-outlook arc imposed on writing that doesn't need it.
Real writing stops when the point is made; it doesn't self-summarize on the way out.

**H. Formatting tells.**
- Em dashes used more than a sentence or two would call for, especially spaced ( — )
  and standing in for a comma, colon, or parenthesis.
- Heavy, mechanical boldface — bolding phrases like a slide deck's "key takeaways"
  rather than for genuine emphasis.
- Title Case section headings (most style guides — including Wikipedia's — use sentence
  case).
- Inline-bold-header bullet lists: every bullet starts with **A Bolded Term:** followed
  by a description, repeated down the whole list. Fine occasionally; a whole document
  built this way is a tell.
- Curly “smart” quotes/apostrophes appearing inconsistently against straight ones in the
  same piece.

**I. Register leakage.** Chatbot-correspondence phrases bleeding into the actual
document: *"I hope this helps," "Certainly!," "Of course!," "Let me know if...,"
"Would you like me to...," "Here is a..."* — these belong in a chat reply, never in the
artifact itself.

**J. Knowledge-cutoff / hedge disclaimers.** *"As of [date]," "based on available
information," "while specific details are limited/scarce,"* speculative filler like
"keeps a low profile" used to paper over something that's simply unknown. If something
isn't known, say so plainly and specifically instead of hedging around it.

## Write toward these (signs of human writing, not just absence of tells)

Removing tells isn't enough on its own — text can be tell-free and still read stiff if
you don't also add back what LLMs systematically under-use:

- Plain "is/has" constructions ("there is a," "it has a") instead of avoiding them.
- Simple everyday verbs over their stiffer synonyms: *wrote* not authored, *used* not
  utilized, *moved* not relocated, *tried* not attempted, *died* not passed away.
- Superlatives and definitive claims where they're actually true — "the first," "the
  only," "one of the best" — instead of hedging everything into vagueness.
- Ordinary hedges and intensifiers — *very, perhaps, tends to* — instead of trimming
  every sentence to sound maximally crisp. That over-trimmed crispness is itself a tell.
- Wordy, natural connective phrases — *in order to, as a result of, the fact that, a
  part of* — instead of algorithmically compressing them out.

## Don't overcorrect: these are not reliable signals on their own

Per WP:AISIGNS's own caveats — flagging these alone produces false positives:

- Perfect grammar (plenty of skilled human writers have it).
- Formal, academic, or "fancy" vocabulary in general (only *specific* words are tells,
  not formality itself).
- Transition words (*Additionally, Notably, Consequently*) used occasionally — only a
  signal when several stack up together with other tells.
- A mix of casual and formal register in the same piece.
- Unsourced claims by themselves.

If a piece of text only has one of these and nothing from the categories above, leave
it alone — don't manufacture edits to justify the pass.

## Reference

`references/ai-tells.md` has the exhaustive word lists (broken out by which LLM
generation popularized each one) and more source examples, for cases the condensed
checklist above doesn't resolve.

## Rules

- Never change facts, numbers, citations, technical claims, or the argument's substance.
  This skill edits *how* something is said, never *what* is said.
- Judge every match in context — see step 4 above. Don't do a mechanical find-and-replace
  on the word lists.
- Default output is the revised text alone, not a report of changes.
