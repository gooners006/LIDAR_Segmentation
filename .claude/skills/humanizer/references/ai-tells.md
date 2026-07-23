# AI writing tells — full reference

Detailed backup for `SKILL.md`'s condensed checklist, drawn from
`Wikipedia:Signs of AI writing` (WP:AISIGNS). Use this when the condensed checklist
doesn't resolve a case, or when you want the finer-grained (era-specific) word lists.

## Contents

1. [Vocabulary overuse by LLM era](#vocabulary-overuse-by-llm-era)
2. [Content-level patterns](#content-level-patterns)
3. [Grammar-level patterns](#grammar-level-patterns)
4. [Formatting-level patterns](#formatting-level-patterns)
5. [Communication-register leakage](#communication-register-leakage)
6. [Older/historical tells](#olderhistorical-tells)
7. [Ineffective indicators (do not use these alone)](#ineffective-indicators)

## Vocabulary overuse by LLM era

Word overuse doesn't mean the word is banned — "underscore" can be a perfectly literal
verb. The signal is elevated *rate* relative to normal prose, especially several from
the same list appearing close together.

**2023–mid-2024 (GPT-4 era):** Additionally (esp. sentence-initial), boasts, bolstered,
crucial, delve, emphasizing, enduring, garner, intricate/intricacies, interplay, key,
landscape (abstract noun, e.g. "the political landscape"), meticulous/meticulously,
pivotal, underscore, tapestry, testament, valuable, vibrant.

**Mid-2024–mid-2025 (GPT-4o era):** align with, bolstered, crucial, emphasizing,
enhance, enduring, fostering, highlighting, pivotal, showcasing, underscore, vibrant.

**Mid-2025+ (GPT-5 era):** emphasizing, enhance, highlighting, showcasing, plus
notability/attribution words (see below).

**Grok-specific:** causal, empirical, correlate, continued heavy "underscore" use.
Grok also skews toward "X rather than Y" constructions and very long output relative to
Gemini/Claude.

**Model differences (general):** Gemini and Claude tend to be more concise and less
prone to "broader significance/legacy" framing than ChatGPT and Grok. Don't assume
every tell applies equally to text from every model.

## Content-level patterns

**Undue emphasis on significance/legacy.** *stands/serves as, is a testament/reminder,
a crucial/pivotal/vital/significant/key role/moment, underscores/highlights its
importance, reflects broader, symbolizing its ongoing/enduring/lasting, contributing to
the, setting the stage for, marking/shaping the, represents/marks a shift, key turning
point, evolving landscape, focal point, indelible mark, deeply rooted.*

**Canned notability/attribution/media-coverage language.** *independent coverage,
[local/regional/national] media outlets, trade publications, profiled in, written by a
leading expert, active social media presence, maintains a strong digital presence.*

**Superficial analysis via dangling participles.** Pattern: a factual sentence + a
tacked-on "-ing" clause claiming significance, often with vague third-party framing —
*highlighting, underscoring, emphasizing, ensuring, reflecting, symbolizing,
contributing to, cultivating, fostering, encompassing, enhancing* + *valuable insights,
align/resonate with.*

**Leads that treat a topic as a proper-noun definition.** "**X** refers to..." instead
of a natural definitional opening.

## Grammar-level patterns

**Copula avoidance.** Documented ~10% drop in "is"/"are" usage in academic writing
post-2023 (Geng & Trotta). Replacements: *serves as/stands as/marks/functions as/
operates as [a], boasts/features/maintains/offers [a], refers to.*

**Negative parallelisms** (three subtypes):
- *Not just X, but also Y* — "It is not only dismissive but also..."
- *Not X, but Y* — "This dispersal is not dissolution. Rather, it constitutes..."
- *X rather than Y* — especially common in Grok output.

**Rule of three.** Triads of adjectives or short phrases ("adjective, adjective,
adjective" / "phrase, phrase, and phrase") used to make superficial coverage look
comprehensive.

**Elegant variation / synonym-cycling.** Over-avoiding word repetition (a
repetition-penalty artifact) produces unnatural synonym cycling for the same referent
across a paragraph. Caveat: some non-native English speakers are also taught to avoid
repetition (e.g., in Italian schools), so this alone isn't proof — look for it stacked
with other tells.

**Words with simple synonyms are under-used.** Empirically, human Wikipedia prose uses
*wrote, moved, used, tried, died* more than AI text does; AI text skews toward
*authored, relocated, utilized, attempted, passed away.* Simple word choice is itself
mildly humanizing.

**Other syntax more common in human writing (so writing toward these helps):** simple
is/has phrases ("there is a," "it has a"); superlative/definitive statements ("one of
the best," "is the only," "was the first"); hedging qualifiers/intensifiers ("very,"
"perhaps," "tends to"); wordy natural constructions ("as a result of," "in order to,"
"all of the," "a part of," "the fact that").

## Formatting-level patterns

- **Title case** in headings (AI chatbots capitalize all main words; most style guides,
  including Wikipedia's, use sentence case).
- **Overuse of boldface** — mechanical bolding of many phrases in a "key takeaways"
  style, inherited from listicles/sales decks/READMEs.
- **Inline-header vertical lists** — every bullet starts with a bolded term + colon,
  then a description, repeated down the list.
- **Overuse of em dashes** — more than typical human density, often spaced ( — ),
  replacing commas/parens/colons. Most useful as a signal when combined with others;
  common on chat/talk-style text especially. Some newer model versions have started
  suppressing this tendency, so absence doesn't rule AI out.
- **Emoji as formatting** — decorating headings/bullets with emoji (mostly seen in
  informal/chat-adjacent text, now rarer).
- **Curly quotation marks/apostrophes** used inconsistently against straight ones.
  Caveat: NOT proof alone — word processors' "smart quotes," OS-level autocorrect, and
  certain style-guide typesetting also produce curly quotes. Also, Gemini and Claude
  models typically do *not* default to curly quotes, so their absence doesn't clear a
  text either.
- **Skipped heading levels** and **thematic-break rules before headings** (a Markdown
  convention that sometimes leaks into non-Markdown output).

## Communication-register leakage

Chatbot correspondence tone accidentally left in the actual content: *I hope this
helps, Of course!, Certainly!, You're absolutely right!, Would you like..., is there
anything else, let me know, more detailed breakdown, here is a.*

**Knowledge-cutoff/hedge disclaimers:** *as of [date], up to my last training update,
as of my last knowledge update, while specific details are limited/scarce,
not widely available/documented/disclosed, based on available information.* Also
watch for speculative filler asserting someone "maintains a low profile" or "keeps
personal details private" as a way of papering over an actual lack of information.

**Unfilled placeholders:** literal bracketed Mad-Libs text left in ("[Describe the
specific section...]", "[Your Name]") — an obvious leftover-artifact tell, rare but
unambiguous when present.

## Older/historical tells

Less common in current-generation output, but still useful for older text:

- **Didactic disclaimers (~2022–2024):** *it's important/critical/crucial to
  note/remember/consider, worth noting, may vary* — advice to an imagined reader,
  especially around safety/controversial/jurisdiction-varying topics.
- **Section summaries:** older LLMs frequently added "Conclusion" sections or ended
  paragraphs by restating the core idea — *In summary, In conclusion, Overall.*
- **Prompt-refusal leakage:** *as an AI language model, as a large language model,
  I cannot offer medical advice but I can..., I'm sorry* — literal refusal boilerplate
  accidentally left in output.

## Ineffective indicators

Do not flag text based on these alone — each has a substantial rate of false positives,
and over-relying on them just produces edits that don't actually address the problem:

- **Perfect grammar** — plenty of skilled human writers have it.
- **Mixed casual/formal register, or "clinical" + "emotional" language together** — can
  reflect a technical-field writer's casual style, youth, neurodivergence, or (in a
  collaborative document) multiple authors.
- **"Bland" or "robotic" prose** in a vague, impressionistic sense — without a specific
  tell from the lists above, this judgment is unreliable.
- **"Fancy"/"academic"/"formal" prose in general** — only *specific* words are
  overused by LLMs; formality itself is not the signal.
- **Transition words in isolation** (*Additionally, Consequently, Notably*) — legitimate
  and common in human essay-style writing too; only meaningful stacked with other tells.
- **Unsourced content** — extremely common in ordinary human-written text as well.
