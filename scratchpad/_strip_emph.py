import glob, re

# Decorative single-word / pure-stress emphases to unwrap. Deliberately EXCLUDES
# any \emph{...} that introduces or defines a term (paradigm names, metric names,
# multi-word technical phrases, class labels on first use).
SAFE = [
    "no", "not", "same", "before", "under", "zero", "other", "lowest", "lower",
    "every", "both", "below", "all", "worst", "worse", "without", "why", "whole",
    "visible", "true", "some", "small", "one", "none", "higher", "down", "best",
    "and", "off", "outward", "rises", "these", "last", "approximately", "primary",
    "final", "redundant", "noisy", "understates", "useful", "coarser", "compact",
    "bounds", "completes", "improve", "improved", "generalises", "reassemble",
    "except", "in full", "paired", "split", "merged", "merging", "raw",
]

files = glob.glob('docs/writing/thesis/ch*.tex')
total = 0
for f in files:
    s = open(f, encoding='utf-8').read()
    n_file = 0
    for w in SAFE:
        pat = '\\emph{' + w + '}'
        c = s.count(pat)
        if c:
            s = s.replace(pat, w)
            n_file += c
    if n_file:
        open(f, 'w', encoding='utf-8').write(s)
        print(f"{f.split(chr(47))[-1]}: stripped {n_file}")
        total += n_file
print("total stripped:", total)
