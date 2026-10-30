"""Thesis figure: per-sequence car acceptance, raw vs completed, with the pooled ceiling.

Reads the saved scoring output; the legend sits above the axes so it covers no bar.
Output: figures/plausibility_acceptance_final.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
SRC = os.path.join(PROJECT_ROOT, "output", "experiments", "plausibility_chen")
OUT = os.path.join(os.path.dirname(__file__), "..", "plausibility_acceptance_final.png")


def main():
    per_seq = json.load(open(os.path.join(SRC, "per_seq.json")))
    summary = json.load(open(os.path.join(SRC, "summary.json")))
    seqs = sorted(per_seq)
    x = np.arange(len(seqs))
    raw = [per_seq[s]["raw_accept"] for s in seqs]
    comp = [per_seq[s]["completed_accept"] for s in seqs]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(x - 0.19, raw, 0.38, color="#c44", label="raw partial")
    ax.bar(x + 0.19, comp, 0.38, color="#3a7", label="completed")
    ax.axhline(summary["ceiling"]["accept"], color="#333", ls="--",
               label=f"ceiling (pooled {summary['ceiling']['accept']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.set_ylabel("car acceptance rate")
    ax.set_xlabel("sequence")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170)
    print("saved", OUT)


if __name__ == "__main__":
    main()
