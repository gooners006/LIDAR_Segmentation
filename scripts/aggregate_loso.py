"""Aggregate leave-one-sequence-out (LOSO) detection folds.

Reads results/loso/fold_*.json (written by evaluate.py --json-out) and reports:
  - a per-fold table (the per-sequence recall spread),
  - a pooled micro-average (TP/FP/FN summed across folds; meanIoU pooled over all
    matched IoUs) -- the single number comparable to the old seq-08 headline,
  - a macro mean +/- std across folds (shows cross-sequence variation).

Outputs results/loso/summary.json and prints a Markdown table for the thesis.
"""
import glob
import json
import os
import re
import statistics

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOSO_DIR = os.path.join(PROJECT_ROOT, "results", "loso")

# Canonical per-fold result files only: fold_<NN>.json. Deliberately excludes
# fold_08_full.json and fold_*_window.json (reference/provenance copies) so a
# sequence is never double-counted in the pool.
FOLD_RE = re.compile(r"^fold_\d+\.json$")


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    paths = sorted(p for p in glob.glob(os.path.join(LOSO_DIR, "fold_*.json"))
                   if FOLD_RE.match(os.path.basename(p)))
    if not paths:
        raise SystemExit(f"No fold_<NN>.json files found in {LOSO_DIR}")

    # Guard against duplicate sequences (e.g. a stray copy sneaking past the filter)
    seen = {}
    for p in paths:
        with open(p) as f:
            seq = json.load(f)["seq"]
        if seq in seen:
            raise SystemExit(f"Duplicate seq {seq}: {seen[seq]} and {p}")
        seen[seq] = p

    folds = []
    for p in paths:
        with open(p) as f:
            folds.append(json.load(f))
    folds.sort(key=lambda d: d["seq"])

    # --- Pooled micro-average ---
    pool_tp = sum(d["tp"] for d in folds)
    pool_fp = sum(d["fp"] for d in folds)
    pool_fn = sum(d["fn"] for d in folds)
    pool_prec, pool_rec, pool_f1 = prf(pool_tp, pool_fp, pool_fn)
    all_ious = [x for d in folds for x in d.get("ious", [])]
    pool_miou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    # --- Macro mean +/- std across folds ---
    def macro(key):
        vals = [d[key] for d in folds]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    macro_stats = {k: macro(k) for k in ("precision", "recall", "f1", "mean_iou")}

    # --- Pooled micro-average EXCLUDING the seq-00 fold ---
    # The classical pipeline params (ground removal, clustering, filters) were
    # hand-tuned on seq 00, so the seq-00 fold retains classical-stage familiarity
    # (its classifier is still properly held out). LOSO does not remove that
    # residual; this second pool drops fold-00 so we can show it barely moves the
    # headline -- i.e. the seq-00 pipeline tuning does not drive the result.
    def pool_subset(subset):
        tp = sum(d["tp"] for d in subset)
        fp = sum(d["fp"] for d in subset)
        fn = sum(d["fn"] for d in subset)
        p, r, f = prf(tp, fp, fn)
        iou = [x for d in subset for x in d.get("ious", [])]
        mi = sum(iou) / len(iou) if iou else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
                "f1": f, "mean_iou": mi, "n_matched_ious": len(iou),
                "frames": sum(d["frames"] for d in subset)}

    folds_excl00 = [d for d in folds if d["seq"] != "00"]
    pool_excl00 = pool_subset(folds_excl00) if folds_excl00 else None

    # --- Print per-fold Markdown table ---
    print(f"LOSO cross-validation over {len(folds)} folds\n")
    hdr = "| Seq | Frames | Precision | Recall | F1 | Mean IoU | TP | FP | FN |"
    sep = "|-----|-------:|----------:|-------:|---:|---------:|---:|---:|---:|"
    print(hdr)
    print(sep)
    for d in folds:
        print(f"| {d['seq']} | {d['frames']} | {d['precision']:.3f} | "
              f"{d['recall']:.3f} | {d['f1']:.3f} | {d['mean_iou']:.3f} | "
              f"{d['tp']} | {d['fp']} | {d['fn']} |")
    print(f"| **pooled (all 11)** | {sum(d['frames'] for d in folds)} | "
          f"**{pool_prec:.3f}** | **{pool_rec:.3f}** | **{pool_f1:.3f}** | "
          f"**{pool_miou:.3f}** | {pool_tp} | {pool_fp} | {pool_fn} |")
    if pool_excl00 is not None:
        pe = pool_excl00
        print(f"| pooled (excl. seq00) | {pe['frames']} | "
              f"{pe['precision']:.3f} | {pe['recall']:.3f} | {pe['f1']:.3f} | "
              f"{pe['mean_iou']:.3f} | {pe['tp']} | {pe['fp']} | {pe['fn']} |")
        print(f"\nSeq-00 pipeline-tuning check: pooled recall "
              f"{pool_rec:.3f} (all) vs {pe['recall']:.3f} (excl. seq00), "
              f"delta={pool_rec - pe['recall']:+.3f}")

    print("\nMacro mean +/- std across folds:")
    for k, (m, s) in macro_stats.items():
        print(f"  {k:10s}: {m:.3f} +/- {s:.3f}")

    summary = {
        "n_folds": len(folds),
        "folds": [
            {k: d[k] for k in ("seq", "frames", "tp", "fp", "fn",
                               "precision", "recall", "f1", "mean_iou",
                               "classifier_ckpt")}
            for d in folds
        ],
        "pooled_micro": {
            "tp": pool_tp, "fp": pool_fp, "fn": pool_fn,
            "precision": pool_prec, "recall": pool_rec, "f1": pool_f1,
            "mean_iou": pool_miou, "n_matched_ious": len(all_ious),
        },
        "pooled_micro_excl_seq00": pool_excl00,
        "macro": {k: {"mean": m, "std": s}
                  for k, (m, s) in macro_stats.items()},
    }
    out = os.path.join(LOSO_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
