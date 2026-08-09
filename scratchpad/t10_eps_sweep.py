"""T10 robustness — does ANY fixed radius match HDBSCAN's recall?

Sweeps dbscan eps / euclidean tolerance on seq 00 (100 frames, per-frame, no
track filter) to check the headline verdict is not an artifact of the single
0.5 m radius used in the main benchmark. HDBSCAN reference (same protocol):
P=0.962 R=0.731 F1=0.831.

Run: .venv\\Scripts\\python.exe scratchpad/t10_eps_sweep.py
"""

import json
import os
import sys

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from classifier import load_classifier  # noqa: E402
from evaluate import (  # noqa: E402
    PROJECT_ROOT, THING_CLASSES_SUPPORTED, evaluate_frame,
)
from pipeline import PIPELINE_CONFIG  # noqa: E402
import glob  # noqa: E402

EPS_VALUES = [0.3, 0.4, 0.5, 0.7]
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth")
OUT = os.path.join(PROJECT_ROOT, "output", "experiments", "t10_clustering", "t10_eps_sweep.json")


def metrics(bins, labs, cls_model, dev, stats):
    tp = fp = fn = 0; ious = []
    for b, l in zip(bins, labs):
        t, f, n, iou = evaluate_frame(
            b, l, 0.3, cls_model=cls_model, cls_device=dev, cls_bbox_stats=stats,
            unknown_threshold=0.50, thing_classes=THING_CLASSES_SUPPORTED)
        tp += t; fp += f; fn += n; ious.extend(iou)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3),
            "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 3)}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, stats = load_classifier(CKPT, dev)
    seq_dir = os.path.join(PROJECT_ROOT, "dataset/sequences/00")
    bins = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))[:100]
    labs = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))[:100]
    results = {}
    for eps in EPS_VALUES:
        for method in ["dbscan", "euclidean"]:
            PIPELINE_CONFIG["clustering_method"] = method
            PIPELINE_CONFIG["dbscan_eps"] = eps
            PIPELINE_CONFIG["euclidean_tolerance"] = eps
            m = metrics(bins, labs, cls_model, dev, stats)
            results[f"{method}_eps{eps}"] = m
            print(f"{method:10s} eps={eps}  P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} mIoU={m['mean_iou']:.3f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    with open(OUT, "w") as f:
        json.dump({"hdbscan_reference": {"precision": 0.962, "recall": 0.731, "f1": 0.831},
                   "sweep": results}, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
