"""Cross-domain classifier evaluation (advisor request, 2026-07-14).

Evaluates any binary classifier checkpoint on either validation domain:
  - synthetic: ShapeNet partial renders (Stage A val split, deterministic)
  - real:      mined SemanticKITTI clusters (Stage B val split, seq 08)

Fills the train-domain x test-domain matrix:
  Stage A (synthetic-trained)      -> real val   [new]
  stage_b_scratch (real-trained)   -> synth val  [new]
  stage_b_best (A->B fine-tuned)   -> synth val  [new, forgetting check]
Existing cells come from training logs / Finding #25.

Semantics: bbox-feature normalization always uses the CHECKPOINT's own
training-time mean/std (deployment behavior); only the eval data changes.

Usage:
  .venv\\Scripts\\python.exe scratchpad/cross_domain_classifier_eval.py \
      --ckpt checkpoints/stage_b_best.pth --domain synthetic
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from classifier import PointNetClassifier, CLASS_LABELS  # noqa: E402
from train_classifier import (  # noqa: E402
    STAGE_B_CONFIG,
    TRAIN_CONFIG,
    ShapeNetClassificationDataset,
    StageBDataset,
    validate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--domain", required=True, choices=["synthetic", "real"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    bbox_mean = np.array(ckpt["bbox_mean"], dtype=np.float32)
    bbox_std = np.array(ckpt["bbox_std"], dtype=np.float32)
    print(f"Checkpoint: {args.ckpt} (epoch {ckpt.get('epoch')}) | domain: {args.domain}")

    if args.domain == "synthetic":
        config = TRAIN_CONFIG.copy()
        val_ds = ShapeNetClassificationDataset(config, split="val")
    else:
        config = STAGE_B_CONFIG.copy()
        val_ds = StageBDataset(config["stage_b_root"], "val", config)
    val_ds.bbox_mean = bbox_mean
    val_ds.bbox_std = bbox_std

    val_loader = data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = PointNetClassifier(
        bbox_feat_dim=ckpt["bbox_feat_dim"],
        num_classes=len(CLASS_LABELS),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()  # unweighted; loss is informational only
    metrics = validate(model, val_loader, criterion, device, config)

    print(f"\nAcc: {metrics['val_acc']:.4f} | Macro F1: {metrics['val_macro_f1']:.4f} | "
          f"Macro F1 (thresh): {metrics['val_macro_f1_thresh']:.4f}")
    for cls in CLASS_LABELS:
        r = metrics["per_class"].get(cls, {})
        print(f"  {cls:12s}  P={r.get('precision', 0):.3f}  "
              f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}  "
              f"n={int(r.get('support', 0))}")
    print(f"Confusion matrix (rows=true, cols=pred, order={CLASS_LABELS}):")
    print(metrics["confusion_matrix"])

    out_dir = os.path.join(PROJECT_ROOT, "output", "experiments",
                           "cross_domain_classifier")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_path = os.path.join(out_dir, f"{tag}__on_{args.domain}.json")
    with open(out_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "domain": args.domain,
            "val_acc": float(metrics["val_acc"]),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "val_macro_f1_thresh": float(metrics["val_macro_f1_thresh"]),
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "n_samples": int(metrics["confusion_matrix"].sum()),
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
