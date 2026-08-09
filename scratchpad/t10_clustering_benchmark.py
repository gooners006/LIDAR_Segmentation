"""T10 — clustering-method benchmark (thesis table, adopt nothing).

Compares production HDBSCAN vs Open3D DBSCAN vs PCL-style Euclidean clustering
at a FIXED cluster_voxel_size=0.10 and identical pre/post pipeline stages.
Reports standard evaluate.py detection metrics (per-frame, no track filter —
see NOTE) plus pure clustering-algorithm runtime per frame.

NOTE (deviation, logged): the metrics are computed with track-level filtering
OFF. A strided seq-08 sample is incompatible with the centroid tracker (it
links consecutive frames; stride-20 makes every detection a length-1 track, so
min_track_length=2 rejects everything). Per-frame eval also isolates clustering
quality from the method-independent track post-filter, which is what this
benchmark is for. seq-00 uses 100 contiguous frames; seq-08 uses a stride-20
sample (state: stride-20 agrees with the full run to 1.4%).

Params are standard and UNTUNED (fairness): eps/tolerance 0.5 m, min size 10.

Run: .venv\\Scripts\\python.exe scratchpad/t10_clustering_benchmark.py
"""

import copy
import json
import os
import sys
import time

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from classifier import load_classifier  # noqa: E402
from evaluate import (  # noqa: E402
    PROJECT_ROOT,
    THING_CLASSES_SUPPORTED,
    evaluate_frame,
    get_frame_detections,
)
from pipeline import PIPELINE_CONFIG, _dispatch_clustering  # noqa: E402
import glob  # noqa: E402
import open3d as o3d  # noqa: E402

METHODS = ["hdbscan", "dbscan", "euclidean"]
IOU_THRESHOLD = 0.3
CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "stage_b_scratch_best.pth")
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "experiments", "t10_clustering")
os.makedirs(OUT_DIR, exist_ok=True)


def frame_paths(seq: str, max_frames: int | None, stride: int):
    seq_dir = os.path.join(PROJECT_ROOT, f"dataset/sequences/{seq}")
    bins = sorted(glob.glob(os.path.join(seq_dir, "velodyne/*.bin")))
    labs = sorted(glob.glob(os.path.join(seq_dir, "labels/*.label")))
    assert len(bins) == len(labs), f"{len(bins)} bins vs {len(labs)} labels"
    if max_frames is not None:
        bins, labs = bins[:max_frames], labs[:max_frames]
    bins, labs = bins[::stride], labs[::stride]
    return bins, labs


def run_metrics(bins, labs, cls_model, cls_device, cls_stats):
    """Per-frame (no track filter) detection metrics for each method."""
    out = {}
    for method in METHODS:
        PIPELINE_CONFIG["clustering_method"] = method
        tp = fp = fn = 0
        ious = []
        for b, l in zip(bins, labs):
            t, f, n, iou = evaluate_frame(
                b, l, IOU_THRESHOLD,
                cls_model=cls_model, cls_device=cls_device, cls_bbox_stats=cls_stats,
                unknown_threshold=0.50, thing_classes=THING_CLASSES_SUPPORTED,
            )
            tp += t; fp += f; fn += n; ious.extend(iou)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[method] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "mean_iou": round(float(np.mean(ious)) if ious else 0.0, 3),
        }
        print(f"    {method:10s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
              f"mIoU={out[method]['mean_iou']:.3f} (TP={tp} FP={fp} FN={fn})")
    return out


def run_timing(bins, labs):
    """Pure clustering-algorithm runtime on IDENTICAL coarse (0.10 m) clouds."""
    cv = PIPELINE_CONFIG["cluster_voxel_size"]
    cfgs = {}
    for m in METHODS:
        c = copy.deepcopy(PIPELINE_CONFIG)
        c["clustering_method"] = m
        c["adaptive_hdbscan"] = False
        cfgs[m] = c
    times = {m: [] for m in METHODS}
    ncl = {m: [] for m in METHODS}
    warmed = False
    # cls_model=None -> get_frame_detections skips the classifier (fast); we only
    # need its objects_pcd (index 4 of the returned tuple).
    PIPELINE_CONFIG["clustering_method"] = "hdbscan"
    for b, l in zip(bins, labs):
        res = get_frame_detections(
            b, l, cls_model=None, thing_classes=THING_CLASSES_SUPPORTED, keep_unknown=False,
        )
        objects_pcd = res[4]
        coarse = objects_pcd.voxel_down_sample(voxel_size=cv)
        coarse_pts = np.asarray(coarse.points)
        if len(coarse_pts) == 0:
            continue
        if not warmed:
            for m in METHODS:
                _dispatch_clustering(coarse_pts, cfgs[m])
            warmed = True
        for m in METHODS:
            t0 = time.perf_counter()
            labels = _dispatch_clustering(coarse_pts, cfgs[m])
            times[m].append((time.perf_counter() - t0) * 1000.0)
            ncl[m].append(int((np.unique(labels) >= 0).sum()))
    out = {}
    for m in METHODS:
        arr = np.array(times[m])
        out[m] = {
            "median_ms": round(float(np.median(arr)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "median_n_clusters": round(float(np.median(ncl[m])), 1),
            "n_frames_timed": len(arr),
        }
        print(f"    {m:10s} median={out[m]['median_ms']:.2f} ms  "
              f"mean={out[m]['mean_ms']:.2f} ms  "
              f"median_clusters={out[m]['median_n_clusters']:.1f}")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_model, cls_stats = load_classifier(CKPT, device)
    print(f"Classifier: {CKPT} on {device}")
    print(f"Fixed cluster_voxel_size={PIPELINE_CONFIG['cluster_voxel_size']}; "
          f"dbscan eps={PIPELINE_CONFIG['dbscan_eps']}/min_pts={PIPELINE_CONFIG['dbscan_min_points']}; "
          f"euclidean tol={PIPELINE_CONFIG['euclidean_tolerance']}/min={PIPELINE_CONFIG['euclidean_min_cluster_size']}")

    scenarios = [
        {"name": "seq00_100", "seq": "00", "max_frames": 100, "stride": 1},
        {"name": "seq08_stride20", "seq": "08", "max_frames": None, "stride": 20},
    ]
    results = {"params": {
        "cluster_voxel_size": PIPELINE_CONFIG["cluster_voxel_size"],
        "dbscan_eps": PIPELINE_CONFIG["dbscan_eps"],
        "dbscan_min_points": PIPELINE_CONFIG["dbscan_min_points"],
        "euclidean_tolerance": PIPELINE_CONFIG["euclidean_tolerance"],
        "euclidean_min_cluster_size": PIPELINE_CONFIG["euclidean_min_cluster_size"],
        "iou_threshold": IOU_THRESHOLD,
        "track_filter": False,
        "classifier": os.path.basename(CKPT),
    }, "scenarios": {}}

    for sc in scenarios:
        bins, labs = frame_paths(sc["seq"], sc["max_frames"], sc["stride"])
        print(f"\n=== {sc['name']}: {len(bins)} frames (seq {sc['seq']}, stride {sc['stride']}) ===")
        print("  metrics (per-frame, no track filter):")
        metrics = run_metrics(bins, labs, cls_model, device, cls_stats)
        print("  timing (pure clustering on identical 0.10 m coarse cloud):")
        timing = run_timing(bins, labs)
        results["scenarios"][sc["name"]] = {
            "seq": sc["seq"], "n_frames": len(bins), "stride": sc["stride"],
            "metrics": metrics, "timing": timing,
        }

    out_path = os.path.join(OUT_DIR, "t10_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
