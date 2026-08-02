"""Per-frame metric time-series for the full seq-08 eval.

Parses the evaluate.py per-frame log lines:
  Frame  27: TP= 9  FP= 1  FN= 1  Prec=0.90  Rec=0.90  F1=0.90  meanIoU=0.89
and plots, over the whole sequence:
  (1) rolling precision / recall / F1
  (2) per-frame GT-car count and FN count (where the misses concentrate)
Also prints the micro-averaged aggregate as a sanity check vs the eval footer.
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = re.compile(
    r"Frame\s+(\d+):\s*TP=\s*(\d+)\s*FP=\s*(\d+)\s*FN=\s*(\d+)")


def parse(log_path):
    fr, tp, fp, fn = [], [], [], []
    with open(log_path) as f:
        for ln in f:
            m = LINE.search(ln)
            if m:
                fr.append(int(m.group(1)))
                tp.append(int(m.group(2)))
                fp.append(int(m.group(3)))
                fn.append(int(m.group(4)))
    return (np.array(fr), np.array(tp, float),
            np.array(fp, float), np.array(fn, float))


def rolling(a, w=51):
    if len(a) < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def main(log_path, out_name="seq08_timeseries.png"):
    fr, tp, fp, fn = parse(log_path)
    if len(fr) == 0:
        print("no per-frame lines parsed yet — is the eval still buffering?")
        return

    prec = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.full_like(tp, np.nan), where=(prec + rec) > 0)
    gt = tp + fn

    # micro aggregate
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F = 2 * P * R / (P + R)
    print(f"frames parsed: {len(fr)}")
    print(f"micro: TP={int(TP)} FP={int(FP)} FN={int(FN)}  "
          f"P={P:.3f} R={R:.3f} F1={F:.3f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    w = 51
    ax1.plot(fr, rolling(np.nan_to_num(prec, nan=np.nanmean(prec)), w),
             color="#1f77b4", label="precision (rolling)")
    ax1.plot(fr, rolling(np.nan_to_num(rec, nan=np.nanmean(rec)), w),
             color="#d62728", label="recall (rolling)")
    ax1.plot(fr, rolling(np.nan_to_num(f1, nan=np.nanmean(f1)), w),
             color="#2ca02c", label="F1 (rolling)")
    ax1.axhline(P, ls=":", color="#1f77b4", alpha=0.6)
    ax1.axhline(R, ls=":", color="#d62728", alpha=0.6)
    ax1.axhline(F, ls=":", color="#2ca02c", alpha=0.6)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("metric")
    ax1.set_title(f"Seq 08 full ({len(fr)} frames) — rolling (w={w}) metrics; "
                  f"dotted = micro aggregate (P={P:.3f} R={R:.3f} F1={F:.3f})")
    ax1.legend(loc="lower left", ncol=3)
    ax1.grid(alpha=0.25)

    ax2.fill_between(fr, 0, rolling(gt, w), color="0.8", label="GT cars / frame (rolling)")
    ax2.plot(fr, rolling(fn, w), color="#ff8c00", label="FN / frame (rolling)")
    ax2.plot(fr, rolling(fp, w), color="#e02020", label="FP / frame (rolling)")
    ax2.set_ylabel("count / frame")
    ax2.set_xlabel("frame index")
    ax2.legend(loc="upper left", ncol=3)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, "output", out_name)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get("TMP", "."), "eval_seq08_full.log")
    main(log)
