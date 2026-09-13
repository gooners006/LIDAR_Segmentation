# Post-fix canonical numbers (height-gate normal-sign fix) — 2026-09-14

All from post-fix JSONs on disk. Pre-fix values shown for the search-and-replace.

## Headline seq08 (track-filtered, tau=0.30, full 4071)  [eval_seq08_full_normalfix.json]
PRE : P? R0.730 F1 0.808   (ablation +clf+track: P0.905 R0.730 F1 0.808, TP25478 FP2676 FN9444)
POST: P0.911 R0.732 F1 0.812   TP25568 FP2513 FN9354

## seq00 full (4541)  [eval_seq00_full_normalfix.json]
POST: P0.941 R0.755 F1 0.838   TP42441 FP2684 FN13749

## LOSO all-11 pooled  [loso_postfix/loso_fold*.json vs results/loso/fold_*.json]
PRE : P0.874 R0.719 F1 0.789   TP117159 FP16839 FN45779
POST: P0.888 R0.717 F1 0.793   TP116797 FP14681 FN46141   (FP -2158)

## Ablation seq08 (tab:ablation-stage)
geom-only : PRE P0.149 R0.775 F1 0.250 (TP27051 FP154612 FN7871) -> POST P0.199 R0.773 F1 0.316 (TP27002 FP108831 FN7920)
+clf      : PRE P0.820 R0.677 F1 0.741 -> POST P0.846 R0.676 F1 0.751 (TP23600 FP4303 FN11322)
+clf+trk  : PRE P0.905 R0.730 F1 0.808 (TP25478 FP2676 FN9444) -> POST P0.911 R0.732 F1 0.812 (TP25568 FP2513 FN9354)

## IoU sensitivity seq08 (tab:iou-sensitivity)
tau=0.25 : 0.811 -> 0.815   (post P0.914 R0.735)
tau=0.30 : 0.808 -> 0.812
tau=0.50 : 0.785 -> 0.788   (post P0.884 R0.711)

## Coarsening: cluster grid 0.05 -> 0.10 (seq08, adopted 0.10)
0.05 : P0.910 R0.710 F1 0.798 (TP24812 FP2441 FN10110)
0.10 : P0.911 R0.732 F1 0.812 (TP25568 FP2513 FN9354)
delta: recall +0.022 (0.710->0.732), F1 +0.014 (0.798->0.812), +756 TP, precision ~flat (0.910->0.911)
PRE-FIX prose was: recall 0.699->0.730, +1655 TP, precision 0.903->0.905, F1 0.788->0.808

## #6 split/merge seq08 full (analyze_clustering, ac_seq08_v05/v10 + distbins)
0.05 : ok 55.1%  split 44.8%  merged 0.2% (56)
0.10 : ok 66.0%  split 33.7%  merged 0.8% (284)
"none newly merge" is FALSE: coarsening 0.05->0.10 cuts split 44.8->33.7 but raises merges 56->284 (0.2%->0.8%). Trade-off.

## Distance-binned SPLIT rates seq08 full @0.10 (ac_seq08_distbins.txt)
0-10m : 53.5% (4108/7679)   [thesis cited 66%]
10-20m: 40.2% (4120/10246)
20-40m: 22.3% (3188/14277)
40m+  :  3.5% (64/1855)     [thesis cited 4% at 30-50m]
Qualitative "splits the majority of very-near cars" STILL HOLDS (53.5% > 50%).

## Distance RECALL table tab:distance (stride-20 seq08, post-fix)  [distance_recall_postfix.json]
             PRE(P)  ->  POST         R      F1     IoU    TP/FP/FN(post)
0-10 : 0.896 -> 0.899 | R0.787 F1 0.839 IoU0.902 | 311/35/84
10-20: 0.855 -> 0.869 | R0.862 F1 0.866 IoU0.926 | 432/65/69
20-40: 0.762 -> 0.799 | R0.550 F1 0.652 IoU0.934 | 401/101/328   (was R0.549 F1 0.638, TP400 FP125 FN329)
40+  : 0.393 -> 0.512 | R0.196 F1 0.284 IoU0.943 | 22/21/90       (was F1 0.262, FP34)
Pooled:0.813 -> 0.840 | R0.671 F1 0.746 IoU0.923 | 1166/222/571   (was F1 0.735, TP1165 FP268 FN572)
Recall column UNCHANGED. Caption "track-filtered headline 0.730" -> 0.732.
