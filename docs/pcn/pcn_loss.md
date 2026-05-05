# PCN Loss — Chamfer Distance

The loss function measures how close the predicted point cloud is to the ground truth. PCN uses **Chamfer Distance (CD)** — a bidirectional nearest-neighbor metric — applied to both the coarse and fine outputs.

**Source:** `src/pcn.py`, functions `chamfer_distance_chunked` and `pcn_loss`

---

## High-level flow

```
Coarse output (B, 1024, 3)     GT subsampled (B, 1024, 3)
         |                              |
         +--------- CD coarse ---------+
                       |
Fine output (B, 16384, 3)      GT full (B, 16384, 3)
         |                              |
    subsample to 4096              subsample to 4096
         |                              |
         +---------- CD fine ----------+
                       |
                       v
           total = cd_coarse + 0.5 * cd_fine
```

---

## What is Chamfer Distance?

Chamfer Distance measures the distance between two point clouds by looking at nearest neighbors in **both directions**.

Given predicted cloud `P` (M points) and ground truth cloud `G` (N points):

```
Direction 1 (pred → gt):
    For every predicted point, find its closest GT point.
    Average all those distances.

Direction 2 (gt → pred):
    For every GT point, find its closest predicted point.
    Average all those distances.

CD = mean(min_dist pred→gt) + mean(min_dist gt→pred)
```

### Why both directions?

Neither direction alone is sufficient:

**Pred → GT only (accuracy):**
```
Prediction: one point at the center of the GT cloud

    GT:  . . . . .        Pred: *
         . . . . .
         . . . . .

    pred→gt distance is small (the one point is close to some GT point)
    But the prediction is useless — it collapsed to a single point
```

This would score well because the single predicted point has a small distance to its nearest GT neighbor. But most of the shape is missing.

**GT → Pred only (coverage):**
```
Prediction: GT cloud + thousands of random noise points

    GT:  . . . . .        Pred: . . . . . x x x x
         . . . . .              . . . . . x x x
         . . . . .              . . . . . x x x x x

    gt→pred distance is small (every GT point has a nearby pred point)
    But prediction has tons of garbage points
```

This would score well because every GT point is covered. But the extra noise points go unpenalized.

**Both directions together:**
- Pred → GT penalizes predicted points far from any GT point (removes noise)
- GT → Pred penalizes GT regions not covered by predictions (ensures coverage)

Together they force the prediction to match the GT both in accuracy and completeness.

---

## Step-by-step: `chamfer_distance_chunked`

### The memory problem

The naive approach computes all pairwise distances at once:

```python
dist = torch.cdist(pred, gt)    # (B, M, N) distance matrix
```

For the fine output with `M = N = 16384`:
- Matrix size: `16384 x 16384 x 4 bytes = 1 GB` per sample
- Batch of 8: `8 GB` just for the distance matrix
- Plus gradients: another `8 GB`
- Total: exceeds the 8 GB VRAM of the RTX 3070 Ti

### The solution: per-sample chunking

Process one sample at a time, and only compute distances to a chunk of the other cloud at once.

```python
B = pred.shape[0]
cd_sum = torch.tensor(0.0, device=pred.device)
```

Initialize an accumulator. We'll sum CD across all samples and divide at the end.

### Direction 1: pred → gt

For each predicted point, find its nearest GT point.

```python
for b in range(B):
    p = pred[b]    # (M, 3)
    g = gt[b]      # (N, 3)

    min_p2g = torch.full((M,), float("inf"), device=p.device)
```

`min_p2g` stores the running minimum distance from each predicted point to any GT point seen so far. Initialized to infinity so any real distance will be smaller.

```python
    for i in range(0, N, chunk_size):
        g_chunk = g[i:i + chunk_size]    # (chunk, 3)
```

Instead of comparing all M predicted points against all N GT points at once, we take N in chunks. With `chunk_size = 2048` and `N = 16384`, that's 8 iterations.

```python
        dist = torch.cdist(p.unsqueeze(0), g_chunk.unsqueeze(0)).squeeze(0)    # (M, chunk)
```

`torch.cdist` computes pairwise Euclidean distances. The unsqueeze/squeeze is needed because `cdist` expects a batch dimension.

The result is an `(M, chunk)` matrix — distance from every predicted point to every GT point in this chunk.

Memory: `16384 x 2048 x 4 bytes = 128 MB`. Manageable.

```python
        min_p2g = torch.min(min_p2g, dist.min(dim=1).values)
```

For each predicted point, find the minimum distance within this chunk (`dist.min(dim=1)`), then update the running minimum. After all chunks, `min_p2g[i]` holds the distance from predicted point `i` to its true nearest GT point.

**Visualization of the chunking:**

```
Full distance matrix (M x N):
+--+--+--+--+--+--+--+--+
|  |  |  |  |  |  |  |  |  M = 16384 rows (pred points)
|  |  |  |  |  |  |  |  |
+--+--+--+--+--+--+--+--+
 c1  c2  c3  c4  c5  c6  c7  c8    N = 16384 cols, 8 chunks of 2048

We compute one chunk at a time:
 [c1] → find min per row → update running min
 [c2] → find min per row → update running min
 ...
 [c8] → find min per row → update running min

Final min_p2g = true row-wise minimum across all columns
```

### Direction 2: gt → pred

Same logic, reversed. For each GT point, find its nearest predicted point:

```python
    min_g2p = torch.full((N,), float("inf"), device=g.device)
    for i in range(0, M, chunk_size):
        p_chunk = p[i:i + chunk_size]
        dist = torch.cdist(g.unsqueeze(0), p_chunk.unsqueeze(0)).squeeze(0)    # (N, chunk)
        min_g2p = torch.min(min_g2p, dist.min(dim=1).values)
```

Now we chunk across predicted points instead of GT points.

### Combine both directions

```python
    cd_sum = cd_sum + min_p2g.mean() + min_g2p.mean()

return cd_sum / B
```

Average the nearest-neighbor distances in each direction, sum them, then average over the batch.

---

## Step-by-step: `pcn_loss`

The combined loss supervises both decoder outputs.

### Coarse loss

```python
cd_coarse = chamfer_distance_chunked(coarse, gt_coarse, chunk_size=1024)
```

| Input | Shape | Description |
|-------|-------|-------------|
| `coarse` | `(B, 1024, 3)` | Decoder coarse output |
| `gt_coarse` | `(B, 1024, 3)` | GT randomly subsampled to 1024 points |

Distance matrix per sample: `1024 x 1024 x 4 bytes = 4 MB`. Small — `chunk_size=1024` means it's computed in a single pass.

### Fine loss — subsampling

The fine output has 16,384 points. Even with chunking, backpropagation through the full CD is expensive. The solution: randomly subsample both sides to 4096 points.

```python
B, N, _ = gt.shape
if N > fine_gt_samples:
    idx = torch.randint(0, N, (B, fine_gt_samples), device=gt.device)
    gt_fine = torch.gather(gt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
```

**How `torch.gather` works here:**

```
gt:     (B, 16384, 3)        full ground truth
idx:    (B, 4096)            random indices into dim=1
```

`idx.unsqueeze(-1).expand(-1, -1, 3)` makes the index tensor `(B, 4096, 3)` — the same index is used for all 3 coordinates of each point.

`torch.gather(gt, 1, ...)` picks 4096 points from the 16384 by their indices along dimension 1.

Result: `gt_fine` is `(B, 4096, 3)` — a random subset of the full GT.

Same for the fine predictions:

```python
M = fine.shape[1]
if M > fine_gt_samples:
    idx_f = torch.randint(0, M, (B, fine_gt_samples), device=fine.device)
    fine_sub = torch.gather(fine, 1, idx_f.unsqueeze(-1).expand(-1, -1, 3))
```

`fine_sub`: `(B, 4096, 3)` — random subset of the fine output.

### Fine CD

```python
cd_fine = chamfer_distance_chunked(fine_sub, gt_fine, chunk_size=2048)
```

Now the distance matrix per sample is `4096 x 4096 x 4 bytes = 64 MB` (computed in 2 chunks of 2048). With backprop this stays well under 8 GB VRAM.

### Total loss

```python
total = cd_coarse + alpha * cd_fine
return total, cd_coarse, cd_fine
```

`alpha = 0.5` downweights the fine loss.

**Why downweight fine?** Early in training, the coarse points are still scattered randomly. The fine stage adds offsets to these bad coarse positions, producing equally bad fine points. The fine CD is noisy and uninformative at this stage. By weighting it at 0.5, we let the coarse skeleton stabilize first. As coarse points converge to reasonable positions, the fine loss becomes meaningful and guides surface detail.

**Why return all three?** Logging. During training we print `cd_coarse` and `cd_fine` separately to monitor which stage is converging and which is struggling.

---

## Memory budget

| Operation | Points | Matrix size | Memory |
|-----------|--------|-------------|--------|
| Coarse CD | 1024 vs 1024 | 1024 x 1024 | 4 MB |
| Fine CD (subsampled) | 4096 vs 4096 | 4096 x 2048 (chunked) | 32 MB |
| **Total peak** | | | **~3 GB** (including model + gradients) |

Fits comfortably in 8 GB VRAM at batch size 8.

---

## Limitations and future work

**CD can produce uneven point distributions.** Chamfer Distance doesn't penalize clustering — if 100 predicted points collapse to the same location, CD only cares that each one is close to some GT point. This can cause "over-concentrated" regions.

**Solution: Earth Mover's Distance (EMD).** EMD finds a 1-to-1 matching between predicted and GT points that minimizes total distance. This forces uniform coverage. The standard PCN approach:
- EMD on coarse output (1024 points — manageable cost)
- CD on fine output (16384 points — EMD too expensive)

EMD requires either a custom CUDA kernel or a Sinkhorn approximation. **TODO: add EMD for the coarse stage.**

**Subsampling introduces variance.** Random 4096-point subsets of the fine output and GT change every forward pass. This adds noise to the gradient but also acts as implicit regularization. On average over many iterations, the loss approximates the full CD.
