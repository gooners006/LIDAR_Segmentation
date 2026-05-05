# PCN Encoder — Stacked PointNet Feature Extractor

The encoder is a permutation-invariant feature extractor for point clouds.

**Input:** `(B, N, 3)`
- `B`: batch size
- `N`: number of points
- `3`: `(x, y, z)`

A point cloud has no ordering, so the network cannot rely on spatial adjacency like CNNs on images. PointNet solves this by:

1. Processing each point independently with shared MLPs
2. Aggregating all points with symmetric pooling (`max`)

---

## High-level flow

```
Input points (B, N, 3)
   |
Shared MLPs (Conv1d)
   |
Per-point local features (B, 256, N)
   |
Global max-pool
   |
Global shape feature (B, 256, 1)
   |
Concatenate global feature back to each point
   |
Second shared MLP (Conv1d)
   |
Final global max-pool
   |
1024-d shape embedding (B, 1024)
```

---

## Step-by-step

### 1. Input transpose

```python
x = xyz.transpose(1, 2)
```

`(B, N, 3)` → `(B, 3, N)`

`Conv1d` expects `(B, C, L)` where:
- `C` = channels/features
- `L` = sequence length

Here: channels = 3 coordinates, length = number of points.

---

### 2. Stage 1 — Local feature extraction

```python
self.stage1 = nn.Sequential(
    nn.Conv1d(3, 128, 1),   # kernel_size=1
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Conv1d(128, 256, 1),
    nn.BatchNorm1d(256),
    nn.ReLU(),
)
```

Kernel size = 1 means each point is processed independently. This is mathematically equivalent to applying the same MLP to every point.

**Per point:** `(x, y, z)` → `3` → `128` → `256`

Every point becomes a 256-dimensional vector.

**Output:** `(B, 256, N)`

**Interpretation:** Each point now encodes local geometric semantics — edge-like structure, flat surface, corner, curvature, etc. No interaction between points yet.

---

### 3. First global pooling

```python
g1 = local_feat.max(dim=2, keepdim=True).values
```

`(B, 256, N)` → `(B, 256, 1)`

Max over all `N` points creates one global descriptor.

**Why max pooling?** Point clouds are unordered. Max pooling is permutation-invariant:

```
max(f(p1), f(p2), ..., f(pN))
```

does not depend on point order. This is the key PointNet idea.

**What does `g1` represent?** Each channel stores the strongest activation over all points. Intuitively:
- One channel activates for wheels
- Another for flat surfaces
- Another for sharp corners

After max pooling: *"Does this shape contain this pattern anywhere?"*

`g1` becomes a coarse summary of the whole object.

---

### 4. Broadcast global feature back

```python
g1_expanded = g1.expand(-1, -1, x.size(2))
```

`(B, 256, 1)` → `(B, 256, N)`

Every point receives the same global context vector.

---

### 5. Concatenate local + global

```python
combined = torch.cat([local_feat, g1_expanded], dim=1)
```

- Local: `(B, 256, N)`
- Global: `(B, 256, N)`
- Result: `(B, 512, N)`

Each point now contains:

```
[local geometry | global object context]
```

**Why this matters:** Suppose a local patch looks ambiguous. A flat patch could belong to an airplane wing, a table, or a car hood. Local-only features may confuse them. But once the point knows the global object context — *"This whole shape looks like a car"* — the local interpretation becomes easier.

This is called **local-global feature fusion**.

---

### 6. Stage 2 — Contextual refinement

```python
self.stage2 = nn.Sequential(
    nn.Conv1d(512, 512, 1),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Conv1d(512, 1024, 1),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
)
```

Processes the enriched point features. Each point already knows its local geometry and the entire object structure.

**Output:** `(B, 1024, N)` — each point has a highly contextualized representation.

---

### 7. Final max pooling

```python
return feat.max(dim=2).values
```

`(B, 1024, N)` → `(B, 1024)`

This becomes the final latent vector.

**What does the 1024-d vector contain?** A compressed representation of the partial object:
- Global geometry
- Semantic structure
- Spatial configuration
- Missing-part priors

The decoder later uses this to reconstruct the complete shape.

---

## Why stacked PointNet is better than single-stage

**Vanilla PointNet:**

```
points -> MLP -> global pool
```

Problem: points never interact before pooling.

**Stacked PointNet:**

```
points
 -> local features
 -> global feature
 -> broadcast global context
 -> refine local features
 -> final global feature
```

Now each point can be interpreted relative to the whole object. This significantly improves shape understanding, completion quality, and segmentation accuracy.

---

## Important limitation

Even stacked PointNet still ignores explicit local neighborhoods. It does **not** model:
- Nearest neighbors
- Graph structure
- Local topology

Every point is processed independently. This is why later methods like PointNet++, DGCNN, and Point Transformer perform better on fine geometric detail.

But PCN intentionally uses PointNet because:
- Simple and fast
- Memory efficient
- Robust for completion tasks

The decoder handles most geometric refinement later through grid folding.
