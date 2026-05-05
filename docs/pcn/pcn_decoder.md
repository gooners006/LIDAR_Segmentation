# PCN Decoder — Coarse-to-Fine with Grid Folding

The decoder takes the 1024-d global feature from the encoder and reconstructs a complete point cloud in two stages: first a coarse skeleton, then fine surface detail via 2D grid folding.

**Input:** `(B, 1024)` — global shape feature from encoder

**Output:**
- Coarse: `(B, 1024, 3)` — 1024 seed points (rough skeleton)
- Fine: `(B, 16384, 3)` — 16,384 surface points (detailed shape)

---

## High-level flow

```
Global feature (B, 1024)
   |
   v
[Coarse stage] — fully-connected layers
   |
   v
1024 seed points (B, 1024, 3)
   |
   v
[Fine stage] — attach 4x4 grid to each seed, predict offsets
   |
   v
16,384 surface points (B, 16384, 3)
```

---

## Step-by-step

### 1. Coarse stage — Predicting the skeleton

```python
self.coarse_fc = nn.Sequential(
    nn.Linear(feat_dim, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, num_coarse * 3),
)
```

Three fully-connected layers expand the 1024-d feature into raw coordinates:

```
(B, 1024) → Linear → (B, 1024) → Linear → (B, 1024) → Linear → (B, 3072)
```

Then reshape:

```python
coarse = self.coarse_fc(feat).reshape(B, self.num_coarse, 3)
```

`(B, 3072)` → `(B, 1024, 3)`

Each group of 3 values becomes one `(x, y, z)` point. The result is 1024 seed points that form the rough skeleton of the complete shape.

**Why fully-connected and not something fancier?** The coarse stage only needs to place ~1000 points in roughly the right positions. FC layers are simple and effective for this. The hard work of capturing surface detail is left to the fine stage.

**What do coarse points look like?** Early in training, they're scattered randomly. As training progresses, they converge to an even spread across the object's surface — like a low-resolution sampling of the complete shape.

---

### 2. Pre-computed 2D folding grid

```python
lin = torch.linspace(-0.05, 0.05, grid_size)     # [-0.05, -0.017, 0.017, 0.05]
grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))
```

This creates a fixed 4x4 grid of 2D coordinates, stored as a buffer (not a learnable parameter):

```
Grid (16 points, 2D):

   (-0.05, -0.05)  (-0.02, -0.05)  (+0.02, -0.05)  (+0.05, -0.05)
   (-0.05, -0.02)  (-0.02, -0.02)  (+0.02, -0.02)  (+0.05, -0.02)
   (-0.05, +0.02)  (-0.02, +0.02)  (+0.02, +0.02)  (+0.05, +0.02)
   (-0.05, +0.05)  (-0.02, +0.05)  (+0.02, +0.05)  (+0.05, +0.05)
```

After reshape: `(16, 2)` — 16 points, each with a `(u, v)` coordinate.

**Why `register_buffer`?** The grid is a constant — it moves to GPU with the model but doesn't get gradients or updates during training.

**Why these specific values?** The range `[-0.05, 0.05]` is small relative to the object scale (meters). Each grid represents a tiny flat patch. The network will learn to deform these patches to match the local surface.

---

### 3. Fine stage — Tiling

This is where the math gets interesting. We need to create 16,384 input vectors — one for each fine point — by combining information from three sources.

#### 3a. Tile coarse points

```python
coarse_tiled = coarse.unsqueeze(2).expand(-1, -1, G, -1).reshape(B, self.num_fine, 3)
```

Each of the 1024 coarse points is repeated 16 times (once per grid point):

```
coarse:        (B, 1024, 3)
unsqueeze(2):  (B, 1024, 1, 3)
expand:        (B, 1024, 16, 3)     ← each point copied 16 times
reshape:       (B, 16384, 3)
```

Concretely, if coarse point #0 is at position `(1.2, 0.5, -0.3)`:

```
Index 0:   (1.2, 0.5, -0.3)   ← for grid point (0,0)
Index 1:   (1.2, 0.5, -0.3)   ← for grid point (0,1)
...
Index 15:  (1.2, 0.5, -0.3)   ← for grid point (3,3)
Index 16:  (next coarse point) ← for grid point (0,0)
...
```

#### 3b. Tile the 2D grid

```python
grid_tiled = self.grid.unsqueeze(0).unsqueeze(0).expand(B, self.num_coarse, -1, -1).reshape(B, self.num_fine, 2)
```

The same 16-point grid is repeated for every coarse point:

```
grid:          (16, 2)
unsqueeze x2:  (1, 1, 16, 2)
expand:        (B, 1024, 16, 2)    ← same grid for every coarse point
reshape:       (B, 16384, 2)
```

So indices 0–15 have the 16 grid UV values, indices 16–31 have the same 16 values again, etc.

#### 3c. Tile the global feature

```python
feat_tiled = feat.unsqueeze(1).expand(-1, self.num_fine, -1)
```

The single 1024-d global feature is copied to all 16,384 points:

```
feat:          (B, 1024)
unsqueeze(1):  (B, 1, 1024)
expand:        (B, 16384, 1024)
```

Every fine point gets the same global shape context.

---

### 4. Concatenation

```python
fold_input = torch.cat([coarse_tiled, grid_tiled, feat_tiled], dim=2)
```

For each of the 16,384 fine points, we concatenate:

```
[coarse_xyz (3) | grid_uv (2) | global_feat (1024)]  =  1029 dimensions
```

| Component | Size | Role |
|-----------|------|------|
| `coarse_xyz` | 3 | Where on the object (anchor position) |
| `grid_uv` | 2 | Where on the local patch (offset identity) |
| `global_feat` | 1024 | What object this is (shape context) |

**Result:** `(B, 16384, 1029)`

**Intuition:** Each fine point knows three things:
- "I'm near coarse point #42, which is on the car hood" (from `coarse_xyz`)
- "I'm the top-left corner of my local patch" (from `grid_uv`)
- "This whole shape is a sedan" (from `global_feat`)

From this, the MLP can predict exactly where this point should go on the surface.

---

### 5. Folding MLP — Predicting offsets

```python
fold_input = fold_input.transpose(1, 2)    # (B, 1029, 16384)
offset = self.fold_mlp(fold_input)         # (B, 3, 16384)
offset = offset.transpose(1, 2)            # (B, 16384, 3)
```

The transpose is needed because `Conv1d` expects `(B, channels, length)`.

The MLP:

```python
self.fold_mlp = nn.Sequential(
    nn.Conv1d(1029, 512, 1),    # 1029 → 512
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Conv1d(512, 512, 1),     # 512 → 512
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Conv1d(512, 3, 1),       # 512 → 3
)
```

Again, `Conv1d` with kernel size 1 = shared MLP applied to every point independently.

```
Input:  1029-d  →  512  →  512  →  3-d offset
```

**Output:** `(B, 16384, 3)` — a 3D offset for each fine point.

**Why `Conv1d` instead of `Linear`?** They're mathematically identical with kernel size 1, but `Conv1d` is more efficient here because we process all 16,384 points in one batched operation without reshaping.

---

### 6. Add offset to coarse position

```python
fine = coarse_tiled + offset
```

`(B, 16384, 3)` + `(B, 16384, 3)` → `(B, 16384, 3)`

Each fine point = its anchor coarse point + learned offset.

**Why residual (additive) instead of direct prediction?** The network only needs to learn small displacements from already-reasonable positions. This is much easier than predicting absolute coordinates from scratch. The coarse stage handles global positioning; the fine stage only handles local surface detail.

```
Before training:                    After training:

  Coarse point at (1.2, 0.5, -0.3)    Same coarse point
        |                                    |
   16 grid points, all                  16 points displaced to
   clustered at the                     follow the curved
   same location                        surface of the car hood
        |                                    |
        v                                    v
     . . . .                              .   .  .
     . . . .                           .    .      .
     . . . .                          .  .    .  .
     . . . .                            .  .   .
   (flat, useless)                  (learned surface patch)
```

---

### 7. Return both outputs

```python
return coarse, fine
```

Both are returned because the loss function supervises both stages:
- Coarse loss ensures the skeleton converges to the right shape
- Fine loss ensures surface detail is accurate

Without coarse supervision, the seed points could drift to bad positions, and the folding stage would have to compensate with huge offsets — leading to unstable training.

---

## Why grid folding works

The key insight is that any smooth surface can be locally approximated by deforming a flat 2D patch. This is the same principle as UV unwrapping in computer graphics — every surface patch is a warped version of a flat square.

The folding MLP learns this warping function:

```
f: (anchor_xyz, grid_uv, global_context) → 3D offset
```

Given the same grid UV but different anchor points, the MLP produces different offsets — a flat patch on the car roof vs. a curved patch on the wheel arch.

Given the same anchor point but different grid UVs, the MLP spreads the 16 points across the local surface — covering a small neighborhood rather than collapsing to one point.

---

## Tensor shape summary

```
feat                (B, 1024)          global feature from encoder
   |
   v
coarse_fc           (B, 3072)          raw FC output
   |
   v reshape
coarse              (B, 1024, 3)       seed points

coarse_tiled        (B, 16384, 3)      each seed repeated 16x
grid_tiled          (B, 16384, 2)      4x4 grid repeated 1024x
feat_tiled          (B, 16384, 1024)   global feat broadcast

fold_input          (B, 16384, 1029)   concatenated
   |
   v transpose → (B, 1029, 16384)
fold_mlp            (B, 3, 16384)      predicted offsets
   |
   v transpose
offset              (B, 16384, 3)

fine = coarse_tiled + offset
                    (B, 16384, 3)      final surface points
```
