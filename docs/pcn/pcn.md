# PCN — Point Completion Network

Reference: Yuan et al., *PCN: Point Completion Network*, 2018

**Source:** `src/pcn.py`

PCN takes a partial point cloud and predicts the complete shape. It has three parts: an encoder that compresses the input into a feature vector, a decoder that generates the completed point cloud in two stages, and a Chamfer Distance loss that measures prediction quality.

```
Partial (B, N, 3)
   |
   v
[Encoder] — stacked PointNet
   |
   v
1024-d feature vector (B, 1024)
   |
   v
[Decoder] — coarse-to-fine with grid folding
   |
   v
Coarse (B, 1024, 3)  +  Fine (B, 16384, 3)
   |
   v
[Loss] — Chamfer Distance (coarse + fine)
```

---

## Encoder

See [pcn_encoder.md](pcn_encoder.md) for a detailed breakdown.

The encoder takes a partial point cloud and compresses it into a single 1024-d feature vector that represents the whole shape.

**Stage 1:** Each point's `(x, y, z)` is passed through shared MLPs (implemented as `Conv1d` with kernel size 1 — this applies the same linear layer to every point independently). After two layers (`3` → `128` → `256`), each point has a 256-d local feature.

**Local-global mixing:** Max-pool across all points to get one 256-d global vector `g1`. Then concatenate `g1` back onto every point's local feature. Now each point has 512-d: its own local geometry + awareness of the full shape. This is the "stacked PointNet" idea — it's more effective than a single pass because each point knows what the rest of the cloud looks like.

**Stage 2:** Another round of shared MLPs (`512` → `512` → `1024`) on the enriched per-point features, followed by a final max-pool → one 1024-d vector that encodes the entire partial shape.

```
Input (B, N, 3)
   |
   | transpose → (B, 3, N)
   v
Stage 1: Conv1d 3→128→256          → per-point local features (B, 256, N)
   |
   v
Max-pool → g1 (B, 256, 1)          → coarse global summary
   |
   | broadcast g1 to all N points
   v
Concat [local | global]            → (B, 512, N)
   |
   v
Stage 2: Conv1d 512→512→1024       → enriched features (B, 1024, N)
   |
   v
Max-pool                            → final embedding (B, 1024)
```

---

## Decoder

See [pcn_decoder.md](pcn_decoder.md) for a detailed breakdown.

Turns the 1024-d feature into a complete point cloud in two stages.

### Coarse stage

Three fully-connected layers (`1024` → `1024` → `1024` → `3072`) predict `1024 x 3` values, reshaped to 1024 seed points. These are the rough skeleton of the completed shape.

`(B, 1024)` → `(B, 3072)` → reshape → `(B, 1024, 3)`

### Fine stage — Grid folding

For each of the 1024 coarse points, attach a 4x4 = 16-point 2D grid. That gives `1024 x 16 = 16,384` points total.

For each of these 16,384 points, concatenate three things:

| Component | Dimensions | Purpose |
|-----------|-----------|---------|
| Coarse point position | 3 | *Where* on the shape |
| Grid UV coordinate | 2 | *Where* on the local patch |
| Global feature | 1024 | Shape context |
| **Total** | **1029** | |

This 1029-d vector goes through an MLP (`1029` → `512` → `512` → `3`) that predicts a 3-d offset. The final fine point = coarse point + offset. The network learns to "unfold" each flat grid patch into the local surface geometry.

```
For each coarse point:
    Flat 4x4 grid patch
         |
         | MLP predicts offset per grid point
         v
    Curved surface patch that follows local geometry

Before training:              After training:
    . . . .                      .   .  .
    . . . .        →          .    .      .
    . . . .                  .  .    .  .
    . . . .                    .  .   .
  (flat grid)              (learned surface)
```

**Output:** Coarse `(B, 1024, 3)` + Fine `(B, 16384, 3)`

---

## Loss — Chamfer Distance

See [pcn_loss.md](pcn_loss.md) for a detailed breakdown.

Chamfer Distance measures how close two point clouds are by finding nearest neighbors in both directions:

```
CD = mean(min distance pred→gt) + mean(min distance gt→pred)
```

Both directions are needed — pred→gt alone allows collapse to a single point, gt→pred alone allows scattering noise everywhere.

### Combined loss

```
total = cd_coarse + 0.5 * cd_fine
```

| Term | What it compares | Point counts | Purpose |
|------|-----------------|--------------|---------|
| `cd_coarse` | Coarse output vs subsampled GT | 1024 vs 1024 | Teaches skeleton shape |
| `cd_fine` | Fine output vs GT (both subsampled) | 4096 vs 4096 | Teaches surface detail |
| `alpha` | 0.5 | — | Downweights fine loss |

The fine output and GT are subsampled from 16,384 to 4,096 points during training to fit in 8 GB VRAM. Peak memory: ~3 GB at batch size 8.

**TODO:** Add EMD (Earth Mover's Distance) for the coarse output to improve point density uniformity.
