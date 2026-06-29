"""PoinTr — Diverse Point Cloud Completion with Geometry-Aware Transformers.

Faithful (scaled-down) re-implementation of PoinTr (Yu et al., ICCV 2021) in
self-contained PyTorch — no custom CUDA ops. The model reformulates completion
as set-to-set translation:

    1. Point Proxy: FPS to N centers + a lightweight DGCNN feature extractor
       turn the partial cloud into a sequence of N proxy tokens (feature +
       position embedding of the center).
    2. Geometry-aware Transformer encoder: self-attention captures semantic
       (long-range) relations; a kNN/DGCNN branch injects explicit local
       geometry. Following the paper's ablation (model E), the geometry-aware
       branch is used on the *first* encoder/decoder block only.
    3. Dynamic Query Generator: a global summary of the encoder memory predicts
       M coarse missing-center coordinates; queries = MLP(concat(global, coord)).
    4. Multi-scale generation: a per-proxy FoldingNet recovers a local patch
       around each predicted center. We predict only the missing part and
       concatenate the input partial to form the complete cloud.

Loss = CD(coarse, GT) + CD(fine, GT), both supervised by the high-res GT
(Sec. 3.6). We reuse ``pcn.chamfer_distance_chunked`` for memory-safe CD.

Adaptations for this thesis pipeline (cars-only, 256-pt partial input, 4096-pt
GT, 8 GB VRAM): trans_dim=256, 4 heads, N=64 proxies, enc/dec depth 4, M=256
queries, fold grid 4x4 → 256*16=4096 missing points. The public contract matches
PCN: ``PoinTr(partial) -> (coarse, fine)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Point-cloud helpers (pure PyTorch, no custom ops)
# ---------------------------------------------------------------------------

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance.

    Args:
        src: (B, M, 3) source points.
        dst: (B, N, 3) destination points.

    Returns:
        (B, M, N) matrix of squared distances ``|src_i - dst_j|^2``.
    """
    return torch.cdist(src, dst) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C) feature/point tensor.
        idx: (B, S) or (B, S, K) integer index into dim 1.

    Returns:
        (B, S, C) or (B, S, K, C) gathered tensor.
    """
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_idx = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest point sampling (FPS).

    Args:
        xyz: (B, N, 3) point coordinates.
        npoint: number of centers to sample.

    Returns:
        (B, npoint) indices of the sampled centers.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    if N <= npoint:
        # Not enough points to sample uniquely; pad by repeating indices.
        idx = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        if N < npoint:
            pad = idx[:, :1].repeat(1, npoint - N)
            idx = torch.cat([idx, pad], dim=1)
        return idx

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)  # deterministic start
    batch_idx = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.min(distance, dist)
        farthest = distance.max(dim=1).indices
    return centroids


def knn_idx(query: torch.Tensor, base: torch.Tensor, k: int) -> torch.Tensor:
    """k nearest neighbours of ``query`` within ``base`` (by coordinates).

    Args:
        query: (B, M, 3) query coordinates.
        base: (B, N, 3) reference coordinates.
        k: number of neighbours.

    Returns:
        (B, M, k) indices into ``base``.
    """
    dist = square_distance(query, base)  # (B, M, N)
    return dist.topk(k, dim=-1, largest=False).indices


# ---------------------------------------------------------------------------
# DGCNN point-proxy feature extractor
# ---------------------------------------------------------------------------

class EdgeConv(nn.Module):
    """DGCNN EdgeConv with optional FPS downsampling.

    For each output centre, gathers k neighbours from the input set and
    aggregates edge features ``[x_i, x_j - x_i]`` via a shared MLP + max-pool.

    Args:
        in_dim: input feature dim.
        out_dim: output feature dim.
        k: neighbourhood size.
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(
        self,
        feat: torch.Tensor,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply EdgeConv, optionally resampling to ``new_xyz``.

        Args:
            feat: (B, N, C) per-point features.
            xyz: (B, N, 3) coordinates of ``feat``.
            new_xyz: (B, S, 3) output centres. If None, ``S = N`` (no downsample).

        Returns:
            (B, S, out_dim) features at the output centres.
        """
        if new_xyz is None:
            new_xyz = xyz
            center_feat = feat
        else:
            # gather the input feature at each output centre (nearest = itself
            # since new_xyz is an FPS subset of xyz)
            nn_self = knn_idx(new_xyz, xyz, 1).squeeze(-1)  # (B, S)
            center_feat = index_points(feat, nn_self)        # (B, S, C)

        nbr = knn_idx(new_xyz, xyz, self.k)          # (B, S, k)
        nbr_feat = index_points(feat, nbr)           # (B, S, k, C)
        center = center_feat.unsqueeze(2).expand_as(nbr_feat)
        edge = torch.cat([center, nbr_feat - center], dim=-1)  # (B, S, k, 2C)
        edge = edge.permute(0, 3, 1, 2)                        # (B, 2C, S, k)
        out = self.mlp(edge).max(dim=-1).values               # (B, out_dim, S)
        return out.transpose(1, 2)                            # (B, S, out_dim)


class PointProxy(nn.Module):
    """Convert a partial cloud to a sequence of N point-proxy tokens.

    Hierarchical DGCNN (paper Supp. A, scaled down) with FPS downsampling,
    then proxy feature = DGCNN feature + MLP position-embedding of the centre.

    Args:
        num_proxies: number of output proxies N (FPS centres).
        trans_dim: transformer token dimension.
        k: DGCNN neighbourhood size.
    """

    def __init__(self, num_proxies: int = 64, trans_dim: int = 256, k: int = 16):
        super().__init__()
        self.num_proxies = num_proxies
        self.input_proj = nn.Sequential(nn.Linear(3, 8), nn.LeakyReLU(0.2))
        self.conv1 = EdgeConv(8, 32, k)     # full resolution
        self.conv2 = EdgeConv(32, 64, k)    # → N*2
        self.conv3 = EdgeConv(64, 128, k)   # → N
        self.feat_proj = nn.Linear(128, trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, trans_dim)
        )

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a partial cloud.

        Args:
            xyz: (B, Npts, 3) partial point cloud.

        Returns:
            Tuple of (centers, tokens) where ``centers`` is (B, N, 3) and
            ``tokens`` is (B, N, trans_dim).
        """
        B, Npts, _ = xyz.shape
        N = self.num_proxies

        feat0 = self.input_proj(xyz)                 # (B, Npts, 8)
        feat1 = self.conv1(feat0, xyz)               # (B, Npts, 32)

        idx2 = farthest_point_sample(xyz, min(N * 2, Npts))
        xyz2 = index_points(xyz, idx2)               # (B, N2, 3)
        feat2 = self.conv2(feat1, xyz, xyz2)         # (B, N2, 64)

        idx3 = farthest_point_sample(xyz2, N)
        centers = index_points(xyz2, idx3)           # (B, N, 3)
        feat3 = self.conv3(feat2, xyz2, centers)     # (B, N, 128)

        tokens = self.feat_proj(feat3) + self.pos_embed(centers)
        return centers, tokens


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class GeometryBranch(nn.Module):
    """kNN/DGCNN geometric branch (Sec. 3.3, Fig. 3).

    Captures explicit local geometry by aggregating neighbour features over a
    coordinate-kNN graph (linear + max-pool, DGCNN-style). Plugged in parallel
    to self-attention; outputs are concatenated and projected back.

    Args:
        dim: token dimension.
        k: geometric neighbourhood size.
    """

    def __init__(self, dim: int, k: int = 8):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """Aggregate local geometric features.

        Args:
            x: (B, T, dim) token features.
            coord: (B, T, 3) token coordinates (proxy/query centres).

        Returns:
            (B, T, dim) geometric features.
        """
        nbr = knn_idx(coord, coord, self.k)          # (B, T, k)
        nbr_feat = index_points(x, nbr)              # (B, T, k, dim)
        center = x.unsqueeze(2).expand_as(nbr_feat)
        edge = torch.cat([center, nbr_feat - center], dim=-1)  # (B, T, k, 2dim)
        return self.linear(edge).max(dim=2).values             # (B, T, dim)


class EncoderBlock(nn.Module):
    """Transformer encoder block, optionally geometry-aware.

    Args:
        dim: token dimension.
        num_heads: attention heads.
        geometry: if True, fuse a kNN geometric branch with self-attention.
        k: geometric neighbourhood size.
        mlp_ratio: FFN hidden expansion.
        dropout: attention/FFN dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        geometry: bool = False,
        k: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.geometry = GeometryBranch(dim, k) if geometry else None
        if geometry:
            self.fuse = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """Run one encoder block.

        Args:
            x: (B, N, dim) tokens.
            coord: (B, N, 3) token coordinates (for the geometry branch).

        Returns:
            (B, N, dim) updated tokens.
        """
        h = self.norm1(x)
        sem, _ = self.attn(h, h, h, need_weights=False)
        if self.geometry is not None:
            geo = self.geometry(h, coord)
            sem = self.fuse(torch.cat([sem, geo], dim=-1))
        x = x + sem
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block (self-attn + cross-attn), optionally geometry-aware.

    Args:
        dim: token dimension.
        num_heads: attention heads.
        geometry: if True, fuse a kNN geometric branch with query self-attention.
        k: geometric neighbourhood size.
        mlp_ratio: FFN hidden expansion.
        dropout: attention/FFN dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        geometry: bool = False,
        k: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.geometry = GeometryBranch(dim, k) if geometry else None
        if geometry:
            self.fuse = nn.Linear(dim * 2, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        q_coord: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder block.

        Args:
            q: (B, M, dim) query tokens.
            memory: (B, N, dim) encoder output.
            q_coord: (B, M, 3) query coordinates (for the geometry branch).

        Returns:
            (B, M, dim) updated query tokens.
        """
        h = self.norm1(q)
        sem, _ = self.self_attn(h, h, h, need_weights=False)
        if self.geometry is not None:
            geo = self.geometry(h, q_coord)
            sem = self.fuse(torch.cat([sem, geo], dim=-1))
        q = q + sem
        h = self.norm2(q)
        cross, _ = self.cross_attn(h, memory, memory, need_weights=False)
        q = q + cross
        q = q + self.ffn(self.norm3(q))
        return q


# ---------------------------------------------------------------------------
# Query generator and multi-scale folding decoder
# ---------------------------------------------------------------------------

class QueryGenerator(nn.Module):
    """Dynamic query generator (Sec. 3.4).

    Summarises encoder memory into a global feature, predicts M coarse missing
    centres, then forms queries = MLP(concat(global, coord)).

    Args:
        dim: token dimension.
        num_query: number of dynamic queries M.
    """

    def __init__(self, dim: int, num_query: int = 256):
        super().__init__()
        self.num_query = num_query
        self.summary = nn.Linear(dim, dim * 2)
        self.coord_pred = nn.Linear(dim * 2, num_query * 3)
        self.query_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate coarse centres and query embeddings.

        Args:
            memory: (B, N, dim) encoder output.

        Returns:
            Tuple of (coord, queries): ``coord`` is (B, M, 3) coarse missing
            centres, ``queries`` is (B, M, dim).
        """
        B = memory.shape[0]
        g = self.summary(memory).max(dim=1).values          # (B, 2*dim)
        coord = self.coord_pred(g).reshape(B, self.num_query, 3)
        g_exp = g.unsqueeze(1).expand(-1, self.num_query, -1)
        queries = self.query_mlp(torch.cat([g_exp, coord], dim=-1))
        return coord, queries


class FoldingDecoder(nn.Module):
    """Per-proxy FoldingNet — recovers a local patch around each centre (Sec. 3.5).

    Each predicted proxy feature conditions a 2-stage 2D-grid folding that
    deforms a ``grid_size x grid_size`` patch; the patch is placed at the
    proxy's centre coordinate.

    Args:
        dim: proxy feature dim.
        grid_size: folding grid side; yields ``grid_size**2`` points per proxy.
    """

    def __init__(self, dim: int, grid_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.num_per = grid_size ** 2

        lin = torch.linspace(-0.5, 0.5, grid_size)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("grid", torch.stack([gx, gy], dim=-1).reshape(-1, 2))

        self.fold1 = nn.Sequential(
            nn.Conv1d(dim + 2, dim, 1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim // 2, 1), nn.BatchNorm1d(dim // 2), nn.ReLU(),
            nn.Conv1d(dim // 2, 3, 1),
        )
        self.fold2 = nn.Sequential(
            nn.Conv1d(dim + 3, dim, 1), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim // 2, 1), nn.BatchNorm1d(dim // 2), nn.ReLU(),
            nn.Conv1d(dim // 2, 3, 1),
        )

    def forward(self, proxies: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Fold a local patch around every proxy centre.

        Args:
            proxies: (B, M, dim) predicted proxy features.
            centers: (B, M, 3) proxy centre coordinates.

        Returns:
            (B, M * grid_size**2, 3) recovered (missing) point cloud.
        """
        B, M, dim = proxies.shape
        G = self.num_per

        feat = proxies.reshape(B * M, dim, 1).expand(-1, -1, G)        # (BM, dim, G)
        grid = self.grid.to(proxies.dtype).t().unsqueeze(0).expand(B * M, -1, -1)  # (BM, 2, G)

        p1 = self.fold1(torch.cat([feat, grid], dim=1))               # (BM, 3, G)
        p2 = self.fold2(torch.cat([feat, p1], dim=1))                 # (BM, 3, G)

        patch = p2.transpose(1, 2).reshape(B, M, G, 3)
        out = patch + centers.unsqueeze(2)                            # place at centre
        return out.reshape(B, M * G, 3)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PoinTr(nn.Module):
    """PoinTr point completion model (scaled-down, self-contained).

    Args:
        num_proxies: number of input point proxies N.
        num_query: number of dynamic queries M (coarse missing centres).
        trans_dim: transformer token dimension.
        num_heads: attention heads.
        enc_depth: number of encoder blocks.
        dec_depth: number of decoder blocks.
        grid_size: folding grid side (fine = M * grid_size**2 missing points).
        knn_dgcnn: kNN for the DGCNN proxy feature extractor.
        knn_geo: kNN for the geometry-aware transformer branch.
    """

    def __init__(
        self,
        num_proxies: int = 64,
        num_query: int = 256,
        trans_dim: int = 256,
        num_heads: int = 4,
        enc_depth: int = 4,
        dec_depth: int = 4,
        grid_size: int = 4,
        knn_dgcnn: int = 16,
        knn_geo: int = 8,
    ):
        super().__init__()
        self.num_query = num_query
        self.grid_size = grid_size

        self.proxy = PointProxy(num_proxies, trans_dim, knn_dgcnn)
        # Geometry-aware branch on the first block only (paper model E).
        self.encoder = nn.ModuleList([
            EncoderBlock(trans_dim, num_heads, geometry=(i == 0), k=knn_geo)
            for i in range(enc_depth)
        ])
        self.query_gen = QueryGenerator(trans_dim, num_query)
        self.decoder = nn.ModuleList([
            DecoderBlock(trans_dim, num_heads, geometry=(i == 0), k=knn_geo)
            for i in range(dec_depth)
        ])
        self.fold = FoldingDecoder(trans_dim, grid_size)

    def forward(self, partial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Complete a partial point cloud.

        Args:
            partial: (B, Npts, 3) partial point cloud.

        Returns:
            Tuple of (coarse, fine):
              - ``coarse`` (B, M + N, 3): predicted missing centres
                concatenated with the input proxy centres (paper's C).
              - ``fine`` (B, M*grid**2 + Npts, 3): folded missing points
                concatenated with the input partial (predict-missing-then-concat).
        """
        centers, tokens = self.proxy(partial)        # (B, N, 3), (B, N, dim)

        x = tokens
        for blk in self.encoder:
            x = blk(x, centers)
        memory = x                                   # (B, N, dim)

        coord, queries = self.query_gen(memory)      # (B, M, 3), (B, M, dim)

        q = queries
        for blk in self.decoder:
            q = blk(q, memory, coord)                # (B, M, dim)

        missing = self.fold(q, coord)                # (B, M*G, 3)

        coarse = torch.cat([coord, centers], dim=1)          # (B, M+N, 3)
        fine = torch.cat([missing, partial], dim=1)          # (B, M*G+Npts, 3)
        return coarse, fine


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pointr_loss(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PoinTr completion loss: CD(coarse, GT) + alpha * CD(fine, GT).

    Both predictions are supervised by the high-res GT (Sec. 3.6). Unlike
    ``pcn.pcn_loss`` (whose fine cloud is 16384 points and must be subsampled
    for VRAM), our fine cloud is only ~4352 points, so we compute the Chamfer
    distance **exactly** — no random subsampling. Subsampling the prediction
    with replacement would drop valid points and inflate CD (GT points get
    forced onto farther survivors), so it is avoided entirely.

    Args:
        coarse: (B, M+N, 3) coarse prediction (missing centres + input centres).
        fine: (B, F, 3) fine prediction (missing folds + input partial).
        gt: (B, N_gt, 3) full ground-truth cloud.
        alpha: weight on the fine CD term.

    Returns:
        Tuple of (total, cd_coarse, cd_fine).
    """
    from pcn import chamfer_distance_chunked

    cd_coarse = chamfer_distance_chunked(coarse, gt, chunk_size=1024)
    cd_fine = chamfer_distance_chunked(fine, gt, chunk_size=2048)
    total = cd_coarse + alpha * cd_fine
    return total, cd_coarse, cd_fine
