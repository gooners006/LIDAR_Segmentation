"""PCN (Point Completion Network) — Yuan et al., 2018.

Encoder: Stacked PointNet (local feat → global pool → concat → second stage) → 1024-d
Decoder: FC → coarse (1024 pts), 4×4 grid folding → fine (16384 pts)
Loss: Chamfer Distance (chunked for memory safety on 8GB VRAM)
"""

import torch
import torch.nn as nn


class PCNEncoder(nn.Module):
    """Stacked PointNet encoder.

    Stage 1: per-point MLP → max pool → global feature g1
    Concat g1 back to per-point features (local + global mixing)
    Stage 2: per-point MLP → max pool → global feature g2 (final)
    """

    def __init__(self, feat_dim: int = 1024):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, feat_dim, 1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) → (B, feat_dim)"""
        x = xyz.transpose(1, 2)  # (B, N, 3) => (B, 3, N)
        local_feat = self.stage1(x)  # (B, 3, N) => (B, 256, N)
        g1 = local_feat.max(dim=2, keepdim=True).values  # (B, 256, N) => (B, 256, 1)
        g1_expanded = g1.expand(-1, -1, x.size(2))  # (B, 256, 1) => (B, 256, N)
        combined = torch.cat([local_feat, g1_expanded], dim=1)  # (B, 512, N)
        feat = self.stage2(combined)  # (B, 512, N) => (B, feat_dim, N)
        return feat.max(dim=2).values  # (B, feat_dim)


class PCNDecoder(nn.Module):
    """Coarse-to-fine decoder with 2D grid folding."""

    def __init__(self, feat_dim: int = 1024, num_coarse: int = 1024, grid_size: int = 4):
        super().__init__()
        self.num_coarse = num_coarse
        self.grid_size = grid_size
        self.num_fine = num_coarse * grid_size ** 2

        self.coarse_fc = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_coarse * 3),
        )

        # Folding MLP: takes [coarse_point(3) + grid_uv(2) + global_feat] → offset(3)
        self.fold_mlp = nn.Sequential(
            nn.Conv1d(feat_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        # Pre-compute 2D folding grid
        lin = torch.linspace(-0.05, 0.05, grid_size)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("grid", torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, feat_dim) → coarse (B, num_coarse, 3), fine (B, num_fine, 3)"""
        B = feat.size(0)
        G = self.grid_size ** 2

        # Coarse: predict seed points
        coarse = self.coarse_fc(feat).reshape(B, self.num_coarse, 3)

        # Fine: fold a 2D grid around each coarse point
        # Expand coarse points: (B, num_coarse, G, 3)
        coarse_tiled = coarse.unsqueeze(2).expand(-1, -1, G, -1).reshape(B, self.num_fine, 3)
        # Expand grid: (B, num_fine, 2)
        grid_tiled = self.grid.unsqueeze(0).unsqueeze(0).expand(B, self.num_coarse, -1, -1).reshape(B, self.num_fine, 2)
        # Expand global feature: (B, num_fine, feat_dim)
        feat_tiled = feat.unsqueeze(1).expand(-1, self.num_fine, -1)

        # Concatenate and predict offsets
        fold_input = torch.cat([coarse_tiled, grid_tiled, feat_tiled], dim=2)  # (B, num_fine, feat+5)
        fold_input = fold_input.transpose(1, 2)  # (B, feat+5, num_fine)
        offset = self.fold_mlp(fold_input).transpose(1, 2)  # (B, num_fine, 3)

        fine = coarse_tiled + offset

        return coarse, fine


class PCN(nn.Module):
    """Point Completion Network.

    Input:  (B, N, 3) partial point cloud
    Output: coarse (B, 1024, 3), fine (B, 16384, 3)
    """

    def __init__(self, feat_dim: int = 1024, num_coarse: int = 1024, grid_size: int = 4):
        super().__init__()
        self.encoder = PCNEncoder(feat_dim)
        self.decoder = PCNDecoder(feat_dim, num_coarse, grid_size)

    def forward(self, partial: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(partial)
        return self.decoder(feat)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def chamfer_distance_chunked(
    pred: torch.Tensor,
    gt: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Memory-safe bidirectional Chamfer Distance (L2, not squared).

    Processes one sample at a time, chunking the inner dimension to control peak memory.
    With chunk_size=2048 and M=N=16384: peak ~16384×2048×4 = 128MB per direction
    (torch.cdist outputs scalar distances, not 3D vectors; backward may allocate more).

    Args:
        pred: (B, M, 3)
        gt:   (B, N, 3)
    Returns:
        Scalar mean CD over batch.
    """
    B = pred.shape[0]
    cd_sum = torch.tensor(0.0, device=pred.device)

    for b in range(B):
        p = pred[b]  # (M, 3)
        g = gt[b]    # (N, 3)
        M, N = p.shape[0], g.shape[0]

        # pred → gt
        min_p2g = torch.full((M,), float("inf"), device=p.device)
        for i in range(0, N, chunk_size):
            g_chunk = g[i:i + chunk_size]  # (chunk, 3)
            dist = torch.cdist(p.unsqueeze(0), g_chunk.unsqueeze(0)).squeeze(0)  # (M, chunk)
            min_p2g = torch.min(min_p2g, dist.min(dim=1).values)

        # gt → pred
        min_g2p = torch.full((N,), float("inf"), device=g.device)
        for i in range(0, M, chunk_size):
            p_chunk = p[i:i + chunk_size]  # (chunk, 3)
            dist = torch.cdist(g.unsqueeze(0), p_chunk.unsqueeze(0)).squeeze(0)  # (N, chunk)
            min_g2p = torch.min(min_g2p, dist.min(dim=1).values)

        cd_sum = cd_sum + min_p2g.mean() + min_g2p.mean()

    return cd_sum / B


def pcn_loss(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    gt: torch.Tensor,
    gt_coarse: torch.Tensor,
    alpha: float = 0.5,
    fine_gt_samples: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined coarse + fine Chamfer loss.

    Subsamples GT to `fine_gt_samples` for the fine loss to fit in 8GB VRAM.

    Args:
        coarse:    (B, num_coarse, 3)
        fine:      (B, num_fine, 3)
        gt:        (B, N_gt, 3) full ground truth
        gt_coarse: (B, num_coarse, 3) subsampled ground truth for coarse loss
        alpha:     weight for fine loss
        fine_gt_samples: subsample GT to this many points for fine CD

    Returns:
        (total_loss, cd_coarse, cd_fine)
    """
    cd_coarse = chamfer_distance_chunked(coarse, gt_coarse, chunk_size=1024)

    # Subsample GT for fine loss to control memory
    B, N, _ = gt.shape
    if N > fine_gt_samples:
        idx = torch.randint(0, N, (B, fine_gt_samples), device=gt.device)
        gt_fine = torch.gather(gt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    else:
        gt_fine = gt

    # Also subsample fine predictions symmetrically
    M = fine.shape[1]
    if M > fine_gt_samples:
        idx_f = torch.randint(0, M, (B, fine_gt_samples), device=fine.device)
        fine_sub = torch.gather(fine, 1, idx_f.unsqueeze(-1).expand(-1, -1, 3))
    else:
        fine_sub = fine

    cd_fine = chamfer_distance_chunked(fine_sub, gt_fine, chunk_size=2048)
    total = cd_coarse + alpha * cd_fine
    return total, cd_coarse, cd_fine
