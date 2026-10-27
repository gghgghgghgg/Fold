from __future__ import annotations

import torch
from torch import nn


def _masked_pool(
    features: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(features.dtype)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    mean = (features * mask_f).sum(dim=1) / count

    masked = features.masked_fill(~mask.unsqueeze(-1), torch.finfo(features.dtype).min)
    maximum = masked.max(dim=1).values
    has_value = mask.any(dim=1, keepdim=True)
    maximum = torch.where(has_value, maximum, torch.zeros_like(maximum))

    weighted_mask = weights.unsqueeze(-1) * mask_f
    weighted = (features * weighted_mask).sum(dim=1) / weighted_mask.sum(dim=1).clamp_min(1e-6)
    return torch.cat([mean, maximum, weighted], dim=-1)


class PointNetFieldEncoder(nn.Module):
    """Part-aware set encoder for a global per-anchor force field."""

    def __init__(
        self,
        num_parts: int = 2,
        part_embedding_dim: int = 8,
        anchor_feature_dim: int = 128,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.num_parts = int(num_parts)
        self.part_embedding = nn.Embedding(self.num_parts, part_embedding_dim)
        input_dim = 3 + 3 + 3 + 1 + 1 + part_embedding_dim
        self.anchor_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, anchor_feature_dim),
            nn.LayerNorm(anchor_feature_dim),
            nn.SiLU(),
            nn.Linear(anchor_feature_dim, anchor_feature_dim),
            nn.SiLU(),
        )
        pooled_dim = 3 * anchor_feature_dim
        self.field_encoder = nn.Sequential(
            nn.Linear((self.num_parts + 1) * pooled_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        positions: torch.Tensor,
        normals: torch.Tensor,
        forces: torch.Tensor,
        part_ids: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        magnitude = torch.linalg.vector_norm(forces, dim=-1)
        if active_mask is None:
            active_mask = magnitude > 1e-6
        part_ids = part_ids.long()
        if part_ids.ndim == 1:
            part_ids = part_ids.unsqueeze(0).expand(positions.shape[0], -1)
        part_embedding = self.part_embedding(part_ids)
        anchor_input = torch.cat(
            [
                positions,
                normals,
                forces,
                torch.log1p(magnitude).unsqueeze(-1),
                active_mask.to(forces.dtype).unsqueeze(-1),
                part_embedding,
            ],
            dim=-1,
        )
        anchor_features = self.anchor_encoder(anchor_input)
        all_mask = torch.ones_like(active_mask, dtype=torch.bool)
        pooled = [_masked_pool(anchor_features, all_mask, magnitude)]
        for part in range(self.num_parts):
            pooled.append(_masked_pool(anchor_features, part_ids == part, magnitude))
        return self.field_encoder(torch.cat(pooled, dim=-1))


class FieldAutoencoder(nn.Module):
    """Field encoder with local-force and generalized-wrench auxiliary heads."""

    def __init__(self, num_parts: int = 2, latent_dim: int = 128):
        super().__init__()
        self.encoder = PointNetFieldEncoder(num_parts=num_parts, latent_dim=latent_dim)
        self.decoder_part_embedding = nn.Embedding(num_parts, 8)
        self.force_decoder = nn.Sequential(
            nn.Linear(latent_dim + 3 + 3 + 8, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        self.wrench_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 7),
        )

    def forward(
        self,
        positions: torch.Tensor,
        normals: torch.Tensor,
        forces: torch.Tensor,
        part_ids: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(positions, normals, forces, part_ids, active_mask)
        if part_ids.ndim == 1:
            part_ids = part_ids.unsqueeze(0).expand(positions.shape[0], -1)
        latent_per_anchor = latent.unsqueeze(1).expand(-1, positions.shape[1], -1)
        decoder_input = torch.cat(
            [latent_per_anchor, positions, normals, self.decoder_part_embedding(part_ids.long())],
            dim=-1,
        )
        force_reconstruction = self.force_decoder(decoder_input)
        wrench_reconstruction = self.wrench_head(latent)
        return force_reconstruction, wrench_reconstruction, latent
