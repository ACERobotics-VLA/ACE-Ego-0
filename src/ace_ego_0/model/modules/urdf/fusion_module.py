"""Gated fusion modules for URDF embodiment tokens."""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class GatedUrdfTokenFusion(nn.Module):
    """Fuse body and chain embeddings into action-expert conditioning tokens."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.body_seed = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.chain_seed = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.body_gate = nn.Linear(2 * token_dim, token_dim)
        self.chain_gate = nn.Linear(2 * token_dim, token_dim)
        self.body_value = nn.Linear(token_dim, token_dim)
        self.chain_value = nn.Linear(token_dim, token_dim)
        self.body_norm = nn.LayerNorm(token_dim)
        self.chain_norm = nn.LayerNorm(token_dim)

    def _fuse_token(
        self,
        *,
        seed: torch.Tensor,
        urdf_embedding: torch.Tensor,
        gate_projection: nn.Linear,
        value_projection: nn.Linear,
        output_norm: nn.LayerNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse one URDF embedding into one learned seed token."""
        expanded_seed = seed.expand(urdf_embedding.shape[0], -1)
        gate = torch.sigmoid(gate_projection(torch.cat((expanded_seed, urdf_embedding), dim=-1)))
        fused_token = output_norm(expanded_seed + gate * value_projection(urdf_embedding))
        return fused_token.unsqueeze(1), gate

    def forward(
        self, body_embeddings: torch.Tensor, chain_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build one body token and one chain token for each batch element."""
        body_tokens, body_gate = self._fuse_token(
            seed=self.body_seed,
            urdf_embedding=body_embeddings,
            gate_projection=self.body_gate,
            value_projection=self.body_value,
            output_norm=self.body_norm,
        )
        chain_tokens, chain_gate = self._fuse_token(
            seed=self.chain_seed,
            urdf_embedding=chain_embeddings,
            gate_projection=self.chain_gate,
            value_projection=self.chain_value,
            output_norm=self.chain_norm,
        )
        diagnostics = {
            "body_gate_mean": body_gate.mean().detach(),
            "body_gate_std": body_gate.std(unbiased=False).detach(),
            "chain_gate_mean": chain_gate.mean().detach(),
            "chain_gate_std": chain_gate.std(unbiased=False).detach(),
        }
        return torch.cat((body_tokens, chain_tokens), dim=1), diagnostics
