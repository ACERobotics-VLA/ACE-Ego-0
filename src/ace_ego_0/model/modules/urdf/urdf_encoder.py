"""Lightweight URDF graph encoder for v1 embodiment tokens."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ResidualGraphMessageLayer(nn.Module):
    """One residual message-passing layer over a dense normalized adjacency matrix."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor, adjacency_norm: torch.Tensor) -> torch.Tensor:
        """Run one residual graph update."""
        messages = adjacency_norm @ hidden_states
        updated_states = self.projection(torch.cat((hidden_states, messages), dim=-1))
        return hidden_states + updated_states


class UrdfGraphEncoder(nn.Module):
    """Encode one static URDF graph into body and chain embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}.")

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([ResidualGraphMessageLayer(hidden_dim) for _ in range(num_layers)])
        self.body_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.chain_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _masked_mean(self, hidden_states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Mean-pool hidden states under a boolean node mask."""
        if mask is None:
            return hidden_states.mean(dim=0)

        mask = mask.to(device=hidden_states.device, dtype=torch.bool)
        if not torch.any(mask):
            return hidden_states.new_zeros((hidden_states.shape[-1],))

        return hidden_states[mask].mean(dim=0)

    def forward(self, graph_tensors: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one cached URDF graph payload into body and chain tokens."""
        node_features = graph_tensors["node_features"]
        adjacency_norm = graph_tensors["adjacency_norm"]
        left_chain_mask = graph_tensors["left_chain_mask"]
        right_chain_mask = graph_tensors["right_chain_mask"]

        hidden_states = self.input_projection(node_features)
        for layer in self.layers:
            hidden_states = layer(hidden_states, adjacency_norm)

        body_embedding = self.body_projection(self._masked_mean(hidden_states, mask=None))
        left_chain_embedding = self._masked_mean(hidden_states, left_chain_mask)
        right_chain_embedding = self._masked_mean(hidden_states, right_chain_mask)
        chain_embedding = self.chain_projection(torch.cat((left_chain_embedding, right_chain_embedding), dim=-1))
        return body_embedding, chain_embedding
