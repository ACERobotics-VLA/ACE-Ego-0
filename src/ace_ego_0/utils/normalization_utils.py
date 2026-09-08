"""
Shared normalization utilities used by training and inference.

This module centralizes normalization / denormalization logic so data pipeline
and inference post-processing stay mathematically consistent.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

EPS = 1e-6


def _to_tensor(value, dtype, device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(value, dtype=dtype, device=device)


def _get_stats_tensor(statistics: Dict, primary_key: str, fallback_key: Optional[str], dtype, device) -> torch.Tensor:
    if primary_key in statistics:
        return _to_tensor(statistics[primary_key], dtype=dtype, device=device)
    if fallback_key is not None and fallback_key in statistics:
        return _to_tensor(statistics[fallback_key], dtype=dtype, device=device)
    raise KeyError(f"Missing statistics keys: {primary_key} (fallback: {fallback_key})")


def normalize_tensor(x: torch.Tensor, mode: str, statistics: Dict) -> torch.Tensor:
    if mode == "identity":
        return x

    if mode == "q99":
        q01 = _get_stats_tensor(statistics, "q01", "min", x.dtype, x.device)
        q99 = _get_stats_tensor(statistics, "q99", "max", x.dtype, x.device)
        mask = q01 != q99
        normalized = torch.zeros_like(x)
        normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (q99[..., mask] - q01[..., mask] + EPS)
        normalized[..., mask] = 2 * normalized[..., mask] - 1
        normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
        return normalized

    if mode == "mean_std":
        mean = _get_stats_tensor(statistics, "mean", None, x.dtype, x.device)
        std = _get_stats_tensor(statistics, "std", None, x.dtype, x.device)
        mask = std != 0
        normalized = torch.zeros_like(x)
        normalized[..., mask] = (x[..., mask] - mean[..., mask]) / (std[..., mask] + EPS)
        normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
        return normalized

    if mode == "min_max":
        # Keep training semantics: min/max path prefers exact min/max stats.
        min_v = _get_stats_tensor(statistics, "min", "q01", x.dtype, x.device)
        max_v = _get_stats_tensor(statistics, "max", "q99", x.dtype, x.device)
        mask = min_v != max_v
        normalized = torch.zeros_like(x)
        normalized[..., mask] = (x[..., mask] - min_v[..., mask]) / (max_v[..., mask] - min_v[..., mask] + EPS)
        normalized[..., mask] = 2 * normalized[..., mask] - 1
        normalized[..., ~mask] = 0
        return normalized

    if mode == "scale":
        min_v = _get_stats_tensor(statistics, "min", None, x.dtype, x.device)
        max_v = _get_stats_tensor(statistics, "max", None, x.dtype, x.device)
        abs_max = torch.max(torch.abs(min_v), torch.abs(max_v))
        mask = abs_max != 0
        normalized = torch.zeros_like(x)
        normalized[..., mask] = x[..., mask] / abs_max[..., mask]
        normalized[..., ~mask] = 0
        return normalized

    if mode == "binary":
        return (x > 0.5).to(x.dtype)

    raise ValueError(f"Invalid normalization mode: {mode}")


def denormalize_tensor(x: torch.Tensor, mode: str, statistics: Dict) -> torch.Tensor:
    if mode == "identity":
        return x

    if mode == "q99":
        q01 = _get_stats_tensor(statistics, "q01", "min", x.dtype, x.device)
        q99 = _get_stats_tensor(statistics, "q99", "max", x.dtype, x.device)
        return (x + 1) / 2 * (q99 - q01 + EPS) + q01

    if mode == "mean_std":
        mean = _get_stats_tensor(statistics, "mean", None, x.dtype, x.device)
        std = _get_stats_tensor(statistics, "std", None, x.dtype, x.device)
        return x * (std + EPS) + mean

    if mode == "min_max":
        min_v = _get_stats_tensor(statistics, "min", "q01", x.dtype, x.device)
        max_v = _get_stats_tensor(statistics, "max", "q99", x.dtype, x.device)
        return (x + 1) / 2 * (max_v - min_v + EPS) + min_v

    if mode == "binary":
        return (x > 0.5).to(x.dtype)

    if mode == "scale":
        min_v = _get_stats_tensor(statistics, "min", None, x.dtype, x.device)
        max_v = _get_stats_tensor(statistics, "max", None, x.dtype, x.device)
        abs_max = torch.max(torch.abs(min_v), torch.abs(max_v))
        return x * abs_max

    raise ValueError(f"Invalid normalization mode: {mode}")


def _get_stats_numpy(statistics: Dict, primary_key: str, fallback_key: Optional[str]) -> np.ndarray:
    if primary_key in statistics:
        return np.array(statistics[primary_key])
    if fallback_key is not None and fallback_key in statistics:
        return np.array(statistics[fallback_key])
    raise KeyError(f"Missing statistics keys: {primary_key} (fallback: {fallback_key})")


def denormalize_actions_numpy(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, np.ndarray],
    dim_modes: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Denormalize action array using per-dimension modes.
    """
    if mask is None:
        mask = np.ones(normalized_actions.shape[-1], dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)

    actions = normalized_actions.copy()

    minmax_mask = (dim_modes == "min_max") & mask
    if np.any(minmax_mask):
        min_v = _get_stats_numpy(action_norm_stats, "min", "q01")
        max_v = _get_stats_numpy(action_norm_stats, "max", "q99")
        clipped = np.clip(normalized_actions, -1, 1)
        actions = np.where(minmax_mask, 0.5 * (clipped + 1) * (max_v - min_v + EPS) + min_v, actions)

    q99_mask = (dim_modes == "q99") & mask
    if np.any(q99_mask):
        q01 = _get_stats_numpy(action_norm_stats, "q01", "min")
        q99 = _get_stats_numpy(action_norm_stats, "q99", "max")
        actions = np.where(q99_mask, 0.5 * (normalized_actions + 1) * (q99 - q01 + EPS) + q01, actions)

    meanstd_mask = (dim_modes == "mean_std") & mask
    if np.any(meanstd_mask):
        mean = _get_stats_numpy(action_norm_stats, "mean", None)
        std = _get_stats_numpy(action_norm_stats, "std", None)
        std = np.where(std < EPS, EPS, std)
        actions = np.where(meanstd_mask, normalized_actions * (std + EPS) + mean, actions)

    binary_mask = (dim_modes == "binary") & mask
    if np.any(binary_mask):
        actions = np.where(binary_mask, (normalized_actions > 0.5).astype(normalized_actions.dtype), actions)

    return actions
