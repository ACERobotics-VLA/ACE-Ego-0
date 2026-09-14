"""Strict checkpoint contracts for ACE-Ego-0 SFT initialization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

HUMAN_SURROGATE_KEY_PREFIX = "action_model.learnable_urdf_embeddings.human_video"
EXPECTED_PRETRAIN_HUMAN_SURROGATE_KEYS = frozenset(
    {
        HUMAN_SURROGATE_KEY_PREFIX,
        f"{HUMAN_SURROGATE_KEY_PREFIX}__0402",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__0403",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__0411_pnppotatotomicrowaveclose",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__0413_pnpcuptodrawerclose",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__ego4d_cooking",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__ego4d_other",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__egodex",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__egoexo4d",
        f"{HUMAN_SURROGATE_KEY_PREFIX}__epic",
    }
)


def _checkpoint_config_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint, got: {checkpoint_path}")
    if len(checkpoint_path.parents) < 2:
        raise ValueError(f"Checkpoint must be stored below a bundle checkpoints directory: {checkpoint_path}")
    return checkpoint_path.parents[1] / "config.yaml"


def read_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the config stored beside one flat model checkpoint."""
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    config_path = _checkpoint_config_path(resolved_checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config must contain a YAML mapping: {config_path}")
    return config


def _config_uses_delta_actions(config: Mapping[str, Any]) -> bool:
    datasets = config.get("datasets", {})
    vla_data = datasets.get("vla_data", {}) if isinstance(datasets, Mapping) else {}
    if not isinstance(vla_data, Mapping):
        raise ValueError("Checkpoint config `datasets.vla_data` must be a mapping.")
    return bool(vla_data.get("delta_action", False))


def validate_checkpoint_action_mode(
    checkpoint_path: str | Path,
    *,
    expected_action_mode: str,
) -> None:
    """Reject absolute/delta cross-initialization before loading any tensors."""
    normalized_mode = str(expected_action_mode).strip().lower()
    if normalized_mode not in {"absolute", "delta"}:
        raise ValueError(f"Expected action mode must be 'absolute' or 'delta', got {expected_action_mode!r}.")

    checkpoint_mode = "delta" if _config_uses_delta_actions(read_checkpoint_config(checkpoint_path)) else "absolute"
    if checkpoint_mode != normalized_mode:
        raise ValueError(
            f"Cannot initialize {normalized_mode} SFT from a {checkpoint_mode} checkpoint: {checkpoint_path}"
        )


def _load_flat_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(state_dict, dict) or not all(isinstance(key, str) for key in state_dict):
        raise TypeError(f"Expected a flat state_dict in checkpoint: {checkpoint_path}")
    return state_dict


def _validate_sft_initialization_keys(
    model: nn.Module,
    checkpoint_state_dict: Mapping[str, torch.Tensor],
) -> set[str]:
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict)
    checkpoint_keys = set(checkpoint_state_dict)
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    allowed_unexpected_keys = unexpected_keys & EXPECTED_PRETRAIN_HUMAN_SURROGATE_KEYS
    disallowed_unexpected_keys = unexpected_keys - EXPECTED_PRETRAIN_HUMAN_SURROGATE_KEYS
    if missing_keys or disallowed_unexpected_keys:
        raise RuntimeError(
            "SFT initialization checkpoint is incompatible: "
            f"missing={sorted(missing_keys)}, unexpected={sorted(disallowed_unexpected_keys)}"
        )
    if allowed_unexpected_keys != EXPECTED_PRETRAIN_HUMAN_SURROGATE_KEYS:
        raise RuntimeError(
            "SFT initialization human-surrogate key set does not match the audited Common23 pretrain contract: "
            f"missing={sorted(EXPECTED_PRETRAIN_HUMAN_SURROGATE_KEYS - allowed_unexpected_keys)}, "
            f"unexpected={sorted(allowed_unexpected_keys - EXPECTED_PRETRAIN_HUMAN_SURROGATE_KEYS)}"
        )

    shape_mismatches = {
        key: (tuple(model_state_dict[key].shape), tuple(checkpoint_state_dict[key].shape))
        for key in model_keys & checkpoint_keys
        if model_state_dict[key].shape != checkpoint_state_dict[key].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"SFT initialization has tensor shape mismatches: {shape_mismatches}")
    return allowed_unexpected_keys


def load_sft_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_action_mode: str,
) -> set[str]:
    """Load a Common23 pretrain checkpoint with one exact, audited exception.

    Production RoboCasa/ARX SFT intentionally drops the ten human-video
    surrogate embeddings from the mixed-data pretrain model. Every robot-policy
    parameter must otherwise be present and shape-compatible.
    """
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    validate_checkpoint_action_mode(
        resolved_checkpoint,
        expected_action_mode=expected_action_mode,
    )
    checkpoint_state_dict = _load_flat_state_dict(resolved_checkpoint)
    allowed_unexpected_keys = _validate_sft_initialization_keys(model, checkpoint_state_dict)

    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != allowed_unexpected_keys:
        raise RuntimeError(
            "Checkpoint compatibility changed while loading: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return allowed_unexpected_keys
