"""Checkpoint configuration and normalization-statistics loading helpers."""

import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ace_ego_0.inference_utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def dict_to_namespace(config: dict[str, Any]):
    """Convert a YAML mapping into an OmegaConf object used by the model builders."""
    return OmegaConf.create(config)


def _resolve_manifest_merged_stats_path(norm_manifest_path: Any) -> Path:
    """Resolve the merged statistics path declared by a normalization manifest."""
    manifest_path = Path(str(norm_manifest_path)).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    merged_stats_path = artifacts.get("merged_stats_path") if isinstance(artifacts, dict) else None
    if not merged_stats_path:
        raise AssertionError(f"Norm manifest `{manifest_path}` does not define artifacts.merged_stats_path")
    merged_path = Path(str(merged_stats_path)).expanduser()
    return merged_path if merged_path.is_absolute() else manifest_path.parent / merged_path


def _normalize_merged_norm_paths(merged_norm_path: Any) -> list[Path]:
    """Normalize one or more configured merged-statistics paths."""
    if merged_norm_path in (None, "", False, []):
        return []
    if isinstance(merged_norm_path, (str, Path)):
        return [Path(str(merged_norm_path)).expanduser()]
    if isinstance(merged_norm_path, (list, tuple)):
        return [Path(str(path)).expanduser() for path in merged_norm_path]
    raise AssertionError(f"Unsupported merged_norm_path value: {merged_norm_path!r}")


def _resolve_configured_norm_statistics_path(global_config: dict[str, Any]) -> Path | None:
    """Resolve a statistics artifact explicitly referenced by checkpoint config."""
    data_config = global_config.get("datasets", {}).get("vla_data", {})
    if not isinstance(data_config, dict):
        return None

    norm_manifest_path = data_config.get("norm_manifest_path")
    if norm_manifest_path not in (None, "", False, []):
        return _resolve_manifest_merged_stats_path(norm_manifest_path)

    merged_norm_paths = _normalize_merged_norm_paths(data_config.get("merged_norm_path"))
    if not merged_norm_paths:
        return None
    if len(merged_norm_paths) != 1:
        raise AssertionError(
            "Checkpoint config uses multiple merged_norm_path files; eval-side norm loading expects one file."
        )
    return merged_norm_paths[0]


def _resolve_dataset_statistics_path(run_dir: Path, global_config: dict[str, Any] | None = None) -> Path:
    """Resolve normalization statistics from a public or legacy checkpoint layout."""
    if global_config is not None:
        configured_path = _resolve_configured_norm_statistics_path(global_config)
        if configured_path is not None:
            if not configured_path.exists():
                raise AssertionError(f"Configured norm statistics file does not exist: {configured_path}")
            return configured_path

    candidates = (
        run_dir / "dataset_statistics.json",
        run_dir / "checkpoints" / "runtime_merged_dataset_statistics.json",
        run_dir / "runtime_merged_dataset_statistics.json",
        run_dir / "checkpoints" / "dataset_statistics.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise AssertionError(f"Missing dataset statistics json for `{run_dir}`; checked: {expected}")


def read_mode_config(pretrained_checkpoint: str | Path, allow_missing_norm_stats: bool = False):
    """Load one checkpoint's YAML configuration and normalization statistics."""
    if not os.path.isfile(pretrained_checkpoint):
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")

    checkpoint_path = Path(pretrained_checkpoint)
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint path, got `{checkpoint_path}`.")
    run_dir = checkpoint_path.parents[1]
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing `config.yaml` for `{run_dir}`")

    global_config = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    if not isinstance(global_config, dict):
        raise ValueError(f"Expected mapping in checkpoint config `{config_path}`.")

    try:
        statistics_path = _resolve_dataset_statistics_path(run_dir, global_config)
    except AssertionError:
        if not allow_missing_norm_stats:
            raise
        return global_config, {}

    with statistics_path.open("r", encoding="utf-8") as handle:
        return global_config, json.load(handle)
