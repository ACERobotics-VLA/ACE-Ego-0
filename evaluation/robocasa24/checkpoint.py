"""Validation and configuration access for published checkpoint bundles."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PUBLIC_CHECKPOINT_FILENAMES = (
    "ace_ego_0_robocasa24_absolute.pt",
    "ace_ego_0_robocasa24_delta.pt",
)
SUPPORTED_FRAMEWORK_NAMES = {"ACE-Ego-0"}


@dataclass(frozen=True)
class CheckpointContract:
    """Validated paths and action semantics for one evaluation checkpoint."""

    root: Path
    weights_path: Path
    config_path: Path
    statistics_path: Path
    framework_name: str
    action_mode: str
    action_horizon: int
    rotation_repr: str
    delta_base_state: str | None
    delta_action_mode: str | None
    delta_quat_mode: str | None
    rot_norm_mode: str


def _resolve_weights_path(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint.resolve()
    checkpoint_dir = checkpoint / "checkpoints"
    for filename in PUBLIC_CHECKPOINT_FILENAMES:
        candidate = checkpoint_dir / filename
        if candidate.is_file():
            return candidate.resolve()
    expected_names = ", ".join(PUBLIC_CHECKPOINT_FILENAMES)
    raise FileNotFoundError(f"Checkpoint weights do not exist in {checkpoint_dir}; expected one of: {expected_names}.")


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return data


def _require_mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping `{key}` in {path}.")
    return value


def inspect_checkpoint(checkpoint: str | Path) -> CheckpointContract:
    """Validate a checkpoint bundle and return its evaluation contract."""
    weights_path = _resolve_weights_path(Path(checkpoint).expanduser())
    root = weights_path.parents[1]
    config_path = root / "config.yaml"
    statistics_path = root / "checkpoints" / "runtime_merged_dataset_statistics.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config does not exist: {config_path}")
    if not statistics_path.is_file():
        raise FileNotFoundError(f"Checkpoint statistics do not exist: {statistics_path}")

    config = _read_mapping(config_path)
    framework = _require_mapping(config, "framework", config_path)
    framework_name = framework.get("name")
    if framework_name not in SUPPORTED_FRAMEWORK_NAMES:
        raise ValueError(
            f"Unsupported framework name {framework_name!r} in {config_path}; "
            f"expected one of {sorted(SUPPORTED_FRAMEWORK_NAMES)}."
        )
    action = _require_mapping(framework, "action_model", config_path)
    datasets = _require_mapping(config, "datasets", config_path)
    vla_data = _require_mapping(datasets, "vla_data", config_path)
    return _build_contract(root, weights_path, config_path, statistics_path, str(framework_name), action, vla_data)


def _build_contract(
    root: Path,
    weights_path: Path,
    config_path: Path,
    statistics_path: Path,
    framework_name: str,
    action: dict[str, Any],
    vla_data: dict[str, Any],
) -> CheckpointContract:
    action_dim = int(action.get("action_dim", 0))
    state_dim = int(action.get("state_dim", 0))
    action_horizon = int(action.get("action_horizon", 0))
    if (action_dim, state_dim, action_horizon) != (23, 23, 30):
        raise ValueError(
            "Published checkpoints require action_dim=23, state_dim=23, and action_horizon=30; "
            f"got {(action_dim, state_dim, action_horizon)}."
        )
    if not bool(action.get("enable_urdf_input", False)):
        raise ValueError("Published checkpoints require framework.action_model.enable_urdf_input=true.")

    is_delta = bool(vla_data.get("delta_action", False))
    return CheckpointContract(
        root=root,
        weights_path=weights_path,
        config_path=config_path,
        statistics_path=statistics_path,
        framework_name=framework_name,
        action_mode="delta" if is_delta else "absolute",
        action_horizon=action_horizon,
        rotation_repr="rot6d",
        delta_base_state=str(vla_data.get("delta_base_state", "first_state")) if is_delta else None,
        delta_action_mode=str(vla_data.get("delta_action_mode", "state_anchor")) if is_delta else None,
        delta_quat_mode=str(vla_data.get("delta_quat_mode", "subtract")) if is_delta else None,
        rot_norm_mode=str(vla_data.get("rot_norm_mode", "q99")),
    )
