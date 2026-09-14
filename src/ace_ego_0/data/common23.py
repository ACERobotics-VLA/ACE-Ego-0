"""Common23 transforms backed only by shipped normalization statistics."""

# Common23 packing and Fourier hand conversion follow the NVIDIA GR00T data
# transform contract; see THIRD_PARTY_NOTICES.md and Apache-2.0 upstream notices.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-6
COMMON23_DIM = 23

ROBOCASA_RAW_SEGMENTS = {
    "left_eef_pos": slice(0, 3),
    "left_eef_rot6d": slice(3, 9),
    "right_eef_pos": slice(9, 12),
    "right_eef_rot6d": slice(12, 18),
    "left_hand": slice(18, 24),
    "right_hand": slice(24, 30),
    "waist": slice(30, 33),
}
ARX_RAW_SEGMENTS = {
    "left_eef": slice(0, 9),
    "left_gripper": slice(9, 10),
    "right_eef": slice(10, 19),
    "right_gripper": slice(19, 20),
}


def _load_json_mapping(path: Path, *, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing shipped {artifact_name}: {path}. ACE-Ego-0 never computes normalization statistics at runtime."
        )
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{artifact_name} must contain a JSON object: {path}")
    return value


@dataclass(frozen=True)
class Common23Norm:
    """Validated raw-schema q99 statistics for one SFT recipe."""

    action_mode: str
    robot_type: str
    norm_key: str
    action: dict[str, Any]
    state: dict[str, Any]
    source_path: Path

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        expected_action_mode: str,
        expected_robot_type: str,
    ) -> "Common23Norm":
        resolved_manifest = Path(manifest_path).expanduser().resolve()
        manifest = _load_json_mapping(resolved_manifest, artifact_name="norm manifest")
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported norm manifest schema in {resolved_manifest}.")

        action_mode = str(manifest.get("action_mode", "")).lower()
        if action_mode != expected_action_mode:
            raise ValueError(
                f"Norm manifest action_mode={action_mode!r} does not match recipe action_mode={expected_action_mode!r}."
            )
        robot_type = str(manifest.get("robot_type", ""))
        if robot_type != expected_robot_type:
            raise ValueError(
                f"Norm manifest robot_type={robot_type!r} does not match recipe robot_type={expected_robot_type!r}."
            )

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts.get("merged_stats_path"):
            raise ValueError(f"Norm manifest must define artifacts.merged_stats_path: {resolved_manifest}")
        stats_path = Path(str(artifacts["merged_stats_path"])).expanduser()
        if not stats_path.is_absolute():
            stats_path = resolved_manifest.parent / stats_path
        stats = _load_json_mapping(stats_path.resolve(), artifact_name="merged normalization statistics")

        norm_key = str(manifest.get("norm_key", ""))
        if not norm_key:
            raise ValueError(f"Norm manifest must define norm_key: {resolved_manifest}")
        expected_norm_key = "gr1" if expected_robot_type == "fourier_gr1_eef_gripper_6d" else "arx"
        if norm_key != expected_norm_key:
            raise ValueError(
                f"Norm manifest for {expected_robot_type!r} must use norm_key={expected_norm_key!r}, got {norm_key!r}."
            )
        if norm_key not in stats or not isinstance(stats[norm_key], dict):
            raise ValueError(f"Merged stats do not contain norm_key={norm_key!r}: {stats_path}")
        stats_block = stats[norm_key]
        action = stats_block.get("action")
        state = stats_block.get("state")
        if not isinstance(action, dict) or not isinstance(state, dict):
            raise ValueError(f"Merged stats must contain action/state mappings under {norm_key!r}.")

        expected_dim = 33 if expected_robot_type == "fourier_gr1_eef_gripper_6d" else 20
        for kind, block in (("action", action), ("state", state)):
            for field in ("q01", "q99"):
                values = block.get(field)
                if not isinstance(values, list) or len(values) != expected_dim:
                    raise ValueError(
                        f"{kind}.{field} in {stats_path} must have {expected_dim} values, "
                        f"got {len(values) if isinstance(values, list) else type(values).__name__}."
                    )
                if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                    raise ValueError(f"{kind}.{field} in {stats_path} contains non-finite values.")
            q01 = np.asarray(block["q01"], dtype=np.float64)
            q99 = np.asarray(block["q99"], dtype=np.float64)
            if np.any(q99 < q01):
                raise ValueError(f"{kind} q99 must be greater than or equal to q01 in {stats_path}.")
        return cls(
            action_mode=action_mode,
            robot_type=robot_type,
            norm_key=norm_key,
            action=action,
            state=state,
            source_path=stats_path.resolve(),
        )


def _normalize_q99(values: np.ndarray, statistics: dict[str, Any], segment: slice) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    q01 = np.asarray(statistics["q01"][segment], dtype=np.float32)
    q99 = np.asarray(statistics["q99"][segment], dtype=np.float32)
    valid = q01 != q99
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[..., valid] = (values[..., valid] - q01[valid]) / (q99[valid] - q01[valid] + EPS) * 2.0 - 1.0
    normalized[..., ~valid] = values[..., ~valid]
    return normalized


def _normalize_gripper(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values * 2.0 - 1.0


HAND_OPEN = np.array([-1.5, -1.5, -1.5, -1.5, -3.0, 3.0], dtype=np.float32)
HAND_CLOSE = np.array([1.5, 1.5, 1.5, 1.5, 3.0, 3.0], dtype=np.float32)


def hand_6d_to_gripper_1d(hand: np.ndarray) -> np.ndarray:
    """Convert Fourier's first four finger joints from [-1.5, 1.5] to one gripper value in [0, 1]."""
    hand = np.asarray(hand, dtype=np.float32)
    if hand.shape[-1] != 6:
        raise ValueError(f"Fourier hand input must be 6D, got {hand.shape}.")
    fingers = np.clip((hand[..., :4] + 1.5) / 3.0, 0.0, 1.0)
    return fingers.mean(axis=-1, keepdims=True, dtype=np.float32)


def gripper_1d_to_hand_6d(gripper: np.ndarray) -> np.ndarray:
    """Convert a scalar gripper value in [0, 1] to Fourier's 6D hand pose."""
    values = np.asarray(gripper, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.shape[-1] != 1:
        values = values[..., np.newaxis]
    values = np.clip(values, 0.0, 1.0)
    hand_action = HAND_OPEN + values * (HAND_CLOSE - HAND_OPEN)
    hand_action[..., 5] = 3.0
    return hand_action.astype(np.float32)


def transform_robocasa_common23(
    *,
    state_components: dict[str, np.ndarray],
    action_components: dict[str, np.ndarray],
    norm: Common23Norm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform raw GR1 camera-EEF fields into normalized Common23 tensors."""
    if norm.robot_type != "fourier_gr1_eef_gripper_6d":
        raise ValueError(f"RoboCasa transform received stats for {norm.robot_type!r}.")

    required = set(ROBOCASA_RAW_SEGMENTS)
    if set(state_components) != required or set(action_components) != required:
        raise ValueError("RoboCasa raw components do not match the reviewed Common23 contract.")
    state_raw = {key: np.asarray(value, dtype=np.float32) for key, value in state_components.items()}
    action_raw = {key: np.asarray(value, dtype=np.float32) for key, value in action_components.items()}

    if norm.action_mode == "delta":
        for key in ("left_eef_pos", "left_eef_rot6d", "right_eef_pos", "right_eef_rot6d", "waist"):
            action_raw[key] = action_raw[key] - state_raw[key]

    state_parts = [
        _normalize_q99(state_raw["left_eef_pos"], norm.state, ROBOCASA_RAW_SEGMENTS["left_eef_pos"]),
        _normalize_q99(state_raw["left_eef_rot6d"], norm.state, ROBOCASA_RAW_SEGMENTS["left_eef_rot6d"]),
        _normalize_q99(state_raw["right_eef_pos"], norm.state, ROBOCASA_RAW_SEGMENTS["right_eef_pos"]),
        _normalize_q99(state_raw["right_eef_rot6d"], norm.state, ROBOCASA_RAW_SEGMENTS["right_eef_rot6d"]),
        _normalize_gripper(hand_6d_to_gripper_1d(state_raw["left_hand"])),
        _normalize_gripper(hand_6d_to_gripper_1d(state_raw["right_hand"])),
        _normalize_q99(state_raw["waist"], norm.state, ROBOCASA_RAW_SEGMENTS["waist"]),
    ]
    action_parts = [
        _normalize_q99(action_raw["left_eef_pos"], norm.action, ROBOCASA_RAW_SEGMENTS["left_eef_pos"]),
        _normalize_q99(action_raw["left_eef_rot6d"], norm.action, ROBOCASA_RAW_SEGMENTS["left_eef_rot6d"]),
        _normalize_q99(action_raw["right_eef_pos"], norm.action, ROBOCASA_RAW_SEGMENTS["right_eef_pos"]),
        _normalize_q99(action_raw["right_eef_rot6d"], norm.action, ROBOCASA_RAW_SEGMENTS["right_eef_rot6d"]),
        _normalize_gripper(hand_6d_to_gripper_1d(action_raw["left_hand"])),
        _normalize_gripper(hand_6d_to_gripper_1d(action_raw["right_hand"])),
        _normalize_q99(action_raw["waist"], norm.action, ROBOCASA_RAW_SEGMENTS["waist"]),
    ]
    state = np.concatenate(state_parts, axis=-1).astype(np.float32, copy=False)
    action = np.concatenate(action_parts, axis=-1).astype(np.float32, copy=False)
    if state.shape != (COMMON23_DIM,) or action.ndim != 2 or action.shape[-1] != COMMON23_DIM:
        raise RuntimeError(f"Invalid RoboCasa Common23 shapes: state={state.shape}, action={action.shape}.")
    return (
        state[None, :],
        action,
        np.ones(COMMON23_DIM, dtype=bool),
        np.ones(COMMON23_DIM, dtype=bool),
    )


def transform_arx_common23(
    *,
    raw_state: np.ndarray,
    raw_actions: np.ndarray,
    norm: Common23Norm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform ARX raw20 EEF vectors into normalized Common23 tensors."""
    if norm.robot_type != "arx_dual_arm_common23":
        raise ValueError(f"ARX transform received stats for {norm.robot_type!r}.")
    raw_state = np.asarray(raw_state, dtype=np.float32)
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    if raw_state.shape != (20,) or raw_actions.ndim != 2 or raw_actions.shape[-1] != 20:
        raise ValueError(f"ARX expects state [20] and actions [T,20], got {raw_state.shape}/{raw_actions.shape}.")

    actions = raw_actions.copy()
    if norm.action_mode == "delta":
        actions[..., ARX_RAW_SEGMENTS["left_eef"]] -= raw_state[ARX_RAW_SEGMENTS["left_eef"]]
        actions[..., ARX_RAW_SEGMENTS["right_eef"]] -= raw_state[ARX_RAW_SEGMENTS["right_eef"]]

    normalized_state = _normalize_q99(raw_state, norm.state, slice(0, 20))
    normalized_actions = _normalize_q99(actions, norm.action, slice(0, 20))
    state = np.concatenate(
        [
            normalized_state[0:3],
            normalized_state[3:9],
            normalized_state[10:13],
            normalized_state[13:19],
            normalized_state[9:10],
            normalized_state[19:20],
            np.zeros(3, dtype=np.float32),
        ]
    )
    action = np.concatenate(
        [
            normalized_actions[..., 0:3],
            normalized_actions[..., 3:9],
            normalized_actions[..., 10:13],
            normalized_actions[..., 13:19],
            normalized_actions[..., 9:10],
            normalized_actions[..., 19:20],
            np.zeros((*normalized_actions.shape[:-1], 3), dtype=np.float32),
        ],
        axis=-1,
    )
    action_mask = np.ones(COMMON23_DIM, dtype=bool)
    state_mask = np.ones(COMMON23_DIM, dtype=bool)
    action_mask[20:23] = False
    return state[None, :], action.astype(np.float32, copy=False), action_mask, state_mask
