from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

import cv2 as cv
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ace_ego_0.model.framework.share_tools import read_mode_config
from ace_ego_0.model.modules.urdf.robot_urdf_registry import resolve_robot_name
from ace_ego_0.utils.rotation_utils import (
    apply_quaternion_delta_np,
    matrix_to_rot6d,
    rot6d_to_matrix,
)
from evaluation.robocasa24.hand_to_gripper import (
    gripper_1d_to_hand_6d,
    hand_6d_to_gripper_1d,
)
from evaluation.robocasa24.ik_utils import (
    IKUnreachableError,
    InverseKinematicsConfig,
    RobotKinematics,
    convert_pose_xyzw_to_wxyz,
    create_sub_urdf,
    fix_continuous_joint_limits,
)
from evaluation.robocasa24.server.tools.websocket_policy_client import WebsocketClientPolicy

LEGACY_EVAL_IMAGE_SIZE = [224, 224]
EPS = 1e-6
ROBOCASA_EVAL_JOINT_NAMES = {
    "waist": ("robot0_torso_waist_yaw", "robot0_torso_waist_pitch", "robot0_torso_waist_roll"),
    "left_arm": tuple(
        f"robot0_l_{name}"
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "wrist_yaw",
            "wrist_roll",
            "wrist_pitch",
        )
    ),
    "right_arm": tuple(
        f"robot0_r_{name}"
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "wrist_yaw",
            "wrist_roll",
            "wrist_pitch",
        )
    ),
    "left_hand": (
        "robot0_l_f_pinky",
        "robot0_l_f_ring",
        "robot0_l_f_middle",
        "robot0_l_f_index",
        "robot0_l_f_thumb_pitch",
        "robot0_l_f_thumb_yaw",
    ),
    "right_hand": (
        "robot0_r_f_pinky",
        "robot0_r_f_ring",
        "robot0_r_f_middle",
        "robot0_r_f_index",
        "robot0_r_f_thumb_pitch",
        "robot0_r_f_thumb_yaw",
    ),
}
ROBOCASA_WAIST_LANGUAGE_PREFIXES = ("locked_waist:", "unlocked_waist:")


def _format_debug_array_summary(name: str, values: np.ndarray) -> str:
    """Format a compact numeric summary for eval action tracing."""
    array = np.asarray(values, dtype=np.float32)
    flattened = array.reshape(-1)
    first_vector = np.round(array.reshape(-1, array.shape[-1])[0], 5).tolist()
    return (
        f"{name}: shape={array.shape}, min={float(flattened.min()):.5f}, "
        f"max={float(flattened.max()):.5f}, first={first_vector}"
    )


# ============================================================================
# Checkpoint/Image Configuration Helpers
# ============================================================================


def _resolve_preserve_raw_image_size_flag(model_config: dict | None) -> bool:
    """Return whether checkpoint config requests raw-resolution image inputs."""
    if not isinstance(model_config, dict):
        return False

    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, dict):
        return False

    vla_data_cfg = datasets_cfg.get("vla_data", {})
    if not isinstance(vla_data_cfg, dict):
        return False

    raw_flag = vla_data_cfg.get("preserve_raw_image_size", False)
    if isinstance(raw_flag, bool):
        return raw_flag
    if isinstance(raw_flag, (int, float)):
        return bool(raw_flag)
    if isinstance(raw_flag, str):
        normalized_flag = raw_flag.strip().lower()
        if normalized_flag in {"1", "true", "yes", "on"}:
            return True
        if normalized_flag in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"`datasets.vla_data.preserve_raw_image_size` must be boolean-compatible, got {raw_flag!r}.")


def _resolve_checkpoint_configured_image_size(model_config: dict | None) -> list[int] | None:
    """Resolve the checkpoint-configured pre-resize image size."""
    if not isinstance(model_config, dict):
        return None

    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, dict):
        return None

    vla_data_cfg = datasets_cfg.get("vla_data", {})
    if not isinstance(vla_data_cfg, dict):
        return None

    raw_size = vla_data_cfg.get("image_size", None)
    if raw_size is None:
        return None
    if isinstance(raw_size, Sequence) and not isinstance(raw_size, (str, bytes)) and len(raw_size) == 2:
        width = int(raw_size[0])
        height = int(raw_size[1])
        if width <= 0 or height <= 0:
            raise TypeError(f"`datasets.vla_data.image_size` must contain positive integers, got {list(raw_size)!r}.")
        return [width, height]
    raise TypeError(
        f"`datasets.vla_data.image_size` must be a length-2 sequence like [width, height], got {raw_size!r}."
    )


def _resolve_checkpoint_default_image_size(policy_ckpt_path: str | Path) -> list[int] | None:
    """Resolve the eval image-size default from the checkpoint config."""
    try:
        model_config, _ = read_mode_config(policy_ckpt_path, allow_missing_norm_stats=True)
    except (AssertionError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        print(
            "*** WARNING: Failed to read checkpoint config for eval image size from "
            f"{policy_ckpt_path}: {error}. Falling back to legacy {LEGACY_EVAL_IMAGE_SIZE}. ***"
        )
        return LEGACY_EVAL_IMAGE_SIZE.copy()

    if _resolve_preserve_raw_image_size_flag(model_config):
        return None

    try:
        configured_image_size = _resolve_checkpoint_configured_image_size(model_config)
    except (TypeError, ValueError) as error:
        print(
            "*** WARNING: Invalid checkpoint `datasets.vla_data.image_size` in "
            f"{policy_ckpt_path}: {error}. Falling back to legacy {LEGACY_EVAL_IMAGE_SIZE}. ***"
        )
        return LEGACY_EVAL_IMAGE_SIZE.copy()

    if configured_image_size is not None:
        return configured_image_size
    return LEGACY_EVAL_IMAGE_SIZE.copy()


def _resolve_boolean_like(raw_flag: object, *, default: bool = False) -> bool:
    """Resolve a checkpoint config value that is expected to behave like a boolean."""
    if raw_flag is None:
        return default
    if isinstance(raw_flag, bool):
        return raw_flag
    if isinstance(raw_flag, (int, float)):
        return bool(raw_flag)
    if isinstance(raw_flag, str):
        normalized_flag = raw_flag.strip().lower()
        if normalized_flag in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized_flag in {"0", "false", "no", "n", "off"}:
            return False
    raise TypeError(f"Expected a boolean-compatible config value, got {raw_flag!r}.")


def _get_checkpoint_vla_data_config(model_config: dict | None) -> dict:
    if not isinstance(model_config, dict):
        return {}
    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, dict):
        return {}
    vla_data_cfg = datasets_cfg.get("vla_data", {})
    return vla_data_cfg if isinstance(vla_data_cfg, dict) else {}


def _resolve_checkpoint_language_source(model_config: dict | None) -> str:
    """Resolve the training-side language source recorded in a checkpoint config."""
    vla_data_cfg = _get_checkpoint_vla_data_config(model_config)
    language_source = vla_data_cfg.get("language_source")
    if language_source is None:
        legacy_flag = vla_data_cfg.get("language_from_episode_remarks")
        if legacy_flag is None:
            return "task_table"
        try:
            return "episode_remarks" if _resolve_boolean_like(legacy_flag, default=False) else "task_table"
        except TypeError as error:
            print(f"WARNING: Invalid checkpoint language_from_episode_remarks={legacy_flag!r}: {error}")
            return "task_table"

    normalized_source = str(language_source).strip().lower()
    if normalized_source in {"task_table", "episode_remarks"}:
        return normalized_source
    print(f"WARNING: Unknown checkpoint language_source={language_source!r}; treating as task_table.")
    return "task_table"


def _strip_robocasa_waist_language_prefix(language: str) -> str:
    """Remove RoboCasa Groot wrapper waist-control prefixes while preserving ordinary instructions."""
    stripped_language = language.strip()
    lower_language = stripped_language.lower()
    for prefix in ROBOCASA_WAIST_LANGUAGE_PREFIXES:
        if lower_language.startswith(prefix):
            return stripped_language[len(prefix) :].strip()
    return stripped_language


def normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    """Normalize xyzw quaternions and make the scalar term non-negative."""
    quat = np.array(quat, dtype=np.float32, copy=True)
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    quat = quat / norms
    quat = np.where(quat[..., 3:4] < 0, -quat, quat)
    return quat


def get_eef_layout(*, use_gripper_mode: bool = True, use_rot6d: bool = False) -> Dict[str, object]:
    """Return the canonical EEF action/state layout for eval-time tensors."""
    rot_dim = 6 if use_rot6d else 4
    hand_dim = 1 if use_gripper_mode else 6

    offset = 0
    left_pos = slice(offset, offset + 3)
    offset += 3
    left_rot = slice(offset, offset + rot_dim)
    offset += rot_dim
    right_pos = slice(offset, offset + 3)
    offset += 3
    right_rot = slice(offset, offset + rot_dim)
    offset += rot_dim
    left_hand = slice(offset, offset + hand_dim)
    offset += hand_dim
    right_hand = slice(offset, offset + hand_dim)
    offset += hand_dim
    waist = slice(offset, offset + 3)
    offset += 3

    return {
        "rot_dim": rot_dim,
        "hand_dim": hand_dim,
        "pose_dim": 2 * (3 + rot_dim),
        "total_dim": offset,
        "left_pos": left_pos,
        "left_rot": left_rot,
        "right_pos": right_pos,
        "right_rot": right_rot,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "waist": waist,
    }


def _rotation_matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to rot6d with the training-time canonical ordering."""
    matrix_np = np.asarray(matrix, dtype=np.float32)
    matrix_torch = torch.from_numpy(matrix_np)
    rot6d_torch = matrix_to_rot6d(matrix_torch)
    return rot6d_torch.numpy().astype(matrix_np.dtype)


def _rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert rot6d to rotation matrices with the training-time canonical ordering."""
    rot6d_np = np.asarray(rot6d, dtype=np.float32)
    rot6d_torch = torch.from_numpy(rot6d_np)
    matrix_torch = rot6d_to_matrix(rot6d_torch)
    return matrix_torch.numpy().astype(rot6d_np.dtype)


def quaternion_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to rot6d."""
    quat = normalize_quaternion_xyzw(quat)
    original_shape = quat.shape[:-1]
    matrix = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().astype(np.float32)
    rot6d = _rotation_matrix_to_rot6d(matrix)
    return rot6d.reshape(*original_shape, 6)


def rot6d_to_quaternion_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """Convert rot6d to xyzw quaternions."""
    rot6d = np.asarray(rot6d, dtype=np.float32)
    original_shape = rot6d.shape[:-1]
    matrix = _rot6d_to_rotation_matrix(rot6d.reshape(-1, 6))
    quat = Rotation.from_matrix(matrix).as_quat().astype(np.float32)
    quat = normalize_quaternion_xyzw(quat)
    return quat.reshape(*original_shape, 4)


def apply_rot6d_delta_np(state_rot6d: np.ndarray, delta_rot6d: np.ndarray) -> np.ndarray:
    """Apply a relative rot6d delta to a reference rot6d orientation."""
    state_matrix = _rot6d_to_rotation_matrix(state_rot6d)
    delta_matrix = _rot6d_to_rotation_matrix(delta_rot6d)
    return _rotation_matrix_to_rot6d(np.matmul(state_matrix, delta_matrix))


def _validate_delta_action_mode(delta_action_mode: str) -> str:
    """Validate the delta anchoring mode."""
    if delta_action_mode not in {"state_anchor", "chunk_anchor"}:
        raise ValueError(
            f"Unsupported delta_action_mode={delta_action_mode!r}. Expected 'state_anchor' or 'chunk_anchor'."
        )
    return delta_action_mode


def _validate_delta_quat_mode(delta_quat_mode: str) -> str:
    """Validate the rotation-delta representation."""
    if delta_quat_mode not in {"so3", "subtract"}:
        raise ValueError(f"Unsupported delta_quat_mode={delta_quat_mode!r}. Expected 'so3' or 'subtract'.")
    return delta_quat_mode


def _ensure_eef_chunk_batch(actions: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    """Ensure an EEF chunk has shape `(B, T, D)` and report whether batch was added."""
    actions = np.asarray(actions, dtype=np.float32)
    squeeze_batch = False
    if actions.ndim == 2:
        actions = actions[None, ...]
        squeeze_batch = True
    elif actions.ndim != 3:
        raise ValueError(f"Expected {name} to have shape (B, T, D) or (T, D), got {actions.shape}")
    return actions, squeeze_batch


def _ensure_eef_reference_batch(reference: np.ndarray | None, actions: np.ndarray, *, name: str) -> np.ndarray:
    """Broadcast a reference EEF state/action to `(B, 1|T, D)` for delta conversion."""
    if reference is None:
        raise ValueError(f"{name} is required for state-anchor delta reconstruction.")

    reference = np.asarray(reference, dtype=np.float32)
    if reference.ndim == 1:
        reference = reference[None, None, :]
    elif reference.ndim == 2:
        if reference.shape[-1] != actions.shape[-1]:
            raise ValueError(f"{name} dim mismatch: {reference.shape} vs {actions.shape}")
        reference = reference[:, None, :]
    elif reference.ndim != 3:
        raise ValueError(f"Expected {name} to have shape (D,), (B, D), or (B, 1|T, D), got {reference.shape}")

    if reference.shape[0] != actions.shape[0] or reference.shape[-1] != actions.shape[-1]:
        raise ValueError(f"{name} shape mismatch: actions={actions.shape}, {name}={reference.shape}")
    if reference.shape[1] not in (1, actions.shape[1]):
        raise ValueError(
            f"{name} time dim must be 1 or match the action chunk length, got {reference.shape[1]} vs {actions.shape[1]}"
        )
    return reference


def _apply_eef_delta_from_reference(
    delta_actions: np.ndarray,
    reference_actions: np.ndarray,
    *,
    use_gripper_mode: bool,
    use_rot6d: bool,
    delta_quat_mode: str,
) -> np.ndarray:
    """Decode EEF deltas relative to a reference state/action."""
    layout = get_eef_layout(use_gripper_mode=use_gripper_mode, use_rot6d=use_rot6d)
    delta_quat_mode = _validate_delta_quat_mode(delta_quat_mode)
    abs_actions = delta_actions + reference_actions

    if delta_actions.shape[-1] != layout["total_dim"]:
        raise ValueError(f"Expected {layout['total_dim']}D EEF actions, got {delta_actions.shape[-1]}D")

    if use_gripper_mode:
        abs_actions[..., layout["left_hand"]] = delta_actions[..., layout["left_hand"]]
        abs_actions[..., layout["right_hand"]] = delta_actions[..., layout["right_hand"]]

    if delta_quat_mode == "subtract":
        return abs_actions

    if use_rot6d:
        abs_actions[..., layout["left_rot"]] = apply_rot6d_delta_np(
            reference_actions[..., layout["left_rot"]],
            delta_actions[..., layout["left_rot"]],
        )
        abs_actions[..., layout["right_rot"]] = apply_rot6d_delta_np(
            reference_actions[..., layout["right_rot"]],
            delta_actions[..., layout["right_rot"]],
        )
    else:
        abs_actions[..., layout["left_rot"]] = apply_quaternion_delta_np(
            reference_actions[..., layout["left_rot"]],
            delta_actions[..., layout["left_rot"]],
        )
        abs_actions[..., layout["right_rot"]] = apply_quaternion_delta_np(
            reference_actions[..., layout["right_rot"]],
            delta_actions[..., layout["right_rot"]],
        )
    return abs_actions


def apply_delta_to_eef_actions(
    delta_actions: np.ndarray,
    base_state: np.ndarray | None = None,
    *,
    use_gripper_mode: bool = True,
    use_rot6d: bool = False,
    delta_action_mode: str = "state_anchor",
    delta_quat_mode: str = "so3",
) -> np.ndarray:
    """Recover absolute EEF actions from delta predictions."""
    delta_action_mode = _validate_delta_action_mode(delta_action_mode)
    delta_quat_mode = _validate_delta_quat_mode(delta_quat_mode)
    delta_actions, squeeze_batch = _ensure_eef_chunk_batch(delta_actions, name="delta_actions")

    if delta_action_mode == "state_anchor":
        reference_actions = _ensure_eef_reference_batch(base_state, delta_actions, name="base_state")
        abs_actions = _apply_eef_delta_from_reference(
            delta_actions,
            reference_actions,
            use_gripper_mode=use_gripper_mode,
            use_rot6d=use_rot6d,
            delta_quat_mode=delta_quat_mode,
        )
        return abs_actions[0] if squeeze_batch else abs_actions

    abs_actions = delta_actions.copy()
    if delta_actions.shape[1] > 1:
        anchor_action = delta_actions[:, :1, :]
        abs_actions[:, 1:, :] = _apply_eef_delta_from_reference(
            delta_actions[:, 1:, :],
            anchor_action,
            use_gripper_mode=use_gripper_mode,
            use_rot6d=use_rot6d,
            delta_quat_mode=delta_quat_mode,
        )
    return abs_actions[0] if squeeze_batch else abs_actions


# EEF Coordinate Transform Utilities
# ============================================================================


def get_body_pose_from_sim(sim, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a body's position and rotation matrix from MuJoCo sim."""
    body_id = sim.model.body_name2id(body_name)
    pos = sim.data.body_xpos[body_id].copy()
    mat = sim.data.body_xmat[body_id].reshape(3, 3).copy()
    return pos, mat


def get_camera_pose_from_sim(sim, camera_name: str = "egoview") -> tuple[np.ndarray, np.ndarray]:
    """Read a camera's position and rotation matrix from MuJoCo sim."""
    cam_id = sim.model.camera_name2id(camera_name)
    pos = sim.data.cam_xpos[cam_id].copy()
    mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    return pos, mat


def _get_eef_body_pose_from_sim(sim, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one arm's EEF body pose from sim as `(position, quat_xyzw)`."""
    body_name = f"robot0_{arm}_eef"
    body_id = sim.model.body_name2id(body_name)
    pos_world = sim.data.body_xpos[body_id].copy().astype(np.float32)
    quat_world_wxyz = sim.data.body_xquat[body_id].copy().astype(np.float32)
    quat_world_xyzw = np.array(
        [
            quat_world_wxyz[1],
            quat_world_wxyz[2],
            quat_world_wxyz[3],
            quat_world_wxyz[0],
        ],
        dtype=np.float32,
    )
    return pos_world, normalize_quaternion_xyzw(quat_world_xyzw)


def get_current_eef_state_in_camera_frame(
    sim,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    waist: np.ndarray,
    use_gripper_mode: bool = True,
    use_rot6d: bool = False,
) -> np.ndarray:
    """Build the current EEF state in camera frame from the live simulator state."""
    cam_pos, cam_mat = get_camera_pose_from_sim(sim, "egoview")
    cam_mat_inv = cam_mat.T
    cam_quat_world = Rotation.from_matrix(cam_mat).as_quat()
    cam_rot = Rotation.from_quat(cam_quat_world)

    left_eef_pos_world, left_eef_quat_world = _get_eef_body_pose_from_sim(sim, "left")
    right_eef_pos_world, right_eef_quat_world = _get_eef_body_pose_from_sim(sim, "right")

    left_eef_pos_cam = cam_mat_inv @ (left_eef_pos_world - cam_pos)
    right_eef_pos_cam = cam_mat_inv @ (right_eef_pos_world - cam_pos)

    left_rot_cam = cam_rot.inv() * Rotation.from_quat(left_eef_quat_world)
    right_rot_cam = cam_rot.inv() * Rotation.from_quat(right_eef_quat_world)
    left_eef_quat_cam = normalize_quaternion_xyzw(left_rot_cam.as_quat().astype(np.float32))
    right_eef_quat_cam = normalize_quaternion_xyzw(right_rot_cam.as_quat().astype(np.float32))

    left_eef_rot = quaternion_xyzw_to_rot6d(left_eef_quat_cam) if use_rot6d else left_eef_quat_cam
    right_eef_rot = quaternion_xyzw_to_rot6d(right_eef_quat_cam) if use_rot6d else right_eef_quat_cam

    layout = get_eef_layout(use_gripper_mode=use_gripper_mode, use_rot6d=use_rot6d)
    batch_size = left_hand.shape[0]
    eef_state = np.zeros((batch_size, 1, layout["total_dim"]), dtype=np.float32)

    for batch_index in range(batch_size):
        eef_state[batch_index, 0, layout["left_pos"]] = left_eef_pos_cam
        eef_state[batch_index, 0, layout["left_rot"]] = left_eef_rot
        eef_state[batch_index, 0, layout["right_pos"]] = right_eef_pos_cam
        eef_state[batch_index, 0, layout["right_rot"]] = right_eef_rot
        eef_state[batch_index, 0, layout["left_hand"]] = left_hand[batch_index, 0, :]
        eef_state[batch_index, 0, layout["right_hand"]] = right_hand[batch_index, 0, :]
        eef_state[batch_index, 0, layout["waist"]] = waist[batch_index, 0, :]
    return eef_state


class PolicyWarper:
    """Eval-side websocket policy wrapper for RoboCasa rollouts."""

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: str | None = None,
        image_size: list[int] | None = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        host: str = "127.0.0.1",
        port: int = 10095,
        n_action_steps: int = 2,
        model_action_horizon: int | None = None,
        ik_urdf_path: str | None = None,
        ik_learning_rate: float = 0.05,
        ik_max_iterations: int = 2000,
        ik_pose_tolerance: float = 0.005,
        ik_unreachable_threshold: float = 1.5,
        use_delta_action: bool = False,
        delta_base_state: str = "first_state",
        delta_rot_norm_mode: str = "min_max",
        delta_stats_path: str | None = None,
        delta_action_mode: str | None = None,
        delta_quat_mode: str | None = None,
    ) -> None:
        if image_size is None:
            image_size = _resolve_checkpoint_default_image_size(policy_ckpt_path)

        if image_size is not None and image_size != [224, 224]:
            raise ValueError("GR1 RoboCasa 24 checkpoints require image_size=[224, 224].")

        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key

        checkpoint_state_info = PolicyWarper._inspect_checkpoint_state_config(policy_ckpt_path)
        checkpoint_delta_info = PolicyWarper._inspect_checkpoint_delta_config(policy_ckpt_path)
        checkpoint_urdf_info = PolicyWarper._inspect_checkpoint_urdf_config(policy_ckpt_path)
        checkpoint_language_info = PolicyWarper._inspect_checkpoint_language_config(policy_ckpt_path)
        self.enable_state_input = checkpoint_state_info["state_supported"]
        self.state_input_format = checkpoint_state_info["state_input_format"]
        self.state_input_dim = checkpoint_state_info["state_dim"]
        self.framework_name = checkpoint_state_info["framework_name"]
        self.enable_urdf_input = checkpoint_urdf_info["urdf_enabled"]
        self.language_source = checkpoint_language_info["language_source"]
        self.strip_robocasa_waist_language_prefix = checkpoint_language_info["strip_waist_prefix"]
        self.robot_name = PolicyWarper._resolve_eval_robot_name(
            policy_ckpt_path,
            urdf_enabled=self.enable_urdf_input,
        )

        self.normalization_mode = PolicyWarper._detect_checkpoint_normalization_mode(policy_ckpt_path)

        print(f"*** RoboCasa 24 inference: unnorm_key={unnorm_key} ***")
        if self.enable_state_input:
            print(
                f"*** State input format: {self.state_input_format} "
                f"(framework={self.framework_name}, dim={self.state_input_dim}) ***"
            )
        if self.enable_urdf_input:
            print(f"*** URDF-conditioned checkpoint detected; eval robot_name={self.robot_name} ***")
        print(
            "*** Eval language handling: "
            f"source={self.language_source}, "
            f"strip_robocasa_waist_prefix={self.strip_robocasa_waist_language_prefix} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.preserve_raw_image_size = self.image_size is None
        self._last_ik_cfg = {"left": None, "right": None}
        self.n_action_steps = n_action_steps
        self.model_action_horizon = model_action_horizon if model_action_horizon is not None else n_action_steps
        self.use_gripper_mode = True
        self.use_rot6d = True
        self._eef_layout = get_eef_layout(use_gripper_mode=True, use_rot6d=True)
        self.ik_urdf_path = ik_urdf_path
        self.ik_learning_rate = ik_learning_rate
        self.ik_max_iterations = ik_max_iterations
        self.ik_pose_tolerance = ik_pose_tolerance
        self.ik_unreachable_threshold = ik_unreachable_threshold
        self._sim = None
        self._get_sim_fn = None
        self.state_norm_stats = None
        self._left_ik_solver = None
        self._right_ik_solver = None
        self._ik_initialized = False
        self._ik_joint_names = {"left": [], "right": []}
        self._ik_name_to_index = {"left": {}, "right": {}}
        self._ik_arm_joint_indices = {"left": [], "right": []}
        self._ik_waist_joint_indices = {"left": [], "right": []}
        self._last_ik_cfg = {"left": None, "right": None}
        self._robot_qpos_addr_cache = None
        debug_action_count = os.getenv("ACE_EGO_0_EVAL_DEBUG_ACTIONS", "0")
        try:
            self._debug_action_stats_remaining = max(0, int(debug_action_count))
        except ValueError as error:
            raise ValueError("ACE_EGO_0_EVAL_DEBUG_ACTIONS must be a non-negative integer.") from error

        if self.state_input_dim != 23:
            raise ValueError(f"GR1 RoboCasa 24 checkpoints require state_dim=23, got {self.state_input_dim}.")
        if self.preserve_raw_image_size:
            print("*** Raw image size preservation enabled for RoboCasa eval input preprocessing ***")
        else:
            print(f"*** RoboCasa eval input images will be resized to {self.image_size} before websocket inference ***")
        print("*** GR1 EEF+gripper inference with camera-frame rot6d and Mink IK enabled ***")

        if self.state_input_format != "raw":
            raise ValueError(f"GR1 RoboCasa 24 checkpoints require raw EEF state input, got {self.state_input_format}.")

        self.use_delta_action = use_delta_action
        self.delta_base_state = delta_base_state
        self.delta_rot_norm_mode = delta_rot_norm_mode
        self.delta_stats_path = delta_stats_path
        checkpoint_is_delta = checkpoint_delta_info.get("delta_action_mode") is not None
        if use_delta_action != checkpoint_is_delta:
            expected_mode = "delta" if checkpoint_is_delta else "absolute"
            raise ValueError(
                f"Checkpoint/action-mode mismatch: this checkpoint is {expected_mode}, "
                f"but use_delta_action={use_delta_action}."
            )
        self.delta_action_mode = _validate_delta_action_mode(
            delta_action_mode or checkpoint_delta_info.get("delta_action_mode") or "state_anchor"
        )
        self.delta_quat_mode = _validate_delta_quat_mode(
            delta_quat_mode or checkpoint_delta_info.get("delta_quat_mode") or "so3"
        )
        if self.use_delta_action:
            if self.delta_action_mode == "state_anchor" and self.delta_base_state != "first_state":
                raise NotImplementedError("Minimal delta eval currently supports only delta_base_state='first_state'.")
            print("*** DELTA ACTION MODE ENABLED: model outputs delta EEF actions ***")
            print(
                "    Delta reconstruction: "
                f"mode={self.delta_action_mode}, rot_mode={self.delta_quat_mode}, "
                f"base_state={self.delta_base_state}, rot_norm_mode={self.delta_rot_norm_mode}"
            )
            if self.delta_stats_path:
                print(f"    Delta stats override: {self.delta_stats_path}")

        self.task_description = None

        print(f"*** Loading checkpoint action/state statistics: {policy_ckpt_path} ***")
        _, norm_stats = read_mode_config(policy_ckpt_path)
        stats_key = self._check_unnorm_key(norm_stats, self.unnorm_key)
        self.action_norm_stats = norm_stats[stats_key]["action"]
        self.state_norm_stats = norm_stats[stats_key]["state"]

        self.action_norm_stats = self._convert_stats_to_eef_gripper_mode(self.action_norm_stats)
        self.state_norm_stats = self._convert_stats_to_eef_gripper_mode(self.state_norm_stats)
        for stats_name, stats in (("action", self.action_norm_stats), ("state", self.state_norm_stats)):
            if len(np.asarray(stats["min"])) != self._eef_layout["total_dim"]:
                raise ValueError(f"{stats_name} stats must be 23D after EEF/gripper conversion.")

    @property
    def sim(self):
        """Return the freshest available simulator handle for EEF eval."""
        if self._get_sim_fn is not None:
            try:
                sim = self._get_sim_fn()
                if sim is not None and hasattr(sim, "data") and hasattr(sim, "model"):
                    return sim
            except Exception as error:
                print(f"WARNING: sim getter failed: {error}")
        return self._sim

    @sim.setter
    def sim(self, value) -> None:
        """Cache one direct simulator handle as a fallback."""
        self._sim = value

    def set_sim_getter(self, get_sim_fn) -> None:
        """Register one callback that returns the current live simulator handle."""
        self._get_sim_fn = get_sim_fn
        print("*** Set sim getter function on model ***")

    def _convert_stats_to_eef_gripper_mode(self, stats_29d: dict) -> dict:
        """Convert EEF+hand checkpoint stats to EEF+gripper checkpoint stats."""
        source_layout = get_eef_layout(use_gripper_mode=False, use_rot6d=self.use_rot6d)
        target_layout = get_eef_layout(use_gripper_mode=True, use_rot6d=self.use_rot6d)
        stats_converted = {}

        for key in ["min", "max", "mean", "std", "q01", "q99"]:
            if key not in stats_29d:
                continue

            old_values = np.asarray(stats_29d[key], dtype=np.float32)
            if old_values.ndim != 1:
                raise ValueError(
                    "Checkpoint EEF stat conversion expects flattened stats arrays. "
                    f"Got shape {old_values.shape} for key {key!r}."
                )
            if len(old_values) == target_layout["total_dim"]:
                stats_converted[key] = old_values.tolist()
                continue
            if len(old_values) != source_layout["total_dim"]:
                raise ValueError(
                    f"Unexpected EEF stats dimension {len(old_values)}; expected "
                    f"{source_layout['total_dim']} or {target_layout['total_dim']}."
                )

            new_values = np.zeros(target_layout["total_dim"], dtype=np.float32)
            new_values[target_layout["left_pos"]] = old_values[source_layout["left_pos"]]
            new_values[target_layout["left_rot"]] = old_values[source_layout["left_rot"]]
            new_values[target_layout["right_pos"]] = old_values[source_layout["right_pos"]]
            new_values[target_layout["right_rot"]] = old_values[source_layout["right_rot"]]

            if key in {"min", "q01"}:
                gripper_value = 0.0
            elif key in {"max", "q99"}:
                gripper_value = 1.0
            else:
                gripper_value = 0.5
            new_values[target_layout["left_hand"].start] = gripper_value
            new_values[target_layout["right_hand"].start] = gripper_value
            new_values[target_layout["waist"]] = old_values[source_layout["waist"]]
            stats_converted[key] = new_values.tolist()

        if "mask" in stats_29d:
            old_mask = list(stats_29d["mask"])
            if len(old_mask) == target_layout["total_dim"]:
                stats_converted["mask"] = old_mask
            else:
                new_mask = [True] * target_layout["total_dim"]
                left_pos = target_layout["left_pos"]
                left_rot = target_layout["left_rot"]
                right_pos = target_layout["right_pos"]
                right_rot = target_layout["right_rot"]
                waist = target_layout["waist"]
                new_mask[left_pos] = old_mask[source_layout["left_pos"]]
                new_mask[left_rot] = old_mask[source_layout["left_rot"]]
                new_mask[right_pos] = old_mask[source_layout["right_pos"]]
                new_mask[right_rot] = old_mask[source_layout["right_rot"]]
                new_mask[target_layout["left_hand"].start] = True
                new_mask[target_layout["right_hand"].start] = True
                new_mask[waist] = old_mask[source_layout["waist"]]
                stats_converted["mask"] = new_mask

        print(
            f"*** Converted EEF checkpoint stats from {source_layout['total_dim']}D to "
            f"{target_layout['total_dim']}D (EEF+gripper mode) ***"
        )
        return stats_converted

    def reset(self, task_description: str | tuple, *, force: bool = False) -> None:
        """Reset episode-local buffers for a new task description."""
        if not force and task_description == self.task_description:
            return
        self.task_description = task_description
        self._last_ik_cfg = {"left": None, "right": None}

    def reset_after_ik_failure(self) -> None:
        """Reset policy state after an episode-ending live-model IK failure."""
        self.reset(self.task_description, force=True)

    def close(self) -> None:
        """Release the eval-side websocket client."""
        self.client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_current_eef_state_from_sim(
        self,
        *,
        left_hand_state: np.ndarray,
        right_hand_state: np.ndarray,
        waist_state: np.ndarray,
    ) -> np.ndarray:
        """Build the current live EEF state from the simulator."""
        if self.sim is None:
            raise RuntimeError("sim is required to build EEF state from the environment.")
        return get_current_eef_state_in_camera_frame(
            self.sim,
            left_hand_state,
            right_hand_state,
            waist_state,
            use_gripper_mode=True,
            use_rot6d=True,
        )

    @staticmethod
    def _resolve_default_ik_urdf_path() -> str:
        """Resolve the GR1 URDF used by the Mink IK solver."""
        candidates = [Path(__file__).resolve().parents[2] / "assets/GR1T2_with_hands.urdf"]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"GR1 URDF not found in expected paths: {candidates}")

    def _initialize_ik_solvers(self) -> None:
        """Initialize left/right Mink IK solvers lazily."""
        if self._ik_initialized:
            return

        urdf_path = Path(self.ik_urdf_path or self._resolve_default_ik_urdf_path())
        fixed_urdf_path = fix_continuous_joint_limits(urdf_path)
        ik_config = InverseKinematicsConfig(
            learning_rate=self.ik_learning_rate,
            max_iterations=self.ik_max_iterations,
            pose_tolerance=self.ik_pose_tolerance,
            unreachable_threshold=self.ik_unreachable_threshold,
        )
        left_sub_urdf = create_sub_urdf(fixed_urdf_path, "torso_waist_roll", "l_palm")
        right_sub_urdf = create_sub_urdf(fixed_urdf_path, "torso_waist_roll", "r_palm")
        self._left_ik_solver = RobotKinematics(left_sub_urdf, "l_palm", config=ik_config)
        self._right_ik_solver = RobotKinematics(right_sub_urdf, "r_palm", config=ik_config)

        for arm, solver in (("left", self._left_ik_solver), ("right", self._right_ik_solver)):
            names = list(solver.robot.joints.actuated_names)
            self._ik_joint_names[arm] = names
            self._ik_name_to_index[arm] = {name: idx for idx, name in enumerate(names)}
            prefix = "l_" if arm == "left" else "r_"
            arm_joint_names = [
                f"{prefix}shoulder_pitch",
                f"{prefix}shoulder_roll",
                f"{prefix}shoulder_yaw",
                f"{prefix}elbow_pitch",
                f"{prefix}wrist_yaw",
                f"{prefix}wrist_roll",
                f"{prefix}wrist_pitch",
            ]
            self._ik_arm_joint_indices[arm] = [self._ik_name_to_index[arm].get(name, -1) for name in arm_joint_names]
            waist_joint_names = ["waist_pan", "waist_pitch", "waist_yaw"]
            self._ik_waist_joint_indices[arm] = [
                self._ik_name_to_index[arm][name] for name in waist_joint_names if name in self._ik_name_to_index[arm]
            ]

        self._ik_initialized = True
        print(f"*** Initialized Mink IK solvers from URDF: {urdf_path} ***")

    def _build_ik_initial_guess(
        self,
        arm: str,
        arm_joints: np.ndarray,
        waist_joints: np.ndarray,
        *,
        base_cfg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build one full Mink joint vector from predicted waist and arm joints."""
        joint_names = self._ik_joint_names[arm]
        if base_cfg is not None and len(base_cfg) == len(joint_names):
            cfg = np.asarray(base_cfg, dtype=np.float32).copy()
        else:
            cfg = np.zeros(len(joint_names), dtype=np.float32)

        waist_joint_names = ["waist_pan", "waist_pitch", "waist_yaw"]
        for index, joint_name in enumerate(waist_joint_names):
            if index >= len(waist_joints):
                continue
            joint_index = self._ik_name_to_index[arm].get(joint_name)
            if joint_index is not None:
                cfg[joint_index] = waist_joints[index]

        prefix = "l_" if arm == "left" else "r_"
        arm_joint_names = [
            f"{prefix}shoulder_pitch",
            f"{prefix}shoulder_roll",
            f"{prefix}shoulder_yaw",
            f"{prefix}elbow_pitch",
            f"{prefix}wrist_yaw",
            f"{prefix}wrist_roll",
            f"{prefix}wrist_pitch",
        ]
        for index, joint_name in enumerate(arm_joint_names):
            if index >= len(arm_joints):
                continue
            joint_index = self._ik_name_to_index[arm].get(joint_name)
            if joint_index is not None:
                cfg[joint_index] = arm_joints[index]
        return cfg

    def _extract_arm_joints(self, ik_cfg: np.ndarray, arm: str) -> np.ndarray:
        """Extract RoboCasa-order 7D arm joints from one Mink solution vector."""
        arm_joints = np.zeros(7, dtype=np.float32)
        for index, joint_index in enumerate(self._ik_arm_joint_indices[arm]):
            if joint_index >= 0:
                arm_joints[index] = ik_cfg[joint_index]
        return arm_joints

    def _get_robot_qpos_addr_cache(self) -> dict[str, tuple[int | None, ...]]:
        """Resolve sim qpos addresses for RoboCasa joint groups once."""
        if self.sim is None:
            raise RuntimeError("sim is required to resolve robot qpos addresses.")
        if self._robot_qpos_addr_cache is not None:
            return self._robot_qpos_addr_cache

        cache: dict[str, tuple[int | None, ...]] = {}
        for group_name, joint_names in ROBOCASA_EVAL_JOINT_NAMES.items():
            qpos_addresses: list[int | None] = []
            for joint_name in joint_names:
                try:
                    joint_id = self.sim.model.joint_name2id(joint_name)
                except ValueError:
                    qpos_addresses.append(None)
                    continue
                qpos_addresses.append(int(self.sim.model.jnt_qposadr[joint_id]))
            cache[group_name] = tuple(qpos_addresses)
        self._robot_qpos_addr_cache = cache
        return cache

    def _capture_sim_state(self) -> dict[str, np.ndarray]:
        """Capture sim state so chunk-local rollout can restore it afterwards."""
        if self.sim is None:
            raise RuntimeError("sim is required to capture simulator state.")
        snapshot = {
            "qpos": self.sim.data.qpos.copy(),
            "qvel": self.sim.data.qvel.copy(),
        }
        if getattr(self.sim.data, "act", None) is not None:
            snapshot["act"] = self.sim.data.act.copy()
        if getattr(self.sim.data, "ctrl", None) is not None:
            snapshot["ctrl"] = self.sim.data.ctrl.copy()
        return snapshot

    def _restore_sim_state(self, snapshot: dict[str, np.ndarray]) -> None:
        """Restore a previously captured sim snapshot."""
        if self.sim is None:
            raise RuntimeError("sim is required to restore simulator state.")
        self.sim.data.qpos[:] = snapshot["qpos"]
        self.sim.data.qvel[:] = snapshot["qvel"]
        if "act" in snapshot and getattr(self.sim.data, "act", None) is not None:
            self.sim.data.act[:] = snapshot["act"]
        if "ctrl" in snapshot and getattr(self.sim.data, "ctrl", None) is not None:
            self.sim.data.ctrl[:] = snapshot["ctrl"]
        self.sim.forward()

    def _set_joint_targets_on_sim(self, qpos_addresses: tuple[int | None, ...], values: np.ndarray) -> None:
        """Assign one joint group to current sim qpos."""
        if self.sim is None:
            raise RuntimeError("sim is required to assign joint targets.")
        flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
        for index, qpos_address in enumerate(qpos_addresses):
            if qpos_address is None or index >= len(flat_values):
                continue
            self.sim.data.qpos[qpos_address] = flat_values[index]

    def _apply_robot_rollout_state_to_sim(
        self,
        *,
        left_arm_joints: np.ndarray,
        right_arm_joints: np.ndarray,
        waist_joints: np.ndarray,
        left_hand_state: np.ndarray,
        right_hand_state: np.ndarray,
    ) -> None:
        """Roll the robot state forward inside sim so later chunk steps can read updated camera pose."""
        qpos_addr_cache = self._get_robot_qpos_addr_cache()
        self._set_joint_targets_on_sim(qpos_addr_cache["waist"], waist_joints)
        self._set_joint_targets_on_sim(qpos_addr_cache["left_arm"], left_arm_joints)
        self._set_joint_targets_on_sim(qpos_addr_cache["right_arm"], right_arm_joints)
        self._set_joint_targets_on_sim(qpos_addr_cache["left_hand"], left_hand_state)
        self._set_joint_targets_on_sim(qpos_addr_cache["right_hand"], right_hand_state)
        self.sim.forward()

    def _transform_eef_pose_to_torso_frame(
        self, eef_pos: np.ndarray, eef_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform one camera-frame EEF pose into the Mink torso frame."""
        if self.sim is None:
            raise RuntimeError("sim is required for camera-to-torso transforms.")
        cam_pos, cam_mat = get_camera_pose_from_sim(self.sim, "egoview")
        torso_pos, torso_mat = get_body_pose_from_sim(self.sim, "robot0_torso_waist_roll")
        eef_mat_world = cam_mat @ Rotation.from_quat(eef_quat).as_matrix()
        eef_pos_world = cam_mat @ eef_pos + cam_pos
        eef_mat_torso = torso_mat.T @ eef_mat_world
        eef_pos_torso = torso_mat.T @ (eef_pos_world - torso_pos)
        return eef_pos_torso.astype(np.float32), normalize_quaternion_xyzw(
            Rotation.from_matrix(eef_mat_torso).as_quat().astype(np.float32)
        )

    def _compute_ik_for_eef_pose(
        self,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        arm: str,
        *,
        current_joint_angles: np.ndarray | None,
        waist_joints: np.ndarray,
    ) -> np.ndarray:
        """Solve one arm IK target in torso/base frame and return RoboCasa-order arm joints."""
        if not self._ik_initialized:
            self._initialize_ik_solvers()

        solver = self._left_ik_solver if arm == "left" else self._right_ik_solver
        target_pose = convert_pose_xyzw_to_wxyz(eef_pos, eef_quat, arm=arm)
        cached_cfg = self._last_ik_cfg[arm]
        if current_joint_angles is not None:
            initial_guess = self._build_ik_initial_guess(
                arm,
                np.asarray(current_joint_angles, dtype=np.float32),
                np.asarray(waist_joints, dtype=np.float32),
                base_cfg=cached_cfg,
            )
        elif cached_cfg is not None:
            initial_guess = np.asarray(cached_cfg, dtype=np.float32).copy()
        else:
            initial_guess = None

        try:
            ik_result = np.asarray(
                solver.compute_inverse_kinematics(target_pose, initial_guess=initial_guess), dtype=np.float32
            )
        except IKUnreachableError as error:
            print(f"WARNING: Mink IK failed for {arm} arm: {error}")
            if current_joint_angles is not None:
                return np.asarray(current_joint_angles, dtype=np.float32).copy()
            return np.zeros(7, dtype=np.float32)

        for index, joint_index in enumerate(self._ik_waist_joint_indices[arm]):
            if index < len(waist_joints):
                ik_result[joint_index] = waist_joints[index]
        self._last_ik_cfg[arm] = ik_result.copy()
        return self._extract_arm_joints(ik_result, arm)

    def _solve_eef_action_chunk_with_ik(
        self,
        *,
        left_eef_pos: np.ndarray,
        left_eef_quat: np.ndarray,
        right_eef_pos: np.ndarray,
        right_eef_quat: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        waist: np.ndarray,
        current_left_arm: np.ndarray,
        current_right_arm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert one predicted EEF action chunk into joint targets with the selected IK backend."""
        if self.sim is None:
            raise RuntimeError("EEF IK chunk solving requires sim to be set.")

        batch_size, time_steps, _ = left_hand.shape
        if batch_size != 1:
            raise NotImplementedError("Minimal EEF IK eval currently supports only batch size 1.")

        left_arm_joints = np.zeros((batch_size, time_steps, 7), dtype=np.float32)
        right_arm_joints = np.zeros((batch_size, time_steps, 7), dtype=np.float32)
        sim_snapshot = self._capture_sim_state() if time_steps > 1 else None

        try:
            for batch_index in range(batch_size):
                current_left = np.asarray(current_left_arm[batch_index], dtype=np.float32).copy()
                current_right = np.asarray(current_right_arm[batch_index], dtype=np.float32).copy()
                for step_index in range(time_steps):
                    self._apply_robot_rollout_state_to_sim(
                        left_arm_joints=current_left,
                        right_arm_joints=current_right,
                        waist_joints=waist[batch_index, step_index],
                        left_hand_state=left_hand[batch_index, step_index],
                        right_hand_state=right_hand[batch_index, step_index],
                    )
                    left_pos_torso, left_quat_torso = self._transform_eef_pose_to_torso_frame(
                        left_eef_pos[batch_index, step_index], left_eef_quat[batch_index, step_index]
                    )
                    right_pos_torso, right_quat_torso = self._transform_eef_pose_to_torso_frame(
                        right_eef_pos[batch_index, step_index], right_eef_quat[batch_index, step_index]
                    )

                    waist_joints = np.asarray(waist[batch_index, step_index], dtype=np.float32)
                    left_arm_joints[batch_index, step_index] = self._compute_ik_for_eef_pose(
                        left_pos_torso,
                        left_quat_torso,
                        "left",
                        current_joint_angles=current_left,
                        waist_joints=waist_joints,
                    )
                    right_arm_joints[batch_index, step_index] = self._compute_ik_for_eef_pose(
                        right_pos_torso,
                        right_quat_torso,
                        "right",
                        current_joint_angles=current_right,
                        waist_joints=waist_joints,
                    )
                    current_left = left_arm_joints[batch_index, step_index]
                    current_right = right_arm_joints[batch_index, step_index]
        finally:
            if sim_snapshot is not None:
                self._restore_sim_state(sim_snapshot)

        return left_arm_joints, right_arm_joints

    def _normalize_task_description(self, task_description):
        if not self.strip_robocasa_waist_language_prefix:
            return task_description
        if isinstance(task_description, str):
            return _strip_robocasa_waist_language_prefix(task_description)
        if isinstance(task_description, tuple):
            return tuple(self._normalize_task_description(item) for item in task_description)
        if isinstance(task_description, list):
            return [self._normalize_task_description(item) for item in task_description]
        return task_description

    def _log_action_trace(
        self,
        *,
        current_eef_state_raw: np.ndarray | None,
        input_state_tensor: np.ndarray | None,
        normalized_actions: np.ndarray,
        raw_actions: np.ndarray,
        unnormalized_model_actions: np.ndarray | None = None,
        raw_action: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Log the numeric state/action contract for a bounded number of eval calls."""
        if raw_action is not None:
            print("*** ROBOCASA EVAL FINAL ENV ACTION TRACE ***")
            for key, values in raw_action.items():
                print(_format_debug_array_summary(f"env.{key}", values))
            return

        print("*** ROBOCASA EVAL MODEL ACTION TRACE ***")
        if current_eef_state_raw is not None:
            print(_format_debug_array_summary("state.raw_eef", current_eef_state_raw))
        if input_state_tensor is not None:
            print(_format_debug_array_summary("state.normalized", input_state_tensor))
        print(_format_debug_array_summary("action.normalized", normalized_actions[:, : self.n_action_steps]))
        if unnormalized_model_actions is not None:
            print(
                _format_debug_array_summary(
                    "action.unnormalized_model_23d",
                    unnormalized_model_actions[:, : self.n_action_steps],
                )
            )
        print(_format_debug_array_summary("action.absolute_23d", raw_actions[:, : self.n_action_steps]))

    def step(self, observations, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        """Run one websocket inference step and convert outputs to RoboCasa actions."""
        task_description = self._normalize_task_description(observations["annotation.human.coarse_action"][0])
        images = observations["video.ego_view"]
        state = {
            "left_arm": observations["state.left_arm"],
            "right_arm": observations["state.right_arm"],
            "left_hand": observations["state.left_hand"],
            "right_hand": observations["state.right_hand"],
            "waist": observations["state.waist"],
        }
        if state["waist"].shape[1] != 1:
            raise NotImplementedError(
                "Minimal EEF eval currently supports only state_delta_indices=[0]. "
                "Temporal EEF-state replay has not been migrated yet."
            )

        if self.sim is None:
            raise RuntimeError("RoboCasa EEF inference requires `model.sim` to be set before inference.")
        current_eef_state_raw = self._build_current_eef_state_from_sim(
            left_hand_state=hand_6d_to_gripper_1d(state["left_hand"])[:, -1:, :],
            right_hand_state=hand_6d_to_gripper_1d(state["right_hand"])[:, -1:, :],
            waist_state=state["waist"][:, -1:, :],
        )
        input_state_tensor = self.normalize_eef_state(current_eef_state_raw)

        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        images = [[self._resize_image(img) for img in sample] for sample in images]
        batch_size = len(images)
        if input_state_tensor.ndim == 3 and input_state_tensor.shape[1] == 1:
            input_state_tensor = input_state_tensor[:, 0, :]
        input_state = [sample for sample in input_state_tensor]

        if isinstance(self.task_description, str):
            instructions = [self.task_description] * batch_size
        else:
            instructions = list(self.task_description)
            if len(instructions) == 1 and batch_size > 1:
                instructions = instructions * batch_size
            if len(instructions) != batch_size:
                raise ValueError(
                    f"Task-description batch size mismatch: expected {batch_size}, got {len(instructions)}."
                )

        examples = []
        for batch_index in range(batch_size):
            example = {
                "image": images[batch_index],
                "lang": instructions[batch_index],
                "state": input_state[batch_index],
                "state_mask": [True] * self.state_input_dim,
                "robot_name": self.robot_name,
            }
            examples.append(example)

        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "action_horizon": self.model_action_horizon,
        }
        response = self.client.predict_action(vla_input)

        if response.get("status") == "error" or not response.get("ok", True):
            error_info = response.get("error", {})
            error_msg = error_info.get("message", "Unknown error") if isinstance(error_info, dict) else str(error_info)
            raise RuntimeError(f"Server inference error: {error_msg}\nFull response: {response}")

        normalized_actions = np.asarray(response["data"]["normalized_actions"], dtype=np.float32)
        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions,
            action_norm_stats=self.action_norm_stats,
            normalization_mode=self.normalization_mode,
        )
        unnormalized_model_actions = raw_actions.copy() if self.use_delta_action else None

        if self.use_delta_action:
            if self.delta_action_mode == "state_anchor" and current_eef_state_raw is None:
                raise RuntimeError(
                    "Delta-action reconstruction requires the current raw EEF state from simulation, but none was available."
                )
            raw_actions = apply_delta_to_eef_actions(
                raw_actions,
                current_eef_state_raw,
                use_gripper_mode=self.use_gripper_mode,
                use_rot6d=self.use_rot6d,
                delta_action_mode=self.delta_action_mode,
                delta_quat_mode=self.delta_quat_mode,
            )

        should_log_action_trace = self._debug_action_stats_remaining > 0
        if should_log_action_trace:
            self._debug_action_stats_remaining -= 1
            self._log_action_trace(
                current_eef_state_raw=current_eef_state_raw,
                input_state_tensor=input_state_tensor,
                normalized_actions=normalized_actions,
                raw_actions=raw_actions,
                unnormalized_model_actions=unnormalized_model_actions,
            )

        predicted_horizon = raw_actions.shape[1]
        if self.n_action_steps > predicted_horizon:
            raise ValueError(
                "Requested `n_action_steps` exceeds model prediction horizon: "
                f"n_action_steps={self.n_action_steps}, predicted_horizon={predicted_horizon}."
            )

        action_slice = slice(0, self.n_action_steps)
        if raw_actions.shape[-1] != self._eef_layout["total_dim"]:
            raise ValueError(f"RoboCasa EEF inference expects 23D actions, got {raw_actions.shape}.")
        layout = self._eef_layout
        left_eef_pos = raw_actions[:, action_slice, layout["left_pos"]]
        right_eef_pos = raw_actions[:, action_slice, layout["right_pos"]]
        left_eef_rot_raw = raw_actions[:, action_slice, layout["left_rot"]]
        right_eef_rot_raw = raw_actions[:, action_slice, layout["right_rot"]]
        left_eef_quat = rot6d_to_quaternion_xyzw(left_eef_rot_raw)
        right_eef_quat = rot6d_to_quaternion_xyzw(right_eef_rot_raw)

        left_gripper = raw_actions[:, action_slice, layout["left_hand"]]
        right_gripper = raw_actions[:, action_slice, layout["right_hand"]]
        left_hand = gripper_1d_to_hand_6d(left_gripper)
        right_hand = gripper_1d_to_hand_6d(right_gripper)
        waist = raw_actions[:, action_slice, layout["waist"]]

        current_left_arm = np.asarray(state["left_arm"][:, -1, :], dtype=np.float32)
        current_right_arm = np.asarray(state["right_arm"][:, -1, :], dtype=np.float32)
        left_arm_joints, right_arm_joints = self._solve_eef_action_chunk_with_ik(
            left_eef_pos=left_eef_pos,
            left_eef_quat=left_eef_quat,
            right_eef_pos=right_eef_pos,
            right_eef_quat=right_eef_quat,
            left_hand=left_hand,
            right_hand=right_hand,
            waist=waist,
            current_left_arm=current_left_arm,
            current_right_arm=current_right_arm,
        )
        raw_action = {
            "action.left_arm": left_arm_joints,
            "action.right_arm": right_arm_joints,
            "action.left_hand": left_hand,
            "action.right_hand": right_hand,
            "action.waist": waist,
        }
        if should_log_action_trace:
            self._log_action_trace(
                current_eef_state_raw=current_eef_state_raw,
                input_state_tensor=input_state_tensor,
                normalized_actions=normalized_actions,
                raw_actions=raw_actions,
                unnormalized_model_actions=unnormalized_model_actions,
                raw_action=raw_action,
            )
        return {"actions": raw_action}

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_mode: str = "min_max",
    ) -> np.ndarray:
        """Undo action normalization using either min/max or q01/q99 stats."""
        if normalization_mode == "q99":
            action_high = np.array(action_norm_stats.get("q99", action_norm_stats.get("max")), dtype=np.float32)
            action_low = np.array(action_norm_stats.get("q01", action_norm_stats.get("min")), dtype=np.float32)
        else:
            action_high = np.array(action_norm_stats.get("max"), dtype=np.float32)
            action_low = np.array(action_norm_stats.get("min"), dtype=np.float32)

        mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool)), dtype=bool)
        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low + EPS) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def _load_checkpoint_config(checkpoint_path: str) -> dict:
        """Read checkpoint config.yaml through the shared config loader."""
        try:
            config, _ = read_mode_config(checkpoint_path, allow_missing_norm_stats=True)
        except (AssertionError, FileNotFoundError, OSError, TypeError, ValueError) as error:
            print(f"WARNING: Failed to inspect checkpoint config at {checkpoint_path}: {error}")
            return {}
        return config if isinstance(config, dict) else {}

    @staticmethod
    def _detect_checkpoint_normalization_mode(checkpoint_path: str) -> str:
        """Infer whether the checkpoint used `min_max` or `q99` normalization."""
        config = PolicyWarper._load_checkpoint_config(checkpoint_path)
        vla_data = config.get("datasets", {}).get("vla_data", {}) if isinstance(config, dict) else {}
        trans_mode = str(vla_data.get("trans_norm_mode", "min_max"))
        rot_mode = str(vla_data.get("rot_norm_mode", trans_mode))
        gripper_mode = str(vla_data.get("gripper_norm_mode", trans_mode))
        modes = [trans_mode, rot_mode, gripper_mode]
        detected_mode = modes[0]
        if len(set(modes)) > 1:
            print(f"WARNING: Inconsistent normalization modes in config: {modes}, using {detected_mode}")
        if detected_mode not in {"min_max", "q99"}:
            print(f"WARNING: Unknown normalization mode {detected_mode!r}, falling back to 'min_max'")
            return "min_max"
        return detected_mode

    @staticmethod
    def _inspect_checkpoint_language_config(checkpoint_path: str) -> dict:
        """Infer eval-side language normalization from the checkpoint's training data config."""
        config = PolicyWarper._load_checkpoint_config(checkpoint_path)
        language_source = _resolve_checkpoint_language_source(config)
        vla_data_cfg = _get_checkpoint_vla_data_config(config)
        explicit_eval_strip = vla_data_cfg.get("eval_strip_robocasa_waist_language_prefix")
        if explicit_eval_strip is not None:
            try:
                strip_waist_prefix = _resolve_boolean_like(explicit_eval_strip, default=False)
            except TypeError as error:
                print(
                    "WARNING: Invalid checkpoint eval_strip_robocasa_waist_language_prefix="
                    f"{explicit_eval_strip!r}: {error}; using auto mode."
                )
                strip_waist_prefix = language_source == "episode_remarks"
        else:
            strip_waist_prefix = language_source == "episode_remarks"
        return {
            "language_source": language_source,
            "strip_waist_prefix": strip_waist_prefix,
        }

    @staticmethod
    def _inspect_checkpoint_state_config(checkpoint_path: str) -> dict:
        """Inspect checkpoint config and determine state-input behavior."""
        config = PolicyWarper._load_checkpoint_config(checkpoint_path)
        framework_cfg = config.get("framework", {}) if isinstance(config, dict) else {}
        action_model_cfg = framework_cfg.get("action_model", {}) if isinstance(framework_cfg, dict) else {}
        framework_name = str(framework_cfg.get("name", "unknown"))
        state_dim = int(action_model_cfg.get("state_dim") or 0)
        legacy_state_cfg = framework_cfg.get("state_encoder", {}) if isinstance(framework_cfg, dict) else {}
        vla_data_cfg = config.get("datasets", {}).get("vla_data", {}) if isinstance(config, dict) else {}

        if framework_name == "QwenGR00T":
            uses_state_from_model_cfg = state_dim > 0
        else:
            state_injection_mode = str(action_model_cfg.get("state_injection_mode", "disabled")).lower()
            state_prompt_mode = str(action_model_cfg.get("state_prompt_mode", "disabled")).lower()
            uses_state_from_model_cfg = any(
                [
                    bool(action_model_cfg.get("enable_state_input", False)),
                    bool(action_model_cfg.get("enable_state_timestep_conditioning", False)),
                    bool(action_model_cfg.get("discrete_state_input", False)),
                    state_injection_mode != "disabled",
                    state_prompt_mode != "disabled",
                    bool(legacy_state_cfg.get("enable_state_input", False)),
                ]
            )

        if isinstance(vla_data_cfg, dict) and "include_state" in vla_data_cfg:
            state_supported = bool(vla_data_cfg.get("include_state", False)) and uses_state_from_model_cfg
        else:
            state_supported = uses_state_from_model_cfg

        state_input_format = "raw"
        if framework_name.startswith("ABot") and state_dim == 58:
            state_supported = True
            state_input_format = "abot_joint_sincos_58d"
        elif not state_supported:
            state_input_format = "none"

        print(f"*** Checkpoint state input support: {state_supported} ***")
        return {
            "framework_name": framework_name,
            "state_dim": state_dim,
            "state_supported": state_supported,
            "state_input_format": state_input_format,
        }

    @staticmethod
    def _inspect_checkpoint_delta_config(checkpoint_path: str) -> dict:
        """Inspect checkpoint config and infer delta-action settings."""
        config = PolicyWarper._load_checkpoint_config(checkpoint_path)
        vla_data_cfg = config.get("datasets", {}).get("vla_data", {}) if isinstance(config, dict) else {}
        delta_action_mode = vla_data_cfg.get("delta_action_mode") if isinstance(vla_data_cfg, dict) else None
        delta_quat_mode = vla_data_cfg.get("delta_quat_mode") if isinstance(vla_data_cfg, dict) else None
        return {
            "delta_action_mode": None if delta_action_mode is None else str(delta_action_mode),
            "delta_quat_mode": None if delta_quat_mode is None else str(delta_quat_mode),
        }

    @staticmethod
    def _inspect_checkpoint_urdf_config(checkpoint_path: str) -> dict:
        """Inspect checkpoint config and determine whether model-side URDF conditioning is enabled."""
        config = PolicyWarper._load_checkpoint_config(checkpoint_path)
        framework_cfg = config.get("framework", {}) if isinstance(config, dict) else {}
        action_model_cfg = framework_cfg.get("action_model", {}) if isinstance(framework_cfg, dict) else {}
        urdf_enabled = bool(action_model_cfg.get("enable_urdf_input", False))
        return {"urdf_enabled": urdf_enabled}

    @staticmethod
    def _resolve_eval_robot_name(
        checkpoint_path: str,
        *,
        urdf_enabled: bool,
    ) -> str | None:
        """Resolve one canonical robot name for URDF-conditioned eval checkpoints."""
        if not urdf_enabled:
            return None

        config = PolicyWarper._load_checkpoint_config(checkpoint_path)
        vla_data_cfg = config.get("datasets", {}).get("vla_data", {}) if isinstance(config, dict) else {}
        candidate_identifiers = [
            vla_data_cfg.get("robot_name"),
            vla_data_cfg.get("robot_type"),
            vla_data_cfg.get("manifest_robot_type"),
            config.get("run_id") if isinstance(config, dict) else None,
            config.get("output_dir") if isinstance(config, dict) else None,
            checkpoint_path,
        ]

        for candidate_identifier in candidate_identifiers:
            if not candidate_identifier:
                continue
            candidate_text = str(candidate_identifier)
            try:
                return resolve_robot_name(candidate_text)
            except KeyError:
                normalized_text = candidate_text.lower()
                if "gr1" in normalized_text:
                    return "GR1"
                if "r1lite" in normalized_text or "galaxea" in normalized_text:
                    return "R1Lite"
                if "agibot" in normalized_text or "genie" in normalized_text:
                    return "G1"

        raise ValueError(
            "Checkpoint enables `framework.action_model.enable_urdf_input=true`, but eval could not infer "
            "robot_name from the checkpoint metadata."
        )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize an input image unless raw-resolution eval is enabled."""
        if self.image_size is None:
            return image
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """Resolve which dataset statistics entry to use for de-normalization."""
        if unnorm_key is None:
            if "gr1" in norm_stats:
                return "gr1"
            assert len(norm_stats) == 1, (
                "Your model was trained on more than one dataset, please pass an `unnorm_key` from "
                f"{norm_stats.keys()} to choose the statistics used for un-normalizing actions."
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            "The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def _normalize_values_with_stats(self, values: np.ndarray, stats: dict, *, name: str) -> np.ndarray:
        if self.normalization_mode == "q99":
            low = stats.get("q01")
            high = stats.get("q99")
            if low is None or high is None:
                print(f"WARNING: q01/q99 {name} stats missing, falling back to min/max")
                low = stats.get("min")
                high = stats.get("max")
        else:
            low = stats.get("min")
            high = stats.get("max")

        if low is None or high is None:
            return values

        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        if low.shape[-1] != values.shape[-1]:
            raise ValueError(f"{name} stats dimension mismatch: stats={low.shape}, values={values.shape}.")

        value_range = np.where((high - low) < EPS, EPS, high - low)
        return 2 * (values - low) / value_range - 1

    def normalize_eef_state(self, eef_state: np.ndarray) -> np.ndarray:
        """Normalize EEF state to `[-1, 1]` using checkpoint or single-task stats."""
        if self.state_norm_stats is None:
            return eef_state

        if self.normalization_mode == "q99":
            state_low = self.state_norm_stats.get("q01")
            state_high = self.state_norm_stats.get("q99")
            if state_low is None or state_high is None:
                print("WARNING: q01/q99 state stats missing, falling back to min/max for EEF state normalization")
                state_low = self.state_norm_stats.get("min")
                state_high = self.state_norm_stats.get("max")
        else:
            state_low = self.state_norm_stats.get("min")
            state_high = self.state_norm_stats.get("max")

        if state_low is None or state_high is None:
            return eef_state

        state_low = np.asarray(state_low, dtype=np.float32)
        state_high = np.asarray(state_high, dtype=np.float32)
        if state_low.shape[-1] != eef_state.shape[-1]:
            raise ValueError(f"State stats dimension mismatch: stats={state_low.shape}, eef_state={eef_state.shape}.")

        range_val = np.where((state_high - state_low) < EPS, EPS, state_high - state_low)
        normalized = 2 * (eef_state - state_low) / range_val - 1
        return np.clip(normalized, -1.0, 1.0)
