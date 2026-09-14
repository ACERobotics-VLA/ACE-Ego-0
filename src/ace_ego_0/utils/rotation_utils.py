"""Common23 rotation helpers used by ACE-Ego-0 training and RoboCasa evaluation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

COMMON23_ACTION_DIM = 23
COMMON23_ROT6D_RANGES: tuple[tuple[int, int], ...] = ((3, 9), (12, 18))
COMMON23_XYZ_RANGES: tuple[tuple[int, int], ...] = ((0, 3), (9, 12))
COMMON23_GRIPPER_INDICES: tuple[int, ...] = (18, 19)


def _require_common23_action_dim(action_dim: int) -> None:
    """Reject action layouts that are not the public Common23 23D space."""
    if action_dim != COMMON23_ACTION_DIM:
        raise ValueError(f"Public ACE-Ego-0 only supports Common23 action_dim=23, got {action_dim}.")


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices."""
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {quaternions.shape}.")
    q = quaternions / (torch.linalg.norm(quaternions, dim=-1, keepdim=True) + 1e-8)
    r, i, j, k = torch.unbind(q, dim=-1)
    two_s = 2.0
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape((*quaternions.shape[:-1], 3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to wxyz quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-8)) / 2.0
    qx = torch.sign(m21 - m12) * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) / 2.0
    qy = torch.sign(m02 - m20) * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8)) / 2.0
    qz = torch.sign(m10 - m01) * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8)) / 2.0
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return q / (torch.linalg.norm(q, dim=-1, keepdim=True) + 1e-8)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rot6d to rotation matrices using the dataset column-major convention.

    The RoboCasa EEF datasets store rot6d as the first two matrix columns
    flattened as ``[R[:, 0], R[:, 1]]``.
    """
    col1 = rot6d[..., :3]
    col1 = col1 / (col1.norm(dim=-1, keepdim=True) + 1e-8)
    col2 = rot6d[..., 3:6]
    dot = (col1 * col2).sum(dim=-1, keepdim=True)
    col2 = col2 - dot * col1
    col2 = col2 / (col2.norm(dim=-1, keepdim=True) + 1e-8)
    col3 = torch.cross(col1, col2, dim=-1)
    return torch.stack([col1, col2, col3], dim=-1)


def matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to rot6d using the dataset column-major convention."""
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _normalize_xyzw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize xyzw quaternions to unit length."""
    return quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp(min=1e-8)


def _canonicalize_xyzw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic xyzw quaternion sign by enforcing non-negative w."""
    sign = torch.where(quaternion[..., 3:4] < 0, -1.0, 1.0)
    return quaternion * sign


def _align_xyzw_quaternion_sign(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Flip target xyzw quaternions when needed so reference·target stays non-negative."""
    dot = (reference * target).sum(dim=-1, keepdim=True)
    sign = torch.where(dot < 0, -1.0, 1.0)
    return target * sign


def _xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternions to wxyz layout."""
    return torch.cat([quaternion[..., 3:4], quaternion[..., :3]], dim=-1)


def _wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to xyzw layout."""
    return torch.cat([quaternion[..., 1:], quaternion[..., 0:1]], dim=-1)


def xyzw_quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert xyzw quaternions to rotation matrices."""
    normalized_quaternion = _normalize_xyzw_quaternion(quaternion)
    return quaternion_to_matrix(_xyzw_to_wxyz(normalized_quaternion))


def matrix_to_xyzw_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to canonicalized xyzw quaternions."""
    quaternion = _wxyz_to_xyzw(matrix_to_quaternion(matrix))
    quaternion = _normalize_xyzw_quaternion(quaternion)
    return _canonicalize_xyzw_quaternion(quaternion)


def apply_quaternion_delta(state_quaternion: torch.Tensor, delta_quaternion: torch.Tensor) -> torch.Tensor:
    """Apply an xyzw quaternion delta to a state quaternion."""
    normalized_state = _normalize_xyzw_quaternion(state_quaternion)
    normalized_delta = _normalize_xyzw_quaternion(delta_quaternion)

    state_matrix = xyzw_quaternion_to_matrix(normalized_state)
    delta_matrix = xyzw_quaternion_to_matrix(normalized_delta)
    action_matrix = torch.matmul(state_matrix, delta_matrix)
    action_quaternion = matrix_to_xyzw_quaternion(action_matrix)
    return _align_xyzw_quaternion_sign(normalized_state, action_quaternion)


def apply_quaternion_delta_np(state_quaternion: np.ndarray, delta_quaternion: np.ndarray) -> np.ndarray:
    """Numpy wrapper around ``apply_quaternion_delta``."""
    state_torch = torch.from_numpy(state_quaternion).float()
    delta_torch = torch.from_numpy(delta_quaternion).float()
    action_torch = apply_quaternion_delta(state_torch, delta_torch)
    return action_torch.numpy().astype(delta_quaternion.dtype)


def identify_rot6d_indices(action_dim: int) -> List[Tuple[int, int]]:
    """Return Common23 rot6d ranges."""
    _require_common23_action_dim(action_dim)
    return list(COMMON23_ROT6D_RANGES)


def identify_quaternion_indices(action_dim: int) -> List[Tuple[int, int]]:
    """Common23 uses rot6d, so there are no quaternion ranges."""
    _require_common23_action_dim(action_dim)
    return []


def identify_xyz_indices(action_dim: int) -> List[Tuple[int, int]]:
    """Return Common23 translation ranges."""
    _require_common23_action_dim(action_dim)
    return list(COMMON23_XYZ_RANGES)


def identify_gripper_indices(action_dim: int) -> List[int]:
    """Return Common23 gripper indices."""
    _require_common23_action_dim(action_dim)
    return list(COMMON23_GRIPPER_INDICES)


def identify_joint_ranges(action_dim: int) -> List[Tuple[int, int]]:
    """Common23 has no separate arm-joint ranges; waist stays on the default norm mode."""
    _require_common23_action_dim(action_dim)
    return []
