"""
Shared rotation conversion utilities.

This module provides optional pytorch3d-backed conversions with pure torch
fallbacks, so runtime can use pytorch3d when available without hard dependency.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import pytorch3d.transforms as pt  # type: ignore
except Exception:
    pt = None


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError(f"Unknown axis letter: {letter}")


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Unknown axis letter: {axis}")

    return torch.stack(R, dim=-1).reshape((*angle.shape, 3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if pt is not None:
        return pt.euler_angles_to_matrix(euler_angles, convention=convention)
    if euler_angles.shape[-1] != 3:
        raise ValueError(f"Invalid input euler angles shape {euler_angles.shape}.")
    if len(convention) != 3 or len(set(convention)) != 3:
        raise ValueError(f"Invalid convention {convention}.")
    matrices = [_axis_angle_rotation(c, euler_angles[..., i]) for i, c in enumerate(convention)]
    return matrices[0] @ matrices[1] @ matrices[2]


def _angle_from_tan(axis: str, other_axis: str, data: torch.Tensor, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
    i1 = _index_from_letter(axis)
    i2 = _index_from_letter(other_axis)
    if horizontal:
        i1, i2 = i2, i1
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    if pt is not None:
        return pt.matrix_to_euler_angles(matrix, convention=convention)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    if len(convention) != 3 or len(set(convention)) != 3:
        raise ValueError(f"Invalid convention {convention}.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        sign = -1.0 if (i0 - i2) in (-1, 2) else 1.0
        central_angle = torch.asin(matrix[..., i0, i2] * sign)
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, dim=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    if pt is not None:
        return pt.quaternion_to_matrix(quaternions)
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
    if pt is not None:
        return pt.matrix_to_quaternion(matrix)
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


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Invalid axis_angle shape {axis_angle.shape}.")
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4:
        raise ValueError(f"Invalid quaternion shape {quaternion.shape}.")
    q = quaternion / (torch.linalg.norm(quaternion, dim=-1, keepdim=True) + 1e-8)
    w = torch.clamp(q[..., :1], -1.0, 1.0)
    xyz = q[..., 1:]
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-8))
    axis = xyz / (sin_half + 1e-8)
    return axis * angle


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    if pt is not None:
        return pt.axis_angle_to_matrix(axis_angle)
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    if pt is not None:
        return pt.matrix_to_axis_angle(matrix)
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert rot6d to rotation matrices using the dataset column-major convention.

    The RoboCasa EEF datasets store rot6d as the first two matrix columns
    flattened as ``[R[:, 0], R[:, 1]]``. We intentionally avoid delegating to
    pytorch3d here because its runtime convention differs from the dataset
    convention used by the training pipeline and evaluation assets.
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


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    return rot6d_to_matrix(rotation_6d)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix_to_rot6d(matrix)


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


def compute_quaternion_delta(action_quaternion: torch.Tensor, state_quaternion: torch.Tensor) -> torch.Tensor:
    """
    Compute xyzw quaternion deltas using SO(3) relative rotations.

    This computes the relative rotation in the state/body frame:
    `R_delta = R_state.T @ R_action`.

    Args:
        action_quaternion: Target rotations in xyzw format, shape (..., 4).
        state_quaternion: Reference rotations in xyzw format, shape (..., 4).

    Returns:
        Quaternion deltas in xyzw format with canonicalized sign.
    """
    normalized_action = _normalize_xyzw_quaternion(action_quaternion)
    normalized_state = _normalize_xyzw_quaternion(state_quaternion)
    aligned_action = _align_xyzw_quaternion_sign(normalized_state, normalized_action)

    action_matrix = xyzw_quaternion_to_matrix(aligned_action)
    state_matrix = xyzw_quaternion_to_matrix(normalized_state)
    delta_matrix = torch.matmul(state_matrix.transpose(-2, -1), action_matrix)
    return matrix_to_xyzw_quaternion(delta_matrix)


def apply_quaternion_delta(state_quaternion: torch.Tensor, delta_quaternion: torch.Tensor) -> torch.Tensor:
    """
    Apply an xyzw quaternion delta to a state quaternion.

    Args:
        state_quaternion: Reference rotations in xyzw format, shape (..., 4).
        delta_quaternion: Relative rotations in xyzw format, shape (..., 4).

    Returns:
        Reconstructed action rotations in canonicalized xyzw format.
    """
    normalized_state = _normalize_xyzw_quaternion(state_quaternion)
    normalized_delta = _normalize_xyzw_quaternion(delta_quaternion)

    state_matrix = xyzw_quaternion_to_matrix(normalized_state)
    delta_matrix = xyzw_quaternion_to_matrix(normalized_delta)
    action_matrix = torch.matmul(state_matrix, delta_matrix)
    action_quaternion = matrix_to_xyzw_quaternion(action_matrix)
    return _align_xyzw_quaternion_sign(normalized_state, action_quaternion)


ROTATION_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "axis_angle_to_matrix": axis_angle_to_matrix,
    "matrix_to_axis_angle": matrix_to_axis_angle,
    "quaternion_to_matrix": quaternion_to_matrix,
    "matrix_to_quaternion": matrix_to_quaternion,
    "rotation_6d_to_matrix": rotation_6d_to_matrix,
    "matrix_to_rotation_6d": matrix_to_rotation_6d,
    "euler_angles_to_matrix": euler_angles_to_matrix,
    "matrix_to_euler_angles": matrix_to_euler_angles,
}


# ============================================================================
# ACE-Ego-0 rot6d utilities.
# ============================================================================


def _get_uas_indices() -> dict:
    """
    Get unified action-space indices.

    Prefer importing from UNIFIED_ACTION_SPACE to keep a single source of truth.
    If importing fails due to optional runtime dependencies (e.g. accelerate),
    fall back to canonical 80D defaults for tool/test-time usability.
    """
    try:
        from ace_ego_0.dataloader.gr00t_lerobot.unified_action_space import UNIFIED_ACTION_SPACE as UAS

        return {
            "right_rot6d": (UAS.RIGHT_EE_ROT6D_1, UAS.RIGHT_EE_ROT6D_6 + 1),
            "left_rot6d": (UAS.LEFT_EE_ROT6D_1, UAS.LEFT_EE_ROT6D_6 + 1),
            "right_xyz": (UAS.RIGHT_EE_X, UAS.RIGHT_EE_Z + 1),
            "left_xyz": (UAS.LEFT_EE_X, UAS.LEFT_EE_Z + 1),
            "right_gripper": UAS.RIGHT_GRIPPER,
            "left_gripper": UAS.LEFT_GRIPPER,
            "right_joint": (UAS.RIGHT_ARM_JOINT1, UAS.RIGHT_ARM_JOINT6 + 1),
            "left_joint": (UAS.LEFT_ARM_JOINT1, UAS.LEFT_ARM_JOINT6 + 1),
        }
    except Exception:
        return {
            "right_rot6d": (10, 16),
            "left_rot6d": (39, 45),
            "right_xyz": (7, 10),
            "left_xyz": (36, 39),
            "right_gripper": 16,
            "left_gripper": 45,
            "right_joint": (0, 6),
            "left_joint": (29, 35),
        }


def compute_rot6d_delta(action_rot6d: torch.Tensor, state_rot6d: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation delta in rot6d space using proper rotation mathematics.

    This computes the relative rotation: R_delta = R_state.T @ R_action
    which represents the rotation needed to go from state to action.

    Args:
        action_rot6d: (T, 6) or (B, T, 6) target rotations
        state_rot6d: (1, 6) or (B, 1, 6) reference rotation (first state)

    Returns:
        (T, 6) or (B, T, 6) rotation deltas in rot6d space
    """
    # Convert rot6d to rotation matrices
    R_action = rot6d_to_matrix(action_rot6d)  # (..., 3, 3)
    R_state = rot6d_to_matrix(state_rot6d)  # (..., 3, 3)

    # Compute relative rotation: R_delta = R_state.T @ R_action
    R_state_T = R_state.transpose(-2, -1)
    R_delta = torch.matmul(R_state_T, R_action)

    # Convert back to rot6d
    delta_rot6d = matrix_to_rot6d(R_delta)

    return delta_rot6d


def apply_rotlta(state_rot6d: torch.Tensor, delta_rot6d: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation delta to state rotation.

    This computes: R_new = R_state @ R_delta

    Args:
        state_rot6d: (..., 6) current rotation state
        delta_rot6d: (..., 6) rotation delta to apply

    Returns:
        (..., 6) new rotation state
    """
    # Convert to matrices
    R_state = rot6d_to_matrix(state_rot6d)
    R_delta = rot6d_to_matrix(delta_rot6d)

    # Apply delta: R_new = R_state @ R_delta
    R_new = torch.matmul(R_state, R_delta)

    # Convert back to rot6d
    return matrix_to_rot6d(R_new)


def compute_rot6d_geodesic_loss(pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic loss for rot6d representations.

    This computes the geodesic distance on SO(3) manifold, which is the
    mathematically correct way to measure rotation errors.

    Args:
        pred_rot6d: (..., 6) predicted rotations
        target_rot6d: (..., 6) target rotations

    Returns:
        (...,) geodesic distances normalized by π
    """
    # Convert rot6d to rotation matrices
    R_pred = rot6d_to_matrix(pred_rot6d)  # (..., 3, 3)
    R_target = rot6d_to_matrix(target_rot6d)  # (..., 3, 3)

    # Compute relative rotation: R_rel = R_pred.T @ R_target
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_target)

    # Compute trace and geodesic distance
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)  # (...,)

    # Geodesic distance: θ = arccos((trace - 1) / 2)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)  # Numerical stability
    theta = torch.acos(cos_theta)

    # Normalize by π to get values in [0, 1]
    return theta / torch.pi


def identify_rot6d_indices(action_dim: int, dataset_name: Optional[str] = None) -> List[Tuple[int, int]]:
    """
    Identify which indices contain rot6d data for different datasets.
    Uses UnifiedActionSpaceConfig constants for the 80D unified space.

    Args:
        action_dim: Total action dimension
        dataset_name: Name of the dataset (optional, for future extensibility)

    Returns:
        List of (start_idx, end_idx) tuples for rot6d dimensions (end_idx exclusive)
    """
    rot6d_indices = []
    idx = _get_uas_indices()

    if action_dim == 80:
        rot6d_indices = [
            idx["right_rot6d"],
            idx["left_rot6d"],
        ]
    elif action_dim == 34:
        pass
    elif action_dim == 20:
        # raw20: left_xyz(3), left_rot6d(6), left_gripper(1), right_xyz(3), right_rot6d(6), right_gripper(1)
        rot6d_indices = [(3, 9), (13, 19)]
    elif action_dim == 23:
        # common23: left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6), left_gripper(1), right_gripper(1), waist(3)
        rot6d_indices = [(3, 9), (12, 18)]

    return rot6d_indices


def identify_quaternion_indices(action_dim: int) -> List[Tuple[int, int]]:
    """Identify xyzw quaternion ranges for action spaces that use quaternions directly."""
    if action_dim == 19:
        return [(3, 7), (10, 14)]
    return []


def identify_xyz_indices(action_dim: int) -> List[Tuple[int, int]]:
    """Identify xyz (translation) indices in the unified action space."""
    idx = _get_uas_indices()

    if action_dim == 80:
        return [
            idx["right_xyz"],
            idx["left_xyz"],
        ]
    if action_dim == 19:
        return [(0, 3), (7, 10)]
    if action_dim == 20:
        # raw20: left_xyz(3), left_rot6d(6), left_gripper(1), right_xyz(3), right_rot6d(6), right_gripper(1)
        return [(0, 3), (10, 13)]
    if action_dim == 23:
        # common23: left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6), left_gripper(1), right_gripper(1), waist(3)
        return [(0, 3), (9, 12)]
    return []


def identify_gripper_indices(action_dim: int) -> List[int]:
    """Identify gripper indices in the unified action space."""
    idx = _get_uas_indices()

    if action_dim == 80:
        return [idx["right_gripper"], idx["left_gripper"]]
    elif action_dim == 19:
        return [14, 15]
    elif action_dim == 14:
        # Galaxea 14D action space: gripper at indices 6 (left) and 13 (right)
        return [6, 13]
    elif action_dim == 16:
        # 16D action space: gripper at indices 8 (left) and 15 (right) - adjust if needed
        return [8, 15]
    elif action_dim == 20:
        # raw20: left_gripper at 9, right_gripper at 19
        return [9, 19]
    elif action_dim == 23:
        # common23: left_gripper at 18, right_gripper at 19
        return [18, 19]
    return []


def identify_joint_ranges(action_dim: int) -> List[Tuple[int, int]]:
    """Identify joint (arm) indices in the unified action space."""
    idx = _get_uas_indices()

    if action_dim == 80:
        return [
            idx["right_joint"],
            idx["left_joint"],
        ]
    if action_dim == 19:
        return [(16, 19)]
    return []


def compute_dual_arm_rot6d_delta(state_action: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
    """
    Compute delta for dual arm action sequence with rot6d rotations.

    This is a convenience function that handles the full dual-arm case,
    applying proper rotation mathematics to rot6d dimensions while using
    simple subtraction for other dimensions.

    Args:
        state_action: (..., action_dim) state action (typically first timestep)
        action_seq: (..., T, action_dim) action sequence

    Returns:
        (..., T, action_dim) delta actions
    """
    action_dim = action_seq.shape[-1]
    rot6d_indices = identify_rot6d_indices(action_dim)

    # Initialize delta with simple subtraction
    delta_seq = action_seq - state_action.unsqueeze(-2)  # Broadcast state to all timesteps

    # Apply proper rotation delta for rot6d dimensions
    for start_idx, end_idx in rot6d_indices:
        if start_idx < action_dim and end_idx <= action_dim:
            state_rot6d = state_action[..., start_idx:end_idx].unsqueeze(-2)  # Add time dim
            action_rot6d = action_seq[..., start_idx:end_idx]

            # Compute proper rotation delta
            delta_rot6d = compute_rot6d_delta(action_rot6d, state_rot6d)
            delta_seq[..., start_idx:end_idx] = delta_rot6d

    return delta_seq


def validate_rot6d(rot6d: torch.Tensor, tolerance: float = 1e-3) -> bool:
    """
    Validate that rot6d representation is valid.

    A valid rot6d should represent the first two columns of an orthogonal matrix.

    Args:
        rot6d: (..., 6) rot6d representation
        tolerance: Numerical tolerance for validation

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Convert to matrix and check if it's a valid rotation matrix
        R = rot6d_to_matrix(rot6d)

        # Check if it's orthogonal: R @ R.T should be identity
        should_be_identity = torch.matmul(R, R.transpose(-2, -1))
        identity = torch.eye(3, device=R.device, dtype=R.dtype)

        # Check orthogonality
        ortho_error = torch.abs(should_be_identity - identity).max()

        # Check determinant is 1 (proper rotation, not reflection)
        det = torch.det(R)
        det_error = torch.abs(det - 1.0).max()

        return ortho_error < tolerance and det_error < tolerance

    except Exception:
        return False


# Numpy versions for compatibility with existing code
def compute_rot6d_delta_np(action_rot6d: np.ndarray, state_rot6d: np.ndarray) -> np.ndarray:
    """Numpy version of compute_rot6d_delta for compatibility."""
    # Convert to torch, compute, convert back
    action_torch = torch.from_numpy(action_rot6d).float()
    state_torch = torch.from_numpy(state_rot6d).float()

    delta_torch = compute_rot6d_delta(action_torch, state_torch)
    return delta_torch.numpy().astype(action_rot6d.dtype)


def compute_quaternion_delta_np(action_quaternion: np.ndarray, state_quaternion: np.ndarray) -> np.ndarray:
    """Numpy version of compute_quaternion_delta for xyzw quaternions."""
    action_torch = torch.from_numpy(action_quaternion).float()
    state_torch = torch.from_numpy(state_quaternion).float()

    delta_torch = compute_quaternion_delta(action_torch, state_torch)
    return delta_torch.numpy().astype(action_quaternion.dtype)


def apply_quaternion_delta_np(state_quaternion: np.ndarray, delta_quaternion: np.ndarray) -> np.ndarray:
    """Numpy version of apply_quaternion_delta for xyzw quaternions."""
    state_torch = torch.from_numpy(state_quaternion).float()
    delta_torch = torch.from_numpy(delta_quaternion).float()

    action_torch = apply_quaternion_delta(state_torch, delta_torch)
    return action_torch.numpy().astype(delta_quaternion.dtype)


def compute_dual_arm_rot6d_delta_np(state_action: np.ndarray, action_seq: np.ndarray) -> np.ndarray:
    """Numpy version of compute_dual_arm_rot6d_delta for compatibility."""
    # Convert to torch, compute, convert back
    state_torch = torch.from_numpy(state_action).float()
    action_torch = torch.from_numpy(action_seq).float()

    delta_torch = compute_dual_arm_rot6d_delta(state_torch, action_torch)
    return delta_torch.numpy().astype(action_seq.dtype)
