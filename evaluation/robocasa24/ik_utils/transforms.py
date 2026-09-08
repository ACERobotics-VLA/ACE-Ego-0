"""
Coordinate transform utilities for EEF pose manipulation.

Adapted from teleoperation project: teleo.robots.sim_gr1_eef_replay
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    frame_pos: np.ndarray,
    frame_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a pose from local frame to parent frame.

    Computes: P_parent = T_frame * P_local

    Args:
        pos: Position in local frame [x, y, z]
        quat: Quaternion in local frame [x, y, z, w] (xyzw format)
        frame_pos: Frame origin position in parent frame [x, y, z]
        frame_quat: Frame orientation in parent frame [x, y, z, w] (xyzw format)

    Returns:
        Tuple of (position, quaternion) in parent frame, quaternion in xyzw format.
    """
    pos = np.asarray(pos, dtype=np.float64).flatten()
    quat = np.asarray(quat, dtype=np.float64).flatten()
    frame_pos = np.asarray(frame_pos, dtype=np.float64).flatten()
    frame_quat = np.asarray(frame_quat, dtype=np.float64).flatten()

    # Create rotation objects (scipy uses xyzw format)
    rot_frame = R.from_quat(frame_quat)
    rot_local = R.from_quat(quat)

    # Transform position: p_parent = frame_pos + rot_frame * p_local
    pos_parent = frame_pos + rot_frame.apply(pos)

    # Transform orientation: rot_parent = rot_frame * rot_local
    rot_parent = rot_frame * rot_local
    quat_parent = rot_parent.as_quat()  # Returns xyzw

    return pos_parent, quat_parent


def inverse_transform_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    frame_pos: np.ndarray,
    frame_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a pose from parent frame to local frame (inverse transform).

    Computes: P_local = T_frame^-1 * P_parent

    Args:
        pos: Position in parent frame [x, y, z]
        quat: Quaternion in parent frame [x, y, z, w] (xyzw format)
        frame_pos: Frame origin position in parent frame [x, y, z]
        frame_quat: Frame orientation in parent frame [x, y, z, w] (xyzw format)

    Returns:
        Tuple of (position, quaternion) in local frame, quaternion in xyzw format.
    """
    pos = np.asarray(pos, dtype=np.float64).flatten()
    quat = np.asarray(quat, dtype=np.float64).flatten()
    frame_pos = np.asarray(frame_pos, dtype=np.float64).flatten()
    frame_quat = np.asarray(frame_quat, dtype=np.float64).flatten()

    # Create rotation objects
    rot_frame = R.from_quat(frame_quat)
    rot_parent = R.from_quat(quat)

    # Inverse of frame rotation
    rot_frame_inv = rot_frame.inv()

    # Transform position: p_local = rot_frame^-1 * (p_parent - frame_pos)
    pos_local = rot_frame_inv.apply(pos - frame_pos)

    # Transform orientation: rot_local = rot_frame^-1 * rot_parent
    rot_local = rot_frame_inv * rot_parent
    quat_local = rot_local.as_quat()  # Returns xyzw

    return pos_local, quat_local


def convert_pose_xyzw_to_wxyz(pos: np.ndarray, quat: np.ndarray, arm: str = "right") -> np.ndarray:
    """Convert pose from separate pos/quat to 7D [wxyz, xyz] format.

    This function also applies a 180° rotation correction to account for the
    coordinate frame difference between MuJoCo's robot0_right_eef/robot0_left_eef
    and URDF's r_palm/l_palm.

    MuJoCo EEF frames vs URDF palm frames:
        - MuJoCo right_eef: rotated -90° around Z relative to r_hand_pitch
        - URDF r_palm: rotated +90° around Z relative to r_hand_pitch
        - Difference: 180° around Z axis

        - MuJoCo left_eef: rotated +90° around Z relative to l_hand_pitch
        - URDF l_palm: rotated -90° around Z relative to l_hand_pitch
        - Difference: 180° around Z axis

    Args:
        pos: 3D position [x, y, z]
        quat: 4D quaternion in xyzw format [qx, qy, qz, qw]
        arm: "right" or "left" arm

    Returns:
        7D pose in [qw, qx, qy, qz, x, y, z] format for pyroki.
    """
    pos = np.asarray(pos, dtype=np.float32).flatten()
    quat = np.asarray(quat, dtype=np.float32).flatten()

    # Input quaternion is in xyzw format: [qx, qy, qz, qw]
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    # Apply 180° rotation around Z axis to correct for MuJoCo EEF vs URDF palm frame difference
    rot_original = R.from_quat([qx, qy, qz, qw])  # scipy uses xyzw
    rot_180_z = R.from_euler("z", 180, degrees=True)
    rot_corrected = rot_original * rot_180_z

    # Convert back to quaternion (scipy returns xyzw)
    quat_corrected = rot_corrected.as_quat()
    qx, qy, qz, qw = quat_corrected

    return np.array([qw, qx, qy, qz, pos[0], pos[1], pos[2]], dtype=np.float32)


# Hand joint control ranges (from XML ctrlrange)
HAND_JOINT_CTRL_RANGES = (
    (0, 1.74),  # pinky_intermediate
    (0, 1.57),  # pinky_proximal
    (0, 1.74),  # ring_intermediate
    (0, 1.57),  # ring_proximal
    (0, 1.74),  # middle_intermediate
    (0, 1.57),  # middle_proximal
    (0, 1.74),  # index_intermediate
    (0, 1.57),  # index_proximal
    (0, 1.23),  # thumb_distal
    (0, 1.22),  # thumb_proximal_pitch
    (0, 1.74),  # thumb_proximal_yaw
)


def format_hand_action(action: np.ndarray) -> np.ndarray:
    """Convert 6 DOF hand action to 11 joint positions.

    Args:
        action: 6D hand action

    Returns:
        11D joint positions (clipped to valid range)
    """
    if len(action) != 6:
        raise ValueError(f"Expected 6D hand action, got {len(action)}")

    # Expand 6D to 11D using the same indices as robocasa
    indices = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
    expanded = action[indices]

    # Clip to joint control ranges
    joint_positions = np.zeros(11)
    for i, (ctrl_min, ctrl_max) in enumerate(HAND_JOINT_CTRL_RANGES):
        joint_positions[i] = np.clip(expanded[i], ctrl_min, ctrl_max)

    return joint_positions
