"""Fourier hand and scalar gripper conversion helpers used by RoboCasa eval."""

import numpy as np

HAND_OPEN = np.array([-1.5, -1.5, -1.5, -1.5, -3.0, 3.0], dtype=np.float32)
HAND_CLOSE = np.array([1.5, 1.5, 1.5, 1.5, 3.0, 3.0], dtype=np.float32)
FINGER_MIN = HAND_OPEN[:4]
FINGER_MAX = HAND_CLOSE[:4]


def hand_6d_to_gripper_1d(hand_action: np.ndarray) -> np.ndarray:
    """Convert Fourier's 6D hand pose to a scalar gripper value in [0, 1]."""
    values = np.asarray(hand_action, dtype=np.float32)
    fingers = values[..., :4]
    normalized = np.clip((fingers - FINGER_MIN) / (FINGER_MAX - FINGER_MIN), 0.0, 1.0)
    return np.mean(normalized, axis=-1, keepdims=True).astype(np.float32)


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
