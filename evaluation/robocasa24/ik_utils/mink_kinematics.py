"""
Inverse Kinematics utilities using MuJoCo + Mink.

This module provides the same public interface as kinematics.py so callers can
switch imports with minimal code changes.
"""

from __future__ import annotations

import logging
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import mink
import mujoco
import numpy as np
import numpy.typing as npt

try:
    from robosuite.utils import transform_utils as T
except ImportError:
    from robosuite.robosuite.utils import transform_utils as T

logger = logging.getLogger(__name__)


__all__ = [
    "IKUnreachableError",
    "InverseKinematicsConfig",
    "RobotKinematics",
]


FloatArray = npt.NDArray[np.float32]


class IKUnreachableError(Exception):
    """Raised when the IK target is unreachable."""


@dataclass(frozen=True)
class InverseKinematicsConfig:
    """Configuration parameters for the Mink IK solver."""

    learning_rate: float = 0.1
    max_iterations: int = 200
    pose_tolerance: float = 0.05
    unreachable_threshold: float = 0.2
    posture_cost: float = 1e-2
    position_cost: float = 1.0
    orientation_cost: float = 0.5
    solver: str = "quadprog"


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_xyzw = T.convert_quat(
        np.asarray(quat_wxyz, dtype=np.float64),
        to="xyzw",
    )
    return np.asarray(T.quat2mat(quat_xyzw), dtype=np.float64)


def _pose_wxyzxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_wxyz_to_rotmat(pose[:4])
    mat[:3, 3] = pose[4:]
    return mat


def _rotation_angle_rad(rot_src: np.ndarray, rot_tgt: np.ndarray) -> float:
    cos_theta = (np.trace(rot_src.T @ rot_tgt) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


# 解耦了MuJoCo在末端坐标系的转换方法
def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions in wxyz order."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _apply_local_z_rotation_wxyz(quat_wxyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply a local-frame Z rotation to a quaternion in wxyz order."""
    half_rad = np.deg2rad(angle_deg) * 0.5
    qz = np.array([np.cos(half_rad), 0.0, 0.0, np.sin(half_rad)], dtype=np.float64)
    out = _quat_mul_wxyz(np.asarray(quat_wxyz, dtype=np.float64), qz)
    norm = np.linalg.norm(out)
    if norm < 1e-12:
        return np.asarray(quat_wxyz, dtype=np.float64)
    return out / norm


def _create_meshless_urdf_copy(urdf_path: Path) -> Path:
    """Create a temporary URDF copy without visual/collision geometry.

    The original pyroki backend loads URDF with mesh loading disabled, so it can
    ignore missing STL assets completely. MuJoCo's URDF importer is stricter and
    tries to resolve mesh files during model compilation. For IK we only need the
    kinematic tree, joints, limits, and inertials, so stripping visual/collision
    blocks keeps the chain intact while avoiding mesh dependency failures.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link_elem in root.findall("link"):
        for child in list(link_elem):
            if child.tag in {"visual", "collision"}:
                link_elem.remove(child)

    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".urdf",
        prefix=f"{urdf_path.stem}_meshless_",
        delete=False,
        dir=urdf_path.parent,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def _resolve_end_effector_link_name(
    requested_link: str,
    available_links: list[str],
) -> str:
    """Resolve end-effector link names across URDF variants.

    Some GR1 URDF variants expose palm links as `l_palm/r_palm`, while others
    only provide `l_hand_pitch/r_hand_pitch` as the terminal arm body. Keep the
    public API unchanged and resolve internally.
    """
    if requested_link in available_links:
        return requested_link
    # TODO: Improve URDF link-name compatibility across MuJoCo-friendly GR1 variants.
    alias_candidates: dict[str, tuple[str, ...]] = {
        "l_palm": ("l_hand_pitch", "l_hand_roll", "l_hand_yaw"),
        "r_palm": ("r_hand_pitch", "r_hand_roll", "r_hand_yaw"),
    }
    candidates = alias_candidates.get(requested_link, ())
    for name in candidates:
        if name in available_links:
            logger.warning(
                "Requested end-effector link '%s' is not present. Using fallback link '%s'.",
                requested_link,
                name,
            )
            return name

    raise ValueError(f"Link '{requested_link}' not found in model. Available links: {tuple(available_links)}.")


class RobotKinematics:
    """Utility class that wraps Mink for FK + iterative IK."""

    def __init__(
        self,
        robot_model: str,
        end_effector_link: str = "eef_link",
        config: Optional[InverseKinematicsConfig] = None,
    ) -> None:
        self.robot_model = str(robot_model)
        self._config = config or InverseKinematicsConfig()

        model_path = Path(self.robot_model).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Robot model not found at {model_path}.")

        compiled_model_path = _create_meshless_urdf_copy(model_path)
        logger.info(
            "Loading Mink IK model from meshless URDF copy %s (source: %s)",
            compiled_model_path,
            model_path,
        )

        self._model = mujoco.MjModel.from_xml_path(str(compiled_model_path))
        self._data = mujoco.MjData(self._model)
        self._configuration = mink.Configuration(self._model)

        self._joint_names: list[str] = []
        self._joint_qpos_indexes: list[int] = []
        lower_limits: list[float] = []
        upper_limits: list[float] = []

        for joint_id in range(self._model.njnt):
            joint = self._model.joint(joint_id)
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if joint.type == mujoco.mjtJoint.mjJNT_BALL:
                continue

            width = len(joint.qpos0)
            start = joint.qposadr[0]
            self._joint_names.append(joint.name)
            self._joint_qpos_indexes.extend(start + np.arange(width).tolist())

            if joint.type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and width == 1:
                joint_range = self._model.jnt_range[joint_id]
                lower_limits.append(float(joint_range[0]))
                upper_limits.append(float(joint_range[1]))
            else:
                lower_limits.extend([-np.inf] * width)
                upper_limits.extend([np.inf] * width)

        self._lower_limits = np.asarray(lower_limits, dtype=np.float32)
        self._upper_limits = np.asarray(upper_limits, dtype=np.float32)

        if not self._joint_qpos_indexes:
            raise ValueError("No actuated joints found in model.")

        q0 = np.asarray(
            self._model.qpos0[self._joint_qpos_indexes],
            dtype=np.float32,
        )
        finite = np.isfinite(self._lower_limits) & np.isfinite(self._upper_limits)
        midpoint = 0.5 * (self._lower_limits + self._upper_limits)
        self._default_cfg = np.where(finite, midpoint, q0).astype(np.float32)
        self._default_cfg = np.clip(
            self._default_cfg,
            self._lower_limits,
            self._upper_limits,
        )

        self._link_names = [self._model.body(i).name for i in range(1, self._model.nbody)]
        self._link_name_to_index: dict[str, int] = {name: idx for idx, name in enumerate(self._link_names)}
        self._requested_ee_link_name = end_effector_link
        self._ee_link_name = _resolve_end_effector_link_name(
            end_effector_link,
            self._link_names,
        )
        # Keep the effective EE frame consistent with the original kinematics
        # convention when we must fall back from palm links to hand links.
        self._target_local_z_comp_deg = 0.0
        if self._requested_ee_link_name != self._ee_link_name:
            if self._requested_ee_link_name == "r_palm" and self._ee_link_name.startswith("r_hand_"):
                # Requested palm frame -> internal right hand frame
                self._target_local_z_comp_deg = -90.0
            elif self._requested_ee_link_name == "l_palm" and self._ee_link_name.startswith("l_hand_"):
                # Requested palm frame -> internal left hand frame
                self._target_local_z_comp_deg = 90.0
            if self._target_local_z_comp_deg != 0.0:
                logger.warning(
                    "Applying EE frame compensation %.1fdeg around local Z (requested='%s', resolved='%s').",
                    self._target_local_z_comp_deg,
                    self._requested_ee_link_name,
                    self._ee_link_name,
                )
        self._ee_body_id = self._model.body(self._ee_link_name).id

        self._posture_task = mink.PostureTask(
            self._model,
            cost=self._config.posture_cost,
            lm_damping=1.0,
        )
        self._ee_task = mink.FrameTask(
            frame_name=self._ee_link_name,
            frame_type="body",
            position_cost=self._config.position_cost,
            orientation_cost=self._config.orientation_cost,
            lm_damping=1.0,
        )

        # Compatibility proxy for callers using solver.robot.joints.*
        self.robot = SimpleNamespace(
            joints=SimpleNamespace(
                actuated_names=tuple(self._joint_names),
                lower_limits=self._lower_limits.copy(),
                upper_limits=self._upper_limits.copy(),
            ),
            links=SimpleNamespace(names=tuple(self._link_names)),
        )

    def _set_configuration_from_joint_cfg(self, joint_cfg: np.ndarray) -> None:
        qpos = np.asarray(self._model.qpos0, dtype=np.float64).copy()
        qpos[self._joint_qpos_indexes] = joint_cfg
        self._configuration.data.qpos[:] = qpos
        self._configuration.update()

    def _ee_pose_from_configuration(self) -> np.ndarray:
        pose = self._configuration.get_transform_frame_to_world(
            self._ee_link_name,
            "body",
        )
        mat = pose.as_matrix()
        # Convert rotation matrix back to quaternion using MuJoCo utility.
        quat_wxyz = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, mat[:3, :3].reshape(9))
        return np.concatenate([quat_wxyz, mat[:3, 3]]).astype(np.float32)

    def _pose_error(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
    ) -> float:
        pos_err = float(np.linalg.norm(current_pose[4:] - target_pose[4:]))
        rot_cur = _quat_wxyz_to_rotmat(current_pose[:4].astype(np.float64))
        rot_tgt = _quat_wxyz_to_rotmat(target_pose[:4].astype(np.float64))
        rot_err = _rotation_angle_rad(rot_cur, rot_tgt)
        return float(np.sqrt(pos_err * pos_err + rot_err * rot_err))

    def compute_forward_kinematics(
        self,
        joint_positions: npt.ArrayLike,
    ) -> FloatArray:
        """Compute all link poses for a given joint configuration."""
        cfg = np.asarray(joint_positions, dtype=np.float32).reshape(-1)
        if cfg.shape[0] != len(self._joint_qpos_indexes):
            raise ValueError(f"Expected joint_positions shape ({len(self._joint_qpos_indexes)},), received {cfg.shape}.")

        data = mujoco.MjData(self._model)
        data.qpos[:] = self._model.qpos0
        data.qpos[self._joint_qpos_indexes] = cfg
        mujoco.mj_kinematics(self._model, data)

        poses: list[np.ndarray] = []
        for body_name in self._link_names:
            body = data.body(body_name)
            pose = np.concatenate([body.xquat.copy(), body.xpos.copy()])
            poses.append(pose.astype(np.float32))
        return np.asarray(poses, dtype=np.float32)

    def compute_end_effector_pose(
        self,
        joint_positions: npt.ArrayLike,
    ) -> FloatArray:
        """Compute the effective end-effector pose."""
        fk = self.compute_forward_kinematics(joint_positions)
        pose = fk[self._link_name_to_index[self._ee_link_name]].copy()
        if self._target_local_z_comp_deg != 0.0:
            # Convert internal hand-frame orientation back to requested palm semantics.
            pose[:4] = _apply_local_z_rotation_wxyz(
                pose[:4].astype(np.float64),
                -self._target_local_z_comp_deg,
            ).astype(np.float32)
        return pose

    def compute_inverse_kinematics(
        self,
        end_effector_pose: npt.ArrayLike,
        initial_guess: Optional[npt.ArrayLike] = None,
    ) -> FloatArray:
        """Solve IK to reach the provided SE(3) pose (wxyzxyz)."""
        target_pose = np.asarray(
            end_effector_pose,
            dtype=np.float32,
        ).reshape(-1)
        if target_pose.shape != (7,):
            raise ValueError(f"Expected ee pose shape (7,), received {target_pose.shape}.")

        if initial_guess is None:
            joint_cfg = self._default_cfg.copy()
        else:
            joint_cfg = np.asarray(initial_guess, dtype=np.float32).reshape(-1)

        if joint_cfg.shape[0] != len(self._joint_qpos_indexes):
            raise ValueError(
                f"Expected initial_guess shape ({len(self._joint_qpos_indexes)},), received {joint_cfg.shape}."
            )

        target_mat = _pose_wxyzxyz_to_matrix(target_pose.astype(np.float64))
        if self._target_local_z_comp_deg != 0.0:
            target_pose = target_pose.copy()
            target_pose[:4] = _apply_local_z_rotation_wxyz(
                target_pose[:4].astype(np.float64),
                self._target_local_z_comp_deg,
            ).astype(np.float32)
            target_mat = _pose_wxyzxyz_to_matrix(target_pose.astype(np.float64))
        self._ee_task.set_target(mink.SE3.from_matrix(target_mat))

        self._set_configuration_from_joint_cfg(joint_cfg)
        self._posture_task.set_target_from_configuration(self._configuration)

        pose_error = np.inf
        for _ in range(self._config.max_iterations):
            vel = mink.solve_ik(
                self._configuration,
                [self._ee_task, self._posture_task],
                self._config.learning_rate,
                self._config.solver,
                1e-5,
            )
            self._configuration.integrate_inplace(
                vel,
                self._config.learning_rate,
            )
            # TODO: Integrating before clipping can briefly push joint_cfg outside limits and trigger warnings.
            joint_cfg = self._configuration.data.qpos[self._joint_qpos_indexes].astype(np.float32)
            joint_cfg = np.clip(
                joint_cfg,
                self._lower_limits,
                self._upper_limits,
            )
            self._set_configuration_from_joint_cfg(joint_cfg)

            current_pose = self._ee_pose_from_configuration()
            pose_error = self._pose_error(current_pose, target_pose)
            if pose_error <= self._config.pose_tolerance:
                return joint_cfg

        if pose_error > self._config.unreachable_threshold:
            raise IKUnreachableError(
                f"Target unreachable. Pose error: {pose_error:.4f} > {self._config.unreachable_threshold}."
            )

        return joint_cfg
