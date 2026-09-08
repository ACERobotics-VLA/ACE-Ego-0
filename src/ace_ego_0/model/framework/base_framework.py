"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
No device placement or optimizer concerns are handled here.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import PretrainedConfig, PreTrainedModel

from ace_ego_0.inference_utils import initialize_overwatch
from ace_ego_0.model.framework.__init__ import build_framework
from ace_ego_0.model.framework.share_tools import dict_to_namespace, read_mode_config

logger = initialize_overwatch(__name__)


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to accept a structured config and register components in ``__init__``.
    """

    def __init__(self, hf_config=None) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """
        if hf_config is None:
            hf_config = PretrainedConfig()
        super().__init__(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        framework_override: Optional[str] = None,
        allow_missing_norm_stats: bool = False,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            framework_override: Optional framework name override from runtime entrypoints.
            allow_missing_norm_stats: Whether checkpoint loading may continue without normalization stats.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(
            pretrained_checkpoint,
            allow_missing_norm_stats=allow_missing_norm_stats,
        )  # read config and norm_stats

        config = dict_to_namespace(model_config)
        model_config = config
        if framework_override is not None:
            logger.info("Overriding framework name: %s -> %s", model_config.framework.name, framework_override)
            model_config.framework.name = framework_override
        # FrameworkModel = cls(config=model_config, **kwargs) # TODO find cls by config
        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats or {}
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        # 使用 mmap=True 进行内存映射加载,大幅减少网络存储的加载时间
        # 使用 weights_only=True 跳过 pickle 安全检查,加速加载
        model_state_dict = torch.load(
            pretrained_checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())

        state_proj_in_checkpoint = any("state_proj" in key for key in checkpoint_keys)
        state_proj_in_dual_expert_checkpoint = any("dual_expert_model.state_proj" in key for key in checkpoint_keys)
        state_proj_in_model = any("state_proj" in key for key in model_keys)
        state_proj_in_dual_expert_model = any("dual_expert_model.state_proj" in key for key in model_keys)

        if state_proj_in_checkpoint and state_proj_in_model:
            if state_proj_in_dual_expert_checkpoint and not state_proj_in_dual_expert_model:
                logger.info("Adapting state_proj keys: dual_expert_model.state_proj.* -> state_proj.*")
                adapted_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith("dual_expert_model.state_proj."):
                        adapted_key = key.replace("dual_expert_model.state_proj.", "state_proj.")
                        adapted_state_dict[adapted_key] = value
                    else:
                        adapted_state_dict[key] = value
                model_state_dict = adapted_state_dict
                checkpoint_keys = set(model_state_dict.keys())
            elif not state_proj_in_dual_expert_checkpoint and state_proj_in_dual_expert_model:
                logger.info("Adapting state_proj keys: state_proj.* -> dual_expert_model.state_proj.*")
                adapted_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith("state_proj.") and not key.startswith("dual_expert_model."):
                        adapted_key = f"dual_expert_model.{key}"
                        adapted_state_dict[adapted_key] = value
                    else:
                        adapted_state_dict[key] = value
                model_state_dict = adapted_state_dict
                checkpoint_keys = set(model_state_dict.keys())

        legacy_state_encoder_unexpected_keys = {
            "action_model.state_encoder.layer1.weight",
            "action_model.state_encoder.layer1.bias",
            "action_model.state_encoder.layer2.weight",
            "action_model.state_encoder.layer2.bias",
        }
        try:
            FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            if not missing_keys and unexpected_keys and unexpected_keys.issubset(legacy_state_encoder_unexpected_keys):
                logger.warning(
                    "Ignoring legacy unexpected state_encoder keys for no-state checkpoint: %s",
                    sorted(unexpected_keys),
                )
                incompatible = FrameworkModel.load_state_dict(model_state_dict, strict=False)
                remaining_missing_keys = set(incompatible.missing_keys)
                remaining_unexpected_keys = set(incompatible.unexpected_keys) - legacy_state_encoder_unexpected_keys
                if not remaining_missing_keys and not remaining_unexpected_keys:
                    return FrameworkModel
                if remaining_missing_keys:
                    logger.warning("Remaining missing keys after legacy fallback: %s", remaining_missing_keys)
                if remaining_unexpected_keys:
                    logger.warning(
                        "Remaining unexpected keys after legacy fallback: %s",
                        remaining_unexpected_keys,
                    )

            raise e

        return FrameworkModel

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.

        Raises:
            AssertionError: If multiple datasets present and key not provided,
                            or provided key not found.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray], norm_config: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Map normalized actions back to original value range.
        Supports per-dimension denormalization for translation, quaternion / rot6d, gripper, and waist dims.

        Args:
            normalized_actions: Array shape [T, D].
            action_norm_stats: Dict containing q01, q99, mean, std, mask.
            norm_config: Optional dict with rot_norm_mode, trans_norm_mode, and gripper_norm_mode.

        Returns:
            np.ndarray: Unnormalized actions.
        """
        from ace_ego_0.utils.normalization_utils import denormalize_actions_numpy
        from ace_ego_0.utils.rotation_utils import (
            identify_gripper_indices,
            identify_joint_ranges,
            identify_quaternion_indices,
            identify_rot6d_indices,
            identify_xyz_indices,
        )

        action_dim = normalized_actions.shape[-1]
        raw_mask = action_norm_stats.get("mask", None)
        if raw_mask is None or len(raw_mask) != action_dim:
            mask = np.ones(action_dim, dtype=bool)
        else:
            mask = np.array(raw_mask, dtype=bool)

        rot_mode = "min_max"
        trans_mode = "min_max"
        gripper_mode = "min_max"
        if norm_config:
            rot_mode = norm_config.get("rot_norm_mode", "min_max")
            trans_mode = norm_config.get("trans_norm_mode", "min_max")
            gripper_mode = norm_config.get("gripper_norm_mode", "min_max")

        # Build per-dimension norm mode array
        dim_modes = np.full(action_dim, "min_max", dtype=object)
        for start, end in identify_rot6d_indices(action_dim):
            dim_modes[start:end] = rot_mode
        for start, end in identify_quaternion_indices(action_dim):
            dim_modes[start:end] = rot_mode
        for start, end in identify_xyz_indices(action_dim):
            dim_modes[start:end] = trans_mode
        for start, end in identify_joint_ranges(action_dim):
            dim_modes[start:end] = trans_mode
        for gi in identify_gripper_indices(action_dim):
            dim_modes[gi] = gripper_mode
        return denormalize_actions_numpy(
            normalized_actions=normalized_actions,
            action_norm_stats=action_norm_stats,
            dim_modes=dim_modes,
            mask=mask,
        )

    @classmethod
    def get_action_stats(self, unnorm_key=None, norm_stats=None):
        """
        Retrieve raw action normalization statistics.
        """
        if norm_stats is None:
            norm_stats = self.norm_stats
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
