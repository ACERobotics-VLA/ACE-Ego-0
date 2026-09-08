"""Checkpoint-compatible ACE-Ego-0 inference policy."""

from typing import Any, List, Optional

import numpy as np
import torch

from ace_ego_0.inference_utils import initialize_overwatch
from ace_ego_0.inference_utils.image import resize_images
from ace_ego_0.inference_utils.image_tools import to_pil_preserve
from ace_ego_0.model.framework.base_framework import baseframework
from ace_ego_0.model.modules.action_model.flow_matching_action_expert import FlowMatchingActionExpert, get_action_model
from ace_ego_0.model.modules.vlm import get_vlm_model
from ace_ego_0.model.tools import FRAMEWORK_REGISTRY

logger = initialize_overwatch(__name__)

SUPPORTED_COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


def _resolve_compute_dtype(config_section: Any, *, field_name: str, default: torch.dtype) -> torch.dtype:
    """Resolve one configured module compute dtype and reject unsupported values."""
    configured_dtype = config_section.get("compute_dtype", default)
    if isinstance(configured_dtype, torch.dtype):
        return configured_dtype
    if not isinstance(configured_dtype, str):
        raise TypeError(f"`{field_name}` must be a string or torch.dtype, got {type(configured_dtype).__name__}.")

    normalized_dtype = configured_dtype.strip().lower()
    if normalized_dtype not in SUPPORTED_COMPUTE_DTYPES:
        raise ValueError(
            f"Unsupported `{field_name}`: {configured_dtype!r}. Expected one of {sorted(SUPPORTED_COMPUTE_DTYPES)}."
        )
    return SUPPORTED_COMPUTE_DTYPES[normalized_dtype]


def _resolve_action_horizon(action_model_cfg) -> int:
    """Resolve one action horizon from action model config and validate consistency."""
    future_window = int(action_model_cfg.get("future_action_window_size", 0))
    past_window = int(action_model_cfg.get("past_action_window_size", 0))
    derived_horizon = future_window + past_window + 1

    configured_horizon = action_model_cfg.get("action_horizon", None)
    if configured_horizon is None:
        return derived_horizon

    configured_horizon = int(configured_horizon)
    if configured_horizon != derived_horizon:
        raise ValueError(
            "Inconsistent action horizon config for ACE-Ego-0: "
            f"`action_horizon`={configured_horizon}, "
            f"`future_action_window_size + past_action_window_size + 1`={derived_horizon}."
        )
    return configured_horizon


def _resolve_state_injection_mode(action_model_cfg) -> str:
    """Resolve one canonical state route so the framework and action head agree on state usage."""
    explicit_mode = action_model_cfg.get("state_injection_mode", None)
    if explicit_mode is None:
        state_design_preset = action_model_cfg.get("state_design_preset", None)
        if state_design_preset is not None and str(state_design_preset).lower() in {"rdt2", "rdt2_like", "rdt2_v1"}:
            explicit_mode = "timestep_conditioning"

    if explicit_mode is not None:
        normalized_mode = str(explicit_mode).lower()
        alias_map = {
            "none": "disabled",
            "prepend_token": "token",
            "raw_state_token": "token",
            "rdt2": "timestep_conditioning",
            "rdt2_like": "timestep_conditioning",
        }
        normalized_mode = alias_map.get(normalized_mode, normalized_mode)
        valid_modes = {"disabled", "token", "timestep_conditioning"}
        if normalized_mode not in valid_modes:
            raise ValueError(
                f"Unsupported `state_injection_mode` for ACE-Ego-0: {explicit_mode!r}. "
                f"Expected one of {sorted(valid_modes)}."
            )
        return normalized_mode

    if bool(action_model_cfg.get("enable_state_timestep_conditioning", False)):
        return "timestep_conditioning"
    if bool(action_model_cfg.get("enable_state_input", False)):
        return "token"
    return "disabled"


def _prepare_state_tensor(
    state: list[Any] | None,
    *,
    state_masks: list[Any] | None = None,
    enable_state_input: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Convert raw batch state arrays to model tensors when state input is enabled."""
    if not enable_state_input or state is None:
        return None

    state_tensor = torch.as_tensor(np.asarray(state), device=device, dtype=dtype)
    if state_tensor.ndim == 2:
        state_tensor = state_tensor.unsqueeze(1)

    if state_masks is not None:
        state_mask_tensor = torch.as_tensor(np.asarray(state_masks), device=device, dtype=dtype)
        if state_mask_tensor.ndim == 3 and state_mask_tensor.shape[1] == 1:
            state_mask_tensor = state_mask_tensor[:, 0]
        if state_mask_tensor.ndim != 2 or state_mask_tensor.shape[-1] != state_tensor.shape[-1]:
            raise ValueError(
                "State masks must have shape [B, state_dim] matching the state tensor; "
                f"got state={tuple(state_tensor.shape)}, mask={tuple(state_mask_tensor.shape)}."
            )
        state_tensor = state_tensor * state_mask_tensor.unsqueeze(1)

    return state_tensor


def _extract_visual_inputs(
    examples: list[dict[str, Any]],
    *,
    convert_to_pil: bool,
) -> list[Any]:
    """Extract a batch of static camera images."""
    if any("video" in example for example in examples):
        raise ValueError("RoboCasa 24 inference accepts static camera images only.")
    images = [example["image"] for example in examples]
    return to_pil_preserve(images) if convert_to_pil else images


@FRAMEWORK_REGISTRY.register("ACE-Ego-0")
class ACEEgoPolicy(baseframework):
    """Inference-only multimodal vision-language-action policy."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical inference configuration.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.vlm_compute_dtype = _resolve_compute_dtype(
            config.framework.qwenvl,
            field_name="framework.qwenvl.compute_dtype",
            default=torch.bfloat16,
        )
        self.action_compute_dtype = _resolve_compute_dtype(
            config.framework.action_model,
            field_name="framework.action_model.compute_dtype",
            default=torch.float32,
        )
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.qwen_vl_interface.to(dtype=self.vlm_compute_dtype)

        # dynamic get llm config
        num_vl_layers, llm_hidden_size = (
            self.qwen_vl_interface.model.config.text_config.num_hidden_layers,
            self.qwen_vl_interface.model.config.hidden_size,
        )
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        self.action_model: FlowMatchingActionExpert = get_action_model(config=self.config)
        self.action_model.to(dtype=self.action_compute_dtype)
        self.state_injection_mode = _resolve_state_injection_mode(config.framework.action_model)
        self.enable_state_input = self.state_injection_mode != "disabled"
        self.config.framework.action_model.state_injection_mode = self.state_injection_mode
        self.config.framework.action_model.enable_state_input = self.enable_state_input

        self.future_action_window_size = int(config.framework.action_model.future_action_window_size)
        self.past_action_window_size = int(config.framework.action_model.past_action_window_size)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_horizon = _resolve_action_horizon(config.framework.action_model)
        self.config.framework.action_model.action_horizon = self.action_horizon

        logger.info(
            "ACE-Ego-0 dtype policy: VLM=%s, action_model=%s.",
            self.vlm_compute_dtype,
            self.action_compute_dtype,
        )
        logger.info("ACE-Ego-0 state input enabled: %s.", self.enable_state_input)
        logger.info("ACE-Ego-0 state injection mode: %s.", self.state_injection_mode)

    def _encode_qwen_hidden_states(
        self,
        *,
        batch_images: list[Any],
        instructions: list[str],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode image-language inputs and return action-head layerwise hidden states."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        with torch.autocast(
            "cuda",
            dtype=self.vlm_compute_dtype,
            enabled=self.vlm_compute_dtype != torch.float32,
        ):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]
        return vl_embs_list, base_hidden

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Reject training calls because this public release exposes inference only."""
        raise RuntimeError(
            "The public ACE-Ego-0 release is inference-only; call predict_action() for RoboCasa evaluation."
        )

    def predict_action(
        self,
        examples: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Inference: sample future actions with flow matching.

        Steps:
          1. Resize images to the configured resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          3. Predict actions with state and URDF conditioning
          4. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        batch_images = _extract_visual_inputs(examples, convert_to_pil=True)
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        state_mask = None
        robot_names = [str(example["robot_name"]) for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        vl_embs_list, base_hidden = self._encode_qwen_hidden_states(
            batch_images=batch_images,
            instructions=instructions,
        )
        action_vl_embs_list = [hidden.to(dtype=self.action_compute_dtype) for hidden in vl_embs_list]

        state_tensor = _prepare_state_tensor(
            state,
            state_masks=state_mask,
            enable_state_input=self.enable_state_input,
            device=base_hidden.device,
            dtype=self.action_compute_dtype,
        )
        # Action expert sampling.
        with torch.autocast("cuda", enabled=False):
            pred_actions = self.action_model.predict_action(
                action_vl_embs_list,
                state_tensor,
                robot_names=robot_names,
                action_horizon=self.action_horizon,
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().float().numpy()
        result = {
            "normalized_actions": normalized_actions,
            "action_horizon": int(self.action_horizon),
        }
        return result
