import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from ace_ego_0.model.modules.vlm.vision_inputs import build_qwen_messages, pin_qwen_processor_image_pixels


class Qwen3VLBackbone(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = (
            os.environ.get("ACE_EGO_0_BASE_VLM")
            or os.environ.get("GR1_ROBOCASA24_BASE_VLM")
            or qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        )

        model_path = Path(model_id).expanduser()
        if not model_path.is_dir() and not model_path.is_absolute():
            repository_root = Path(__file__).resolve().parents[5]
            for bundled_model_path in (repository_root / model_path, repository_root / "assets" / model_path):
                if bundled_model_path.is_dir():
                    model_path = bundled_model_path
                    break
        if model_path.is_dir() and (model_path / "config.json").is_file():
            model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            model = Qwen3VLForConditionalGeneration._from_config(
                model_config,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        image_size = config.datasets.vla_data.get("image_size", None)
        if not bool(config.datasets.vla_data.get("preserve_raw_image_size", False)):
            pin_qwen_processor_image_pixels(processor, image_size)

        self.model = model
        self.processor = processor
        self.config = config
        self.gradient_checkpointing_enabled = False
        self._original_use_cache = getattr(self.model.config, "use_cache", None)

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable memory-saving activation checkpointing during full SFT."""
        self.gradient_checkpointing_enabled = bool(enabled)
        if self.gradient_checkpointing_enabled:
            if not hasattr(self.model, "gradient_checkpointing_enable"):
                raise RuntimeError("The configured Qwen3-VL model does not support gradient checkpointing.")
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
            return

        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        if self._original_use_cache is not None and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self._original_use_cache

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to the underlying Qwen3-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def build_qwenvl_inputs(
        self,
        images=None,
        instructions=None,
        **kwargs,
    ):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        if instructions is None:
            raise ValueError("`instructions` must be provided.")
        cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", None)
        if images is None:
            raise ValueError("ACE-Ego-0 requires image inputs.")
        messages = build_qwen_messages(images=images, instructions=instructions, cot_prompt=cot_prompt)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return batch_inputs.to(self.model.device)
