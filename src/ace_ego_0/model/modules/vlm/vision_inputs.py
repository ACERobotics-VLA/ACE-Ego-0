"""Qwen image message construction for RoboCasa inference."""

from __future__ import annotations

from typing import Any


def pin_qwen_processor_image_pixels(processor: Any, image_size: Any | None) -> None:
    """Lock Qwen's smart resize to the configured training pixel budget."""
    if image_size is None:
        return
    try:
        width, height = (int(image_size[0]), int(image_size[1]))
    except (TypeError, ValueError, IndexError, KeyError) as error:
        raise ValueError(f"`datasets.vla_data.image_size` must be [width, height], got {image_size!r}.") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"`datasets.vla_data.image_size` must contain positive values, got {image_size!r}.")

    pixel_budget = width * height
    image_processor = processor.image_processor
    image_processor.size = {"shortest_edge": pixel_budget, "longest_edge": pixel_budget}
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = pixel_budget
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = pixel_budget


def build_qwen_messages(
    *,
    images: list[list[Any]],
    instructions: list[str],
    cot_prompt: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Build batched Qwen messages from one or more static camera images."""
    if len(images) != len(instructions):
        raise ValueError("Visual observations and instructions must have the same batch size.")

    messages: list[list[dict[str, Any]]] = []
    for sample_index, instruction in enumerate(instructions):
        content = [{"type": "image", "image": image} for image in images[sample_index]]

        prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt is not None else instruction
        content.append({"type": "text", "text": prompt})
        messages.append([{"role": "user", "content": content}])
    return messages
