"""Qwen image message construction for RoboCasa inference."""

from __future__ import annotations

from typing import Any


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
