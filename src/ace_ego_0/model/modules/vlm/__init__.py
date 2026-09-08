"""Qwen3-VL backbone factory used by the released checkpoints."""


def get_vlm_model(config):
    """Build the configured Qwen3-VL interface."""
    vlm_name = str(config.framework.qwenvl.base_vlm)
    if "Qwen3-VL" not in vlm_name and "qwen3-vl-4b-instruct" not in vlm_name:
        raise NotImplementedError(f"Only Qwen3-VL is included in this release, got {vlm_name!r}.")
    from .qwen3_vl import Qwen3VLBackbone

    return Qwen3VLBackbone(config)
