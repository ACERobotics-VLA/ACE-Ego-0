"""Small recursive YAML composition for public SFT recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_sft_config(config_path: str | Path, *, _seen: frozenset[Path] = frozenset()) -> Any:
    """Load one recipe and recursively merge its optional `base_config`."""
    resolved_path = Path(config_path).expanduser().resolve()
    if resolved_path in _seen:
        raise ValueError(f"Recursive SFT base_config reference: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"SFT config does not exist: {resolved_path}")

    config = OmegaConf.load(resolved_path)
    base_config = config.pop("base_config", None)
    if base_config is not None:
        base_path = Path(str(base_config)).expanduser()
        if not base_path.is_absolute():
            base_path = resolved_path.parent / base_path
        parent = load_sft_config(base_path, _seen=_seen | {resolved_path})
        config = OmegaConf.merge(parent, config)
    OmegaConf.resolve(config)
    return config
