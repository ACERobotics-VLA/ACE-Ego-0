#!/usr/bin/env python3
"""Validate the simulator dependencies and task registrations for RoboCasa 24."""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger(__name__)
EXPECTED_VERSIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "numpy": "1.26.4",
    "mujoco": "3.2.6",
    "robosuite": "1.5.1",
    "robocasa": "0.2.0",
    "mink": "0.0.5",
}


def _distribution_version(distribution: str) -> str | None:
    """Return an installed distribution version, or ``None`` when it is absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _has_files(path: Path) -> bool:
    """Return whether a directory contains at least one regular file."""
    return path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))


def main() -> int:
    """Check versions, RoboCasa assets, and all 24 registered environments."""
    failures: list[str] = []

    if sys.version_info[:2] != (3, 10):
        failures.append(f"Python 3.10 is required; found {sys.version.split()[0]}")

    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = _distribution_version(distribution)
        if actual != expected:
            failures.append(f"{distribution}=={expected} is required; found {actual or 'not installed'}")
        else:
            LOGGER.info("%s==%s", distribution, actual)

    try:
        import torch
    except ImportError as error:
        failures.append(f"PyTorch import failed: {error}")
    else:
        if not torch.cuda.is_available():
            failures.append("CUDA is not available; RoboCasa evaluation requires an NVIDIA GPU.")
        else:
            LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(0))

    try:
        import gymnasium as gym
        import robocasa
        import robocasa.utils.gym_utils
    except (ImportError, AssertionError) as error:
        failures.append(f"RoboCasa imports failed: {error}")
    else:
        asset_root = Path(robocasa.__file__).resolve().parent / "models" / "assets"
        for relative_path in (
            "arenas",
            "fixtures",
            "objects",
            "scenes",
            "textures",
            "objects/sketchfab",
            "objects/lightwheel",
        ):
            asset_path = asset_root / relative_path
            if not _has_files(asset_path):
                failures.append(f"RoboCasa asset directory is missing or empty: {asset_path}")

        from evaluation.robocasa24.tasks import ROBOCASA24_TASKS, task_env_name

        for task_name in ROBOCASA24_TASKS:
            environment_name = task_env_name(task_name)
            try:
                gym.spec(environment_name)
            except (gym.error.Error, KeyError) as error:
                failures.append(f"Environment is not registered: {environment_name} ({error})")

    if failures:
        for failure in failures:
            LOGGER.error(failure)
        return 1

    LOGGER.info("All 24 GR1 RoboCasa environments are registered and assets are present.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
