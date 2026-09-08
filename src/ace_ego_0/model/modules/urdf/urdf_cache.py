"""Disk caching for static URDF graph tensors."""

from __future__ import annotations

import fcntl
import logging
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from ace_ego_0.model.modules.urdf.robot_urdf_registry import RobotUrdfSpec
from ace_ego_0.model.modules.urdf.urdf_graph_builder import build_urdf_graph_tensors

logger = logging.getLogger(__name__)


def resolve_urdf_cache_path(cache_dir: Path | str, spec: RobotUrdfSpec) -> Path:
    """Resolve the cache file path for one robot URDF graph payload."""
    return Path(cache_dir).expanduser() / f"{spec.cache_key}.pkl"


def _resolve_urdf_cache_lock_path(cache_path: Path) -> Path:
    """Resolve the lock file path for one robot URDF graph payload."""
    return cache_path.parent / f"{cache_path.name}.lock"


def _resolve_temporary_cache_path(cache_path: Path) -> Path:
    """Resolve a process-unique temporary path for one cache write."""
    unique_token = f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex}"
    return cache_path.parent / f"{cache_path.name}.tmp-{unique_token}"


def _resolve_lock_timeout_seconds() -> float:
    """Resolve how long ranks may wait for one URDF cache writer."""
    raw_value = os.environ.get("ACE_EGO_0_URDF_CACHE_LOCK_TIMEOUT_SECONDS", "300")
    try:
        timeout_seconds = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid ACE_EGO_0_URDF_CACHE_LOCK_TIMEOUT_SECONDS=%r; falling back to 300 seconds.",
            raw_value,
        )
        timeout_seconds = 300.0
    return max(timeout_seconds, 0.0)


@contextmanager
def _urdf_cache_write_lock(cache_path: Path):
    """Acquire an exclusive lock for one cache path during build or save."""
    lock_path = _resolve_urdf_cache_lock_path(cache_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    timeout_seconds = _resolve_lock_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                break
            except BlockingIOError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout_seconds:.1f}s waiting for URDF cache lock {lock_path}"
                    ) from exc
                time.sleep(0.25)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_urdf_graph_cache(cache_path: Path | str, payload: dict[str, Any]) -> None:
    """Persist one URDF graph payload to disk atomically."""
    cache_path = Path(cache_path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _resolve_temporary_cache_path(cache_path)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def load_urdf_graph_cache(cache_path: Path | str) -> dict[str, Any]:
    """Load one cached URDF graph payload from disk."""
    return torch.load(Path(cache_path).expanduser(), map_location="cpu")


def build_and_save_urdf_graph_cache(spec: RobotUrdfSpec, cache_dir: Path | str) -> dict[str, Any]:
    """Build the graph tensors for one robot and store them in the cache directory."""
    cache_path = resolve_urdf_cache_path(cache_dir, spec)

    with _urdf_cache_write_lock(cache_path):
        if cache_path.exists():
            payload = load_urdf_graph_cache(cache_path)
            cached_urdf_path = payload.get("urdf_path")
            if cached_urdf_path is None or Path(cached_urdf_path).expanduser().resolve() == spec.urdf_path.resolve():
                logger.info("Reusing existing URDF graph cache for %s at %s.", spec.robot_name, cache_path)
                return payload
            logger.info(
                "Replacing URDF graph cache %s because it was built from %s instead of %s.",
                cache_path,
                cached_urdf_path,
                spec.urdf_path,
            )

        payload = build_urdf_graph_tensors(spec)
        save_urdf_graph_cache(cache_path, payload)

    logger.info("Saved URDF graph cache for %s to %s.", spec.robot_name, cache_path)
    return payload


def load_or_build_urdf_graph_cache(
    spec: RobotUrdfSpec,
    cache_dir: Path | str,
    *,
    auto_build: bool = True,
) -> dict[str, Any]:
    """Load one URDF cache payload, optionally building it when missing."""
    cache_path = resolve_urdf_cache_path(cache_dir, spec)
    if cache_path.exists():
        payload = load_urdf_graph_cache(cache_path)
        cached_urdf_path = payload.get("urdf_path")
        if cached_urdf_path is None or Path(cached_urdf_path).expanduser().resolve() == spec.urdf_path.resolve():
            return payload
        logger.info(
            "Ignoring URDF graph cache %s because it was built from %s instead of %s.",
            cache_path,
            cached_urdf_path,
            spec.urdf_path,
        )

    if not auto_build:
        raise FileNotFoundError(f"URDF cache file does not exist: {cache_path}")

    logger.info("URDF cache for %s was missing at %s. Building it lazily.", spec.robot_name, cache_path)
    return build_and_save_urdf_graph_cache(spec, cache_dir)
