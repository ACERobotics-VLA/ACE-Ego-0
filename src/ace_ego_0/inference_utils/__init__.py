"""Small inference-only utilities used by the ACE-Ego-0 runtime."""

import logging


def initialize_overwatch(name: str) -> logging.Logger:
    """Return a standard logger for an inference module."""
    return logging.getLogger(name)


__all__ = ["initialize_overwatch"]
