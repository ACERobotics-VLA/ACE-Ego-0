#!/usr/bin/env python3
"""Run one ACE-Ego-0 episode on one GR1 RoboCasa 24 task."""

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from evaluation.robocasa24.cli import evaluate_main  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(evaluate_main([*sys.argv[1:], "--episodes", "1"]))
