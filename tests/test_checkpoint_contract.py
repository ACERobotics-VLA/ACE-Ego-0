from pathlib import Path

import pytest
import yaml

from evaluation.robocasa24.checkpoint import inspect_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_bundle_or_skip(bundle_name: str, weight_name: str) -> Path:
    """Return a local checkpoint bundle, or skip when weights are not downloaded yet."""
    bundle_path = REPO_ROOT / "checkpoints/robocasa24" / bundle_name
    weight_path = bundle_path / "checkpoints" / weight_name
    if not weight_path.is_file():
        pytest.skip(f"Checkpoint weights are not downloaded: {weight_path}")
    return bundle_path


def test_absolute_checkpoint_contract() -> None:
    contract = inspect_checkpoint(
        _checkpoint_bundle_or_skip(
            "ace-ego-0-absolute",
            "ace_ego_0_robocasa24_absolute.pt",
        )
    )
    assert contract.framework_name == "ACE-Ego-0"
    assert contract.action_mode == "absolute"
    assert contract.action_horizon == 30
    assert contract.rotation_repr == "rot6d"


def test_delta_checkpoint_contract() -> None:
    contract = inspect_checkpoint(
        _checkpoint_bundle_or_skip(
            "ace-ego-0-delta",
            "ace_ego_0_robocasa24_delta.pt",
        )
    )
    assert contract.framework_name == "ACE-Ego-0"
    assert contract.action_mode == "delta"
    assert contract.delta_action_mode == "state_anchor"
    assert contract.delta_base_state == "first_state"
    assert contract.delta_quat_mode == "subtract"


def test_public_checkpoint_configs_exclude_training_only_sections() -> None:
    forbidden_keys = {"trainer", "optimizer", "scheduler", "co_training", "loss_groups", "data_mix"}
    for bundle_name in ("ace-ego-0-absolute", "ace-ego-0-delta"):
        config_path = REPO_ROOT / "checkpoints/robocasa24" / bundle_name / "config.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        assert isinstance(config, dict)
        assert forbidden_keys.isdisjoint(config)
        assert forbidden_keys.isdisjoint(config["framework"])
        assert forbidden_keys.isdisjoint(config["framework"]["action_model"])
        assert forbidden_keys.isdisjoint(config["datasets"]["vla_data"])
