from ace_ego_0 import ACEEgoPolicy, FlowMatchingActionExpert, Qwen3VLBackbone
from ace_ego_0.model.tools import FRAMEWORK_REGISTRY


def test_public_model_names_are_registered() -> None:
    assert FlowMatchingActionExpert.__name__ == "FlowMatchingActionExpert"
    assert Qwen3VLBackbone.__name__ == "Qwen3VLBackbone"
    assert FRAMEWORK_REGISTRY["ACE-Ego-0"] is ACEEgoPolicy
