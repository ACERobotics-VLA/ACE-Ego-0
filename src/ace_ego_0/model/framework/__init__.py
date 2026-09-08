"""Factory for the ACE-Ego-0 inference framework."""

from ace_ego_0.model.tools import FRAMEWORK_REGISTRY


def build_framework(cfg):
    """Build the ACE-Ego-0 framework declared by a checkpoint configuration."""
    framework_name = cfg.framework.get("name", cfg.framework.get("framework_py", None))
    if framework_name in {"ACE-Ego-0", "ACE_Ego_0"}:
        from ace_ego_0.model.framework.ace_ego_policy import ACEEgoPolicy

        return ACEEgoPolicy(cfg)
    if framework_name not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {framework_name!r} is not included in this evaluation release.")
    return FRAMEWORK_REGISTRY[framework_name](cfg)


__all__ = ["FRAMEWORK_REGISTRY", "build_framework"]
