"""Robot-name registry for URDF-conditioned ace_ego_0 baselines."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotUrdfSpec:
    """Static URDF metadata for one supported robot embodiment."""

    robot_name: str
    urdf_path: Path
    base_link: str
    eef_link_names: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    @property
    def cache_key(self) -> str:
        """Return the canonical cache key used for disk persistence."""
        return self.robot_name


def _normalize_name(name: str) -> str:
    """Normalize one robot identifier for alias matching."""
    return "".join(character.lower() for character in name if character.isalnum())


def _require_non_empty_string(value: object, field_name: str) -> str:
    """Validate one configuration value is a non-empty string."""
    normalized_value = str(value).strip()
    if not normalized_value:
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return normalized_value


def resolve_urdf_spec_overrides(raw_overrides: object) -> dict[str, RobotUrdfSpec]:
    """Parse configuration-provided URDF specifications keyed by robot identifiers.

    Override entries are intentionally kept local to one model instance. This allows a
    dataset-specific URDF to be supplied without mutating the process-wide registry or
    assigning an unverified description to another dataset.
    """
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, Mapping) and not hasattr(raw_overrides, "items"):
        raise TypeError("`urdf_robot_specs` must be a mapping from robot identifiers to URDF metadata.")

    resolved_specs: dict[str, RobotUrdfSpec] = {}
    for identifier, raw_spec in raw_overrides.items():
        identifier_text = _require_non_empty_string(identifier, "urdf_robot_specs key")
        if not isinstance(raw_spec, Mapping) and not hasattr(raw_spec, "items"):
            raise TypeError(f"URDF override for {identifier_text!r} must be a mapping.")

        if "urdf_path" not in raw_spec:
            raise ValueError(f"URDF override for {identifier_text!r} is missing `urdf_path`.")
        if "base_link" not in raw_spec:
            raise ValueError(f"URDF override for {identifier_text!r} is missing `base_link`.")
        if "eef_link_names" not in raw_spec:
            raise ValueError(f"URDF override for {identifier_text!r} is missing `eef_link_names`.")

        raw_eef_link_names = raw_spec["eef_link_names"]
        if isinstance(raw_eef_link_names, str) or not hasattr(raw_eef_link_names, "__iter__"):
            raise TypeError(f"`eef_link_names` for {identifier_text!r} must be a non-empty sequence.")
        eef_link_names = tuple(
            _require_non_empty_string(link_name, f"eef_link_names for {identifier_text!r}")
            for link_name in raw_eef_link_names
        )
        if not eef_link_names:
            raise ValueError(f"`eef_link_names` for {identifier_text!r} must not be empty.")

        robot_name = _require_non_empty_string(raw_spec.get("robot_name", identifier_text), "URDF robot_name")
        raw_aliases = raw_spec.get("aliases", ())
        if isinstance(raw_aliases, str) or not hasattr(raw_aliases, "__iter__"):
            raise TypeError(f"`aliases` for {identifier_text!r} must be a sequence when provided.")
        aliases = tuple(_require_non_empty_string(alias, f"aliases for {identifier_text!r}") for alias in raw_aliases)
        spec = RobotUrdfSpec(
            robot_name=robot_name,
            urdf_path=Path(_require_non_empty_string(raw_spec["urdf_path"], "URDF urdf_path")).expanduser(),
            base_link=_require_non_empty_string(raw_spec["base_link"], "URDF base_link"),
            eef_link_names=eef_link_names,
            aliases=aliases,
        )

        for lookup_name in (identifier_text, robot_name, *aliases):
            normalized_name = _normalize_name(lookup_name)
            existing_spec = resolved_specs.get(normalized_name)
            if existing_spec is not None and existing_spec != spec:
                raise ValueError(f"URDF override identifier {lookup_name!r} is configured more than once.")
            resolved_specs[normalized_name] = spec

    return resolved_specs


ROBOT_URDF_SPECS: dict[str, RobotUrdfSpec] = {
    "GR1": RobotUrdfSpec(
        robot_name="GR1",
        urdf_path=Path(__file__).resolve().parents[5] / "assets/GR1T2_with_hands.urdf",
        base_link="base",
        eef_link_names=("l_hand_pitch", "r_hand_pitch"),
        aliases=("gr1",),
    ),
}


ROBOT_TYPE_TO_CANONICAL_ROBOT_NAME: dict[str, str] = {
    "fourier_gr1_eef_gripper_6d": "GR1",
}


def _resolve_default_urdf_spec(canonical_name: str) -> RobotUrdfSpec:
    """Resolve a built-in URDF spec, honoring a local runtime path override."""
    spec = ROBOT_URDF_SPECS[canonical_name]
    if canonical_name == "GR1":
        configured_path = os.environ.get("ACE_EGO_0_GR1_URDF")
        if configured_path:
            return RobotUrdfSpec(
                robot_name=spec.robot_name,
                urdf_path=Path(configured_path).expanduser().resolve(),
                base_link=spec.base_link,
                eef_link_names=spec.eef_link_names,
                aliases=spec.aliases,
            )
    return spec


_ROBOT_NAME_ALIASES: dict[str, str] = {}
for canonical_name, spec in ROBOT_URDF_SPECS.items():
    _ROBOT_NAME_ALIASES[_normalize_name(canonical_name)] = canonical_name
    for alias in spec.aliases:
        _ROBOT_NAME_ALIASES[_normalize_name(alias)] = canonical_name

for robot_type, canonical_name in ROBOT_TYPE_TO_CANONICAL_ROBOT_NAME.items():
    _ROBOT_NAME_ALIASES[_normalize_name(robot_type)] = canonical_name


def list_supported_robot_names() -> list[str]:
    """Return the canonical robot names supported by the URDF registry."""
    return list(ROBOT_URDF_SPECS.keys())


def resolve_robot_name(
    robot_type_or_name: str,
    *,
    urdf_spec_overrides: Mapping[str, RobotUrdfSpec] | None = None,
) -> str:
    """Resolve one robot type or alias, consulting optional local URDF overrides first."""
    normalized_name = _normalize_name(robot_type_or_name)
    if urdf_spec_overrides is not None:
        override_spec = urdf_spec_overrides.get(normalized_name)
        if override_spec is not None:
            return override_spec.robot_name

    resolved_name = _ROBOT_NAME_ALIASES.get(normalized_name)
    if resolved_name is not None:
        return resolved_name

    raise KeyError(f"Unsupported robot identifier for URDF conditioning: {robot_type_or_name!r}.")


def get_robot_urdf_spec(
    robot_name: str,
    *,
    require_urdf_file: bool = True,
    urdf_spec_overrides: Mapping[str, RobotUrdfSpec] | None = None,
) -> RobotUrdfSpec:
    """Return URDF metadata, optionally requiring the source URDF file to exist.

    A prebuilt graph cache is sufficient for inference and training. Callers that
    load such a cache can disable the source-file check; callers that may build a
    missing cache should keep the default strict check.
    """
    canonical_name = resolve_robot_name(robot_name, urdf_spec_overrides=urdf_spec_overrides)
    spec = urdf_spec_overrides.get(_normalize_name(robot_name)) if urdf_spec_overrides is not None else None
    if spec is None and urdf_spec_overrides is not None:
        spec = urdf_spec_overrides.get(_normalize_name(canonical_name))
    if spec is None:
        spec = _resolve_default_urdf_spec(canonical_name)
    if require_urdf_file and not spec.urdf_path.exists():
        raise FileNotFoundError(f"URDF file for {canonical_name} does not exist: {spec.urdf_path}")
    return spec
