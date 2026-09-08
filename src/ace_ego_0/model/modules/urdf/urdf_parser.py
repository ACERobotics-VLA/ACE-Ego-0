"""URDF parsing helpers for static embodiment conditioning."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedJoint:
    """One URDF joint record normalized for graph feature extraction."""

    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    limit_lower: float
    limit_upper: float
    has_limit: bool


@dataclass(frozen=True)
class ParsedUrdfModel:
    """Minimal URDF model representation needed by the v1 graph builder."""

    robot_name: str
    base_link: str
    joints: tuple[ParsedJoint, ...]


def _parse_float_vector(raw_value: str | None, expected_length: int, default: tuple[float, ...]) -> tuple[float, ...]:
    """Parse one whitespace-separated numeric vector from URDF XML."""
    if raw_value in (None, ""):
        return default

    parsed_values = tuple(float(value) for value in raw_value.strip().split())
    if len(parsed_values) != expected_length:
        raise ValueError(f"Expected {expected_length} values, got {len(parsed_values)} from {raw_value!r}.")
    return parsed_values


def _resolve_base_link(root: ET.Element, child_links: set[str], parent_links: set[str]) -> str:
    """Resolve the base link of one XML-parsed URDF tree."""
    root_candidates = sorted(parent_links - child_links)
    if root_candidates:
        return root_candidates[0]

    link_names = [element.attrib["name"] for element in root.findall("link") if "name" in element.attrib]
    if len(link_names) == 1:
        return link_names[0]

    robot_name = root.attrib.get("name", "<unknown>")
    raise ValueError(f"Could not resolve a unique base link for URDF robot {robot_name!r}.")


def _parse_urdf_with_xml(urdf_path: Path) -> ParsedUrdfModel:
    """Parse one URDF using only the Python standard library."""
    root = ET.parse(urdf_path).getroot()
    joints: list[ParsedJoint] = []
    child_links: set[str] = set()
    parent_links: set[str] = set()

    for joint_element in root.findall("joint"):
        joint_name = joint_element.attrib["name"]
        joint_type = joint_element.attrib.get("type", "fixed")

        parent_element = joint_element.find("parent")
        child_element = joint_element.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"Joint {joint_name!r} is missing a parent or child link.")

        parent_link = parent_element.attrib["link"]
        child_link = child_element.attrib["link"]
        parent_links.add(parent_link)
        child_links.add(child_link)

        origin_element = joint_element.find("origin")
        axis_element = joint_element.find("axis")
        limit_element = joint_element.find("limit")

        origin_xyz = _parse_float_vector(
            None if origin_element is None else origin_element.attrib.get("xyz"),
            expected_length=3,
            default=(0.0, 0.0, 0.0),
        )
        origin_rpy = _parse_float_vector(
            None if origin_element is None else origin_element.attrib.get("rpy"),
            expected_length=3,
            default=(0.0, 0.0, 0.0),
        )
        axis_default = (1.0, 0.0, 0.0) if joint_type in {"revolute", "prismatic", "continuous"} else (0.0, 0.0, 0.0)
        axis = _parse_float_vector(
            None if axis_element is None else axis_element.attrib.get("xyz"),
            expected_length=3,
            default=axis_default,
        )

        has_limit = limit_element is not None and {"lower", "upper"}.issubset(limit_element.attrib)
        if has_limit:
            limit_lower = float(limit_element.attrib["lower"])
            limit_upper = float(limit_element.attrib["upper"])
        else:
            limit_lower = 0.0
            limit_upper = 0.0

        joints.append(
            ParsedJoint(
                name=joint_name,
                joint_type=joint_type,
                parent_link=parent_link,
                child_link=child_link,
                axis=axis,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                has_limit=has_limit,
            )
        )

    base_link = _resolve_base_link(root, child_links=child_links, parent_links=parent_links)
    return ParsedUrdfModel(robot_name=root.attrib.get("name", urdf_path.stem), base_link=base_link, joints=tuple(joints))


def _parse_urdf_with_yourdfpy(urdf_path: Path) -> ParsedUrdfModel:
    """Parse one URDF with yourdfpy when its optional dependencies are available."""
    from yourdfpy import URDF

    urdf_model = URDF.load(urdf_path, load_meshes=False, build_scene_graph=False)
    raw_base_link = urdf_model.base_link
    base_link = None if raw_base_link is None else str(raw_base_link)
    joints: list[ParsedJoint] = []
    parent_links: set[str] = set()
    child_links: set[str] = set()

    for joint in urdf_model.robot.joints:
        joint_type = getattr(joint, "type", "fixed")
        axis_default = (1.0, 0.0, 0.0) if joint_type in {"revolute", "prismatic", "continuous"} else (0.0, 0.0, 0.0)
        axis = tuple(float(value) for value in (joint.axis if joint.axis is not None else axis_default))
        limit = getattr(joint, "limit", None)
        has_limit = limit is not None and limit.lower is not None and limit.upper is not None
        joints.append(
            ParsedJoint(
                name=str(joint.name),
                joint_type=str(joint_type),
                parent_link=str(joint.parent),
                child_link=str(joint.child),
                axis=axis,
                origin_xyz=(0.0, 0.0, 0.0)
                if joint.origin is None
                else tuple(float(value) for value in joint.origin[:3, 3]),
                origin_rpy=(0.0, 0.0, 0.0),
                limit_lower=0.0 if not has_limit else float(limit.lower),
                limit_upper=0.0 if not has_limit else float(limit.upper),
                has_limit=bool(has_limit),
            )
        )
        parent_links.add(str(joint.parent))
        child_links.add(str(joint.child))

    if base_link in {None, "None"}:
        root_candidates = sorted(parent_links - child_links)
        if len(root_candidates) != 1:
            raise ValueError(f"Could not resolve a unique base link for URDF robot {urdf_path!s}: {root_candidates}.")
        base_link = root_candidates[0]
    assert base_link is not None

    return ParsedUrdfModel(robot_name=str(urdf_model.robot.name), base_link=base_link, joints=tuple(joints))


def parse_urdf_file(urdf_path: Path | str) -> ParsedUrdfModel:
    """Parse one URDF file, preferring yourdfpy and falling back to XML parsing."""
    urdf_path = Path(urdf_path)
    try:
        return _parse_urdf_with_yourdfpy(urdf_path)
    except ModuleNotFoundError as error:
        logger.info("Falling back to XML URDF parsing for %s because yourdfpy is unavailable: %s", urdf_path, error)
    except ImportError as error:
        logger.info("Falling back to XML URDF parsing for %s because yourdfpy import failed: %s", urdf_path, error)
    except Exception as error:
        logger.warning("yourdfpy parsing failed for %s. Falling back to XML parser: %s", urdf_path, error)

    return _parse_urdf_with_xml(urdf_path)
