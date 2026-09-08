"""
URDF processing utilities for IK.

Adapted from teleoperation project: teleo.robots.sim_gr1_eef_replay
"""

import logging
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yourdfpy

logger = logging.getLogger(__name__)


def fix_continuous_joint_limits(urdf_path: Path | str) -> Path:
    """Fix URDF by adding default limits to continuous joints.

    Pyroki requires all joints to have velocity limits, but continuous joints
    in URDF often don't have them.

    Args:
        urdf_path: Path to original URDF file (can be str or Path).

    Returns:
        Path to fixed URDF file (in temp directory).
    """
    # Convert to Path if string
    if isinstance(urdf_path, str):
        urdf_path = Path(urdf_path)

    with open(urdf_path, "r") as f:
        urdf_content = f.read()

    # Pattern to match continuous joints
    joint_pattern = re.compile(
        r'(<joint\s+(?:[^>]*\s+)?name="([^"]+)"[^>]*type="continuous"[^>]*>.*?)'
        r"(</joint>)",
        re.DOTALL,
    )

    def add_limits(match):
        joint_block = match.group(1)
        closing_tag = match.group(3)

        if "<limit" in joint_block:
            return match.group(0)

        limit_tag = (
            '    <limit\n      lower="-3.1416"\n      upper="3.1416"\n      effort="100"\n      velocity="31.4" />\n  '
        )
        return joint_block + limit_tag + closing_tag

    fixed_content = joint_pattern.sub(add_limits, urdf_content)

    temporary_dir = urdf_path.parent / ".gr1_ik_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False, dir=temporary_dir)
    temp_file.write(fixed_content)
    temp_file.close()

    logger.info(f"Fixed URDF written to {temp_file.name}")
    return Path(temp_file.name)


def create_sub_urdf(
    urdf_path: Path | str,
    base_link: str,
    end_link_or_output: str | Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create a sub-URDF with base_link as the root.

    This extracts only the links and joints from base_link to end_link,
    effectively making base_link the new root of the kinematic chain.

    Args:
        urdf_path: Original URDF file (can be str or Path).
        base_link: Name of the link to use as new root.
        end_link_or_output: Either end_link (str) or output_path (Path) for backward compatibility.
        output_path: Path to write the sub-URDF (optional, auto-generates temp file if None).

    Returns:
        Path to the created sub-URDF.
    """
    # Convert to Path if string
    if isinstance(urdf_path, str):
        urdf_path = Path(urdf_path)

    # Handle backward compatibility: if third arg is a Path, treat it as output_path
    end_link = None
    if end_link_or_output is not None:
        if isinstance(end_link_or_output, Path):
            output_path = end_link_or_output
        elif isinstance(end_link_or_output, str):
            # Check if it looks like a path (contains / or ends with .urdf)
            if "/" in end_link_or_output or end_link_or_output.endswith(".urdf"):
                output_path = Path(end_link_or_output)
            else:
                end_link = end_link_or_output

    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=False,
        load_collision_meshes=False,
    )

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    active_links = set()
    active_joints = set()

    if end_link is not None:
        # Find path from base_link to end_link
        def find_path_to_end(current_link, target_link, path_links, path_joints):
            if current_link == target_link:
                return True
            for joint in urdf.joint_map.values():
                if joint.parent == current_link:
                    if find_path_to_end(joint.child, target_link, path_links, path_joints):
                        path_links.add(current_link)
                        path_links.add(joint.child)
                        path_joints.add(joint.name)
                        return True
            return False

        find_path_to_end(base_link, end_link, active_links, active_joints)
    else:
        # Traverse all children from base_link
        def traverse_link(link_name):
            if link_name in active_links:
                return
            active_links.add(link_name)

            for joint in urdf.joint_map.values():
                if joint.parent == link_name:
                    active_joints.add(joint.name)
                    traverse_link(joint.child)

        traverse_link(base_link)

    new_robot = ET.Element("robot", name=f"{urdf_path.stem}_sub")

    for link_name in active_links:
        for link_elem in root.findall("link"):
            if link_elem.get("name") == link_name:
                new_robot.append(link_elem)
                break

    for joint_name in active_joints:
        for joint_elem in root.findall("joint"):
            if joint_elem.get("name") == joint_name:
                new_robot.append(joint_elem)
                break

    # Auto-generate output path if not provided
    if output_path is None:
        suffix = f"_{base_link}_to_{end_link}" if end_link else f"_{base_link}"
        temporary_dir = urdf_path.parent / ".gr1_ik_tmp"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".urdf", prefix=f"{urdf_path.stem}{suffix}_", delete=False, dir=temporary_dir
        )
        output_path = Path(temp_file.name)
        temp_file.close()

    new_tree = ET.ElementTree(new_robot)
    new_tree.write(output_path, encoding="utf-8", xml_declaration=True)

    logger.info(f"Created sub-URDF with {len(active_links)} links and {len(active_joints)} joints")
    logger.info(f"Base link: {base_link}, End link: {end_link}, Output: {output_path}")

    return output_path
