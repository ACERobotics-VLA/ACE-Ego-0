"""Graph feature construction for static URDF conditioning."""

from __future__ import annotations

import logging
import math
from typing import Any

import networkx as nx
import torch

from ace_ego_0.model.modules.urdf.robot_urdf_registry import RobotUrdfSpec
from ace_ego_0.model.modules.urdf.urdf_parser import ParsedJoint, ParsedUrdfModel, parse_urdf_file

logger = logging.getLogger(__name__)

JOINT_TYPE_ORDER = ("revolute", "prismatic", "continuous", "fixed")
FEATURE_NAMES = [
    "joint_type_revolute",
    "joint_type_prismatic",
    "joint_type_continuous",
    "joint_type_fixed",
    "axis_x",
    "axis_y",
    "axis_z",
    "origin_x",
    "origin_y",
    "origin_z",
    "origin_roll_sin",
    "origin_pitch_sin",
    "origin_yaw_sin",
    "origin_roll_cos",
    "origin_pitch_cos",
    "origin_yaw_cos",
    "limit_lower",
    "limit_upper",
    "has_limit",
    "is_actuated",
    "depth",
    "degree",
    "parent_count",
    "child_count",
    "distance_to_left_eef",
    "distance_to_right_eef",
    "on_left_chain",
    "on_right_chain",
    "on_main_chain",
]


def _joint_type_one_hot(joint_type: str) -> list[float]:
    """Return a deterministic one-hot encoding for one URDF joint type."""
    return [1.0 if joint_type == expected_type else 0.0 for expected_type in JOINT_TYPE_ORDER]


def _build_relationship_maps(parsed_urdf: ParsedUrdfModel) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    """Build parent and child joint maps from the URDF link tree."""
    child_joint_by_link = {joint.child_link: joint.name for joint in parsed_urdf.joints}
    child_joints_by_parent_link: dict[str, list[str]] = {}
    parent_joint_by_joint: dict[str, str | None] = {}

    for joint in parsed_urdf.joints:
        child_joints_by_parent_link.setdefault(joint.parent_link, []).append(joint.name)
        parent_joint_by_joint[joint.name] = child_joint_by_link.get(joint.parent_link)

    return parent_joint_by_joint, child_joints_by_parent_link


def _extract_chain(parsed_urdf: ParsedUrdfModel, base_link: str, eef_link_name: str) -> list[str]:
    """Extract the ordered root-to-EEF joint chain for one EEF link."""
    joint_by_child_link = {joint.child_link: joint for joint in parsed_urdf.joints}
    known_links = {parsed_urdf.base_link, *(joint.parent_link for joint in parsed_urdf.joints)} | {
        joint.child_link for joint in parsed_urdf.joints
    }
    if eef_link_name not in known_links:
        raise ValueError(f"EEF link {eef_link_name!r} is not present in URDF robot {parsed_urdf.robot_name!r}.")
    if base_link not in known_links:
        raise ValueError(f"Base link {base_link!r} is not present in URDF robot {parsed_urdf.robot_name!r}.")

    chain: list[str] = []
    current_link = eef_link_name

    while current_link in joint_by_child_link:
        joint = joint_by_child_link[current_link]
        chain.append(joint.name)
        current_link = joint.parent_link

    chain.reverse()
    if current_link != base_link:
        raise ValueError(
            f"EEF chain for {eef_link_name!r} terminated at link {current_link!r}, "
            f"not configured base link {base_link!r}."
        )
    if not chain:
        raise ValueError(f"EEF link {eef_link_name!r} does not have a joint chain from base link {base_link!r}.")
    return chain


def _build_joint_graph(
    parsed_urdf: ParsedUrdfModel,
) -> tuple[nx.Graph, dict[str, int], dict[str, str | None], dict[str, list[str]]]:
    """Build one undirected joint graph from URDF parent-child link relations."""
    graph = nx.Graph()
    joint_name_to_index = {joint.name: index for index, joint in enumerate(parsed_urdf.joints)}
    parent_joint_by_joint, child_joints_by_parent_link = _build_relationship_maps(parsed_urdf)

    for joint in parsed_urdf.joints:
        graph.add_node(joint.name)

    for joint in parsed_urdf.joints:
        parent_joint_name = parent_joint_by_joint[joint.name]
        if parent_joint_name is not None:
            graph.add_edge(joint.name, parent_joint_name)

    return graph, joint_name_to_index, parent_joint_by_joint, child_joints_by_parent_link


def _compute_joint_depths(parsed_urdf: ParsedUrdfModel, parent_joint_by_joint: dict[str, str | None]) -> dict[str, int]:
    """Compute root-to-joint depth in joint-count units."""
    joint_depths: dict[str, int] = {}
    for joint in parsed_urdf.joints:
        depth = 0
        parent_joint_name = parent_joint_by_joint[joint.name]
        while parent_joint_name is not None:
            depth += 1
            parent_joint_name = parent_joint_by_joint[parent_joint_name]
        joint_depths[joint.name] = depth
    return joint_depths


def _compute_distance_features(
    graph: nx.Graph,
    joint_names: list[str],
    left_chain: list[str],
    right_chain: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute shortest-path distances from each joint to each arm EEF joint."""
    left_target = left_chain[-1] if left_chain else None
    right_target = right_chain[-1] if right_chain else None
    left_distances: dict[str, int] = {}
    right_distances: dict[str, int] = {}

    for joint_name in joint_names:
        if left_target is None:
            left_distances[joint_name] = -1
        else:
            try:
                left_distances[joint_name] = int(nx.shortest_path_length(graph, joint_name, left_target))
            except nx.NetworkXNoPath:
                left_distances[joint_name] = -1

        if right_target is None:
            right_distances[joint_name] = -1
        else:
            try:
                right_distances[joint_name] = int(nx.shortest_path_length(graph, joint_name, right_target))
            except nx.NetworkXNoPath:
                right_distances[joint_name] = -1

    return left_distances, right_distances


def _build_adjacency_tensor(graph: nx.Graph, joint_names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct adjacency and row-normalized adjacency tensors for message passing."""
    adjacency = torch.zeros((len(joint_names), len(joint_names)), dtype=torch.float32)
    for source_name, target_name in graph.edges():
        source_index = joint_names.index(source_name)
        target_index = joint_names.index(target_name)
        adjacency[source_index, target_index] = 1.0
        adjacency[target_index, source_index] = 1.0

    adjacency_with_self = adjacency + torch.eye(len(joint_names), dtype=torch.float32)
    row_sums = adjacency_with_self.sum(dim=-1, keepdim=True).clamp_min(1.0)
    adjacency_norm = adjacency_with_self / row_sums
    return adjacency, adjacency_norm


def _encode_joint_features(
    joint: ParsedJoint,
    *,
    graph: nx.Graph,
    parent_joint_by_joint: dict[str, str | None],
    child_joints_by_parent_link: dict[str, list[str]],
    joint_depths: dict[str, int],
    left_distances: dict[str, int],
    right_distances: dict[str, int],
    left_chain_set: set[str],
    right_chain_set: set[str],
) -> list[float]:
    """Encode one joint into the dense feature vector used by the v1 encoder."""
    parent_joint_count = 1 if parent_joint_by_joint[joint.name] is not None else 0
    child_joint_count = len(child_joints_by_parent_link.get(joint.child_link, []))
    degree = int(graph.degree(joint.name))
    on_left_chain = 1.0 if joint.name in left_chain_set else 0.0
    on_right_chain = 1.0 if joint.name in right_chain_set else 0.0
    on_main_chain = 1.0 if on_left_chain or on_right_chain else 0.0
    roll, pitch, yaw = joint.origin_rpy

    return [
        *_joint_type_one_hot(joint.joint_type),
        *joint.axis,
        *joint.origin_xyz,
        math.sin(roll),
        math.sin(pitch),
        math.sin(yaw),
        math.cos(roll),
        math.cos(pitch),
        math.cos(yaw),
        joint.limit_lower,
        joint.limit_upper,
        1.0 if joint.has_limit else 0.0,
        0.0 if joint.joint_type == "fixed" else 1.0,
        float(joint_depths[joint.name]),
        float(degree),
        float(parent_joint_count),
        float(child_joint_count),
        float(left_distances[joint.name]),
        float(right_distances[joint.name]),
        on_left_chain,
        on_right_chain,
        on_main_chain,
    ]


def build_urdf_graph_tensors(spec: RobotUrdfSpec) -> dict[str, Any]:
    """Build the cached URDF graph tensors for one registered robot."""
    parsed_urdf = parse_urdf_file(spec.urdf_path)
    graph, joint_name_to_index, parent_joint_by_joint, child_joints_by_parent_link = _build_joint_graph(parsed_urdf)
    joint_names = [joint.name for joint in parsed_urdf.joints]
    joint_depths = _compute_joint_depths(parsed_urdf, parent_joint_by_joint)

    left_eef_link = spec.eef_link_names[0]
    right_eef_link = spec.eef_link_names[1] if len(spec.eef_link_names) > 1 else spec.eef_link_names[0]
    left_chain = _extract_chain(parsed_urdf, spec.base_link, left_eef_link)
    right_chain = _extract_chain(parsed_urdf, spec.base_link, right_eef_link)
    left_distances, right_distances = _compute_distance_features(graph, joint_names, left_chain, right_chain)
    adjacency, adjacency_norm = _build_adjacency_tensor(graph, joint_names)

    left_chain_set = set(left_chain)
    right_chain_set = set(right_chain)
    node_features = torch.tensor(
        [
            _encode_joint_features(
                joint,
                graph=graph,
                parent_joint_by_joint=parent_joint_by_joint,
                child_joints_by_parent_link=child_joints_by_parent_link,
                joint_depths=joint_depths,
                left_distances=left_distances,
                right_distances=right_distances,
                left_chain_set=left_chain_set,
                right_chain_set=right_chain_set,
            )
            for joint in parsed_urdf.joints
        ],
        dtype=torch.float32,
    )

    left_chain_mask = torch.tensor([joint_name in left_chain_set for joint_name in joint_names], dtype=torch.bool)
    right_chain_mask = torch.tensor([joint_name in right_chain_set for joint_name in joint_names], dtype=torch.bool)
    chain_mask = left_chain_mask | right_chain_mask

    return {
        "robot_name": spec.robot_name,
        "urdf_path": str(spec.urdf_path),
        "base_link": spec.base_link,
        "eef_link_names": list(spec.eef_link_names),
        "joint_names": joint_names,
        "left_chain_joint_names": left_chain,
        "right_chain_joint_names": right_chain,
        "feature_names": FEATURE_NAMES,
        "node_features": node_features,
        "adjacency": adjacency,
        "adjacency_norm": adjacency_norm,
        "left_chain_mask": left_chain_mask,
        "right_chain_mask": right_chain_mask,
        "chain_mask": chain_mask,
        "feature_dim": int(node_features.shape[-1]),
        "num_nodes": int(node_features.shape[0]),
        "joint_name_to_index": joint_name_to_index,
    }
