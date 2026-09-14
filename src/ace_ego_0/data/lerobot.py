"""Small, read-only LeRobot v2.x reader for RoboCasa24 and ARX SFT."""

# The LeRobot disk and PyAV access contract is adapted from NVIDIA GR00T;
# see THIRD_PARTY_NOTICES.md and Apache-2.0 upstream notices.

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ace_ego_0.data.common23 import (
    Common23Norm,
    transform_arx_common23,
    transform_robocasa_common23,
)

SUPPORTED_ROBOT_TYPES = {
    "fourier_gr1_eef_gripper_6d",
    "arx_dual_arm_common23",
}
ROBOCASA_PARQUET_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "state.left_eef_pos_cam",
    "state.left_eef_rot6d_cam",
    "state.right_eef_pos_cam",
    "state.right_eef_rot6d_cam",
    "action.left_eef_pos_cam",
    "action.left_eef_rot6d_cam",
    "action.right_eef_pos_cam",
    "action.right_eef_rot6d_cam",
)
ARX_PARQUET_COLUMNS = (
    "observation.state",
    "actions",
    "prompt",
    "timestamp",
)
ROBOCASA_VIDEO_KEYS = ("ego_view",)
ARX_VIDEO_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def ordered_video_keys(robot_type: str) -> tuple[str, ...]:
    """Return the paper/eval camera order for one public SFT robot type."""
    if robot_type == "fourier_gr1_eef_gripper_6d":
        return ROBOCASA_VIDEO_KEYS
    if robot_type == "arx_dual_arm_common23":
        return ARX_VIDEO_KEYS
    raise ValueError(
        f"Unsupported public SFT robot_type={robot_type!r}; "
        f"expected one of {sorted(SUPPORTED_ROBOT_TYPES)}."
    )


def ordered_video_entries(modality: Mapping[str, Any], robot_type: str) -> list[Mapping[str, Any]]:
    """Walk `modality.json` video entries in the pinned camera order."""
    video = modality.get("video")
    if not isinstance(video, dict):
        raise ValueError("modality.json must contain a video mapping.")
    entries: list[Mapping[str, Any]] = []
    for key in ordered_video_keys(robot_type):
        entry = video.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"modality.json is missing video key {key!r}.")
        entries.append(entry)
    return entries


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required LeRobot {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"LeRobot {name} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required LeRobot {name}: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{name} line {line_number} must be a JSON object: {path}")
            records.append(value)
    return records


def _scalar(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected a scalar timestamp, got shape {array.shape}.")
    return float(array[0])


def _strip_waist_prefix(language: str) -> str:
    stripped = str(language).strip()
    for prefix in ("locked_waist:", "unlocked_waist:"):
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _decode_video_frame(video_path: Path, timestamp: float) -> Image.Image:
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot video: {video_path}")
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        container.seek(max(int(timestamp / time_base), 0), stream=stream)
        closest_frame = None
        closest_difference = float("inf")
        for frame in container.decode(stream):
            frame_timestamp = float(frame.pts * time_base)
            difference = abs(frame_timestamp - timestamp)
            if difference <= closest_difference:
                closest_frame = frame
                closest_difference = difference
            elif frame_timestamp > timestamp:
                break
        if closest_frame is None:
            raise ValueError(f"Unable to decode frame at {timestamp:.6f}s from {video_path}")
        return Image.fromarray(closest_frame.to_ndarray(format="rgb24"))
    finally:
        container.close()


def _resolve_config_path(config_path: Any, *, base_dir: Path) -> Path:
    path = Path(str(config_path)).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _parse_dataset_manifest(
    manifest_path: Path,
    *,
    default_robot_type: str | None,
) -> list[tuple[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    if manifest_path.suffix.lower() in {".yaml", ".yml"}:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        entries = manifest.get("datasets") if isinstance(manifest, dict) else manifest
        if not isinstance(entries, list):
            raise ValueError(f"YAML dataset manifest must contain a list: {manifest_path}")
        resolved: list[tuple[str, str]] = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not entry.get("name"):
                raise ValueError(f"Every dataset manifest entry must define `name`: {manifest_path}")
            robot_type = str(entry.get("robot_type") or default_robot_type or "")
            resolved.append((str(entry["name"]), robot_type))
        return resolved

    resolved = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        resolved.append((stripped, str(default_robot_type or "")))
    if not resolved:
        raise ValueError(f"Dataset manifest contains no dataset paths: {manifest_path}")
    return resolved


class LeRobotSFTDataset(Dataset):
    """One reviewed RoboCasa or ARX dataset root."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        robot_type: str,
        norm: Common23Norm,
        action_horizon: int,
        image_size: Sequence[int],
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.robot_type = str(robot_type)
        if self.robot_type not in SUPPORTED_ROBOT_TYPES:
            raise ValueError(
                f"Unsupported public SFT robot_type={self.robot_type!r}; "
                f"expected one of {sorted(SUPPORTED_ROBOT_TYPES)}."
            )
        self.norm = norm
        self.action_horizon = int(action_horizon)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
            raise ValueError(f"image_size must be [width, height], got {image_size!r}.")
        self.image_size = (int(image_size[0]), int(image_size[1]))

        meta_dir = self.dataset_path / "meta"
        self.info = _read_json(meta_dir / "info.json", name="info.json")
        self.modality = _read_json(meta_dir / "modality.json", name="modality.json")
        self.episodes = _read_jsonl(meta_dir / "episodes.jsonl", name="episodes.jsonl")
        if str(self.info.get("codebase_version")) not in {"v2.0", "v2.1"}:
            raise ValueError(
                f"Unsupported LeRobot codebase_version={self.info.get('codebase_version')!r} "
                f"in {meta_dir / 'info.json'}."
            )
        if len(self.episodes) != int(self.info.get("total_episodes", -1)):
            raise ValueError(f"episodes.jsonl count does not match info.json in {self.dataset_path}.")
        self.episode_lengths = [int(record["length"]) for record in self.episodes]
        self.cumulative_lengths = np.cumsum(self.episode_lengths).tolist()
        if not self.cumulative_lengths or self.cumulative_lengths[-1] != int(self.info.get("total_frames", -1)):
            raise ValueError(f"Episode lengths do not match total_frames in {self.dataset_path}.")
        self._table_cache: OrderedDict[int, dict[str, list[Any]]] = OrderedDict()
        self._validate_modality_contract()

    def _validate_modality_contract(self) -> None:
        if self.robot_type == "fourier_gr1_eef_gripper_6d":
            if str(self.info.get("robot_type")) != "GR1ArmsAndWaistFourierHands":
                raise ValueError(f"Unexpected RoboCasa robot_type in {self.dataset_path / 'meta/info.json'}.")
            expected_video = set(ROBOCASA_VIDEO_KEYS)
            required_state = {"left_hand", "right_hand", "waist"}
            required_action = {"left_hand", "right_hand", "waist"}
        else:
            if str(self.info.get("robot_type")) != "arx_dual_arm":
                raise ValueError(f"Unexpected ARX robot_type in {self.dataset_path / 'meta/info.json'}.")
            expected_video = set(ARX_VIDEO_KEYS)
            required_state = {"left_eef", "left_gripper", "right_eef", "right_gripper"}
            required_action = required_state
        video = self.modality.get("video")
        state = self.modality.get("state")
        action = self.modality.get("action")
        if not isinstance(video, dict) or set(video) != expected_video:
            raise ValueError(f"Unexpected video modality contract in {self.dataset_path}.")
        if not isinstance(state, dict) or not required_state.issubset(state):
            raise ValueError(f"Missing required state modalities in {self.dataset_path}.")
        if not isinstance(action, dict) or not required_action.issubset(action):
            raise ValueError(f"Missing required action modalities in {self.dataset_path}.")

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def _resolve_frame(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.cumulative_lengths, index)
        previous_total = 0 if episode_index == 0 else self.cumulative_lengths[episode_index - 1]
        return episode_index, index - previous_total

    def _episode_path(self, template_key: str, episode_index: int, **values: Any) -> Path:
        template = self.info.get(template_key)
        if not isinstance(template, str):
            raise ValueError(f"info.json is missing {template_key!r} in {self.dataset_path}.")
        chunk_size = int(self.info.get("chunks_size", 1000))
        relative_path = template.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            **values,
        )
        return self.dataset_path / relative_path

    def _load_episode_columns(self, episode_index: int) -> dict[str, list[Any]]:
        cached = self._table_cache.get(episode_index)
        if cached is not None:
            self._table_cache.move_to_end(episode_index)
            return cached
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError("Training data loading requires `pip install -e '.[train]'`.") from error

        columns = ROBOCASA_PARQUET_COLUMNS if self.robot_type == "fourier_gr1_eef_gripper_6d" else ARX_PARQUET_COLUMNS
        parquet_path = self._episode_path("data_path", episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing LeRobot parquet episode: {parquet_path}")
        table = pq.read_table(parquet_path, columns=list(columns))
        loaded = {column: table[column].to_pylist() for column in columns}
        expected_length = self.episode_lengths[episode_index]
        if any(len(values) != expected_length for values in loaded.values()):
            raise ValueError(f"Parquet row count does not match episodes.jsonl: {parquet_path}")
        self._table_cache[episode_index] = loaded
        while len(self._table_cache) > 2:
            self._table_cache.popitem(last=False)
        return loaded

    def _load_images(self, episode_index: int, timestamp: float) -> list[Image.Image]:
        images: list[Image.Image] = []
        for video_entry in ordered_video_entries(self.modality, self.robot_type):
            original_key = str(video_entry["original_key"])
            video_path = self._episode_path("video_path", episode_index, video_key=original_key)
            images.append(_decode_video_frame(video_path, timestamp).resize(self.image_size))
        return images

    def _robocasa_sample(
        self,
        episode_index: int,
        frame_index: int,
        columns: dict[str, list[Any]],
    ) -> dict[str, Any]:
        last_frame = self.episode_lengths[episode_index] - 1
        action_indices = [min(frame_index + offset, last_frame) for offset in range(self.action_horizon)]
        state_vector = np.asarray(columns["observation.state"][frame_index], dtype=np.float32)
        action_vectors = np.asarray([columns["action"][index] for index in action_indices], dtype=np.float32)
        state_components = {
            "left_eef_pos": np.asarray(columns["state.left_eef_pos_cam"][frame_index]),
            "left_eef_rot6d": np.asarray(columns["state.left_eef_rot6d_cam"][frame_index]),
            "right_eef_pos": np.asarray(columns["state.right_eef_pos_cam"][frame_index]),
            "right_eef_rot6d": np.asarray(columns["state.right_eef_rot6d_cam"][frame_index]),
            "left_hand": state_vector[7:13],
            "right_hand": state_vector[29:35],
            "waist": state_vector[41:44],
        }
        action_components = {
            "left_eef_pos": np.asarray([columns["action.left_eef_pos_cam"][index] for index in action_indices]),
            "left_eef_rot6d": np.asarray([columns["action.left_eef_rot6d_cam"][index] for index in action_indices]),
            "right_eef_pos": np.asarray([columns["action.right_eef_pos_cam"][index] for index in action_indices]),
            "right_eef_rot6d": np.asarray([columns["action.right_eef_rot6d_cam"][index] for index in action_indices]),
            "left_hand": action_vectors[:, 7:13],
            "right_hand": action_vectors[:, 29:35],
            "waist": action_vectors[:, 41:44],
        }
        state, action, mask, state_mask = transform_robocasa_common23(
            state_components=state_components,
            action_components=action_components,
            norm=self.norm,
        )
        episode_record = self.episodes[episode_index]
        language = episode_record.get("remarks") or episode_record.get("description")
        if not language:
            tasks = episode_record.get("tasks", [])
            language = tasks[0] if tasks else ""
        timestamp = _scalar(columns["timestamp"][frame_index])
        return {
            "image": self._load_images(episode_index, timestamp),
            "lang": _strip_waist_prefix(str(language)),
            "action": action,
            "state": state,
            "mask": mask,
            "state_mask": state_mask,
            "robot_name": "GR1",
            "robot_type": self.robot_type,
        }

    def _arx_sample(
        self,
        episode_index: int,
        frame_index: int,
        columns: dict[str, list[Any]],
    ) -> dict[str, Any]:
        last_frame = self.episode_lengths[episode_index] - 1
        action_indices = [min(frame_index + offset, last_frame) for offset in range(self.action_horizon)]
        state, action, mask, state_mask = transform_arx_common23(
            raw_state=np.asarray(columns["observation.state"][frame_index]),
            raw_actions=np.asarray([columns["actions"][index] for index in action_indices]),
            norm=self.norm,
        )
        language = str(columns["prompt"][frame_index]).strip()
        if not language:
            tasks = self.episodes[episode_index].get("tasks", [])
            language = str(tasks[0]) if tasks else ""
        timestamp = _scalar(columns["timestamp"][frame_index])
        return {
            "image": self._load_images(episode_index, timestamp),
            "lang": language,
            "action": action,
            "state": state,
            "mask": mask,
            "state_mask": state_mask,
            "robot_name": "ARX5Dual",
            "robot_type": self.robot_type,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self._resolve_frame(index)
        columns = self._load_episode_columns(episode_index)
        if self.robot_type == "fourier_gr1_eef_gripper_6d":
            sample = self._robocasa_sample(episode_index, frame_index, columns)
        else:
            sample = self._arx_sample(episode_index, frame_index, columns)
        sample["dataset_name"] = self.dataset_path.name
        sample["domain"] = "robocasa24" if self.robot_type == "fourier_gr1_eef_gripper_6d" else "arx"
        return sample


def identity_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep heterogeneous image lists and metadata as model-ready samples."""
    return batch


def build_sft_dataset(config: Any, *, config_path: str | Path | None = None) -> Dataset:
    """Build the reviewed all-samples-once mixture from one public recipe."""
    config_dir = Path(config_path).expanduser().resolve().parent if config_path is not None else Path.cwd()
    data_config = config.datasets.vla_data
    data_root = _resolve_config_path(data_config.data_root_dir, base_dir=config_dir)
    manifest_path = _resolve_config_path(data_config.dataset_manifest_path, base_dir=config_dir)
    robot_type = str(data_config.get("manifest_robot_type", ""))
    datasets = _parse_dataset_manifest(manifest_path, default_robot_type=robot_type or None)
    if not datasets:
        raise ValueError(f"No datasets configured by {manifest_path}.")

    action_mode = "delta" if bool(data_config.get("delta_action", False)) else "absolute"
    norm_manifest_path = _resolve_config_path(data_config.norm_manifest_path, base_dir=config_dir)
    resolved_robot_types = {entry_robot_type for _, entry_robot_type in datasets}
    if len(resolved_robot_types) != 1:
        raise ValueError("A public SFT recipe must use one Common23 robot_type and one norm manifest.")
    resolved_robot_type = next(iter(resolved_robot_types))
    norm = Common23Norm.from_manifest(
        norm_manifest_path,
        expected_action_mode=action_mode,
        expected_robot_type=resolved_robot_type,
    )
    action_horizon = int(data_config.action_horizon)
    image_size = data_config.image_size
    dataset_objects = [
        LeRobotSFTDataset(
            data_root / relative_path,
            robot_type=entry_robot_type,
            norm=norm,
            action_horizon=action_horizon,
            image_size=image_size,
        )
        for relative_path, entry_robot_type in datasets
    ]
    return dataset_objects[0] if len(dataset_objects) == 1 else ConcatDataset(dataset_objects)


def build_sft_dataloader(config: Any, *, config_path: str | Path | None = None) -> DataLoader:
    """Build a shuffled identity-collated DataLoader without any stats side effects."""
    data_config = config.datasets.vla_data
    dataset = build_sft_dataset(config, config_path=config_path)
    num_workers = int(data_config.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(data_config.per_device_batch_size),
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": identity_collate,
        "pin_memory": True,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(data_config.get("persistent_workers", True))
    return DataLoader(**kwargs)
