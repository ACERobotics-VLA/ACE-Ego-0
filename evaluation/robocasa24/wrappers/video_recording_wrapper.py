import os
import uuid
from pathlib import Path

import av
import gymnasium as gym
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: list[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int | None = 0,
    allow_negative: bool = False,
) -> tuple[list[int], list[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = np.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile:v": "high"},
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, _global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, _c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for _i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.file_path = None

        self.step_count = 0

        self.is_success = False
        self.episode_idx = 0
        self.task_name = None  # Will be set from env_name

    def _finalize_recording(self) -> None:
        """Stop and rename the current episode recording, if one exists."""
        self.video_recorder.stop()
        if self.video_dir is None or self.file_path is None or not self.file_path.exists():
            self.file_path = None
            return

        status = "success" if self.is_success else "fail"
        if self.task_name:
            new_filestem = f"{self.task_name}_ep{self.episode_idx:03d}_{status}"
        else:
            new_filestem = f"{self.file_path.stem}_{status}"
        new_file_path = self.video_dir / f"{new_filestem}.mp4"
        os.replace(self.file_path, new_file_path)
        self.episode_idx += 1
        self.file_path = None

    def set_task_name(self, task_name: str):
        """Set the task name for video file naming."""
        # Extract short task name from full env name like "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
        if "/" in task_name:
            task_name = task_name.split("/")[-1]
        # Remove robot suffix if present
        if "_GR1" in task_name:
            task_name = task_name.split("_GR1")[0]
        elif "_Env" in task_name:
            task_name = task_name.replace("_Env", "")
        self.task_name = task_name

    def reset(self, **kwargs):
        self._finalize_recording()
        result = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.is_success = False
        if self.video_dir is not None:
            self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
            self.video_recorder.start(self.file_path)
            frame = self.env.render()
            if frame.dtype != np.uint8:
                raise TypeError(f"Expected uint8 reset frame, got {frame.dtype}.")
            self.video_recorder.write_frame(frame)
        return result

    def step(self, action):
        result = super().step(action)
        self.step_count += 1

        # Always check success status, not just on render frames
        # result is (obs, reward, done, truncated, info) - info is at index -1
        info = result[-1]
        if "success" in info:
            success_val = info["success"]
            # Handle different formats: bool, array, or nested
            if isinstance(success_val, bool):
                self.is_success |= success_val
            elif hasattr(success_val, "__iter__"):
                # If it's an array/list, check if any value is True
                self.is_success |= bool(any(success_val))
            else:
                self.is_success |= bool(success_val)

        # Only render on specified intervals
        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.video_recorder.write_frame(frame)
        return result

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path

    def close(self):
        """Finalize the active recording before closing the wrapped environment."""
        self._finalize_recording()
        super().close()
