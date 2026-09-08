"""Command-line entry points for ACE-Ego-0 RoboCasa 24 evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

import websockets.exceptions
import websockets.sync.client

from evaluation.robocasa24.checkpoint import CheckpointContract, inspect_checkpoint
from evaluation.robocasa24.tasks import ROBOCASA24_TASKS, select_tasks, task_env_name

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
EVALUATION_ROOT = REPO_ROOT / "evaluation" / "robocasa24"
DEFAULT_URDF_PATH = REPO_ROOT / "assets" / "GR1T2_with_hands.urdf"
DEFAULT_VLM_PATH = REPO_ROOT / "assets" / "qwen3-vl-4b-instruct"
EPISODE_RESULT_PATTERN = re.compile(r"Episode\s+\d+/\d+:\s+(SUCCESS|FAILED)")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def _task_range(args: argparse.Namespace) -> tuple[int, int]:
    if args.task_index is not None:
        return args.task_index, args.task_index
    return (
        0 if args.task_start is None else args.task_start,
        len(ROBOCASA24_TASKS) - 1 if args.task_end is None else args.task_end,
    )


def _runtime_environment(base_vlm: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    import_paths = [str(REPO_ROOT), str(SRC_ROOT)]
    if python_path:
        import_paths.append(python_path)
    environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    environment.setdefault("WANDB_MODE", "disabled")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ACE_EGO_0_BASE_VLM"] = base_vlm or str(DEFAULT_VLM_PATH)
    return environment


def _server_command(
    contract: CheckpointContract,
    host: str,
    port: int,
    urdf_path: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATION_ROOT / "server/server_policy.py"),
        "--ckpt_path",
        str(contract.weights_path),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if urdf_path is not None:
        command.extend(["--urdf", str(urdf_path)])
    return command


def _evaluation_command(
    args: argparse.Namespace,
    contract: CheckpointContract,
    task_name: str,
    video_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(EVALUATION_ROOT / "simulation_env.py"),
        "--args.env-name",
        task_env_name(task_name),
        "--args.host",
        args.host,
        "--args.port",
        str(args.port),
        "--args.n-episodes",
        str(args.episodes),
        "--args.n-envs",
        "1",
        "--args.max-episode-steps",
        str(args.max_episode_steps),
        "--args.n-action-steps",
        str(contract.action_horizon),
        "--args.n-exec-steps",
        str(args.exec_steps),
        "--args.video-out-path",
        str(video_dir),
        "--args.pretrained-path",
        str(contract.weights_path),
        "--args.ik-urdf-path",
        str(args.urdf),
        "--args.ik-learning-rate",
        str(args.ik_learning_rate),
        "--args.ik-max-iterations",
        str(args.ik_max_iterations),
        "--args.ik-pose-tolerance",
        str(args.ik_pose_tolerance),
        "--args.ik-unreachable-threshold",
        str(args.ik_unreachable_threshold),
        "--args.seed",
        str(args.seed),
    ]
    if contract.action_mode == "delta":
        command.extend(
            [
                "--args.use-delta-action",
                "--args.delta-base-state",
                str(contract.delta_base_state),
                "--args.delta-rot-norm-mode",
                contract.rot_norm_mode,
                "--args.delta-action-mode",
                str(contract.delta_action_mode),
                "--args.delta-quat-mode",
                str(contract.delta_quat_mode),
            ]
        )
    return command


def _wait_for_server(host: str, port: int, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited during startup with code {process.returncode}.")
        try:
            with websockets.sync.client.connect(
                f"ws://{host}:{port}",
                compression=None,
                proxy=None,
                open_timeout=1,
                close_timeout=1,
            ) as connection:
                connection.recv(timeout=1)
                return
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException):
            time.sleep(2)
    raise TimeoutError(f"Policy server did not become ready within {timeout} seconds.")


@contextmanager
def _managed_server(
    args: argparse.Namespace,
    contract: CheckpointContract,
    environment: dict[str, str],
    log_file: IO[str],
) -> Iterator[None]:
    if args.connect_only:
        yield
        return
    process = subprocess.Popen(
        _server_command(contract, args.host, args.port, args.urdf),
        cwd=REPO_ROOT,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(args.host, args.port, process, args.server_timeout)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_and_tee(command: list[str], environment: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return process.wait()


def _parse_task_result(task_index: int, task_name: str, returncode: int, log_path: Path) -> dict[str, object]:
    log_text = log_path.read_text(encoding="utf-8")
    episode_results = EPISODE_RESULT_PATTERN.findall(log_text)
    successes = episode_results.count("SUCCESS")
    failures = episode_results.count("FAILED")
    return {
        "task_index": task_index,
        "task_name": task_name,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / len(episode_results) if episode_results else 0.0,
        "returncode": returncode,
        "log": str(log_path),
    }


def _write_summary(
    output_dir: Path,
    contract: CheckpointContract,
    episodes: int,
    results: list[dict[str, object]],
) -> Path:
    total_successes = sum(int(result["successes"]) for result in results)
    total_episodes = sum(int(result["successes"]) + int(result["failures"]) for result in results)
    summary = {
        "checkpoint": str(contract.weights_path),
        "action_mode": contract.action_mode,
        "episodes_per_task": episodes,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "tasks": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def build_evaluate_parser() -> argparse.ArgumentParser:
    """Build the public evaluation argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate ACE-Ego-0 checkpoints on the GR1 RoboCasa 24 tasks.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-start", type=int)
    parser.add_argument("--task-end", type=int)
    parser.add_argument("--episodes", type=_positive_int, default=50)
    parser.add_argument("--max-episode-steps", type=_positive_int, default=600)
    parser.add_argument("--exec-steps", type=_positive_int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Connect to a local policy server started separately with scripts/run_server.py.",
    )
    parser.add_argument("--base-vlm")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--server-timeout", type=_positive_int, default=600)
    parser.add_argument("--ik-learning-rate", type=float, default=0.05)
    parser.add_argument("--ik-max-iterations", type=_positive_int, default=2000)
    parser.add_argument("--ik-pose-tolerance", type=float, default=0.005)
    parser.add_argument("--ik-unreachable-threshold", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def evaluate_main(argv: list[str] | None = None) -> int:
    """Run one task slice and save structured results."""
    args = build_evaluate_parser().parse_args(argv)
    contract = inspect_checkpoint(args.checkpoint)
    if not args.urdf.is_file():
        raise FileNotFoundError(f"Mink IK URDF does not exist: {args.urdf}")
    task_start, task_end = _task_range(args)
    selected_tasks = select_tasks(task_start, task_end)
    environment = _runtime_environment(args.base_vlm)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{contract.root.name}_{task_start}-{task_end}_{timestamp}"

    if args.dry_run:
        print(json.dumps({"server": _server_command(contract, args.host, args.port, args.urdf)}, indent=2))
        for task_index, task_name in selected_tasks:
            command = _evaluation_command(args, contract, task_name, output_dir / "videos" / task_name)
            print(json.dumps({"task_index": task_index, "command": command}, indent=2))
        return 0

    (output_dir / "logs").mkdir(parents=True)
    (output_dir / "videos").mkdir()
    results: list[dict[str, object]] = []
    with (output_dir / "logs/server.log").open("w", encoding="utf-8") as server_log:
        with _managed_server(args, contract, environment, server_log):
            for task_index, task_name in selected_tasks:
                logger.info("Evaluating task %d/%d: %s", task_index, len(ROBOCASA24_TASKS) - 1, task_name)
                video_dir = output_dir / "videos" / task_name
                video_dir.mkdir()
                log_path = output_dir / "logs" / f"{task_index:02d}_{task_name}.log"
                command = _evaluation_command(args, contract, task_name, video_dir)
                returncode = _run_and_tee(command, environment, log_path)
                results.append(_parse_task_result(task_index, task_name, returncode, log_path))

    summary_path = _write_summary(output_dir, contract, args.episodes, results)
    logger.info("Evaluation summary: %s", summary_path)
    return int(any(int(result["returncode"]) != 0 for result in results))


def run_server_main(argv: list[str] | None = None) -> int:
    """Start a policy server for a validated checkpoint bundle."""
    parser = argparse.ArgumentParser(description="Start the ACE-Ego-0 RoboCasa 24 policy server.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--base-vlm")
    args = parser.parse_args(argv)
    contract = inspect_checkpoint(args.checkpoint)
    if not args.urdf.is_file():
        raise FileNotFoundError(f"GR1 URDF does not exist: {args.urdf}")
    environment = _runtime_environment(args.base_vlm)
    return subprocess.call(
        _server_command(contract, args.host, args.port, args.urdf),
        cwd=REPO_ROOT,
        env=environment,
    )
