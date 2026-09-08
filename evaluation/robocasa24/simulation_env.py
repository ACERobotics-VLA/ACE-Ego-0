import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
import tyro
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from evaluation.robocasa24.model2robocasa_interface import PolicyWarper
from evaluation.robocasa24.wrappers.multistep_wrapper import MultiStepWrapper
from evaluation.robocasa24.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2  # 和10 是什么关系
    fps: int = 10  # BUG 数据集中应该是 20?
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    stabilization_steps: int = 2
    seed: int = 7


def _build_reset_seed(seed: int, n_envs: int) -> int | list[int]:
    """Build the initial seed payload for the environment reset."""
    if n_envs == 1:
        return seed
    return [seed + offset for offset in range(n_envs)]


class SimulationInferenceEnv:
    """Client for running simulations with a model."""

    def __init__(self, model: Optional[PolicyWarper] = None):
        """Initialize the simulation client with a model."""
        self.model = model
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the model based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop("video.ego_view_bg_crop_pad_res256_freq20")
        return self.model.step(observations)

    def _build_hold_action_from_observation(
        self, observations: Dict[str, Any], n_action_steps: int
    ) -> Dict[str, np.ndarray]:
        """Build an action chunk that keeps the current robot state fixed."""
        state_to_action_keys = {
            "state.left_arm": "action.left_arm",
            "state.right_arm": "action.right_arm",
            "state.left_hand": "action.left_hand",
            "state.right_hand": "action.right_hand",
            "state.waist": "action.waist",
        }

        hold_action = {}
        for state_key, action_key in state_to_action_keys.items():
            if state_key not in observations:
                raise KeyError(f"Missing required observation key for hold action: {state_key}")

            current_value = observations[state_key][:, -1:, :]
            hold_action[action_key] = np.repeat(current_value, n_action_steps, axis=1)

        return hold_action

    def _stabilize_environment(self, observations: Dict[str, Any], config: SimulationConfig) -> Dict[str, Any]:
        """Advance the env with hold actions so physics and observations can settle."""
        if config.stabilization_steps <= 0:
            return observations

        print(f"Performing {config.stabilization_steps} stabilization steps...")
        hold_action = self._build_hold_action_from_observation(
            observations=observations,
            n_action_steps=config.multistep.n_action_steps,
        )
        current_obs = observations
        for _ in range(config.stabilization_steps):
            current_obs, _, _, _, _ = self.env.step(hold_action)
            self._clear_rollout_tracking_after_stabilization()
        return current_obs

    def _clear_rollout_tracking_after_stabilization(self) -> None:
        """Drop pre-rollout wrapper counters so stabilization steps do not consume the episode budget."""
        if self.env is None:
            return

        cleared_with_vector_call = False
        if hasattr(self.env, "call"):
            try:
                self.env.call("clear_rollout_tracking")
                cleared_with_vector_call = True
            except (AttributeError, NotImplementedError, TypeError):
                pass
        if not cleared_with_vector_call:
            for base_env in getattr(self.env, "envs", []):
                current_env = base_env
                for _ in range(10):
                    if hasattr(current_env, "clear_rollout_tracking"):
                        current_env.clear_rollout_tracking()
                        break
                    if hasattr(current_env, "env"):
                        current_env = current_env.env
                        continue
                    break

        # SyncVectorEnv records a truncation from the stabilization step and would
        # otherwise auto-reset on the first policy step.  Stabilization is explicitly
        # outside the rollout budget, so clear only the vector autoreset latch while
        # retaining the already-settled observation.
        autoreset_envs = getattr(self.env, "_autoreset_envs", None)
        if autoreset_envs is not None:
            if isinstance(autoreset_envs, np.ndarray):
                autoreset_envs[...] = False
            else:
                self.env._autoreset_envs = False

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def _update_sim_on_model(self, verbose: bool = True) -> bool:
        """Refresh the model-side sim handle after env reset and wrapper changes."""
        if self.model is None:
            return False

        def get_sim_from_env():
            base_env = self.env.envs[0]
            current_env = base_env
            for _ in range(10):
                if hasattr(current_env, "unwrapped"):
                    unwrapped = current_env.unwrapped
                    if hasattr(unwrapped, "env") and hasattr(unwrapped.env, "sim"):
                        sim = unwrapped.env.sim
                        if sim is not None and hasattr(sim, "data") and hasattr(sim, "model"):
                            return sim
                    if hasattr(unwrapped, "sim"):
                        sim = unwrapped.sim
                        if sim is not None and hasattr(sim, "data") and hasattr(sim, "model"):
                            return sim
                if hasattr(current_env, "sim"):
                    sim = current_env.sim
                    if sim is not None and hasattr(sim, "data") and hasattr(sim, "model"):
                        return sim
                if hasattr(current_env, "env"):
                    current_env = current_env.env
                    continue
                break
            return None

        try:
            if hasattr(self.model, "set_sim_getter"):
                self.model.set_sim_getter(get_sim_from_env)
                sim = self.model.sim
                if sim is not None:
                    if verbose:
                        print("*** Set sim getter on model, current sim valid ***")
                    return True
                if verbose:
                    print("WARNING: sim getter set but returned None")
                return False

            sim = get_sim_from_env()
            if sim is not None:
                self.model.sim = sim
                if verbose:
                    print("*** Set sim object directly on model ***")
                return True
            if verbose:
                print("WARNING: Could not find valid sim object")
            return False
        except Exception as error:
            if verbose:
                print(f"WARNING: Could not update sim on model: {error}")
            return False

    def run_simulation(self, config: SimulationConfig, model: Optional[PolicyWarper] = None) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes.

        Args:
            config: Configuration for the simulation
            model: The model to use for inference. If None, uses the model from __init__
        """
        # Use the provided model or fall back to the instance model
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided. Please provide a model either in __init__ or run_simulation")

        start_time = time.time()
        print(f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments")
        print(f"Seed: {config.seed}")
        print(
            f"Max episode steps: {config.multistep.max_episode_steps}, Action steps per inference: {config.multistep.n_action_steps}"
        )
        print("-" * 60)
        # Set up the environment
        self.env = self.setup_environment(config)

        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        ik_failure_count = 0
        # Initial environment reset
        obs, _ = self.env.reset(seed=_build_reset_seed(config.seed, config.n_envs))
        obs = self._stabilize_environment(obs, config)
        self._update_sim_on_model(verbose=True)
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the model
            actions = self._get_actions_from_model(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    # Print real-time episode result
                    status = "SUCCESS" if current_successes[env_idx] else "FAILED"
                    termination_reason = "terminated" if terminations[env_idx] else "truncated (max steps)"
                    print(
                        f"Episode {completed_episodes + 1:3d}/{config.n_episodes}: {status:7s} | "
                        f"Steps: {current_lengths[env_idx]:4d} | {termination_reason}"
                    )
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
                    if config.n_envs == 1 and completed_episodes < config.n_episodes:
                        obs, _ = self.env.reset()
                        obs = self._stabilize_environment(obs, config)
                        self._update_sim_on_model(verbose=True)
                        next_obs = obs
            obs = next_obs
        # Close without an extra reset.  Resetting here starts a fresh recorder and
        # leaves an unlabelled UUID video behind after the completed episode.
        self.env.close()
        self.env = None
        if self.model is not None and hasattr(self.model, "close"):
            self.model.close()
        print("-" * 60)
        print(f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds")
        print(f"IK failures: {ik_failure_count}")
        assert len(episode_successes) >= config.n_episodes, (
            f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        )
        return config.env_name, episode_successes

    def _get_actions_from_model(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the model."""
        # Get actions from the model
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Default joint position controller. Mink IK also emits joint targets into this path.
    env = gym.make(config.env_name, enable_render=True, seed=config.seed + idx)

    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
        # Set task name for better video file naming
        env.set_task_name(config.env_name)
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    model: PolicyWarper,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
    stabilization_steps: int = 2,
    seed: int = 7,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        model: The model to use for inference
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
        stabilization_steps: Number of hold-action steps after each reset.
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(n_action_steps=n_action_steps, max_episode_steps=max_episode_steps),
        stabilization_steps=stabilization_steps,
        seed=seed,
    )
    # Create client and run simulation
    client = SimulationInferenceEnv(model=model)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size: Optional[List[int]] = None  # None means follow the checkpoint image-size policy.

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    env_name: str = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    n_episodes: int = 50  # Number of steps to wait for objects to stabilize i n sim
    n_envs: int = 1  # Number of rollouts per task
    max_episode_steps: int = 360  #
    n_action_steps: int = 3
    n_exec_steps: Optional[int] = None
    stabilization_steps: int = 2  # Number of hold-action steps after each env reset.

    ik_urdf_path: Optional[str] = None

    #################################################################################################################
    # IK solver parameters
    #################################################################################################################
    ik_learning_rate: float = 0.05
    ik_max_iterations: int = 2000
    ik_pose_tolerance: float = 0.005
    ik_unreachable_threshold: float = 1.5

    #################################################################################################################
    # Delta-action parameters
    #################################################################################################################
    use_delta_action: bool = False  # Reconstruct absolute EEF actions from delta predictions
    delta_base_state: str = "first_state"  # Delta reference state strategy
    delta_rot_norm_mode: str = "q99"
    delta_stats_path: Optional[str] = None
    delta_action_mode: Optional[str] = None  # "state_anchor" or "chunk_anchor"
    delta_quat_mode: Optional[str] = None  # "so3" or "subtract"

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "outputs/videos"  # Path to save videos.

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = "checkpoints/robocasa24/ace-ego-0-absolute/checkpoints/ace_ego_0_robocasa24_absolute.pt"


def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    exec_steps = args.n_exec_steps if args.n_exec_steps is not None else args.n_action_steps

    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path,  # to get unnormalization stats
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=exec_steps,
        model_action_horizon=args.n_action_steps,
        ik_urdf_path=args.ik_urdf_path,
        ik_learning_rate=args.ik_learning_rate,
        ik_max_iterations=args.ik_max_iterations,
        ik_pose_tolerance=args.ik_pose_tolerance,
        ik_unreachable_threshold=args.ik_unreachable_threshold,
        use_delta_action=args.use_delta_action,
        delta_base_state=args.delta_base_state,
        delta_rot_norm_mode=args.delta_rot_norm_mode,
        delta_stats_path=args.delta_stats_path,
        delta_action_mode=args.delta_action_mode,
        delta_quat_mode=args.delta_quat_mode,
    )
    run_evaluation(
        env_name=args.env_name,
        model=model,
        video_dir=args.video_out_path,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=exec_steps,
        max_episode_steps=args.max_episode_steps,
        stabilization_steps=args.stabilization_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)
