"""Minimal Accelerate/DeepSpeed trainer for public ACE-Ego-0 SFT recipes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import get_scheduler

from ace_ego_0.data import build_sft_dataloader
from ace_ego_0.model.framework import build_framework
from ace_ego_0.training.checkpointing import load_sft_initialization


def validate_sft_config(config: Any) -> str:
    """Validate the intentionally narrow public training contract."""
    action_config = config.framework.action_model
    data_config = config.datasets.vla_data
    expected = {
        "action_dim": 23,
        "state_dim": 23,
        "action_horizon": 30,
        "future_action_window_size": 29,
        "past_action_window_size": 0,
    }
    for field, expected_value in expected.items():
        actual = int(action_config.get(field))
        if actual != expected_value:
            raise ValueError(f"Public Common23 SFT requires {field}={expected_value}, got {actual}.")
    if str(action_config.get("state_injection_mode")) != "timestep_conditioning":
        raise ValueError("Public Common23 SFT requires state_injection_mode=timestep_conditioning.")
    if not bool(action_config.get("enable_urdf_input", False)):
        raise ValueError("Public Common23 SFT requires URDF morphology conditioning.")
    if bool(action_config.get("urdf_auto_build_cache", True)):
        raise ValueError("Public SFT is cache-only; set urdf_auto_build_cache=false.")
    if data_config.get("merged_norm_path") or data_config.get("runtime_merged_statistics_path"):
        raise ValueError("Use norm_manifest_path only; runtime stats merge is not supported.")
    if bool(data_config.get("dynamic_action_chunking", False)):
        raise ValueError("Dynamic action chunking is not part of the public SFT release.")

    action_mode = "delta" if bool(data_config.get("delta_action", False)) else "absolute"
    if action_mode == "delta":
        delta_contract = {
            "delta_xyz": True,
            "delta_rot": True,
            "delta_action_mode": "state_anchor",
            "delta_base_state": "first_state",
            "delta_quat_mode": "subtract",
        }
        for field, expected_value in delta_contract.items():
            if data_config.get(field) != expected_value:
                raise ValueError(f"Public delta SFT requires datasets.vla_data.{field}={expected_value!r}.")
    return action_mode


def build_param_groups(model: torch.nn.Module, config: Any) -> list[dict[str, Any]]:
    """Reproduce the production base/Qwen/action parameter grouping."""
    learning_rates = config.trainer.learning_rate
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    for module_name in ("qwen_vl_interface", "action_model"):
        module = getattr(model, module_name)
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"Training module {module_name!r} has no trainable parameters.")
        assigned.update(id(parameter) for parameter in parameters)
        groups.append(
            {
                "params": parameters,
                "lr": float(learning_rates[module_name]),
                "name": module_name,
            }
        )
    base_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in assigned
    ]
    if base_parameters:
        groups.append(
            {
                "params": base_parameters,
                "lr": float(learning_rates.base),
                "name": "base",
            }
        )
    return groups


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Any,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LRScheduler]:
    optimizer_config = config.trainer.optimizer
    if bool(optimizer_config.get("foreach", False)):
        raise ValueError("Public ACE-Ego-0 SFT requires trainer.optimizer.foreach=false.")
    optimizer = torch.optim.AdamW(
        build_param_groups(model, config),
        lr=float(config.trainer.learning_rate.base),
        betas=tuple(float(value) for value in optimizer_config.betas),
        eps=float(optimizer_config.eps),
        weight_decay=float(optimizer_config.weight_decay),
        foreach=False,
    )
    scheduler_kwargs = {
        key: float(value) if hasattr(value, "__float__") else value
        for key, value in dict(config.trainer.scheduler_specific_kwargs).items()
    }
    scheduler = get_scheduler(
        name=str(config.trainer.lr_scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=int(config.trainer.num_warmup_steps),
        num_training_steps=int(config.trainer.max_train_steps),
        scheduler_specific_kwargs=scheduler_kwargs,
    )
    return optimizer, scheduler


def _resolve_norm_stats_path(config: Any, *, config_path: Path) -> Path:
    raw_manifest_path = Path(str(config.datasets.vla_data.norm_manifest_path)).expanduser()
    manifest_path = raw_manifest_path if raw_manifest_path.is_absolute() else config_path.parent / raw_manifest_path
    manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    merged_path = Path(str(manifest["artifacts"]["merged_stats_path"])).expanduser()
    return merged_path if merged_path.is_absolute() else (manifest_path.parent / merged_path).resolve()


def _inference_config(config: Any) -> Any:
    exported = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    exported.pop("trainer", None)
    exported.pop("trackers", None)
    exported.pop("run_root_dir", None)
    data_config = exported.datasets.vla_data
    for key in (
        "data_root_dir",
        "dataset_manifest_path",
        "manifest_robot_type",
        "norm_manifest_path",
        "per_device_batch_size",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
    ):
        data_config.pop(key, None)
    robot_type = str(config.datasets.vla_data.manifest_robot_type)
    data_config.robot_name = "GR1" if robot_type == "fourier_gr1_eef_gripper_6d" else "ARX5Dual"
    data_config.include_state = True
    data_config.rot_norm_mode = "q99"
    data_config.trans_norm_mode = "q99"
    data_config.gripper_norm_mode = "q99"
    if robot_type == "fourier_gr1_eef_gripper_6d":
        data_config.language_source = "episode_remarks"
    return exported


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: Any,
    config_path: Path,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"steps_{step}_pytorch_model.pt"
    state_dict = accelerator.get_state_dict(model)
    accelerator.save(state_dict, checkpoint_path)
    if accelerator.is_main_process:
        OmegaConf.save(config=config, f=output_dir / "training_config.yaml")
        OmegaConf.save(config=_inference_config(config), f=output_dir / "config.yaml")
        stats_path = _resolve_norm_stats_path(config, config_path=config_path)
        shutil.copyfile(stats_path, checkpoint_dir / "runtime_merged_dataset_statistics.json")
    accelerator.wait_for_everyone()
    return checkpoint_path


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _initialize_wandb(config: Any, accelerator: Accelerator):
    trackers = {str(name).lower() for name in config.get("trackers", [])}
    if "wandb" not in trackers or not accelerator.is_main_process:
        return None
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B logging requires `pip install -e '.[wandb]'`.") from error
    return wandb.init(
        project=str(config.get("wandb_project", "ace-ego-0-sft")),
        name=str(config.run_id),
        config=OmegaConf.to_container(config, resolve=True),
    )


def train_sft(config: Any, *, config_path: str | Path) -> Path:
    """Run one full-parameter public SFT recipe and return the final flat checkpoint."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    action_mode = validate_sft_config(config)
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=int(config.trainer.get("gradient_accumulation_steps", 1)),
        step_scheduler_with_optimizer=False,
    )
    if accelerator.mixed_precision != "no":
        raise RuntimeError("ACE-Ego-0 requires Accelerate mixed_precision=no.")
    set_seed(int(config.get("seed", 42)), device_specific=True)

    model = build_framework(cfg=config)
    pretrained_checkpoint = Path(str(config.trainer.pretrained_checkpoint)).expanduser()
    if not pretrained_checkpoint.is_absolute():
        pretrained_checkpoint = (resolved_config_path.parent / pretrained_checkpoint).resolve()
    load_sft_initialization(
        model,
        pretrained_checkpoint,
        expected_action_mode=action_mode,
    )
    model.set_gradient_checkpointing(bool(config.trainer.get("enable_gradient_checkpointing", True)))
    dataloader = build_sft_dataloader(config, config_path=resolved_config_path)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    output_root = Path(str(config.run_root_dir)).expanduser()
    if not output_root.is_absolute():
        output_root = (resolved_config_path.parent / output_root).resolve()
    output_dir = output_root / str(config.run_id)
    metrics_path = output_dir / "checkpoints" / "training_metrics.jsonl"
    wandb_run = _initialize_wandb(config, accelerator)
    max_steps = int(config.trainer.max_train_steps)
    save_interval = int(config.trainer.save_interval)
    logging_frequency = int(config.trainer.get("logging_frequency", 20))
    if max_steps <= 0 or save_interval <= 0 or logging_frequency <= 0:
        raise ValueError("max_train_steps, save_interval, and logging_frequency must be positive.")

    completed_steps = 0
    last_checkpoint: Path | None = None
    while completed_steps < max_steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                output = model(examples=batch)
                loss = output["total_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {completed_steps + 1}: {loss}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(config.trainer.gradient_clipping),
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            completed_steps += 1
            if completed_steps == 1 or completed_steps % logging_frequency == 0:
                reduced_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
                if accelerator.is_main_process:
                    metrics = {
                        "step": completed_steps,
                        "loss": float(reduced_loss.item()),
                        "learning_rates": {
                            str(group.get("name", index)): float(group["lr"])
                            for index, group in enumerate(optimizer.param_groups)
                        },
                    }
                    _write_jsonl(metrics_path, metrics)
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=completed_steps)
            if completed_steps % save_interval == 0 or completed_steps == max_steps:
                last_checkpoint = _save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    config=config,
                    config_path=resolved_config_path,
                    output_dir=output_dir,
                    step=completed_steps,
                )
            if completed_steps >= max_steps:
                break

    if last_checkpoint is None:
        raise RuntimeError("Training completed without writing a checkpoint.")
    if wandb_run is not None:
        wandb_run.finish()
    accelerator.end_training()
    return last_checkpoint
