#!/usr/bin/env python3
"""Train ACE-Ego-0 on one public RoboCasa24 or ARX SFT recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from ace_ego_0.training import load_sft_config, train_sft, validate_sft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Public SFT YAML recipe.")
    parser.add_argument("--data-root", type=Path, help="Override datasets.vla_data.data_root_dir.")
    parser.add_argument("--norm-manifest", type=Path, help="Override datasets.vla_data.norm_manifest_path.")
    parser.add_argument("--pretrained-checkpoint", type=Path, help="Override trainer.pretrained_checkpoint.")
    parser.add_argument("--output-dir", type=Path, help="Override run_root_dir.")
    parser.add_argument("--run-id", type=str, help="Override run_id.")
    parser.add_argument("--max-steps", type=int, help="Override trainer.max_train_steps (for smoke tests).")
    parser.add_argument("--per-device-batch-size", type=int, help="Override per-device batch size.")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader workers.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the resolved config without training."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_sft_config(config_path)
    if args.data_root is not None:
        config.datasets.vla_data.data_root_dir = str(args.data_root.expanduser().resolve())
    if args.norm_manifest is not None:
        config.datasets.vla_data.norm_manifest_path = str(args.norm_manifest.expanduser().resolve())
    if args.pretrained_checkpoint is not None:
        config.trainer.pretrained_checkpoint = str(args.pretrained_checkpoint.expanduser().resolve())
    if args.output_dir is not None:
        config.run_root_dir = str(args.output_dir.expanduser().resolve())
    if args.run_id is not None:
        config.run_id = str(args.run_id)
    if args.max_steps is not None:
        config.trainer.max_train_steps = args.max_steps
        config.trainer.save_interval = min(int(config.trainer.save_interval), args.max_steps)
    if args.per_device_batch_size is not None:
        config.datasets.vla_data.per_device_batch_size = args.per_device_batch_size
    if args.num_workers is not None:
        config.datasets.vla_data.num_workers = args.num_workers
    OmegaConf.resolve(config)
    action_mode = validate_sft_config(config)
    if args.dry_run:
        print(f"Validated ACE-Ego-0 {action_mode} SFT recipe: {config_path}")
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    final_checkpoint = train_sft(config, config_path=config_path)
    print(f"Training complete: {final_checkpoint}")


if __name__ == "__main__":
    main()
