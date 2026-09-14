<p align="center">
  <img src="assets/ACE_Logo.png" alt="ACE Robotics" height="72">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/cuhk_logo.png" alt="The Chinese University of Hong Kong" height="72">
</p>

<h1 align="center">ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining</h1>

<p align="center">
  <a href="https://acerobotics2025.github.io/ACE-Ego-0/"><strong>Project Page</strong></a> ·
  <a href="https://arxiv.org/pdf/2606.17200"><strong>Paper</strong></a> ·
  <a href="#-overview"><strong>Overview</strong></a> ·
  <a href="#-code"><strong>Code</strong></a> ·
  <a href="#-models"><strong>Models</strong></a> ·
  <a href="#-results"><strong>Results</strong></a> ·
  <a href="#sft-training"><strong>Training</strong></a> ·
  <a href="#-citation"><strong>Citation</strong></a>
</p>

## 🔥 News

- **2026-09**: RoboCasa 24 camera-space EEF SFT data is released on Hugging Face.
- **2026-09**: Lightweight RoboCasa 24 and ARX SFT code is released with Common23 pretrain initialization contracts.
- **2026-09**: GR1 RoboCasa 24 inference and evaluation code is released.
- **2026-06**: Project materials for *ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining* are available on the project page.

## 📖 Overview

<p align="center">
  <img src="assets/figures/teaser.png" alt="ACE-Ego-0 teaser" width="95%">
</p>

**ACE-Ego-0** is a unified vision-language-action (VLA) pretraining framework that combines egocentric human videos, multi-embodiment robot demonstrations, and simulation rollouts for robot policy learning.

## ✨ Highlights

- **Human + robot pretraining**: Uses 4.53K hours of robot/simulation data and 1.48K hours of pseudo-action-labeled egocentric human data.
- **Camera-space action alignment**: Represents human pseudo-actions and robot end-effector trajectories in the observation-centric camera frame.
- **Morphology conditioning**: Conditions the action expert with robot URDF graph embeddings and learned human surrogate tokens.
- **Strong transfer**: Achieves 72.8% average success on RoboCasa GR1 TableTop, 91.12% / 90.62% on RoboTwin 2.0 Easy / Hard, and 78.3% average success on real bimanual ARX tasks.

## 🧠 Method

<p align="center">
  <img src="assets/figures/method.png" alt="ACE-Ego-0 method overview" width="95%">
</p>

ACE-Ego-0 resolves four core mismatches between egocentric human video and robot trajectories:

1. **Spatial mismatch**: Human and robot motions are normalized through camera-space action representations.
2. **Embodiment mismatch**: Robot morphology and human surrogate embodiment information condition the action model.
3. **Temporal mismatch**: Action chunking aligns heterogeneous video and trajectory horizons.
4. **Label-quality mismatch**: Reliable robot actions supervise the primary objective, while noisier human pseudo-actions contribute through auxiliary losses.

## 📊 Results

| Benchmark | Metric | ACE-Ego-0 |
| --- | ---: | ---: |
| RoboCasa GR1 TableTop | Average success | **72.8%** |
| RoboTwin 2.0 Easy | Average success | **91.12%** |
| RoboTwin 2.0 Hard | Average success | **90.62%** |
| Real bimanual ARX tasks | Average success | **78.3%** |

## 💻 Code

This repository contains the ACE-Ego-0 codebase. The current public surface is
GR1 RoboCasa 24 inference/evaluation plus Common23 SFT for RoboCasa 24 and ARX.

The release does **not** include Common23 449-mixture pretraining,
normalization-statistics computation, RoboCasa365, optimizer state, internal
cluster launchers, RoboTwin evaluation, or real-robot ARX deployment.

### Installation

The setup below assumes a Linux host with `conda`, `git`, `tar`, and `curl` or `wget`. The release was tested with Python 3.10, an NVIDIA GPU, and a CUDA 12.4-compatible driver. From a fresh clone:

```bash
git clone https://github.com/ACERobotics-VLA/ACE-Ego-0.git
cd ACE-Ego-0

conda create -n ace-ego-robocasa24 python=3.10 -y
conda activate ace-ego-robocasa24

python -m pip install --upgrade pip setuptools wheel

# CUDA 12.4 PyTorch wheels.
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Runtime dependencies, the repository, and the Hugging Face CLI.
python -m pip install "huggingface_hub>=0.34,<1" -r requirements.txt -e . einops

# Prebuilt Flash Attention wheel for Python 3.10, CUDA 12.4, and PyTorch 2.6.
python -m pip install \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu124torch2.6-cp310-cp310-linux_x86_64.whl
```

For SFT, install the training extra after the base environment. This adds only
DeepSpeed and PyArrow. Weights & Biases remains optional:

```bash
python -m pip install -e ".[train]"
python -m pip install -e ".[wandb]"
```

Install the pinned RoboSuite and GR1 RoboCasa task package, then download the tabletop assets:

```bash
bash scripts/install_robocasa.sh
```

The script stores external sources under `third_party/`, installs them in editable mode, downloads the required RoboCasa assets, and runs the environment checker. The asset download is large but safe to rerun. To install the Python packages first and download assets later:

```bash
SKIP_ROBOCASA_ASSETS=1 bash scripts/install_robocasa.sh
```

Verify the complete setup:

```bash
python -m pip check
python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
python scripts/check_environment.py
```

The two `MUJOCO_GL` variables are for headless rendering; adjust `MUJOCO_EGL_DEVICE_ID` if the desired GPU is not device 0. If the Flash Attention wheel does not match the local driver or PyTorch installation, install the corresponding prebuilt wheel before running evaluation.

### Source layout

```text
src/ace_ego_0/             ACE-Ego-0 policy and model components
src/ace_ego_0/data/        Read-only RoboCasa24/ARX LeRobot data path
src/ace_ego_0/training/    Checkpoint contracts and lightweight SFT trainer
evaluation/robocasa24/     RoboCasa task runner, websocket server, and Mink IK
configs/                   RoboCasa24/ARX SFT and ZeRO-2 recipes
scripts/                   Training, environment setup, smoke tests, and evaluation
assets/                    GR1 URDF/meshes, URDF caches, and Qwen lightweight files
checkpoints/               Pretrain contracts and RoboCasa24 policy bundles
```

## 🤗 Models

The released Absolute and Delta GR1 RoboCasa 24 checkpoint bundles are available for download from [Hugging Face](https://huggingface.co/acerobotics2025/ACE-Ego-0).

RoboCasa 24 SFT data is hosted at
[ACERobotics/Robocasa_Camera_Space_Eef](https://huggingface.co/datasets/ACERobotics/Robocasa_Camera_Space_Eef).
ARX SFT data remains separately distributed.

### Common23 pretrain initializers

RoboCasa 24 and ARX SFT start from matching Common23 checkpoints:

```text
checkpoints/pretrain/
├── ace-ego-0-pretrain-absolute/
│   ├── config.yaml
│   ├── SHA256SUMS
│   └── checkpoints/ace_ego_0_pretrain_absolute.pt
└── ace-ego-0-pretrain-delta/
    ├── config.yaml
    ├── SHA256SUMS
    └── checkpoints/ace_ego_0_pretrain_delta.pt
```

Download them from the same Hugging Face model repository:

```bash
hf download "acerobotics2025/ACE-Ego-0" \
  --repo-type model \
  --include "checkpoints/pretrain/**" \
  --local-dir .
```

Expected SHA256 values are `02ee20977add7c41ce3a8f3b1164ef2928d88e02884ebd29d550d0c234448d3c`
for Absolute and `f1ae71ce55f4bf609f70af83797081789f41e90f47dc041bae10bae0c3bd06b0`
for Delta. These are initialization models, not target-task policies. Do not use their pretraining
statistics for RoboCasa or ARX inference; SFT loads the stats distributed with each target dataset.

## 🤖 GR1 URDF and Offline Cache

The repository includes `assets/GR1T2_with_hands.urdf` and the referenced mesh files for the released GR1 embodiment. You may use this URDF directly or provide a compatible local URDF with `--urdf`.

`assets/urdf_cache/GR1.pkl` is the canonical 34-node morphology cache used during training
(`SHA256 dea552de1d4449e8c5db5cf7215b80a171c155b5c430dda00e44402e361acc1d`).
Do not rebuild it from the 58-joint with-hands URDF: that file is an IK/render asset and produces a
different model input. Mink IK still reads the source URDF at runtime, so the `--urdf` path must exist.

ARX SFT uses the cache-only `assets/urdf_cache/ARX5Dual.pkl`
(`SHA256 7ebd6dad97105fa17a049dbec75912f3c32d24193ccd84562a78620e6dbd3fa6`).

## 🎯 RoboCasa 24 Evaluation

Run one episode of task index 2 with the Absolute policy:

```bash
python scripts/smoke_test.py \
  --checkpoint checkpoints/robocasa24/ace-ego-0-absolute \
  --urdf assets/GR1T2_with_hands.urdf \
  --task-index 2
```

Run one task for 50 episodes with the Delta policy:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/robocasa24/ace-ego-0-delta \
  --urdf assets/GR1T2_with_hands.urdf \
  --task-index 13 \
  --episodes 50
```

Run all 24 tasks for one episode each:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/robocasa24/ace-ego-0-absolute \
  --urdf assets/GR1T2_with_hands.urdf \
  --task-start 0 \
  --task-end 23 \
  --episodes 1
```

Run the complete 24-task benchmark with 50 episodes per task:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/robocasa24/ace-ego-0-delta \
  --urdf assets/GR1T2_with_hands.urdf \
  --task-start 0 \
  --task-end 23 \
  --episodes 50
```

The evaluator reads Absolute/Delta action semantics from the corresponding config. For Delta Action, it automatically uses the recorded reconstruction settings.

Each run writes per-task logs, videos, and `summary.json` under `outputs/`. These are local evaluation artifacts and are ignored by Git.

### Explicit local server mode

Start the policy server in one terminal:

```bash
python scripts/run_server.py \
  --checkpoint checkpoints/robocasa24/ace-ego-0-delta \
  --urdf /absolute/path/to/GR1T2_with_hands.urdf \
  --host 127.0.0.1 \
  --port 5678
```

Connect the evaluator from a second terminal:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/robocasa24/ace-ego-0-delta \
  --urdf /absolute/path/to/GR1T2_with_hands.urdf \
  --connect-only \
  --host 127.0.0.1 \
  --port 5678 \
  --task-start 0 \
  --task-end 23 \
  --episodes 1
```

Use `--dry-run` to inspect generated server and simulator commands without starting CUDA inference or MuJoCo.

## SFT training

The public trainer is Common23-only. Install the training extra first
(`python -m pip install -e ".[train]"`), then download the matching Absolute or
Delta [Common23 pretrain initializer](#common23-pretrain-initializers).

### RoboCasa 24

RoboCasa SFT covers the 24 GR1 TableTop tasks, not RoboCasa365. The two recipes
are `configs/robocasa24/absolute_sft.yaml` and `configs/robocasa24/delta_sft.yaml`.

Download the public dataset into the default recipe root. Keep the 24
`gr1_unified.*_1000` directory names and the shipped `norm/` artifacts unchanged:

```bash
hf download ACERobotics/Robocasa_Camera_Space_Eef \
  --repo-type dataset \
  --local-dir data/robocasa24
```

The Absolute/Delta recipes read:

```text
data/robocasa24/
├── gr1_unified.*_1000/
│   ├── data/
│   ├── videos/
│   └── meta/
│       ├── info.json
│       ├── modality.json
│       └── episodes.jsonl
└── norm/{absolute,delta}/
    ├── norm_manifest.json
    └── merged_stats.json
```

Each task directory only needs those three `meta/` files. Extra leftovers such as
`meta/stats_gr00t*.json` are ignored. Set `ACE_EGO_0_DATA_ROOT` if the dataset
lives outside `data/robocasa24`.

The RoboCasa training contract is:

- one `ego_view` camera, resized to 224×224 in the loader;
- language from `episodes.jsonl` `remarks`, with `locked_waist:` /
  `unlocked_waist:` stripped;
- Common23 23D state/action and horizon 30;
- Fourier hands converted with the first four fingers `(x + 1.5) / 3`, then
  mapped to `[-1, 1]`;
- Absolute recipes initialize from the Absolute pretrain bundle;
- Delta recipes initialize from the Delta bundle and apply
  `state_anchor + first_state + subtract` to xyz / rot6d / waist **before** q99.
  `first_state` is the current chunk observation, not episode frame 0. Gripper
  stays absolute;
- q99 statistics are loaded only from `norm/{absolute,delta}/`, key `"gr1"`,
  raw dim 33. The trainer never scans parquet or recomputes norms;
- morphology uses the shipped 34-node `assets/urdf_cache/GR1.pkl`. Do not rebuild
  it from `assets/GR1T2_with_hands.urdf`;
- Absolute and Delta initializers must not be crossed.

Default recipe schedule: 200k steps, per-device batch size 16, Qwen LR `2e-5`,
action-expert LR `1e-4`. Production launch uses the 8-process ZeRO-2 profile
with `mixed_precision=no`.

Validate a recipe without allocating the model:

```bash
python scripts/train.py \
  --config configs/robocasa24/absolute_sft.yaml \
  --dry-run
```

Launch Absolute or Delta SFT:

```bash
accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --mixed_precision no \
  scripts/train.py \
  --config configs/robocasa24/absolute_sft.yaml

accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --mixed_precision no \
  scripts/train.py \
  --config configs/robocasa24/delta_sft.yaml
```

Paths can be overridden without editing YAML:

```bash
python scripts/train.py \
  --config configs/robocasa24/absolute_sft.yaml \
  --data-root /path/to/robocasa24 \
  --norm-manifest /path/to/robocasa24/norm/absolute/norm_manifest.json \
  --pretrained-checkpoint /path/to/ace_ego_0_pretrain_absolute.pt \
  --output-dir /path/to/runs
```

`--max-steps` is available for short smokes; it also lowers `save_interval` so a
checkpoint is written at the last step.

Training is full-parameter SFT: VLM BF16, action expert FP32, four diffusion
repeats, AdamW parameter groups, gradient clipping, and cosine-with-min-LR.

Each run writes:

```text
<run_root>/<run_id>/
├── training_config.yaml
├── config.yaml
└── checkpoints/
    ├── steps_<N>_pytorch_model.pt
    ├── runtime_merged_dataset_statistics.json
    └── training_metrics.jsonl
```

`training_config.yaml` is the full recipe. `config.yaml` is the stripped
inference config used by evaluation: the trainer writes back `include_state`,
`rot/trans/gripper_norm_mode=q99`, and `language_source=episode_remarks`. The
copied statistics file is the shipped RoboCasa merged stats, not a newly computed
table. The trainer does not save optimizer state or implement resume.

To evaluate a trained run, pass the `.pt` file. The matching `config.yaml` and
`checkpoints/runtime_merged_dataset_statistics.json` must stay beside it:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/training/ace_ego_0_robocasa24_absolute_sft/checkpoints/steps_200000_pytorch_model.pt \
  --urdf assets/GR1T2_with_hands.urdf \
  --task-index 2 \
  --episodes 1
```

Released RoboCasa eval bundles already follow this contract; do not mix their
`config.yaml` or statistics with a different Absolute/Delta run.

### ARX

ARX: `cam_high`, `cam_left_wrist`, and `cam_right_wrist`, 256×256, raw20 mapped to
Common23 with the three waist action dimensions masked.

Launch the current formal ARX task:

```bash
accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --mixed_precision no \
  scripts/train.py \
  --config configs/arx/stack_ceramic_bowls_delta_sft.yaml
```

Paths can be supplied without editing YAML:

```bash
python scripts/train.py \
  --config configs/arx/stack_ceramic_bowls_absolute_sft.yaml \
  --data-root /path/to/arx \
  --norm-manifest /path/to/arx/norm/stack_ceramic_bowls/absolute/norm_manifest.json \
  --pretrained-checkpoint /path/to/ace_ego_0_pretrain_absolute.pt \
  --output-dir /path/to/runs
```

The current formal ARX target is `stack_ceramic_bowls`. Other ARX recipes remain in
`configs/arx/` but are not the validated training target.

See [docs/README.md](docs/README.md) for the document map.
[docs/sft_training_implementation.md](docs/sft_training_implementation.md) records the
implementation and contracts.
[docs/sft_training_status.md](docs/sft_training_status.md) records the data layout and
acceptance runs. The original scoped plan remains in
[docs/open_source_sft_training.md](docs/open_source_sft_training.md).

## 📝 Citation

If you find ACE-Ego-0 useful, please cite:

```bibtex
@misc{li2026aceego0unifyingegocentrichuman,
  title         = {ACE-Ego-0: Unifying Egocentric Human and Robotic Data for VLA Pretraining},
  author        = {Hao Li and Ganlong Zhao and Yufei Liu and Haotian Hou and Guoquan Ye and Tongyan Fang and Chunxiao Liu and Siyuan Huang and Jianbo Liu and Xiaogang Wang and Hongsheng Li},
  year          = {2026},
  eprint        = {2606.17200},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2606.17200}
}
```

## 📬 Contact

For questions about the ACE-Ego-0 release, please open an issue in this repository.

## License

The ACE-Ego-0 code is MIT licensed. RoboCasa, RoboSuite, Qwen3-VL, datasets,
robot assets, and other dependencies retain their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
