# ACE-Ego-0 SFT 实现与验收记录

本文记录公开训练代码落地后的数据布局、本地辅助脚本、集群验收和当前正式范围。
设计合同见 [`sft_training_implementation.md`](sft_training_implementation.md)，
原始计划见 [`open_source_sft_training.md`](open_source_sft_training.md)。

## 1. 当前结论

公开仓库已经具备可复现的轻量 SFT：

- RoboCasa 24 Absolute / Delta。
- ARX 正式验收任务只有 `stack_ceramic_bowls` Absolute / Delta。
- 训练代码只读 shipped norm，不计算、不 merge、不写 stats cache。
- 2 卡 2-step 和 8 卡 100-step 都已在 A800 训练池跑通，loss 有限，checkpoint 落盘。

`heat_sandwich`、`pour_coffee_beans`、`scan_pencil_case_new` 的 recipe 仍留在
`configs/arx/`，但当前正式训练和集群验收不再覆盖这三个任务。

公开 Git 不包含集群 launcher、W&B key 或本地数据包。`data/` 与 `scripts/local/`
已被 `.gitignore`。

## 2. 公开代码与配置

```text
src/ace_ego_0/data/common23.py      q99、RoboCasa raw33 / ARX raw20 → Common23
src/ace_ego_0/data/lerobot.py       只读 LeRobot v2.x reader
src/ace_ego_0/training/             config 合并、pretrain 初始化、ZeRO-2 trainer
scripts/train.py                    统一 CLI
configs/common_sft.yaml             共享模型 / trainer 合同
configs/robocasa24/                 24-task manifest + abs/delta recipe
configs/arx/                        ARX base + task recipe
configs/accelerate/zero2.yaml       Accelerate ZeRO-2，mixed_precision=no
```

公开 Git 只跟踪 `configs/` 中的 recipe。本机正式训练包装在 gitignored 的
`scripts/local/formal/`，继承公开 yaml 后写入绝对 data / pretrain / output 路径。

`scripts/train.py` 覆盖项：

```text
--config
--data-root
--norm-manifest
--pretrained-checkpoint
--output-dir
--run-id
--max-steps
--per-device-batch-size
--num-workers
--dry-run
```

`--max-steps` 会同时把 `save_interval` 压到不超过该步数，便于 smoke / 100-step
测试写出 `steps_<N>_pytorch_model.pt`。

### 相机顺序

ARX 图像不再按 `modality.json` 的 dict 遍历顺序读取，而是固定：

1. `cam_high`
2. `cam_left_wrist`
3. `cam_right_wrist`

实现为 `ordered_video_keys()` / `ordered_video_entries()`。RoboCasa 固定
`ego_view`。

### PYTHONPATH

`/data/lh/projects/envs/starVLA` 可能默认导入其他仓库里的旧 `ace_ego_0`。
训练启动必须：

```bash
export PYTHONPATH=/data/lh/projects/ACE-Ego-0/src:/data/lh/projects/ACE-Ego-0
```

不要把旧路径放在前面。

## 3. 数据布局

默认 recipe 读取：

```text
data/
├── robocasa24/
│   ├── gr1_unified.*_1000/          24 个完整目录名
│   └── norm/{absolute,delta}/
│       ├── norm_manifest.json
│       └── merged_stats.json
└── arx/
    ├── v0.1/heat_sandwich/
    ├── v0.1/stack_ceramic_bowls/session_*/
    ├── v0.1/pour_coffee_beans/
    ├── v0.1/scan_pencil_case_new/
    └── norm/<task>/{absolute,delta}/
        ├── norm_manifest.json
        └── merged_stats.json
```

每个 dataset 根目录只保留训练会读的文件：

```text
data/
videos/
meta/info.json
meta/modality.json
meta/episodes.jsonl
```

`experiments/`、`meta/metadata.json`、`meta/stats.json`、`meta/steps_data_index.pkl`、
`meta/tasks.jsonl` 和 per-dataset `stats_gr00t*.json` 不进入本地 overlay。真正 q99
归一化只使用 `norm_manifest.json` 指向的 merged stats。

### 本地打包

`scripts/local/package_sft_data.py` 是内部辅助脚本，不进入公开 Git。它：

- 复制内部 0820 RoboCasa / 0912 ARX 的
  `runtime_merged_dataset_statistics.json` 为 `merged_stats.json`。
- 写出 schema v1 `norm_manifest.json`。
- 把已有 LeRobot 根目录 symlink 到 `data/`。
- 不为 RoboCasa 或 bowls 重新扫描 parquet。

`pour_coffee_beans` 和 `scan_pencil_case_new` 的源 `meta/` 只有
`info.json` / `episodes.jsonl` / `tasks.jsonl`。本地 overlay 会：

- symlink `data/` 与 `videos/`；
- 用 heat 同款 `modality.json`。

这只是本地可跑布局。公开数据和训练都不需要 per-dataset `stats_gr00t*.json`。
`stack_ceramic_bowls` merged stats 使用 0912 pub run 的 `parent_task` 口径，
不按 7 个 session 分别再 merge。

当前本地核验规模：

```text
RoboCasa 24 absolute mixture     5,820,277 frames
heat_sandwich                    203,443
pour_coffee_beans                209,045
scan_pencil_case_new             239,762
stack_ceramic_bowls 7 sessions   222,509
```

## 4. 正式 recipe

公开合同 yaml 在 `configs/`。当前正式训练使用的本地包装在
`scripts/local/formal/`：

| 任务 | YAML | SH | 预算 |
|---|---|---|---|
| RoboCasa abs | `robocasa24_absolute_sft.yaml` | `robocasa24_absolute_sft.sh` | 8 卡 / 200k / bs16 |
| RoboCasa delta | `robocasa24_delta_sft.yaml` | `robocasa24_delta_sft.sh` | 8 卡 / 200k / bs16 |
| bowls abs | `stack_ceramic_bowls_absolute_sft.yaml` | `stack_ceramic_bowls_absolute_sft.sh` | 8 卡 / 100k / bs16 |
| bowls delta | `stack_ceramic_bowls_delta_sft.yaml` | `stack_ceramic_bowls_delta_sft.sh` | 8 卡 / 100k / bs16 |

这些 yaml 继承 `configs/` 中的公开 recipe，并把 data / norm / pretrain /
output 写成绝对路径。正式超参与内部 0820 / 0912 对齐：

```text
mixed_precision            no
DeepSpeed                  ZeRO-2
VLM                        BF16
action expert              FP32
AdamW foreach              false
gradient clip              1.0
warmup                     5000
cosine min LR              5e-7
RoboCasa Qwen LR           2e-5
ARX base/Qwen LR           1e-5
action LR                  1e-4
```

公开 Accelerate 入口：

```bash
export PYTHONPATH=/data/lh/projects/ACE-Ego-0/src:/data/lh/projects/ACE-Ego-0

python scripts/train.py \
  --config configs/robocasa24/absolute_sft.yaml \
  --dry-run

accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --mixed_precision no \
  scripts/train.py \
  --config configs/robocasa24/absolute_sft.yaml

accelerate launch \
  --config_file configs/accelerate/zero2.yaml \
  --mixed_precision no \
  scripts/train.py \
  --config configs/arx/stack_ceramic_bowls_delta_sft.yaml
```

## 5. 集群约定

训练 / 模型 debug 走 A800 池 `sensebot-a100-a800-liujianbo`，不要打到 4090
评测池。公开仓库不发布 ACP 脚本；本机辅助文件在 gitignored 的
`scripts/local/`。

| 用途 | 脚本 | 默认资源 |
|---|---|---|
| 2-step smoke | `scripts/local/submit_acp_sft_smoke.sh` | 2 GPU，bs1 |
| 正式 / 100-step 测试 | `scripts/local/formal/*.sh` | 8 GPU，bs16 |

ACP 异构 SKU 最多 4 种，且每个候选的 GPU 数必须等于 `NUM_PROCESSES`。

2 卡 smoke 实际可用集合：

```text
N2lS.Ii.I60a.2,N3lS.Ii.I60.2,n3ls.ii.i60a.2,N3lS.Ii.I60.2.16c208g
```

只写两种 2×A800 + 24c/240GiB 时，即使池里有卡，worker 也可能一直 Pending。
补上 A100 2 卡和缩小 CPU/内存切片后，2-step smoke 才能进入 RUNNING。

8 卡正式集合：

```text
N2lS.Ii.I60a.8,N3lS.Ii.I60.8,n3ls.ii.i60a.8,N3lS.Ii.I60.88c896g
```

默认 `DRY_RUN=1`。确认 startup command 后再 `DRY_RUN=0`。100-step 测试通过：

```bash
MAX_TRAIN_STEPS=100 \
RUN_ID=<unique_run_id> \
JOB_NAME=<unique_job_name> \
DRY_RUN=0 bash scripts/local/formal/robocasa24_absolute_sft.sh
```

## 6. 验收结果

### 单元测试

公开仓库当前不附带 `tests/`。训练不依赖这些单测。

### 真实样本

本地 layout 上，episode 0 / frame 0 可解码：

```text
RoboCasa   1 × 224², action (30, 23), robot=GR1
heat       3 × 256², action (30, 23), robot=ARX5Dual
pour       3 × 256², action (30, 23), robot=ARX5Dual
bowls      3 × 256², action (30, 23), robot=ARX5Dual
```

此前与内部 loader 对齐的数值误差仍成立：

```text
RoboCasa absolute  action max error 2.38e-7, state max error 9.54e-7
RoboCasa delta     action max error 1.49e-7, state max error 9.54e-7
ARX absolute       action max error 2.98e-7, state max error 3.70e-6
ARX delta          action max error 4.77e-7, state max error 3.70e-6
```

### 2 卡 2-step smoke

作业提交到 `sensebot-a100-a800-liujianbo`，全部 SUCCEEDED：

| Job | Recipe | step1 loss | step2 loss |
|---|---|---|---|
| `pt-eoyxgb6z` | RoboCasa abs | 0.49 | 0.64 |
| `pt-k0sbr3qy` | RoboCasa delta | 0.50 | 0.65 |
| `pt-nb5e8jes` | bowls abs | 0.59 | 只记 step1（logging_frequency=20） |
| `pt-buserjup` | bowls delta | 0.87 | 只记 step1 |

四个 run 都写出约 12G 的 `steps_2_pytorch_model.pt`。产物在
`outputs/acp_smoke/`。

### 8 卡 100-step

同样提交到 `sensebot-a100-a800-liujianbo`，全部 SUCCEEDED：

| Job | Recipe | step1 → step100 |
|---|---|---|
| `pt-b31h62f8` | RoboCasa abs | 0.44 → 0.18 |
| `pt-egz73l9j` | RoboCasa delta | 0.49 → 0.23 |
| `pt-1mk5zk1u` | bowls abs | 0.76 → 0.19 |
| `pt-72m9e81n` | bowls delta | 0.62 → 0.31 |

四个 run 都写出 `steps_100_pytorch_model.pt`。产物在
`outputs/training/*_100step/`。没有 OOM、非有限 loss 或 traceback。

### 仍未做的正式长训

完整 RoboCasa 200k 与 bowls 100k 还没有提交。100-step 只证明 8 卡启动、数据、
初始化、ZeRO-2 和 checkpoint 合同，不代替长训收敛或评测。

## 7. 维护边界

1. 公开代码不计算 norm，也不写入 stats cache。
2. 公开 Git 不提交 ACP 脚本、W&B key、内部绝对路径数据包。
3. 训练和评测分池：A800 训练，4090 仿真 / eval。
4. ARX 正式范围当前只验收 `stack_ceramic_bowls` 的 7 个 session。
5. 不要交换 ARX 三路相机顺序。
6. 不要把 Delta 移到 Common23 concat 或 q99 之后。
7. 不要从 `GR1T2_with_hands.urdf` 重建 34 节点 `GR1.pkl`。
8. Absolute / Delta pretrain 禁止交叉初始化。
9. 异构 SKU 不要超过 4 个，也不要混用不同 GPU 数。
