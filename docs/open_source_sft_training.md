# ACE-Ego-0 开源 SFT 训练计划

> 状态（2026-09）：公开轻量 SFT 已经落地。设计合同见
> [`sft_training_implementation.md`](sft_training_implementation.md)，
> 数据布局与集群验收见 [`sft_training_status.md`](sft_training_status.md)。
>
> 计划原文按当时范围写了 4 个 ARX 任务 recipe。落地后的正式验收任务收窄为
> `stack_ceramic_bowls`；其余 recipe 仍留在 `configs/arx/`，但不再作为当前
> 正式训练目标。本文保留原计划，不再按后续范围改写正文。

## 1. 目标与边界

在当前 ACE-Ego-0 eval 仓库上增加一层可验证的轻量训练实现，而不是迁入整个内部
`starVLA` 仓库。

本次发布：

- RoboCasa GR1 TableTop 24（公开名 `robocasa24`）Absolute / Delta SFT。
- 4 个具有正式 recipe 的 ARX 任务：
  `heat_sandwich`、`stack_ceramic_bowls`、`pour_coffee_beans`、
  `scan_pencil_case_new`。
- Common23 Absolute 0428 和 Delta 0427 两个 SFT 初始化 bundle。
- 原始 LeRobot v2.0（RoboCasa）/ v2.1（ARX）schema 的训练数据，以及预先计算好的
  normalization stats。

明确不发布：

- RoboCasa365、Mobile Native34 或其他 action space。
- Common23 449-mix pretrain 训练代码。
- dataset statistics 计算、runtime merge 或 stats cache 写入代码。
- human-video 数据清洗、dynamic chunking、Memory、Ceph、ACP/集群脚本。
- optimizer state、ARX 真机 serving 或 RoboTwin evaluation。

论文包含 6 个 ARX 评测任务；当前训练仓只有上述 4 个任务具备正式、可复现的
0912 recipe。本次不推测另外两个任务的训练合同。

## 2. 公开命名与发布物

- 方法和 framework：`ACE-Ego-0`
- Python package：`ace_ego_0`
- 动作空间：`Common23 camera-space EEF`
- 机器人：`GR1`、`ARX5Dual`
- 初始化模型：
  - `ace-ego-0-pretrain-absolute`
  - `ace-ego-0-pretrain-delta`
- RoboCasa SFT/eval 模型继续使用：
  - `ace-ego-0-absolute`
  - `ace-ego-0-delta`

公开文档、命令和新配置不使用 `starVLA` / `QwenPI` 作为模型名称。为兼容已经训练的
checkpoint，PyTorch module attribute 和 state-dict key 不改名。

两个 pretrain bundle 是 SFT 初始化产物，不是可直接用于目标任务 serving 的 policy。
每个 bundle 包含最小模型 config、原始 `.pt` 和 SHA256 manifest；不附 449-mix runtime
stats，避免将 pretrain stats 错用于 RoboCasa 或 ARX。

## 3. 模型和 checkpoint 合同

### 3.1 固定模型参数

- Qwen3-VL-4B-Instruct + Flow-Matching Action Expert。
- `action_dim=23`、`state_dim=23`、`action_horizon=30`。
- `future_action_window_size=29`、`past_action_window_size=0`。
- State Timestep Conditioning（内部兼容字段 `state_design_preset: rdt2`）。
- URDF morphology conditioning 使用 body token 和 chain token。
- VLM BF16；action expert FP32。

### 3.2 GR1 / ARX morphology cache

模型侧 GR1 必须使用训练时的 34 节点 morphology cache：

```text
assets/urdf_cache/GR1.pkl
SHA256 dea552de1d4449e8c5db5cf7215b80a171c155b5c430dda00e44402e361acc1d
```

`assets/GR1T2_with_hands.urdf` 是 58 关节的仿真 / Mink IK 资产，不用于重建模型
cache。此前从该 URDF 生成的 58 节点 cache 与训练输入不同，必须替换。

ARX 使用 cache-only：

```text
assets/urdf_cache/ARX5Dual.pkl
SHA256 7ebd6dad97105fa17a049dbec75912f3c32d24193ccd84562a78620e6dbd3fa6
```

公开训练禁止自动重建 cache。

### 3.3 Pretrain 与 SFT 的加载合同

完整 pretrain bundle 的 config 声明 checkpoint 中全部 human surrogate token，因此完整
bundle 可以 strict load。

目标 SFT 模型不包含 human token。初始化时：

- `missing_keys` 必须为空。
- 只允许
  `action_model.learnable_urdf_embeddings.human_video*`
  这 10 个已审计 key 出现在 `unexpected_keys`。
- 任何其他 missing / unexpected key 立即失败。

这与 0820 RoboCasa / 0912 ARX 的生产初始化行为一致，同时避免普通
`strict=False` 静默吞掉结构错误。最终 SFT state dict 不含 human token，与现有
RoboCasa eval 模型拓扑一致。

Absolute 只能从 0428 Absolute 初始化；Delta 只能从 0427 Delta 初始化。bundle
metadata 与训练 recipe 不一致时启动即失败。

训练 config 和导出的 inference config 分离。后者继续满足现有 eval bundle 白名单，不含
`trainer`、`optimizer`、`scheduler`、`data_mix` 或内部路径。

## 4. 最小训练前向

`FlowMatchingActionExpert.forward()` 只实现生产 recipe 使用的 velocity 参数化：

1. 从 `Beta(1.5, 1.0)` 采样时间，应用 `noise_s=0.999`。
2. 对无效 action 维同时 mask clean action 和 sampled noise。
3. 构造 `x_t = (1 - t) * noise + t * action`。
4. target velocity 为 `action - noise`。
5. 复用现有 action encoder、state timestep conditioning、URDF tokens 和 DiT。
6. 使用 mask 后的 velocity MSE 作为唯一 backward objective。

Common23 的 left/right xyz、rot6d、gripper、waist loss group 只用于诊断日志，不改变
`total_loss` 的权重。ARX zero-padded waist 的 mask 为 False，不能贡献 loss。

不迁移未启用的 sample parameterization、geodesic loss、action-mask conditioner、
human auxiliary loss、render image 或 temporal memory 分支；配置要求这些能力时明确报错。

`ACEEgoPolicy.forward()` 接受 dataset 输出的 `list[dict]`，支持：

- RoboCasa 单目和 ARX 三目静态图片。
- `lang`、`action`、`state`、`state_mask`、`mask`、`robot_name`。
- policy 层 `repeated_diffusion_steps=4`。

现有 `predict_action()` 和 RoboCasa websocket/eval 接口不变。

## 5. 数据合同

公开数据维持现有 LeRobot v2.0 / v2.1 磁盘 schema。公开代码提供基于 pyarrow / PyAV 的白名单
reader，不依赖完整 `lerobot` Python package。

变换顺序固定为：

```text
raw keys
  -> optional per-key delta (state_anchor + first_state + subtract)
  -> q99 normalization
  -> Common23 concatenation + mask
```

禁止先拼成 23D 再做 delta。Delta 路径对 xyz / rot6d / waist 做 subtract；gripper
保持绝对值。

### 5.1 RoboCasa 24

- Video key：`video.ego_view`
- Language：episode `remarks`，prompt 为 `{instruction}`
- Image：224×224
- Robot type：`fourier_gr1_eef_gripper_6d` -> `GR1`
- Fourier left/right 6D hand 先转换为 1D gripper
- Common23 的 23 个 action/state 维全部有效
- Manifest 保留生产 24 个完整 dataset 名，不用 `<task>_1000` 猜测短名

### 5.2 ARX

- Video keys：`video.cam_high`、`video.cam_left_wrist`、
  `video.cam_right_wrist`
- Language：`annotation.human.action.task_description`
- Image：256×256
- Disk robot type：`arx_dual_arm`
- Training robot type：`arx_dual_arm_common23`
- Model robot name：`ARX5Dual`
- raw20：left EEF 9D + gripper 1D + right EEF 9D + gripper 1D
- Common23：拆出 xyz / rot6d 后追加 3 个 zero-padded waist 维，waist mask=False
- `stack_ceramic_bowls` manifest 保留全部 7 个 session

### 5.3 Shipped stats

每个数据集至少包含 `meta/info.json`、`meta/modality.json`、absolute stats 和 delta
stats。每个公开 recipe 通过不可变 `norm_manifest.json` 指向已经发布的 merged stats。

- RoboCasa abs / delta 分别提供 24-task merged stats。
- ARX 每个任务分别提供 abs / delta merged stats。
- `pour_coffee_beans` 和 `scan_pencil_case_new` 必须在数据发布前补齐原数据缺少的 stats。
- `stack_ceramic_bowls` 直接发布按 `parent_task` 口径合并好的 stats。

loader 对 stats 文件、key、维度和 abs/delta metadata 做严格校验。缺失或不匹配时立即
失败；不会扫描 parquet 计算 stats，也不会写 cache。

## 6. Trainer 和 recipe

公开入口：

```bash
python scripts/train.py --config configs/robocasa24/absolute_sft.yaml
python scripts/train.py --config configs/arx/heat_sandwich_delta_sft.yaml
```

trainer 只包含：

- Accelerate + DeepSpeed ZeRO-2。
- 全参数 SFT；不使用 LoRA。
- base / Qwen / action expert 分组学习率。
- AdamW、gradient clipping 1.0。
- 5000-step warmup + cosine-with-min-LR。
- flat `.pt` checkpoint、JSONL metrics；W&B 为独立可选依赖。

启动必须使用 Accelerate `mixed_precision=no`；模型内部自行保持 VLM BF16 / action FP32，
`reduce_in_full_precision=true`，AdamW `foreach=false`。

公开 recipe 共 10 份：

- RoboCasa：Absolute / Delta 各一份，bs16，200k step，base LR `1e-5`，
  Qwen LR `2e-5`，action LR `1e-4`。
- ARX：4 task × Absolute / Delta，bs16，100k step，base/Qwen LR `1e-5`，
  action LR `1e-4`。

所有超参数逐字段对齐内部 0820 / 0912 production recipe，只允许改变公开名称、数据路径、
checkpoint 路径和输出路径。

不实现 optimizer-state resume、dynamic horizon、内置周期 benchmark eval、集群 launcher
或 norm 生成。

## 7. 依赖和仓库卫生

- package 名从 eval-only 名称调整为 `ace-ego-0`。
- `diffusers` 是当前推理模型的直接依赖，放基础依赖。
- `[train]` 只加入 import closure 需要的 `pyarrow==14.0.1`、
  `deepspeed==0.16.9` 等最小项。
- W&B 放独立 optional extra。
- `.pt` 继续通过 Hugging Face 发布，不进入普通 Git history。
- `.gitignore` 只放行审核过的 GR1 / ARX cache 和本计划文档。

README 需要分别说明：

- 数据与 pretrain bundle 下载。
- 单卡 / 多卡训练命令与硬件预期。
- Pretrain initialization、RoboCasa SFT/eval、ARX SFT 的用途。
- RoboCasa24 不是 RoboCasa365，ARX 本次不提供 serving。
- MIT 代码、Qwen3-VL、RoboCasa/RoboSuite 和公开数据各自的许可证 / 再分发条款。

## 8. 实施顺序与验证闸门

1. 修复 `.gitignore`，锁定 34 节点 GR1 与 ARX cache；跑现有测试和 RoboCasa
   abs/delta 1-episode smoke。
2. 实现训练 forward。固定 seed 和同一个序列化 batch，对齐内部实现的 noisy trajectory、
   target/pred velocity、mask、total/group loss。
3. 实现 loader。固定 `(episode, frame)` 对齐 RoboCasa 和 ARX 的
   image/lang/action/state/mask/robot_name，并验证 delta-before-normalize。
4. 实现安全 pretrain 初始化和 checkpoint 导出；验证 pretrain strict load、
   SFT allowlisted load、最终 SFT strict load。
5. 实现 trainer 和 10 份 recipe；执行 RoboCasa abs/delta 各 2-step、ARX 至少一个任务
   abs/delta 各 2-step。
6. 验证基础 eval 安装和 `pip install -e ".[train]"`；运行 tests、Ruff、RoboCasa smoke。
7. 扫描内部绝对路径、用户名、W&B/ACP 凭据和 RoboCasa365/Mobile 入口；两个 pretrain
   `.pt` 必须与 0428 / 0427 源文件 SHA256 完全一致。

## 9. 完成标准

当时计划的完成标准：

- 使用公开 raw LeRobot v2.x 数据、随数据发布的 stats 和两个 pretrain bundle，可启动
  RoboCasa 24 与 4 个 ARX task 的 abs/delta SFT。
- loader 和单步 flow-matching loss 都通过内部生产实现的固定样本数值对齐，而不只满足
  “能跑”。
- norm 永不在运行时计算；RoboCasa365、449-mix pretrain、ARX serving 和内部基础设施不进入
  公开代码。
- 现有 RoboCasa eval、checkpoint contract 和 CLI 不回归。
- 模型 morphology conditioning 使用 canonical 34 节点 GR1 cache。

落地结果：

- 公开代码、10 份 recipe、只读 loader、安全初始化和 ZeRO-2 trainer 均已合入。
- 固定样本数值对齐、checkpoint contract、RoboCasa 1-episode smoke 均已通过。
- 正式验收范围后来收窄为 RoboCasa 24 + `stack_ceramic_bowls`。
- A800 池已完成 2 卡 2-step 与 8 卡 100-step；完整 200k / 100k 长训尚未提交。
- 细节见 [`sft_training_status.md`](sft_training_status.md)。
- 落地后的空目录和内部仓遗留代码已按
  [`sft_training_implementation.md`](sft_training_implementation.md) 第 13 节清理。
