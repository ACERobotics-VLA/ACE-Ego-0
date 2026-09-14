# ACE-Ego-0 公开训练文档

本目录只记录公开 SFT 的设计、实现和验收，不包含内部 `starVLA` 全量训练仓。

| 文档 | 用途 |
|---|---|
| [sft_training_implementation.md](sft_training_implementation.md) | 设计合同、代码结构、Common23 / checkpoint / trainer 细节；第 13 节记录已完成的冗余清理 |
| [sft_training_status.md](sft_training_status.md) | 落地后的数据布局、本地辅助脚本、集群验收和当前正式范围 |
| [open_source_sft_training.md](open_source_sft_training.md) | 最初的开源范围与实施计划，保留作历史依据 |

当前正式训练范围：

- RoboCasa GR1 TableTop 24 Absolute / Delta
- ARX `stack_ceramic_bowls` Absolute / Delta

公开入口是 `scripts/train.py` 与 `configs/`。`data/`、`scripts/local/` 和集群
launcher 不进入公开 Git。
