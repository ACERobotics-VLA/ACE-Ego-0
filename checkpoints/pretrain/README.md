# ACE-Ego-0 Common23 pretrain initialization

The two model files are distributed through Hugging Face and are intentionally
not stored in Git:

```text
ace-ego-0-pretrain-absolute/checkpoints/ace_ego_0_pretrain_absolute.pt
ace-ego-0-pretrain-delta/checkpoints/ace_ego_0_pretrain_delta.pt
```

These checkpoints initialize target-data SFT. They are not target-task policies,
and their original 449-mixture normalization statistics must not be used for
RoboCasa or ARX inference. Each SFT recipe loads normalization statistics from
the released target dataset instead.

The bundle configs retain the ten human-video surrogate parameters so each full
pretrain state dict can be loaded strictly. The SFT initializer intentionally
drops exactly those ten parameters and rejects every other state-dict mismatch.
