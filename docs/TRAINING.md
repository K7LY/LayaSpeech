# Local Laya fine-tuning

The PoC includes a single-device trainer rather than CI/cloud training.

**GitHub Actions is not used for model training or tests.**

The trainer consumes the chunked `state/questions/gold` JSONL produced by
`layaspeech export-jsonl` or `layaspeech build-corpus`.

## Objective

The local trainer follows Laya's published fine-tuning recipe at a smaller
single-device scale:

- noisy grouped logit samples;
- Laya's strictly proper `proper_reward`;
- group-relative policy-gradient advantage;
- soft cross-entropy guidance.

Training and validation are split at the exported JSONL row/chunk level before
the questions are flattened into individual Laya training items.

## RTX 5070 starting point

The defaults are deliberately conservative:

- multilingual base checkpoint;
- micro batch 2;
- gradient accumulation 8;
- group size 4;
- gradient checkpointing when supported;
- BF16 on CUDA hardware that supports it, otherwise FP16.

These are starting values, not a guarantee of fitting every PyTorch/CUDA build.

## Output

The output directory is saved in the same basic structure consumed by
`laya.load(local_path)`:

```text
checkpoint/
├── model.safetensors
├── rl_agent_config.json
├── training_meta.json
├── encoder/
└── tokenizer/
```

A rolling `checkpoint_latest/` is also overwritten after each epoch.

Model weights remain ignored by the repository's `.gitignore`.
