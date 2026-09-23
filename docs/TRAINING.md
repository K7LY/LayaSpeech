# Local Laya fine-tuning

LayaSpeech includes a single-device trainer rather than CI/cloud training. **GitHub Actions is not used for model training or tests.**

For complete setup, start with [Getting Started](GETTING_STARTED.md).

Install the NVIDIA training environment explicitly on Windows or Linux:

```bash
uv sync --extra train --extra cuda
```

The `cuda` extra selects PyTorch's CUDA 12.8 wheels. A plain `--extra train` install may resolve a CPU-only PyTorch build and is not sufficient for GPU training.

## Input and objective

The trainer consumes chunked `state/questions/gold` JSONL produced by `export-jsonl` or `build-corpus`.

It follows Laya's published fine-tuning approach at a smaller single-device scale: noisy grouped logit samples, Laya's `proper_reward`, group-relative policy-gradient advantage, and soft cross-entropy guidance.

## Command

```bash
uv run layaspeech train data/my-corpus/train.jsonl --output-dir checkpoints/layaspeech --device cuda
```

Useful options include `--epochs`, `--micro-batch`, `--grad-accum`, `--group-size`, `--model-id`, `--subfolder`, and `--device`.

For a 12 GB-class GPU, defaults are micro batch 2, gradient accumulation 8, and group size 4. Reduce `--micro-batch` to 1 first if VRAM is insufficient.

## Output

The checkpoint is compatible with `laya.load(local_path)` and contains `model.safetensors`, `rl_agent_config.json`, `training_meta.json`, `encoder/`, and `tokenizer/`. A rolling `checkpoint_latest/` is overwritten after each epoch.
