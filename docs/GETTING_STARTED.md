# Getting Started

This guide takes a fresh LayaSpeech checkout to the first synthesized WAV. The current PoC is easiest to run on Windows with an NVIDIA GPU.

## 1. Prerequisites

Install Git, Python 3.11+, uv, and Miniconda/Miniforge. GPU training also requires a PyTorch-compatible NVIDIA/CUDA environment. The current runtime backend requires a separately running VOICEVOX-compatible Engine.

The teacher TTS is intentionally not bundled. It may be any command-line program that accepts text and writes a WAV file.

## 2. Clone and install

```powershell
git clone https://github.com/K7LY/LayaSpeech.git
cd LayaSpeech
uv sync --extra dev --extra analysis --extra alignment --extra train
uv run pytest
uv run layaspeech targets examples/sample_prosody.json
```

For routine local checks:

```powershell
./scripts/check.ps1
```

No GitHub Actions workflow is required or used.

## 3. Install Montreal Forced Aligner

```powershell
conda create -n layaspeech-aligner -c conda-forge montreal-forced-aligner -y
conda activate layaspeech-aligner
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_katakana_mfa
mfa version
```

The `mfa` executable must be visible on PATH when LayaSpeech runs, or pass its full path with `--mfa-executable`.

## 4. Check Japanese mora segmentation

```powershell
uv run layaspeech moras "こんにちは"
```

You should receive Katakana mora tokens. For the first smoke test, avoid sentences containing standalone `ッ` or `ー`; the current aligner deliberately rejects those until validated.

## 5. Connect a teacher TTS

```powershell
Copy-Item examples/teacher-config.example.json teacher.json
```

Edit `teacher.json`:

```json
{
  "teacher_id": "my-teacher",
  "command": [
    "python",
    "D:/path/to/teacher_cli.py",
    "--text",
    "{text}",
    "--output",
    "{output}"
  ],
  "timeout_s": 300
}
```

The command must accept `{text}`, write a WAV exactly to `{output}`, and exit with code 0. LayaSpeech invokes it directly without a shell. For Irodori-TTS or another teacher, use a small wrapper script if its native CLI does not already fit this interface.

## 6. Prepare a manifest

Create `data/smoke-manifest.jsonl`:

```json
{"id":"u000001","text":"今日はいい天気ですね。"}
{"id":"u000002","text":"本当にそう思いますか？"}
```

Start with a few short sentences. An optional `speaker_id` may be included per row.

## 7. Build the teacher corpus

```powershell
uv run layaspeech build-corpus data/smoke-manifest.jsonl `
  --teacher-config teacher.json `
  --output-dir data/smoke-corpus `
  --mfa-executable mfa
```

This performs teacher synthesis, OpenJTalk reading conversion, MFA alignment, WORLD F0/energy extraction, high-resolution prosody export, and Laya training-data export.

Output:

```text
data/smoke-corpus/
├── audio/
├── prosody/
└── train.jsonl
```

The build resumes from existing files by default.

To analyze an existing WAV without a teacher:

```powershell
uv run layaspeech analyze "今日はいい天気ですね。" .\teacher.wav --output .\prosody.json --mfa-executable mfa
uv run layaspeech targets .\prosody.json
```

## 8. Fine-tune Laya

```powershell
uv run layaspeech train data/smoke-corpus/train.jsonl `
  --output-dir checkpoints/smoke `
  --device cuda `
  --epochs 1
```

The default starts from Laya's multilingual checkpoint. The first run downloads the upstream checkpoint. Initial 12 GB-class GPU defaults are micro batch 2, gradient accumulation 8, and RL group size 4. If VRAM is insufficient, retry with `--micro-batch 1`.

## 9. Inspect the learned policy

```powershell
uv run layaspeech predict "今日はいい天気ですね。" --checkpoint checkpoints/smoke --device cuda
```

The result is JSON containing predicted prosody controls per mora.

## 10. Start VOICEVOX Engine

Start a VOICEVOX-compatible Engine separately. The default endpoint is `http://127.0.0.1:50021`.

Find a style ID:

```powershell
Invoke-RestMethod http://127.0.0.1:50021/speakers | ConvertTo-Json -Depth 10
```

Use a returned `style.id` as `--speaker`.

## 11. Generate the first WAV

```powershell
uv run layaspeech synthesize "今日はいい天気ですね。" `
  --checkpoint checkpoints/smoke `
  --device cuda `
  --speaker <STYLE_ID> `
  --output outputs/smoke.wav
```

The runtime derives moras, runs the fine-tuned Laya policy in bounded batches, modifies a VOICEVOX AudioQuery, calls streaming synthesis, and writes the streamed WAV bytes.

## 12. What this proves

A successful smoke test proves the plumbing:

```text
teacher -> WAV -> alignment -> acoustic features -> Laya training data
        -> fine-tuned checkpoint -> prosody prediction -> VOICEVOX -> WAV
```

It does not yet prove that distilled prosody is better than neutral synthesis. That requires a larger corpus and held-out quality/latency evaluation.

## Troubleshooting

**MFA executable not found:** activate the MFA Conda environment or pass the full executable path with `--mfa-executable`.

**`NotImplementedError` mentioning `ッ` or `ー`:** use a sentence without those moras for the initial PoC.

**Laya falls back to CPU on an RTX 50-series GPU:** install a PyTorch build that supports the GPU's CUDA architecture, then rerun with `--device cuda`.

**VOICEVOX mora count mismatch:** OpenJTalk and VOICEVOX produced different mora counts. LayaSpeech stops rather than applying controls to the wrong moras.

**Training runs out of VRAM:** retry with `--micro-batch 1`.
