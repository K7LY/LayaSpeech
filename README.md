# LayaSpeech

**LayaSpeech** is an experimental framework for distilling expressive speaking behavior into [Laya](https://github.com/NandhaKishorM/laya) and using the resulting low-latency prosody policy for real-time speech synthesis.

LayaSpeech does not generate waveforms with Laya. Laya predicts **how to speak**: relative pitch, duration, energy, pauses, and related prosodic controls. A separate synthesizer produces the audio.

> **Status:** early proof of concept. The pipeline is implemented, but real teacher-corpus training and end-to-end quality/latency validation are still in progress.

## How it works

```text
Training:
Teacher TTS -> WAV -> OpenJTalk + MFA -> WORLD -> prosody dataset -> fine-tuned Laya

Runtime:
Text + context -> fine-tuned Laya -> prosody controls -> synthesizer -> streaming audio
```

The first PoC targets Japanese and predicts five controls per mora: relative pitch level, pitch direction, relative duration, relative energy, and following pause.

Teacher-side data keeps the original F0 contour and timings, so changing the Laya target representation later does not require regenerating teacher audio.

## Quick start

For installation through the first synthesized WAV, follow **[Getting Started](docs/GETTING_STARTED.md)**.

Once configured, the main flow is:

```bash
uv run layaspeech build-corpus examples/manifest.jsonl --teacher-config teacher.json --output-dir data/my-corpus
uv run layaspeech train data/my-corpus/train.jsonl --output-dir checkpoints/layaspeech
uv run layaspeech predict "今日はいい天気ですね。" --checkpoint checkpoints/layaspeech
uv run layaspeech synthesize "今日はいい天気ですね。" --checkpoint checkpoints/layaspeech --speaker 1 --output outputs/example.wav
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) — installation through first synthesis
- [PoC design](docs/POC.md) — scope and success criteria
- [Teacher adapters](docs/TEACHERS.md) — connect an external teacher TTS
- [Acoustic analysis](docs/ANALYSIS.md) — OpenJTalk, MFA, and WORLD
- [Local training](docs/TRAINING.md) — single-GPU Laya fine-tuning
- [VOICEVOX backend](docs/VOICEVOX.md) — runtime backend and mappings

## What is deliberately not distributed

This repository contains framework code and pipeline tooling. It does **not** distribute teacher TTS weights, teacher-generated speech datasets, reference speaker recordings, proprietary voice data, or project-specific fine-tuned Laya checkpoints.

Users are responsible for supplying teacher models and training data they have the right to use.

## Development

LayaSpeech targets Python 3.11+.

```powershell
./scripts/check.ps1
```

The check script runs dependency setup, pytest, Python compilation, and Ruff locally. **No GitHub Actions workflows are used.**

## Current limitations

- MFA mora-token alignment still needs validation against real teacher WAVs.
- Standalone `ッ` and `ー` are conservatively rejected by the current aligner.
- A real teacher corpus and fine-tuned checkpoint still need to be produced.
- Speech quality and first-audio latency have not yet been benchmarked.
- VOICEVOX AudioQuery currently maps pitch level, duration, and existing phrase-boundary pauses. Pitch direction and per-mora energy are retained for future backends.
