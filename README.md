# LayaSpeech

**LayaSpeech** is an experimental framework for real-time expressive speech synthesis using [Laya](https://github.com/NandhaKishorM/laya) as a low-latency prosody controller.

Instead of generating audio directly with Laya, LayaSpeech distills speaking behavior from a teacher TTS system and learns **how speech should be delivered**: pitch, duration, energy, pauses, and related prosodic controls.

The predicted controls are passed to a lightweight speech synthesizer for low-latency or streaming audio generation.

## Concept

### Training pipeline

```text
Teacher TTS
    ↓
Generated speech
    ↓
Alignment / acoustic analysis
    ↓
High-resolution prosody dataset
    ↓
Laya target conversion
    ↓
Fine-tuned Laya
```

### Runtime pipeline

```text
Text + context
      ↓
Fine-tuned Laya
      ↓
Prosody controls
      ↓
Streaming synthesizer
      ↓
Audio
```

The goal is to separate **how to speak** from **audio generation itself**.

A relatively heavy TTS model can be used offline as a teacher, while a lightweight Laya-based policy and synthesizer handle real-time inference.

## PoC scope

The first PoC targets Japanese speech and predicts five controls per focused mora:

- relative pitch level (7 levels)
- pitch direction (`falling` / `level` / `rising`)
- relative duration (7 levels)
- relative energy (7 levels)
- following pause (`none` / `short` / `medium` / `long`)

Teacher-side acoustic data retains the original F0 contour and timings instead of only the quantized Laya targets. This keeps the dataset reusable if the target representation changes later.

See [`docs/POC.md`](docs/POC.md) for the current boundaries and success criteria.

## Repository layout

```text
src/layaspeech/
├── schema.py       # teacher-agnostic high-resolution prosody schema
├── targets.py      # acoustic features -> compact Laya targets
├── policy.py       # Laya request/inference adapter
├── adapters.py     # teacher, analyzer, synthesizer interfaces
└── cli.py

examples/
tests/
docs/
```

## Development

Python 3.11+ is used for the PoC.

```bash
uv sync --extra dev
uv run pytest
```

Inspect the target conversion using the synthetic example:

```bash
uv run layaspeech targets examples/sample_prosody.json
```

Laya itself is an optional dependency during dataset/schema development:

```bash
uv sync --extra laya
```

The initial Japanese runtime uses Laya's multilingual checkpoint unless a fine-tuned checkpoint is supplied.

## What is not included

This repository contains the framework and training/runtime plumbing only.

It does **not** distribute:

- teacher TTS model weights
- teacher-generated speech datasets
- reference speaker recordings
- proprietary or third-party voice data
- project-specific fine-tuned Laya checkpoints

Users are responsible for supplying teacher models and training data that they have the right to use.

## Status

Early PoC.

Completed foundation:

- common high-resolution prosody schema
- relative speaker/teacher baseline calculation
- first 7-level target quantization
- Laya request builder for five prosody controls
- swappable teacher/analyzer/synthesizer interfaces
- dependency-free dataset inspection CLI
- unit tests for schema and target conversion

Next blocker: automatic Japanese mora alignment and acoustic feature extraction from teacher-generated audio.
