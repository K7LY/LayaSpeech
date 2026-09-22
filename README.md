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

See [`docs/POC.md`](docs/POC.md) and [`docs/ANALYSIS.md`](docs/ANALYSIS.md).

## Repository layout

```text
src/layaspeech/
├── schema.py       # teacher-agnostic high-resolution prosody schema
├── analysis.py     # WORLD F0/energy extraction from aligned moras
├── targets.py      # acoustic features -> compact Laya targets
├── dataset.py      # Laya-compatible training JSONL export
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

Inspect target conversion using the synthetic example:

```bash
uv run layaspeech targets examples/sample_prosody.json
```

Export analyzed utterances to the `state/questions/gold` JSONL shape used by Laya fine-tuning:

```bash
uv run layaspeech export-jsonl examples/sample_prosody.json -o train.jsonl
```

Optional components:

```bash
uv sync --extra laya
uv sync --extra analysis
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
- Laya-compatible fine-tuning JSONL export
- swappable teacher/analyzer/synthesizer interfaces
- WORLD-based F0 and energy extraction for aligned moras
- dependency-free dataset inspection/export CLI
- unit tests for schema, targets, dataset export, and analysis helpers

Current blocker: automatic Japanese phone-to-mora alignment from forced-aligner output.
