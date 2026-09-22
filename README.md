# LayaSpeech

LayaSpeech is an experimental framework for real-time expressive speech synthesis using Laya as a low-latency prosody controller.

Instead of generating audio directly with Laya, LayaSpeech distills speaking behavior from a teacher TTS system and learns how speech should be delivered: pitch, duration, energy, pauses, emphasis, and other prosodic controls.

The predicted parameters are then passed to a lightweight speech synthesizer for low-latency or streaming audio generation.

## Concept

### Training pipeline

```text
Teacher TTS
    ↓
Generated speech
    ↓
Alignment / acoustic analysis
    ↓
Pitch / duration / energy / pause / ...
    ↓
Laya training dataset
    ↓
Fine-tuned Laya
```

### Runtime pipeline

```text
Text + context
      ↓
Fine-tuned Laya
      ↓
Prosody parameters
      ↓
Streaming synthesizer
      ↓
Audio
```

The goal is to separate how to speak from audio generation itself.

A relatively heavy TTS model can therefore be used offline as a teacher, while a lightweight Laya-based policy and synthesizer handle real-time inference.

## Planned components

- Teacher TTS adapter interface
- Automatic speech generation pipeline
- Phoneme / mora alignment
- Pitch (F0) extraction
- Duration extraction
- Energy extraction
- Pause and phrase-boundary extraction
- Common prosody dataset format
- Laya fine-tuning pipeline
- Real-time prosody inference
- Streaming synthesis interface
- Pluggable synthesizer backends

## What is not included

This repository is intended to contain the framework and training pipeline only.

It does not distribute:

- Teacher TTS model weights
- Teacher-generated speech datasets
- Reference speaker recordings
- Proprietary or third-party voice data
- Project-specific fine-tuned Laya checkpoints

Users are responsible for supplying teacher models and training data that they have the right to use.

## Status

Early experimental project.

The initial goal is to validate whether a small Laya-based decision model can reproduce useful prosodic behavior from a higher-quality TTS system while remaining fast enough for real-time speech generation.
