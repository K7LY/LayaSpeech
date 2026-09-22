# LayaSpeech PoC

## Question

Can a fine-tuned Laya checkpoint predict useful Japanese prosody controls quickly
enough to drive a streaming synthesizer?

The PoC is deliberately smaller than a production TTS system. It should prove or
disprove the idea before GUI, networking, MLX, or a production audio backend are
added.

## Frozen v0 boundaries

### Teacher side

A `TeacherAdapter` produces audio. An `AcousticAnalyzer` converts the known text
and that audio into `UtteranceProsody`.

Teacher model weights, generated wave files, speaker recordings, and project
checkpoints are data inputs and are not part of the repository.

### Common acoustic schema

Raw analyzed data keeps:

- mora boundaries
- consonant/vowel durations
- high-resolution voiced F0 points
- energy
- following pause

The raw schema intentionally keeps more information than the Laya target space.

### Laya target space

The first PoC predicts five controls per focused mora:

- pitch level: 7 ordinal levels
- pitch direction: falling / level / rising
- duration: 7 ordinal levels
- energy: 7 ordinal levels
- following pause: none / short / medium / long

All numeric controls are relative to a speaker/teacher baseline. This avoids
hard-coding one voice's absolute F0 or loudness into the policy.

### Runtime side

A `SynthesizerAdapter` consumes the predicted controls and yields audio chunks.
The concrete synthesizer is intentionally not selected yet.

## Milestones

1. **Schema / target conversion** — complete in the initial scaffold.
2. **Teacher pipeline** — generate owned/synthetic Japanese utterances and
   automatically extract mora timing, F0, energy, and pauses.
3. **Dataset export** — create Laya fine-tuning records from the analyzed corpus.
4. **Fine-tune** — specialize the multilingual Laya base on the prosody targets.
5. **Offline validation** — compare predicted controls with held-out teacher data.
6. **Streaming prototype** — connect predictions to one lightweight synthesizer.
7. **Latency benchmark** — measure first-audio latency and sustained real-time factor.

## PoC success criteria

The idea is worth continuing if all of the following are true on held-out Japanese
sentences:

- predictions beat simple per-feature majority/median baselines;
- generated speech is measurably closer to the teacher than neutral prosody;
- Laya inference does not dominate end-to-end first-audio latency;
- the runtime can synthesize the next chunk while the current chunk is playing.

Exact quality thresholds should be set after the first dataset exists; choosing
them now would be arbitrary.
