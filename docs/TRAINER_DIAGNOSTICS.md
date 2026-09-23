# Trainer memorization diagnostics

This note records the September 2026 Irodori-TTS trainer audit. Generated WAV,
datasets, weights, checkpoints, and detailed prediction artifacts remain local
and are not part of the repository.

## Finding

The trainer can memorize the supplied labels. Duration-only probes reached
100% raw-head and policy-semantic training accuracy for 1, 4, and 10
utterances. The original collapsed checkpoint was undertrained: it used one
RLCD+CE epoch over five controls. RLCD+CE also needed roughly 2.5–3 times as
many epochs as CE-only to reach 95% accuracy in the controlled probes.

| Utterances | Decisions | CE epoch to 95% | RLCD+CE epoch to 95% | Final accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11 | 9 | 27 | 100% |
| 4 | 57 | 17 | 45 | 100% |
| 10 | 155 | 18 | 44 | 100% |

The failure was not caused by label-index inversion, missing option markers,
zero gradients, inability of the Laya head to represent the labels, or
multi-task interference. A four-control CE model also memorized every training
decision after 30 epochs.

## Generalization and target evidence

Single-control models memorized all 368 training moras, but most failed to beat
the held-out majority baseline:

| Control | Train | Held-out | Held-out majority | Held-out MAE |
| --- | ---: | ---: | ---: | ---: |
| pitch direction | 100% | 34.1% | 34.1% | n/a |
| duration | 100% | 29.3% | 42.7% | 2.04 |
| global pitch | 100% | 6.1% | 17.1% | 2.80 |
| energy | 100% | 28.0% | 15.9% | 1.17 |

The no-reference Irodori corpus does not have a stable speaker register.
Utterance median F0 ranged from about 88 Hz to 621 Hz, while the target used a
single corpus-wide median. This made speaker/voice identity look like prosody.
Utterance-local pitch residuals improved held-out pitch from 6.1% / 2.80 MAE
to 43.9% / 0.70 MAE, though 30 sentences still did not beat its 51.2%
train-majority baseline. All held-out baselines in this note use the majority
class selected from training labels; the held-out labels do not select their
own baseline class.

The acoustic audit also found 13 within-mora F0 jumps above eight semitones and
one clear isolated jump candidate. Those cases need review or filtering, but
they do not explain the dominant pitch extremes: most extremes were coherent
whole-utterance voice-register changes. Duration and energy extremes also
contain phoneme identity and devoicing effects; notably 12 of the 20 lowest
energy moras were `ス`.

## Recommended workflow

1. Establish a CE-only memorization baseline before enabling RLCD:

   ```bash
   uv run layaspeech train train.jsonl \
     --output-dir checkpoint \
     --device cuda \
     --loss-mode ce \
     --control duration_level
   ```

2. For a no-reference teacher that changes voice register, export local pitch
   residuals:

   ```bash
   uv run layaspeech export-jsonl prosody/*.json \
     --output train.jsonl \
     --pitch-reference utterance
   ```

3. Prefer a fixed reference voice when available. Increase to 100 and 500
   sentences only after the teacher identity and target definition are stable.
4. Treat pause as boundary detection followed by boundary-only length
   prediction. The 30-sentence audit had only 33 train and 9 held-out boundary
   examples, which was insufficient for held-out performance.

## Causal classification

- **A — trainer bug:** rejected as the primary cause.
- **B — RLCD/loss problem:** secondary contributor. It converges, but much more
  slowly than CE in this supervised setting; one epoch was inadequate.
- **C — multi-task interference:** rejected for memorization; four controls
  jointly reached 100% train accuracy.
- **D — data shortage:** major cause of held-out failure.
- **E — teacher feature/target design:** major cause, especially global pitch
  with a no-reference teacher and phoneme-dependent duration/energy targets.
- **F — Laya suitability:** not supported by the evidence; Laya memorized every
  tested target set.

The immediate collapse is best explained by an undertrained RLCD+CE schedule.
The remaining held-out failure is best explained by **D + E**, with the pitch
target defect directly demonstrated and partially corrected.
