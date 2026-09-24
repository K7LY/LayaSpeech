# Tsukuyomi-chan fixed-reference PoC

## Goal

Repeat the Irodori-TTS distillation PoC with a **fixed Japanese reference voice**
so speaker/register variation is not learned as prosody.

## Reference source

Use **Tsukuyomi-chan Corpus Vol.1** (CV. Rei Yumesaki / 夢前黎).

Official page:
https://tyc.rei-yumesaki.net/material/corpus/

Official ZIP currently linked by the project:
https://tyc.rei-yumesaki.net/files/voice/tyc-corpus1.zip

The corpus contains 100 Japanese utterances from one speaker. The official page
states that research/development where the original voice or generated voice is
not externally published can be used without restriction.

Do not commit the corpus, extracted clips, generated teacher WAVs, or checkpoints.

## Fixed reference construction

Irodori-TTS v4.1-Small recommends multiple short, clean clips from the same
speaker rather than one uninterrupted long recording.

1. Download and unpack Tsukuyomi-chan Corpus Vol.1 outside tracked repository data.
2. Inspect all corpus WAVs and reject any clip with obvious clipping, long silence,
   or unusual acting/noise.
3. Select several neutral/ordinary-reading clips totaling **25–35 seconds**.
4. Keep the selected ordered clip list fixed for the entire experiment.
5. Use Irodori `--ref-wavs` with the same ordered list for every teacher utterance.
6. Keep default reference loudness normalization (`--ref-normalize-db -16`).
7. Keep a fixed inference seed and all other inference parameters fixed.
8. If supported conveniently, precompute reference latents once and reuse them;
   otherwise reuse the exact same WAV list.

Do not use `--no-ref` for this experiment.

## Target design

Use the corrected target design from the previous PoC:

- **pitch:** utterance-local pitch residual, not a corpus-global F0 median
- **duration:** current relative duration target, unless diagnostics show a new issue
- **energy:** current relative energy target, unless diagnostics show a new issue
- **pause:** keep separate from the four main controls; boundary scarcity must not
  dominate the main training task

The first primary comparison should use the four non-pause controls.

## Corpus size and split

Generate one fixed 500-utterance teacher corpus, then derive nested subsets from it:

- 30 utterances
- 100 utterances
- 500 utterances

Use the **same held-out set** for all three training sizes.

The 30 and 100 subsets must be subsets of the 500 training pool so the learning
curve measures data scale rather than different sentence composition.

Prefer varied Japanese sentences covering statement/question/exclamation,
punctuation, short/medium length, and different mora/phoneme contexts.

## Training

Primary distillation experiment:

- CE-only first
- enough epochs to converge; do not repeat the previous one-epoch undertraining
- measure train and held-out metrics every epoch
- select the best checkpoint by held-out metric, not final epoch

After CE establishes a working baseline, optionally compare RLCD+CE using a
training budget large enough to account for its slower convergence.

## Evaluation

For each 30 / 100 / 500 training size, report:

- train accuracy by control
- held-out accuracy by control
- majority baseline
- ordinal MAE for pitch/duration/energy
- prediction class distribution
- confusion matrices
- collapse diagnostics

Also report macro averages, but do not let pause dominate the headline metric.

Compare at minimum:

1. upstream Laya
2. majority/neutral baseline
3. fine-tuned Laya (CE-only)
4. RLCD+CE if run

## Audio validation

After the best checkpoint is selected:

1. Generate 5–10 held-out sentences through normal VOICEVOX.
2. Generate the same sentences through LayaSpeech-controlled VOICEVOX.
3. Save paired files locally only.
4. Measure first-audio latency and Laya chunk latency.
5. Do not commit or publish Tsukuyomi-derived audio during this private PoC.

## Repository rules

- No GitHub Actions.
- No Tsukuyomi corpus files.
- No reference WAVs or latents.
- No Irodori weights.
- No generated teacher WAVs.
- No generated datasets.
- No fine-tuned checkpoints.
- Commit only generic framework fixes, diagnostics, evaluation tools, and docs.

## Success criterion

This PoC succeeds if increasing the fixed-reference dataset from 30 → 100 → 500
produces a clear held-out learning curve and the fine-tuned policy beats simple
baselines on at least the main pitch/duration/energy/direction controls without
class collapse.

If not, identify whether the remaining limitation is target design, feature
extraction, model capacity/architecture, or insufficient textual context.
