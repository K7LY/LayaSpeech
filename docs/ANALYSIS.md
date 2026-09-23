# Acoustic analysis plan

The PoC splits teacher-audio analysis into two stages:

1. **forced alignment** — determine Japanese mora/phone timing;
2. **acoustic extraction** — measure F0 and energy inside those boundaries.

## Acoustic extraction

`layaspeech.analysis.extract_world_features()` accepts aligned mora boundaries
and extracts:

- voiced F0 samples using WORLD DIO + StoneMask;
- RMS energy in dBFS;
- mora duration;
- consonant/vowel duration supplied by the aligner;
- inter-mora pause duration.

Install the optional dependencies with:

```bash
uv sync --extra analysis
```

The PoC uses `pyworld-prebuilt` so Windows/Python 3.11 can use the same
`import pyworld` API without requiring a local C++ build.

## Japanese frontend and forced alignment

OpenJTalk is used only to obtain the spoken Katakana reading:

```text
Japanese text
    ↓
pyopenjtalk.g2p(..., kana=True)
    ↓
Katakana reading
    ↓
mora split
```

Install it with:

```bash
uv sync --extra alignment
```

The first forced-alignment backend is Montreal Forced Aligner (MFA).

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner -y
conda activate aligner

mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_katakana_mfa
```

### Mora-token alignment

Instead of aligning normal words and trying to infer mora boundaries afterward,
LayaSpeech writes the Katakana reading with one mora per whitespace-separated
token:

```text
コンニチワ
↓
コ ン ニ チ ワ
```

`MfaMoraAligner` sends that transcript to `mfa align_one` with tokenization
disabled and JSON output enabled. MFA's resulting `words` tier is therefore
already the mora tier. The `phones` tier is used only to measure consonant and
vowel duration within each mora.

This avoids a fragile phone-to-mora regrouping pass.

### Current conservative limitation

Standalone sokuon (`ッ`) and prolonged sound marks (`ー`) are rejected by the
initial aligner until their behavior with the Katakana G2P model is validated
against real MFA output. The splitter recognizes them; the aligner refuses to
silently produce questionable labels.

## End-to-end analyzer

`MfaWorldAnalyzer` now represents the intended automatic path:

```text
(text, WAV)
    ↓
OpenJTalk Katakana reading
    ↓
MFA mora-token alignment
    ↓
WORLD F0 + energy extraction
    ↓
UtteranceProsody
    ↓
Laya training JSONL
```

The remaining work here is environmental validation against real MFA/Irodori
outputs and expanding special-mora coverage.

## Pitch reference

The default `speaker` pitch reference compares mora F0 with a corpus-wide
speaker median. This is appropriate only when the teacher corpus uses a stable
speaker identity. A no-reference TTS teacher may change voice register between
utterances; in that case the global target confounds speaker identity with
prosody.

Use utterance-local residual pitch targets for that corpus:

```bash
uv run layaspeech export-jsonl prosody/*.json \
  --output train.jsonl \
  --pitch-reference utterance
```

This subtracts each utterance's median F0 before seven-level quantization. It
does not change pitch-direction, duration, energy, or pause targets. Prefer a
fixed reference voice and `speaker` mode when the teacher supports one.
