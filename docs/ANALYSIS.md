# Acoustic analysis plan

The PoC splits teacher-audio analysis into two independent stages:

1. **forced alignment** — determine Japanese phone/mora timing;
2. **acoustic extraction** — measure F0 and energy inside those boundaries.

This keeps the alignment backend replaceable without changing the common
`UtteranceProsody` schema.

## Acoustic extraction

`layaspeech.analysis.extract_world_features()` is implemented.

It accepts already-aligned mora boundaries and extracts:

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

## Forced alignment

The first backend target is Montreal Forced Aligner (MFA), because official
Japanese acoustic, dictionary, and G2P models are available.

Typical setup:

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner -y
conda activate aligner

mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
mfa model download g2p japanese_mfa
```

MFA produces phone/word boundaries. LayaSpeech still needs one explicit bridge:

```text
MFA phone intervals
      ↓
Japanese phone -> mora grouping
      ↓
AlignedMora[]
      ↓
WORLD acoustic extraction
      ↓
UtteranceProsody
```

That phone-to-mora bridge is intentionally not guessed in the initial scaffold.
Japanese long vowels, moraic nasal, geminates, palatalized consonants, and
devoiced vowels need deterministic handling before corpus generation starts.

Once that mapping is implemented, the teacher-data path becomes fully automatic
from `(text, wav)` to Laya training JSONL.
