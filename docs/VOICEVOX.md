# VOICEVOX backend

The first concrete runtime backend targets a separately running
VOICEVOX-compatible Engine over HTTP.

LayaSpeech does not bundle VOICEVOX code, voices, or model weights.

## Why this backend

The current VOICEVOX Engine exposes:

- editable per-mora pitch;
- editable consonant/vowel duration;
- AudioQuery-based synthesis;
- `/streaming_synthesis` for incremental audio delivery.

That is enough to validate the core LayaSpeech idea without writing a vocoder
for the PoC.

## Mapping

The normal VOICEVOX speech AudioQuery can represent only part of the common
LayaSpeech control schema.

| LayaSpeech control | VOICEVOX PoC mapping |
| --- | --- |
| pitch level | multiply existing mora F0 by a semitone-derived ratio |
| duration level | scale consonant and vowel durations |
| pause after | adjust an existing phrase-boundary `pause_mora` |
| pitch direction | not represented by normal AudioQuery |
| per-mora energy | not represented by normal AudioQuery |

The unsupported controls remain in the common schema so a later frame-level or
custom synthesizer can use them.

## Streaming

`VoicevoxSynthesizer.synthesize_stream()`:

1. requests a normal AudioQuery;
2. applies LayaSpeech controls;
3. posts the modified query to `/streaming_synthesis`;
4. yields the returned WAV bytes incrementally.

The Laya policy currently plans all prosody chunks for one sentence before
starting the VOICEVOX request. Sentence/clause segmentation should happen above
this layer when integrating with a token-streaming LLM.
