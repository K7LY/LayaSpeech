# VOICEVOX backend

The first concrete runtime backend targets a separately running VOICEVOX-compatible Engine over HTTP. LayaSpeech does not bundle VOICEVOX code, voices, or model weights.

For complete setup, start with [Getting Started](GETTING_STARTED.md).

## Engine

The default endpoint is `http://127.0.0.1:50021`. Find available style IDs with:

```powershell
Invoke-RestMethod http://127.0.0.1:50021/speakers | ConvertTo-Json -Depth 10
```

Use a returned `style.id` as `--speaker`.

## Current mapping

| LayaSpeech control | VOICEVOX PoC mapping |
| --- | --- |
| pitch level | multiply existing mora F0 by a semitone-derived ratio |
| duration level | scale consonant and vowel durations |
| pause after | adjust an existing phrase-boundary `pause_mora` |
| pitch direction | not represented by normal AudioQuery |
| per-mora energy | not represented by normal AudioQuery |

Unsupported controls remain in the common schema for future backends.

## Streaming

`VoicevoxSynthesizer.synthesize_stream()` requests an AudioQuery, applies supported controls, posts it to `/streaming_synthesis`, and yields returned WAV bytes incrementally.

The current runtime plans the sentence's prosody before starting the VOICEVOX request. Sentence/clause segmentation should sit above this layer when integrating a token-streaming LLM.
