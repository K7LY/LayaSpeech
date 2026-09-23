# Teacher adapters

LayaSpeech deliberately does not depend on a specific teacher TTS.

The first adapter is `CommandTeacherAdapter`, which launches an external
process without a shell. This keeps third-party model code, model weights, voice
data, and licenses outside this repository.

## Configuration

Create a JSON file:

```json
{
  "teacher_id": "my-teacher",
  "command": [
    "python",
    "teacher_cli.py",
    "--text",
    "{text}",
    "--output",
    "{output}"
  ],
  "timeout_s": 300
}
```

Only two placeholders are interpreted:

- `{text}`
- `{output}`

The command must write a WAV file to `{output}`.

## Manifest

Input text is JSONL:

```json
{"id":"u000001","text":"今日はいい天気ですね。"}
{"id":"u000002","text":"本当にそう思いますか？"}
```

An optional `speaker_id` can be included per row.

## Corpus build

After installing the alignment and analysis dependencies and configuring MFA:

```bash
uv run layaspeech build-corpus examples/manifest.jsonl \
  --teacher-config teacher.json \
  --output-dir data/my-corpus
```

The build is resumable by default:

```text
data/my-corpus/
├── audio/      # teacher WAVs
├── prosody/    # high-resolution UtteranceProsody JSON
└── train.jsonl # chunked Laya state/questions/gold records
```

The repository's `.gitignore` excludes generated audio/data directories so
teacher outputs are not accidentally committed.
