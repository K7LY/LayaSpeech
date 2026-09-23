# Teacher adapters

LayaSpeech deliberately does not depend on a specific teacher TTS. For a complete first-time setup, start with [Getting Started](GETTING_STARTED.md).

The initial `CommandTeacherAdapter` launches an external process without a shell, keeping third-party code, weights, voice data, and licenses outside this repository.

## Configuration

```json
{
  "teacher_id": "my-teacher",
  "command": ["python", "teacher_cli.py", "--text", "{text}", "--output", "{output}"],
  "timeout_s": 300
}
```

Supported placeholders are `{text}` and `{output}`. The command must write a WAV to `{output}` and exit successfully. Pipes and shell redirection are not interpreted; wrap complex invocation in a script.

## Manifest

```json
{"id":"u000001","text":"今日はいい天気ですね。"}
{"id":"u000002","text":"本当にそう思いますか？"}
```

An optional `speaker_id` may be included.

## Corpus build

```bash
uv run layaspeech build-corpus examples/manifest.jsonl --teacher-config teacher.json --output-dir data/my-corpus
```

The build is resumable and produces `audio/`, `prosody/`, and `train.jsonl`. Generated data directories are ignored by Git.
