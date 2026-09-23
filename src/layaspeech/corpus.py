"""Resumable offline teacher -> acoustic corpus pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import AcousticAnalyzer, TeacherAdapter
from .dataset import export_jsonl
from .schema import UtteranceProsody


@dataclass(frozen=True, slots=True)
class CorpusItem:
    utterance_id: str
    text: str
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id must not be empty")
        if not self.text:
            raise ValueError("text must not be empty")


def load_manifest(path: str | Path) -> tuple[CorpusItem, ...]:
    """Load JSONL rows containing at least id/utterance_id and text."""

    items: list[CorpusItem] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on manifest line {line_number}") from exc

            utterance_id = str(row.get("utterance_id") or row.get("id") or "")
            item = CorpusItem(
                utterance_id=utterance_id,
                text=str(row.get("text") or ""),
                speaker_id=row.get("speaker_id"),
            )
            if item.utterance_id in seen:
                raise ValueError(f"duplicate utterance_id: {item.utterance_id}")
            seen.add(item.utterance_id)
            items.append(item)

    if not items:
        raise ValueError("manifest contains no utterances")
    return tuple(items)


@dataclass(frozen=True, slots=True)
class CorpusBuildResult:
    analyzed: tuple[Path, ...]
    training_jsonl: Path


def build_corpus(
    *,
    items: Iterable[CorpusItem],
    teacher: TeacherAdapter,
    analyzer: AcousticAnalyzer,
    output_dir: str | Path,
    teacher_id: str | None = None,
    chunk_size: int = 4,
    resume: bool = True,
) -> CorpusBuildResult:
    """Generate teacher audio, analyze it, and export Laya training JSONL."""

    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    prosody_dir = output_dir / "prosody"
    audio_dir.mkdir(parents=True, exist_ok=True)
    prosody_dir.mkdir(parents=True, exist_ok=True)

    utterances: list[UtteranceProsody] = []
    analyzed_paths: list[Path] = []

    for item in items:
        audio_path = audio_dir / f"{item.utterance_id}.wav"
        prosody_path = prosody_dir / f"{item.utterance_id}.json"

        if resume and prosody_path.exists():
            utterance = UtteranceProsody.load(prosody_path)
        else:
            if not (resume and audio_path.exists()):
                teacher.synthesize(item.text, audio_path)

            utterance = analyzer.analyze(
                utterance_id=item.utterance_id,
                text=item.text,
                audio_path=audio_path,
                teacher_id=teacher_id,
                speaker_id=item.speaker_id,
            )
            utterance.save(prosody_path)

        utterances.append(utterance)
        analyzed_paths.append(prosody_path)

    training_path = output_dir / "train.jsonl"
    export_jsonl(utterances, training_path, chunk_size=chunk_size)
    return CorpusBuildResult(
        analyzed=tuple(analyzed_paths),
        training_jsonl=training_path,
    )
