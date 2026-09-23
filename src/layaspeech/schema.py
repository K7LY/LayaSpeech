"""Common, teacher-agnostic prosody schema.

The raw acoustic representation intentionally preserves more detail than Laya
will predict. This lets target quantization change without regenerating teacher
audio or rerunning acoustic analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class F0Point:
    """One voiced F0 sample relative to the start of a mora."""

    offset_s: float
    hz: float

    def __post_init__(self) -> None:
        if self.offset_s < 0:
            raise ValueError("offset_s must be >= 0")
        if self.hz <= 0:
            raise ValueError("hz must be > 0")


@dataclass(frozen=True, slots=True)
class MoraFeatures:
    """Acoustic features for one Japanese mora."""

    text: str
    start_s: float
    end_s: float
    consonant: str | None = None
    vowel: str | None = None
    consonant_duration_s: float = 0.0
    vowel_duration_s: float = 0.0
    energy_dbfs: float | None = None
    pause_after_s: float = 0.0
    f0_contour: tuple[F0Point, ...] = ()

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("mora text must not be empty")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("mora timing must satisfy 0 <= start_s < end_s")
        if self.consonant_duration_s < 0 or self.vowel_duration_s < 0:
            raise ValueError("phoneme durations must be >= 0")
        if self.pause_after_s < 0:
            raise ValueError("pause_after_s must be >= 0")
        duration = self.end_s - self.start_s
        if self.consonant_duration_s + self.vowel_duration_s > duration + 1e-6:
            raise ValueError("phoneme durations cannot exceed mora duration")
        if any(point.offset_s > duration + 1e-6 for point in self.f0_contour):
            raise ValueError("F0 point lies outside the mora")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class UtteranceProsody:
    """Teacher-derived features for one utterance."""

    utterance_id: str
    text: str
    sample_rate: int
    moras: tuple[MoraFeatures, ...]
    speaker_id: str | None = None
    teacher_id: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id must not be empty")
        if not self.text:
            raise ValueError("text must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        previous_end = 0.0
        for mora in self.moras:
            if mora.start_s + 1e-6 < previous_end:
                raise ValueError("moras must be ordered and non-overlapping")
            previous_end = mora.end_s

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["moras"] = [
            {
                **{k: v for k, v in asdict(mora).items() if k != "f0_contour"},
                "f0_contour": [asdict(point) for point in mora.f0_contour],
            }
            for mora in self.moras
        ]
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UtteranceProsody:
        moras = tuple(
            MoraFeatures(
                text=mora["text"],
                start_s=float(mora["start_s"]),
                end_s=float(mora["end_s"]),
                consonant=mora.get("consonant"),
                vowel=mora.get("vowel"),
                consonant_duration_s=float(mora.get("consonant_duration_s", 0.0)),
                vowel_duration_s=float(mora.get("vowel_duration_s", 0.0)),
                energy_dbfs=(
                    None if mora.get("energy_dbfs") is None else float(mora["energy_dbfs"])
                ),
                pause_after_s=float(mora.get("pause_after_s", 0.0)),
                f0_contour=tuple(
                    F0Point(offset_s=float(point["offset_s"]), hz=float(point["hz"]))
                    for point in mora.get("f0_contour", [])
                ),
            )
            for mora in data["moras"]
        )
        return cls(
            utterance_id=str(data["utterance_id"]),
            text=str(data["text"]),
            sample_rate=int(data["sample_rate"]),
            moras=moras,
            speaker_id=data.get("speaker_id"),
            teacher_id=data.get("teacher_id"),
            schema_version=int(data.get("schema_version", 1)),
        )

    @classmethod
    def from_json(cls, payload: str) -> UtteranceProsody:
        return cls.from_dict(json.loads(payload))

    @classmethod
    def load(cls, path: str | Path) -> UtteranceProsody:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")
