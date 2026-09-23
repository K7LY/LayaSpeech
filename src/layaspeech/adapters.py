"""Interfaces separating third-party teachers and synthesizers from LayaSpeech."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import UtteranceProsody


@dataclass(frozen=True, slots=True)
class TeacherResult:
    audio_path: Path
    metadata: dict[str, Any] | None = None


class TeacherAdapter(ABC):
    """Offline teacher TTS interface."""

    @abstractmethod
    def synthesize(self, text: str, output_path: Path) -> TeacherResult:
        raise NotImplementedError


class AcousticAnalyzer(ABC):
    """Convert teacher audio + known text into the common prosody schema."""

    @abstractmethod
    def analyze(
        self,
        *,
        utterance_id: str,
        text: str,
        audio_path: Path,
        teacher_id: str | None = None,
        speaker_id: str | None = None,
    ) -> UtteranceProsody:
        raise NotImplementedError


class SynthesizerAdapter(ABC):
    """Runtime backend receiving predicted prosody controls."""

    @abstractmethod
    def synthesize_stream(
        self,
        *,
        text: str,
        prosody: Iterable[dict[str, Any]],
    ) -> Iterable[bytes]:
        """Yield encoded or PCM audio chunks."""
        raise NotImplementedError
