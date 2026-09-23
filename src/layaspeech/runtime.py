"""Streaming-oriented runtime planning.

This module does not generate waveforms. It prepares bounded prosody chunks so a
synthesizer backend can render the next chunk while the current one is playing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .alignment import text_to_katakana_moras
from .policy import LayaProsodyPolicy


@dataclass(frozen=True, slots=True)
class ProsodyChunk:
    start_index: int
    end_index: int
    moras: tuple[str, ...]
    controls: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class StreamingProsodyPlanner:
    policy: LayaProsodyPolicy
    chunk_size: int = 4

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

    def plan(
        self,
        text: str,
        *,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> Iterator[ProsodyChunk]:
        """Yield predicted chunks incrementally instead of waiting for the full sentence."""

        moras = text_to_katakana_moras(text)
        for start in range(0, len(moras), self.chunk_size):
            end = min(len(moras), start + self.chunk_size)
            controls = self.policy.predict_chunk(
                text=text,
                moras=moras,
                start_index=start,
                end_index=end,
                previous_text=previous_text,
                next_text=next_text,
            )
            yield ProsodyChunk(
                start_index=start,
                end_index=end,
                moras=moras[start:end],
                controls=tuple(controls),
            )
