"""Streaming-oriented runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .adapters import SynthesizerAdapter
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
        """Yield predicted chunks instead of building one huge decision request."""

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


@dataclass(slots=True)
class StreamingSpeechRuntime:
    """Connect a fine-tuned Laya policy to a streaming synthesizer backend."""

    planner: StreamingProsodyPlanner
    synthesizer: SynthesizerAdapter

    def synthesize_stream(
        self,
        text: str,
        *,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> Iterator[bytes]:
        controls: list[dict[str, Any]] = []
        for chunk in self.planner.plan(
            text,
            previous_text=previous_text,
            next_text=next_text,
        ):
            controls.extend(chunk.controls)

        yield from self.synthesizer.synthesize_stream(
            text=text,
            prosody=controls,
        )
