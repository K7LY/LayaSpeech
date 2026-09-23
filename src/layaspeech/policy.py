"""Thin Laya adapter for low-latency prosody inference.

Laya is intentionally isolated here so dataset generation and acoustic analysis
remain usable without torch/transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_LEVELS = [
    "very low",
    "low",
    "slightly low",
    "neutral",
    "slightly high",
    "high",
    "very high",
]

_DURATION_LEVELS = [
    "much shorter",
    "shorter",
    "slightly shorter",
    "normal",
    "slightly longer",
    "longer",
    "much longer",
]


def _questions_for_mora(index: int, mora: str) -> dict[str, dict[str, Any]]:
    prefix = f"m{index:04d}"
    focus = f"mora index {index} ({mora!r})"
    return {
        f"{prefix}_pitch_level": {
            "type": "score",
            "instructions": f"Relative pitch of {focus} versus this speaker's baseline.",
            "criteria": _LEVELS,
        },
        f"{prefix}_pitch_direction": {
            "type": "choice",
            "instructions": f"Pitch movement across {focus}.",
            "criteria": {
                "falling": "pitch falls",
                "level": "pitch stays approximately level",
                "rising": "pitch rises",
            },
        },
        f"{prefix}_duration_level": {
            "type": "score",
            "instructions": f"Relative duration of {focus} versus normal speech.",
            "criteria": _DURATION_LEVELS,
        },
        f"{prefix}_energy_level": {
            "type": "score",
            "instructions": f"Relative vocal energy of {focus} versus normal speech.",
            "criteria": _LEVELS,
        },
        f"{prefix}_pause_after": {
            "type": "choice",
            "instructions": f"Pause immediately after {focus}.",
            "criteria": {
                "none": "no meaningful pause",
                "short": "brief pause",
                "medium": "clear phrase pause",
                "long": "long pause or sentence boundary",
            },
        },
    }


def build_chunk_request(
    *,
    text: str,
    moras: list[str] | tuple[str, ...],
    start_index: int,
    end_index: int,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one Laya forward-pass request for several consecutive moras."""

    if not 0 <= start_index < end_index <= len(moras):
        raise ValueError("chunk must satisfy 0 <= start_index < end_index <= len(moras)")

    state = {
        "utterance": text,
        "moras": list(moras),
        "chunk_start": start_index,
        "chunk_end": end_index,
        "previous_context": previous_text or "",
        "next_context": next_text or "",
    }

    questions: dict[str, Any] = {}
    for index in range(start_index, end_index):
        questions.update(_questions_for_mora(index, moras[index]))
    return state, questions


def build_mora_request(
    *,
    text: str,
    moras: list[str] | tuple[str, ...],
    mora_index: int,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper returning the five unprefixed questions for one mora."""

    state, prefixed = build_chunk_request(
        text=text,
        moras=moras,
        start_index=mora_index,
        end_index=mora_index + 1,
        previous_text=previous_text,
        next_text=next_text,
    )
    prefix = f"m{mora_index:04d}_"
    questions = {key.removeprefix(prefix): value for key, value in prefixed.items()}
    state["focus_index"] = mora_index
    state["focus_mora"] = moras[mora_index]
    return state, questions


def _score_to_level(answer: dict[str, Any]) -> int:
    score = float(answer["score"])
    return max(-3, min(3, round(score) - 3))


def _parse_one(answers: dict[str, Any], index: int, mora: str) -> dict[str, Any]:
    prefix = f"m{index:04d}"
    return {
        "mora_index": index,
        "text": mora,
        "pitch_level": _score_to_level(answers[f"{prefix}_pitch_level"]),
        "pitch_direction": answers[f"{prefix}_pitch_direction"]["choice"],
        "duration_level": _score_to_level(answers[f"{prefix}_duration_level"]),
        "energy_level": _score_to_level(answers[f"{prefix}_energy_level"]),
        "pause_after": answers[f"{prefix}_pause_after"]["choice"],
    }


@dataclass(slots=True)
class LayaProsodyPolicy:
    """Inference wrapper around a Laya checkpoint."""

    model_id: str = "convaiinnovations/laya"
    subfolder: str | None = "multilingual"
    device: str | None = None
    agent: Any | None = None

    def __post_init__(self) -> None:
        if self.agent is None:
            try:
                import laya
            except ImportError as exc:
                raise RuntimeError(
                    "Laya is not installed. Install with: uv sync --extra laya"
                ) from exc
            self.agent = laya.load(
                self.model_id,
                device=self.device,
                subfolder=self.subfolder,
            )

    def predict_chunk(
        self,
        *,
        text: str,
        moras: list[str] | tuple[str, ...],
        start_index: int,
        end_index: int,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict several moras in one Laya forward pass."""

        state, questions = build_chunk_request(
            text=text,
            moras=moras,
            start_index=start_index,
            end_index=end_index,
            previous_text=previous_text,
            next_text=next_text,
        )
        result = self.agent.predict(state, questions)
        answers = result["answers"]
        return [
            _parse_one(answers, index, moras[index])
            for index in range(start_index, end_index)
        ]

    def predict_utterance(
        self,
        *,
        text: str,
        moras: list[str] | tuple[str, ...],
        chunk_size: int = 4,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict a full utterance using bounded multi-mora forward passes."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        predictions: list[dict[str, Any]] = []
        for start in range(0, len(moras), chunk_size):
            end = min(len(moras), start + chunk_size)
            predictions.extend(
                self.predict_chunk(
                    text=text,
                    moras=moras,
                    start_index=start,
                    end_index=end,
                    previous_text=previous_text,
                    next_text=next_text,
                )
            )
        return predictions

    def predict_mora(
        self,
        *,
        text: str,
        moras: list[str] | tuple[str, ...],
        mora_index: int,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> dict[str, Any]:
        """Predict one mora; mainly useful for debugging."""

        return self.predict_chunk(
            text=text,
            moras=moras,
            start_index=mora_index,
            end_index=mora_index + 1,
            previous_text=previous_text,
            next_text=next_text,
        )[0]
