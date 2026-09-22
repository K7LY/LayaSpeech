"""Thin Laya adapter for PoC inference.

Laya is intentionally isolated here so the schema and dataset pipeline remain
usable without installing torch/transformers.
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


def build_mora_request(
    *,
    text: str,
    moras: list[str] | tuple[str, ...],
    mora_index: int,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one batched Laya request for a focused mora."""

    if not 0 <= mora_index < len(moras):
        raise IndexError("mora_index out of range")

    state = {
        "utterance": text,
        "moras": list(moras),
        "focus_index": mora_index,
        "focus_mora": moras[mora_index],
        "previous_context": previous_text or "",
        "next_context": next_text or "",
    }
    questions = {
        "pitch_level": {
            "type": "score",
            "instructions": "Relative pitch of the focused mora versus this speaker's baseline.",
            "criteria": _LEVELS,
        },
        "pitch_direction": {
            "type": "choice",
            "instructions": "Pitch movement across the focused mora.",
            "criteria": {
                "falling": "pitch falls",
                "level": "pitch stays approximately level",
                "rising": "pitch rises",
            },
        },
        "duration_level": {
            "type": "score",
            "instructions": "Relative duration of the focused mora versus normal speech.",
            "criteria": [
                "much shorter",
                "shorter",
                "slightly shorter",
                "normal",
                "slightly longer",
                "longer",
                "much longer",
            ],
        },
        "energy_level": {
            "type": "score",
            "instructions": "Relative vocal energy of the focused mora versus normal speech.",
            "criteria": _LEVELS,
        },
        "pause_after": {
            "type": "choice",
            "instructions": "Pause immediately after the focused mora.",
            "criteria": {
                "none": "no meaningful pause",
                "short": "brief pause",
                "medium": "clear phrase pause",
                "long": "long pause or sentence boundary",
            },
        },
    }
    return state, questions


def _score_to_level(answer: dict[str, Any]) -> int:
    score = float(answer["score"])
    return max(-3, min(3, round(score) - 3))


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

    def predict_mora(
        self,
        *,
        text: str,
        moras: list[str] | tuple[str, ...],
        mora_index: int,
        previous_text: str | None = None,
        next_text: str | None = None,
    ) -> dict[str, Any]:
        state, questions = build_mora_request(
            text=text,
            moras=moras,
            mora_index=mora_index,
            previous_text=previous_text,
            next_text=next_text,
        )
        result = self.agent.predict(state, questions)
        answers = result["answers"]
        return {
            "pitch_level": _score_to_level(answers["pitch_level"]),
            "pitch_direction": answers["pitch_direction"]["choice"],
            "duration_level": _score_to_level(answers["duration_level"]),
            "energy_level": _score_to_level(answers["energy_level"]),
            "pause_after": answers["pause_after"]["choice"],
            "raw": result,
        }
