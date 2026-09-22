"""Export analyzed prosody as Laya-compatible fine-tuning rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .policy import build_mora_request
from .schema import UtteranceProsody
from .targets import MoraTarget, ProsodyBaseline, derive_targets


def _one_hot(keys: list[str], selected: str) -> dict[str, float]:
    if selected not in keys:
        raise ValueError(f"selected value {selected!r} is not in {keys!r}")
    return {key: 1.0 if key == selected else 0.0 for key in keys}


def build_training_case(
    utterance: UtteranceProsody,
    target: MoraTarget,
) -> dict[str, Any]:
    """Build one focused-mora case using Laya's state/questions/gold convention."""

    moras = tuple(mora.text for mora in utterance.moras)
    state, questions = build_mora_request(
        text=utterance.text,
        moras=moras,
        mora_index=target.mora_index,
    )

    gold = {
        "pitch_level": {
            "probabilities": _one_hot(
                [str(i) for i in range(7)],
                str(target.pitch_level + 3),
            )
        },
        "pitch_direction": {
            "probabilities": _one_hot(
                list(questions["pitch_direction"]["criteria"]),
                target.pitch_direction,
            )
        },
        "duration_level": {
            "probabilities": _one_hot(
                [str(i) for i in range(7)],
                str(target.duration_level + 3),
            )
        },
        "energy_level": {
            "probabilities": _one_hot(
                [str(i) for i in range(7)],
                str(target.energy_level + 3),
            )
        },
        "pause_after": {
            "probabilities": _one_hot(
                list(questions["pause_after"]["criteria"]),
                target.pause_after,
            )
        },
    }
    return {
        "utterance_id": utterance.utterance_id,
        "mora_index": target.mora_index,
        "state": state,
        "questions": questions,
        "gold": gold,
    }


def to_upstream_row(case: dict[str, Any]) -> dict[str, str]:
    """Serialize fields like the dataset consumed by Laya's fine-tune notebook."""

    return {
        "state": json.dumps(case["state"], ensure_ascii=False),
        "questions": json.dumps(case["questions"], ensure_ascii=False),
        "gold": json.dumps(case["gold"], ensure_ascii=False),
    }


def export_jsonl(
    utterances: Iterable[UtteranceProsody],
    output_path: str | Path,
) -> int:
    """Export one Laya training case per mora using a corpus-wide baseline."""

    records = tuple(utterances)
    if not records:
        raise ValueError("at least one utterance is required")

    baseline = ProsodyBaseline.from_utterances(records)
    count = 0
    with Path(output_path).open("w", encoding="utf-8", newline="\n") as handle:
        for utterance in records:
            for target in derive_targets(utterance, baseline):
                row = to_upstream_row(build_training_case(utterance, target))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count
