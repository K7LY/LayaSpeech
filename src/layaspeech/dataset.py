"""Export analyzed prosody as Laya-compatible fine-tuning rows."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from .diagnostics import level_to_score_index
from .policy import build_chunk_request
from .schema import UtteranceProsody
from .targets import MoraTarget, ProsodyBaseline, derive_targets


def _one_hot(keys: list[str], selected: str) -> dict[str, float]:
    if selected not in keys:
        raise ValueError(f"selected value {selected!r} is not in {keys!r}")
    return {key: 1.0 if key == selected else 0.0 for key in keys}


def _gold_for_target(
    target: MoraTarget,
    questions: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    prefix = f"m{target.mora_index:04d}"
    return {
        f"{prefix}_pitch_level": {
            "probabilities": _one_hot(
                [str(i) for i in range(7)],
                str(level_to_score_index(target.pitch_level)),
            )
        },
        f"{prefix}_pitch_direction": {
            "probabilities": _one_hot(
                list(questions[f"{prefix}_pitch_direction"]["criteria"]),
                target.pitch_direction,
            )
        },
        f"{prefix}_duration_level": {
            "probabilities": _one_hot(
                [str(i) for i in range(7)],
                str(level_to_score_index(target.duration_level)),
            )
        },
        f"{prefix}_energy_level": {
            "probabilities": _one_hot(
                [str(i) for i in range(7)],
                str(level_to_score_index(target.energy_level)),
            )
        },
        f"{prefix}_pause_after": {
            "probabilities": _one_hot(
                list(questions[f"{prefix}_pause_after"]["criteria"]),
                target.pause_after,
            )
        },
    }


def build_training_chunk_case(
    utterance: UtteranceProsody,
    targets: tuple[MoraTarget, ...],
    *,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    """Build one multi-mora case matching the runtime request structure."""

    moras = tuple(mora.text for mora in utterance.moras)
    state, questions = build_chunk_request(
        text=utterance.text,
        moras=moras,
        start_index=start_index,
        end_index=end_index,
    )
    gold: dict[str, Any] = {}
    for target in targets[start_index:end_index]:
        gold.update(_gold_for_target(target, questions))

    return {
        "utterance_id": utterance.utterance_id,
        "start_index": start_index,
        "end_index": end_index,
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
    *,
    chunk_size: int = 4,
    pitch_reference: Literal["speaker", "utterance"] = "speaker",
) -> int:
    """Export chunked Laya training cases using a corpus-wide baseline."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    records = tuple(utterances)
    if not records:
        raise ValueError("at least one utterance is required")

    baseline = ProsodyBaseline.from_utterances(records)
    count = 0
    with Path(output_path).open("w", encoding="utf-8", newline="\n") as handle:
        for utterance in records:
            targets = derive_targets(
                utterance,
                baseline,
                pitch_reference=pitch_reference,
            )
            for start in range(0, len(targets), chunk_size):
                end = min(len(targets), start + chunk_size)
                row = to_upstream_row(
                    build_training_chunk_case(
                        utterance,
                        targets,
                        start_index=start,
                        end_index=end,
                    )
                )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count
