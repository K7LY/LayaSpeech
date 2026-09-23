"""Dependency-light metrics for semantic prosody control predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

CONTROLS = (
    "pitch_level",
    "pitch_direction",
    "duration_level",
    "energy_level",
    "pause_after",
)
ORDINAL_CONTROLS = frozenset({"pitch_level", "duration_level", "energy_level"})


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def evaluate_control_records(
    gold: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    majority_source: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compare aligned mora controls and derive baselines from training data only."""

    gold_records = tuple(gold)
    predicted_records = tuple(predictions)
    majority_records = tuple(majority_source)
    if (
        not gold_records
        or len(gold_records) != len(predicted_records)
        or not majority_records
    ):
        raise ValueError("gold/predictions must have the same non-zero length and majority source")

    controls: dict[str, Any] = {}
    total_correct = 0
    for control in CONTROLS:
        gold_values = [record[control] for record in gold_records]
        predicted_values = [record[control] for record in predicted_records]
        majority_value = Counter(record[control] for record in majority_records).most_common(1)[0][0]
        correct = sum(
            expected == actual
            for expected, actual in zip(gold_values, predicted_values, strict=True)
        )
        total_correct += correct
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for expected, actual in zip(gold_values, predicted_values, strict=True):
            confusion[str(expected)][str(actual)] += 1

        ordinal_mae = None
        if control in ORDINAL_CONTROLS:
            ordinal_mae = sum(
                abs(int(expected) - int(actual))
                for expected, actual in zip(gold_values, predicted_values, strict=True)
            ) / len(gold_values)

        controls[control] = {
            "accuracy": correct / len(gold_values),
            "majority_value": majority_value,
            "majority_accuracy": sum(value == majority_value for value in gold_values)
            / len(gold_values),
            "ordinal_mae": ordinal_mae,
            "gold_distribution": _distribution(gold_values),
            "prediction_distribution": _distribution(predicted_values),
            "confusion": {
                expected: dict(sorted(actual.items()))
                for expected, actual in sorted(confusion.items())
            },
        }

    return {
        "count": len(gold_records),
        "overall_accuracy": total_correct / (len(gold_records) * len(CONTROLS)),
        "controls": controls,
    }
