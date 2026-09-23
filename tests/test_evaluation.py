import pytest

from layaspeech.evaluation import evaluate_control_records


def test_evaluation_reports_control_metrics_and_train_only_majority() -> None:
    gold = [
        {
            "pitch_level": -3,
            "pitch_direction": "falling",
            "duration_level": -1,
            "energy_level": 0,
            "pause_after": "none",
        },
        {
            "pitch_level": 3,
            "pitch_direction": "rising",
            "duration_level": 2,
            "energy_level": 1,
            "pause_after": "short",
        },
    ]
    predictions = [
        {
            "pitch_level": -2,
            "pitch_direction": "falling",
            "duration_level": -1,
            "energy_level": 1,
            "pause_after": "none",
        },
        {
            "pitch_level": 3,
            "pitch_direction": "level",
            "duration_level": 0,
            "energy_level": 1,
            "pause_after": "none",
        },
    ]
    majority_source = [
        {
            "pitch_level": 1,
            "pitch_direction": "level",
            "duration_level": 2,
            "energy_level": -1,
            "pause_after": "none",
        },
        {
            "pitch_level": 1,
            "pitch_direction": "level",
            "duration_level": 2,
            "energy_level": -1,
            "pause_after": "none",
        },
        {
            "pitch_level": 0,
            "pitch_direction": "rising",
            "duration_level": 0,
            "energy_level": 2,
            "pause_after": "long",
        },
    ]

    result = evaluate_control_records(gold, predictions, majority_source)

    assert result["count"] == 2
    assert result["overall_accuracy"] == pytest.approx(0.5)
    pitch = result["controls"]["pitch_level"]
    assert pitch["accuracy"] == 0.5
    assert pitch["ordinal_mae"] == 0.5
    assert pitch["majority_value"] == 1
    assert pitch["majority_accuracy"] == 0.0
    assert pitch["gold_distribution"] == {"-3": 1, "3": 1}
    assert pitch["prediction_distribution"] == {"-2": 1, "3": 1}
    assert pitch["confusion"] == {"-3": {"-2": 1}, "3": {"3": 1}}
    direction = result["controls"]["pitch_direction"]
    assert direction["ordinal_mae"] is None
    assert direction["majority_value"] == "level"
    assert direction["majority_accuracy"] == 0.0


def test_evaluation_rejects_misaligned_records() -> None:
    record = {
        "pitch_level": 0,
        "pitch_direction": "level",
        "duration_level": 0,
        "energy_level": 0,
        "pause_after": "none",
    }
    with pytest.raises(ValueError, match="same non-zero length"):
        evaluate_control_records([record], [], [record])
