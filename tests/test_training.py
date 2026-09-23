import pytest

from layaspeech.training import TrainingConfig, _target_for_question


def test_score_target_is_normalized() -> None:
    q = {"type": "score", "criteria": ["a", "b", "c"]}
    gold = {"probabilities": {"0": 0.0, "1": 2.0, "2": 0.0}}
    assert _target_for_question(q, gold) == [0.0, 1.0, 0.0]


def test_choice_target_uses_criteria_order() -> None:
    q = {"type": "choice", "criteria": {"x": "X", "y": "Y"}}
    gold = {"probabilities": {"y": 1.0}}
    assert _target_for_question(q, gold) == [0.0, 1.0]


def test_training_config_rejects_invalid_group_size() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(group_size=1)
