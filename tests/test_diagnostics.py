import sys
from types import ModuleType, SimpleNamespace

import pytest

from layaspeech.diagnostics import (
    audit_training_rows,
    level_to_score_index,
    score_index_to_level,
)


def test_score_level_indices_round_trip() -> None:
    assert level_to_score_index(-3) == 0
    assert level_to_score_index(0) == 3
    assert level_to_score_index(3) == 6

    assert score_index_to_level(0) == -3
    assert score_index_to_level(3) == 0
    assert score_index_to_level(6) == 3


def test_audit_preserves_question_and_choice_order(monkeypatch: pytest.MonkeyPatch) -> None:
    common = ModuleType("laya.common")
    common.QTYPES = {"choice": 0, "score": 1, "noul": 2}
    common.render_options = lambda question: list(question["crit"])
    common.build_sequence = lambda *_args, **_kwargs: ([101, 103, 103, 102], [1, 2])
    laya = ModuleType("laya")
    laya.common = common
    monkeypatch.setitem(sys.modules, "laya", laya)
    monkeypatch.setitem(sys.modules, "laya.common", common)

    rows = [
        {
            "state": {"focus": "ア"},
            "questions": {
                "m0000_direction": {
                    "type": "choice",
                    "instructions": "direction",
                    "criteria": {"falling": "down", "level": "flat"},
                },
                "m0000_pause": {
                    "type": "choice",
                    "instructions": "pause",
                    "criteria": {"none": "no", "short": "yes"},
                },
            },
            "gold": {
                "m0000_direction": {"probabilities": {"falling": 0.0, "level": 1.0}},
                "m0000_pause": {"probabilities": {"none": 1.0, "short": 0.0}},
            },
        }
    ]

    records = audit_training_rows(
        rows,
        SimpleNamespace(tok=object(), cfg={"max_len": 32, "head_max_len": 16}),
    )

    assert [record["qid"] for record in records] == ["m0000_direction", "m0000_pause"]
    assert list(records[0]["criteria"]) == ["falling", "level"]
    assert records[0]["label"] == 1
    assert records[0]["marker_count"] == records[0]["expected_option_count"] == 2


def test_audit_rejects_marker_option_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    common = ModuleType("laya.common")
    common.QTYPES = {"choice": 0, "score": 1, "noul": 2}
    common.render_options = lambda question: list(question["crit"])
    common.build_sequence = lambda *_args, **_kwargs: ([101, 103, 102], [1])
    laya = ModuleType("laya")
    laya.common = common
    monkeypatch.setitem(sys.modules, "laya", laya)
    monkeypatch.setitem(sys.modules, "laya.common", common)

    rows = [
        {
            "state": {},
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"a": "A", "b": "B"},
                }
            },
            "gold": {"q": {"probabilities": {"a": 1.0, "b": 0.0}}},
        }
    ]

    with pytest.raises(ValueError, match="marker/option mismatch"):
        audit_training_rows(rows, SimpleNamespace(tok=object(), cfg={}))
