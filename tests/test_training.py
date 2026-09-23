import sys
from types import ModuleType, SimpleNamespace

import pytest

from layaspeech.training import (
    TrainingConfig,
    _build_items,
    _diagnostic_snapshot,
    _supervised_cross_entropy,
    _target_for_question,
    combine_training_loss,
)


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


def test_training_config_validates_loss_controls_and_steps() -> None:
    config = TrainingConfig(
        loss_mode="ce",
        control_names=("duration_level",),
        diagnostic_steps=(0, 1, 10),
    )
    assert config.loss_mode == "ce"

    with pytest.raises(ValueError, match="loss_mode"):
        TrainingConfig(loss_mode="mystery")
    with pytest.raises(ValueError, match="control"):
        TrainingConfig(control_names=("duration",))
    with pytest.raises(ValueError, match="diagnostic"):
        TrainingConfig(diagnostic_steps=(-1,))


def test_build_items_filters_exact_control_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    common = ModuleType("laya.common")
    common.QTYPES = {"choice": 0, "score": 1, "noul": 2}
    common.render_options = lambda question: list(question["crit"])
    common.build_sequence = lambda *_args, **_kwargs: ([1, 2, 3], [1, 2])
    laya = ModuleType("laya")
    laya.common = common
    monkeypatch.setitem(sys.modules, "laya", laya)
    monkeypatch.setitem(sys.modules, "laya.common", common)
    row = {
        "state": {},
        "questions": {
            "m0000_duration_level": {
                "type": "score",
                "instructions": "duration",
                "criteria": ["short", "long"],
            },
            "m0000_pitch_level": {
                "type": "score",
                "instructions": "pitch",
                "criteria": ["low", "high"],
            },
        },
        "gold": {
            "m0000_duration_level": {"probabilities": {"0": 1.0, "1": 0.0}},
            "m0000_pitch_level": {"probabilities": {"0": 0.0, "1": 1.0}},
        },
    }
    agent = SimpleNamespace(tok=object(), cfg={})

    items = _build_items([row], agent, control_names=("duration_level",))
    assert [item["qid"] for item in items] == ["m0000_duration_level"]
    with pytest.raises(ValueError, match="no valid"):
        _build_items([row], agent, control_names=("energy_level",))


def test_ce_only_loss_ignores_rlcd_term() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])
    ce = _supervised_cross_entropy(logits, target, mask)

    assert combine_training_loss(ce, torch.tensor(99.0), "ce") is ce
    assert torch.isclose(combine_training_loss(ce, torch.tensor(2.0), "rlcd_ce"), ce + 2.0)


def test_diagnostic_snapshot_reports_gradients_logits_and_entropy() -> None:
    torch = pytest.importorskip("torch")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.scorer = torch.nn.Linear(2, 2)

        def forward(self, values):
            return self.scorer(self.encoder(values))

    model = TinyModel()
    logits = model(torch.tensor([[1.0, -1.0]]))
    logits.sum().backward()
    snapshot = _diagnostic_snapshot(model, logits, torch.tensor([[True, True]]))

    assert snapshot["gradient_norm"] > 0
    assert snapshot["encoder_gradient_norm"] > 0
    assert snapshot["decision_head_gradient_norm"] > 0
    assert snapshot["logits_std"] >= 0
    assert snapshot["softmax_entropy"] >= 0
    assert len(snapshot["predicted_top1_class"]) == 1
