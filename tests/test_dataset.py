import json

from layaspeech.dataset import build_training_case, export_jsonl, to_upstream_row
from layaspeech.schema import F0Point, MoraFeatures, UtteranceProsody
from layaspeech.targets import ProsodyBaseline, derive_targets


def sample() -> UtteranceProsody:
    return UtteranceProsody(
        utterance_id="u1",
        text="あ",
        sample_rate=24000,
        moras=(
            MoraFeatures(
                text="あ",
                start_s=0.0,
                end_s=0.1,
                vowel="a",
                vowel_duration_s=0.1,
                energy_dbfs=-18.0,
                pause_after_s=0.0,
                f0_contour=(F0Point(0.05, 200.0),),
            ),
        ),
    )


def test_training_case_matches_laya_dataset_shape() -> None:
    utterance = sample()
    baseline = ProsodyBaseline.from_utterances([utterance])
    target = derive_targets(utterance, baseline)[0]
    case = build_training_case(utterance, target)
    row = to_upstream_row(case)

    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])

    assert state["focus_mora"] == "あ"
    assert set(questions) == set(gold)
    assert gold["pitch_level"]["probabilities"]["3"] == 1.0


def test_export_jsonl_writes_one_row_per_mora(tmp_path) -> None:
    output = tmp_path / "train.jsonl"
    count = export_jsonl([sample()], output)
    rows = output.read_text(encoding="utf-8").splitlines()

    assert count == 1
    assert len(rows) == 1
    assert set(json.loads(rows[0])) == {"state", "questions", "gold"}
