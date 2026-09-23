import json

from layaspeech.dataset import build_training_chunk_case, export_jsonl, to_upstream_row
from layaspeech.schema import F0Point, MoraFeatures, UtteranceProsody
from layaspeech.targets import ProsodyBaseline, derive_targets


def sample() -> UtteranceProsody:
    return UtteranceProsody(
        utterance_id="u1",
        text="あい",
        sample_rate=24000,
        moras=(
            MoraFeatures(
                text="あ",
                start_s=0.0,
                end_s=0.1,
                vowel="a",
                vowel_duration_s=0.1,
                energy_dbfs=-18.0,
                f0_contour=(F0Point(0.05, 200.0),),
            ),
            MoraFeatures(
                text="い",
                start_s=0.1,
                end_s=0.2,
                vowel="i",
                vowel_duration_s=0.1,
                energy_dbfs=-17.0,
                f0_contour=(F0Point(0.05, 210.0),),
            ),
        ),
    )


def test_training_case_matches_laya_dataset_shape() -> None:
    utterance = sample()
    baseline = ProsodyBaseline.from_utterances([utterance])
    targets = derive_targets(utterance, baseline)
    case = build_training_chunk_case(utterance, targets, start_index=0, end_index=2)
    row = to_upstream_row(case)

    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])

    assert state["chunk_start"] == 0
    assert state["chunk_end"] == 2
    assert len(questions) == 10
    assert set(questions) == set(gold)


def test_export_jsonl_chunks_moras(tmp_path) -> None:
    output = tmp_path / "train.jsonl"
    count = export_jsonl([sample()], output, chunk_size=1)
    rows = output.read_text(encoding="utf-8").splitlines()

    assert count == 2
    assert len(rows) == 2
    assert set(json.loads(rows[0])) == {"state", "questions", "gold"}
