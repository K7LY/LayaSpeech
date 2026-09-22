from layaspeech.policy import build_mora_request
from layaspeech.schema import F0Point, MoraFeatures, UtteranceProsody
from layaspeech.targets import ProsodyBaseline, derive_targets


def sample() -> UtteranceProsody:
    return UtteranceProsody(
        utterance_id="sample-1",
        text="そう",
        sample_rate=24000,
        teacher_id="test",
        moras=(
            MoraFeatures(
                text="そ",
                start_s=0.0,
                end_s=0.10,
                consonant="s",
                vowel="o",
                consonant_duration_s=0.03,
                vowel_duration_s=0.07,
                energy_dbfs=-20.0,
                pause_after_s=0.0,
                f0_contour=(
                    F0Point(0.04, 180.0),
                    F0Point(0.07, 190.0),
                ),
            ),
            MoraFeatures(
                text="う",
                start_s=0.10,
                end_s=0.24,
                vowel="u",
                vowel_duration_s=0.14,
                energy_dbfs=-17.0,
                pause_after_s=0.35,
                f0_contour=(
                    F0Point(0.02, 200.0),
                    F0Point(0.10, 230.0),
                ),
            ),
        ),
    )


def test_schema_roundtrip() -> None:
    utterance = sample()
    restored = UtteranceProsody.from_json(utterance.to_json())
    assert restored == utterance


def test_targets_are_bounded_and_pause_is_detected() -> None:
    utterance = sample()
    baseline = ProsodyBaseline.from_utterances([utterance])
    targets = derive_targets(utterance, baseline)

    assert len(targets) == 2
    assert all(-3 <= t.pitch_level <= 3 for t in targets)
    assert all(-3 <= t.duration_level <= 3 for t in targets)
    assert all(-3 <= t.energy_level <= 3 for t in targets)
    assert targets[1].pause_after == "long"
    assert targets[1].pitch_direction == "rising"


def test_laya_request_batches_five_controls() -> None:
    state, questions = build_mora_request(
        text="そう",
        moras=("そ", "う"),
        mora_index=1,
    )
    assert state["focus_mora"] == "う"
    assert set(questions) == {
        "pitch_level",
        "pitch_direction",
        "duration_level",
        "energy_level",
        "pause_after",
    }
