import math

import pytest

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


def test_pitch_levels_reserve_endpoints_for_large_deviations() -> None:
    baseline = ProsodyBaseline(f0_hz=100.0, mora_duration_s=0.1, energy_dbfs=-20.0)

    def pitch_level(semitones: float) -> int:
        hz = 100.0 * math.pow(2.0, semitones / 12.0)
        utterance = UtteranceProsody(
            utterance_id=f"pitch-{semitones}",
            text="カ",
            sample_rate=24_000,
            moras=(
                MoraFeatures(
                    text="カ",
                    start_s=0.0,
                    end_s=0.1,
                    vowel="a",
                    vowel_duration_s=0.1,
                    energy_dbfs=-20.0,
                    f0_contour=(F0Point(0.05, hz),),
                ),
            ),
        )
        return derive_targets(utterance, baseline)[0].pitch_level

    assert pitch_level(-13.0) == -3
    assert pitch_level(-5.0) == -1
    assert pitch_level(0.0) == 0
    assert pitch_level(5.0) == 1
    assert pitch_level(13.0) == 3


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


def test_utterance_pitch_reference_removes_voice_register() -> None:
    def utterance(identifier: str, scale: float) -> UtteranceProsody:
        return UtteranceProsody(
            utterance_id=identifier,
            text="あい",
            sample_rate=24_000,
            moras=(
                MoraFeatures(
                    text="あ",
                    start_s=0.0,
                    end_s=0.1,
                    vowel="a",
                    vowel_duration_s=0.1,
                    energy_dbfs=-20.0,
                    f0_contour=(F0Point(0.05, 100.0 * scale),),
                ),
                MoraFeatures(
                    text="い",
                    start_s=0.1,
                    end_s=0.2,
                    vowel="i",
                    vowel_duration_s=0.1,
                    energy_dbfs=-20.0,
                    f0_contour=(F0Point(0.05, 200.0 * scale),),
                ),
            ),
        )

    low = utterance("low", 1.0)
    high = utterance("high", 4.0)
    baseline = ProsodyBaseline.from_utterances([low, high])

    low_targets = derive_targets(low, baseline, pitch_reference="utterance")
    high_targets = derive_targets(high, baseline, pitch_reference="utterance")

    assert [target.pitch_level for target in low_targets] == [
        target.pitch_level for target in high_targets
    ]


def test_pitch_reference_rejects_unknown_mode() -> None:
    utterance = sample()
    baseline = ProsodyBaseline.from_utterances([utterance])
    with pytest.raises(ValueError, match="pitch_reference"):
        derive_targets(utterance, baseline, pitch_reference="mystery")
