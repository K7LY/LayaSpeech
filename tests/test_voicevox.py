import pytest

from layaspeech.backends.voicevox import apply_voicevox_controls


def query() -> dict:
    return {
        "accent_phrases": [
            {
                "moras": [
                    {
                        "text": "カ",
                        "consonant": "k",
                        "consonant_length": 0.04,
                        "vowel": "a",
                        "vowel_length": 0.08,
                        "pitch": 200.0,
                    },
                    {
                        "text": "ナ",
                        "consonant": "n",
                        "consonant_length": 0.04,
                        "vowel": "a",
                        "vowel_length": 0.08,
                        "pitch": 210.0,
                    },
                ],
                "accent": 1,
                "pause_mora": {
                    "text": "、",
                    "vowel": "pau",
                    "vowel_length": 0.15,
                    "pitch": 0.0,
                },
            }
        ],
        "speedScale": 1.0,
        "pitchScale": 0.0,
        "intonationScale": 1.0,
        "volumeScale": 1.0,
        "prePhonemeLength": 0.1,
        "postPhonemeLength": 0.1,
        "outputSamplingRate": 24000,
        "outputStereo": False,
    }


def controls() -> list[dict]:
    return [
        {
            "pitch_level": 1,
            "pitch_direction": "rising",
            "duration_level": -1,
            "energy_level": 0,
            "pause_after": "none",
        },
        {
            "pitch_level": -1,
            "pitch_direction": "falling",
            "duration_level": 2,
            "energy_level": 2,
            "pause_after": "long",
        },
    ]


def test_apply_voicevox_controls_changes_pitch_duration_and_existing_pause() -> None:
    original = query()
    modified = apply_voicevox_controls(original, controls())
    moras = modified["accent_phrases"][0]["moras"]

    assert moras[0]["pitch"] > original["accent_phrases"][0]["moras"][0]["pitch"]
    assert moras[0]["vowel_length"] < 0.08
    assert moras[1]["pitch"] < 210.0
    assert moras[1]["vowel_length"] > 0.08
    assert modified["accent_phrases"][0]["pause_mora"]["vowel_length"] == 0.40
    assert original["accent_phrases"][0]["pause_mora"]["vowel_length"] == 0.15


def test_apply_voicevox_controls_rejects_mora_count_mismatch() -> None:
    with pytest.raises(ValueError, match="mora count mismatch"):
        apply_voicevox_controls(query(), controls()[:1])
