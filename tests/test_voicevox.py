from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from layaspeech.backends.voicevox import VoicevoxSynthesizer, apply_voicevox_controls


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


def test_synthesizer_falls_back_when_streaming_endpoint_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> BytesIO:
        del timeout
        url = str(request.full_url)
        requested_urls.append(url)
        if "/streaming_synthesis" in url:
            raise HTTPError(url, 404, "not found", {}, None)
        return BytesIO(b"RIFF-compatible-wave")

    monkeypatch.setattr("layaspeech.backends.voicevox.urlopen", fake_urlopen)
    monkeypatch.setattr(VoicevoxSynthesizer, "audio_query", lambda self, text: query())

    synthesizer = VoicevoxSynthesizer(speaker=8)
    audio = b"".join(synthesizer.synthesize_stream(text="カナ", prosody=controls()))

    assert audio == b"RIFF-compatible-wave"
    assert "/streaming_synthesis" in requested_urls[0]
    assert "/synthesis" in requested_urls[1]
