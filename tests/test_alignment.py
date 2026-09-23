from pathlib import Path
from subprocess import CompletedProcess

import pytest

from layaspeech.alignment import MfaMoraAligner, parse_mfa_json, split_katakana_moras


def test_split_katakana_moras() -> None:
    assert split_katakana_moras("コンニチワ") == ("コ", "ン", "ニ", "チ", "ワ")
    assert split_katakana_moras("キャット") == ("キャ", "ッ", "ト")
    assert split_katakana_moras("スーパー") == ("ス", "ー", "パ", "ー")


def test_parse_mfa_json_uses_word_tier_as_moras() -> None:
    payload = {
        "tiers": {
            "words": {
                "type": "interval",
                "entries": [
                    [0.00, 0.10, "カ"],
                    [0.10, 0.22, "ナ"],
                ],
            },
            "phones": {
                "type": "interval",
                "entries": [
                    [0.00, 0.04, "k"],
                    [0.04, 0.10, "a"],
                    [0.10, 0.15, "n"],
                    [0.15, 0.22, "a"],
                ],
            },
        }
    }

    result = parse_mfa_json(payload, ("カ", "ナ"))

    assert result[0].text == "カ"
    assert result[0].consonant == "k"
    assert result[0].vowel == "a"
    assert result[0].consonant_duration_s == pytest.approx(0.04)
    assert result[0].vowel_duration_s == pytest.approx(0.06)
    assert result[1].start_s == 0.10


def test_parse_mfa_json_rejects_count_mismatch() -> None:
    payload = {
        "tiers": {
            "words": {"entries": [[0.0, 0.1, "カ"]]},
            "phones": {"entries": [[0.0, 0.1, "a"]]},
        }
    }
    with pytest.raises(ValueError, match="count mismatch"):
        parse_mfa_json(payload, ("カ", "ナ"))


def test_mfa_aligner_uses_robust_default_beam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "sample.wav"
    audio.touch()
    observed_command: list[str] = []

    def fake_run(command: list[str], **_: object) -> CompletedProcess[str]:
        observed_command.extend(command)
        output_path = Path(command[6])
        output_path.write_text(
            '{"tiers":{"words":{"entries":[[0,0.1,"\u30ab"]]},'
            '"phones":{"entries":[[0,0.1,"a"]]}}}',
            encoding="utf-8",
        )
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("layaspeech.alignment.subprocess.run", fake_run)
    monkeypatch.setattr(
        "layaspeech.alignment.text_to_katakana_moras", lambda text: ("カ",)
    )

    MfaMoraAligner().align("カ", audio)

    beam_index = observed_command.index("--beam")
    assert observed_command[beam_index + 1] == "100"
