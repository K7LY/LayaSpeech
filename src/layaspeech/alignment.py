"""Japanese mora alignment using OpenJTalk reading + Montreal Forced Aligner."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters import AcousticAnalyzer
from .analysis import AlignedMora, extract_world_features
from .schema import UtteranceProsody

_SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
_COMBINING_MARKS = frozenset("゙゚")
_SKIP_CHARS = frozenset(
    " \t\r\n、。！？!?・，,.：:；;（）()［］[]「」『』【】〈〉《》“”\"'…‥"
)
_VOWEL_PHONES = frozenset(
    {
        "a",
        "aː",
        "e",
        "eː",
        "i",
        "iː",
        "i̥",
        "o",
        "oː",
        "ɨ",
        "ɨː",
        "ɨ̥",
        "ɯ",
        "ɯː",
        "ɯ̥",
    }
)


def split_katakana_moras(reading: str) -> tuple[str, ...]:
    """Split a Katakana reading into orthographic mora tokens."""

    moras: list[str] = []
    for char in reading:
        if char in _SKIP_CHARS:
            continue
        if char in _SMALL_KANA or char in _COMBINING_MARKS:
            if not moras:
                raise ValueError(f"small/combining kana {char!r} has no base kana")
            moras[-1] += char
            continue
        moras.append(char)

    if not moras:
        raise ValueError("no mora could be extracted from the reading")
    return tuple(moras)


def text_to_katakana_moras(text: str) -> tuple[str, ...]:
    """Use OpenJTalk to obtain the spoken Katakana reading, then split it."""

    try:
        import pyopenjtalk
    except ImportError as exc:
        raise RuntimeError(
            "Japanese frontend dependencies are not installed. "
            "Install with: uv sync --extra alignment"
        ) from exc

    reading = pyopenjtalk.g2p(text, kana=True)
    return split_katakana_moras(reading)


def _tier_entries(payload: dict[str, Any], suffix: str) -> list[list[Any]]:
    tiers = payload.get("tiers", {})
    for name, tier in tiers.items():
        normalized = name.strip().lower()
        if normalized == suffix or normalized.endswith(f" - {suffix}"):
            return list(tier.get("entries", []))
    raise ValueError(f"MFA JSON does not contain a {suffix!r} tier")


def _is_silence(label: str) -> bool:
    return label.strip().lower() in {"", "sil", "sp", "spn", "<eps>", "<unk>"}


def parse_mfa_json(
    payload: dict[str, Any],
    moras: tuple[str, ...],
) -> tuple[AlignedMora, ...]:
    """Convert MFA JSON word/phone tiers into aligned mora boundaries.

    The transcript passed to MFA contains one Katakana mora per whitespace-
    separated word, so the word tier is already the required mora tier.
    """

    word_entries = [
        entry
        for entry in _tier_entries(payload, "words")
        if len(entry) >= 3 and not _is_silence(str(entry[2]))
    ]
    phone_entries = [
        entry
        for entry in _tier_entries(payload, "phones")
        if len(entry) >= 3 and not _is_silence(str(entry[2]))
    ]

    if len(word_entries) != len(moras):
        labels = [str(entry[2]) for entry in word_entries]
        raise ValueError(
            "MFA mora count mismatch: "
            f"expected {len(moras)} {moras!r}, got {len(word_entries)} {labels!r}"
        )

    aligned: list[AlignedMora] = []
    for index, (mora, word) in enumerate(zip(moras, word_entries, strict=True)):
        start, end = float(word[0]), float(word[1])
        if end <= start:
            raise ValueError(f"invalid MFA word interval for mora {index}: {word!r}")

        phones = []
        for phone in phone_entries:
            p_start, p_end, label = float(phone[0]), float(phone[1]), str(phone[2])
            midpoint = (p_start + p_end) / 2.0
            if start - 1e-4 <= midpoint <= end + 1e-4:
                phones.append((p_start, p_end, label))

        vowel_duration = sum(
            p_end - p_start
            for p_start, p_end, label in phones
            if label in _VOWEL_PHONES
        )
        consonant_duration = sum(
            p_end - p_start
            for p_start, p_end, label in phones
            if label not in _VOWEL_PHONES
        )

        vowel_labels = [label for _, _, label in phones if label in _VOWEL_PHONES]
        consonant_labels = [label for _, _, label in phones if label not in _VOWEL_PHONES]

        aligned.append(
            AlignedMora(
                text=mora,
                start_s=start,
                end_s=end,
                consonant="+".join(consonant_labels) or None,
                vowel="+".join(vowel_labels) or None,
                consonant_duration_s=consonant_duration,
                vowel_duration_s=vowel_duration,
            )
        )

    return tuple(aligned)


@dataclass(slots=True)
class MfaMoraAligner:
    """Run MFA on a single file with one Katakana mora per transcript token."""

    executable: str = "mfa"
    dictionary_model: str = "japanese_mfa"
    acoustic_model: str = "japanese_mfa"
    g2p_model: str = "japanese_katakana_mfa"
    beam: int = 100
    timeout_s: float = 180.0

    def align(self, text: str, audio_path: str | Path) -> tuple[AlignedMora, ...]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        moras = text_to_katakana_moras(text)
        unsupported = [mora for mora in moras if mora in {"ッ", "ー"}]
        if unsupported:
            raise NotImplementedError(
                "The initial MFA mora-token PoC does not yet validate standalone "
                "sokuon/prolonged-sound-mark alignment: "
                + ", ".join(unsupported)
            )

        with tempfile.TemporaryDirectory(prefix="layaspeech-mfa-") as temporary:
            temporary_path = Path(temporary)
            transcript_path = temporary_path / "utterance.txt"
            output_path = temporary_path / "alignment.json"
            transcript_path.write_text(" ".join(moras) + "\n", encoding="utf-8")

            command = [
                self.executable,
                "align_one",
                str(audio_path),
                str(transcript_path),
                self.dictionary_model,
                self.acoustic_model,
                str(output_path),
                "--output_format",
                "json",
                "--no_tokenization",
                "--g2p_model_path",
                self.g2p_model,
                "--beam",
                str(self.beam),
                "--clean",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "MFA executable was not found. Install/activate Montreal Forced Aligner "
                    "or set MfaMoraAligner(executable=...)."
                ) from exc

            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(f"MFA alignment failed ({completed.returncode}): {details}")
            if not output_path.exists():
                raise RuntimeError("MFA completed without producing alignment JSON")

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            return parse_mfa_json(payload, moras)


@dataclass(slots=True)
class MfaWorldAnalyzer(AcousticAnalyzer):
    """End-to-end PoC analyzer: text + WAV -> MFA boundaries -> WORLD features."""

    aligner: MfaMoraAligner

    def analyze(
        self,
        *,
        utterance_id: str,
        text: str,
        audio_path: Path,
        teacher_id: str | None = None,
        speaker_id: str | None = None,
    ) -> UtteranceProsody:
        aligned = self.aligner.align(text, audio_path)
        return extract_world_features(
            utterance_id=utterance_id,
            text=text,
            audio_path=audio_path,
            aligned_moras=aligned,
            teacher_id=teacher_id,
            speaker_id=speaker_id,
        )
