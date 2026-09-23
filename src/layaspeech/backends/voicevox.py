"""VOICEVOX Engine adapter for the first real-time synthesis PoC.

The adapter talks to an already-running VOICEVOX-compatible HTTP engine. No
VOICEVOX model weights or engine code are bundled with LayaSpeech.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..adapters import SynthesizerAdapter


_PITCH_SEMITONES = {
    -3: -5.0,
    -2: -3.0,
    -1: -1.25,
    0: 0.0,
    1: 1.25,
    2: 3.0,
    3: 5.0,
}

_DURATION_MULTIPLIER = {
    -3: 0.60,
    -2: 0.78,
    -1: 0.90,
    0: 1.00,
    1: 1.10,
    2: 1.28,
    3: 1.55,
}

_PAUSE_SECONDS = {
    "none": 0.0,
    "short": 0.08,
    "medium": 0.20,
    "long": 0.40,
}


def _flatten_moras(query: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], bool]]:
    flattened: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    for phrase in query.get("accent_phrases", []):
        moras = phrase.get("moras", [])
        for index, mora in enumerate(moras):
            flattened.append((phrase, mora, index == len(moras) - 1))
    return flattened


def apply_voicevox_controls(
    query: dict[str, Any],
    controls: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the subset of LayaSpeech controls exposed by normal AudioQuery.

    Supported:
    - relative mora pitch
    - consonant/vowel duration
    - pause duration when VOICEVOX already has a phrase-boundary pause mora

    Not directly representable by normal speech AudioQuery:
    - within-mora pitch direction
    - per-mora energy

    Those controls remain in the common schema for richer future backends.
    """

    result = deepcopy(query)
    moras = _flatten_moras(result)
    controls = list(controls)
    if len(moras) != len(controls):
        raise ValueError(
            "VOICEVOX/LayaSpeech mora count mismatch: "
            f"query has {len(moras)}, controls have {len(controls)}"
        )

    for (phrase, mora, is_phrase_end), control in zip(moras, controls, strict=True):
        pitch_level = int(control["pitch_level"])
        duration_level = int(control["duration_level"])

        pitch = float(mora.get("pitch", 0.0) or 0.0)
        if pitch > 0:
            semitones = _PITCH_SEMITONES[pitch_level]
            mora["pitch"] = pitch * (2.0 ** (semitones / 12.0))

        duration_multiplier = _DURATION_MULTIPLIER[duration_level]
        if mora.get("consonant_length") is not None:
            mora["consonant_length"] = max(
                0.01,
                float(mora["consonant_length"]) * duration_multiplier,
            )
        mora["vowel_length"] = max(
            0.01,
            float(mora["vowel_length"]) * duration_multiplier,
        )

        if is_phrase_end:
            pause_class = str(control.get("pause_after", "none"))
            pause_mora = phrase.get("pause_mora")
            if pause_mora is not None and pause_class in _PAUSE_SECONDS:
                pause_mora["vowel_length"] = _PAUSE_SECONDS[pause_class]

    return result


@dataclass(slots=True)
class VoicevoxSynthesizer(SynthesizerAdapter):
    """HTTP adapter for a local VOICEVOX-compatible engine."""

    speaker: int
    base_url: str = "http://127.0.0.1:50021"
    segment_length: float = 0.3
    timeout_s: float = 60.0
    read_size: int = 65536

    def _url(self, path: str, **params: Any) -> str:
        return (
            self.base_url.rstrip("/")
            + path
            + ("?" + urlencode(params) if params else "")
        )

    def _post_json(
        self,
        path: str,
        *,
        params: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = Request(
            self._url(path, **params),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"VOICEVOX request failed: {exc}") from exc

    def audio_query(self, text: str) -> dict[str, Any]:
        return self._post_json(
            "/audio_query",
            params={"text": text, "speaker": self.speaker},
        )

    def synthesize_stream(
        self,
        *,
        text: str,
        prosody: Iterable[dict[str, Any]],
    ) -> Iterator[bytes]:
        query = apply_voicevox_controls(self.audio_query(text), prosody)
        request = Request(
            self._url(
                "/streaming_synthesis",
                speaker=self.speaker,
                segment_length=self.segment_length,
            ),
            data=json.dumps(query).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            response = urlopen(request, timeout=self.timeout_s)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"VOICEVOX streaming synthesis failed: {exc}") from exc

        try:
            while True:
                chunk = response.read(self.read_size)
                if not chunk:
                    break
                yield chunk
        finally:
            response.close()
