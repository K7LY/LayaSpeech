"""Acoustic feature extraction for already-aligned Japanese moras."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

from .schema import F0Point, MoraFeatures, UtteranceProsody


@dataclass(frozen=True, slots=True)
class AlignedMora:
    """Time-aligned mora boundary produced by a forced aligner."""

    text: str
    start_s: float
    end_s: float
    consonant: str | None = None
    vowel: str | None = None
    consonant_duration_s: float = 0.0
    vowel_duration_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("aligned mora text must not be empty")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("aligned mora timing must satisfy 0 <= start_s < end_s")
        if self.consonant_duration_s < 0 or self.vowel_duration_s < 0:
            raise ValueError("phoneme durations must be >= 0")


def _energy_dbfs(samples: "object") -> float:
    """Return RMS energy in dBFS for a normalized floating-point signal."""

    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    return 20.0 * math.log10(max(rms, 1e-6))


def extract_world_features(
    *,
    utterance_id: str,
    text: str,
    audio_path: str | Path,
    aligned_moras: Iterable[AlignedMora],
    speaker_id: str | None = None,
    teacher_id: str | None = None,
    frame_period_ms: float = 5.0,
) -> UtteranceProsody:
    """Extract F0 and energy with WORLD using externally supplied mora timings."""

    try:
        import numpy as np
        import pyworld as pw
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Acoustic analysis dependencies are not installed. "
            "Install with: uv sync --extra analysis"
        ) from exc

    audio, sample_rate = sf.read(str(audio_path), dtype="float64", always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError("audio must be mono or stereo")
    if sample_rate < 16000:
        raise ValueError("WORLD analysis requires audio sampled at 16 kHz or above")

    coarse_f0, time_axis = pw.dio(audio, sample_rate, frame_period=frame_period_ms)
    f0 = pw.stonemask(audio, coarse_f0, time_axis, sample_rate)

    boundaries = tuple(aligned_moras)
    features: list[MoraFeatures] = []
    for index, mora in enumerate(boundaries):
        start_sample = max(0, int(round(mora.start_s * sample_rate)))
        end_sample = min(len(audio), int(round(mora.end_s * sample_rate)))
        if end_sample <= start_sample:
            raise ValueError(f"empty audio span for mora {index}: {mora.text}")

        mask = (time_axis >= mora.start_s) & (time_axis <= mora.end_s) & (f0 > 0)
        contour = tuple(
            F0Point(offset_s=float(t - mora.start_s), hz=float(hz))
            for t, hz in zip(time_axis[mask], f0[mask], strict=True)
        )
        pause_after = (
            max(0.0, boundaries[index + 1].start_s - mora.end_s)
            if index + 1 < len(boundaries)
            else 0.0
        )

        features.append(
            MoraFeatures(
                text=mora.text,
                start_s=mora.start_s,
                end_s=mora.end_s,
                consonant=mora.consonant,
                vowel=mora.vowel,
                consonant_duration_s=mora.consonant_duration_s,
                vowel_duration_s=mora.vowel_duration_s,
                energy_dbfs=_energy_dbfs(audio[start_sample:end_sample]),
                pause_after_s=pause_after,
                f0_contour=contour,
            )
        )

    return UtteranceProsody(
        utterance_id=utterance_id,
        text=text,
        sample_rate=int(sample_rate),
        moras=tuple(features),
        speaker_id=speaker_id,
        teacher_id=teacher_id,
    )
