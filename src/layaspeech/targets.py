"""Convert high-resolution teacher features into compact Laya targets."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median
from typing import Literal

from .schema import MoraFeatures, UtteranceProsody

PitchDirection = Literal["falling", "level", "rising"]
PauseClass = Literal["none", "short", "medium", "long"]


@dataclass(frozen=True, slots=True)
class ProsodyBaseline:
    """Speaker/teacher reference values used for relative targets."""

    f0_hz: float
    mora_duration_s: float
    energy_dbfs: float

    @classmethod
    def from_utterances(cls, utterances: Iterable[UtteranceProsody]) -> ProsodyBaseline:
        f0: list[float] = []
        duration: list[float] = []
        energy: list[float] = []

        for utterance in utterances:
            for mora in utterance.moras:
                f0.extend(point.hz for point in mora.f0_contour)
                duration.append(mora.duration_s)
                if mora.energy_dbfs is not None:
                    energy.append(mora.energy_dbfs)

        if not f0:
            raise ValueError("cannot build baseline without voiced F0 samples")
        if not duration:
            raise ValueError("cannot build baseline without moras")
        if not energy:
            raise ValueError("cannot build baseline without energy samples")

        return cls(
            f0_hz=float(median(f0)),
            mora_duration_s=float(median(duration)),
            energy_dbfs=float(median(energy)),
        )


@dataclass(frozen=True, slots=True)
class MoraTarget:
    """Discrete System-1 controls predicted by Laya for one mora."""

    mora_index: int
    text: str
    pitch_level: int
    pitch_direction: PitchDirection
    duration_level: int
    energy_level: int
    pause_after: PauseClass

    def __post_init__(self) -> None:
        for name in ("pitch_level", "duration_level", "energy_level"):
            value = getattr(self, name)
            if not -3 <= value <= 3:
                raise ValueError(f"{name} must be in [-3, 3]")


def _seven_level(value: float, thresholds: tuple[float, ...]) -> int:
    if len(thresholds) != 6:
        raise ValueError("seven-level quantization needs six thresholds")
    return bisect_right(thresholds, value) - 3


def _representative_f0(mora: MoraFeatures) -> float | None:
    values = [point.hz for point in mora.f0_contour if point.hz > 0]
    return float(median(values)) if values else None


def _pitch_direction(mora: MoraFeatures, threshold_semitones: float = 0.75) -> PitchDirection:
    values = [point.hz for point in mora.f0_contour if point.hz > 0]
    if len(values) < 2:
        return "level"

    split = max(1, len(values) // 3)
    start = median(values[:split])
    end = median(values[-split:])
    delta = 12.0 * math.log2(end / start)

    if delta >= threshold_semitones:
        return "rising"
    if delta <= -threshold_semitones:
        return "falling"
    return "level"


def _pause_class(seconds: float) -> PauseClass:
    if seconds < 0.03:
        return "none"
    if seconds < 0.12:
        return "short"
    if seconds < 0.30:
        return "medium"
    return "long"


def derive_targets(
    utterance: UtteranceProsody,
    baseline: ProsodyBaseline,
) -> tuple[MoraTarget, ...]:
    """Quantize teacher acoustics into the first LayaSpeech PoC target space."""

    if baseline.f0_hz <= 0 or baseline.mora_duration_s <= 0:
        raise ValueError("baseline F0 and duration must be > 0")

    targets: list[MoraTarget] = []
    for index, mora in enumerate(utterance.moras):
        representative_f0 = _representative_f0(mora)
        if representative_f0 is None:
            pitch_delta_st = 0.0
        else:
            pitch_delta_st = 12.0 * math.log2(representative_f0 / baseline.f0_hz)

        duration_ratio = mora.duration_s / baseline.mora_duration_s
        energy_delta_db = (
            0.0 if mora.energy_dbfs is None else mora.energy_dbfs - baseline.energy_dbfs
        )

        targets.append(
            MoraTarget(
                mora_index=index,
                text=mora.text,
                pitch_level=_seven_level(
                    pitch_delta_st,
                    (-4.0, -2.0, -0.75, 0.75, 2.0, 4.0),
                ),
                pitch_direction=_pitch_direction(mora),
                duration_level=_seven_level(
                    duration_ratio,
                    (0.70, 0.85, 0.95, 1.05, 1.18, 1.40),
                ),
                energy_level=_seven_level(
                    energy_delta_db,
                    (-6.0, -3.0, -1.0, 1.0, 3.0, 6.0),
                ),
                pause_after=_pause_class(mora.pause_after_s),
            )
        )

    return tuple(targets)
