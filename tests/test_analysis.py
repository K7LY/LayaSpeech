from layaspeech.analysis import AlignedMora, _energy_dbfs


def test_aligned_mora_validation() -> None:
    mora = AlignedMora(
        text="か",
        start_s=0.1,
        end_s=0.2,
        consonant="k",
        vowel="a",
        consonant_duration_s=0.04,
        vowel_duration_s=0.06,
    )
    assert mora.end_s - mora.start_s == 0.1


def test_energy_dbfs_for_half_scale_signal() -> None:
    value = _energy_dbfs([0.5, -0.5, 0.5, -0.5])
    assert -6.1 < value < -6.0
