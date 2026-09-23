from layaspeech.policy import build_chunk_request, build_mora_request


def test_chunk_request_batches_multiple_moras() -> None:
    state, questions = build_chunk_request(
        text="こんにちは",
        moras=("コ", "ン", "ニ", "チ", "ワ"),
        start_index=0,
        end_index=4,
    )
    assert state["chunk_start"] == 0
    assert state["chunk_end"] == 4
    assert len(questions) == 20
    assert "m0003_pause_after" in questions


def test_single_mora_compatibility_request() -> None:
    state, questions = build_mora_request(
        text="そう",
        moras=("ソ", "ウ"),
        mora_index=1,
    )
    assert state["focus_mora"] == "ウ"
    assert set(questions) == {
        "pitch_level",
        "pitch_direction",
        "duration_level",
        "energy_level",
        "pause_after",
    }
