from layaspeech.cli import build_parser


def test_synthesize_cli() -> None:
    args = build_parser().parse_args(
        [
            "synthesize",
            "こんにちは",
            "--checkpoint",
            "checkpoint",
            "--output",
            "out.wav",
            "--speaker",
            "1",
        ]
    )
    assert args.command == "synthesize"
    assert args.voicevox_url == "http://127.0.0.1:50021"
    assert args.segment_length == 0.3
