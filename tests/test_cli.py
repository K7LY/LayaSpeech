from layaspeech.cli import build_parser


def test_analyze_cli_parses_required_inputs() -> None:
    args = build_parser().parse_args(
        ["analyze", "こんにちは", "voice.wav", "-o", "out.json"]
    )
    assert args.command == "analyze"
    assert args.text == "こんにちは"
    assert args.mfa_executable == "mfa"


def test_moras_cli_parses_text() -> None:
    args = build_parser().parse_args(["moras", "こんにちは"])
    assert args.command == "moras"
    assert args.text == "こんにちは"
