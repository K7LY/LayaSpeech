from layaspeech.cli import build_parser


def test_train_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["train", "train.jsonl", "--output-dir", "checkpoint"]
    )
    assert args.command == "train"
    assert args.subfolder == "multilingual"
    assert args.micro_batch == 2
    assert args.loss_mode == "rlcd_ce"
    assert args.control == []
    assert args.diagnostic_step is None


def test_train_cli_accepts_diagnostic_controls() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "train.jsonl",
            "--output-dir",
            "checkpoint",
            "--loss-mode",
            "ce",
            "--control",
            "duration_level",
            "--diagnostic-step",
            "25",
        ]
    )
    assert args.loss_mode == "ce"
    assert args.control == ["duration_level"]
    assert args.diagnostic_step == [25]


def test_predict_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["predict", "こんにちは", "--checkpoint", "checkpoint"]
    )
    assert args.command == "predict"
    assert args.chunk_size == 4
