from layaspeech.cli import build_parser


def test_train_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["train", "train.jsonl", "--output-dir", "checkpoint"]
    )
    assert args.command == "train"
    assert args.subfolder == "multilingual"
    assert args.micro_batch == 2


def test_predict_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["predict", "こんにちは", "--checkpoint", "checkpoint"]
    )
    assert args.command == "predict"
    assert args.chunk_size == 4
