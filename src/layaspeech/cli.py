"""Small dependency-free CLI for inspecting and exporting PoC datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import export_jsonl
from .schema import UtteranceProsody
from .targets import ProsodyBaseline, derive_targets


def _targets(path: Path) -> int:
    utterance = UtteranceProsody.load(path)
    baseline = ProsodyBaseline.from_utterances([utterance])
    targets = derive_targets(utterance, baseline)
    payload = [
        {
            "mora_index": target.mora_index,
            "text": target.text,
            "pitch_level": target.pitch_level,
            "pitch_direction": target.pitch_direction,
            "duration_level": target.duration_level,
            "energy_level": target.energy_level,
            "pause_after": target.pause_after,
        }
        for target in targets
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _export(inputs: list[Path], output: Path) -> int:
    utterances = [UtteranceProsody.load(path) for path in inputs]
    count = export_jsonl(utterances, output)
    print(f"exported {count} mora training cases to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="layaspeech")
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser(
        "targets",
        help="quantize one analyzed utterance into PoC Laya targets",
    )
    targets.add_argument("input", type=Path)

    export = subparsers.add_parser(
        "export-jsonl",
        help="export analyzed utterances to Laya-compatible state/questions/gold JSONL",
    )
    export.add_argument("inputs", type=Path, nargs="+")
    export.add_argument("-o", "--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "targets":
        return _targets(args.input)
    if args.command == "export-jsonl":
        return _export(args.inputs, args.output)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
