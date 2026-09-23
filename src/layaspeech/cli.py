"""CLI for the LayaSpeech proof of concept."""

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


def _export(inputs: list[Path], output: Path, chunk_size: int) -> int:
    utterances = [UtteranceProsody.load(path) for path in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = export_jsonl(utterances, output, chunk_size=chunk_size)
    print(f"exported {count} chunk training cases to {output}")
    return 0


def _moras(text: str) -> int:
    from .alignment import text_to_katakana_moras

    print(json.dumps(text_to_katakana_moras(text), ensure_ascii=False))
    return 0


def _analyze(
    *,
    text: str,
    audio: Path,
    output: Path,
    utterance_id: str,
    teacher_id: str | None,
    speaker_id: str | None,
    mfa_executable: str,
) -> int:
    from .alignment import MfaMoraAligner, MfaWorldAnalyzer

    analyzer = MfaWorldAnalyzer(MfaMoraAligner(executable=mfa_executable))
    utterance = analyzer.analyze(
        utterance_id=utterance_id,
        text=text,
        audio_path=audio,
        teacher_id=teacher_id,
        speaker_id=speaker_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    utterance.save(output)
    print(f"wrote {len(utterance.moras)} aligned moras to {output}")
    return 0


def _build_corpus(
    *,
    manifest: Path,
    output_dir: Path,
    teacher_config: Path,
    chunk_size: int,
    mfa_executable: str,
    no_resume: bool,
) -> int:
    from .alignment import MfaMoraAligner, MfaWorldAnalyzer
    from .corpus import build_corpus, load_manifest
    from .teacher_config import load_command_teacher

    teacher, teacher_id = load_command_teacher(teacher_config)
    result = build_corpus(
        items=load_manifest(manifest),
        teacher=teacher,
        analyzer=MfaWorldAnalyzer(MfaMoraAligner(executable=mfa_executable)),
        output_dir=output_dir,
        teacher_id=teacher_id,
        chunk_size=chunk_size,
        resume=not no_resume,
    )
    print(
        f"analyzed {len(result.analyzed)} utterances; "
        f"training data: {result.training_jsonl}"
    )
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
        help="export analyzed utterances to chunked Laya fine-tuning JSONL",
    )
    export.add_argument("inputs", type=Path, nargs="+")
    export.add_argument("-o", "--output", type=Path, required=True)
    export.add_argument("--chunk-size", type=int, default=4)

    moras = subparsers.add_parser(
        "moras",
        help="show OpenJTalk Katakana mora segmentation for Japanese text",
    )
    moras.add_argument("text")

    analyze = subparsers.add_parser(
        "analyze",
        help="run MFA + WORLD on one known-text WAV and write UtteranceProsody JSON",
    )
    analyze.add_argument("text")
    analyze.add_argument("audio", type=Path)
    analyze.add_argument("-o", "--output", type=Path, required=True)
    analyze.add_argument("--utterance-id", default="utterance-001")
    analyze.add_argument("--teacher-id")
    analyze.add_argument("--speaker-id")
    analyze.add_argument("--mfa-executable", default="mfa")

    corpus = subparsers.add_parser(
        "build-corpus",
        help="teacher TTS -> MFA/WORLD analysis -> Laya training JSONL",
    )
    corpus.add_argument("manifest", type=Path)
    corpus.add_argument("--teacher-config", type=Path, required=True)
    corpus.add_argument("--output-dir", type=Path, required=True)
    corpus.add_argument("--chunk-size", type=int, default=4)
    corpus.add_argument("--mfa-executable", default="mfa")
    corpus.add_argument("--no-resume", action="store_true")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "targets":
        return _targets(args.input)
    if args.command == "export-jsonl":
        return _export(args.inputs, args.output, args.chunk_size)
    if args.command == "moras":
        return _moras(args.text)
    if args.command == "analyze":
        return _analyze(
            text=args.text,
            audio=args.audio,
            output=args.output,
            utterance_id=args.utterance_id,
            teacher_id=args.teacher_id,
            speaker_id=args.speaker_id,
            mfa_executable=args.mfa_executable,
        )
    if args.command == "build-corpus":
        return _build_corpus(
            manifest=args.manifest,
            output_dir=args.output_dir,
            teacher_config=args.teacher_config,
            chunk_size=args.chunk_size,
            mfa_executable=args.mfa_executable,
            no_resume=args.no_resume,
        )
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
