import pytest

from layaspeech.teachers import render_teacher_command


def test_render_teacher_command() -> None:
    command = render_teacher_command(
        ("python", "teacher.py", "--text", "{text}", "--output", "{output}"),
        text="こんにちは",
        output_path="out.wav",
    )
    assert command[:4] == ["python", "teacher.py", "--text", "こんにちは"]
    assert command[-2] == "--output"
    assert command[-1].endswith("out.wav")


def test_render_teacher_command_rejects_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="unknown teacher command placeholder"):
        render_teacher_command(("{voice}",), text="x", output_path="out.wav")
