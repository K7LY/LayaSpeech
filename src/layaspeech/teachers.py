"""Generic teacher adapters.

Third-party TTS packages stay outside LayaSpeech. The command adapter lets users
connect any CLI-capable teacher without adding its weights or dependencies here.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .adapters import TeacherAdapter, TeacherResult


def render_teacher_command(
    command: Sequence[str],
    *,
    text: str,
    output_path: str | Path,
) -> list[str]:
    """Expand the two supported placeholders without invoking a shell."""

    values = {"text": text, "output": str(Path(output_path).resolve())}
    try:
        return [part.format_map(values) for part in command]
    except KeyError as exc:
        raise ValueError(
            f"unknown teacher command placeholder: {exc.args[0]!r}; "
            "supported placeholders are {text} and {output}"
        ) from exc


@dataclass(slots=True)
class CommandTeacherAdapter(TeacherAdapter):
    """Run an external teacher TTS command.

    Example command:
        ("python", "teacher.py", "--text", "{text}", "--output", "{output}")
    """

    command: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    timeout_s: float = 300.0

    def synthesize(self, text: str, output_path: Path) -> TeacherResult:
        if not self.command:
            raise ValueError("teacher command must not be empty")

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = render_teacher_command(
            self.command,
            text=text,
            output_path=output_path,
        )

        environment = os.environ.copy()
        if self.env:
            environment.update(self.env)

        completed = subprocess.run(
            rendered,
            cwd=self.cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"teacher command failed ({completed.returncode}): {details}"
            )
        if not output_path.exists():
            raise RuntimeError(
                "teacher command completed successfully but did not create "
                f"{output_path}"
            )

        return TeacherResult(
            audio_path=output_path,
            metadata={
                "command": rendered,
                "stdout": completed.stdout.strip(),
            },
        )
