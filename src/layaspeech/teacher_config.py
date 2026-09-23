"""Small JSON configuration loader for external teacher commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .teachers import CommandTeacherAdapter


def load_command_teacher(path: str | Path) -> tuple[CommandTeacherAdapter, str | None]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    command = payload.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError("teacher config 'command' must be an array of strings")

    cwd = payload.get("cwd")
    env = payload.get("env")
    if env is not None and (
        not isinstance(env, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
    ):
        raise ValueError("teacher config 'env' must map strings to strings")

    teacher = CommandTeacherAdapter(
        command=tuple(command),
        cwd=None if cwd is None else Path(cwd),
        env=env,
        timeout_s=float(payload.get("timeout_s", 300.0)),
    )
    teacher_id = payload.get("teacher_id")
    return teacher, None if teacher_id is None else str(teacher_id)
