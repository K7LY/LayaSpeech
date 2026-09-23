$ErrorActionPreference = "Stop"

uv sync --extra dev
uv run pytest
uv run python -m compileall -q src tests
uv run ruff check src tests

Write-Host "LayaSpeech local checks passed."
