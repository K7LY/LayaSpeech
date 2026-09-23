$ErrorActionPreference = "Stop"

uv sync --extra dev --extra analysis
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run python -m compileall -q src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "LayaSpeech local checks passed."
