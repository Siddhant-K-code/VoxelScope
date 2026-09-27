# SPDX-License-Identifier: Apache-2.0
.PHONY: sync test lint typecheck demo clean
sync:
	uv sync --frozen

test:
	uv run pytest

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

demo:
	uv run python -c "from pathlib import Path; import shutil; p=Path('build/demo'); shutil.rmtree(p) if p.exists() else None"
	uv run voxelscope fixture build --output build/demo
	uv run voxelscope verify --bundle build/demo

clean:
	uv run python -c "from pathlib import Path; import shutil; p=Path('build'); shutil.rmtree(p) if p.exists() else None"
