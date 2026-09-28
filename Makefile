# SPDX-License-Identifier: Apache-2.0
.PHONY: sync test lint typecheck custody-validate demo build validate clean
sync:
	uv sync --frozen

test:
	uv run pytest

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

custody-validate:
	uv run voxelscope source verify --registry research/source-registry-v1.json
	uv run voxelscope source one-volume --record research/one-volume-source-decision-v1.json --plan research/one-volume-acquisition-plan-v1.json
	uv run voxelscope custody plan --registry research/source-registry-v1.json --plan research/acquisition-plan-v1.json
	uv run voxelscope milestone4-public verify --bundle research/milestone-4
	uv run voxelscope custody scan-public --root .

demo:
	uv run python -c "from pathlib import Path; import shutil; p=Path('build/demo'); shutil.rmtree(p) if p.exists() else None"
	uv run voxelscope fixture build --output build/demo
	uv run voxelscope verify --bundle build/demo

build:
	uv build --offline

validate: lint typecheck test custody-validate demo build

clean:
	uv run python -c "from pathlib import Path; import shutil; p=Path('build'); shutil.rmtree(p) if p.exists() else None"
