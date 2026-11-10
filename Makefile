# Dev mode: `make all SAMPLE=1` (2026 only). Full: `make all`.
SAMPLE ?=
FLAG := $(if $(SAMPLE),--sample,)
RUN := uv run absengine

.PHONY: setup data runvalue overturn value policy backtest players export serve web web-dev test lint all

setup:
	uv sync --all-extras

data:
	$(RUN) data $(FLAG)

runvalue:
	$(RUN) runvalue $(FLAG)

overturn:  # also builds ABS-zone geometry features
	$(RUN) overturn $(FLAG)

value:
	$(RUN) value $(FLAG)

policy:
	$(RUN) policy $(FLAG)

backtest:
	$(RUN) backtest $(FLAG)

players:
	$(RUN) players $(FLAG)

export:
	$(RUN) export-web $(FLAG)

serve:
	uv run uvicorn absengine.api.main:app --host 0.0.0.0 --port 8000

web: export
	cd web && npm ci && npm run build

web-dev:
	cd web && npm ci && npm run dev

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run mypy src

test:
	uv run pytest
	cd web && npm ci && npm test

all: data runvalue overturn value policy backtest players export
