.PHONY: sync run plot lint all

sync:
	uv sync

run:
	uv run python experiment.py

plot:
	uv run python plot.py

lint:
	uv run ruff check . && uv run ruff format --check .

all: sync run plot lint
