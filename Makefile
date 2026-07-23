INSTALL := uv
PYTHON := uv run python

FLAKE8 := uv run flake8
MYPY := uv run mypy

SRC := src

.PHONY: help install run debug test lint lint-strict clean

help:
	@echo "Available commands:"
	@echo "  make install"
	@echo "  make run"
	@echo "  make debug"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make lint-strict"
	@echo "  make clean"

install:
	$(INSTALL) sync
	uv pip install pydantic

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

test:
	uv run pytest

lint:
	$(FLAKE8) $(SRC)
	$(MYPY) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--follow-imports=skip \
		--disallow-untyped-defs \
		--check-untyped-defs \
		$(SRC)

lint-strict:
	$(FLAKE8) $(SRC)
	$(MYPY) --strict $(SRC)

clean:
	find . -type d \( \
		-name "__pycache__" -o \
		-name ".mypy_cache" -o \
		-name ".pytest_cache" \
	\) -exec rm -rf {} +