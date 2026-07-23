.PHONY: install run debug clean lint lint-strict

# Install project dependencies
install:
	uv sync
	uv pip install pydantic

# Run the main script
run:
	uv run python -m src

# Run the main script under pdb (Python's built-in debugger)
debug:
	uv run python -m pdb main.py

# Remove temporary files and caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Lint the project (flake8 + mypy)
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Stricter lint pass (optional)
lint-strict:
	uv run flake8 .
	uv run mypy . --strict