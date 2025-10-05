.PHONY: install test lint clean

install:
	uv sync

test:
	pytest tests/ -v --cov=src/memxlnet --cov-report=term-missing

lint:
	@echo "Running ruff..."
	uv run ruff check --fix src/ tests/ examples/
	@echo "Running mypy..."
	uv run mypy src/
	@echo "Running formatter..."
	uv run ruff format src/ tests/ examples/
	@echo "Linting complete!"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	@echo "Cleaning cache directories..."
	rm -rf .cache/ cache*/
	@echo "Cleaning test outputs..."
	rm -rf test_output/ test_resume_output/
	@echo "Clean complete!"
