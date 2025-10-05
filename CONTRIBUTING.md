# Contributing to MemXLNet-QA

Thank you for your interest in contributing to MemXLNet-QA!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paper-revise
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # Or: uv pip install --system -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_memory.py -v

# Run with coverage
pytest tests/ --cov=src/memxlnet --cov-report=html
```

### Code Quality

```bash
# Lint code
make lint

# Auto-format code
make format

# Clean build artifacts
make clean
```

### Training and Evaluation

```bash
# Basic training
make train

# Phase 2 training (recommended)
make train-phase2

# Evaluation
make evaluate
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 120 characters
- Use ruff for formatting and linting
- Add docstrings to all public functions and classes

## Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(training): add progressive segment curriculum learning

- Implement multi-stage training with increasing segments
- Add warmup controls for base model and global softmax
- Update configuration to support segment progression

Closes #123
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add/update tests as needed
4. Ensure all tests pass
5. Update documentation
6. Submit pull request with clear description

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- Questions about usage

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
