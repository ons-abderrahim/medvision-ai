# Contributing to MedVision AI

Thank you for your interest in contributing! We welcome contributions of all kinds.

## How to Contribute

### Reporting Bugs
Open an issue with:
- A clear title and description
- Steps to reproduce
- Expected vs actual behaviour
- Your environment (OS, Python, PyTorch versions)

### Feature Requests
Open an issue with the `enhancement` label. Describe the use case and why it would benefit the project.

### Pull Requests

1. **Fork** the repository and create your branch from `develop`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Make your changes.** Ensure:
   - All existing tests pass: `pytest tests/ -v`
   - New features have tests
   - Code is formatted: `black src/ tests/`
   - No lint errors: `ruff check src/ tests/`
   - Type annotations added where sensible

4. **Write a clear commit message** following [Conventional Commits](https://conventionalcommits.org):
   ```
   feat: add BioViL model integration
   fix: correct SHAP background sampling
   docs: update MODEL_CARD with ISIC results
   ```

5. **Open a PR** against the `develop` branch with a description of your changes.

## Code Style

- **Formatter:** black (line-length 100)
- **Linter:** ruff
- **Type checker:** mypy
- **Docstrings:** Google style

## Medical AI Ethics

Given the medical context of this project:
- Never add features that could enable autonomous clinical diagnosis
- Always document limitations and failure modes in model cards
- Flag any changes that could affect model safety or fairness
