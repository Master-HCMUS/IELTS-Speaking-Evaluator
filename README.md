# IELTS Speaking Part 2 Evaluation – PoC

Spec-driven local application evaluating IELTS Speaking Task 2 responses.

## Status
Early foundation: storage backend + core models scaffolded.

## Quick Start
```bash
# (Optional) create & activate virtual environment
pip install -e .[dev]
pytest
```

## Project Structure
- `docs/` – Specification & planning (`speficication.md`, `plan/`)
- `src/models` – Core data models
- `src/storage` – Storage backend abstraction & local implementation
- `tests/` – Pytest suite

## Development Principles
See `.github/copilot-instructions.md`.
