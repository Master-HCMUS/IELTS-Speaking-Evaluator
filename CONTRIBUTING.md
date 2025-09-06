# Contributing Guidelines

This project follows a spec-driven workflow.

## Process
1. Read `docs/speficication.md` and relevant `docs/plan/Pxx_*.md` file.
2. Pick a task ID (e.g., FEAT-3) and open/change only code relevant to that task.
3. Add task ID + spec section reference in docstrings or comments.
4. Write/extend tests before final implementation when feasible.
5. Run lint & tests before submitting PR.

## Coding Standards
- Python 3.10+
- Ruff + mypy strict.
- Structured logging only via `logging_util.structured_logger`.

## Tests
```bash
pytest
```

## Pull Requests
Include checklist (see `.github/copilot-instructions.md`).
