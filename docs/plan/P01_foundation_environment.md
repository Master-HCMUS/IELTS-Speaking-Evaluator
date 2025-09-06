# P01 – Repo & Dev Environment Foundation (Priority 1)

Goal: Establish minimal, repeatable development environment enabling rapid iteration (local-first) and Azure audio API calls.

## Outcomes
- Project skeleton (e.g., `app/`, `engine/`, `scripts/`, `tests/`).
- Package management & linting configured (Python + optional TypeScript/Electron shell decision documented).
- .env / secure config pattern (do not commit secrets).
- Logging & basic error boundary.

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| FND-1 | Tech Stack Decision | Choose Electron+React vs pure web vs CLI-first; document rationale. | 2h |  | Done | Decision: Python core + future Electron shell; start CLI tests now. |
| FND-2 | Repo Init | Initialize git, license, README, contribution guidelines. | 1h |  | Done | LICENSE, README, CONTRIBUTING added |
| FND-3 | Python Environment | Create `pyproject.toml` (poetry / hatch) or `requirements.txt`; set versions. | 1h |  | Done | pyproject with deps |
| FND-4 | Core Folders | Create module folders (`audio`, `transcribe`, `features`, `scoring`, `storage`, `export`, `ui`). | 1h |  | In Progress | partial (storage, models, config) |
| FND-5 | Lint & Format | Ruff/Flake8 + Black + mypy baseline. | 2h |  | Pending | ruff+mypy config added; black not yet configured |
| FND-6 | Logging Setup | Structured logging utility (UTC timestamps). | 1h |  | Done | structured logger module |
| FND-7 | Config Loader | Implement config dataclass (endpoint, key, model, api_version). | 1h |  | Done | load_config implemented |
| FND-8 | Health Script | Simple script validating Azure creds & model list. | 1h |  | Done | scripts/health_check.py created |
| FND-9 | Test Harness | Pytest baseline + first smoke test. | 1h |  | Done | storage backend test |

## Definition of Done
- Running `pytest` succeeds.
- `python scripts/health_check.py` validates Azure connectivity (mock if offline).
- Documented in root README.
