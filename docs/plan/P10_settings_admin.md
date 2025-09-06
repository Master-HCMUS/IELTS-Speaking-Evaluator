# P10 – Settings & Configuration Module (Priority 2)

Goal: UI + backend for managing Azure credentials, model selection, and feature toggles.

## Settings
- Endpoint URL / Region
- Deployment/Model ID
- API Version
- Auth method (API Key vs AAD)
- Realtime mode toggle (flag)

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| CFG-1 | Settings Model | Persisted config object with validation. | 1h |  | Not Started | |
| CFG-2 | Secure Storage | OS keyring or encrypted local file for key. | 2h |  | Not Started | |
| CFG-3 | Settings UI | Form + test connection button. | 3h |  | Not Started | |
| CFG-4 | Validation Logic | Ping model & version check. | 2h |  | Not Started | |
| CFG-5 | Realtime Toggle | Persist & broadcast to components. | 1h |  | Not Started | |
| CFG-6 | Tests | Validation + secure storage mock. | 2h |  | Not Started | |

## Definition of Done
- Changing settings updates subsequent transcription requests without restart.
