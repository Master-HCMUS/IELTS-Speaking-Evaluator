# P02 – Data Model & Local Storage (Priority 1)

Goal: Implement a structured, human-readable **file-based storage layout** (no local DB) that can be cleanly swapped or augmented with Azure cloud storage (Blob / Cosmos DB) later.

## Approach Summary
Local first using a deterministic directory + JSON schema. Introduce an abstraction layer (`StorageBackend`) with a concrete `LocalFileStorageBackend` now and future `AzureBlobStorageBackend` / `CosmosStorageBackend` placeholders.

## Directory Layout (proposed)
```
data/
	index.json                # Lightweight catalog of sessions (id, created_at, cue_card_id, duration, overall_band, model_version)
	sessions/
		<session_id>/
			session.json          # High-level metadata (ids, timestamps, links)
			audio.wav             # Captured mono 16kHz audio
			transcript.json       # { words: [{w, start_ms, end_ms, conf}], text }
			features.json         # Structured metrics grouped by criterion
			scores.json           # Per-criterion + overall bands + rationales
			notes.txt             # (Optional) user prep notes
			export/
				report.pdf          # (Optional) generated on demand
				snapshot.json       # Frozen full snapshot for sharing
```

## JSON Schemas (high-level, versioned)
- Add top-level field `schema_version`: integer (start at 1).
- `features.json` example shape:
	```json
	{
		"schema_version": 1,
		"fluency": {"wpm": 132.5, "mean_length_of_run": 7.2, "pause_rate_per_min": 18, ...},
		"lexical": {"ttr": 0.58, "rare_word_share": 0.12, ...},
		"grammar": {"error_density_per_100": 7, ...},
		"pronunciation": {"wer_proxy": 0.11, "rhythm_variance": 0.23, ...}
	}
	```

## Extensibility for Azure
- `StorageBackend` interface methods: `save_session(meta)`, `save_artifact(session_id, kind, data|bytes)`, `load_session(session_id)`, `list_sessions(filters)`, `delete_session(session_id)`, `export_snapshot(session_id)`.
- Azure Blob mapping: each `<session_id>` becomes a virtual folder/container prefix; index materialized as Azure Table / Cosmos collection later.
- Keep JSON small & atomic (avoid massive monolith); allow streaming upload of audio.

## Index Management
- `index.json` is an array OR object keyed by session id (choose object for O(1) lookup). Include minimal metadata only.
- Update index atomically: write temp file then rename to prevent corruption.

## Concurrency & Safety
- Simple file lock via OS advisory lock or `.lock` file for write operations (future multi-process scenario).
- Validate schema_version; perform in-place upgrader if version mismatch in future.

## Tasks (Revised)
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| DM-1 | Finalize File Layout | Confirm directory + file naming; document. | 1h |  | Done | Implemented in backend + data/.gitkeep |
| DM-2 | Schema Definitions | Draft Pydantic/dataclass models + JSON schema export. | 2h |  | Done | `models/core.py` created |
| DM-3 | Storage Backend Interface | Define `StorageBackend` protocol + errors. | 1h |  | Done | Interface + errors added |
| DM-4 | Local Implementation | Implement file write/read, atomic index updates. | 3h |  | Done | LocalFileStorageBackend |
| DM-5 | Index Service | Functions to list sessions, filter, sort by date. | 2h |  | Done | list_sessions implemented |
| DM-6 | Snapshot Export | Compose consolidated `snapshot.json`. | 1h |  | Done | export_snapshot method |
| DM-7 | Validation Utilities | Schema version check & integrity scan command. | 2h |  | Done | integrity_scan added |
| DM-8 | Azure Abstraction Hooks | Stub classes + interface tests (no real calls). | 1h |  | Done | AzureBlobStorageBackend stub |
| DM-9 | Unit & Integration Tests | Create/save/load/delete lifecycle + corruption handling. | 3h |  | Done | test updated for lifecycle |

## Definition of Done
- Creating a session writes `session.json`, `audio.wav` (placeholder if not yet recorded), and updates `index.json` atomically.
- `list_sessions()` returns metadata sourced only from `index.json` (no deep scans).
- Running an integrity script validates all referenced files & schema versions.
- Swapping to a future Azure backend only requires implementing the interface (no changes to callers).

## Future Considerations
- Optional compression for transcript/features (gzip) if size grows.
- Add `checksum` field in index for audio integrity.
- Background compaction / cleanup of orphaned artifacts.
