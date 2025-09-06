# P05 – Transcription Orchestration & Options (Priority 1)

Goal: Orchestrate transcription lifecycle, manage model selection, and allow fallback modes.

## Outcomes
- Unified transcription service interface.
- Primary path: batch STT.
- Optional flags for realtime or audio completions (hook only in MVP).
- Normalization (punctuation, casing, disfluency markers optional).

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| TRN-1 | Interface Design | Define `Transcriber` protocol/base class. | 1h |  | Not Started | |
| TRN-2 | Batch Adapter | Implement batch STT adapter calling P04 client. | 2h |  | Not Started | |
| TRN-3 | Normalization | Rules for punctuation & casing; optional raw toggle. | 2h |  | Not Started | |
| TRN-4 | Word Timing Merge | Ensure alignment structure uniform. | 1h |  | Not Started | |
| TRN-5 | Error Modes | Graceful fallback & user messaging. | 1h |  | Not Started | |
| TRN-6 | Tests | Adapter + normalization tests. | 2h |  | Not Started | |

## Definition of Done
- Single function: `get_transcript(session_id) -> Transcript` used by scoring pipeline.
