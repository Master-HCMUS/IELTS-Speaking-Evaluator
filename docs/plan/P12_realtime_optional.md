# P12 – Realtime GPT-4o Audio Mode (Priority 3)

Goal: Optional enhancement allowing low-latency streaming transcription or interactive coaching (scoped prototype only).

## Scope (MVP for this feature)
- Connect to Realtime API; stream mic audio; receive partial transcripts.
- Show live transcript draft before final batch transcript.
- Feature flag controlled (off by default).

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| RT-1 | Connection Handler | WebSocket / Realtime session setup. | 3h |  | Not Started | |
| RT-2 | Audio Chunking | Stream PCM frames sized for latency vs quality. | 2h |  | Not Started | |
| RT-3 | Partial Transcript UI | Live updating area. | 2h |  | Not Started | |
| RT-4 | Merge Logic | Merge realtime partial with final batch transcript. | 2h |  | Not Started | |
| RT-5 | Flag Integration | Toggle from settings (P10). | 1h |  | Not Started | |
| RT-6 | Tests | Mock streaming session & state transitions. | 3h |  | Not Started | |

## Definition of Done
- Demonstrated latency improvement (qualitative) in dev log.
