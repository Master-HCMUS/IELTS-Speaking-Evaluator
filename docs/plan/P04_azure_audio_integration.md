# P04 – Azure AI OpenAI Audio Integration (Priority 1)

Goal: Enable secure, configurable calls to Azure Audio API (batch STT) and framework hooks for optional realtime & audio completions.

## Outcomes
- Config-driven client (endpoint, region, api_version, deployment/model id, key/token).
- Batch STT function: submit WAV -> transcript with timestamps.
- Error handling & retry (transient failures, latency measurement).
- Placeholder interfaces for realtime & audio completion expansion.

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| AZA-1 | SDK Decision | Use Azure OpenAI official SDK (chosen) – document rationale (maintainability, auth). | 0.5h |  | Done | Using azure-openai SDK |
| AZA-1a | Client Factory | Implement factory wrapper to inject into transcriber. | 1h |  | Done | ClientFactory implemented |
| AZA-2 | Auth Module | API key + optional AAD token support. | 2h |  | Done | API key + DefaultAzureCredential fallback |
| AZA-3 | Batch STT Submit | Upload WAV, poll/completed retrieval. | 3h |  | Done | Implemented direct call (simple sync) |
| AZA-4 | Timestamp Parsing | Normalize timestamps per word. | 2h |  | Done | Handles words, segments, fallback span |
| AZA-5 | Retry & Metrics | Exponential backoff, record latency. | 2h |  | Done | Implemented in transcribe() |
| AZA-6 | Config UI Hook | Expose model/endpoint selectors (stub). | 1h |  | Pending | Await basic UI layer |
| AZA-7 | Unit Tests | Mocked API responses. | 2h |  | Done | test_transcriber.py added |

## Definition of Done
- Function call returns deterministic structure: list[word, start_ms, end_ms, confidence].
- Latency logged and under target for sample file.
