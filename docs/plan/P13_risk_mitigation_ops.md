# P13 – Risk Mitigations & Ops Considerations (Priority 2)

Goal: Track mitigation actions for identified risks (model drift, latency, accent bias, expectation setting).

## Risks & Actions
| Risk | Impact | Mitigation Task | Owner | Status |
|------|--------|-----------------|-------|--------|
| Model Version Drift | Inconsistent scoring over time | Expose version in settings; store version per session |  | Not Started |
| Latency Variability | Poor UX wait times | Add progress states & retry with backoff |  | Not Started |
| Accent Bias | Unfair scoring | Collect sample accents; review WER metrics |  | Not Started |
| User Over-Reliance | Misinterpreting scores as official | Prominent disclaimer + onboarding text |  | Not Started |
| Local Resource Constraints | Performance issues on low-end devices | Optimize feature extraction; measure CPU time |  | Not Started |

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| RSK-1 | Version Tagging | Persist model + api_version with session. | 1h |  | Not Started | |
| RSK-2 | Latency Metrics | Capture timing logs & simple dashboard (dev). | 2h |  | Not Started | |
| RSK-3 | Disclaimer UX | Implement persistent banner (ties to P09). | 1h |  | Not Started | |
| RSK-4 | Accent Benchmark Plan | Outline evaluation dataset spec. | 2h |  | Not Started | |
| RSK-5 | Perf Profiling Script | Time feature extraction stages. | 2h |  | Not Started | |

## Definition of Done
- Risk table updated with status and evidence notes.
