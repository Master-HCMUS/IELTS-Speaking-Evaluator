# P06 – Local Feature Extraction Engine (Priority 1)

Goal: Compute all analytic features locally from transcript + timings (and audio if needed) feeding scoring mapping.

## Feature Groups
- Fluency: WPM, mean length of run, pause rate/duration, filled pauses, repetitions.
- Coherence: discourse markers density, structural sequence heuristics.
- Lexical: type-token ratio (windowed), rare-word share (frequency list), idiom/collocation heuristic.
- Grammar: error density (simple grammar checker or heuristic), complexity (mean clause length, subordinate markers).
- Pronunciation: WER proxy (alignment vs transcript?), rhythm metrics (inter-word timing variance).

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| FEAT-1 | Tokenization | Implement robust tokenizer word + filler detection. | 2h |  | Not Started | |
| FEAT-2 | Timing Analysis | Pause segmentation thresholds (e.g., >250ms). | 2h |  | Not Started | |
| FEAT-3 | Fluency Metrics | Implement runs, WPM, repetitions. | 3h |  | Not Started | |
| FEAT-4 | Coherence Metrics | Discourse marker list & density calc. | 2h |  | Not Started | |
| FEAT-5 | Lexical Metrics | TTR, rare word share (wordfreq list). | 3h |  | Not Started | |
| FEAT-6 | Grammar Heuristics | Simple pattern-based error counts. | 3h |  | Not Started | |
| FEAT-7 | Pronunciation Proxies | Rhythm variance, WER vs ref (if alt). | 3h |  | Not Started | |
| FEAT-8 | Data Structure | Feature dataclass & JSON schema. | 1h |  | Not Started | |
| FEAT-9 | Unit Tests | Each metric with synthetic transcript fixtures. | 3h |  | Not Started | |

## Definition of Done
- `extract_features(transcript, audio_meta) -> FeatureSet` returns all metrics with docstrings.
