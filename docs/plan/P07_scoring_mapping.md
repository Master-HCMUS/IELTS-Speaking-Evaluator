# P07 – IELTS Scoring & Band Mapping (Priority 1)

Goal: Translate feature metrics into 0–9 band estimates per criterion with transparent rationales.

## Approach
- Rule-based tiers referencing descriptor language (configurable thresholds YAML).
- Weighted aggregation for overall band (simple mean or descriptor-based rounding).
- Evidence packaging (e.g., list of long pauses, rare word examples).

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| SCORE-1 | Threshold Schema | Design YAML/JSON config for bands. | 2h |  | Done | thresholds.json added |
| SCORE-2 | Mapping Engine | Implement per-criterion mapping functions. | 3h |  | Done | engine.score_features |
| SCORE-3 | Rationale Builder | Generate human-readable explanations. | 2h |  | Done | rationale parts recorded |
| SCORE-4 | Overall Aggregation | Compute final band with rules. | 1h |  | Done | weighted mean + 0.5 rounding |
| SCORE-5 | Calibration Hook | Placeholder for future ML calibration. | 1h |  | Done | future extension TODO comment |
| SCORE-6 | Tests | Unit tests with synthetic feature sets for boundaries. | 3h |  | Done | test_scoring_engine.py |

## Definition of Done
- Function: `score_features(feature_set) -> ScoreSet` including rationales.
- Config changes require no code changes (hot reload in dev acceptable).
