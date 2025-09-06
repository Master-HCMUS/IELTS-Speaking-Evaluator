# Copilot Project Instructions – IELTS Speaking Evaluation (Spec‑Driven Development)

These guidelines steer AI code suggestions so all contributions remain tightly aligned with the technical specification in `docs/speficication.md` and implementation plans under `docs/plan/`.

## Core Principles
1. Spec First: Do not invent features not described in the spec or approved plan files.
2. Plan Traceability: Every new module/function should map to a task ID from a `Pxx_*.md` plan file (add the ID in comments / docstrings).
3. Local‑First: Optimize for offline/local processing; Azure calls only where defined (transcription, optional realtime, audio completions).
4. Separation of Concerns: Follow component boundaries (audio, transcription, features, scoring, storage, export, ui shell). Prefer clean architecture principles.
5. Extensibility: Design storage & scoring so future Azure backends / ML calibration can slot in without breaking existing interfaces.
6. Reference `docs/speficication.md` for authoritative PoC requirements and acceptance criteria.
7. Reference `docs/plan/*` — feature and supporting implementation plans

## Change workflow (how to respond when editing files)
- Start with a brief, step-by-step plan.
- Group changes by file, using the file path as the header.
- For each file:
  - Give a short summary of what changes are needed.
  - Provide exactly one code block per file.
  - The code block must:
    - Start with four backticks and a language id (e.g., markdown, typescript, python).
    - Include a first-line comment with the absolute filepath.
    - Show only the changes; use comments like “...existing code...” for unchanged regions.
- Keep responses concise, professional, and deterministic. Do not include chain-of-thought; prefer succinct, actionable notes.

## Testing Expectations
- Each new feature must include at least one unit test (pytest assumed) referencing the task ID.
- Use synthetic small transcripts/audio metadata for deterministic tests.
- Avoid network calls in tests; mock Azure responses.

## Error Handling & Logging
- Use structured logging (JSON or key=value) for pipeline stages (capture, transcribe, features, score, export) including latency metrics.

## Performance Targets
- Keep feature extraction functions pure & fast (target << 500ms for 2‑minute transcript on average hardware).
- Defer heavy NLP/ML (e.g., large transformer models) unless explicitly added to plan.

## Security & Secrets
- Never hardcode API keys; use environment variables or a secure local key storage helper.
- Redact secrets in logs.

## Accessibility & Internationalization
- UI text centralization: pull strings from a single resource map for future i18n (task IDs in P11).

## Code Comment Template (Example)
```python
# Task: FEAT-3 (P06) – Implement runs, WPM, repetitions
# Spec Ref: Section 5.2 Fluency metrics
```

## PR / Change Checklist (Embed in PR Template Later)
- [ ] Linked Task ID(s): ...
- [ ] Spec section(s) referenced: ...
- [ ] Tests added/updated
- [ ] No new unplanned dependencies
- [ ] Storage schema unchanged or version bumped with migrator

## Out of Scope (Reject Suggestions)
- Full ML band prediction models
- Cloud sync / multi-user dashboards
- Phoneme-level pronunciation APIs (future extension)
- Adding databases unless P02 is revised

## When In Doubt
Ask for decision making
Consult `P00_overview.md` index and associated plan file; if ambiguity remains, add a TODO with the task ID and minimal stub—do not over-engineer.