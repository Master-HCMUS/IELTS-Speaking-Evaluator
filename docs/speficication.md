# Technical Specification — Local IELTS Speaking Task 2 Judge (with Azure AI Audio)

**Author:** Product Management  
**Owner:** Phong Nguyen (M365)  
**Status:** Draft for PoC

---

## 1) Product Overview

### 1.1 Goal  
Build a **local desktop/web application** that evaluates **IELTS Speaking – Part 2 (Long Turn)** performances. The app records the candidate, transcribes speech using **Azure AI OpenAI audio models**, computes analytic features locally, and estimates **band scores** for the four official criteria (Fluency & Coherence, Lexical Resource, Grammatical Range & Accuracy, Pronunciation). It presents transparent feedback and evidence to help learners improve.

### 1.2 Key Outcomes
- Accurate, explainable **per‑criterion** scoring approximating IELTS descriptors.  
- Local‑first processing with **selective calls to Azure AI OpenAI audio** for transcription/audio.  
- Clear visual feedback (timing, pauses, lexical diversity, grammar notes, pronunciation proxies).  
- Exportable session reports for learners and coaches.

---

## 2) Users & Use Cases

- **Learners** practicing Part 2 speeches, seeking actionable feedback aligned with IELTS criteria.  
- **Teachers/Coaches** reviewing analytics and trends to guide instruction.  
- **Self‑study platforms** embedding the judge as a local tool with cloud ASR.

**Typical Flow**  
1) User selects a cue card; app shows **1‑minute prep timer** and **2‑minute speaking timer**.  
2) App records audio locally → calls **Azure AI OpenAI audio** for transcription.  
3) Local scoring engine computes features → maps to estimated bands per criterion with explanations.  
4) User reviews transcript, timing, scores, and tips; exports PDF/JSON.

---

## 3) Scope & Requirements

### 3.1 Functional Requirements

**A. Task 2 Session Management**
- Present cue card and two timers: **Prep (60 s)** and **Speak (≤ 120 s)**.  
- Allow simple note‑taking during prep.

**B. Audio Capture & Transport**
- Record **mono 16 kHz** WAV locally; show audio level meter and re‑record option.
- Call **Azure AI OpenAI audio** services for STT/TTS:  
  - **Audio API** for **speech‑to‑text** (Whisper path).  
  - **GPT‑4o audio** **Realtime API** for low‑latency interactions (optional).  
  - **Audio completions** for generating audio feedback or model answers.

**C. Transcription Options**
- **Primary**: Audio API for STT (batch).  
- **Optional**: GPT‑4o audio for realtime or audio completions.  
- Expose model selection (e.g., `gpt-4o-mini-audio-preview`) and API version (e.g., `2025‑01‑01‑preview`).

**D. Feature Extraction (Local)**
- **Fluency**: WPM, mean length of run, pause rate/duration, filled‑pause detection, self‑repetition counts.  
- **Coherence**: discourse markers density, structural sequence heuristics.  
- **Lexical resource**: type‑token ratio, rare‑word share, collocation/idiom signals.  
- **Grammar**: error density (heuristic), complexity metrics.  
- **Pronunciation**: intelligibility via STT alignment/WER, prosodic rhythm.

**E. Score Mapping**
- Align outputs to **four IELTS criteria** bands **0–9** using rubric‑derived rules and/or lightweight calibration.  
- Provide **evidence** (highlighted disfluencies, pause heatmap, lexical charts).  
- Display **disclaimer**: scores are **training feedback**, not official.

**F. Feedback & Reporting**
- Present transcript with time‑coded words, per‑criterion summaries, and improvement tips.  
- Export **PDF** and **JSON**.

**G. Administration & Configuration**
- Configure Azure endpoint, model deployment name, API version, and region.  
- Toggle real‑time audio vs. batch transcription modes.

### 3.2 Non‑Functional Requirements
- **Performance:**  
  - STT latency ≤ 5 s for a 2‑minute utterance when using Azure audio APIs.  
- **Internationalization:**  
  - UI in English first; design for i18n.  
- **Accessibility:**  
  - Keyboard navigation; captions; high‑contrast theme.

---

## 4) Azure AI OpenAI Integration

### 4.1 Supported Capabilities
- **Audio API** for STT, translation, and TTS.  
- **GPT‑4o audio** for realtime low‑latency conversations and audio generation.  
- **Audio completions** via `/chat/completions` with supported voices (Alloy, Echo, Shimmer).

### 4.2 Configuration Parameters
- **Endpoint & Region:** Azure OpenAI resource endpoint.  
- **Model ID:** e.g., `gpt-4o-mini-audio-preview`.  
- **API Version:** e.g., `2025‑01‑01‑preview`.  
- **Auth:** Azure API key or AAD token.

### 4.3 Modes to Support
1) **Batch STT (Audio API)** – Submit WAV, receive transcript + timestamps.  
2) **Realtime** – Stream mic to GPT‑4o audio Realtime API (optional).  
3) **Audio Completions** – Generate audio feedback or model answers.

---

## 5) Scoring Framework (IELTS‑Aligned)

### 5.1 Criteria Anchoring
- Use official **IELTS Speaking Band Descriptors** for **Fluency & Coherence, Lexical Resource, Grammatical Range & Accuracy, Pronunciation**.

### 5.2 Signals
- **Fluency & Coherence:** WPM, pauses, filled‑pause rate, discourse markers.  
- **Lexical Resource:** TTR, rare‑word share, idioms.  
- **Grammar:** error density, complexity.  
- **Pronunciation:** WER proxy, rhythm stability.

### 5.3 Band Estimation
- Per‑criterion score 0–9 via rule‑based mapping; aggregate to overall band.  
- Provide explanations referencing rubric language and evidence.

---

## 6) UX & Reporting

- **Record Screen:** timers, audio level, start/stop, redo.  
- **Results Screen:**  
  - Transcript with timestamps; pause heatmap.  
  - Four cards (Fluency, Lexical, Grammar, Pronunciation) with band estimate and tips.  
- **Export:** PDF + JSON.  
- **Model Settings:** region, endpoint, model, API version, realtime toggle.

---

## 7) Data Model

**Entities**
- Session: id, timestamps, cue card id.  
- Audio: local file path, sample rate.  
- Transcript: text, word timings, confidences.  
- Features: fluency, coherence, lexical, grammar, pronunciation metrics.  
- Scores: per‑criterion band, overall, rationale.  
- Export: PDF path, JSON snapshot.

**Storage Strategy**  
- Chosen: structured local file-based storage (human-readable JSON + deterministic directory layout) for PoC.  
- Rationale: simpler setup, transparent artifacts, easy manual inspection, no DB dependency.  
- Abstraction: `StorageBackend` interface allows future swap/augmentation with Azure Blob or Cosmos DB without refactoring core logic.  
- Layout (summary):  
  - `data/index.json` (catalog of sessions: id, timestamps, cue_card_id, overall_band, model_version).  
  - `data/sessions/<session_id>/session.json` (metadata)  
  - `.../audio.wav`, `transcript.json`, `features.json`, `scores.json`, optional `notes.txt`  
  - `.../export/report.pdf`, `.../export/snapshot.json` (on-demand)  
- Versioning: each JSON includes `schema_version` (start at 1).  
- Atomic updates: write temp then rename for `index.json` to avoid corruption.  
- Future Azure: implement `AzureBlobStorageBackend` that mirrors directory semantics via virtual prefixes; optional centralized index in Table/Cosmos.

---

## 8) Architecture (High‑Level)

**Components**
1) **UI Shell** (desktop Electron / web SPA)  
2) **Audio Subsystem** (capture, WAV writer)  
3) **Azure Audio Connector** (Audio API, Realtime API, audio completions)  
4) **Scoring Engine** (local feature extraction + rubric mapping)  
5) **Storage Layer** (local DB/files)  
6) **Export Service** (PDF/JSON)

---

## 9) Risks & Mitigations

- **Model availability & version drift**: expose model/version in settings; pin by default.  
- **Latency/network variability**: show progress feedback; allow retry.  
- **Accent bias**: emphasize intelligibility and evidence; evaluate across accents.  
- **User expectations**: clear disclaimer that this is **practice feedback** only.

---

## 10) Future Extensions

- **Azure Speech Pronunciation Assessment** for detailed phoneme scoring.  
- **Realtime coaching** using GPT‑4o audio Realtime API.  
- **Personalized study plans** based on weakness patterns.  
- **Educator dashboard** for cohort analytics.

---