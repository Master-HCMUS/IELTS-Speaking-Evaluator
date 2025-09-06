# P03 – Audio Capture & WAV Pipeline (Priority 1)

Goal: Robust local audio capture adhering to mono 16 kHz WAV requirement with level metering and re-record option.

## Outcomes
- Cross-platform microphone capture (Windows priority) -> PCM 16-bit mono 16kHz.
- Real-time audio level (RMS) visualization feed.
- Save/overwrite logic (re-record).

## Tasks
| ID | Task | Description | Est | Owner | Status | Notes |
|----|------|-------------|-----|-------|--------|-------|
| AUD-1 | Lib Selection | Choose PyAudio / sounddevice / web MediaRecorder (if web). | 1h |  | Done | Chosen: sounddevice + numpy |
| AUD-2 | Capture Module | Implement start/stop, buffer, write WAV utility. | 3h |  | Done | AudioRecorder + DummyRecorder |
| AUD-3 | Level Meter | Compute short-window RMS for UI. | 1h |  | Done | RMS callback implemented |
| AUD-4 | Re-record Flow | Delete old file & state reset. | 1h |  | Done | discard() clears frames |
| AUD-5 | Duration Guard | Hard stop at 120s speak timer. | 1h |  | Done | stop logic / callback guard |
| AUD-6 | Unit Tests | WAV properties test, re-record logic, duration cap. | 2h |  | Done | test_audio_capture.py |

## Definition of Done
- Generated WAV passes: mono, 16kHz, 16-bit, correct duration tolerance.
- Level meter responds within <150ms update latency.
