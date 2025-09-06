"""Audio capture module.

Tasks: AUD-2, AUD-3, AUD-4, AUD-5 (P03)
Spec Refs: Audio Capture (3.1 B), Requirements for mono 16kHz WAV

Note: Real microphone capture requires system audio devices; tests will mock dependencies.
"""
from __future__ import annotations
import wave
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # 16 kHz
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (16-bit)

RMSCallback = Callable[[float], None]

class AudioCaptureError(Exception):
    pass

@dataclass
class CaptureResult:
    audio_path: Path
    duration_sec: float

class AudioRecorder:
    """Stateful audio recorder supporting level metering & duration guard."""

    def __init__(self, rms_callback: Optional[RMSCallback] = None, max_duration_sec: int = 120):
        self._rms_callback = rms_callback
        self._max_duration = max_duration_sec
        self._frames: list[np.ndarray] = []
        self._start_time: float | None = None
        self._stop_event = threading.Event()
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):  # pragma: no cover (device IO)
        if status:
            # Device warnings could be logged
            pass
        if self._stop_event.is_set():
            raise sd.CallbackAbort
        # Convert to mono int16
        data = (indata.copy()).astype(np.float32)
        # Compute RMS for level meter
        if self._rms_callback is not None:
            rms = float(np.sqrt(np.mean(np.square(data))))
            self._rms_callback(rms)
        with self._lock:
            self._frames.append(data)
        # Duration guard
        if self._start_time and (time.time() - self._start_time) >= self._max_duration:
            self.stop()
            raise sd.CallbackAbort

    def start(self):  # pragma: no cover (device IO)
        if self._stream is not None:
            raise AudioCaptureError("Already recording")
        self._frames.clear()
        self._stop_event.clear()
        self._start_time = time.time()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> float:  # pragma: no cover (device IO)
        if self._stream is None:
            raise AudioCaptureError("Not recording")
        self._stop_event.set()
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if self._start_time is None:
            raise AudioCaptureError("No start time recorded")
        duration = time.time() - self._start_time
        self._start_time = None
        return duration

    def discard(self):
        with self._lock:
            self._frames.clear()
        self._start_time = None
        self._stop_event.set()
        if self._stream is not None:  # pragma: no cover
            self._stream.abort()
            self._stream = None

    def write_wav(self, path: Path) -> CaptureResult:
        """Write captured frames to WAV file (16k mono, 16-bit)."""
        with self._lock:
            if not self._frames:
                raise AudioCaptureError("No audio captured")
            concatenated = np.concatenate(self._frames, axis=0)
        # Ensure int16
        if concatenated.dtype != np.int16:
            concatenated = (concatenated * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(concatenated.tobytes())
        duration = len(concatenated) / SAMPLE_RATE
        return CaptureResult(audio_path=path, duration_sec=duration)

# Utility for tests (simulate frames)
class DummyRecorder(AudioRecorder):
    def __init__(self, rms_callback: Optional[RMSCallback] = None, max_duration_sec: int = 120):
        super().__init__(rms_callback, max_duration_sec)

    def feed(self, samples: np.ndarray):
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        with self._lock:
            self._frames.append(samples.astype(np.int16))
        if self._rms_callback:
            rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            self._rms_callback(rms)
