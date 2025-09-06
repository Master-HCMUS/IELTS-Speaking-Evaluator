"""Tests for audio capture (dummy recorder).

Tasks: AUD-2, AUD-3, AUD-4, AUD-5, AUD-6
"""
from pathlib import Path
import numpy as np
from audio.capture import DummyRecorder, AudioCaptureError, SAMPLE_RATE


def test_dummy_record_and_wav(tmp_path):
    rms_values = []
    rec = DummyRecorder(rms_callback=lambda v: rms_values.append(v))
    # Simulate 0.5s of a 440Hz tone
    t = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 440 * t))
    rec.feed(samples.astype(np.float32))
    out = rec.write_wav(tmp_path / "test.wav")
    assert abs(out.duration_sec - 0.5) < 0.02
    assert rms_values, "RMS callback not triggered"


def test_write_without_audio_raises(tmp_path):
    rec = DummyRecorder()
    try:
        rec.write_wav(tmp_path / "empty.wav")
        assert False, "Expected AudioCaptureError"
    except AudioCaptureError:
        pass
