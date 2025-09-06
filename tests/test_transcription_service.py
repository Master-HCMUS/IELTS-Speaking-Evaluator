"""Tests for transcription service orchestration.

Tasks: TRN-2..TRN-6
"""
from pathlib import Path
from models.core import Transcript, WordTiming
from transcription.service import TranscriptionService, NormalizationOptions
from transcription.transcriber import Transcriber, TranscriptionError
from storage.backend import LocalFileStorageBackend
from models.core import SessionMeta
import uuid
from datetime import datetime


class DummyTranscriber(Transcriber):
    def __init__(self, fail=False):
        self.fail = fail
    def transcribe(self, audio_path: Path) -> Transcript:  # type: ignore[override]
        if self.fail:
            raise TranscriptionError("boom")
        words = [
            WordTiming(w="Hello", start_ms=0, end_ms=300),
            WordTiming(w="Hello", start_ms=310, end_ms=600),
            WordTiming(w="world", start_ms=610, end_ms=900),
        ]
        return Transcript(words=words, text="Hello Hello world")


def setup_session(tmp_path, monkeypatch):
    """Isolated session setup with dedicated temp data root.

    We patch storage backend constants BEFORE init to avoid cross-test state leakage.
    """
    import storage.backend as sb
    monkeypatch.setattr(sb, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(sb, "INDEX_FILE", tmp_path / "data" / "index.json")
    backend = LocalFileStorageBackend()
    backend.init()
    sid = f"sess_{uuid.uuid4().hex[:8]}"
    backend.create_session(SessionMeta(session_id=sid, created_at=datetime.utcnow(), cue_card_id=None, model_version=None))
    return backend, sid


def test_service_success(tmp_path, monkeypatch):
    backend, sid = setup_session(tmp_path, monkeypatch)
    service = TranscriptionService(backend, DummyTranscriber(), NormalizationOptions(lowercase=True))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFfake")
    tr = service.get_transcript(sid, audio)
    assert tr.text == "hello hello world"  # normalized lowercase
    assert len(tr.words) == 2  # merged duplicate Hello tokens


def test_service_fallback(tmp_path, monkeypatch):
    backend, sid = setup_session(tmp_path, monkeypatch)
    service = TranscriptionService(backend, DummyTranscriber(fail=True))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFfake")
    tr = service.get_transcript(sid, audio)
    assert tr.text == "__TRANSCRIPTION_FAILED__"
    assert tr.words == []