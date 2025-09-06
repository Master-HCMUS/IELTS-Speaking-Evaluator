"""Tests for AzureOpenAITranscriber offline behavior (mock).

Tasks: AZA-1a, AZA-5, AZA-7, AZA-2, AZA-3, AZA-4
"""
from pathlib import Path
from transcription.transcriber import AzureOpenAITranscriber, ClientFactory, TranscriptionError
from models.core import Transcript

class DummyFactory(ClientFactory):
    def create(self):  # type: ignore[override]
        class _DummyClient:
            class audio:  # type: ignore
                class transcriptions:  # type: ignore
                    @staticmethod
                    def create(model, file, response_format):  # noqa: D401
                        # Simulated verbose response with per-word timings
                        return {
                            "text": "dummy transcript",
                            "words": [
                                {"word": "dummy", "start": 0.0, "end": 0.5, "confidence": 0.95},
                                {"word": "transcript", "start": 0.5, "end": 1.0, "confidence": 0.90},
                            ],
                        }
        return _DummyClient()


def test_transcriber_dummy(monkeypatch, tmp_path):
    # Bypass real factory client creation to stay offline
    transcriber = AzureOpenAITranscriber(client_factory=DummyFactory())
    t = transcriber.transcribe(tmp_path / "fake.wav")
    assert isinstance(t, Transcript)
    assert t.text == "dummy transcript"
    assert len(t.words) == 2


def test_transcriber_retries(monkeypatch, tmp_path):
    class FailingTranscriber(AzureOpenAITranscriber):
        def _call_api(self, audio_path):  # type: ignore[override]
            raise RuntimeError("fail")
    ft = FailingTranscriber(client_factory=DummyFactory(), max_retries=2, backoff_sec=0)
    try:
        ft.transcribe(tmp_path / "fake.wav")
        assert False, "Expected TranscriptionError"
    except TranscriptionError as e:
        assert "2 attempts" in str(e)


def test_word_normalization():
    class Dummy(AzureOpenAITranscriber):
        def _call_api(self, audio_path):  # type: ignore[override]
            raw = [
                {"word": "hi", "start": 0.0, "end": 0.25, "confidence": 0.9},
                {"word": "there", "start": 0.25, "end": 0.75, "confidence": 0.88},
            ]
            words = self._normalize_words(raw)
            return Transcript(words=words, text="hi there")
    t = Dummy(client_factory=DummyFactory())
    tr = t.transcribe(Path("fake.wav"))
    assert tr.words[0].start_ms == 0 and tr.words[0].end_ms == 250
    assert tr.words[1].start_ms == 250 and tr.words[1].end_ms == 750
