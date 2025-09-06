"""Transcription interface stubs.

Task: TRN-1 (P05) – Interface Design
Decision: Use Azure OpenAI SDK for batch STT.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Callable, Optional
import time
import hashlib
from models.core import Transcript, WordTiming
from logging_util.structured_logger import logger
from config.app_config import load_config
from pydantic import BaseModel

try:  # azure-openai / openai might not be installed in minimal envs
    from openai import AzureOpenAI  # type: ignore
    from azure.identity import DefaultAzureCredential  # type: ignore
except Exception:  # pragma: no cover
    AzureOpenAI = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore

class TranscriptionError(Exception):
    pass

class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript: ...

# Placeholder Azure implementation stub (to be completed in P04/P05 tasks)
class ClientFactory:
    """Creates AzureOpenAI clients (Task: AZA-1a) with API key or AAD token.

    Auth precedence: API key > AAD (DefaultAzureCredential) if key absent.
    """

    def __init__(self, cfg=None):
        self._cfg = cfg or load_config()

    def create(self):  # pragma: no cover (network not executed in tests)
        if AzureOpenAI is None:
            raise TranscriptionError("azure-openai SDK not available")
        if self._cfg.api_key:
            return AzureOpenAI(
                api_key=self._cfg.api_key,
                api_version=self._cfg.api_version,
                azure_endpoint=self._cfg.endpoint,
            )
        if DefaultAzureCredential is None:
            raise TranscriptionError("No API key and azure-identity not available for AAD auth")
        cred = DefaultAzureCredential()
        # azure-openai python SDK may not yet support token credential directly; placeholder
        return AzureOpenAI(
            api_key=None,
            api_version=self._cfg.api_version,
            azure_endpoint=self._cfg.endpoint,
            azure_ad_token_provider=lambda: cred.get_token("https://cognitiveservices.azure.com/.default").token,  # type: ignore[arg-type]
        )


class _Word(BaseModel):
    word: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


class AzureOpenAITranscriber(Transcriber):
    """Batch STT transcriber using Azure OpenAI Audio API (Tasks: AZA-2..AZA-5)."""

    def __init__(self, client_factory: ClientFactory, max_retries: int = 3, backoff_sec: float = 1.0):
        self._factory = client_factory
        self._max_retries = max_retries
        self._backoff_sec = backoff_sec

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]

    def transcribe(self, audio_path: Path) -> Transcript:
        start = time.time()
        attempt = 0
        last_err: Exception | None = None
        while attempt < self._max_retries:
            attempt += 1
            try:
                transcript = self._call_api(audio_path)
                latency = round((time.time() - start) * 1000, 1)
                logger.info("transcribe_success", latency_ms=latency, attempts=attempt, audio_hash=self._hash_file(audio_path))
                return transcript
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.error("transcribe_attempt_failed", attempt=attempt, error=str(e))
                if attempt >= self._max_retries:
                    break
                time.sleep(self._backoff_sec * attempt)
        raise TranscriptionError(f"Transcription failed after {self._max_retries} attempts: {last_err}")

    def _normalize_words(self, raw_segments: list[dict]) -> List[WordTiming]:
        words: List[WordTiming] = []
        for seg in raw_segments:
            start = int(seg.get("start", 0) * 1000)
            end = int(seg.get("end", start/1000 + 0.5) * 1000) if isinstance(seg.get("end"), (int,float)) else start + 500
            words.append(WordTiming(
                w=seg.get("word", ""),
                start_ms=start,
                end_ms=end,
                confidence=seg.get("confidence"),
            ))
        return words

    def _call_api(self, audio_path: Path) -> Transcript:  # pragma: no cover (network path exercised via mock tests)
        """Invoke Azure OpenAI audio transcription.

        Tasks: AZA-2 (Auth), AZA-3 (Batch STT Submit), AZA-4 (Timestamp Parsing).
        Strategy:
          1. Acquire client (API key or AAD) via factory.
          2. Stream file to transcription endpoint.
          3. Request verbose/word-level timestamps if supported.
          4. Normalize heterogeneous response shapes to internal Transcript.
        Fallback:
          - If API does not return per-word timings, fabricate a single span covering whole text.
        """
        if not audio_path.exists():  # early validation for clearer error
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        client = self._factory.create()
        cfg = self._factory._cfg  # type: ignore[attr-defined]
        response = None
        try:
            with audio_path.open('rb') as f:
                # Modern OpenAI/Azure pattern; timestamp granularity may vary by model.
                # We request verbose JSON to attempt to get 'words'.
                response = client.audio.transcriptions.create(  # type: ignore[call-arg]
                    model=cfg.model,
                    file=f,
                    response_format="verbose_json",
                )
        except Exception as e:  # noqa: BLE001
            # Surface as domain exception for retry loop.
            raise TranscriptionError(f"API call failed: {e}") from e

        # Response object might be SDK model or dict; unify access.
        def _get(obj, name, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        text = _get(response, "text", "") or ""

        raw_segments: list[dict] = []
        # Prefer explicit 'words'
        words_payload = _get(response, "words")
        if words_payload and isinstance(words_payload, list):
            for w in words_payload:
                if isinstance(w, dict):
                    raw_segments.append(w)
                else:  # SDK object with attributes
                    raw_segments.append({
                        "word": _get(w, "word", ""),
                        "start": _get(w, "start"),
                        "end": _get(w, "end"),
                        "confidence": _get(w, "confidence"),
                    })
        else:
            # Try segments -> words nesting (some variants expose segments with timing)
            segments = _get(response, "segments")
            if segments and isinstance(segments, list):
                for seg in segments:
                    if isinstance(seg, dict):
                        start = seg.get("start")
                        end = seg.get("end")
                        token_word = seg.get("text", "").strip()
                        if token_word:
                            raw_segments.append({"word": token_word, "start": start, "end": end})
            # Fallback single span
            if not raw_segments:
                raw_segments = [{"word": text.strip() or "(empty)", "start": 0.0, "end": max(len(text.split())*0.3, 0.5)}]

        words = self._normalize_words(raw_segments)
        return Transcript(words=words, text=text)
