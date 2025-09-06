"""Transcription interface stubs.

Task: TRN-1 (P05) – Interface Design
Decision: Use Azure OpenAI SDK for batch STT.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol
from models.core import Transcript, WordTiming

class TranscriptionError(Exception):
    pass

class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript: ...

# Placeholder Azure implementation stub (to be completed in P04/P05 tasks)
class AzureOpenAITranscriber(Transcriber):
    def __init__(self, config_loader, client_factory):  # dependency injection for testability
        self._config_loader = config_loader
        self._client_factory = client_factory

    def transcribe(self, audio_path: Path) -> Transcript:  # pragma: no cover (stub)
        raise NotImplementedError("Batch STT integration pending (AZA tasks)")
