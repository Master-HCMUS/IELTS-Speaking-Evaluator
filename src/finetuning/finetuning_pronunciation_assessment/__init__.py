"""
Pronunciation Assessment Fine-tuning Module for Whisper Models.

This module provides functionality to fine-tune Whisper models for pronunciation assessment
while maintaining transcription capabilities. It supports both word-level and phone-level
assessment granularities.

Key Features:
- Maintains Whisper's transcription capabilities (decoder retained)
- Adds assessment heads for word-level and phone-level scoring
- Multi-objective training (ASR + assessment losses)
- Supports SpeechOcean762 dataset with detailed word/phone annotations
"""

from .whisper_pronunciation_model import WhisperPronunciationAssessmentModel
from .data_processor import SpeechOcean762PronunciationProcessor
from .training_config import PronunciationTrainingConfig
from .trainer import PronunciationAssessmentTrainer

__all__ = [
    'WhisperPronunciationAssessmentModel',
    'SpeechOcean762PronunciationProcessor',
    'PronunciationTrainingConfig',
    'PronunciationAssessmentTrainer'
]