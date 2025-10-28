"""
STT Whisper Evaluation Module

This module provides tools for evaluating fine-tuned Whisper models
for speech-to-text (STT) and pronunciation assessment against the 
SpeechOcean762 dataset with expert human annotations.

Components:
    - core: Main evaluation logic (StandaloneWhisperModelEvaluator, StandaloneWhisperPronunciationAssessor)
    - cli: Command-line interface for running evaluations

Usage:
    from evaluation.stt_whisper_evaluation.core import StandaloneWhisperModelEvaluator
    
    # Or use the CLI:
    python -m src.evaluation.stt_whisper_evaluation.cli --model-path "models/whisper_development" --max-samples 50
"""

from .core import (
    StandaloneWhisperModelEvaluator,
    StandaloneWhisperPronunciationAssessor,
    StandaloneEvaluationMetrics,
    WhisperEvaluationResult
)

__all__ = [
    'StandaloneWhisperModelEvaluator',
    'StandaloneWhisperPronunciationAssessor',
    'StandaloneEvaluationMetrics',
    'WhisperEvaluationResult'
]
