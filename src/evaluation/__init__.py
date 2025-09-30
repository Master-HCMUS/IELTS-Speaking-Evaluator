"""
Evaluation package for IELTS Speaking Assessment

This package provides evaluation tools for comparing pronunciation assessment models
(Azure Speech, fine-tuned Whisper) with expert human annotations from benchmark datasets.
"""

from .dataset_evaluator import SpeechOcean762Evaluator, EvaluationMetrics
from .whisper_evaluator import (
    WhisperModelEvaluator, 
    WhisperPronunciationAssessor,
    WhisperEvaluationResult
)

__all__ = [
    'SpeechOcean762Evaluator', 
    'EvaluationMetrics',
    'WhisperModelEvaluator',
    'WhisperPronunciationAssessor', 
    'WhisperEvaluationResult'
]