"""
Evaluation package for IELTS Speaking Assessment

This package provides evaluation tools for comparing pronunciation assessment models
(Azure Speech, fine-tuned Whisper) with expert human annotations from benchmark datasets.
"""

try:
    from .dataset_evaluator import SpeechOcean762Evaluator, EvaluationMetrics
    AZURE_EVALUATOR_AVAILABLE = True
except ImportError:
    AZURE_EVALUATOR_AVAILABLE = False
    print("Warning: Azure Speech evaluator not available")

try:
    from .whisper_evaluator import (
        WhisperModelEvaluator, 
        WhisperPronunciationAssessor,
        WhisperEvaluationResult
    )
    WHISPER_EVALUATOR_AVAILABLE = True
except ImportError:
    WHISPER_EVALUATOR_AVAILABLE = False
    print("Warning: Whisper evaluator not available")

try:
    from .standalone_whisper_evaluator import (
        StandaloneWhisperModelEvaluator,
        StandaloneWhisperPronunciationAssessor,
        StandaloneEvaluationMetrics
    )
    STANDALONE_WHISPER_EVALUATOR_AVAILABLE = True
except ImportError:
    STANDALONE_WHISPER_EVALUATOR_AVAILABLE = False
    print("Warning: Standalone Whisper evaluator not available")

# Build __all__ based on what's available
__all__ = []

if AZURE_EVALUATOR_AVAILABLE:
    __all__.extend(['SpeechOcean762Evaluator', 'EvaluationMetrics'])

if WHISPER_EVALUATOR_AVAILABLE:
    __all__.extend([
        'WhisperModelEvaluator',
        'WhisperPronunciationAssessor', 
        'WhisperEvaluationResult'
    ])

if STANDALONE_WHISPER_EVALUATOR_AVAILABLE:
    __all__.extend([
        'StandaloneWhisperModelEvaluator',
        'StandaloneWhisperPronunciationAssessor',
        'StandaloneEvaluationMetrics'
    ])