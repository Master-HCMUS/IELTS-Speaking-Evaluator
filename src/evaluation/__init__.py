"""
Evaluation package for IELTS Speaking Assessment

This package provides evaluation tools for comparing pronunciation assessment models
(Azure Speech, fine-tuned Whisper, multi-objective Whisper) with expert human annotations
from benchmark datasets.

Sub-packages:
    - azure_speech_evaluation: Evaluate Azure Speech API (SpeechOcean762Evaluator)
    - stt_whisper_evaluation: Evaluate STT-only Whisper models (StandaloneWhisperModelEvaluator)
    - multi_objective_whisper_evaluation: Evaluate multi-objective Whisper (with assessment heads)

Usage Examples:
    # Azure Speech evaluation
    from evaluation.azure_speech_evaluation import SpeechOcean762Evaluator
    
    # STT Whisper evaluation
    from evaluation.stt_whisper_evaluation import StandaloneWhisperModelEvaluator
    
    # Multi-objective Whisper evaluation
    from evaluation.multi_objective_whisper_evaluation import MultiObjectiveWhisperModelEvaluator
    
    # Or use CLI commands:
    python -m src.evaluation.azure_speech_evaluation.cli --max-samples 50
    python -m src.evaluation.stt_whisper_evaluation.cli --model-path "models/whisper_development"
    python -m src.evaluation.multi_objective_whisper_evaluation.cli --model-path "models/kaggle"
"""

# Azure Speech evaluation
try:
    from .azure_speech_evaluation import SpeechOcean762Evaluator
    AZURE_EVALUATOR_AVAILABLE = True
except ImportError:
    AZURE_EVALUATOR_AVAILABLE = False
    print("Warning: Azure Speech evaluator not available")

# STT Whisper model evaluation
try:
    from .stt_whisper_evaluation import (
        StandaloneWhisperModelEvaluator,
        StandaloneWhisperPronunciationAssessor,
        StandaloneEvaluationMetrics
    )
    STT_WHISPER_EVALUATOR_AVAILABLE = True
except ImportError:
    STT_WHISPER_EVALUATOR_AVAILABLE = False
    print("Warning: STT Whisper evaluator not available")

# Multi-objective Whisper model evaluation (with pronunciation assessment heads)
try:
    from .multi_objective_whisper_evaluation import (
        MultiObjectiveWhisperModelEvaluator,
        MultiObjectiveWhisperAssessor,
        MultiObjectiveEvaluationMetrics,
        MultiObjectiveEvaluationResult
    )
    MULTI_OBJECTIVE_WHISPER_EVALUATOR_AVAILABLE = True
except ImportError:
    MULTI_OBJECTIVE_WHISPER_EVALUATOR_AVAILABLE = False
    print("Warning: Multi-objective Whisper evaluator not available")

# Build __all__ based on what's available
__all__ = []

if AZURE_EVALUATOR_AVAILABLE:
    __all__.extend(['SpeechOcean762Evaluator'])

if STT_WHISPER_EVALUATOR_AVAILABLE:
    __all__.extend([
        'StandaloneWhisperModelEvaluator',
        'StandaloneWhisperPronunciationAssessor',
        'StandaloneEvaluationMetrics'
    ])

if MULTI_OBJECTIVE_WHISPER_EVALUATOR_AVAILABLE:
    __all__.extend([
        'MultiObjectiveWhisperModelEvaluator',
        'MultiObjectiveWhisperAssessor',
        'MultiObjectiveEvaluationMetrics',
        'MultiObjectiveEvaluationResult'
    ])