"""
Multi-Objective Whisper Evaluation Module

This module provides tools for evaluating fine-tuned multi-objective Whisper models
(with pronunciation assessment heads) against the SpeechOcean762 dataset with expert
human annotations.

The multi-objective model includes:
- Transcription (STT): Cross-Entropy loss on decoder outputs
- Pronunciation Assessment: MSE loss on 9 assessment objectives
  - Word-level: accuracy, stress, total
  - Phone-level: accuracy
  - Utterance-level: accuracy, fluency, prosodic, completeness, total

Components:
    - core: Main evaluation logic (MultiObjectiveWhisperModelEvaluator, MultiObjectiveWhisperAssessor)
    - cli: Command-line interface for running evaluations

Usage:
    from evaluation.multi_objective_whisper_evaluation import MultiObjectiveWhisperModelEvaluator
    
    # Initialize evaluator
    evaluator = MultiObjectiveWhisperModelEvaluator(
        "src/finetuning/finetuning_pronunciation_assessment/models/kaggle"
    )
    
    # Load dataset and run evaluation
    evaluator.load_dataset(split='test', max_samples=100)
    metrics = evaluator.run_evaluation(save_results=True)
    
    # Print summary
    evaluator.print_evaluation_summary(metrics)
    
    # Or use the CLI:
    python -m src.evaluation.multi_objective_whisper_evaluation.cli \\
        --model-path "src/finetuning/finetuning_pronunciation_assessment/models/kaggle" \\
        --quick-test
"""

from .core import (
    MultiObjectiveWhisperModelEvaluator,
    MultiObjectiveWhisperAssessor,
    MultiObjectiveEvaluationMetrics,
    MultiObjectiveEvaluationResult
)

__all__ = [
    'MultiObjectiveWhisperModelEvaluator',
    'MultiObjectiveWhisperAssessor',
    'MultiObjectiveEvaluationMetrics',
    'MultiObjectiveEvaluationResult'
]
