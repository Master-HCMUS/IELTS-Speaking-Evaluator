"""
Evaluation package for IELTS Speaking Assessment

This package provides evaluation tools for comparing Azure Speech pronunciation
assessment with expert human annotations from benchmark datasets.
"""

from .dataset_evaluator import SpeechOcean762Evaluator, EvaluationMetrics

__all__ = ['SpeechOcean762Evaluator', 'EvaluationMetrics']