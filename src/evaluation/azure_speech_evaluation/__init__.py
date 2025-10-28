"""
Azure Speech Evaluation Module

This module provides tools for evaluating Azure Speech pronunciation assessment
against the SpeechOcean762 dataset with expert human annotations.

Components:
    - core: Main evaluation logic (SpeechOcean762Evaluator)
    - cli: Command-line interface for running evaluations

Usage:
    from evaluation.azure_speech_evaluation.core import SpeechOcean762Evaluator
    
    # Or use the CLI:
    python -m src.evaluation.azure_speech_evaluation.cli --max-samples 50
"""

from .core import SpeechOcean762Evaluator

__all__ = ['SpeechOcean762Evaluator']
