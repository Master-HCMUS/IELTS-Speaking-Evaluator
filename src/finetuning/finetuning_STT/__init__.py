"""
Speech-to-Text (STT) Fine-tuning Module for Whisper Models.

This module provides comprehensive functionality to fine-tune OpenAI Whisper models
on the SpeechOcean762 dataset for improved speech recognition capabilities.

The fine-tuning approach here uses the standard Whisper encoder-decoder architecture
and trains on (audio, text) pairs to improve transcription accuracy on pronunciation-
varied speech data.
"""

from .whisper_finetuner import WhisperFineTuner
from .training_config import TrainingConfig
from .data_processor import SpeechOcean762DataProcessor

__all__ = [
    'WhisperFineTuner',
    'TrainingConfig', 
    'SpeechOcean762DataProcessor'
]