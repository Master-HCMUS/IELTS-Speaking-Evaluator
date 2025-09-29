"""
Fine-tuning package for Whisper models on SpeechOcean762 dataset.

This package provides functionality to fine-tune OpenAI Whisper models
for improved speech recognition and pronunciation assessment on the
SpeechOcean762 dataset.
"""

from .whisper_finetuner import WhisperFineTuner
from .data_processor import SpeechOcean762DataProcessor
from .training_config import TrainingConfig

__all__ = [
    'WhisperFineTuner',
    'SpeechOcean762DataProcessor', 
    'TrainingConfig'
]