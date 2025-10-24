"""
Fine-tuning package for Whisper models.

This package provides different approaches for fine-tuning Whisper models:

1. finetuning_STT: Standard Speech-to-Text fine-tuning on SpeechOcean762 dataset
   - Uses encoder-decoder architecture
   - Trains on (audio, text) pairs
   - Improves transcription accuracy

2. (Future) Other fine-tuning approaches can be added here, such as:
   - Pronunciation assessment fine-tuning
   - Multi-task learning approaches
   - Custom regression models
"""

# Import from STT fine-tuning submodule for backward compatibility
try:
    from .finetuning_STT import WhisperFineTuner, TrainingConfig, SpeechOcean762DataProcessor
except ImportError:
    # Fallback if import fails
    WhisperFineTuner = None
    TrainingConfig = None
    SpeechOcean762DataProcessor = None

__all__ = [
    'WhisperFineTuner',
    'SpeechOcean762DataProcessor', 
    'TrainingConfig'
]