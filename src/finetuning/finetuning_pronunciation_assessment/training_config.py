"""
Training configuration for pronunciation assessment fine-tuning.

This module provides configuration classes for training Whisper models with
pronunciation assessment capabilities, including multi-objective loss settings
and assessment-specific hyperparameters.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch


@dataclass
class PronunciationTrainingConfig:
    """
    Configuration class for pronunciation assessment training.
    
    This configuration handles both ASR and pronunciation assessment training
    parameters, including loss weights and granularity settings.
    """
    
    # Model configuration
    whisper_model_name: str = "openai/whisper-tiny"
    assessment_dropout: float = 0.1
    freeze_whisper_layers: int = 0
    
    # Training data configuration
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    include_transcription: bool = True  # Include ASR training
    
    # Audio preprocessing
    sampling_rate: int = 16000
    max_audio_length: float = 30.0
    normalize_audio: bool = True
    
    # Training parameters
    output_dir: str = "whisper_pronunciation_assessment"
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Gradient and optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    lr_scheduler: str = "linear"
    
    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_total_loss"
    greater_is_better: bool = False
    
    # Multi-objective loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'asr': 1.0,                    # ASR (transcription) loss
        'word_accuracy': 1.0,          # Word-level accuracy
        'word_stress': 0.5,            # Word-level stress (lower weight)
        'word_total': 1.0,             # Word-level total score
        'phone_accuracy': 1.0,         # Phone-level accuracy
        'utterance_accuracy': 1.0,     # Utterance-level accuracy
        'utterance_fluency': 1.0,      # Utterance-level fluency
        'utterance_prosodic': 1.0,     # Utterance-level prosodic
        'utterance_completeness': 0.1, # Lower weight (99.6% is 10)
        'utterance_total': 1.0         # Utterance-level total
    })
    
    # Training granularities
    train_word_level: bool = True
    train_phone_level: bool = True
    train_utterance_level: bool = True
    
    # Hardware and performance
    use_cuda: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 0
    
    # Whisper-specific parameters
    forced_decoder_ids: Optional[List] = None
    suppress_tokens: Optional[List] = None
    
    # Experiment tracking
    run_name: Optional[str] = None
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set default run name if not provided
        if self.run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = self.whisper_model_name.split("/")[-1]
            self.run_name = f"{model_short}_pronunciation_{timestamp}"
        
        # Validate loss weights
        self._validate_loss_weights()
        
        # Set CUDA availability
        if self.use_cuda and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.use_cuda = False
            self.fp16 = False
    
    def _validate_loss_weights(self):
        """Validate and adjust loss weights based on training granularities."""
        if not self.train_word_level:
            # Remove word-level loss weights
            word_keys = [k for k in self.loss_weights.keys() if k.startswith('word_')]
            for key in word_keys:
                self.loss_weights.pop(key, None)
        
        if not self.train_phone_level:
            # Remove phone-level loss weights
            phone_keys = [k for k in self.loss_weights.keys() if k.startswith('phone_')]
            for key in phone_keys:
                self.loss_weights.pop(key, None)
        
        if not self.train_utterance_level:
            # Remove utterance-level loss weights
            utterance_keys = [k for k in self.loss_weights.keys() if k.startswith('utterance_')]
            for key in utterance_keys:
                self.loss_weights.pop(key, None)
        
        if not self.include_transcription:
            # Remove ASR loss weight
            self.loss_weights.pop('asr', None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, file_path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @classmethod
    def load(cls, file_path: str) -> 'PronunciationTrainingConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def get_device(self) -> torch.device:
        """Get the training device."""
        if self.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def print_config(self):
        """Print configuration summary."""
        print("Pronunciation Assessment Training Configuration")
        print("=" * 50)
        print(f"Model: {self.whisper_model_name}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Training Granularities:")
        print(f"  - Word-level: {self.train_word_level}")
        print(f"  - Phone-level: {self.train_phone_level}")
        print(f"  - Utterance-level: {self.train_utterance_level}")
        print(f"  - Include Transcription: {self.include_transcription}")
        print(f"Training Parameters:")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Max Train Samples: {self.max_train_samples or 'All'}")
        print(f"  - Max Eval Samples: {self.max_eval_samples or 'All'}")
        print(f"Loss Weights:")
        for loss_name, weight in self.loss_weights.items():
            print(f"  - {loss_name}: {weight}")
        print(f"Device: {self.get_device()}")
        print("=" * 50)


# Preset configurations for different use cases

def get_quick_test_config() -> PronunciationTrainingConfig:
    """Configuration for quick testing with minimal data."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="pronunciation_quick_test",
        max_train_samples=100,
        max_eval_samples=50,
        num_epochs=1,
        batch_size=4,
        eval_steps=25,
        save_steps=50,
        logging_steps=10,
        warmup_steps=10,
        # Reduce assessment complexity for quick test
        train_phone_level=False,
        loss_weights={
            'asr': 1.0,
            'word_accuracy': 1.0,
            'word_total': 1.0,
            'utterance_accuracy': 1.0,
            'utterance_total': 1.0
        }
    )


def get_development_config() -> PronunciationTrainingConfig:
    """Configuration for development with moderate data."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="pronunciation_development",
        max_train_samples=1000,
        max_eval_samples=200,
        num_epochs=3,
        batch_size=8,
        eval_steps=100,
        save_steps=200,
        logging_steps=20,
        warmup_steps=100,
        # Include all granularities
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True
    )


def get_production_config() -> PronunciationTrainingConfig:
    """Configuration for production training with full dataset."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-base",
        output_dir="pronunciation_production",
        max_train_samples=None,  # Use all data
        max_eval_samples=None,   # Use all data
        num_epochs=10,           # Increased for better convergence
        batch_size=12,           # Slightly reduced for stability
        eval_batch_size=24,      # Proportionally adjusted
        learning_rate=3e-6,      # Lower for stable long training
        weight_decay=0.005,      # Reduced weight decay
        warmup_steps=2000,       # Longer warmup for large training
        eval_steps=250,          # More frequent evaluation
        save_steps=500,          # More frequent saving
        logging_steps=25,        # More frequent logging
        gradient_accumulation_steps=4,  # Better gradient estimates
        max_grad_norm=0.5,       # Stricter gradient clipping
        lr_scheduler="cosine",   # Better for long training
        save_total_limit=5,      # Keep more checkpoints
        # Full multi-granularity training
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True,
        # Optimized loss weights for long training
        loss_weights={
            'asr': 0.7,                    # Balanced ASR weight
            'word_accuracy': 1.0,
            'word_stress': 0.4,            # Slightly reduced
            'word_total': 1.0,
            'phone_accuracy': 1.5,         # Higher emphasis on phones
            'utterance_accuracy': 1.2,     # Increased utterance focus
            'utterance_fluency': 1.0,
            'utterance_prosodic': 0.8,     # Slightly reduced
            'utterance_completeness': 0.1, # Keep low (most samples are 10)
            'utterance_total': 1.3         # Higher total score emphasis
        }
    )


def get_transcription_only_config() -> PronunciationTrainingConfig:
    """Configuration for transcription-only training (ASR baseline)."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="pronunciation_asr_only",
        max_train_samples=1000,
        max_eval_samples=200,
        num_epochs=3,
        batch_size=16,
        # Only transcription training
        train_word_level=False,
        train_phone_level=False,
        train_utterance_level=False,
        include_transcription=True,
        loss_weights={
            'asr': 1.0  # Only ASR loss
        }
    )


def get_assessment_only_config() -> PronunciationTrainingConfig:
    """Configuration for assessment-only training (no transcription)."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="pronunciation_assessment_only",
        max_train_samples=1000,
        max_eval_samples=200,
        num_epochs=5,
        batch_size=8,
        # Only assessment training
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=False,  # No ASR training
        freeze_whisper_layers=4,      # Freeze more layers since no ASR
        loss_weights={
            'word_accuracy': 1.0,
            'word_stress': 0.5,
            'word_total': 1.0,
            'phone_accuracy': 1.0,
            'utterance_accuracy': 1.0,
            'utterance_fluency': 1.0,
            'utterance_prosodic': 1.0,
            'utterance_completeness': 0.1,
            'utterance_total': 1.0
        }
    )


def get_phone_focused_config() -> PronunciationTrainingConfig:
    """Configuration focused on phone-level assessment."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="pronunciation_phone_focused",
        max_train_samples=1000,
        max_eval_samples=200,
        num_epochs=4,
        batch_size=8,
        # Focus on phone-level
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True,
        loss_weights={
            'asr': 0.5,                    # Lower ASR weight
            'word_accuracy': 0.8,          # Lower word weight
            'word_total': 0.8,
            'phone_accuracy': 2.0,         # High phone weight
            'utterance_accuracy': 1.0,
            'utterance_total': 1.0
        }
    )