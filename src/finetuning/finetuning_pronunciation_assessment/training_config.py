"""
Training configuration for Whisper pronunciation assessment fine-tuning.

Supports multiple training modes with predefined configurations:
- Quick Test: Fast iteration and debugging
- Development: Experimentation with reasonable dataset size
- Production: Full training on complete dataset
- Specialized: Transcription-only, assessment-only, phone-focused modes
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PronunciationTrainingConfig:
    """Configuration for pronunciation assessment training."""
    
    # Model configuration
    whisper_model_name: str = "openai/whisper-tiny"
    output_dir: str = "models/pronunciation_development"
    
    # Dataset configuration
    max_train_samples: Optional[int] = 1000
    max_eval_samples: Optional[int] = 200
    
    # Training hyperparameters
    batch_size: int = 4
    eval_batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Audio configuration
    sampling_rate: int = 16000
    max_audio_length: float = 30.0
    
    # Training granularities
    train_word_level: bool = True
    train_phone_level: bool = True
    train_utterance_level: bool = True
    include_transcription: bool = True
    
    # Aliases for trainer compatibility
    @property
    def use_word_level_assessment(self) -> bool:
        return self.train_word_level
    
    @property
    def use_phone_level_assessment(self) -> bool:
        return self.train_phone_level
    
    @property
    def use_utterance_level_assessment(self) -> bool:
        return self.train_utterance_level
    
    @property
    def use_transcription(self) -> bool:
        return self.include_transcription
    
    # Loss weights for multi-objective training
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'transcription': 1.0,          # Transcription loss (decoder)
        'word_accuracy': 1.0,          # Word-level accuracy
        'word_stress': 0.5,            # Word-level stress
        'word_total': 1.0,             # Word-level total
        'phone_accuracy': 1.0,         # Phone-level accuracy
        'utterance_accuracy': 1.0,     # Utterance accuracy
        'utterance_fluency': 1.0,      # Utterance fluency
        'utterance_prosodic': 1.0,     # Utterance prosodic
        'utterance_completeness': 0.1, # Lower weight (99.6% is 10)
        'utterance_total': 1.0         # Utterance total
    })
    
    # Optimization
    save_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    warmup_ratio: float = 0.1  # 10% of total steps for warmup
    
    # Device configuration
    device: str = "cuda"
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def get_quick_test_config() -> PronunciationTrainingConfig:
    """Get configuration for quick test mode."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="models/pronunciation_quick_test",
        max_train_samples=100,
        max_eval_samples=50,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-4,
        train_phone_level=False,  # Skip phone level for speed
        loss_weights={
            'asr': 1.0,
            'word_accuracy': 1.0,
            'word_stress': 0.5,
            'word_total': 1.0,
            'utterance_accuracy': 1.0,
            'utterance_fluency': 1.0,
            'utterance_prosodic': 1.0,
            'utterance_completeness': 0.1,
            'utterance_total': 1.0
        }
    )


def get_development_config() -> PronunciationTrainingConfig:
    """Get configuration for development mode."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="models/pronunciation_development",
        max_train_samples=1000,
        max_eval_samples=200,
        batch_size=4,
        num_epochs=3,
        learning_rate=1e-5,
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True,
        loss_weights={
            'asr': 1.0,
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


def get_production_config() -> PronunciationTrainingConfig:
    """Get configuration for production mode."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-base",
        output_dir="models/pronunciation_production",
        max_train_samples=None,  # Use full dataset
        max_eval_samples=None,
        batch_size=8,
        eval_batch_size=16,
        num_epochs=30,
        learning_rate=5e-6,
        warmup_steps=1000,
        gradient_accumulation_steps=2,
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True,
        loss_weights={
            'asr': 1.0,
            'word_accuracy': 1.2,
            'word_stress': 0.6,
            'word_total': 1.0,
            'phone_accuracy': 1.2,
            'utterance_accuracy': 1.0,
            'utterance_fluency': 1.0,
            'utterance_prosodic': 1.0,
            'utterance_completeness': 0.1,
            'utterance_total': 1.0
        }
    )


def get_transcription_only_config() -> PronunciationTrainingConfig:
    """Get configuration for transcription-only baseline."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="models/pronunciation_transcription_only",
        max_train_samples=1000,
        max_eval_samples=200,
        batch_size=4,
        num_epochs=3,
        learning_rate=1e-5,
        train_word_level=False,
        train_phone_level=False,
        train_utterance_level=False,
        include_transcription=True,
        loss_weights={
            'asr': 1.0,
            'word_accuracy': 0.0,
            'word_stress': 0.0,
            'word_total': 0.0,
            'phone_accuracy': 0.0,
            'utterance_accuracy': 0.0,
            'utterance_fluency': 0.0,
            'utterance_prosodic': 0.0,
            'utterance_completeness': 0.0,
            'utterance_total': 0.0
        }
    )


def get_assessment_only_config() -> PronunciationTrainingConfig:
    """Get configuration for assessment-only training."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="models/pronunciation_assessment_only",
        max_train_samples=1000,
        max_eval_samples=200,
        batch_size=4,
        num_epochs=3,
        learning_rate=1e-5,
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=False,
        loss_weights={
            'asr': 0.0,
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
    """Get configuration for phone-focused training."""
    return PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir="models/pronunciation_phone_focused",
        max_train_samples=1000,
        max_eval_samples=200,
        batch_size=4,
        num_epochs=3,
        learning_rate=1e-5,
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True,
        loss_weights={
            'asr': 0.8,
            'word_accuracy': 0.8,
            'word_stress': 0.4,
            'word_total': 0.8,
            'phone_accuracy': 1.5,  # Higher weight for phones
            'utterance_accuracy': 0.8,
            'utterance_fluency': 0.8,
            'utterance_prosodic': 0.8,
            'utterance_completeness': 0.1,
            'utterance_total': 0.8
        }
    )


# Mapping of mode names to config getters
TRAINING_MODES = {
    'quick-test': get_quick_test_config,
    'development': get_development_config,
    'production': get_production_config,
    'transcription-only': get_transcription_only_config,
    'assessment-only': get_assessment_only_config,
    'phone-focused': get_phone_focused_config,
}


def get_config(mode: str = 'development') -> PronunciationTrainingConfig:
    """
    Get training configuration by mode name.
    
    Args:
        mode: Training mode name
        
    Returns:
        PronunciationTrainingConfig instance
        
    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in TRAINING_MODES:
        raise ValueError(
            f"Unknown training mode '{mode}'. "
            f"Available modes: {', '.join(TRAINING_MODES.keys())}"
        )
    
    return TRAINING_MODES[mode]()
