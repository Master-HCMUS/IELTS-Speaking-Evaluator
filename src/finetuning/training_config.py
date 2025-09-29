"""
Training configuration for Whisper fine-tuning on SpeechOcean762 dataset.

This module defines configuration classes and constants for training
Whisper models on pronunciation assessment tasks.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration class for Whisper fine-tuning training."""
    
    # Model configuration
    model_name: str = "openai/whisper-tiny"
    cache_dir: Optional[str] = None
    
    # Dataset configuration
    dataset_name: str = "mispeech/speechocean762"
    train_split: str = "train"
    eval_split: str = "test"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # Audio preprocessing
    sampling_rate: int = 16000
    max_audio_length: float = 30.0  # seconds
    normalize_audio: bool = True
    
    # Training hyperparameters
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization
    optimizer: str = "adamw"
    lr_scheduler: str = "linear"
    
    # Evaluation and logging
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_wer"
    greater_is_better: bool = False
    
    # Output configuration
    output_dir: str = "whisper_finetuned"
    run_name: Optional[str] = None
    
    # Hardware configuration
    use_cuda: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 0  # Set to 0 for Windows compatibility
    
    # Special tokens and processing
    forced_decoder_ids: Optional[Dict[str, Any]] = None
    suppress_tokens: Optional[list] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.run_name is None:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_short = self.model_name.split("/")[-1]
            self.run_name = f"{model_short}_speechocean762_{timestamp}"
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Predefined configurations for different use cases
def get_quick_test_config() -> TrainingConfig:
    """Get configuration for quick testing with minimal resources."""
    return TrainingConfig(
        model_name="openai/whisper-tiny",
        batch_size=4,
        eval_batch_size=8,
        num_epochs=1,
        max_train_samples=100,
        max_eval_samples=50,
        eval_steps=50,
        save_steps=100,
        logging_steps=25,
        output_dir="whisper_quick_test"
    )


def get_development_config() -> TrainingConfig:
    """Get configuration for development and experimentation."""
    return TrainingConfig(
        model_name="openai/whisper-tiny",
        batch_size=8,
        eval_batch_size=16,
        num_epochs=3,
        max_train_samples=1000,
        max_eval_samples=200,
        eval_steps=200,
        save_steps=500,
        output_dir="whisper_development"
    )


def get_production_config() -> TrainingConfig:
    """Get configuration for production fine-tuning."""
    return TrainingConfig(
        model_name="openai/whisper-small",
        batch_size=16,
        eval_batch_size=32,
        num_epochs=5,
        learning_rate=5e-6,
        warmup_steps=1000,
        eval_steps=1000,
        save_steps=2000,
        output_dir="whisper_production"
    )


def get_large_model_config() -> TrainingConfig:
    """Get configuration for large model fine-tuning."""
    return TrainingConfig(
        model_name="openai/whisper-base",
        batch_size=4,
        eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_epochs=3,
        learning_rate=1e-6,
        warmup_steps=2000,
        eval_steps=2000,
        save_steps=4000,
        fp16=True,
        output_dir="whisper_large_model"
    )