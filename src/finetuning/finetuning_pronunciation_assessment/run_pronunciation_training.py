"""
Main script for running pronunciation assessment fine-tuning.

This script provides an easy-to-use interface for training Whisper models
with pronunciation assessment capabilities using different preset configurations.
"""

import os
import sys
import argparse
from pathlib import Path

# Disable torchcodec early to avoid FFmpeg issues
os.environ["DATASETS_DISABLE_TORCHCODEC"] = "1"

# Add src to Python path and handle imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finetuning.finetuning_pronunciation_assessment.training_config import (
    PronunciationTrainingConfig,
    get_quick_test_config,
    get_development_config,
    get_production_config,
    get_transcription_only_config,
    get_assessment_only_config,
    get_phone_focused_config
)
from finetuning.finetuning_pronunciation_assessment.trainer import PronunciationAssessmentTrainer


def run_quick_test():
    """Run a quick test with minimal data for testing setup."""
    print("🚀 Starting Quick Test Pronunciation Assessment Fine-tuning...")
    print("=" * 60)
    
    # Get quick test configuration
    config = get_quick_test_config()
    config.output_dir = str(script_dir / "models" / "pronunciation_quick_test")
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Training Granularities:")
    print(f"  - Word-level: {config.train_word_level}")
    print(f"  - Phone-level: {config.train_phone_level}")
    print(f"  - Utterance-level: {config.train_utterance_level}")
    print(f"  - Include Transcription: {config.include_transcription}")
    print()
    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Quick Test Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    # Print evaluation losses
    eval_results = results.get('eval_results', {})
    print("Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    return results


def run_development_training():
    """Run development training with moderate data for real fine-tuning."""
    print("🚀 Starting Development Pronunciation Assessment Fine-tuning...")
    print("=" * 60)
    
    # Get development configuration
    config = get_development_config()
    config.output_dir = str(script_dir / "models" / "pronunciation_development")
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Training Granularities:")
    print(f"  - Word-level: {config.train_word_level}")
    print(f"  - Phone-level: {config.train_phone_level}")
    print(f"  - Utterance-level: {config.train_utterance_level}")
    print(f"  - Include Transcription: {config.include_transcription}")
    print()
    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Development Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    # Print evaluation losses
    eval_results = results.get('eval_results', {})
    print("Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    return results


def run_production_training():
    """Run production training with optimized settings for best performance."""
    print("🚀 Starting Production Pronunciation Assessment Fine-tuning...")
    print("=" * 60)
    
    # Get production configuration
    config = get_production_config()
    config.output_dir = str(script_dir / "models" / "pronunciation_production")
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples if config.max_train_samples else 'All'}")
    print(f"Max Evaluation Samples: {config.max_eval_samples if config.max_eval_samples else 'All'}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Warmup Steps: {config.warmup_steps}")
    print(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print(f"Training Granularities:")
    print(f"  - Word-level: {config.train_word_level}")
    print(f"  - Phone-level: {config.train_phone_level}")
    print(f"  - Utterance-level: {config.train_utterance_level}")
    print(f"  - Include Transcription: {config.include_transcription}")
    print()
    
    print("⚠️  Warning: Production training requires significant computational resources!")
    print("   - Estimated time: 6-12 hours")
    print("   - GPU memory: 12GB+ recommended")
    print("   - Disk space: 3GB+ for model storage")
    print()

    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Production Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    # Print evaluation losses
    eval_results = results.get('eval_results', {})
    print("Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    return results


def run_transcription_only():
    """Run transcription-only training (ASR baseline)."""
    print("🚀 Starting Transcription-Only Fine-tuning (ASR Baseline)...")
    print("=" * 60)
    
    # Get transcription-only configuration
    config = get_transcription_only_config()
    config.output_dir = str(script_dir / "models" / "pronunciation_asr_only")
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Training Mode: ASR Only (Baseline)")
    print()
    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Transcription-Only Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    return results


def run_assessment_only():
    """Run assessment-only training (no transcription)."""
    print("🚀 Starting Assessment-Only Fine-tuning...")
    print("=" * 60)
    
    # Get assessment-only configuration
    config = get_assessment_only_config()
    config.output_dir = str(script_dir / "models" / "pronunciation_assessment_only")
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Frozen Whisper Layers: {config.freeze_whisper_layers}")
    print(f"Training Mode: Assessment Only (No Transcription)")
    print()
    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Assessment-Only Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    return results


def run_phone_focused():
    """Run phone-focused training."""
    print("🚀 Starting Phone-Focused Fine-tuning...")
    print("=" * 60)
    
    # Get phone-focused configuration
    config = get_phone_focused_config()
    config.output_dir = str(script_dir / "models" / "pronunciation_phone_focused")
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Training Mode: Phone-Level Focused")
    print("Loss Weights:")
    for loss_name, weight in config.loss_weights.items():
        print(f"  {loss_name}: {weight}")
    print()
    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Phone-Focused Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    return results


def run_custom_training():
    """Run custom training with user-specified parameters."""
    print("🚀 Starting Custom Pronunciation Assessment Fine-tuning...")
    print("=" * 60)
    
    # Create custom configuration
    config = PronunciationTrainingConfig(
        whisper_model_name="openai/whisper-tiny",
        output_dir=str(script_dir / "models" / "pronunciation_custom"),
        batch_size=8,
        num_epochs=3,
        learning_rate=1e-5,
        max_train_samples=1000,
        max_eval_samples=200,
        warmup_steps=100,
        eval_steps=100,
        save_steps=200,
        logging_steps=20,
        # Multi-granularity training
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        include_transcription=True
    )
    
    print(f"Model: {config.whisper_model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Training Granularities:")
    print(f"  - Word-level: {config.train_word_level}")
    print(f"  - Phone-level: {config.train_phone_level}")
    print(f"  - Utterance-level: {config.train_utterance_level}")
    print(f"  - Include Transcription: {config.include_transcription}")
    print()
    
    # Initialize and run fine-tuning
    trainer = PronunciationAssessmentTrainer(config)
    results = trainer.train()
    
    print("\n🎉 Custom Training Completed!")
    print("=" * 60)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    
    return results


def main():
    """Main function to run pronunciation assessment fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper models for pronunciation assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  --quick-test         Quick test with minimal data (100 train, 50 eval, 1 epoch)
  --development        Development training (1000 train, 200 eval, 3 epochs) 
  --production         Production training (all samples, whisper-small, 5 epochs)
  --transcription-only ASR baseline (transcription only, no assessment)
  --assessment-only    Assessment only (no transcription training)
  --phone-focused      Phone-level focused training
  --custom             Custom training with balanced settings

Examples:
  python run_pronunciation_training.py --quick-test
  python run_pronunciation_training.py --production
  python run_pronunciation_training.py --phone-focused
  python run_pronunciation_training.py  # Interactive mode
        """
    )
    
    # Training mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal data"
    )
    mode_group.add_argument(
        "--development",
        action="store_true", 
        help="Run development training with moderate data"
    )
    mode_group.add_argument(
        "--production",
        action="store_true",
        help="Run production training with full dataset"
    )
    mode_group.add_argument(
        "--transcription-only",
        action="store_true",
        help="Run transcription-only training (ASR baseline)"
    )
    mode_group.add_argument(
        "--assessment-only",
        action="store_true",
        help="Run assessment-only training (no transcription)"
    )
    mode_group.add_argument(
        "--phone-focused",
        action="store_true",
        help="Run phone-level focused training"
    )
    mode_group.add_argument(
        "--custom",
        action="store_true",
        help="Run custom training with balanced settings"
    )
    
    args = parser.parse_args()
    
    # If any training mode is specified via CLI, run it directly
    if args.quick_test:
        run_quick_test()
    elif args.development:
        run_development_training()
    elif args.production:
        run_production_training()
    elif args.transcription_only:
        run_transcription_only()
    elif args.assessment_only:
        run_assessment_only()
    elif args.phone_focused:
        run_phone_focused()
    elif args.custom:
        run_custom_training()
    else:
        # No CLI arguments, run interactive mode
        _run_interactive_mode()


def _run_interactive_mode():
    """Run the interactive mode for training selection."""
    print("Whisper Pronunciation Assessment Fine-tuning")
    print("=" * 60)
    print()
    print("Choose training mode:")
    print("1. Quick Test (100 train, 50 eval, 1 epoch, word+utterance level)")
    print("2. Development (1000 train, 200 eval, 3 epochs, all granularities)")
    print("3. Production (all samples, whisper-small, 5 epochs, all granularities)")
    print("4. Transcription-Only (ASR baseline, no assessment)")
    print("5. Assessment-Only (no transcription, assessment heads only)")
    print("6. Phone-Focused (emphasis on phone-level assessment)")
    print("7. Custom (balanced multi-granularity training)")
    print("8. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                run_quick_test()
                break
            elif choice == "2":
                run_development_training()
                break
            elif choice == "3":
                run_production_training()
                break
            elif choice == "4":
                run_transcription_only()
                break
            elif choice == "5":
                run_assessment_only()
                break
            elif choice == "6":
                run_phone_focused()
                break
            elif choice == "7":
                run_custom_training()
                break
            elif choice == "8":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, 7, or 8.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()