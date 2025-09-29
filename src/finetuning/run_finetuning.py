"""
Simple script to load and fine-tune OpenAI Whisper-tiny model on SpeechOcean762 dataset.

This script provides an easy-to-use interface for fine-tuning Whisper models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to Python path and handle imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finetuning.training_config import (
    TrainingConfig, 
    get_quick_test_config, 
    get_development_config,
    get_production_config,
    get_large_model_config
)
from finetuning.whisper_finetuner import WhisperFineTuner


def run_quick_test():
    """Run a quick test with minimal data for testing setup."""
    print("🚀 Starting Quick Test Fine-tuning...")
    print("=" * 50)
    
    # Get quick test configuration
    config = get_quick_test_config()
    config.output_dir = str(script_dir / "models" / "whisper_quick_test")
    
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print()
    
    # Initialize and run fine-tuning
    fine_tuner = WhisperFineTuner(config)
    results = fine_tuner.train()
    
    print("\n🎉 Quick Test Completed!")
    print("=" * 50)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    print(f"WER: {results['eval_results'].get('eval_wer', 'N/A')}")
    print(f"BLEU Score: {results['eval_results'].get('eval_bleu', 'N/A')}")
    print(f"Character Accuracy: {results['eval_results'].get('eval_char_accuracy', 'N/A')}")
    
    return results


def run_development_training():
    """Run development training with moderate data for real fine-tuning."""
    print("🚀 Starting Development Fine-tuning...")
    print("=" * 50)
    
    # Get development configuration
    config = get_development_config()
    config.output_dir = str(script_dir / "models" / "whisper_development")
    
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print()
    
    # Initialize and run fine-tuning
    fine_tuner = WhisperFineTuner(config)
    results = fine_tuner.train()
    
    print("\n🎉 Development Training Completed!")
    print("=" * 50)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    print(f"WER: {results['eval_results'].get('eval_wer', 'N/A')}")
    print(f"BLEU Score: {results['eval_results'].get('eval_bleu', 'N/A')}")
    print(f"Character Accuracy: {results['eval_results'].get('eval_char_accuracy', 'N/A')}")
    
    return results


def run_production_training():
    """Run production training with optimized settings for best performance."""
    print("🚀 Starting Production Fine-tuning...")
    print("=" * 50)
    
    # Get production configuration
    config = get_production_config()
    config.output_dir = str(script_dir / "models" / "whisper_production")
    
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples if config.max_train_samples else 'All'}")
    print(f"Max Evaluation Samples: {config.max_eval_samples if config.max_eval_samples else 'All'}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Warmup Steps: {config.warmup_steps}")
    print()
    
    # Initialize and run fine-tuning
    fine_tuner = WhisperFineTuner(config)
    results = fine_tuner.train()
    
    print("\n🎉 Production Training Completed!")
    print("=" * 50)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    print(f"WER: {results['eval_results'].get('eval_wer', 'N/A')}")
    print(f"BLEU Score: {results['eval_results'].get('eval_bleu', 'N/A')}")
    print(f"Character Accuracy: {results['eval_results'].get('eval_char_accuracy', 'N/A')}")
    
    return results


def run_large_model_training():
    """Run training with larger Whisper model for better performance."""
    print("🚀 Starting Large Model Fine-tuning...")
    print("=" * 50)
    
    # Get large model configuration
    config = get_large_model_config()
    config.output_dir = str(script_dir / "models" / "whisper_large_model")
    
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples if config.max_train_samples else 'All'}")
    print(f"Max Evaluation Samples: {config.max_eval_samples if config.max_eval_samples else 'All'}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Warmup Steps: {config.warmup_steps}")
    print()
    
    print("⚠️  Warning: Large model training requires significant computational resources!")
    print("   - Estimated time: 4-8 hours")
    print("   - GPU memory: 8GB+ recommended")
    print("   - Disk space: 2GB+ for model storage")
    print()
    
    confirm = input("Do you want to continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Training cancelled.")
        return None
    
    # Initialize and run fine-tuning
    fine_tuner = WhisperFineTuner(config)
    results = fine_tuner.train()
    
    print("\n🎉 Large Model Training Completed!")
    print("=" * 50)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    print(f"WER: {results['eval_results'].get('eval_wer', 'N/A')}")
    print(f"BLEU Score: {results['eval_results'].get('eval_bleu', 'N/A')}")
    print(f"Character Accuracy: {results['eval_results'].get('eval_char_accuracy', 'N/A')}")
    
    return results


def run_custom_training():
    """Run custom training with user-specified parameters."""
    print("🚀 Starting Custom Fine-tuning...")
    print("=" * 50)
    
    # Create custom configuration
    config = TrainingConfig(
        model_name="openai/whisper-tiny",
        output_dir=str(script_dir / "models" / "whisper_custom"),
        batch_size=16,
        num_epochs=5,
        learning_rate=1e-5,
        max_train_samples=1000,  # Adjust as needed
        max_eval_samples=200,    # Adjust as needed
        warmup_steps=500,
        save_steps=100,
        eval_steps=100,
        logging_steps=10
    )
    
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples}")
    print(f"Max Evaluation Samples: {config.max_eval_samples}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print()
    
    # Initialize and run fine-tuning
    fine_tuner = WhisperFineTuner(config)
    results = fine_tuner.train()
    
    print("\n🎉 Custom Training Completed!")
    print("=" * 50)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    print(f"WER: {results['eval_results'].get('eval_wer', 'N/A')}")
    print(f"BLEU Score: {results['eval_results'].get('eval_bleu', 'N/A')}")
    print(f"Character Accuracy: {results['eval_results'].get('eval_char_accuracy', 'N/A')}")
    
    return results


def main():
    """Main function to run fine-tuning with user selection."""
    parser = argparse.ArgumentParser(
        description="Fine-tune OpenAI Whisper models on SpeechOcean762 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Modes:
  --quick-test     Quick test with minimal data (100 train, 50 eval, 1 epoch)
  --development    Development training (1000 train, 200 eval, 3 epochs)
  --production     Production training (all samples, whisper-small, 5 epochs)
  --large-model    Large model training (all samples, whisper-base, 3 epochs)
  --custom         Custom training (1000 train, 200 eval, 5 epochs)

Examples:
  python run_finetuning.py --quick-test
  python run_finetuning.py --production
  python run_finetuning.py --large-model
  python run_finetuning.py  # Interactive mode
        """
    )
    
    # Training mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run quick test with minimal data for testing setup"
    )
    mode_group.add_argument(
        "--development", 
        action="store_true",
        help="Run development training with moderate data"
    )
    mode_group.add_argument(
        "--production", 
        action="store_true",
        help="Run production training with optimized settings"
    )
    mode_group.add_argument(
        "--large-model", 
        action="store_true",
        help="Run training with larger Whisper model"
    )
    mode_group.add_argument(
        "--custom", 
        action="store_true",
        help="Run custom training with predefined parameters"
    )
    
    # Optional arguments for customization
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for the trained model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Whisper model name to use (e.g., openai/whisper-tiny, openai/whisper-small)"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        help="Maximum number of training samples to use"
    )
    parser.add_argument(
        "--max-eval-samples", 
        type=int,
        help="Maximum number of evaluation samples to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for training"
    )
    
    args = parser.parse_args()
    
    # If any training mode is specified via CLI, run it directly
    if args.quick_test:
        config = get_quick_test_config()
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            config.output_dir = str(script_dir / "models" / "whisper_quick_test")
        _apply_custom_args(config, args)
        _run_training("Quick Test", config)
        
    elif args.development:
        config = get_development_config()
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            config.output_dir = str(script_dir / "models" / "whisper_development")
        _apply_custom_args(config, args)
        _run_training("Development", config)
        
    elif args.production:
        config = get_production_config()
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            config.output_dir = str(script_dir / "models" / "whisper_production")
        _apply_custom_args(config, args)
        _run_training("Production", config)
        
    elif args.large_model:
        config = get_large_model_config()
        if args.output_dir:
            config.output_dir = args.output_dir
        else:
            config.output_dir = str(script_dir / "models" / "whisper_large_model")
        _apply_custom_args(config, args)
        
        print("⚠️  Warning: Large model training requires significant computational resources!")
        print("   - Estimated time: 4-8 hours")
        print("   - GPU memory: 8GB+ recommended")
        print("   - Disk space: 2GB+ for model storage")
        print()
        
        confirm = input("Do you want to continue? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Training cancelled.")
            return
        
        _run_training("Large Model", config)
        
    elif args.custom:
        config = TrainingConfig(
            model_name="openai/whisper-tiny",
            output_dir=str(script_dir / "models" / "whisper_custom"),
            batch_size=16,
            num_epochs=5,
            learning_rate=1e-5,
            max_train_samples=1000,
            max_eval_samples=200,
            warmup_steps=500,
            save_steps=100,
            eval_steps=100,
            logging_steps=10
        )
        if args.output_dir:
            config.output_dir = args.output_dir
        _apply_custom_args(config, args)
        _run_training("Custom", config)
        
    else:
        # No CLI arguments, run interactive mode
        _run_interactive_mode()


def _apply_custom_args(config, args):
    """Apply custom arguments to configuration."""
    if args.model_name:
        config.model_name = args.model_name
    if args.max_train_samples:
        config.max_train_samples = args.max_train_samples
    if args.max_eval_samples:
        config.max_eval_samples = args.max_eval_samples
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate


def _run_training(mode_name, config):
    """Run training with the given configuration."""
    print(f"🚀 Starting {mode_name} Fine-tuning...")
    print("=" * 50)
    
    print(f"Model: {config.model_name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Max Training Samples: {config.max_train_samples if config.max_train_samples else 'All'}")
    print(f"Max Evaluation Samples: {config.max_eval_samples if config.max_eval_samples else 'All'}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    if hasattr(config, 'warmup_steps'):
        print(f"Warmup Steps: {config.warmup_steps}")
    if hasattr(config, 'gradient_accumulation_steps') and config.gradient_accumulation_steps > 1:
        print(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print()
    
    # Initialize and run fine-tuning
    fine_tuner = WhisperFineTuner(config)
    results = fine_tuner.train()
    
    print(f"\n🎉 {mode_name} Training Completed!")
    print("=" * 50)
    print(f"Model saved to: {config.output_dir}")
    print(f"Training Loss: {results['train_results']['training_loss']:.4f}")
    print(f"WER: {results['eval_results'].get('eval_wer', 'N/A')}")
    print(f"BLEU Score: {results['eval_results'].get('eval_bleu', 'N/A')}")
    print(f"Character Accuracy: {results['eval_results'].get('eval_char_accuracy', 'N/A')}")
    
    return results


def _run_interactive_mode():
    """Run the original interactive mode."""
    print("OpenAI Whisper-tiny Fine-tuning on SpeechOcean762 Dataset")
    print("========================================================")
    print()
    print("Choose training mode:")
    print("1. Quick Test (100 train samples, 50 eval samples, 1 epoch)")
    print("2. Development (1000 train samples, 200 eval samples, 3 epochs)")
    print("3. Production (all samples, whisper-small, 5 epochs)")
    print("4. Large Model (all samples, whisper-base, 3 epochs)")
    print("5. Custom (1000 train samples, 200 eval samples, 5 epochs)")
    print("6. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
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
                run_large_model_training()
                break
            elif choice == "5":
                run_custom_training()
                break
            elif choice == "6":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()