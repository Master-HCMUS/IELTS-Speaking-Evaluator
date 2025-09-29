"""
Simple script to load and fine-tune OpenAI Whisper-tiny model on SpeechOcean762 dataset.

This script provides an easy-to-use interface for fine-tuning Whisper models.
"""

import os
import sys
from pathlib import Path

# Add src to Python path and handle imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from finetuning.training_config import TrainingConfig, get_quick_test_config, get_development_config
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
    print("OpenAI Whisper-tiny Fine-tuning on SpeechOcean762 Dataset")
    print("========================================================")
    print()
    print("Choose training mode:")
    print("1. Quick Test (10 train samples, 5 eval samples, 1 epoch)")
    print("2. Development (1000 train samples, 200 eval samples, 3 epochs)")
    print("3. Custom (1000 train samples, 200 eval samples, 5 epochs)")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                run_quick_test()
                break
            elif choice == "2":
                run_development_training()
                break
            elif choice == "3":
                run_custom_training()
                break
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()