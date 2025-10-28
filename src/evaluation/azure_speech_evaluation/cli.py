"""
CLI interface for running SpeechOcean762 dataset evaluation.

This module provides a command-line interface for evaluating Azure Speech
pronunciation assessment against the SpeechOcean762 dataset.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config_manager import ConfigManager
from evaluation.azure_speech_evaluation.core import SpeechOcean762Evaluator


def main():
    """Main CLI function for dataset evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Azure Speech pronunciation assessment against SpeechOcean762 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.evaluation.evaluate_dataset                    # Evaluate full test set
  python -m src.evaluation.evaluate_dataset --max-samples 50   # Evaluate 50 samples
  python -m src.evaluation.evaluate_dataset --split train      # Evaluate training set
  python -m src.evaluation.evaluate_dataset --no-save          # Don't save results
        """
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'train', 'validation'],
        help='Dataset split to evaluate (default: test)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )
    
    parser.add_argument(
        '--config-path',
        type=str,
        default='config/audio_config.json',
        help='Path to configuration file (default: config/audio_config.json)'
    )
    
    args = parser.parse_args()
    
    print("🎯 SpeechOcean762 Dataset Evaluation")
    print("=" * 50)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager(args.config_path)
        
        # Check if Azure Speech is configured
        if not config_manager.is_speech_configured():
            print("❌ Azure Speech service not configured!")
            print("\nPlease configure Azure Speech in your .env file:")
            print("   AZURE_SPEECH_KEY=your-speech-api-key")
            print("   AZURE_SPEECH_REGION=your-speech-region")
            print("   AZURE_SPEECH_LOCALE=en-US")
            sys.exit(1)
        
        # Initialize evaluator
        evaluator = SpeechOcean762Evaluator(config_manager)
        
        # Load dataset
        print(f"\n📥 Loading dataset (split: {args.split})...")
        if not evaluator.load_dataset(split=args.split, max_samples=args.max_samples):
            print("❌ Failed to load dataset")
            sys.exit(1)
        
        # Run evaluation
        print(f"\n🚀 Starting evaluation...")
        save_results = not args.no_save
        
        metrics = evaluator.run_evaluation(
            max_samples=args.max_samples,
            save_results=save_results
        )
        
        # Print summary
        evaluator.print_evaluation_summary(metrics)
        
        if save_results:
            print(f"\n📁 Results saved to 'evaluation_results/' directory")
        
        print(f"\n🎉 Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 Evaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()