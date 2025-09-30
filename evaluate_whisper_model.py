#!/usr/bin/env python3
"""
Evaluate Whisper Model Script

Simple script to evaluate fine-tuned Whisper models for pronunciation assessment
against the SpeechOcean762 dataset using the same methodology as Azure Speech evaluation.

Usage:
    python evaluate_whisper_model.py --model-path ./whisper_development
    python evaluate_whisper_model.py --model-path ./whisper_production --max-samples 100
    python evaluate_whisper_model.py --model-path ./whisper_development --split validation
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from evaluation.whisper_evaluator import WhisperModelEvaluator
    from config_manager import ConfigManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Whisper model for pronunciation assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model in whisper_development folder
  python evaluate_whisper_model.py --model-path ./whisper_development
  
  # Evaluate with limited samples for quick testing
  python evaluate_whisper_model.py --model-path ./whisper_development --max-samples 50
  
  # Evaluate on validation split
  python evaluate_whisper_model.py --model-path ./whisper_production --split validation
  
  # Use specific device
  python evaluate_whisper_model.py --model-path ./whisper_development --device cuda
        """
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to fine-tuned Whisper model directory"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        choices=["test", "train", "validation"],
        help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        help="Maximum number of samples to evaluate (default: all samples)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run quick test with 10 samples"
    )
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save detailed results to file"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        print("Make sure the path points to a directory containing fine-tuned Whisper model files")
        return 1
    
    # Check if it looks like a valid model directory
    required_files = ["config.json", "pytorch_model.bin"]
    alternative_files = ["model.safetensors", "generation_config.json"]
    
    has_config = (model_path / "config.json").exists()
    has_model = any((model_path / f).exists() for f in required_files + alternative_files)
    
    if not (has_config and has_model):
        print(f"⚠️ Warning: {model_path} doesn't look like a valid Whisper model directory")
        print("Expected files: config.json and (pytorch_model.bin or model.safetensors)")
        print(f"Contents: {list(model_path.iterdir())}")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Override max_samples for quick test
    if args.quick_test:
        args.max_samples = 10
        print("🚀 Running quick test with 10 samples")
    
    print("=" * 80)
    print("🎯 WHISPER MODEL PRONUNCIATION EVALUATION")
    print("=" * 80)
    print(f"📁 Model path: {model_path}")
    print(f"📊 Dataset split: {args.split}")
    print(f"🔢 Max samples: {args.max_samples or 'all'}")
    print(f"💻 Device: {args.device}")
    print("=" * 80)
    
    try:
        # Initialize configuration manager (optional)
        config_manager = None
        try:
            config_manager = ConfigManager()
        except:
            print("ℹ️ Config manager not available (Azure Speech features disabled)")
        
        # Initialize evaluator
        print("🔧 Initializing Whisper model evaluator...")
        evaluator = WhisperModelEvaluator(str(model_path), config_manager)
        
        # Load dataset
        print(f"📥 Loading SpeechOcean762 dataset...")
        if not evaluator.load_dataset(split=args.split, max_samples=args.max_samples):
            print("❌ Failed to load dataset")
            return 1
        
        # Run evaluation
        print(f"\n🚀 Starting evaluation...")
        print("This may take a while depending on the number of samples...")
        
        metrics = evaluator.run_evaluation(
            max_samples=args.max_samples,
            save_results=not args.no_save
        )
        
        # Print summary
        evaluator.print_evaluation_summary(metrics)
        
        # Performance summary
        print(f"\n🎊 EVALUATION COMPLETED SUCCESSFULLY!")
        
        if not args.no_save:
            print(f"📁 Results saved to evaluation_results/ directory")
        
        # Quick interpretation for user
        avg_correlation = (
            metrics.accuracy_correlation + metrics.fluency_correlation +
            metrics.completeness_correlation + metrics.prosodic_correlation
        ) / 4
        
        print(f"\n📊 QUICK SUMMARY:")
        print(f"   Average correlation with expert scores: {avg_correlation:.3f}")
        print(f"   Success rate: {(metrics.successful_assessments/metrics.total_samples)*100:.1f}%")
        
        if avg_correlation > 0.7:
            print("   🎉 Excellent performance! Strong correlation with human experts.")
        elif avg_correlation > 0.5:
            print("   👍 Good performance! Moderate correlation with human experts.")
        elif avg_correlation > 0.3:
            print("   ⚠️ Fair performance. Model may benefit from additional fine-tuning.")
        else:
            print("   ❌ Poor performance. Model needs significant improvement.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)