#!/usr/bin/env python3
"""
Standalone Whisper Model Evaluation CLI

Simple, self-contained script to evaluate fine-tuned Whisper models.
No dependencies on the Azure Speech evaluation framework.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Whisper model for pronunciation assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model with quick test
  python evaluate_whisper_standalone.py --model-path "src/finetuning/models/whisper_development" --quick-test
  
  # Evaluate with specific number of samples
  python evaluate_whisper_standalone.py --model-path "src/finetuning/models/whisper_development" --max-samples 50
  
  # Evaluate on validation split
  python evaluate_whisper_standalone.py --model-path "src/finetuning/models/whisper_development" --split validation
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
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run quick test with 10 samples"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Import the standalone evaluator
    try:
        from .core import StandaloneWhisperModelEvaluator
    except ImportError:
        # Fallback for direct execution
        try:
            from core import StandaloneWhisperModelEvaluator
        except ImportError as e:
            print(f"❌ Failed to import standalone evaluator: {e}")
            print("Make sure core.py is in the same directory")
            return 1
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return 1
    
    # Override max_samples for quick test
    if args.quick_test:
        args.max_samples = 10
        print("🚀 Running quick test with 10 samples")
    
    print("=" * 80)
    print("🎯 STANDALONE WHISPER MODEL PRONUNCIATION EVALUATION")
    print("=" * 80)
    print(f"📁 Model path: {model_path}")
    print(f"📊 Dataset split: {args.split}")
    print(f"🔢 Max samples: {args.max_samples or 'all'}")
    print(f"💻 Device: {args.device}")
    print("=" * 80)
    
    try:
        # Initialize evaluator
        print("🔧 Initializing Whisper model evaluator...")
        evaluator = StandaloneWhisperModelEvaluator(str(model_path))
        print("✅ Evaluator initialized")
        
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
            save_results=True
        )
        
        # Print summary
        evaluator.print_evaluation_summary(metrics)
        
        # Performance summary
        print(f"\n🎊 EVALUATION COMPLETED SUCCESSFULLY!")
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