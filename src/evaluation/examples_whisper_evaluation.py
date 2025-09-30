"""
Whisper Model Evaluation Example

This script demonstrates how to evaluate fine-tuned Whisper models for 
pronunciation assessment using the SpeechOcean762 dataset.

Example usage scenarios:
1. Evaluate a fine-tuned model against expert annotations
2. Compare different fine-tuned models
3. Quick testing with limited samples
"""

import sys
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
if current_dir.name == "evaluation":
    # We're in src/evaluation, need to go up to src
    src_dir = current_dir.parent
    sys.path.insert(0, str(src_dir))
else:
    # We're in project root
    sys.path.insert(0, str(current_dir / "src"))

from evaluation.whisper_evaluator import WhisperModelEvaluator
from config_manager import ConfigManager


def evaluate_model_example(model_path: str, max_samples: int = 50):
    """
    Example function showing how to evaluate a Whisper model.
    
    Args:
        model_path: Path to fine-tuned Whisper model
        max_samples: Number of samples to evaluate (for quick testing)
    """
    print("🎯 Whisper Model Evaluation Example")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        print(f"🔧 Initializing evaluator for model: {model_path}")
        evaluator = WhisperModelEvaluator(model_path)
        
        # Load dataset
        print(f"📥 Loading SpeechOcean762 dataset (max {max_samples} samples)...")
        if not evaluator.load_dataset(split="test", max_samples=max_samples):
            print("❌ Failed to load dataset")
            return
        
        # Run evaluation
        print("🚀 Running evaluation...")
        metrics = evaluator.run_evaluation(max_samples=max_samples)
        
        # Print results
        evaluator.print_evaluation_summary(metrics)
        
        # Show some sample results
        print("\n📝 Sample Results:")
        successful_results = [r for r in evaluator.evaluation_results if r.success]
        for i, result in enumerate(successful_results[:3]):  # Show first 3 results
            print(f"\nSample {i+1}:")
            print(f"  Text: {result.text}")
            print(f"  Predicted: {result.predicted_text}")
            print(f"  WER: {result.word_error_rate:.3f}")
            print(f"  Expert Accuracy: {result.expert_scores['accuracy']:.1f}")
            print(f"  Whisper Accuracy: {result.pronunciation_scores['accuracy']:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def compare_models_example(model_paths: list, max_samples: int = 30):
    """
    Example function showing how to compare multiple Whisper models.
    
    Args:
        model_paths: List of paths to fine-tuned Whisper models
        max_samples: Number of samples to evaluate per model
    """
    print("🎯 Whisper Model Comparison Example")
    print("=" * 50)
    
    results = {}
    
    for model_path in model_paths:
        try:
            print(f"\n🔧 Evaluating model: {model_path}")
            evaluator = WhisperModelEvaluator(model_path)
            
            if not evaluator.load_dataset(split="test", max_samples=max_samples):
                print(f"❌ Failed to load dataset for {model_path}")
                continue
            
            metrics = evaluator.run_evaluation(max_samples=max_samples, save_results=False)
            
            results[Path(model_path).name] = {
                'accuracy_correlation': metrics.accuracy_correlation,
                'fluency_correlation': metrics.fluency_correlation,
                'completeness_correlation': metrics.completeness_correlation,
                'prosodic_correlation': metrics.prosodic_correlation,
                'success_rate': metrics.successful_assessments / metrics.total_samples,
                'avg_correlation': (
                    metrics.accuracy_correlation + metrics.fluency_correlation +
                    metrics.completeness_correlation + metrics.prosodic_correlation
                ) / 4
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {model_path}: {e}")
    
    # Print comparison
    print("\n📊 MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Avg Corr':<10} {'Accuracy':<10} {'Fluency':<10} {'Complete':<10} {'Prosodic':<10} {'Success%':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['avg_correlation']:<10.3f} "
              f"{metrics['accuracy_correlation']:<10.3f} "
              f"{metrics['fluency_correlation']:<10.3f} "
              f"{metrics['completeness_correlation']:<10.3f} "
              f"{metrics['prosodic_correlation']:<10.3f} "
              f"{metrics['success_rate']*100:<10.1f}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['avg_correlation'])
        print(f"\n🏆 Best performing model: {best_model[0]} "
              f"(avg correlation: {best_model[1]['avg_correlation']:.3f})")


def quick_test_example():
    """Quick test with a small number of samples."""
    print("🚀 Quick Test Example")
    print("=" * 30)
    
    # Check if we have any fine-tuned models
    possible_paths = [
        "./whisper_development",
        "./whisper_production", 
        "./src/finetuning/whisper_finetuned",
        "./whisper_finetuned"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path:
        print(f"Found model at: {model_path}")
        evaluate_model_example(model_path, max_samples=10)
    else:
        print("❌ No fine-tuned model found in expected locations:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\nTo create a fine-tuned model, run:")
        print("   python src/finetuning/run_finetuning.py --quick-test")


def main():
    """Main example runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whisper Model Evaluation Examples")
    parser.add_argument("--mode", choices=["single", "compare", "quick"], default="quick",
                       help="Evaluation mode")
    parser.add_argument("--model-path", type=str, help="Path to model for single evaluation")
    parser.add_argument("--model-paths", nargs="+", help="Paths to models for comparison")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to evaluate")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.model_path:
            print("❌ --model-path required for single mode")
            return
        evaluate_model_example(args.model_path, args.max_samples)
    
    elif args.mode == "compare":
        if not args.model_paths:
            print("❌ --model-paths required for compare mode")
            return
        compare_models_example(args.model_paths, args.max_samples)
    
    else:  # quick mode
        quick_test_example()


if __name__ == "__main__":
    main()