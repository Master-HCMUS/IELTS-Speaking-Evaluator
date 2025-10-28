#!/usr/bin/env python3
"""
Multi-Objective Whisper Model Evaluation CLI

Command-line interface for evaluating fine-tuned multi-objective Whisper models
(with pronunciation assessment heads) against the SpeechOcean762 dataset.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate multi-objective Whisper model for pronunciation assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with quick test (10 samples)
  python -m src.evaluation.multi_objective_whisper_evaluation.cli \\
      --model-path "src/finetuning/finetuning_pronunciation_assessment/models/kaggle" \\
      --quick-test
  
  # Evaluate with specific number of samples
  python -m src.evaluation.multi_objective_whisper_evaluation.cli \\
      --model-path "src/finetuning/finetuning_pronunciation_assessment/models/kaggle" \\
      --max-samples 100
  
  # Evaluate on validation split
  python -m src.evaluation.multi_objective_whisper_evaluation.cli \\
      --model-path "src/finetuning/finetuning_pronunciation_assessment/models/kaggle" \\
      --split validation --max-samples 50
        """
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to fine-tuned multi-objective Whisper model directory"
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
    
    from .core import MultiObjectiveWhisperModelEvaluator

    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model path does not exist: {model_path}")
        return 1
    
    # Override max_samples for quick test
    if args.quick_test:
        args.max_samples = 10
        print("[INFO] Running quick test with 10 samples")
    
    print("=" * 80)
    print("MULTI-OBJECTIVE WHISPER MODEL PRONUNCIATION EVALUATION")
    print("=" * 80)
    print(f"[MODEL] {model_path}")
    print(f"[SPLIT] {args.split}")
    print(f"[SAMPLES] {args.max_samples or 'all'}")
    print(f"[DEVICE] {args.device}")
    print("=" * 80)
    
    try:
        # Initialize evaluator
        print("[*] Initializing multi-objective Whisper model evaluator...")
        evaluator = MultiObjectiveWhisperModelEvaluator(str(model_path))
        print("[OK] Evaluator initialized")
        
        # Load dataset
        print(f"[*] Loading SpeechOcean762 dataset...")
        if not evaluator.load_dataset(split=args.split, max_samples=args.max_samples):
            print("[ERROR] Failed to load dataset")
            return 1
        
        # Run evaluation
        print(f"\n[*] Starting evaluation...")
        print("[*] This may take a while depending on the number of samples...")
        
        metrics = evaluator.run_evaluation(
            max_samples=args.max_samples,
            save_results=True
        )
        
        # Print summary
        evaluator.print_evaluation_summary(metrics)
        
        # Performance summary
        print(f"\n[OK] EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"[*] Results saved to evaluation_results/ directory")
        
        # Quick interpretation
        avg_correlation = (
            metrics.accuracy_correlation + metrics.fluency_correlation +
            metrics.completeness_correlation + metrics.prosodic_correlation
        ) / 4
        
        print(f"\n[SUMMARY]")
        print(f"   Average assessment correlation: {avg_correlation:.3f}")
        print(f"   Transcription WER: {metrics.wer:.3f}")
        print(f"   Success rate: {(metrics.successful_assessments/metrics.total_samples)*100:.1f}%")
        
        if avg_correlation > 0.7:
            print("   [RESULT] Excellent performance! Strong correlation with human experts.")
        elif avg_correlation > 0.5:
            print("   [RESULT] Good performance! Moderate correlation with human experts.")
        elif avg_correlation > 0.3:
            print("   [RESULT] Fair performance. Model may benefit from additional fine-tuning.")
        else:
            print("   [RESULT] Poor performance. Model needs significant improvement.")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[*] Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
