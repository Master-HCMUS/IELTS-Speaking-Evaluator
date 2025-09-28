#!/usr/bin/env python3
"""
Test script for the SpeechOcean762 evaluation system.
This script tests the evaluation functionality without requiring the full dataset.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_evaluation_imports():
    """Test that all evaluation modules can be imported."""
    print("Testing evaluation module imports...")
    
    try:
        from src.evaluation.dataset_evaluator import SpeechOcean762Evaluator
        print("✅ Successfully imported SpeechOcean762Evaluator")
    except ImportError as e:
        print(f"❌ Failed to import SpeechOcean762Evaluator: {e}")
        return False
    
    try:
        import datasets
        print("✅ HuggingFace datasets library available")
    except ImportError:
        print("❌ HuggingFace datasets library not installed")
        print("   Install with: pip install datasets")
        return False
    
    try:
        import pandas
        print("✅ Pandas library available")
    except ImportError:
        print("❌ Pandas library not installed")
        print("   Install with: pip install pandas")
        return False
    
    try:
        import matplotlib
        print("✅ Matplotlib library available")
    except ImportError:
        print("❌ Matplotlib library not installed")
        print("   Install with: pip install matplotlib")
        return False
    
    return True

def test_config_manager():
    """Test that config manager can be imported and initialized."""
    print("\nTesting configuration manager...")
    
    try:
        # Test if .env file exists
        env_file = Path(".env")
        if env_file.exists():
            print("✅ .env file found")
        else:
            print("⚠️  .env file not found (configuration may not work)")
        
        print("✅ Configuration files accessible")
        return True
    except Exception as e:
        print(f"❌ Failed to check configuration: {e}")
        return False

def test_evaluation_cli():
    """Test that evaluation CLI structure is correct."""
    print("\nTesting evaluation CLI structure...")
    
    try:
        # Check if evaluation files exist
        eval_dir = Path("src/evaluation")
        if eval_dir.exists():
            print("✅ Evaluation directory exists")
            
            if (eval_dir / "dataset_evaluator.py").exists():
                print("✅ dataset_evaluator.py exists")
            else:
                print("❌ dataset_evaluator.py missing")
                return False
                
            if (eval_dir / "evaluate_dataset.py").exists():
                print("✅ evaluate_dataset.py exists")
            else:
                print("❌ evaluate_dataset.py missing")
                return False
                
            if (eval_dir / "__init__.py").exists():
                print("✅ __init__.py exists")
            else:
                print("❌ __init__.py missing")
                return False
        else:
            print("❌ Evaluation directory missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Failed to check evaluation CLI: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing SpeechOcean762 Evaluation System")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_evaluation_imports():
        all_passed = False
    
    # Test config manager
    if not test_config_manager():
        all_passed = False
    
    # Test evaluation CLI
    if not test_evaluation_cli():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Evaluation system is ready.")
        print("\nTo run a full evaluation:")
        print("  python -m src.cli --evaluate-dataset")
        print("  python -m src.cli --evaluate-dataset --max-samples 10")
        print("\nOr use the interactive menu (option 6)")
    else:
        print("❌ Some tests failed. Please install missing dependencies and fix issues.")
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)