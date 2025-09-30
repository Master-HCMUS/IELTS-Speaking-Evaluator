#!/usr/bin/env python3
"""
Test script for Whisper Model Evaluation

This script tests the Whisper evaluation functionality without requiring
a fine-tuned model by using a pre-trained Whisper model.
"""

import sys
from pathlib import Path
import tempfile
import os

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_whisper_pronunciation_assessor():
    """Test the WhisperPronunciationAssessor with a pre-trained model."""
    print("🧪 Testing WhisperPronunciationAssessor...")
    
    try:
        from evaluation.whisper_evaluator import WhisperPronunciationAssessor
        
        # Use a small pre-trained model for testing
        print("📥 Loading pre-trained Whisper model (this may take a moment)...")
        assessor = WhisperPronunciationAssessor("openai/whisper-tiny")
        print("✅ Model loaded successfully")
        
        # Create a simple test audio file (sine wave)
        import numpy as np
        import soundfile as sf
        
        # Generate a simple sine wave (440 Hz for 1 second)
        sample_rate = 16000
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            temp_audio_path = tmp_file.name
        
        try:
            # Test transcription
            print("🎵 Testing audio transcription...")
            result = assessor.transcribe_audio(temp_audio_path, "hello world")
            print(f"📝 Transcription result: {result}")
            
            # Test pronunciation assessment
            print("🎯 Testing pronunciation assessment...")
            scores = assessor.assess_pronunciation(result)
            print(f"📊 Pronunciation scores: {scores}")
            
            print("✅ WhisperPronunciationAssessor test passed!")
            return True
            
        finally:
            # Clean up
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ WhisperPronunciationAssessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_whisper_evaluator_setup():
    """Test WhisperModelEvaluator initialization without running full evaluation."""
    print("\n🧪 Testing WhisperModelEvaluator setup...")
    
    try:
        from evaluation.whisper_evaluator import WhisperModelEvaluator
        
        # Test with pre-trained model
        print("📥 Initializing evaluator with pre-trained model...")
        evaluator = WhisperModelEvaluator("openai/whisper-tiny")
        print("✅ Evaluator initialized successfully")
        
        # Test dataset loading (with minimal samples)
        print("📊 Testing dataset loading...")
        if evaluator.load_dataset(split="test", max_samples=1):
            print("✅ Dataset loaded successfully")
            print("✅ WhisperModelEvaluator test passed!")
            return True
        else:
            print("❌ Failed to load dataset")
            return False
            
    except Exception as e:
        print(f"❌ WhisperModelEvaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test that all modules can be imported correctly."""
    print("🧪 Testing imports...")
    
    try:
        from evaluation.whisper_evaluator import (
            WhisperModelEvaluator,
            WhisperPronunciationAssessor, 
            WhisperEvaluationResult
        )
        print("✅ All Whisper evaluation imports successful")
        
        from evaluation import (
            WhisperModelEvaluator,
            WhisperPronunciationAssessor,
            WhisperEvaluationResult
        )
        print("✅ Package-level imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Whisper Model Evaluation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("WhisperPronunciationAssessor Test", test_whisper_pronunciation_assessor),
        ("WhisperModelEvaluator Setup Test", test_whisper_evaluator_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print(f"\n{'='*20} TEST SUMMARY {'='*20}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<35} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Whisper evaluation system is ready to use.")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)