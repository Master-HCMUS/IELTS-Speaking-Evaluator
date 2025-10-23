"""
Test script to verify local fine-tuned Whisper model loading and inference.

This script tests the local transcription service with the fine-tuned model.
"""

import sys
from pathlib import Path

# Add src to path and handle imports properly
script_dir = Path(__file__).parent
src_dir = script_dir
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import exceptions first
from exceptions import ConfigurationError, TranscriptionError, AudioFileError

# Now import the services
try:
    from config_manager import ConfigManager
    from local_transcription_service import LocalWhisperTranscriptionService
    from combined_transcription_service import CombinedTranscriptionService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the src directory and all dependencies are installed.")
    sys.exit(1)


def test_local_model():
    """Test loading and using the local fine-tuned model."""
    print("🧪 Testing Local Fine-tuned Whisper Model")
    print("=" * 50)
    
    # Path to the fine-tuned model
    model_path = "finetuning/models/whisper_development"
    
    # Check if model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ Model path not found: {model_path}")
        print(f"Expected path: {model_path_obj.absolute()}")
        return False
    
    print(f"✅ Model path found: {model_path}")
    
    try:
        # Initialize local transcription service
        print("\n🤖 Initializing local transcription service...")
        service = LocalWhisperTranscriptionService(
            model_path=model_path,
            device="auto"
        )
        
        # Test connection
        print("\n🔍 Testing model connection...")
        result = service.test_connection()
        
        if result["status"] == "success":
            print("✅ Model test successful!")
            print(f"   Model parameters: {result.get('model_parameters', 'Unknown'):,}")
            print(f"   Device: {result.get('device', 'Unknown')}")
            print(f"   Test transcription: '{result.get('test_transcription', 'N/A')}'")
        else:
            print(f"❌ Model test failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Get model info
        print("\n📊 Model Information:")
        model_info = service.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing local model: {e}")
        return False


def test_combined_service():
    """Test the combined transcription service."""
    print("\n\n🔄 Testing Combined Transcription Service")
    print("=" * 50)
    
    try:
        # Initialize config manager
        config_manager = ConfigManager()
        
        # Enable local Whisper
        local_config = config_manager.get_local_whisper_config()
        local_config["enabled"] = True
        local_config["model_path"] = "finetuning/models/whisper_development"
        local_config["prefer_local"] = True
        config_manager.save_local_whisper_config(local_config)
        
        print("✅ Local Whisper configuration enabled")
        
        # Initialize combined service
        print("\n🔄 Initializing combined transcription service...")
        combined_service = CombinedTranscriptionService(config_manager)
        
        # Get service info
        service_info = combined_service.get_service_info()
        print(f"✅ Active service: {service_info.get('service_type', 'Unknown')}")
        
        # Test connection
        print("\n🔍 Testing combined service connection...")
        result = combined_service.test_connection()
        
        if result["status"] == "success":
            print("✅ Combined service test successful!")
            if result.get("service_type") == "local":
                print(f"   Using local model: {result.get('model_path', 'Unknown')}")
                print(f"   Device: {result.get('device', 'Unknown')}")
            elif result.get("service_type") == "azure":
                print(f"   Using Azure endpoint: {result.get('endpoint', 'Unknown')}")
        else:
            print(f"❌ Combined service test failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Show available services
        print("\n📋 Available Services:")
        available = combined_service.get_available_services()
        for service_type, is_available in available.items():
            if service_type != "active":
                status = "✅" if is_available else "❌"
                print(f"   {service_type.title()}: {status}")
        print(f"   Active: {available.get('active', 'None')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing combined service: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Fine-tuned Whisper Model Test Suite")
    print("=" * 60)
    
    # Test local model directly
    local_success = test_local_model()
    
    # Test combined service
    combined_success = test_combined_service()
    
    # Summary
    print("\n\n📋 Test Summary")
    print("=" * 30)
    print(f"Local Model Test: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"Combined Service Test: {'✅ PASS' if combined_success else '❌ FAIL'}")
    
    if local_success and combined_success:
        print("\n🎉 All tests passed! Your fine-tuned model is ready for use.")
        print("\nTo use the fine-tuned model in the main application:")
        print("1. Run: python -m src.cli --local-whisper-config")
        print("2. Enable the local Whisper model")
        print("3. Set the model path to: finetuning/models/whisper_development")
        print("4. Choose 'prefer local' if you want it to be the default")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())