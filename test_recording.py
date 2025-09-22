"""
Test script to verify the new sd.rec() approach works correctly.
"""

import time
from src.audio_recorder import AudioRecorder
from src.exceptions import AudioRecordingError, AudioDeviceError

def test_recording():
    """Test the new recording approach."""
    print("🧪 Testing AudioRecorder with sd.rec() approach")
    print("=" * 50)
    
    try:
        # Initialize recorder with recommended settings for your devices
        recorder = AudioRecorder(
            sample_rate=44100,
            channels=1,  # Mono for speech
            dtype='int16'
        )
        
        print(f"✅ Recorder initialized successfully")
        print(f"📊 Using device: {recorder.selected_device['name']}")
        print()
        
        # Start recording
        print("Starting 3-second test recording...")
        recorder.start_recording()
        
        # Record for 3 seconds
        time.sleep(3)
        
        # Stop and get data
        audio_data = recorder.stop_recording()
        
        if len(audio_data) > 0:
            # Save the test recording
            output_path = recorder.save_recording(audio_data, "recordings/test_recording.wav")
            print(f"✅ Test recording successful!")
            print(f"📁 Saved to: {output_path}")
            
            # Check audio quality
            duration = len(audio_data) / recorder.sample_rate
            max_amplitude = max(abs(audio_data)) if len(audio_data) > 0 else 0
            
            print(f"📊 Audio quality check:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Max amplitude: {max_amplitude}")
            print(f"   Data type: {audio_data.dtype}")
            print(f"   Shape: {audio_data.shape}")
            
            if max_amplitude > 100:  # For int16, good signal should be > 100
                print("✅ Good signal level detected")
            else:
                print("⚠️  Low signal level - check microphone volume")
                
        else:
            print("❌ No audio data recorded")
            
    except AudioDeviceError as e:
        print(f"❌ Audio device error: {e}")
        print("💡 Check your microphone connection and permissions")
    except AudioRecordingError as e:
        print(f"❌ Recording error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recording()