"""
Simple example script demonstrating how to use the AudioRecorder class.

This script shows basic usage patterns for recording audio programmatically
without using the CLI interface.
"""

import time
from pathlib import Path
from src.audio_recorder import AudioRecorder
from src.config_manager import ConfigManager
from src.exceptions import AudioRecordingError, AudioDeviceError


def simple_recording_example():
    """Demonstrate basic audio recording functionality."""
    print("🎙️  Simple Audio Recording Example")
    print("=" * 40)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        audio_config = config_manager.get_audio_config()
        
        # Create recorder with configuration
        recorder = AudioRecorder(
            sample_rate=audio_config['sample_rate'],
            channels=audio_config['channels'],
            dtype=audio_config['dtype']
        )
        
        print(f"📊 Recording Configuration:")
        print(f"   Sample Rate: {audio_config['sample_rate']} Hz")
        print(f"   Channels: {audio_config['channels']}")
        print(f"   Data Type: {audio_config['dtype']}")
        print()
        
        # List available devices
        print("📻 Available Audio Devices:")
        recorder.list_audio_devices()
        
        # Start recording
        print("\n🎙️  Starting 5-second recording...")
        recorder.start_recording()
        
        # Record for 5 seconds
        for i in range(5, 0, -1):
            print(f"🔴 Recording... {i} seconds remaining")
            time.sleep(1)
        
        # Stop recording
        print("🛑 Stopping recording...")
        audio_data = recorder.stop_recording()
        
        if len(audio_data) > 0:
            # Save the recording
            output_path = Path("recordings/example_recording.wav")
            saved_path = recorder.save_recording(audio_data, output_path)
            print(f"✅ Recording saved successfully!")
            print(f"📁 File location: {saved_path}")
            
            # Display recording information
            duration = len(audio_data) / audio_config['sample_rate']
            file_size = saved_path.stat().st_size
            print(f"📊 Duration: {duration:.2f} seconds")
            print(f"📊 File size: {file_size / 1024:.1f} KB")
        else:
            print("⚠️  No audio data was recorded")
            
    except AudioDeviceError as e:
        print(f"❌ Audio device error: {e}")
        print("💡 Try checking your microphone connection")
    except AudioRecordingError as e:
        print(f"❌ Recording error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def interactive_recording_example():
    """Demonstrate interactive recording with user input."""
    print("\n🎙️  Interactive Recording Example")
    print("=" * 40)
    
    try:
        # Initialize recorder with default settings
        recorder = AudioRecorder()
        
        while True:
            # Get user input
            print("\nPress Enter to start recording, or 'q' to quit:")
            user_input = input().strip().lower()
            
            if user_input == 'q':
                print("👋 Goodbye!")
                break
            
            # Start recording
            print("🎙️  Recording started. Press Enter to stop...")
            recorder.start_recording()
            
            # Wait for user to stop
            input()
            
            # Stop and save
            audio_data = recorder.stop_recording()
            
            if len(audio_data) > 0:
                # Ask for filename
                filename = input("Enter filename (or press Enter for auto-generated): ").strip()
                
                if filename and not filename.lower().endswith('.wav'):
                    filename += '.wav'
                
                output_path = Path(f"recordings/{filename}") if filename else None
                saved_path = recorder.save_recording(audio_data, output_path)
                print(f"✅ Saved: {saved_path}")
            else:
                print("⚠️  No audio recorded")
                
    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run the example demonstrations."""
    print("🎙️  IELTS Speaking Audio Recorder - Examples")
    print("=" * 50)
    print()
    
    while True:
        print("Choose an example to run:")
        print("1. Simple 5-second recording")
        print("2. Interactive recording")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            simple_recording_example()
        elif choice == '2':
            interactive_recording_example()
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()