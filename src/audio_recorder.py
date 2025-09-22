"""
Audio recording module for capturing microphone input.

This module provides a clean interface for recording audio from the user's microphone
using cross-platform libraries. It handles device initialization, recording sessions,
and saving audio files with proper exception handling.
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from .exceptions import AudioRecordingError, AudioDeviceError


class AudioRecorder:
    """
    A cross-platform audio recorder that captures microphone input and saves to WAV files.
    
    This class provides methods to start, stop, and manage audio recording sessions
    with configurable audio parameters and robust error handling.
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 1, dtype: str = 'int16', device_id: Optional[int] = None):
        """
        Initialize the AudioRecorder with specified audio parameters.
        
        Args:
            sample_rate (int): Audio sample rate in Hz. Default is 44100.
            channels (int): Number of audio channels. Default is 1 (mono).
            dtype (str): Audio data type. Default is 'int16'.
            device_id (Optional[int]): Specific device ID to use. If None, uses best available.
            
        Raises:
            AudioDeviceError: If no audio input device is available.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device_id = device_id
        self.is_recording = False
        self.recording_data: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        
        # Validate audio device availability and select best device
        self.selected_device = self._check_and_select_device()
        
    def _check_and_select_device(self) -> dict:
        """
        Check available audio devices and select the best one for recording.
        
        Returns:
            dict: Selected device information
            
        Raises:
            AudioDeviceError: If no suitable audio input device is found.
        """
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': idx,
                        'name': device['name'],
                        'max_channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate'],
                        'device_info': device
                    })
            
            if not input_devices:
                raise AudioDeviceError("No audio input devices found")
            
            # If specific device ID is provided, use it
            if self.device_id is not None:
                selected_device = next((d for d in input_devices if d['id'] == self.device_id), None)
                if selected_device is None:
                    raise AudioDeviceError(f"Device ID {self.device_id} not found")
                
                # Validate channels compatibility
                if self.channels > selected_device['max_channels']:
                    print(f"⚠️  Warning: Requested {self.channels} channels, but device only supports {selected_device['max_channels']}. Adjusting to {min(self.channels, selected_device['max_channels'])}.")
                    self.channels = min(self.channels, selected_device['max_channels'])
                
                return selected_device
            
            # Auto-select best device based on quality criteria
            # Prioritize devices with good names and appropriate channel support
            quality_devices = []
            
            for device in input_devices:
                score = 0
                name_lower = device['name'].lower()
                
                # Prefer real microphones over system mixers
                if 'microphone' in name_lower or 'mic' in name_lower:
                    score += 10
                elif 'realtek' in name_lower and 'mic' in name_lower:
                    score += 8
                elif 'array' in name_lower:
                    score += 6
                
                # Penalize system capture devices
                if 'stereo mix' in name_lower or 'sound mapper' in name_lower or 'capture driver' in name_lower:
                    score -= 5
                
                # Prefer devices that match our sample rate
                if abs(device['sample_rate'] - self.sample_rate) < 1000:
                    score += 5
                
                # Check channel compatibility
                if device['max_channels'] >= self.channels:
                    score += 3
                elif device['max_channels'] == 2 and self.channels == 1:
                    score += 2  # Can convert stereo to mono
                
                quality_devices.append((score, device))
            
            # Sort by score and select the best
            quality_devices.sort(key=lambda x: x[0], reverse=True)
            best_device = quality_devices[0][1]
            
            # Ensure channel compatibility
            if self.channels > best_device['max_channels']:
                print(f"⚠️  Auto-adjusting channels from {self.channels} to {best_device['max_channels']} to match device capabilities.")
                self.channels = min(self.channels, best_device['max_channels'])
            
            print(f"🎙️  Selected audio device: {best_device['name']} (ID: {best_device['id']})")
            print(f"📊 Device specs: {best_device['max_channels']} channels, {best_device['sample_rate']}Hz")
            
            return best_device
            
        except Exception as e:
            raise AudioDeviceError(f"Failed to initialize audio devices: {e}")
    
    def _check_audio_devices(self) -> None:
        """
        Check if audio input devices are available.
        
        Raises:
            AudioDeviceError: If no audio input device is found.
        """
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                raise AudioDeviceError("No audio input devices found")
                
            # Set default input device
            sd.default.device = sd.default.device[0], sd.default.device[1]
            
        except Exception as e:
            raise AudioDeviceError(f"Failed to initialize audio devices: {e}")
    
    def start_recording(self) -> None:
        """
        Start recording audio from the microphone.
        
        Raises:
            AudioRecordingError: If recording is already in progress or fails to start.
        """
        if self.is_recording:
            raise AudioRecordingError("Recording is already in progress")
        
        try:
            with self._lock:
                self.is_recording = True
                self.recording_data = None
            
            print(f"🎙️  Recording started")
            print(f"📊 Device: {self.selected_device['name']}")
            print(f"📊 Settings: {self.sample_rate}Hz, {self.channels} channel{'s' if self.channels > 1 else ''}, {self.dtype}")
            
            # Start recording using sd.rec() - much simpler and more reliable
            # We'll record for a very long duration (1 hour) and stop it manually
            self.recording_data = sd.rec(
                frames=self.sample_rate * 3600,  # 1 hour max
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',  # Use float32 internally for better quality
                device=self.selected_device['id']
            )
            
        except Exception as e:
            self.is_recording = False
            raise AudioRecordingError(f"Failed to start recording: {e}")
    
    def stop_recording(self) -> np.ndarray:
        """
        Stop recording and return the captured audio data.
        
        Returns:
            np.ndarray: The recorded audio data as a numpy array.
            
        Raises:
            AudioRecordingError: If no recording is in progress.
        """
        if not self.is_recording:
            raise AudioRecordingError("No recording in progress")
        
        try:
            with self._lock:
                self.is_recording = False
            
            # Stop the recording
            sd.stop()
            
            # Wait for recording to finish
            sd.wait()
            
            # Get the recorded data
            if self.recording_data is not None:
                # Find where actual recording ends (remove trailing silence)
                # Calculate RMS to find the end of meaningful audio
                if len(self.recording_data.shape) > 1:
                    # Multi-channel: calculate RMS across channels
                    rms = np.sqrt(np.mean(self.recording_data**2, axis=1))
                else:
                    # Mono: calculate RMS directly
                    rms = np.abs(self.recording_data)
                
                # Find the last point with meaningful audio (above 0.1% of max)
                max_amplitude = np.max(rms)
                threshold = max_amplitude * 0.001
                
                # Find last sample above threshold
                last_meaningful_sample = len(rms) - 1
                for i in range(len(rms) - 1, -1, -1):
                    if rms[i] > threshold:
                        # Add a small buffer (0.1 seconds)
                        last_meaningful_sample = min(len(rms) - 1, i + int(self.sample_rate * 0.1))
                        break
                
                # Trim to actual content
                if last_meaningful_sample > 0:
                    recorded_data = self.recording_data[:last_meaningful_sample + 1]
                else:
                    recorded_data = self.recording_data
                
                # Convert to mono if multi-channel input but mono output requested
                if len(recorded_data.shape) > 1 and self.channels == 1:
                    recorded_data = np.mean(recorded_data, axis=1)
                
                # Convert to the requested dtype
                if self.dtype == 'int16':
                    # Ensure no clipping and convert to int16
                    recorded_data = np.clip(recorded_data, -1.0, 1.0)
                    recorded_data = (recorded_data * 32767).astype(np.int16)
                elif self.dtype == 'float32':
                    recorded_data = recorded_data.astype(np.float32)
                
                print("🛑 Recording stopped")
                duration = len(recorded_data) / self.sample_rate
                print(f"📊 Recorded {duration:.2f} seconds of audio")
                
                return recorded_data
            else:
                print("⚠️  No audio data recorded")
                return np.array([], dtype=self.dtype)
                
        except Exception as e:
            raise AudioRecordingError(f"Failed to stop recording: {e}")
    
    def save_recording(self, audio_data: np.ndarray, output_path: Optional[Path] = None) -> Path:
        """
        Save the recorded audio data to a WAV file.
        
        Args:
            audio_data: The audio data to save
            output_path: Optional path for the output file. If None, generates a timestamp-based name.
            
        Returns:
            Path: The path where the file was saved.
            
        Raises:
            AudioRecordingError: If saving fails.
        """
        try:
            if output_path is None:
                # Generate a timestamp-based filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"recordings/recording_{timestamp}.wav")
            
            # Ensure the output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the WAV file
            wavfile.write(str(output_path), self.sample_rate, audio_data)
            
            # Get file size for user feedback
            file_size = output_path.stat().st_size
            duration = len(audio_data) / self.sample_rate
            
            print(f"💾 Audio saved to: {output_path}")
            print(f"📊 Duration: {duration:.2f}s, Size: {file_size/1024:.1f}KB")
            
            return output_path
            
        except Exception as e:
            raise AudioRecordingError(f"Failed to save recording: {e}")
    
    def get_recording_info(self) -> dict:
        """
        Get information about the current recording configuration.
        
        Returns:
            dict: Recording configuration information.
        """
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "dtype": self.dtype,
            "is_recording": self.is_recording,
            "recording_active": self.recording_data is not None,
            "selected_device": self.selected_device['name'] if hasattr(self, 'selected_device') else "Unknown"
        }
    
    def set_device(self, device_id: int) -> bool:
        """
        Set a specific audio input device.
        
        Args:
            device_id (int): The device ID to use
            
        Returns:
            bool: True if device was set successfully, False otherwise
        """
        if self.is_recording:
            print("❌ Cannot change device while recording")
            return False
            
        try:
            # Store old device for rollback
            old_device = self.selected_device if hasattr(self, 'selected_device') else None
            
            # Set new device
            self.device_id = device_id
            self.selected_device = self._check_and_select_device()
            
            print(f"✅ Device changed to: {self.selected_device['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to set device {device_id}: {e}")
            # Rollback to old device if possible
            if old_device:
                self.selected_device = old_device
            return False
    
    def list_audio_devices(self) -> None:
        """Print available audio input devices with recommendations."""
        try:
            devices = sd.query_devices()
            print("\n📻 Available Audio Input Devices:")
            print("-" * 50)
            
            current_device_id = self.selected_device['id'] if hasattr(self, 'selected_device') else None
            
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    marker = "👈 CURRENT" if idx == current_device_id else ""
                    quality_indicator = ""
                    
                    name_lower = device['name'].lower()
                    if 'microphone' in name_lower and 'realtek' in name_lower:
                        quality_indicator = "⭐ RECOMMENDED"
                    elif 'microphone' in name_lower or 'mic' in name_lower:
                        quality_indicator = "✅ GOOD"
                    elif 'stereo mix' in name_lower or 'sound mapper' in name_lower:
                        quality_indicator = "⚠️  SYSTEM DEVICE"
                    
                    print(f"{idx}: {device['name']} {marker}")
                    print(f"    Max input channels: {device['max_input_channels']}")
                    print(f"    Default sample rate: {device['default_samplerate']}")
                    if quality_indicator:
                        print(f"    {quality_indicator}")
                    print()
                    
        except Exception as e:
            print(f"❌ Failed to list audio devices: {e}")