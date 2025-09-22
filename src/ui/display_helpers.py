"""
Display helper utilities for the IELTS Speaking Audio Recorder.

This module provides utility functions for formatting and displaying
information in a consistent and user-friendly manner.
"""

from pathlib import Path
from typing import Dict, Any, List
import time


class DisplayHelpers:
    """
    Utility class for formatting and displaying information.
    
    Provides static methods for consistent formatting of various types
    of information throughout the application.
    """
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to a human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            str: Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {minutes}m {remaining_seconds:.1f}s"
    
    @staticmethod
    def format_file_size(bytes_size: int) -> str:
        """
        Format file size in bytes to a human-readable string.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            str: Formatted file size string
        """
        if bytes_size < 1024:
            return f"{bytes_size} B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.1f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.1f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.1f} GB"
    
    @staticmethod
    def format_audio_config(config: Dict[str, Any]) -> str:
        """
        Format audio configuration for display.
        
        Args:
            config: Audio configuration dictionary
            
        Returns:
            str: Formatted configuration string
        """
        sample_rate = config.get('sample_rate', 'Unknown')
        channels = config.get('channels', 'Unknown')
        dtype = config.get('dtype', 'Unknown')
        
        channel_text = 'Mono' if channels == 1 else 'Stereo' if channels == 2 else f'{channels} channels'
        
        return f"{sample_rate} Hz, {channel_text}, {dtype}"
    
    @staticmethod
    def display_audio_devices(devices: List[Dict[str, Any]], current_device_id: int = None) -> None:
        """
        Display audio devices in a formatted table.
        
        Args:
            devices: List of device information dictionaries
            current_device_id: ID of currently selected device
        """
        print("\n📻 Available Audio Input Devices:")
        print("-" * 70)
        
        for device in devices:
            device_id = device['id']
            name = device['name']
            max_channels = device['max_channels']
            sample_rate = device.get('sample_rate', 'Unknown')
            
            # Truncate long device names
            if len(name) > 40:
                name = name[:37] + "..."
            
            # Mark current device
            marker = "👉" if device_id == current_device_id else "  "
            
            # Quality indicator
            quality = DisplayHelpers._get_device_quality_indicator(device)
            
            print(f"{marker} [{device_id:2d}] {name:<42} {quality} ({max_channels} ch, {sample_rate}Hz)")
        
        print("-" * 70)
    
    @staticmethod
    def _get_device_quality_indicator(device: Dict[str, Any]) -> str:
        """
        Get a quality indicator for an audio device.
        
        Args:
            device: Device information dictionary
            
        Returns:
            str: Quality indicator emoji
        """
        name_lower = device['name'].lower()
        
        if 'microphone' in name_lower or 'mic' in name_lower:
            return "🎙️"
        elif 'array' in name_lower:
            return "🔊"
        elif 'stereo mix' in name_lower or 'capture' in name_lower:
            return "⚠️"
        else:
            return "🔵"
    
    @staticmethod
    def display_transcription_result(result: Dict[str, Any]) -> None:
        """
        Display transcription results in a formatted way.
        
        Args:
            result: Transcription result dictionary
        """
        print("\n📝 Transcription Results:")
        print("=" * 50)
        print(f"Text: {result['text']}")
        print(f"Language: {result['language']}")
        print(f"Duration: {DisplayHelpers.format_duration(result['duration'])}")
        print(f"Word count: {result['word_count']}")
        print(f"Processing time: {DisplayHelpers.format_duration(result['processing_time'])}")
        
        if result.get('confidence_score'):
            confidence_percent = (result['confidence_score'] * 100) if result['confidence_score'] < 0 else result['confidence_score']
            print(f"Confidence score: {confidence_percent:.1f}%")
        
        if result.get('speech_confidence'):
            speech_percent = result['speech_confidence'] * 100
            print(f"Speech confidence: {speech_percent:.1f}%")
    
    @staticmethod
    def display_file_list(files: List[Path], title: str = "Available Files") -> None:
        """
        Display a list of files with their details.
        
        Args:
            files: List of file paths
            title: Title for the file list
        """
        print(f"\n{title}:")
        if not files:
            print("❌ No files found.")
            return
        
        for i, file_path in enumerate(files, 1):
            file_size = file_path.stat().st_size
            size_str = DisplayHelpers.format_file_size(file_size)
            
            # Get file modification time
            mod_time = file_path.stat().st_mtime
            mod_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mod_time))
            
            print(f"  [{i}] {file_path.name:<30} ({size_str}, {mod_str})")
    
    @staticmethod
    def display_progress_bar(progress: float, width: int = 50, message: str = "") -> None:
        """
        Display a progress bar.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            width: Width of the progress bar
            message: Optional message to display with the progress bar
        """
        filled_width = int(width * progress)
        bar = "█" * filled_width + "░" * (width - filled_width)
        percentage = progress * 100
        
        if message:
            print(f"\r{message} [{bar}] {percentage:.1f}%", end="", flush=True)
        else:
            print(f"\r[{bar}] {percentage:.1f}%", end="", flush=True)
    
    @staticmethod
    def format_timestamp() -> str:
        """
        Get a formatted timestamp for file naming.
        
        Returns:
            str: Formatted timestamp string
        """
        return time.strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def display_azure_status(is_configured: bool, endpoint: str = "", deployment: str = "") -> str:
        """
        Get a formatted Azure OpenAI status string.
        
        Args:
            is_configured: Whether Azure OpenAI is configured
            endpoint: Azure endpoint URL
            deployment: Deployment name
            
        Returns:
            str: Formatted status string
        """
        if is_configured:
            return "✅ Configured"
        else:
            return "❌ Not configured"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50) -> str:
        """
        Truncate text to a maximum length with ellipsis.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            str: Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    @staticmethod
    def center_text(text: str, width: int = 80, fill_char: str = " ") -> str:
        """
        Center text within a given width.
        
        Args:
            text: Text to center
            width: Total width
            fill_char: Character to use for padding
            
        Returns:
            str: Centered text
        """
        return text.center(width, fill_char)