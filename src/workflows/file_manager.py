"""
File management utilities for the IELTS Speaking Audio Recorder.

This module handles file operations including listing audio files,
managing transcriptions, and organizing recordings.
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..ui.menu_system import MenuSystem
from ..ui.display_helpers import DisplayHelpers


class FileManager:
    """
    Manages file operations for audio recordings and transcriptions.
    
    This class provides utilities for file discovery, transcription
    management, and file organization.
    """
    
    def __init__(self, recordings_dir: str = "recordings"):
        """
        Initialize the file manager.
        
        Args:
            recordings_dir: Directory where recordings are stored
        """
        self.recordings_dir = Path(recordings_dir)
        self.ensure_directories_exist()
    
    def ensure_directories_exist(self) -> None:
        """Ensure that required directories exist."""
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
    
    def get_audio_files(self, extensions: List[str] = None) -> List[Path]:
        """
        Get list of audio files in the recordings directory.
        
        Args:
            extensions: List of file extensions to include (default: ['wav'])
            
        Returns:
            List[Path]: List of audio file paths sorted by modification time
        """
        if extensions is None:
            extensions = ['wav']
        
        audio_files = []
        for ext in extensions:
            pattern = f"*.{ext.lower()}"
            audio_files.extend(self.recordings_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        audio_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        return audio_files
    
    def select_audio_file(self, menu_system: MenuSystem) -> Optional[Path]:
        """
        Allow user to select an audio file from available recordings.
        
        Args:
            menu_system: Menu system for user interactions
            
        Returns:
            Path or None: Selected file path or None if cancelled
        """
        menu_system.display_section_header("Select Audio File")
        
        # Get available audio files
        audio_files = self.get_audio_files()
        
        if not audio_files:
            menu_system.display_error("No audio files found in recordings directory.")
            return None
        
        # Display files
        DisplayHelpers.display_file_list(audio_files, "Available audio files")
        
        # Get user selection
        choice = menu_system.get_numeric_choice(
            "Enter file number to select (or press Enter to cancel): ",
            min_val=1,
            max_val=len(audio_files)
        )
        
        if choice is not None:
            return audio_files[choice - 1]
        else:
            menu_system.display_info("File selection cancelled.")
            return None
    
    def save_transcription(self, result: Dict[str, Any], audio_file_path: Path) -> Path:
        """
        Save transcription results to a text file.
        
        Args:
            result: Transcription result dictionary
            audio_file_path: Path to the original audio file
            
        Returns:
            Path: Path to the saved transcription file
            
        Raises:
            Exception: If saving fails
        """
        # Generate transcription filename
        base_name = audio_file_path.stem
        transcript_path = audio_file_path.parent / f"{base_name}_transcript.txt"
        
        # Create detailed transcription content
        content = self._create_transcription_content(result, audio_file_path)
        
        # Save to file
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return transcript_path
    
    def _create_transcription_content(self, result: Dict[str, Any], audio_file_path: Path) -> str:
        """
        Create formatted transcription content for saving.
        
        Args:
            result: Transcription result dictionary
            audio_file_path: Path to the original audio file
            
        Returns:
            str: Formatted transcription content
        """
        content = f"""Speech-to-Text Transcription
Audio File: {audio_file_path.name}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Language: {result['language']}
Duration: {DisplayHelpers.format_duration(result['duration'])}
Word Count: {result['word_count']}
Processing Time: {DisplayHelpers.format_duration(result['processing_time'])}

Transcribed Text:
{result['text']}
"""
        
        # Add confidence information if available
        if result.get('confidence_score'):
            confidence_percent = (result['confidence_score'] * 100) if result['confidence_score'] < 0 else result['confidence_score']
            content += f"\nConfidence Score: {confidence_percent:.1f}%"
        
        if result.get('speech_confidence'):
            speech_percent = result['speech_confidence'] * 100
            content += f"\nSpeech Confidence: {speech_percent:.1f}%"
        
        # Add timing information if available
        if result.get('segments'):
            content += "\n\nSegment Timing:\n"
            for i, segment in enumerate(result['segments'], 1):
                start_time = segment['start']
                end_time = segment['end']
                content += f"[{start_time:.1f}s - {end_time:.1f}s] {segment['text']}\n"
        
        # Add word-level timing if available
        if result.get('words') and len(result['words']) > 0:
            content += "\n\nWord-level Timing:\n"
            for word_data in result['words'][:50]:  # Limit to first 50 words to avoid huge files
                word = word_data['word']
                start = word_data['start']
                end = word_data['end']
                content += f"{word} [{start:.1f}s-{end:.1f}s] "
            
            if len(result['words']) > 50:
                content += f"\n... ({len(result['words']) - 50} more words)"
        
        return content
    
    def get_transcription_files(self) -> List[Path]:
        """
        Get list of transcription files in the recordings directory.
        
        Returns:
            List[Path]: List of transcription file paths
        """
        transcript_files = list(self.recordings_dir.glob("*_transcript.txt"))
        # Sort by modification time (newest first)
        transcript_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return transcript_files
    
    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """
        Clean up old audio and transcription files.
        
        Args:
            max_age_days: Maximum age in days for files to keep
            
        Returns:
            int: Number of files deleted
        """
        import time
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0
        
        # Check all files in recordings directory
        for file_path in self.recordings_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception:
                    pass  # Ignore errors
        
        return deleted_count
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information about the recordings directory.
        
        Returns:
            Dict containing storage statistics
        """
        audio_files = self.get_audio_files()
        transcript_files = self.get_transcription_files()
        
        # Calculate total sizes
        total_audio_size = sum(f.stat().st_size for f in audio_files)
        total_transcript_size = sum(f.stat().st_size for f in transcript_files)
        
        # Calculate total duration (approximate from file sizes)
        # Assume average 1.5MB per minute for 44.1kHz mono WAV
        estimated_duration_minutes = total_audio_size / (1.5 * 1024 * 1024)
        
        return {
            'audio_files_count': len(audio_files),
            'transcript_files_count': len(transcript_files),
            'total_audio_size': total_audio_size,
            'total_transcript_size': total_transcript_size,
            'total_size': total_audio_size + total_transcript_size,
            'estimated_duration_minutes': estimated_duration_minutes,
            'recordings_directory': str(self.recordings_dir)
        }
    
    def display_storage_info(self, menu_system: MenuSystem) -> None:
        """
        Display storage information to the user.
        
        Args:
            menu_system: Menu system for displaying information
        """
        info = self.get_storage_info()
        
        menu_system.display_section_header("Storage Information")
        
        print(f"Recordings Directory: {info['recordings_directory']}")
        print(f"Audio Files: {info['audio_files_count']}")
        print(f"Transcription Files: {info['transcript_files_count']}")
        print(f"Total Audio Size: {DisplayHelpers.format_file_size(info['total_audio_size'])}")
        print(f"Total Transcript Size: {DisplayHelpers.format_file_size(info['total_transcript_size'])}")
        print(f"Total Storage Used: {DisplayHelpers.format_file_size(info['total_size'])}")
        print(f"Estimated Audio Duration: {DisplayHelpers.format_duration(info['estimated_duration_minutes'] * 60)}")
        
        menu_system.wait_for_enter()
    
    def organize_files_by_date(self) -> None:
        """
        Organize files into subdirectories by date.
        This can help manage large numbers of recordings.
        """
        audio_files = self.get_audio_files()
        
        for file_path in audio_files:
            # Get file creation date
            creation_time = file_path.stat().st_mtime
            date_str = time.strftime("%Y-%m-%d", time.localtime(creation_time))
            
            # Create date directory
            date_dir = self.recordings_dir / date_str
            date_dir.mkdir(exist_ok=True)
            
            # Move file
            new_path = date_dir / file_path.name
            if not new_path.exists():
                file_path.rename(new_path)
                
                # Also move corresponding transcript if it exists
                transcript_path = file_path.parent / f"{file_path.stem}_transcript.txt"
                if transcript_path.exists():
                    new_transcript_path = date_dir / transcript_path.name
                    if not new_transcript_path.exists():
                        transcript_path.rename(new_transcript_path)