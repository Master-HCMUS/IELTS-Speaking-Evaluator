"""
Menu system for the IELTS Speaking Audio Recorder CLI.

This module handles all menu displays, user input collection, and menu navigation
logic, providing a clean separation between UI logic and business logic.
"""

from typing import Dict, Callable, Any, Optional
import sys


class MenuSystem:
    """
    Handles menu display and user input for the CLI application.
    
    This class provides a clean interface for displaying menus, collecting
    user input, and handling menu navigation with proper error handling.
    """
    
    def __init__(self):
        """Initialize the menu system."""
        self.current_menu = "main"
        self.menu_handlers: Dict[str, Callable] = {}
    
    def register_handler(self, menu_key: str, handler: Callable) -> None:
        """
        Register a handler function for a menu option.
        
        Args:
            menu_key: The menu key (e.g., '1', '2', 'q')
            handler: The function to call when this option is selected
        """
        self.menu_handlers[menu_key] = handler
    
    def display_welcome(self) -> None:
        """Display the welcome message."""
        print("🎙️  IELTS Speaking Audio Recorder + AI Transcription")
        print("=" * 55)
        print("Cross-platform audio recording with Azure OpenAI Whisper")
        print("Record speech, get instant transcriptions, save results")
        print("Press Ctrl+C at any time to stop and save recording")
        print()
    
    def display_main_menu(self) -> None:
        """Display the main menu options."""
        print("📋 Available Commands:")
        print("  [1] Start Recording")
        print("  [2] Record + Transcribe")
        print("  [3] Transcribe Existing File")
        print("  [4] Assess Pronunciation")
        print("  [5] Comprehensive Assessment (Transcribe + Pronunciation)")
        print("  [6] Evaluate Against SpeechOcean762 Dataset")
        print("  [7] List Audio Devices")
        print("  [8] Select Audio Device")
        print("  [9] Configure Settings")
        print("  [10] Configure Azure OpenAI")
        print("  [11] View Current Settings")
        print("  [12] Test Azure Connection")
        print("  [h] Help")
        print("  [q] Quit")
        print()
    
    def get_user_choice(self, prompt: str = "Enter your choice: ") -> str:
        """
        Get user input with proper error handling.
        
        Args:
            prompt: The prompt to display to the user
            
        Returns:
            str: The user's input (stripped and lowercased)
        """
        try:
            return input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            sys.exit(0)
    
    def get_user_input(self, prompt: str, default: str = "") -> str:
        """
        Get user input with optional default value.
        
        Args:
            prompt: The prompt to display
            default: Default value if user presses Enter
            
        Returns:
            str: The user's input or default value
        """
        try:
            user_input = input(prompt).strip()
            return user_input if user_input else default
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            sys.exit(0)
    
    def get_yes_no_choice(self, prompt: str, default: bool = True) -> bool:
        """
        Get a yes/no choice from the user.
        
        Args:
            prompt: The prompt to display
            default: Default value if user presses Enter
            
        Returns:
            bool: True for yes, False for no
        """
        while True:
            default_text = "(Y/n)" if default else "(y/N)"
            choice = self.get_user_input(f"{prompt} {default_text}: ").lower()
            
            if not choice:
                return default
            elif choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def get_numeric_choice(self, prompt: str, min_val: int, max_val: int, default: Optional[int] = None) -> Optional[int]:
        """
        Get a numeric choice from the user within a specified range.
        
        Args:
            prompt: The prompt to display
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            default: Default value if user presses Enter
            
        Returns:
            int or None: The user's choice or None if cancelled
        """
        while True:
            try:
                user_input = self.get_user_input(prompt)
                
                if not user_input:
                    if default is not None:
                        return default
                    print("Input cancelled.")
                    return None
                
                value = int(user_input)
                if min_val <= value <= max_val:
                    return value
                else:
                    print(f"Please enter a number between {min_val} and {max_val}")
                    
            except ValueError:
                print("Please enter a valid number")
    
    def display_error(self, message: str) -> None:
        """
        Display an error message with consistent formatting.
        
        Args:
            message: The error message to display
        """
        print(f"❌ {message}")
    
    def display_success(self, message: str) -> None:
        """
        Display a success message with consistent formatting.
        
        Args:
            message: The success message to display
        """
        print(f"✅ {message}")
    
    def display_warning(self, message: str) -> None:
        """
        Display a warning message with consistent formatting.
        
        Args:
            message: The warning message to display
        """
        print(f"⚠️  {message}")
    
    def display_info(self, message: str) -> None:
        """
        Display an info message with consistent formatting.
        
        Args:
            message: The info message to display
        """
        print(f"ℹ️  {message}")
    
    def wait_for_enter(self, message: str = "Press Enter to continue...") -> None:
        """
        Wait for user to press Enter before continuing.
        
        Args:
            message: The message to display
        """
        try:
            input(f"\n{message}")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            sys.exit(0)
    
    def clear_screen(self) -> None:
        """Clear the screen (cross-platform)."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_section_header(self, title: str, width: int = 50) -> None:
        """
        Display a section header with consistent formatting.
        
        Args:
            title: The section title
            width: Width of the header line
        """
        print(f"\n{title}")
        print("-" * min(len(title), width))
    
    def display_config_option(self, label: str, current_value: Any, description: str = "") -> None:
        """
        Display a configuration option with current value.
        
        Args:
            label: The option label
            current_value: The current value
            description: Optional description
        """
        print(f"{label}: {current_value}")
        if description:
            print(f"  {description}")
    
    def display_help(self) -> None:
        """Display comprehensive help information."""
        help_text = """
🆘 IELTS Speaking Audio Recorder Help
=====================================

Commands:
  1. Start Recording - Begin recording audio from your microphone
  2. Record + Transcribe - Record audio and automatically transcribe with Azure OpenAI
  3. Transcribe Existing File - Convert existing WAV files to text
  4. Assess Pronunciation - Evaluate pronunciation of existing audio files
  5. Comprehensive Assessment - Full analysis (transcription + pronunciation)
  6. List Audio Devices - Show available microphone inputs
  7. Select Audio Device - Choose a specific microphone
  8. Configure Settings - Change audio parameters (sample rate, channels, etc.)
  9. Configure Azure OpenAI - Set up speech-to-text transcription
  10. View Current Settings - Display current configuration
  11. Test Azure Connection - Verify Azure OpenAI connectivity
  h. Help - Show this help message
  q. Quit - Exit the application

Recording Process:
  • Press Enter to start recording
  • Speak into your microphone
  • Press Enter again to stop recording
  • Choose a filename or use auto-generated name
  • Files are saved in the 'recordings/' directory as WAV files

Transcription Process:
  • Configure Azure OpenAI endpoint and API key in .env file
  • Record audio or select existing file
  • Get real-time speech-to-text results
  • Save transcriptions as text files

Pronunciation Assessment:
  • Configure Azure Speech service API key and region in .env file
  • Select existing audio file for assessment
  • Provide reference text or use transcription
  • Get detailed pronunciation scores (accuracy, fluency, completeness)
  • View word-level analysis and feedback
  • Save assessment results as text files

Comprehensive Assessment:
  • Combines transcription and pronunciation evaluation
  • Uses transcribed text as reference for pronunciation scoring
  • Provides complete speech analysis in one step

Azure OpenAI Setup:
  • Create an Azure OpenAI resource in Azure portal
  • Deploy a Whisper model (e.g., 'whisper')
  • Get your endpoint URL and API key
  • Add configuration to .env file

Azure Speech Service Setup:
  • Create an Azure Speech service resource
  • Get your API key and region
  • Add AZURE_SPEECH_API_KEY and AZURE_SPEECH_REGION to .env file

Tips:
  • Use Ctrl+C to interrupt and save current recording
  • For best quality, use 44100 Hz sample rate
  • Mono (1 channel) is sufficient for speech recording
  • Check your microphone levels before recording
  • Auto-transcribe can be enabled for seamless workflow

Supported Audio Formats:
  • Output: WAV files only
  • Sample Rates: 8000, 16000, 22050, 44100, 48000 Hz
  • Channels: 1 (Mono) or 2 (Stereo)
  • Data Types: int16, float32

Transcription Features:
  • Multiple language support (auto-detection available)
  • Word-level timestamps
  • Confidence scores
  • Detailed segment timing
  • Automatic text file generation

Troubleshooting:
  • If no audio devices are found, check microphone connections
  • If recording fails, try different sample rates
  • Ensure 'recordings/' directory is writable
  • For Azure issues, verify 'az login' and permissions
  • Check Azure OpenAI endpoint and deployment names
        """
        print(help_text)
        self.wait_for_enter()
    
    def run_menu_loop(self, menu_handlers: Dict[str, Callable]) -> None:
        """
        Run the main menu loop.
        
        Args:
            menu_handlers: Dictionary mapping menu choices to handler functions
        """
        self.menu_handlers = menu_handlers
        
        while True:
            try:
                self.display_main_menu()
                choice = self.get_user_choice()
                
                if choice in self.menu_handlers:
                    try:
                        self.menu_handlers[choice]()
                    except Exception as e:
                        self.display_error(f"Unexpected error: {e}")
                        self.wait_for_enter("Press Enter to continue...")
                elif choice in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                else:
                    self.display_error("Invalid choice. Please try again.")
                
                print()  # Add spacing between menu iterations
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.display_error(f"Unexpected error: {e}")
                self.wait_for_enter("Press Enter to continue...")