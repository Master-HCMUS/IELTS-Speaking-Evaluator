"""
Simplified command-line interface for the IELTS Speaking Audio Recorder.

This module serves as a thin orchestration layer that coordinates
between the UI components and business logic workflows.
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Callable

from .config_manager import ConfigManager
from .ui.menu_system import MenuSystem
from .workflows.workflow_orchestrator import WorkflowOrchestrator
from .workflows.configuration_handlers import ConfigurationHandlers
from .workflows.file_manager import FileManager


class AudioRecorderCLI:
    """
    Simplified CLI that orchestrates the application workflow.
    
    This class acts as a thin coordination layer between the UI
    and business logic components.
    """
    
    def __init__(self):
        """Initialize the CLI with all necessary components."""
        self.config_manager = ConfigManager()
        self.menu_system = MenuSystem()
        self.workflow_orchestrator = WorkflowOrchestrator(self.config_manager, self.menu_system)
        self.config_handlers = ConfigurationHandlers(self.config_manager, self.menu_system)
        self.file_manager = FileManager()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n🛑 Interrupt received. Stopping...")
        
        # Try to save any ongoing recording
        if (self.workflow_orchestrator.recorder and 
            self.workflow_orchestrator.recorder.is_recording):
            try:
                audio_data = self.workflow_orchestrator.recorder.stop_recording()
                if len(audio_data) > 0:
                    save_path = self.workflow_orchestrator.recorder.save_recording(audio_data)
                    self.menu_system.display_success(f"Emergency save completed: {save_path}")
            except Exception as e:
                self.menu_system.display_error(f"Error during emergency save: {e}")
        
        sys.exit(0)
    
    def _transcribe_existing_file(self) -> None:
        """Handle transcription of existing files."""
        selected_file = self.file_manager.select_audio_file(self.menu_system)
        if selected_file:
            self.workflow_orchestrator.transcribe_file(selected_file)
        self.menu_system.wait_for_enter()
    
    def _show_storage_info(self) -> None:
        """Show storage information."""
        self.file_manager.display_storage_info(self.menu_system)
    
    def run(self):
        """Run the main CLI application."""
        self.menu_system.display_welcome()
        
        # Define menu handlers
        menu_handlers: Dict[str, Callable] = {
            '1': lambda: self.workflow_orchestrator.record_audio(auto_transcribe=False),
            '2': lambda: self.workflow_orchestrator.record_audio(auto_transcribe=True),
            '3': self._transcribe_existing_file,
            '4': self.workflow_orchestrator.list_audio_devices,
            '5': self.workflow_orchestrator.select_audio_device,
            '6': self.config_handlers.configure_audio_settings,
            '7': self.config_handlers.configure_azure_openai,
            '8': self.config_handlers.view_current_settings,
            '9': self.workflow_orchestrator.test_azure_connection,
            's': self._show_storage_info,  # Hidden storage info option
            'h': self.menu_system.display_help,
            'help': self.menu_system.display_help,
        }
        
        # Run the menu loop
        self.menu_system.run_menu_loop(menu_handlers)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="IELTS Speaking Audio Recorder - Record audio and transcribe with Azure OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli                    # Start interactive mode
  python -m src.cli --quick            # Quick record mode
  python -m src.cli --devices          # List audio devices
  python -m src.cli --config           # Configure settings
  python -m src.cli --azure-config     # Configure Azure OpenAI
        """
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick recording mode (start recording immediately)'
    )
    
    parser.add_argument(
        '--transcribe', '-t',
        action='store_true',
        help='Quick record and transcribe mode'
    )
    
    parser.add_argument(
        '--devices', '-d',
        action='store_true',
        help='List available audio devices and exit'
    )
    
    parser.add_argument(
        '--config', '-c',
        action='store_true',
        help='Open audio configuration menu and exit'
    )
    
    parser.add_argument(
        '--azure-config', '-a',
        action='store_true',
        help='Open Azure OpenAI configuration menu and exit'
    )
    
    parser.add_argument(
        '--test-azure',
        action='store_true',
        help='Test Azure OpenAI connection and exit'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output filename for recording (WAV format)'
    )
    
    return parser


def main():
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()
    
    cli = AudioRecorderCLI()
    
    try:
        if args.devices:
            # List devices and exit
            cli.workflow_orchestrator.list_audio_devices()
            
        elif args.config:
            # Configure audio settings and exit
            cli.config_handlers.configure_audio_settings()
            
        elif args.azure_config:
            # Configure Azure OpenAI and exit
            cli.config_handlers.configure_azure_openai()
            
        elif args.test_azure:
            # Test Azure connection and exit
            cli.workflow_orchestrator.test_azure_connection()
            
        elif args.quick or args.transcribe:
            # Quick recording mode (with optional transcription)
            if not cli.workflow_orchestrator.initialize_recorder():
                sys.exit(1)
            
            mode_text = "Quick Record + Transcribe" if args.transcribe else "Quick Recording"
            cli.menu_system.display_info(f"{mode_text} Mode")
            cli.menu_system.display_info("Press Ctrl+C to stop recording")
            
            try:
                # Start recording immediately
                cli.workflow_orchestrator.recorder.start_recording()
                cli.menu_system.display_info("🔴 Recording started. Press Ctrl+C to stop...")
                
                # Wait for interrupt
                while cli.workflow_orchestrator.recorder.is_recording:
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                cli.menu_system.display_info("🛑 Stopping recording...")
                audio_data = cli.workflow_orchestrator.recorder.stop_recording()
                
                if len(audio_data) > 0:
                    # Determine output path
                    if args.output:
                        output_path = Path(args.output)
                        if not str(output_path).lower().endswith('.wav'):
                            output_path = output_path.with_suffix('.wav')
                    else:
                        output_path = None
                    
                    # Save recording
                    saved_path = cli.workflow_orchestrator.recorder.save_recording(audio_data, output_path)
                    cli.menu_system.display_success(f"Recording saved: {saved_path}")
                    
                    # Transcribe if requested
                    if args.transcribe:
                        if cli.config_manager.is_azure_configured():
                            cli.workflow_orchestrator.transcribe_file(saved_path)
                        else:
                            cli.menu_system.display_warning("Azure OpenAI not configured for transcription.")
                            if cli.menu_system.get_yes_no_choice("Configure now?"):
                                cli.config_handlers.configure_azure_openai()
                else:
                    cli.menu_system.display_warning("No audio data recorded")
        else:
            # Interactive mode
            cli.run()
            
    except Exception as e:
        cli.menu_system.display_error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()