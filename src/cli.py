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
    
    def _assess_pronunciation_file(self) -> None:
        """Handle pronunciation assessment of existing files."""
        selected_file = self.file_manager.select_audio_file(self.menu_system)
        if selected_file:
            # Ask for reference text
            reference_text = self.menu_system.get_user_input(
                "Enter reference text (or press Enter to use transcription): "
            )
            reference_text = reference_text if reference_text.strip() else None
            
            self.workflow_orchestrator.assess_pronunciation(selected_file, reference_text)
        self.menu_system.wait_for_enter()
    
    def _comprehensive_assessment_file(self) -> None:
        """Handle comprehensive assessment (transcription + pronunciation) of existing files."""
        selected_file = self.file_manager.select_audio_file(self.menu_system)
        if selected_file:
            self.workflow_orchestrator.comprehensive_assessment(selected_file)
        self.menu_system.wait_for_enter()
    
    def _run_dataset_evaluation(self) -> None:
        """Run evaluation against SpeechOcean762 dataset."""
        self.menu_system.display_section_header("SpeechOcean762 Dataset Evaluation")
        
        try:
            from .evaluation.dataset_evaluator import SpeechOcean762Evaluator
            
            # Check if datasets library is available
            try:
                import datasets
            except ImportError:
                self.menu_system.display_error("HuggingFace datasets library not installed.")
                self.menu_system.display_info("Install with: pip install datasets")
                return
            
            # Check Azure Speech configuration
            if not self.config_manager.is_speech_configured():
                self.menu_system.display_error("Azure Speech service not configured.")
                self.menu_system.display_info("Please configure Azure Speech settings first.")
                return
            
            # Get evaluation parameters
            max_samples = self.menu_system.get_numeric_choice(
                "Number of samples to evaluate (Enter for all, or specify number): ",
                min_val=1,
                max_val=10000
            )
            
            # Initialize evaluator
            evaluator = SpeechOcean762Evaluator(self.config_manager)
            
            # Load dataset
            self.menu_system.display_info("Loading SpeechOcean762 dataset...")
            if not evaluator.load_dataset(split="test", max_samples=max_samples):
                self.menu_system.display_error("Failed to load dataset")
                return
            
            # Confirm evaluation
            total_samples = len(evaluator.dataset)
            if not self.menu_system.get_yes_no_choice(
                f"Run evaluation on {total_samples} samples? This may take a while."
            ):
                return
            
            # Run evaluation
            self.menu_system.display_info("Running evaluation... This may take several minutes.")
            metrics = evaluator.run_evaluation(max_samples=max_samples, save_results=True)
            
            # Show results
            evaluator.print_evaluation_summary(metrics)
            self.menu_system.display_success("Evaluation completed! Results saved to 'evaluation_results/' directory.")
            
        except ImportError as e:
            self.menu_system.display_error(f"Evaluation module not available: {e}")
        except Exception as e:
            self.menu_system.display_error(f"Evaluation failed: {e}")
    
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
            '4': self._assess_pronunciation_file,
            '5': self._comprehensive_assessment_file,
            '6': self._run_dataset_evaluation,
            '7': self.workflow_orchestrator.list_audio_devices,
            '8': self.workflow_orchestrator.select_audio_device,
            '9': self.config_handlers.configure_audio_settings,
            '10': self.config_handlers.configure_azure_openai,
            '11': self.config_handlers.view_current_settings,
            '12': self.workflow_orchestrator.test_azure_connection,
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
  python -m src.cli                              # Start interactive mode
  python -m src.cli --quick                      # Quick record mode
  python -m src.cli --transcribe                 # Quick record and transcribe
  python -m src.cli --assess-pronunciation file.wav  # Assess pronunciation of audio file
  python -m src.cli --comprehensive file.wav     # Full assessment (transcription + pronunciation)
  python -m src.cli --evaluate-dataset           # Evaluate against SpeechOcean762 dataset
  python -m src.cli --evaluate-dataset --max-samples 100  # Evaluate on 100 samples
  python -m src.cli --devices                    # List audio devices
  python -m src.cli --config                     # Configure settings
  python -m src.cli --azure-config               # Configure Azure OpenAI
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
        '--assess-pronunciation', '-p',
        type=str,
        metavar='AUDIO_FILE',
        help='Assess pronunciation of an audio file'
    )
    
    parser.add_argument(
        '--comprehensive', '-comp',
        type=str,
        metavar='AUDIO_FILE',
        help='Perform comprehensive assessment (transcription + pronunciation) of an audio file'
    )
    
    parser.add_argument(
        '--reference-text', '-r',
        type=str,
        help='Reference text for pronunciation assessment (if not provided, will use transcription)'
    )
    
    parser.add_argument(
        '--evaluate-dataset', '-eval',
        action='store_true',
        help='Evaluate pronunciation assessment against SpeechOcean762 dataset'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for --evaluate-dataset)'
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
            
        elif args.assess_pronunciation:
            # Assess pronunciation of audio file
            audio_file = Path(args.assess_pronunciation)
            if not audio_file.exists():
                cli.menu_system.display_error(f"Audio file not found: {audio_file}")
                sys.exit(1)
            
            reference_text = args.reference_text if args.reference_text else None
            cli.workflow_orchestrator.assess_pronunciation(audio_file, reference_text)
            
        elif args.comprehensive:
            # Comprehensive assessment (transcription + pronunciation)
            audio_file = Path(args.comprehensive)
            if not audio_file.exists():
                cli.menu_system.display_error(f"Audio file not found: {audio_file}")
                sys.exit(1)
            
            cli.workflow_orchestrator.comprehensive_assessment(audio_file)
            
        elif args.evaluate_dataset:
            # Evaluate pronunciation assessment against SpeechOcean762 dataset
            try:
                from .evaluation.dataset_evaluator import SpeechOcean762Evaluator
                
                # Check if datasets library is available
                try:
                    import datasets
                except ImportError:
                    cli.menu_system.display_error("HuggingFace datasets library not installed.")
                    cli.menu_system.display_info("Install with: pip install datasets pandas matplotlib")
                    sys.exit(1)
                
                # Check Azure Speech configuration
                if not cli.config_manager.is_speech_configured():
                    cli.menu_system.display_error("Azure Speech service not configured.")
                    cli.menu_system.display_info("Please configure Azure Speech settings first.")
                    sys.exit(1)
                
                # Initialize evaluator
                evaluator = SpeechOcean762Evaluator(cli.config_manager)
                
                # Load dataset
                cli.menu_system.display_info("Loading SpeechOcean762 dataset...")
                if not evaluator.load_dataset(split="test", max_samples=args.max_samples):
                    cli.menu_system.display_error("Failed to load dataset")
                    sys.exit(1)
                
                # Run evaluation
                total_samples = len(evaluator.dataset)
                cli.menu_system.display_info(f"Running evaluation on {total_samples} samples...")
                metrics = evaluator.run_evaluation(max_samples=args.max_samples, save_results=True)
                
                # Show results
                evaluator.print_evaluation_summary(metrics)
                cli.menu_system.display_success("Evaluation completed! Results saved to 'evaluation_results/' directory.")
                
            except ImportError as e:
                cli.menu_system.display_error(f"Evaluation module not available: {e}")
                sys.exit(1)
            except Exception as e:
                cli.menu_system.display_error(f"Evaluation failed: {e}")
                sys.exit(1)
            
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