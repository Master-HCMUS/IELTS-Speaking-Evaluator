"""
Workflow orchestrator for audio recording and transcription operations.

This module contains the core business logic for coordinating recording,
transcription, and file management operations.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from ..audio_recorder import AudioRecorder
from ..transcription_service import AzureOpenAITranscriptionService
from ..config_manager import ConfigManager
from ..ui.menu_system import MenuSystem
from ..ui.display_helpers import DisplayHelpers
from ..exceptions import (
    AudioRecordingError,
    AudioDeviceError,
    ConfigurationError,
    TranscriptionError,
    AzureAuthenticationError,
    AzureAPIError
)


class WorkflowOrchestrator:
    """
    Orchestrates recording and transcription workflows.
    
    This class coordinates the audio recording, transcription, and file
    management operations while keeping the business logic separate
    from the UI concerns.
    """
    
    def __init__(self, config_manager: ConfigManager, menu_system: MenuSystem):
        """
        Initialize the workflow orchestrator.
        
        Args:
            config_manager: Configuration manager instance
            menu_system: Menu system for user interactions
        """
        self.config_manager = config_manager
        self.menu_system = menu_system
        self.recorder: Optional[AudioRecorder] = None
        self.transcription_service: Optional[AzureOpenAITranscriptionService] = None
    
    def initialize_recorder(self) -> bool:
        """
        Initialize the audio recorder with current configuration.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            config = self.config_manager.get_audio_config()
            self.recorder = AudioRecorder(
                sample_rate=config['sample_rate'],
                channels=config['channels'],
                dtype=config['dtype']
            )
            return True
        except (AudioDeviceError, ConfigurationError) as e:
            self.menu_system.display_error(f"Failed to initialize audio recorder: {e}")
            return False
        except Exception as e:
            self.menu_system.display_error(f"Unexpected error during initialization: {e}")
            return False
    
    def initialize_transcription_service(self) -> bool:
        """
        Initialize the Azure OpenAI transcription service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            azure_config = self.config_manager.get_azure_openai_config()
            
            if not azure_config.get('endpoint'):
                self.menu_system.display_error("Azure OpenAI endpoint not configured. Please configure Azure settings first.")
                return False
            
            if not azure_config.get('api_key'):
                self.menu_system.display_error("Azure OpenAI API key not configured. Please add your API key to the .env file.")
                return False
            
            self.transcription_service = AzureOpenAITranscriptionService(
                endpoint=azure_config['endpoint'],
                api_key=azure_config['api_key'],
                deployment_name=azure_config['deployment_name'],
                api_version=azure_config['api_version']
            )
            return True
            
        except AzureAuthenticationError as e:
            self.menu_system.display_error(f"Azure authentication failed: {e}")
            self.menu_system.display_info("Please check your API key in the .env file and verify your Azure OpenAI resource settings.")
            return False
        except Exception as e:
            self.menu_system.display_error(f"Failed to initialize transcription service: {e}")
            return False
    
    def record_audio(self, auto_transcribe: bool = False) -> None:
        """
        Handle the audio recording workflow.
        
        Args:
            auto_transcribe: Whether to automatically transcribe after recording
        """
        if not self.recorder and not self.initialize_recorder():
            return
        
        try:
            self.menu_system.display_info("Preparing to record...")
            self.menu_system.wait_for_enter("Press Enter to start recording: ")
            
            # Start recording
            self.recorder.start_recording()
            
            # Wait for user to stop recording
            self.menu_system.wait_for_enter("🔴 Recording... Press Enter to stop: ")
            
            # Stop recording and get audio data
            audio_data = self.recorder.stop_recording()
            
            if len(audio_data) > 0:
                saved_path = self._save_recording(audio_data)
                
                if saved_path:
                    # Handle transcription if requested
                    transcription_settings = self.config_manager.get_transcription_settings()
                    should_transcribe = auto_transcribe or transcription_settings.get('auto_transcribe', False)
                    
                    if should_transcribe and self.config_manager.is_azure_configured():
                        self.transcribe_file(saved_path)
                    elif should_transcribe:
                        self.menu_system.display_warning("Auto-transcription enabled but Azure OpenAI not configured.")
                        if self.menu_system.get_yes_no_choice("Configure Azure OpenAI now?"):
                            from .configuration_handlers import ConfigurationHandlers
                            config_handlers = ConfigurationHandlers(self.config_manager, self.menu_system)
                            config_handlers.configure_azure_openai()
                    
                    # Ask if user wants to record another
                    if self.menu_system.get_yes_no_choice("Record another?"):
                        self.record_audio(auto_transcribe)
            else:
                self.menu_system.display_warning("No audio data was recorded.")
                
        except AudioRecordingError as e:
            self.menu_system.display_error(f"Recording error: {e}")
        except KeyboardInterrupt:
            self.menu_system.display_warning("Recording interrupted by user")
            self._handle_interrupted_recording()
        except Exception as e:
            self.menu_system.display_error(f"Unexpected error: {e}")
    
    def _save_recording(self, audio_data) -> Optional[Path]:
        """
        Save the recorded audio with user-specified filename.
        
        Args:
            audio_data: The recorded audio data
            
        Returns:
            Path or None: The saved file path or None if saving failed
        """
        try:
            self.menu_system.display_info("Saving recording...")
            filename = self.menu_system.get_user_input("Enter filename (or press Enter for auto-generated): ")
            
            if filename:
                # Add .wav extension if not present
                if not filename.lower().endswith('.wav'):
                    filename += '.wav'
                output_path = Path("recordings") / filename
            else:
                output_path = None
            
            # Save the recording
            saved_path = self.recorder.save_recording(audio_data, output_path)
            self.menu_system.display_success(f"Recording saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            self.menu_system.display_error(f"Failed to save recording: {e}")
            return None
    
    def _handle_interrupted_recording(self) -> None:
        """Handle recording interruption gracefully."""
        if self.recorder and self.recorder.is_recording:
            try:
                audio_data = self.recorder.stop_recording()
                if len(audio_data) > 0:
                    save_path = self.recorder.save_recording(audio_data)
                    self.menu_system.display_success(f"Recording saved: {save_path}")
            except Exception as e:
                self.menu_system.display_error(f"Error saving interrupted recording: {e}")
    
    def transcribe_file(self, file_path: Path) -> None:
        """
        Transcribe an audio file using Azure OpenAI Whisper.
        
        Args:
            file_path: Path to the audio file to transcribe
        """
        if not self.transcription_service and not self.initialize_transcription_service():
            return
        
        try:
            self.menu_system.display_info(f"Transcribing {file_path.name}...")
            
            # Get transcription settings
            settings = self.config_manager.get_transcription_settings()
            language = settings.get('language', 'auto')
            if language == 'auto':
                language = None  # Let Whisper auto-detect
            
            # Transcribe the file
            result = self.transcription_service.transcribe_audio_file(file_path, language)
            
            # Display results
            DisplayHelpers.display_transcription_result(result)
            
            # Offer to save transcription
            if self.menu_system.get_yes_no_choice("Save transcription to file?"):
                self._save_transcription(result, file_path)
                
        except TranscriptionError as e:
            self.menu_system.display_error(f"Transcription error: {e}")
        except AzureAPIError as e:
            self.menu_system.display_error(f"Azure API error: {e}")
        except Exception as e:
            self.menu_system.display_error(f"Unexpected transcription error: {e}")
    
    def _save_transcription(self, result: Dict[str, Any], audio_file_path: Path) -> None:
        """
        Save transcription results to a text file.
        
        Args:
            result: Transcription result dictionary
            audio_file_path: Path to the original audio file
        """
        try:
            from .file_manager import FileManager
            file_manager = FileManager()
            
            transcript_path = file_manager.save_transcription(result, audio_file_path)
            self.menu_system.display_success(f"Transcription saved to: {transcript_path}")
            
        except Exception as e:
            self.menu_system.display_error(f"Error saving transcription: {e}")
    
    def list_audio_devices(self) -> None:
        """List available audio input devices."""
        if not self.recorder and not self.initialize_recorder():
            return
        
        self.recorder.list_audio_devices()
        self.menu_system.wait_for_enter()
    
    def select_audio_device(self) -> None:
        """Allow user to select a specific audio input device."""
        if not self.recorder and not self.initialize_recorder():
            return
        
        self.menu_system.display_section_header("Audio Device Selection")
        
        # Show current device
        if hasattr(self.recorder, 'selected_device'):
            current_info = self.recorder.get_recording_info()
            self.menu_system.display_info(f"Current device: {current_info.get('selected_device', 'Unknown')}")
        
        # List all devices
        self.recorder.list_audio_devices()
        
        device_id = self.menu_system.get_numeric_choice(
            "Enter device ID to select (or press Enter to cancel): ",
            min_val=0,
            max_val=100  # Reasonable upper limit
        )
        
        if device_id is not None:
            if self.recorder.set_device(device_id):
                self.menu_system.display_success("Device selection successful!")
                # Reinitialize recorder with new device
                self.recorder = None
            else:
                self.menu_system.display_error("Device selection failed!")
        else:
            self.menu_system.display_info("Device selection cancelled.")
        
        self.menu_system.wait_for_enter()
    
    def test_azure_connection(self) -> None:
        """Test the Azure OpenAI connection with enhanced diagnostics."""
        self.menu_system.display_section_header("Testing Azure OpenAI Connection")
        
        if not self.config_manager.is_azure_configured():
            self.menu_system.display_error("Azure OpenAI not configured. Please configure it first.")
            self.menu_system.wait_for_enter()
            return
        
        try:
            # Initialize or get transcription service
            if not self.transcription_service and not self.initialize_transcription_service():
                self.menu_system.wait_for_enter()
                return
            
            # Test connection
            success = self.transcription_service.test_connection()
            
            if success:
                self.menu_system.display_success("Azure OpenAI connection test successful!")
                
                # Show service info
                info = self.transcription_service.get_service_info()
                print(f"\nService Information:")
                print(f"  Endpoint: {info['endpoint']}")
                print(f"  Deployment: {info['deployment_name']}")
                print(f"  API Version: {info['api_version']}")
                print(f"  Max File Size: {info['max_file_size_mb']} MB")
            else:
                self.menu_system.display_error("Azure OpenAI connection test failed.")
                self.menu_system.display_info("Check your endpoint, deployment name, and authentication.")
                
        except Exception as e:
            error_message = str(e).lower()
            self.menu_system.display_error(f"Connection test error: {e}")
            
            # Provide tenant-specific diagnostics if it's a tenant issue
            if "tenant" in error_message and "does not match" in error_message:
                self.menu_system.display_warning("\nTenant mismatch detected. Running diagnostics...")
                
                try:
                    diagnostics = self.transcription_service.diagnose_tenant_issue()
                    
                    print(f"\n🔍 Diagnostic Results:")
                    print(f"Status: {diagnostics['status']}")
                    
                    if diagnostics.get('current_tenant'):
                        print(f"Current Azure CLI Tenant: {diagnostics['current_tenant']}")
                    if diagnostics.get('subscription_id'):
                        print(f"Current Subscription: {diagnostics['subscription_id']}")
                    if diagnostics.get('account_name'):
                        print(f"Account: {diagnostics['account_name']}")
                    
                    print(f"\n💡 Suggested Solutions:")
                    for suggestion in diagnostics.get('suggestions', []):
                        print(f"  • {suggestion}")
                        
                except Exception as diag_error:
                    self.menu_system.display_warning(f"Could not run diagnostics: {diag_error}")
            
            elif "authentication" in error_message or "credential" in error_message:
                print(f"\n💡 Authentication troubleshooting:")
                print(f"  • Run: az login")
                print(f"  • Check: az account show")
                print(f"  • Verify you have access to the Azure OpenAI resource")
                print(f"  • Ensure your endpoint URL is correct")
        
        self.menu_system.wait_for_enter()