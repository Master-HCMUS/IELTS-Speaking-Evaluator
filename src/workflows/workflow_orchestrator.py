"""
Workflow orchestrator for audio recording and transcription operations.

This module contains the core business logic for coordinating recording,
transcription, and file management operations.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from ..audio_recorder import AudioRecorder
from ..transcription_service import AzureOpenAITranscriptionService
from ..pronunciation_service import AzureSpeechPronunciationService
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
        self.pronunciation_service: Optional[AzureSpeechPronunciationService] = None
    
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
    
    def initialize_pronunciation_service(self) -> bool:
        """
        Initialize the Azure Speech pronunciation service.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            azure_speech_config = self.config_manager.get_azure_speech_config()
            
            if not azure_speech_config.get('speech_key'):
                self.menu_system.display_error("Azure Speech API key not configured. Please add your API key to the .env file.")
                return False
            
            if not azure_speech_config.get('speech_region'):
                self.menu_system.display_error("Azure Speech region not configured. Please configure Azure Speech settings first.")
                return False
            
            self.pronunciation_service = AzureSpeechPronunciationService(
                speech_key=azure_speech_config['speech_key'],
                speech_region=azure_speech_config['speech_region'],
                locale=azure_speech_config.get('locale', 'en-US')
            )
            return True
            
        except Exception as e:
            self.menu_system.display_error(f"Failed to initialize pronunciation service: {e}")
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
    
    def assess_pronunciation(self, audio_file_path: Path, reference_text: str = None) -> None:
        """
        Assess pronunciation of an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            reference_text: Reference text for pronunciation assessment
        """
        if not self.pronunciation_service and not self.initialize_pronunciation_service():
            return
        
        try:
            # If no reference text provided, try to get transcription first
            if not reference_text:
                if not self.transcription_service and not self.initialize_transcription_service():
                    self.menu_system.display_error("Need reference text for pronunciation assessment. Please transcribe the audio first.")
                    return
                
                self.menu_system.display_info("Getting transcription for pronunciation reference...")
                transcription_result = self.transcription_service.transcribe_audio_file(audio_file_path)
                reference_text = transcription_result.get('text', '').strip()
                
                if not reference_text:
                    self.menu_system.display_error("Could not get transcription for pronunciation assessment.")
                    return
            
            self.menu_system.display_info("Assessing pronunciation...")
            self.menu_system.display_info(f"Reference text: {reference_text}")
            
            # Perform pronunciation assessment
            result = self.pronunciation_service.assess_pronunciation_from_file(audio_file_path, reference_text)
            
            if result and 'overall_scores' in result:
                self._display_pronunciation_results(result)
                self._save_pronunciation_results(audio_file_path, result)
            else:
                self.menu_system.display_error("Pronunciation assessment failed or returned no results.")
                
        except Exception as e:
            self.menu_system.display_error(f"Pronunciation assessment error: {e}")
    
    def _display_pronunciation_results(self, result: Dict[str, Any]) -> None:
        """Display pronunciation assessment results."""
        self.menu_system.display_section_header("Pronunciation Assessment Results")
        
        # Overall scores
        overall_scores = result.get('overall_scores', {})
        accuracy = overall_scores.get('accuracy_score', 0)
        fluency = overall_scores.get('fluency_score', 0)
        completeness = overall_scores.get('completeness_score', 0)
        pronunciation_score = overall_scores.get('pronunciation_score', 0)
        
        self.menu_system.display_info(f"Overall Pronunciation Score: {pronunciation_score:.1f}/100")
        self.menu_system.display_info(f"Accuracy Score: {accuracy:.1f}/100")
        self.menu_system.display_info(f"Fluency Score: {fluency:.1f}/100")
        self.menu_system.display_info(f"Completeness Score: {completeness:.1f}/100")
        
        # Show recognized text vs reference
        recognized_text = result.get('recognized_text', '')
        reference_text = result.get('reference_text', '')
        if recognized_text and reference_text:
            self.menu_system.display_section_header("Text Comparison")
            self.menu_system.display_info(f"Reference:  {reference_text}")
            self.menu_system.display_info(f"Recognized: {recognized_text}")
        
        # Word-level details
        word_scores = result.get('word_level_scores', [])
        if word_scores:
            self.menu_system.display_section_header("Word-level Analysis")
            for word_info in word_scores:
                word = word_info.get('word', 'Unknown')
                word_accuracy = word_info.get('accuracy_score', 0)
                error_type = word_info.get('error_type', 'None')
                
                status = "✓" if error_type == "None" else "✗"
                self.menu_system.display_info(f"{status} {word}: {word_accuracy:.1f} ({error_type})")
        
        # Feedback
        feedback = result.get('detailed_feedback', [])
        if feedback:
            self.menu_system.display_section_header("Feedback")
            for item in feedback:
                self.menu_system.display_info(f"• {item}")
    
    def _save_pronunciation_results(self, audio_file_path: Path, result: Dict[str, Any]) -> None:
        """Save pronunciation assessment results."""
        try:
            # Create pronunciation results filename
            audio_stem = audio_file_path.stem
            results_filename = f"{audio_stem}_pronunciation.txt"
            results_path = Path("transcriptions") / results_filename
            
            # Ensure transcriptions directory exists
            results_path.parent.mkdir(exist_ok=True)
            
            # Format results for saving
            overall_scores = result.get('overall_scores', {})
            content = f"Pronunciation Assessment Results\n"
            content += f"{'=' * 40}\n\n"
            content += f"Audio File: {audio_file_path.name}\n"
            content += f"Processing Time: {result.get('processing_time', 0):.2f} seconds\n"
            content += f"Recognition Status: {result.get('recognition_status', 'Unknown')}\n\n"
            
            # Text comparison
            reference_text = result.get('reference_text', '')
            recognized_text = result.get('recognized_text', '')
            if reference_text or recognized_text:
                content += f"Text Comparison:\n"
                content += f"- Reference:  {reference_text}\n"
                content += f"- Recognized: {recognized_text}\n\n"
            
            content += f"Overall Scores:\n"
            content += f"- Pronunciation Score: {overall_scores.get('pronunciation_score', 0):.1f}/100\n"
            content += f"- Accuracy Score: {overall_scores.get('accuracy_score', 0):.1f}/100\n"
            content += f"- Fluency Score: {overall_scores.get('fluency_score', 0):.1f}/100\n"
            content += f"- Completeness Score: {overall_scores.get('completeness_score', 0):.1f}/100\n\n"
            
            # Word-level details
            word_scores = result.get('word_level_scores', [])
            if word_scores:
                content += f"Word-level Analysis:\n"
                for word_info in word_scores:
                    word = word_info.get('word', 'Unknown')
                    word_accuracy = word_info.get('accuracy_score', 0)
                    error_type = word_info.get('error_type', 'None')
                    content += f"- {word:12}: {word_accuracy:5.1f} ({error_type})\n"
                content += "\n"
            
            # Feedback
            feedback = result.get('detailed_feedback', [])
            if feedback:
                content += f"Feedback:\n"
                for item in feedback:
                    content += f"- {item}\n"
            
            # Save to file
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.menu_system.display_success(f"Pronunciation results saved: {results_path}")
            
        except Exception as e:
            self.menu_system.display_error(f"Error saving pronunciation results: {e}")
    
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
    
    def comprehensive_assessment(self, audio_file_path: Path) -> None:
        """
        Perform comprehensive assessment including transcription and pronunciation.
        
        Args:
            audio_file_path: Path to the audio file
        """
        self.menu_system.display_section_header("Comprehensive Speech Assessment")
        self.menu_system.display_info(f"Analyzing: {audio_file_path.name}")
        
        try:
            # Step 1: Transcription
            if not self.transcription_service and not self.initialize_transcription_service():
                return
            
            self.menu_system.display_info("Step 1: Getting transcription...")
            transcription_result = self.transcription_service.transcribe_audio_file(audio_file_path)
            
            if not transcription_result or not transcription_result.get('text'):
                self.menu_system.display_error("Transcription failed. Cannot proceed with pronunciation assessment.")
                return
            
            transcribed_text = transcription_result['text'].strip()
            self.menu_system.display_success("Transcription completed!")
            self.menu_system.display_info(f"Transcribed text: {transcribed_text}")
            
            # Save transcription
            self._save_transcription_results(audio_file_path, transcription_result)
            
            # Step 2: Pronunciation Assessment
            if not self.pronunciation_service and not self.initialize_pronunciation_service():
                self.menu_system.display_warning("Pronunciation service not available. Transcription completed only.")
                return
            
            self.menu_system.display_info("Step 2: Assessing pronunciation...")
            pronunciation_result = self.pronunciation_service.assess_pronunciation_from_file(audio_file_path, transcribed_text)
            
            if pronunciation_result and 'overall_scores' in pronunciation_result:
                self.menu_system.display_success("Pronunciation assessment completed!")
                self._display_pronunciation_results(pronunciation_result)
                self._save_pronunciation_results(audio_file_path, pronunciation_result)
            else:
                self.menu_system.display_warning("Pronunciation assessment failed, but transcription was successful.")
            
            self.menu_system.display_section_header("Assessment Complete")
            self.menu_system.display_success("Comprehensive assessment finished!")
            
        except Exception as e:
            self.menu_system.display_error(f"Comprehensive assessment error: {e}")
        
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