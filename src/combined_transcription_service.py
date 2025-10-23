"""
Combined transcription service that can use either Azure OpenAI or local fine-tuned models.

This module provides a unified interface for transcription that automatically selects
between Azure OpenAI Whisper and local fine-tuned Whisper models based on configuration.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

# Handle both relative and absolute imports
try:
    from .config_manager import ConfigManager
    from .transcription_service import AzureOpenAITranscriptionService
    from .local_transcription_service import LocalWhisperTranscriptionService
    from .exceptions import TranscriptionError, ConfigurationError
except ImportError:
    from config_manager import ConfigManager
    from transcription_service import AzureOpenAITranscriptionService
    from local_transcription_service import LocalWhisperTranscriptionService
    from exceptions import TranscriptionError, ConfigurationError


class CombinedTranscriptionService:
    """
    Combined transcription service that can use Azure OpenAI or local models.
    
    This service automatically selects the appropriate transcription backend based on
    configuration preferences and availability.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the combined transcription service.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.azure_service: Optional[AzureOpenAITranscriptionService] = None
        self.local_service: Optional[LocalWhisperTranscriptionService] = None
        self.active_service = None
        self.active_service_type = None
        
        # Initialize the appropriate service
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """
        Initialize the appropriate transcription service based on configuration.
        
        Raises:
            ConfigurationError: If no valid transcription service can be initialized
        """
        # Check if we should use local Whisper
        if self.config_manager.should_use_local_whisper():
            try:
                self._initialize_local_service()
                return
            except Exception as e:
                print(f"⚠️  Failed to initialize local Whisper service: {e}")
                print("Falling back to Azure OpenAI...")
        
        # Try to initialize Azure OpenAI service
        if self.config_manager.is_azure_configured():
            try:
                self._initialize_azure_service()
                return
            except Exception as e:
                print(f"⚠️  Failed to initialize Azure OpenAI service: {e}")
                
                # If Azure fails and local is available, try local as fallback
                if self.config_manager.is_local_whisper_configured():
                    print("Falling back to local Whisper...")
                    try:
                        self._initialize_local_service()
                        return
                    except Exception as local_e:
                        print(f"⚠️  Local Whisper fallback also failed: {local_e}")
        
        # If we reach here, no service could be initialized
        available_services = []
        if self.config_manager.is_azure_configured():
            available_services.append("Azure OpenAI")
        if self.config_manager.is_local_whisper_configured():
            available_services.append("Local Whisper")
        
        if not available_services:
            raise ConfigurationError(
                "No transcription service is configured. Please configure either:\n"
                "1. Azure OpenAI Whisper (endpoint, API key, deployment)\n"
                "2. Local fine-tuned Whisper model"
            )
        else:
            raise TranscriptionError(
                f"Failed to initialize any transcription service. "
                f"Available services: {', '.join(available_services)}"
            )
    
    def _initialize_azure_service(self) -> None:
        """Initialize Azure OpenAI transcription service."""
        azure_config = self.config_manager.get_azure_openai_config()
        
        self.azure_service = AzureOpenAITranscriptionService(
            endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            deployment_name=azure_config["deployment_name"],
            api_version=azure_config["api_version"]
        )
        
        self.active_service = self.azure_service
        self.active_service_type = "azure"
        
        print(f"🌐 Using Azure OpenAI Whisper for transcription")
    
    def _initialize_local_service(self) -> None:
        """Initialize local Whisper transcription service."""
        whisper_config = self.config_manager.get_local_whisper_config()
        
        self.local_service = LocalWhisperTranscriptionService(
            model_path=whisper_config["model_path"],
            device=whisper_config["device"]
        )
        
        self.active_service = self.local_service
        self.active_service_type = "local"
        
        print(f"🤖 Using local fine-tuned Whisper for transcription")
    
    def transcribe_audio_file(self, file_path: Union[str, Path], language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file using the active transcription service.
        
        Args:
            file_path: Path to the audio file to transcribe
            language: Optional language code (e.g., 'en', 'es', 'fr'). If None, auto-detect
            
        Returns:
            Dict containing transcription results with text, confidence, and metadata
            
        Raises:
            TranscriptionError: If no active service or transcription fails
        """
        if not self.active_service:
            raise TranscriptionError("No active transcription service")
        
        # Use configured language if not specified
        if language is None:
            if self.active_service_type == "azure":
                azure_config = self.config_manager.get_azure_openai_config()
                language = azure_config.get("language", "auto")
            else:
                whisper_config = self.config_manager.get_local_whisper_config()
                language = whisper_config.get("language", "auto")
        
        # Convert "auto" to None for API compatibility
        if language == "auto":
            language = None
        # Add service type to result
        result = self.active_service.transcribe_audio_file(file_path, language)
        result["service_type"] = self.active_service_type
        result["service_info"] = self.get_service_info()
        
        return result
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the active transcription service.
        
        Returns:
            Dict containing test results
        """
        if not self.active_service:
            return {
                "status": "error",
                "error": "No active transcription service"
            }
        
        if self.active_service_type == "azure":
            # For Azure, we'll do a simple connectivity test
            try:
                # Test by trying to access the client
                client = self.active_service.client
                return {
                    "status": "success",
                    "service_type": "azure",
                    "endpoint": self.active_service.endpoint,
                    "deployment": self.active_service.deployment_name
                }
            except Exception as e:
                return {
                    "status": "error",
                    "service_type": "azure",
                    "error": str(e)
                }
        else:
            # For local, use the built-in test method
            result = self.active_service.test_connection()
            result["service_type"] = "local"
            return result
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the active transcription service.
        
        Returns:
            Dict containing service information
        """
        if not self.active_service:
            return {"status": "no_active_service"}
        
        info = {
            "service_type": self.active_service_type,
            "status": "active"
        }
        
        if self.active_service_type == "azure":
            azure_config = self.config_manager.get_azure_openai_config()
            info.update({
                "endpoint": self.active_service.endpoint,
                "deployment": self.active_service.deployment_name,
                "api_version": self.active_service.api_version,
                "language": azure_config.get("language", "auto")
            })
        else:
            info.update(self.active_service.get_model_info())
            whisper_config = self.config_manager.get_local_whisper_config()
            info["language"] = whisper_config.get("language", "auto")
        
        return info
    
    def switch_to_azure(self) -> bool:
        """
        Switch to Azure OpenAI transcription service.
        
        Returns:
            bool: True if successfully switched to Azure
        """
        try:
            if self.config_manager.is_azure_configured():
                self._initialize_azure_service()
                return True
            else:
                print("❌ Azure OpenAI is not configured")
                return False
        except Exception as e:
            print(f"❌ Failed to switch to Azure OpenAI: {e}")
            return False
    
    def switch_to_local(self) -> bool:
        """
        Switch to local Whisper transcription service.
        
        Returns:
            bool: True if successfully switched to local
        """
        try:
            if self.config_manager.is_local_whisper_configured():
                self._initialize_local_service()
                return True
            else:
                print("❌ Local Whisper is not configured")
                return False
        except Exception as e:
            print(f"❌ Failed to switch to local Whisper: {e}")
            return False
    
    def get_available_services(self) -> Dict[str, bool]:
        """
        Get information about which services are available.
        
        Returns:
            Dict with service availability
        """
        return {
            "azure": self.config_manager.is_azure_configured(),
            "local": self.config_manager.is_local_whisper_configured(),
            "active": self.active_service_type
        }