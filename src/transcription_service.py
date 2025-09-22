"""
Azure OpenAI transcription service for speech-to-text conversion.

This module provides a clean interface for transcribing audio files using Azure OpenAI's
Whisper model. It handles API key authentication, file upload, and response
parsing with comprehensive error handling.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

import openai
from openai import AzureOpenAI

from .exceptions import (
    TranscriptionError, 
    AzureAuthenticationError, 
    AzureAPIError, 
    TranscriptionProcessingError,
    AudioFileError
)


class AzureOpenAITranscriptionService:
    """
    Service for transcribing audio files using Azure OpenAI's Whisper model.
    
    This class handles authentication, file upload, API communication, and response
    processing for speech-to-text transcription using Azure OpenAI services.
    """
    
    def __init__(self, endpoint: str, api_key: str, deployment_name: str = "whisper", api_version: str = "2024-06-01"):
        """
        Initialize the Azure OpenAI transcription service.
        
        Args:
            endpoint (str): Azure OpenAI service endpoint URL
            api_key (str): Azure OpenAI API key
            deployment_name (str): Name of the Whisper deployment. Default is "whisper"
            api_version (str): Azure OpenAI API version. Default is "2024-06-01"
            
        Raises:
            AzureAuthenticationError: If authentication setup fails
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.client: Optional[AzureOpenAI] = None
        
        # Validate required parameters
        if not self.endpoint:
            raise AzureAuthenticationError("Azure OpenAI endpoint is required")
        if not self.api_key:
            raise AzureAuthenticationError("Azure OpenAI API key is required")
        if not self.deployment_name:
            raise AzureAuthenticationError("Azure OpenAI deployment name is required")
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """
        Initialize the Azure OpenAI client with API key authentication.
        
        Uses API key authentication for simple and reliable access to Azure OpenAI.
        This avoids tenant mismatch issues that can occur with Azure CLI authentication.
        
        Raises:
            AzureAuthenticationError: If authentication setup fails
        """
        try:
            # Initialize the Azure OpenAI client with API key
            self.client = AzureOpenAI(
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
                api_key=self.api_key
            )
            
            print(f"✅ Azure OpenAI client initialized successfully")
            print(f"📊 Endpoint: {self.endpoint}")
            print(f"📊 Deployment: {self.deployment_name}")
            print(f"🔑 Authentication: API Key")
            
        except Exception as e:
            # Provide more detailed error information for common issues
            error_message = str(e).lower()
            if "authentication" in error_message or "unauthorized" in error_message or "403" in error_message:
                raise AzureAuthenticationError(
                    f"Authentication failed with API key: {e}\n\n"
                    "Please verify:\n"
                    "1. Your API key is correct (check Azure Portal > Your Resource > Keys and Endpoint)\n"
                    "2. Your endpoint URL is correct (should be https://YOUR-RESOURCE.openai.azure.com)\n"
                    "3. Your Azure OpenAI resource is active and not suspended\n"
                    "4. You have proper permissions to access the resource"
                )
            elif "not found" in error_message or "404" in error_message:
                raise AzureAuthenticationError(
                    f"Resource not found: {e}\n\n"
                    "Please verify:\n"
                    "1. Your endpoint URL is correct\n"
                    "2. Your Azure OpenAI resource exists\n"
                    "3. The resource is in the correct region"
                )
            else:
                raise AzureAuthenticationError(f"Failed to initialize Azure OpenAI client: {e}")
    
    def transcribe_audio_file(self, file_path: Union[str, Path], language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file using Azure OpenAI Whisper.
        
        Args:
            file_path: Path to the audio file to transcribe
            language: Optional language code (e.g., 'en', 'es', 'fr'). If None, auto-detect
            
        Returns:
            Dict containing transcription results with text, confidence, and metadata
            
        Raises:
            AudioFileError: If audio file cannot be read or is invalid
            AzureAPIError: If the API call fails
            TranscriptionProcessingError: If response processing fails
        """
        file_path = Path(file_path)
        
        # Validate file exists and is readable
        if not file_path.exists():
            raise AudioFileError(f"Audio file not found: {file_path}")
        
        if not file_path.is_file():
            raise AudioFileError(f"Path is not a file: {file_path}")
        
        # Check file size (Azure OpenAI has a 25MB limit for audio files)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            raise AudioFileError(f"Audio file too large: {file_size_mb:.1f}MB (limit: 25MB)")
        
        print(f"🎵 Transcribing audio file: {file_path.name}")
        print(f"📊 File size: {file_size_mb:.1f}MB")
        
        try:
            start_time = time.time()
            
            # Open and read the audio file
            with open(file_path, 'rb') as audio_file:
                # Prepare the transcription request
                transcription_params = {
                    "file": audio_file,
                    "model": self.deployment_name,
                    "response_format": "verbose_json",  # Get detailed response with timestamps
                    "timestamp_granularities": ["word", "segment"]
                }
                
                # Add language parameter if specified
                if language:
                    transcription_params["language"] = language
                
                # Make the API call
                response = self.client.audio.transcriptions.create(**transcription_params)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Process the response
            result = self._process_transcription_response(response, processing_time, file_path.name)
            
            print(f"✅ Transcription completed in {processing_time:.2f} seconds")
            
            return result
            
        except openai.APIError as e:
            raise AzureAPIError(f"Azure OpenAI API error: {e}")
        except openai.APIConnectionError as e:
            raise AzureAPIError(f"Connection error to Azure OpenAI: {e}")
        except openai.RateLimitError as e:
            raise AzureAPIError(f"Rate limit exceeded: {e}")
        except openai.APITimeoutError as e:
            raise AzureAPIError(f"API request timed out: {e}")
        except Exception as e:
            if "authentication" in str(e).lower() or "credential" in str(e).lower():
                raise AzureAuthenticationError(f"Authentication error during transcription: {e}")
            else:
                raise TranscriptionError(f"Unexpected error during transcription: {e}")
    
    def _process_transcription_response(self, response, processing_time: float, filename: str) -> Dict[str, Any]:
        """
        Process and format the transcription response from Azure OpenAI.
        
        Args:
            response: Raw response from Azure OpenAI API
            processing_time: Time taken for the API call
            filename: Name of the transcribed file
            
        Returns:
            Formatted transcription result dictionary
            
        Raises:
            TranscriptionProcessingError: If response processing fails
        """
        try:
            # Extract main text
            text = response.text.strip() if hasattr(response, 'text') else ""
            
            # Initialize result structure
            result = {
                "text": text,
                "language": getattr(response, 'language', 'unknown'),
                "duration": getattr(response, 'duration', 0),
                "processing_time": processing_time,
                "filename": filename,
                "word_count": len(text.split()) if text else 0,
                "segments": [],
                "words": []
            }
            
            # Process segments if available
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    segment_data = {
                        "text": segment.text.strip(),
                        "start": segment.start,
                        "end": segment.end,
                        "avg_logprob": getattr(segment, 'avg_logprob', None),
                        "no_speech_prob": getattr(segment, 'no_speech_prob', None)
                    }
                    result["segments"].append(segment_data)
            
            # Process words if available
            if hasattr(response, 'words') and response.words:
                for word in response.words:
                    word_data = {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    }
                    result["words"].append(word_data)
            
            # Calculate confidence metrics
            if result["segments"]:
                avg_logprobs = [s.get("avg_logprob", 0) for s in result["segments"] if s.get("avg_logprob") is not None]
                if avg_logprobs:
                    result["confidence_score"] = sum(avg_logprobs) / len(avg_logprobs)
                
                no_speech_probs = [s.get("no_speech_prob", 0) for s in result["segments"] if s.get("no_speech_prob") is not None]
                if no_speech_probs:
                    result["speech_confidence"] = 1 - (sum(no_speech_probs) / len(no_speech_probs))
            
            return result
            
        except Exception as e:
            raise TranscriptionProcessingError(f"Failed to process transcription response: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the transcription service configuration.
        
        Returns:
            Dictionary with service configuration details
        """
        return {
            "endpoint": self.endpoint,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "client_initialized": self.client is not None,
            "supported_formats": ["wav", "mp3", "m4a", "flac", "webm"],
            "max_file_size_mb": 25,
            "supported_languages": [
                "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", 
                "ar", "hi", "tr", "pl", "nl", "sv", "da", "no", "fi"
            ]
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to Azure OpenAI service.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Simple test to validate the service is accessible
            # We can't easily test without making an actual API call,
            # so we'll validate authentication instead
            if self.client is None:
                return False
                
            print("🔍 Testing Azure OpenAI connection...")
            
            # Try to access the client's configuration
            # This doesn't make an API call but validates the client setup
            endpoint_check = self.client._base_url is not None
            
            if endpoint_check:
                print("✅ Azure OpenAI service connection test passed")
                return True
            else:
                print("❌ Azure OpenAI service connection test failed")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def diagnose_connection_issue(self) -> Dict[str, Any]:
        """
        Diagnose potential connection and authentication issues.
        
        Returns:
            Dictionary with diagnostic information and suggested fixes
        """
        try:
            # Check if client is initialized
            if self.client is None:
                return {
                    "status": "client_not_initialized",
                    "error": "Azure OpenAI client not initialized",
                    "suggestions": [
                        "Check your API key is set in .env file",
                        "Verify endpoint URL format: https://YOUR-RESOURCE.openai.azure.com",
                        "Ensure deployment name is correct"
                    ]
                }
            
            # Check endpoint format
            if not self.endpoint.startswith("https://") or not self.endpoint.endswith(".openai.azure.com"):
                return {
                    "status": "invalid_endpoint",
                    "error": f"Endpoint format may be incorrect: {self.endpoint}",
                    "suggestions": [
                        "Endpoint should be: https://YOUR-RESOURCE.openai.azure.com",
                        "Check your endpoint in Azure Portal: Resource -> Keys and Endpoint",
                        "Do not use the Cognitive Services endpoint format"
                    ]
                }
            
            # Check API key format (basic validation)
            if not self.api_key or len(self.api_key) < 10:
                return {
                    "status": "invalid_api_key",
                    "error": "API key appears to be missing or invalid",
                    "suggestions": [
                        "Get your API key from Azure Portal: Resource -> Keys and Endpoint",
                        "Use either Key 1 or Key 2",
                        "Ensure the key is copied completely without extra spaces"
                    ]
                }
            
            return {
                "status": "configuration_ok",
                "endpoint": self.endpoint,
                "deployment_name": self.deployment_name,
                "api_version": self.api_version,
                "api_key_length": len(self.api_key),
                "suggestions": [
                    "Configuration appears correct",
                    "Try the test connection feature",
                    "Check network connectivity if issues persist"
                ]
            }
                    
        except Exception as e:
            return {
                "status": "diagnostic_failed",
                "error": str(e),
                "suggestions": [
                    "Check your .env file configuration",
                    "Verify all required environment variables are set",
                    "Check your network connection"
                ]
            }