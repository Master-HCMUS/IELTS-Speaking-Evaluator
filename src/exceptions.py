"""
Custom exceptions for the audio recording system.

This module defines specific exceptions that can be raised during audio recording
operations, providing clear error messages and categorization for better error handling.
"""


class AudioRecordingError(Exception):
    """
    Base exception for audio recording related errors.
    
    This exception is raised when general audio recording operations fail,
    such as starting/stopping recordings or processing audio data.
    """
    pass


class AudioDeviceError(AudioRecordingError):
    """
    Exception raised when audio device operations fail.
    
    This includes errors related to device initialization, device availability,
    or device configuration issues.
    """
    pass


class AudioFileError(AudioRecordingError):
    """
    Exception raised when audio file operations fail.
    
    This includes errors related to reading, writing, or processing audio files.
    """
    pass


class ConfigurationError(AudioRecordingError):
    """
    Exception raised when configuration-related operations fail.
    
    This includes invalid audio parameters, configuration file issues,
    or incompatible settings.
    """
    pass


class TranscriptionError(Exception):
    """
    Base exception for transcription-related errors.
    
    This exception is raised when transcription operations fail,
    such as API calls, authentication, or response processing.
    """
    pass


class AzureAuthenticationError(TranscriptionError):
    """
    Exception raised when Azure authentication fails.
    
    This includes errors related to managed identity authentication,
    token acquisition, or credential validation.
    """
    pass


class AzureAPIError(TranscriptionError):
    """
    Exception raised when Azure OpenAI API calls fail.
    
    This includes HTTP errors, API rate limiting, service unavailability,
    or invalid responses from the Azure OpenAI service.
    """
    pass


class TranscriptionProcessingError(TranscriptionError):
    """
    Exception raised when transcription response processing fails.
    
    This includes errors related to parsing API responses, extracting text,
    or handling unexpected response formats.
    """
    pass