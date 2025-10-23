"""
Local Whisper transcription service for speech-to-text conversion using fine-tuned models.

This module provides a clean interface for transcribing audio files using locally stored
fine-tuned Whisper models. It handles model loading, audio preprocessing, and inference
with comprehensive error handling.
"""

import json
import time
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union

from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Handle both relative and absolute imports
try:
    from .exceptions import (
        TranscriptionError, 
        TranscriptionProcessingError,
        AudioFileError
    )
except ImportError:
    from exceptions import (
        TranscriptionError, 
        TranscriptionProcessingError,
        AudioFileError
    )


class LocalWhisperTranscriptionService:
    """
    Service for transcribing audio files using locally stored fine-tuned Whisper models.
    
    This class handles model loading, audio preprocessing, and inference for speech-to-text
    transcription using fine-tuned Whisper models.
    """
    
    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        """
        Initialize the local Whisper transcription service.
        
        Args:
            model_path: Path to the fine-tuned Whisper model directory
            device: Device to use for inference ("auto", "cuda", "cpu")
            
        Raises:
            TranscriptionError: If model loading fails
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        
        # Model components
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None
        
        # Validate model path
        if not self.model_path.exists():
            raise TranscriptionError(f"Model path not found: {model_path}")
        
        if not self.model_path.is_dir():
            raise TranscriptionError(f"Model path is not a directory: {model_path}")
        
        # Load model
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self) -> None:
        """
        Load the fine-tuned Whisper model and processor.
        
        Raises:
            TranscriptionError: If model loading fails
        """
        try:
            print(f"🤖 Loading fine-tuned Whisper model from: {self.model_path}")
            print(f"🔧 Using device: {self.device}")
            
            # Load model components
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(self.model_path)
            ).to(self.device)
            
            self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"✅ Model loaded successfully")
            print(f"📊 Model parameters: {self.model.num_parameters():,}")
            print(f"🎯 Model type: Fine-tuned Whisper")
            
        except Exception as e:
            raise TranscriptionError(f"Failed to load model: {e}")
    
    def transcribe_audio_file(self, file_path: Union[str, Path], language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file using the fine-tuned Whisper model.
        
        Args:
            file_path: Path to the audio file to transcribe
            language: Optional language code (e.g., 'en', 'es', 'fr'). If None, auto-detect
            
        Returns:
            Dict containing transcription results with text, confidence, and metadata
            
        Raises:
            AudioFileError: If audio file cannot be read or is invalid
            TranscriptionProcessingError: If inference fails
        """
        file_path = Path(file_path)
        
        # Validate file exists and is readable
        if not file_path.exists():
            raise AudioFileError(f"Audio file not found: {file_path}")
        
        if not file_path.is_file():
            raise AudioFileError(f"Path is not a file: {file_path}")
        
        # Check file size (reasonable limit for local processing)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # More generous limit for local processing
            raise AudioFileError(f"Audio file too large: {file_size_mb:.1f}MB (recommended limit: 100MB)")
        
        print(f"🎵 Transcribing audio file: {file_path.name}")
        print(f"📊 File size: {file_size_mb:.1f}MB")
        
        try:
            start_time = time.time()
            
            # Load and preprocess audio
            audio = self._load_audio(file_path)
            print("Language: ", language)
            # Prepare inputs (language is handled during generation, not preprocessing)
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                # Prepare generation parameters
                generation_kwargs = {
                    "max_length": 448,  # Standard Whisper max length
                    "num_beams": 5,     # Use beam search for better quality
                    "do_sample": False,
                    "early_stopping": True
                }
                
                # Add language token if specified
                if language and language != "auto":
                    # For English, we use decoder_input_ids to guide generation
                    if language.lower() in ["en", "english"]:
                        decoder_input_ids = torch.tensor(
                            [[50258, 50259]]  # [start_token, english_token]
                        ).to(self.device)
                        generation_kwargs["decoder_input_ids"] = decoder_input_ids
                
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    **generation_kwargs
                )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"⏱️  Transcription completed in {processing_time:.2f} seconds")
            print(f"📝 Transcription: {transcription}")
            
            # Build response in similar format to Azure OpenAI service
            result = {
                "text": transcription.strip(),
                "language": language if language else "auto-detected",
                "confidence": None,  # Fine-tuned models don't provide confidence scores directly
                "confidence_score": None,
                "word_count": len(transcription.strip().split()),
                "duration": len(audio) / 16000,  # Approximate duration in seconds
                "processing_time": processing_time,
                "model_info": {
                    "model_path": str(self.model_path),
                    "model_type": "fine-tuned-whisper",
                    "device": str(self.device),
                    "parameters": self.model.num_parameters()
                },
                "audio_info": {
                    "filename": file_path.name,
                    "size_mb": file_size_mb,
                    "duration": len(audio) / 16000  # Approximate duration in seconds
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Transcription failed: {e}"
            print(f"❌ {error_msg}")
            raise TranscriptionProcessingError(error_msg)
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess audio file for Whisper.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Preprocessed audio array
            
        Raises:
            AudioFileError: If audio loading fails
        """
        try:
            # Load audio with librosa (Whisper expects 16kHz)
            audio, sample_rate = librosa.load(str(file_path), sr=16000, mono=True)
            
            # Normalize audio to [-1, 1] range
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            
            # Pad or trim to 30 seconds (Whisper's expected input length)
            target_length = 16000 * 30  # 30 seconds at 16kHz
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            return audio
            
        except Exception as e:
            raise AudioFileError(f"Failed to load audio file: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the model by performing a simple inference.
        
        Returns:
            Dict containing test results
        """
        try:
            # Create a simple test audio (1 second of silence)
            test_audio = np.zeros(16000)  # 1 second of silence at 16kHz
            
            inputs = self.processor(
                test_audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_length=10,  # Very short for test
                    do_sample=False
                )
            
            test_transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            return {
                "status": "success",
                "model_path": str(self.model_path),
                "device": str(self.device),
                "test_transcription": test_transcription,
                "model_parameters": self.model.num_parameters()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model_path": str(self.model_path),
                "device": str(self.device)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        if not self.model or not self.processor:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "parameters": self.model.num_parameters(),
            "model_type": "fine-tuned-whisper",
            "vocab_size": self.processor.tokenizer.vocab_size if self.processor.tokenizer else None
        }