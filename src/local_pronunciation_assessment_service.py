"""
Local pronunciation assessment service using fine-tuned Whisper encoder with assessment heads.

This module provides pronunciation scoring using a locally trained model with:
- Whisper encoder for audio encoding
- Custom assessment heads for scoring

Note: This service is for ASSESSMENT ONLY, not transcription.
For transcription, use Azure OpenAI or standard Whisper model.
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LocalPronunciationAssessmentService:
    """
    Service for assessing pronunciation using locally trained model.
    
    This class handles pronunciation assessment using the fine-tuned
    Whisper encoder with custom assessment heads.
    """
    
    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        """
        Initialize the local pronunciation assessment service.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to use for inference ("auto", "cuda", "cpu")
            
        Raises:
            Exception: If model loading fails
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = None
        
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        if not self.model_path.is_dir():
            raise NotADirectoryError(f"Model path is not a directory: {model_path}")
        print("Initializing LocalPronunciationAssessmentService...")
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
        Load the fine-tuned pronunciation assessment model.
        
        Raises:
            Exception: If model loading fails
        """
        try:
            print(f"Loading pronunciation assessment model from: {self.model_path}")
            print(f"Using device: {self.device}")
            
            # Import here to avoid JAX issues
            import sys
            import os
            os.environ['JAX_PLATFORMS'] = 'cpu'
            
            # Monkey-patch numpy.dtypes if needed
            import numpy as np
            if not hasattr(np, 'dtypes'):
                class DummyDtypes:
                    pass
                np.dtypes = DummyDtypes()
            
            from transformers import WhisperModel
            # Import from relative path
            import sys
            from pathlib import Path
            finetuning_path = Path(__file__).parent / "finetuning" / "finetuning_pronunciation_assessment"
            if str(finetuning_path) not in sys.path:
                sys.path.insert(0, str(finetuning_path))
            from whisper_pronunciation_model import WhisperPronunciationAssessmentModel
            
            # Load model
            self.model = WhisperPronunciationAssessmentModel(
                model_name="openai/whisper-base"  # Can be overridden, but we'll load weights
            )
            
            # Load saved weights - try different checkpoint formats
            checkpoint_found = False
            
            # Try .pt files first (look for any .pt file in directory)
            pt_files = list(self.model_path.glob("*.pt"))
            if pt_files:
                pt_path = pt_files[0]  # Use first .pt file found
                print(f"Loading checkpoint from {pt_path}")
                state_dict = torch.load(pt_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                checkpoint_found = True
            
            # Try pytorch_model.bin
            elif (self.model_path / "pytorch_model.bin").exists():
                checkpoint_path = self.model_path / "pytorch_model.bin"
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                checkpoint_found = True
            
            # Try model.safetensors
            elif (self.model_path / "model.safetensors").exists():
                logger.info(f"Found model.safetensors, loading with torch...")
                # For now, skip safetensors as it requires additional handling
                logger.warning("safetensors format found but not fully supported yet")
            
            if not checkpoint_found:
                logger.warning(f"No checkpoint found in {self.model_path}. Using random weights.")
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Pronunciation assessment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_and_preprocess_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess audio file to mel-spectrogram.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Mel-spectrogram tensor [1, 80, 3000]
        """
        file_path = Path(file_path)
        
        try:
            # Load audio with librosa (16kHz)
            audio, sr = librosa.load(str(file_path), sr=16000, mono=True)
            
            # Compute mel-spectrogram (Whisper format: 80 bins, 3000 timesteps)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                fmin=0,
                fmax=8000
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec = (mel_spec + 40) / 40
            
            # Pad or truncate to 3000 timesteps
            target_length = 3000
            if mel_spec.shape[1] < target_length:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, target_length - mel_spec.shape[1])))
            else:
                mel_spec = mel_spec[:, :target_length]
            
            # Add batch dimension: [1, 80, 3000]
            mel_spec = np.expand_dims(mel_spec, axis=0)
            
            return mel_spec
        
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def assess_pronunciation(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Assess pronunciation of an audio file.
        
        Args:
            file_path: Path to audio file to assess
            
        Returns:
            Dict with pronunciation scores
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            logger.info(f"Assessing pronunciation: {file_path.name}")
            
            # Load and preprocess audio
            mel_spec = self._load_and_preprocess_audio(file_path)
            mel_spec = torch.from_numpy(mel_spec).float().to(self.device)
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(mel_spec)
            
            # Extract scores
            scores = {}
            
            # Utterance-level scores (single value per utterance)
            if 'utterance_accuracy_logits' in predictions:
                scores['utterance_accuracy'] = float(predictions['utterance_accuracy_logits'].item())
            if 'utterance_fluency_logits' in predictions:
                scores['utterance_fluency'] = float(predictions['utterance_fluency_logits'].item())
            if 'utterance_prosodic_logits' in predictions:
                scores['utterance_prosodic'] = float(predictions['utterance_prosodic_logits'].item())
            if 'utterance_completeness_logits' in predictions:
                scores['utterance_completeness'] = float(predictions['utterance_completeness_logits'].item())
            if 'utterance_total_logits' in predictions:
                scores['utterance_total'] = float(predictions['utterance_total_logits'].item())
            
            # Word-level scores (averaged across sequence)
            if 'word_accuracy_logits' in predictions:
                word_acc = predictions['word_accuracy_logits'].mean(dim=1)
                scores['word_accuracy'] = float(word_acc.item())
            if 'word_stress_logits' in predictions:
                word_stress = predictions['word_stress_logits'].mean(dim=1)
                scores['word_stress'] = float(word_stress.item())
            if 'word_total_logits' in predictions:
                word_total = predictions['word_total_logits'].mean(dim=1)
                scores['word_total'] = float(word_total.item())
            
            # Phone-level score (averaged across sequence)
            if 'phone_accuracy_logits' in predictions:
                phone_acc = predictions['phone_accuracy_logits'].mean(dim=1)
                scores['phone_accuracy'] = float(phone_acc.item())
            
            # Normalize scores to 0-100 range if needed
            scores = self._normalize_scores(scores)
            
            result = {
                "status": "success",
                "file": file_path.name,
                "scores": scores,
                "model_path": str(self.model_path),
                "device": str(self.device)
            }
            
            logger.info(f"Assessment complete: {scores}")
            return result
        
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file": file_path.name
            }
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to 0-100 range if they're outside this range.
        
        Args:
            scores: Dictionary of scores
            
        Returns:
            Normalized scores
        """
        normalized = {}
        
        for key, value in scores.items():
            # If value is outside 0-100, apply sigmoid and scale
            if value < 0 or value > 100:
                # Apply sigmoid to bound between 0-1, then scale to 0-100
                normalized[key] = float(torch.sigmoid(torch.tensor(value)).item() * 100)
            else:
                normalized[key] = value
        
        return normalized
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "device": str(self.device),
            "model_type": "pronunciation-assessment",
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
