"""
Local pronunciation assessment service using fine-tuned Whisper model.

This module provides pronunciation scoring using a locally trained model with:
- WhisperForConditionalGeneration for both transcription and assessment
- Frame-level assessment heads (word accuracy, stress, phone accuracy)
- Utterance-level assessment heads (accuracy, fluency, prosodic, completeness, total)

Output format compatible with SpeechOcean762 dataset standard.
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
        
        Returns scores in SpeechOcean762 format:
        - word_level: accuracy, stress, total (per frame)
        - phone_level: accuracy (per frame)
        - utterance_level: accuracy, fluency, prosodic, completeness, total
        
        Args:
            file_path: Path to audio file to assess
            
        Returns:
            Dict with pronunciation scores in SpeechOcean762 format
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            logger.info(f"Assessing pronunciation: {file_path.name}")
            
            # Load and preprocess audio
            mel_spec = self._load_and_preprocess_audio(file_path)
            mel_spec = torch.from_numpy(mel_spec).float().to(self.device)
            
            # Get predictions using the new method
            with torch.no_grad():
                scores_dict = self.model.predict_assessment_scores(mel_spec)
            
            # Structure output in SpeechOcean762 format
            result = {
                "status": "success",
                "file": file_path.name,
                "scores": {
                    "word_level": {},
                    "phone_level": {},
                    "utterance_level": {}
                },
                "model_path": str(self.model_path),
                "device": str(self.device)
            }
            
            # Extract and normalize word-level scores
            word_level = scores_dict.get('word_level', {})
            if 'accuracy' in word_level:
                # Frame-level scores: [batch, seq_len]
                accuracy_scores = word_level['accuracy'].cpu().numpy()
                result['scores']['word_level']['accuracy'] = self._normalize_frame_scores(
                    accuracy_scores[0] if accuracy_scores.ndim > 1 else accuracy_scores
                )
            
            if 'stress' in word_level:
                stress_scores = word_level['stress'].cpu().numpy()
                result['scores']['word_level']['stress'] = self._normalize_frame_scores(
                    stress_scores[0] if stress_scores.ndim > 1 else stress_scores
                )
            
            if 'total' in word_level:
                total_scores = word_level['total'].cpu().numpy()
                result['scores']['word_level']['total'] = self._normalize_frame_scores(
                    total_scores[0] if total_scores.ndim > 1 else total_scores
                )
            
            # Extract and normalize phone-level scores
            phone_level = scores_dict.get('phone_level', {})
            if 'accuracy' in phone_level:
                phone_scores = phone_level['accuracy'].cpu().numpy()
                result['scores']['phone_level']['accuracy'] = self._normalize_frame_scores(
                    phone_scores[0] if phone_scores.ndim > 1 else phone_scores
                )
            
            # Extract and normalize utterance-level scores (single values)
            utterance_level = scores_dict.get('utterance_level', {})
            
            if 'accuracy' in utterance_level:
                acc = utterance_level['accuracy'].item()
                result['scores']['utterance_level']['accuracy'] = self._normalize_utterance_score(acc)
            
            if 'fluency' in utterance_level:
                flu = utterance_level['fluency'].item()
                result['scores']['utterance_level']['fluency'] = self._normalize_utterance_score(flu)
            
            if 'prosodic' in utterance_level:
                pro = utterance_level['prosodic'].item()
                result['scores']['utterance_level']['prosodic'] = self._normalize_utterance_score(pro)
            
            if 'completeness' in utterance_level:
                com = utterance_level['completeness'].item()
                result['scores']['utterance_level']['completeness'] = self._normalize_utterance_score(com)
            
            if 'total' in utterance_level:
                tot = utterance_level['total'].item()
                result['scores']['utterance_level']['total'] = self._normalize_utterance_score(tot)
            
            # Add averages for convenience
            result['scores']['word_level']['average'] = self._compute_average(
                result['scores']['word_level'].get('accuracy', [])
            )
            result['scores']['phone_level']['average'] = self._compute_average(
                result['scores']['phone_level'].get('accuracy', [])
            )
            
            logger.info(f"Assessment complete. Utterance scores: {result['scores']['utterance_level']}")
            return result
        
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file": file_path.name
            }
    
    def _normalize_frame_scores(self, scores: np.ndarray) -> list:
        """
        Normalize frame-level scores to 0-10 range.
        
        Args:
            scores: Array of frame scores
            
        Returns:
            List of normalized scores
        """
        if isinstance(scores, (int, float)):
            scores = np.array([scores])
        
        # Apply sigmoid to bound between 0-1, then scale to 0-10
        normalized = torch.sigmoid(torch.tensor(scores)).numpy() * 10
        return normalized.tolist()
    
    def _normalize_utterance_score(self, score: float) -> float:
        """
        Normalize utterance-level score to 0-10 range.
        
        Args:
            score: Single utterance score
            
        Returns:
            Normalized score (0-10)
        """
        # Apply sigmoid to bound between 0-1, then scale to 0-10
        normalized = float(torch.sigmoid(torch.tensor(score)).item() * 10)
        return normalized
    
    def _compute_average(self, scores: Union[list, np.ndarray]) -> float:
        """
        Compute average of frame-level scores.
        
        Args:
            scores: List or array of scores
            
        Returns:
            Average score
        """
        if not scores or len(scores) == 0:
            return 0.0
        return float(np.mean(scores))
    
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
