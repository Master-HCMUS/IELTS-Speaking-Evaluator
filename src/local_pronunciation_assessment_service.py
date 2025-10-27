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
from transformers import WhisperProcessor

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
    
    def _generate_transcription(self, mel_spec: torch.Tensor) -> Optional[str]:
        """
        Generate transcription from mel-spectrogram using the model's decoder.
        
        Args:
            mel_spec: Mel-spectrogram tensor [batch, 80, 3000]
            
        Returns:
            Transcription text or None if generation fails
        """
        try:
            if self.model is None:
                logger.warning("Model not loaded, cannot generate transcription")
                return None
            
            # Generate token IDs using the model
            with torch.no_grad():
                generated_ids = self.model.generate_transcription(
                    mel_spec,
                    max_length=128,
                    num_beams=1
                )
            
            # Decode tokens to text using WhisperProcessor
            processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Return first transcription from batch
            if transcription and len(transcription) > 0:
                return transcription[0].strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to generate transcription: {e}")
            return None

    
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
    
    def assess_pronunciation(self, file_path: Union[str, Path], target_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess pronunciation of an audio file.
        
        Returns scores in SpeechOcean762 format:
        - word_level: accuracy, stress, total (per frame)
        - phone_level: accuracy (per frame)
        - utterance_level: accuracy, fluency, prosodic, completeness, total
        - transcript: Transcribed text from audio
        
        Args:
            file_path: Path to audio file to assess
            target_text: Optional target text the user should have said
                        If provided, scores are penalized if transcript doesn't match
            
        Returns:
            Dict with pronunciation scores and transcript in SpeechOcean762 format
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
            
            # Generate transcription
            transcript = self._generate_transcription(mel_spec)
            
            # Structure output in SpeechOcean762 format
            result = {
                "status": "success",
                "file": file_path.name,
                "transcript": transcript if transcript else "Unable to generate transcript",
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
            
            # Apply content matching penalty if target text provided
            if target_text:
                similarity = self._calculate_text_similarity(target_text, transcript)
                result['content_match'] = {
                    'target': target_text,
                    'similarity': similarity,
                    'match_percentage': similarity * 100
                }
                
                # Apply penalty to scores if content doesn't match
                if similarity < 0.75:
                    result = self._apply_content_penalty(result, similarity)
            
            logger.info(f"Assessment complete. Utterance scores: {result['scores']['utterance_level']}")
            logger.info(f"Transcript: {transcript}")
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using sequence matching.
        
        Args:
            text1: Target text
            text2: Transcribed text
            
        Returns:
            Similarity ratio (0-1)
        """
        from difflib import SequenceMatcher
        
        # Normalize texts: lowercase and strip whitespace
        text1_norm = text1.lower().strip()
        text2_norm = text2.lower().strip()
        
        # Use SequenceMatcher to calculate similarity
        matcher = SequenceMatcher(None, text1_norm, text2_norm)
        return matcher.ratio()
    
    def _apply_content_penalty(self, result: Dict[str, Any], similarity: float) -> Dict[str, Any]:
        """
        Apply penalty to scores if transcribed content doesn't match target.
        
        Args:
            result: Assessment result dictionary
            similarity: Text similarity ratio (0-1)
            
        Returns:
            Result with penalized scores
        """
        # Calculate penalty factor based on similarity
        if similarity < 0.5:
            penalty_factor = 0.3 + (similarity * 0.4)  # Range: 0.3-0.7
        elif similarity < 0.75:
            penalty_factor = 0.85
        else:
            penalty_factor = 1.0  # No penalty
        
        # Apply penalty to utterance-level scores
        utterance_scores = result['scores']['utterance_level']
        for key in utterance_scores:
            if key != 'total':
                utterance_scores[key] = utterance_scores[key] * penalty_factor
            else:
                # Total score gets heavier penalty
                utterance_scores[key] = utterance_scores[key] * max(penalty_factor - 0.1, 0.2)
        
        # Apply penalty to word-level scores (frame-level accuracy)
        word_level = result['scores']['word_level']
        if 'accuracy' in word_level and isinstance(word_level['accuracy'], list):
            word_level['accuracy'] = [score * penalty_factor for score in word_level['accuracy']]
        if 'average' in word_level:
            word_level['average'] = word_level['average'] * penalty_factor
        
        # Apply penalty to phone-level scores
        phone_level = result['scores']['phone_level']
        if 'accuracy' in phone_level and isinstance(phone_level['accuracy'], list):
            phone_level['accuracy'] = [score * penalty_factor for score in phone_level['accuracy']]
        if 'average' in phone_level:
            phone_level['average'] = phone_level['average'] * penalty_factor
        
        result['penalty_applied'] = {
            'similarity': similarity,
            'penalty_factor': penalty_factor,
            'reason': 'Content mismatch detected'
        }
        
        return result
    

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
