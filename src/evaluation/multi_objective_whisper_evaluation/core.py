"""
Multi-Objective Whisper Model Evaluation Module

This module provides functionality to evaluate fine-tuned Whisper models with
pronunciation assessment heads against the SpeechOcean762 dataset with expert
human annotations.

The multi-objective model trains on:
- Transcription (STT): Cross-Entropy loss on decoder outputs
- Pronunciation Assessment: MSE loss on 9 assessment objectives (word, phone, utterance levels)

Evaluation computes:
- Transcription metrics: WER, CER, BLEU
- Assessment correlations: Pearson r with expert scores
- MAE and RMSE for assessment accuracy
"""

import os
import json
import time
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Audio processing
import librosa
import soundfile as sf

# Transformers and models
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    AutoProcessor
)

# Evaluation metrics
from scipy import stats
import evaluate

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Local imports - handle both relative and absolute imports
try:
    from ...finetuning.finetuning_pronunciation_assessment.whisper_pronunciation_model import (
        WhisperPronunciationAssessmentModel
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    try:
        from finetuning.finetuning_pronunciation_assessment.whisper_pronunciation_model import (
            WhisperPronunciationAssessmentModel
        )
    except ImportError:
        pass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@dataclass
class MultiObjectiveEvaluationMetrics:
    """Container for multi-objective model evaluation metrics."""
    
    # Transcription metrics
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    bleu: float  # BLEU score
    
    # Correlation metrics (with expert annotations)
    accuracy_correlation: float
    fluency_correlation: float
    completeness_correlation: float
    prosodic_correlation: float
    
    # Mean Absolute Error (MAE)
    accuracy_mae: float
    fluency_mae: float
    completeness_mae: float
    prosodic_mae: float
    
    # Root Mean Square Error (RMSE)
    accuracy_rmse: float
    fluency_rmse: float
    completeness_rmse: float
    prosodic_rmse: float
    
    # Sample statistics
    total_samples: int
    successful_assessments: int
    failed_assessments: int
    
    # Score distributions
    expert_score_stats: Dict[str, Dict[str, float]]
    model_score_stats: Dict[str, Dict[str, float]]


@dataclass
class MultiObjectiveEvaluationResult:
    """Container for individual evaluation results."""
    
    # Basic info
    model_path: str
    sample_idx: int
    text: str
    speaker: str
    success: bool
    
    # Transcription results
    predicted_text: str = ""
    reference_text: str = ""
    
    # Transcription metrics
    word_error_rate: float = 0.0
    character_error_rate: float = 0.0
    bleu_score: float = 0.0
    
    # Pronunciation assessment scores
    model_scores: Dict[str, float] = None
    expert_scores: Dict[str, float] = None
    
    # Word-level scores (similar to Azure Speech API)
    word_level_scores: List[Dict[str, Any]] = None
    
    # Error information
    error: str = ""
    
    def __post_init__(self):
        if self.model_scores is None:
            self.model_scores = {
                'accuracy': 0.0, 'fluency': 0.0,
                'completeness': 0.0, 'prosodic': 0.0
            }
        if self.expert_scores is None:
            self.expert_scores = {
                'accuracy': 0.0, 'fluency': 0.0,
                'completeness': 0.0, 'prosodic': 0.0
            }
        if self.word_level_scores is None:
            self.word_level_scores = []


class MultiObjectiveWhisperAssessor:
    """Multi-objective Whisper model assessor for pronunciation evaluation."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the multi-objective Whisper assessor."""
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        
        # Model components
        self.model = None
        self.processor = None
        self.feature_extractor = None
        
        # Evaluation metrics
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        
        # Load model
        self._load_model()
        
        logger.info(f"Initialized MultiObjectiveWhisperAssessor with model: {model_path}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the fine-tuned multi-objective Whisper model."""
        try:
            logger.info(f"Loading multi-objective Whisper model from: {self.model_path}")
            
            # Check if this is a .pt file or a model directory
            model_path_str = str(self.model_path)
            is_pt_file = model_path_str.endswith('.pt')
            
            if is_pt_file:
                # Load from .pt checkpoint file
                logger.info("Loading from .pt checkpoint file...")
                
                # Initialize model with base Whisper first
                self.model = WhisperPronunciationAssessmentModel(
                    model_name="openai/whisper-base",
                    train_transcription=True,
                    train_word_level=True,
                    train_phone_level=True,
                    train_utterance_level=True,
                    freeze_encoder=False,
                    freeze_decoder=False
                )
                
                # Load checkpoint weights
                checkpoint = torch.load(model_path_str, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Handle checkpoint format with state_dict key
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    # Handle direct state dict
                    self.model.load_state_dict(checkpoint, strict=False)
                
                logger.info("Checkpoint loaded successfully")
                
                # For .pt files, use the processor from the base model that was fine-tuned
                # Since we're loading a fine-tuned whisper-base model, use whisper-base processor
                try:
                    self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
                    logger.info("Loaded WhisperProcessor for fine-tuned model (whisper-base)")
                except:
                    logger.info("Using AutoProcessor as fallback")
                    self.processor = AutoProcessor.from_pretrained("openai/whisper-base")
            else:
                # Load from model directory (standard HuggingFace format)
                logger.info("Loading from model directory...")
                
                self.model = WhisperPronunciationAssessmentModel(
                    model_name=model_path_str,
                    train_transcription=True,
                    train_word_level=True,
                    train_phone_level=True,
                    train_utterance_level=True,
                    freeze_encoder=False,
                    freeze_decoder=False
                )
                
                self.processor = AutoProcessor.from_pretrained(model_path_str)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.feature_extractor = self.processor.feature_extractor
            
            logger.info("Model loaded successfully")
            logger.info(f"Model device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_and_assess(self, audio_path: str, reference_text: str = "") -> Dict[str, Any]:
        """Transcribe audio and compute pronunciation assessment."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare input features
            input_features = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                # Transcription
                predicted_ids = self.model.generate_transcription(input_features)
                predicted_text = self.processor.tokenizer.decode(
                    predicted_ids[0], skip_special_tokens=True
                ).strip()
                
                # Assessment scores
                assessment_scores = self.model.predict_assessment_scores(input_features)
                
                # Generate phonemes using the new method
                phoneme_result = self.model.generate_phonemes(
                    input_features,
                    return_confidence=True,
                    return_frame_level=True,
                    collapse_repeated=True
                )
            
            # Clean reference text
            reference_text = reference_text.strip()
            
            # Calculate transcription quality metrics
            quality_metrics = self._calculate_transcription_metrics(
                predicted_text, reference_text
            )
            
            # Extract assessment scores
            model_scores = self._extract_assessment_scores(assessment_scores)
            
            # Generate word-level scores from phonemes and predicted text
            word_level_scores = self._generate_word_level_scores(
                predicted_text, phoneme_result, model_scores
            )
            
            return {
                'predicted_text': predicted_text,
                'reference_text': reference_text,
                'quality_metrics': quality_metrics,
                'model_scores': model_scores,
                'word_level_scores': word_level_scores,
                'phoneme_result': phoneme_result,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Transcription and assessment failed: {e}")
            return {
                'predicted_text': "",
                'reference_text': reference_text,
                'quality_metrics': {},
                'model_scores': {
                    'accuracy': 0.0, 'fluency': 0.0,
                    'completeness': 0.0, 'prosodic': 0.0
                },
                'word_level_scores': [],
                'phoneme_result': None,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_transcription_metrics(self, predicted: str, reference: str) -> Dict[str, float]:
        """Calculate transcription quality metrics."""
        metrics = {}
        
        try:
            # Word Error Rate (WER)
            if reference and predicted:
                wer = self.wer_metric.compute(
                    predictions=[predicted],
                    references=[reference]
                )
                metrics['wer'] = wer
                metrics['word_accuracy'] = max(0, 1 - wer)
            else:
                metrics['wer'] = 1.0 if not predicted else 0.0
                metrics['word_accuracy'] = 0.0 if not predicted else 1.0
            
            # Character Error Rate (CER)
            cer = self._calculate_cer(predicted, reference)
            metrics['cer'] = cer
            metrics['char_accuracy'] = max(0, 1 - cer)
            
            # BLEU Score
            if reference and predicted:
                bleu = self.bleu_metric.compute(
                    predictions=[predicted],
                    references=[[reference]],
                    max_order=4
                )
                metrics['bleu'] = bleu['bleu']
            else:
                metrics['bleu'] = 0.0
            
            # Length ratio (completeness indicator)
            if reference:
                metrics['length_ratio'] = len(predicted.split()) / len(reference.split())
                metrics['completeness_ratio'] = min(1.0, metrics['length_ratio'])
            else:
                metrics['length_ratio'] = 0.0
                metrics['completeness_ratio'] = 0.0
            
            # Overall transcription quality
            word_acc = metrics['word_accuracy']
            char_acc = metrics['char_accuracy']
            bleu = metrics['bleu']
            completeness = metrics['completeness_ratio']
            
            quality_score = (
                0.4 * word_acc +
                0.3 * char_acc +
                0.2 * bleu +
                0.1 * completeness
            )
            metrics['overall_quality'] = quality_score
            
        except Exception as e:
            logger.error(f"Error calculating transcription metrics: {e}")
            metrics = {
                'wer': 1.0, 'word_accuracy': 0.0,
                'cer': 1.0, 'char_accuracy': 0.0,
                'bleu': 0.0, 'length_ratio': 0.0,
                'completeness_ratio': 0.0, 'overall_quality': 0.0
            }
        
        return metrics
    
    def _calculate_cer(self, predicted: str, reference: str) -> float:
        """Calculate Character Error Rate."""
        if not reference:
            return 1.0 if predicted else 0.0
        
        ref_chars = list(reference.replace(" ", ""))
        pred_chars = list(predicted.replace(" ", ""))
        
        m, n = len(ref_chars), len(pred_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == pred_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_distance = dp[m][n]
        cer = edit_distance / len(ref_chars) if ref_chars else 0.0
        return cer
    
    def _extract_assessment_scores(self, assessment_scores: Dict[str, Any]) -> Dict[str, float]:
        """Extract assessment scores from model predictions and keep in [0-1] range."""
        scores = {
            'accuracy': 0.0,
            'fluency': 0.0,
            'completeness': 0.0,
            'prosodic': 0.0
        }
        
        try:
            # Use utterance-level scores as primary
            utterance = assessment_scores.get('utterance_level', {})
            
            if 'accuracy' in utterance:
                raw_value = utterance['accuracy'].mean().item() if hasattr(utterance['accuracy'], 'mean') else float(utterance['accuracy'])
                scores['accuracy'] = self._clamp_to_01_range(raw_value)
            
            if 'fluency' in utterance:
                raw_value = utterance['fluency'].mean().item() if hasattr(utterance['fluency'], 'mean') else float(utterance['fluency'])
                scores['fluency'] = self._clamp_to_01_range(raw_value)
            
            if 'completeness' in utterance:
                raw_value = utterance['completeness'].mean().item() if hasattr(utterance['completeness'], 'mean') else float(utterance['completeness'])
                scores['completeness'] = self._clamp_to_01_range(raw_value)
            
            if 'prosodic' in utterance:
                raw_value = utterance['prosodic'].mean().item() if hasattr(utterance['prosodic'], 'mean') else float(utterance['prosodic'])
                scores['prosodic'] = self._clamp_to_01_range(raw_value)
            
        except Exception as e:
            logger.warning(f"Failed to extract assessment scores: {e}")
        
        return scores
    
    def _clamp_to_01_range(self, value: float) -> float:
        """
        Clamp model output to [0-1] range.
        
        The model should output values in [0-1] range, but we clamp to ensure
        consistency with normalized expert scores.
        """
        return max(0.0, min(1.0, float(value)))
    
    def _normalize_score(self, value: float, from_range: Tuple[float, float], 
                         to_range: Tuple[float, float]) -> float:
        """Normalize score from one range to another."""
        from_min, from_max = from_range
        to_min, to_max = to_range
        
        # Normalize to 0-1
        if from_max - from_min == 0:
            normalized = 0.5
        else:
            normalized = (value - from_min) / (from_max - from_min)
        
        # Scale to target range
        scaled = normalized * (to_max - to_min) + to_min
        
        # Clamp to target range
        return max(to_min, min(to_max, scaled))
    
    def _generate_word_level_scores(
        self,
        predicted_text: str,
        phoneme_result: Optional[Dict[str, Any]],
        model_scores: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate word-level scores similar to Azure Speech Service API format.
        
        Args:
            predicted_text: Transcribed text
            phoneme_result: Result from model.generate_phonemes()
            model_scores: Overall model assessment scores
            
        Returns:
            List of word-level score dictionaries
        """
        word_level_scores = []
        
        if not predicted_text or not phoneme_result:
            return word_level_scores
        
        try:
            # Get phoneme symbols and confidence scores
            phoneme_symbols = phoneme_result.get('phoneme_symbols', [])
            phoneme_confidence = phoneme_result.get('phoneme_confidence', [])
            
            # Split text into words
            words = predicted_text.split()
            
            if not words or not phoneme_symbols:
                return word_level_scores
            
            # Estimate phonemes per word (simple approximation)
            # This is a rough mapping since we don't have perfect word-phoneme alignment
            phonemes_per_word = len(phoneme_symbols) // len(words) if words else 0
            phoneme_remainder = len(phoneme_symbols) % len(words) if words else 0
            
            phoneme_idx = 0
            
            for word_idx, word in enumerate(words):
                # Calculate how many phonemes for this word
                num_phonemes_for_word = phonemes_per_word
                if word_idx < phoneme_remainder:
                    num_phonemes_for_word += 1
                
                # Extract phonemes for this word
                word_phonemes = []
                word_phoneme_confidences = []
                
                for _ in range(num_phonemes_for_word):
                    if phoneme_idx < len(phoneme_symbols):
                        phoneme_symbol = phoneme_symbols[phoneme_idx]
                        phoneme_conf = phoneme_confidence[phoneme_idx] if phoneme_idx < len(phoneme_confidence) else 5.0
                        
                        word_phonemes.append(phoneme_symbol)
                        word_phoneme_confidences.append(phoneme_conf)
                        phoneme_idx += 1
                
                # Calculate word accuracy score based on phoneme confidence and model scores
                if word_phoneme_confidences:
                    avg_phoneme_confidence = sum(word_phoneme_confidences) / len(word_phoneme_confidences)
                    # Scale from [0-10] to [0-100] and blend with model accuracy
                    word_accuracy = (avg_phoneme_confidence * 10) * 0.7 + (model_scores.get('accuracy', 0) * 10) * 0.3
                    word_accuracy = min(100.0, max(0.0, word_accuracy))
                else:
                    word_accuracy = model_scores.get('accuracy', 0) * 10
                
                # Create phoneme details in Azure format
                phoneme_details = []
                for ph_symbol, ph_conf in zip(word_phonemes, word_phoneme_confidences):
                    phoneme_details.append({
                        "phoneme": ph_symbol.lower(),  # Azure uses lowercase
                        "accuracy_score": ph_conf * 10  # Scale to [0-100]
                    })
                
                # Determine error type based on accuracy
                if word_accuracy >= 80:
                    error_type = "None"
                elif word_accuracy >= 60:
                    error_type = "Mispronunciation"
                else:
                    error_type = "Omission"
                
                word_score = {
                    "word": word.lower(),
                    "accuracy_score": word_accuracy,
                    "error_type": error_type,
                    "phonemes": phoneme_details
                }
                
                word_level_scores.append(word_score)
            
        except Exception as e:
            logger.warning(f"Error generating word-level scores: {e}")
            # Fallback: create basic word scores without phoneme details
            words = predicted_text.split()
            base_accuracy = model_scores.get('accuracy', 0) * 10
            
            for word in words:
                word_level_scores.append({
                    "word": word.lower(),
                    "accuracy_score": base_accuracy,
                    "error_type": "None" if base_accuracy >= 80 else "Mispronunciation",
                    "phonemes": []
                })
        
        return word_level_scores


class MultiObjectiveWhisperModelEvaluator:
    """Evaluator for multi-objective fine-tuned Whisper models on SpeechOcean762 dataset."""
    
    def __init__(self, model_path: str):
        """Initialize the multi-objective Whisper model evaluator."""
        self.model_path = Path(model_path)
        
        # Initialize assessor
        self.assessor = MultiObjectiveWhisperAssessor(str(model_path))
        
        # Dataset and results
        self.dataset = None
        self.evaluation_results: List[MultiObjectiveEvaluationResult] = []
        
        logger.info(f"Initialized MultiObjectiveWhisperModelEvaluator with model: {model_path}")
    
    def _normalize_expert_score(self, score: float, min_val: float = 0, max_val: float = 10) -> float:
        """
        Normalize expert score from [0-10] to [0-1] to match model output range.
        
        This is critical for fair comparison - the model outputs normalized scores [0-1]
        while expert annotations are in raw scale [0-10].
        
        Args:
            score: Expert score in [0-10] range
            min_val: Minimum value in original scale (default: 0)
            max_val: Maximum value in original scale (default: 10)
            
        Returns:
            Normalized score in [0-1] range
        """
        if max_val <= min_val:
            return 0.5  # Default to middle value if invalid range
        
        # Clamp score to valid range before normalizing
        clamped_score = max(min_val, min(max_val, score))
        normalized = (clamped_score - min_val) / (max_val - min_val)
        
        return float(normalized)
    
    def load_dataset(self, split: str = "test", max_samples: Optional[int] = None) -> bool:
        """Load the SpeechOcean762 dataset."""
        if not DATASETS_AVAILABLE:
            print("[ERROR] HuggingFace datasets library not available. Please install with:")
            print("   pip install datasets")
            return False
        
        try:
            print(f"[*] Loading SpeechOcean762 dataset ({split} split)...")
            
            self.dataset = load_dataset("mispeech/speechocean762", split=split)
            
            if max_samples and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))
            
            print(f"[OK] Dataset loaded successfully: {len(self.dataset)} samples")
            
            sample = next(iter(self.dataset))
            print(f"[SAMPLE] Text: {sample['text']}")
            print(f"[SAMPLE] Speaker: {sample['speaker']} ({sample['gender']}, age {sample['age']})")
            print(f"[SAMPLE] Expert scores - Accuracy: {sample['accuracy']}, Fluency: {sample['fluency']}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            return False
    
    def _save_audio_sample(self, audio_data: np.ndarray, sampling_rate: int) -> str:
        """Save audio sample to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        sf.write(temp_path, audio_data, sampling_rate)
        return temp_path
    
    def evaluate_sample(self, sample: Dict[str, Any], sample_idx: int) -> MultiObjectiveEvaluationResult:
        """Evaluate a single sample from the dataset."""
        try:
            # Save audio to temporary file
            audio_path = self._save_audio_sample(sample['audio']['array'], sample['audio']['sampling_rate'])
            
            # Transcribe and assess
            result = self.assessor.transcribe_and_assess(audio_path, reference_text=sample['text'])
            
            # Extract expert scores and normalize to [0-1] to match model output range
            expert_scores = {
                'accuracy': self._normalize_expert_score(float(sample['accuracy'])),
                'fluency': self._normalize_expert_score(float(sample['fluency'])),
                'completeness': self._normalize_expert_score(float(sample['completeness'])),
                'prosodic': self._normalize_expert_score(float(sample['prosodic']))
            }
            
            # Create evaluation result
            eval_result = MultiObjectiveEvaluationResult(
                model_path=str(self.model_path),
                sample_idx=sample_idx,
                text=sample['text'],
                speaker=sample['speaker'],
                success=result['success'],
                predicted_text=result.get('predicted_text', ''),
                reference_text=result.get('reference_text', ''),
                word_error_rate=result.get('quality_metrics', {}).get('wer', 1.0),
                character_error_rate=result.get('quality_metrics', {}).get('cer', 1.0),
                bleu_score=result.get('quality_metrics', {}).get('bleu', 0.0),
                model_scores=result.get('model_scores', {
                    'accuracy': 0.0, 'fluency': 0.0,
                    'completeness': 0.0, 'prosodic': 0.0
                }),
                expert_scores=expert_scores,
                word_level_scores=result.get('word_level_scores', []),
                error=result.get('error', '')
            )
            
            # Clean up temporary file
            try:
                os.remove(audio_path)
            except:
                pass
            
            return eval_result
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample_idx}: {e}")
            return MultiObjectiveEvaluationResult(
                model_path=str(self.model_path),
                sample_idx=sample_idx,
                text=sample.get('text', ''),
                speaker=sample.get('speaker', 'unknown'),
                success=False,
                word_level_scores=[],
                error=str(e)
            )
    
    def run_evaluation(self, max_samples: Optional[int] = None, 
                      save_results: bool = True) -> MultiObjectiveEvaluationMetrics:
        """Run evaluation on the loaded dataset."""
        if not self.dataset:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return None
        
        print(f"[*] Starting multi-objective Whisper model evaluation on {len(self.dataset)} samples...")
        print(f"[MODEL] {self.model_path}")
        
        samples_to_evaluate = list(self.dataset)
        if max_samples:
            samples_to_evaluate = samples_to_evaluate[:max_samples]
        
        self.evaluation_results = []
        
        for idx, sample in enumerate(samples_to_evaluate):
            print(f"  [{idx+1}/{len(samples_to_evaluate)}] Evaluating sample {idx+1}...", end='\r')
            result = self.evaluate_sample(sample, idx)
            self.evaluation_results.append(result)
        
        print(f"\n[OK] Evaluation complete on {len(samples_to_evaluate)} samples")
        
        metrics = self._calculate_metrics()
        
        if save_results:
            self._save_evaluation_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> MultiObjectiveEvaluationMetrics:
        """Calculate evaluation metrics from results."""
        successful_results = [r for r in self.evaluation_results if r.success]
        
        if not successful_results:
            logger.error("No successful evaluations to calculate metrics")
            return MultiObjectiveEvaluationMetrics(
                wer=1.0, cer=1.0, bleu=0.0,
                accuracy_correlation=0.0, fluency_correlation=0.0,
                completeness_correlation=0.0, prosodic_correlation=0.0,
                accuracy_mae=10.0, fluency_mae=10.0,
                completeness_mae=10.0, prosodic_mae=10.0,
                accuracy_rmse=10.0, fluency_rmse=10.0,
                completeness_rmse=10.0, prosodic_rmse=10.0,
                total_samples=len(self.evaluation_results),
                successful_assessments=0,
                failed_assessments=len(self.evaluation_results),
                expert_score_stats={}, model_score_stats={}
            )
        
        print(f"[*] Calculating metrics from {len(successful_results)} successful evaluations...")
        
        # Extract transcription metrics
        wer_scores = [r.word_error_rate for r in successful_results]
        cer_scores = [r.character_error_rate for r in successful_results]
        bleu_scores = [r.bleu_score for r in successful_results]
        
        avg_wer = np.mean(wer_scores) if wer_scores else 1.0
        avg_cer = np.mean(cer_scores) if cer_scores else 1.0
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # Extract scores for analysis
        expert_scores = {
            'accuracy': [r.expert_scores['accuracy'] for r in successful_results],
            'fluency': [r.expert_scores['fluency'] for r in successful_results],
            'completeness': [r.expert_scores['completeness'] for r in successful_results],
            'prosodic': [r.expert_scores['prosodic'] for r in successful_results],
        }
        
        model_scores = {
            'accuracy': [r.model_scores['accuracy'] for r in successful_results],
            'fluency': [r.model_scores['fluency'] for r in successful_results],
            'completeness': [r.model_scores['completeness'] for r in successful_results],
            'prosodic': [r.model_scores['prosodic'] for r in successful_results],
        }
        
        # Calculate correlations
        correlations = {}
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            if len(expert_scores[metric]) > 1:
                try:
                    r, _ = stats.pearsonr(expert_scores[metric], model_scores[metric])
                    # Handle NaN correlation (e.g., when model scores are all constant)
                    if np.isnan(r):
                        correlations[f'{metric}_correlation'] = 0.0
                    else:
                        correlations[f'{metric}_correlation'] = float(r)
                except Exception as e:
                    logger.warning(f"Failed to calculate correlation for {metric}: {e}")
                    correlations[f'{metric}_correlation'] = 0.0
            else:
                correlations[f'{metric}_correlation'] = 0.0
        
        # Calculate MAE and RMSE
        mae_rmse = {}
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            mae = np.mean(np.abs(np.array(model_scores[metric]) - np.array(expert_scores[metric])))
            rmse = np.sqrt(np.mean((np.array(model_scores[metric]) - np.array(expert_scores[metric])) ** 2))
            mae_rmse[f'{metric}_mae'] = mae
            mae_rmse[f'{metric}_rmse'] = rmse
        
        # Calculate score statistics
        expert_stats = {}
        model_stats = {}
        
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            expert_stats[metric] = {
                'mean': float(np.mean(expert_scores[metric])),
                'std': float(np.std(expert_scores[metric])),
                'min': float(np.min(expert_scores[metric])),
                'max': float(np.max(expert_scores[metric]))
            }
            model_stats[metric] = {
                'mean': float(np.mean(model_scores[metric])),
                'std': float(np.std(model_scores[metric])),
                'min': float(np.min(model_scores[metric])),
                'max': float(np.max(model_scores[metric]))
            }
        
        return MultiObjectiveEvaluationMetrics(
            wer=float(avg_wer),
            cer=float(avg_cer),
            bleu=float(avg_bleu),
            **correlations,
            **mae_rmse,
            total_samples=len(self.evaluation_results),
            successful_assessments=len(successful_results),
            failed_assessments=len(self.evaluation_results) - len(successful_results),
            expert_score_stats=expert_stats,
            model_score_stats=model_stats
        )
    
    def _save_evaluation_results(self, metrics: MultiObjectiveEvaluationMetrics):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        model_name = self.model_path.name if self.model_path.name else "multi_objective_whisper_model"
        
        # Convert results to dictionaries
        detailed_results_data = []
        for result in self.evaluation_results:
            detailed_results_data.append({
                'sample_idx': result.sample_idx,
                'text': result.text,
                'speaker': result.speaker,
                'success': result.success,
                'predicted_text': result.predicted_text,
                'reference_text': result.reference_text,
                'metrics': {
                    'wer': result.word_error_rate,
                    'cer': result.character_error_rate,
                    'bleu': result.bleu_score
                },
                'model_scores': result.model_scores,
                'expert_scores': result.expert_scores,
                'word_level_scores': result.word_level_scores,
                'error': result.error
            })
        
        # Save detailed results as JSON
        detailed_results = {
            'evaluation_info': {
                'timestamp': timestamp,
                'model_path': str(self.model_path),
                'model_name': model_name,
                'dataset': 'speechocean762',
                'evaluation_type': 'multi_objective_whisper_pronunciation_assessment',
                'total_samples': metrics.total_samples,
                'successful_assessments': metrics.successful_assessments,
                'failed_assessments': metrics.failed_assessments
            },
            'transcription_metrics': {
                'wer': float(metrics.wer),
                'cer': float(metrics.cer),
                'bleu': float(metrics.bleu)
            },
            'assessment_metrics': {
                'correlations': {
                    'accuracy': float(metrics.accuracy_correlation),
                    'fluency': float(metrics.fluency_correlation),
                    'completeness': float(metrics.completeness_correlation),
                    'prosodic': float(metrics.prosodic_correlation)
                },
                'mae': {
                    'accuracy': float(metrics.accuracy_mae),
                    'fluency': float(metrics.fluency_mae),
                    'completeness': float(metrics.completeness_mae),
                    'prosodic': float(metrics.prosodic_mae)
                },
                'rmse': {
                    'accuracy': float(metrics.accuracy_rmse),
                    'fluency': float(metrics.fluency_rmse),
                    'completeness': float(metrics.completeness_rmse),
                    'prosodic': float(metrics.prosodic_rmse)
                }
            },
            'score_statistics': {
                'expert': metrics.expert_score_stats,
                'model': metrics.model_score_stats
            },
            'individual_results': detailed_results_data
        }
        
        detailed_results = convert_numpy_types(detailed_results)
        
        json_path = results_dir / f"multi_objective_whisper_{model_name}_evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Detailed results saved: {json_path}")
    
    def print_evaluation_summary(self, metrics: MultiObjectiveEvaluationMetrics):
        """Print a comprehensive evaluation summary."""
        print("\n" + "=" * 80)
        print("MULTI-OBJECTIVE WHISPER MODEL PRONUNCIATION EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\n[MODEL INFO]")
        print(f"   Model path: {self.model_path}")
        print(f"   Model name: {self.model_path.name}")
        print(f"   NOTE: All scores normalized to [0-1] range for fair comparison")
        
        print("\n[STATISTICS]")
        print(f"   Total samples: {metrics.total_samples}")
        print(f"   Successful assessments: {metrics.successful_assessments}")
        print(f"   Failed assessments: {metrics.failed_assessments}")
        success_rate = (metrics.successful_assessments / metrics.total_samples) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        print(f"\n[TRANSCRIPTION]")
        print(f"   WER (Word Error Rate): {metrics.wer:.3f}")
        print(f"   CER (Character Error Rate): {metrics.cer:.3f}")
        print(f"   BLEU Score: {metrics.bleu:.3f}")
        
        print(f"\n[SCORE NORMALIZATION INFO]")
        print(f"   Expert scores: Normalized from [0-10] to [0-1]")
        print(f"   Model scores: Direct output in [0-1] range")
        print(f"   This ensures fair comparison between model and expert assessments")
        
        print(f"\n[CORRELATION WITH EXPERTS]")
        print(f"   Accuracy:    {metrics.accuracy_correlation:.3f}")
        print(f"   Fluency:     {metrics.fluency_correlation:.3f}")
        print(f"   Completeness: {metrics.completeness_correlation:.3f}")
        print(f"   Prosodic:    {metrics.prosodic_correlation:.3f}")
        
        print(f"\n[MAE - Mean Absolute Error (normalized [0-1] scale)]")
        print(f"   Accuracy:    {metrics.accuracy_mae:.3f}")
        print(f"   Fluency:     {metrics.fluency_mae:.3f}")
        print(f"   Completeness: {metrics.completeness_mae:.3f}")
        print(f"   Prosodic:    {metrics.prosodic_mae:.3f}")
        
        print(f"\n[RMSE - Root Mean Square Error (normalized [0-1] scale)]")
        print(f"   Accuracy:    {metrics.accuracy_rmse:.3f}")
        print(f"   Fluency:     {metrics.fluency_rmse:.3f}")
        print(f"   Completeness: {metrics.completeness_rmse:.3f}")
        print(f"   Prosodic:    {metrics.prosodic_rmse:.3f}")
        
        print(f"\n[INTERPRETATION]")
        avg_correlation = np.mean([
            metrics.accuracy_correlation, metrics.fluency_correlation,
            metrics.completeness_correlation, metrics.prosodic_correlation
        ])
        
        if avg_correlation > 0.7:
            print(f"   [EXCELLENT] Strong correlation with human experts.")
        elif avg_correlation > 0.5:
            print(f"   [GOOD] Moderate correlation with human experts.")
        elif avg_correlation > 0.3:
            print(f"   [FAIR] Model may benefit from additional fine-tuning.")
        else:
            print(f"   [POOR] Model needs significant improvement.")
        
        print(f"   Average correlation: {avg_correlation:.3f}")
        
        print("=" * 80)
