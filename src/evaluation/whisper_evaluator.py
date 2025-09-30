"""
Fine-tuned Whisper Model Evaluation Module for Pronunciation Assessment

This module provides functionality to evaluate fine-tuned Whisper models
against the SpeechOcean762 dataset with expert human annotations, using
the same methodology as the Azure Speech evaluation.

The evaluation focuses on pronunciation assessment capabilities by:
- Loading fine-tuned Whisper models
- Processing audio files from SpeechOcean762 dataset
- Extracting pronunciation-related features from transcriptions
- Comparing results with expert annotations
- Computing correlation metrics similar to Azure Speech evaluation
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
    WhisperTokenizer,
    WhisperFeatureExtractor
)

# Evaluation metrics
from scipy import stats
import evaluate

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Local imports
from .dataset_evaluator import EvaluationMetrics, convert_numpy_types
from ..config_manager import ConfigManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WhisperEvaluationResult:
    """Container for Whisper model evaluation results."""
    
    # Basic info
    model_path: str
    sample_idx: int
    text: str
    speaker: str
    success: bool
    
    # Transcription results
    predicted_text: str = ""
    reference_text: str = ""
    
    # Confidence and quality metrics
    confidence_score: float = 0.0
    transcription_quality: float = 0.0
    
    # Pronunciation assessment scores (derived from transcription quality)
    pronunciation_scores: Dict[str, float] = None
    
    # Expert annotations
    expert_scores: Dict[str, float] = None
    
    # Error information
    error: str = ""
    
    # Additional metrics
    word_error_rate: float = 0.0
    character_error_rate: float = 0.0
    bleu_score: float = 0.0
    
    def __post_init__(self):
        if self.pronunciation_scores is None:
            self.pronunciation_scores = {
                'accuracy': 0.0,
                'fluency': 0.0, 
                'completeness': 0.0,
                'prosodic': 0.0
            }
        
        if self.expert_scores is None:
            self.expert_scores = {
                'accuracy': 0.0,
                'fluency': 0.0,
                'completeness': 0.0,
                'prosodic': 0.0
            }


class WhisperPronunciationAssessor:
    """
    Pronunciation assessment using fine-tuned Whisper models.
    
    This class provides pronunciation assessment capabilities by:
    1. Using transcription quality as a proxy for pronunciation quality
    2. Analyzing transcription confidence and accuracy
    3. Computing WER, CER, and BLEU scores
    4. Mapping these metrics to pronunciation assessment scores
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the Whisper pronunciation assessor.
        
        Args:
            model_path: Path to fine-tuned Whisper model
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        
        # Model components
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        
        # Evaluation metrics
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        
        # Load model
        self._load_model()
        
        logger.info(f"Initialized WhisperPronunciationAssessor with model: {model_path}")
    
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
        """Load the fine-tuned Whisper model and processor."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load model components
            self.model = WhisperForConditionalGeneration.from_pretrained(
                str(self.model_path)
            ).to(self.device)
            
            self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
            self.tokenizer = self.processor.tokenizer
            self.feature_extractor = self.processor.feature_extractor
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully")
            logger.info(f"Model device: {self.device}")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, reference_text: str = "") -> Dict[str, Any]:
        """
        Transcribe audio and compute quality metrics.
        
        Args:
            audio_path: Path to audio file
            reference_text: Reference transcription for comparison
            
        Returns:
            Dictionary containing transcription and quality metrics
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare input features
            input_features = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription with attention scores
            with torch.no_grad():
                # Generate with return_dict_in_generate for attention scores
                generated_outputs = self.model.generate(
                    input_features,
                    max_length=225,
                    num_beams=5,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    output_scores=True
                )
                
                predicted_ids = generated_outputs.sequences
                
                # Calculate confidence from generation scores
                confidence_score = self._calculate_confidence(generated_outputs)
            
            # Decode transcription
            predicted_text = self.tokenizer.decode(
                predicted_ids[0], skip_special_tokens=True
            ).strip().upper()
            
            # Clean reference text
            reference_text = reference_text.strip().upper()
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                predicted_text, reference_text
            )
            
            return {
                'predicted_text': predicted_text,
                'reference_text': reference_text,
                'confidence_score': confidence_score,
                'quality_metrics': quality_metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                'predicted_text': "",
                'reference_text': reference_text,
                'confidence_score': 0.0,
                'quality_metrics': {},
                'success': False,
                'error': str(e)
            }
    
    def _calculate_confidence(self, generated_outputs) -> float:
        """
        Calculate confidence score from generation outputs.
        
        Args:
            generated_outputs: Model generation outputs with scores
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if hasattr(generated_outputs, 'scores') and generated_outputs.scores:
                # Calculate average probability of generated tokens
                scores = generated_outputs.scores
                probabilities = []
                
                for score in scores:
                    # Apply softmax to get probabilities
                    probs = torch.softmax(score, dim=-1)
                    # Get max probability for each position
                    max_prob = torch.max(probs, dim=-1).values
                    probabilities.extend(max_prob.cpu().numpy())
                
                if probabilities:
                    return float(np.mean(probabilities))
            
            # Fallback: return moderate confidence
            return 0.5
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _calculate_quality_metrics(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        Calculate transcription quality metrics.
        
        Args:
            predicted: Predicted transcription
            reference: Reference transcription
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            # Word Error Rate (WER)
            if reference and predicted:
                wer = self.wer_metric.compute(
                    predictions=[predicted], references=[reference]
                )
                metrics['wer'] = wer
                metrics['word_accuracy'] = max(0, 1 - wer)  # Convert to accuracy
            else:
                metrics['wer'] = 1.0
                metrics['word_accuracy'] = 0.0
            
            # Character Error Rate (CER)
            cer = self._calculate_cer(predicted, reference)
            metrics['cer'] = cer
            metrics['char_accuracy'] = max(0, 1 - cer)
            
            # BLEU Score
            if reference and predicted:
                try:
                    bleu = self.bleu_metric.compute(
                        predictions=[predicted.split()],
                        references=[[reference.split()]]
                    )
                    metrics['bleu'] = bleu['bleu']
                except:
                    metrics['bleu'] = 0.0
            else:
                metrics['bleu'] = 0.0
            
            # Length ratio (completeness indicator)
            if reference:
                length_ratio = len(predicted.split()) / len(reference.split())
                metrics['length_ratio'] = length_ratio
                metrics['completeness_ratio'] = min(1.0, length_ratio)
            else:
                metrics['length_ratio'] = 0.0
                metrics['completeness_ratio'] = 0.0
            
            # Overall transcription quality (composite score)
            word_acc = metrics['word_accuracy']
            char_acc = metrics['char_accuracy']
            bleu = metrics['bleu']
            completeness = metrics['completeness_ratio']
            
            # Weighted average of quality indicators
            quality_score = (
                0.4 * word_acc +      # Word accuracy is most important
                0.3 * char_acc +      # Character accuracy
                0.2 * bleu +          # BLEU score for fluency
                0.1 * completeness    # Completeness
            )
            metrics['overall_quality'] = quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            # Return zero metrics on error
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
        
        # Simple character-level edit distance calculation
        ref_chars = list(reference.replace(" ", ""))
        pred_chars = list(predicted.replace(" ", ""))
        
        # Use dynamic programming for edit distance
        m, n = len(ref_chars), len(pred_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill dp table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_chars[i-1] == pred_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_distance = dp[m][n]
        cer = edit_distance / len(ref_chars) if ref_chars else 0.0
        return cer
    
    def assess_pronunciation(self, transcription_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert transcription quality metrics to pronunciation assessment scores.
        
        This method maps transcription quality to pronunciation scores on a 0-10 scale
        to match the expert annotation format.
        
        Args:
            transcription_result: Result from transcribe_audio()
            
        Returns:
            Dictionary with pronunciation scores (0-10 scale)
        """
        if not transcription_result['success']:
            return {
                'accuracy': 0.0,
                'fluency': 0.0,
                'completeness': 0.0,
                'prosodic': 0.0
            }
        
        quality_metrics = transcription_result['quality_metrics']
        confidence = transcription_result['confidence_score']
        
        # Map transcription quality to pronunciation scores (0-10 scale)
        
        # Accuracy: Based on word accuracy and character accuracy
        word_acc = quality_metrics.get('word_accuracy', 0)
        char_acc = quality_metrics.get('char_accuracy', 0)
        accuracy_score = (0.7 * word_acc + 0.3 * char_acc) * 10
        
        # Fluency: Based on BLEU score and confidence
        bleu = quality_metrics.get('bleu', 0)
        fluency_score = (0.6 * bleu + 0.4 * confidence) * 10
        
        # Completeness: Based on length ratio and word accuracy
        completeness_ratio = quality_metrics.get('completeness_ratio', 0)
        completeness_score = (0.8 * completeness_ratio + 0.2 * word_acc) * 10
        
        # Prosodic: Based on overall quality and confidence (proxy)
        overall_quality = quality_metrics.get('overall_quality', 0)
        prosodic_score = (0.7 * overall_quality + 0.3 * confidence) * 10
        
        # Ensure scores are in valid range
        scores = {
            'accuracy': max(0, min(10, accuracy_score)),
            'fluency': max(0, min(10, fluency_score)),
            'completeness': max(0, min(10, completeness_score)),
            'prosodic': max(0, min(10, prosodic_score))
        }
        
        return scores


class WhisperModelEvaluator:
    """
    Evaluator for fine-tuned Whisper models on SpeechOcean762 dataset.
    
    This class provides comprehensive evaluation of Whisper models for pronunciation
    assessment, using the same methodology as the Azure Speech evaluation.
    """
    
    def __init__(self, model_path: str, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the Whisper model evaluator.
        
        Args:
            model_path: Path to fine-tuned Whisper model
            config_manager: Optional configuration manager
        """
        self.model_path = Path(model_path)
        self.config_manager = config_manager
        
        # Initialize pronunciation assessor
        self.pronunciation_assessor = WhisperPronunciationAssessor(str(model_path))
        
        # Dataset and results
        self.dataset = None
        self.evaluation_results: List[WhisperEvaluationResult] = []
        
        logger.info(f"Initialized WhisperModelEvaluator with model: {model_path}")
    
    def load_dataset(self, split: str = "test", max_samples: Optional[int] = None) -> bool:
        """
        Load the SpeechOcean762 dataset.
        
        Args:
            split: Dataset split to load ("test", "train", "validation")  
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            bool: True if dataset loaded successfully
        """
        if not DATASETS_AVAILABLE:
            print("❌ HuggingFace datasets library not available. Please install with:")
            print("   pip install datasets")
            return False
        
        try:
            print(f"📥 Loading SpeechOcean762 dataset ({split} split)...")
            
            # Load the dataset
            self.dataset = load_dataset("mispeech/speechocean762", split=split)
            
            # Limit samples if specified
            if max_samples and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))
                print(f"📊 Limited to {max_samples} samples for evaluation")
            
            print(f"✅ Dataset loaded successfully: {len(self.dataset)} samples")
            
            # Show a sample
            sample = next(iter(self.dataset))
            print(f"📝 Sample text: {sample['text']}")
            print(f"👤 Speaker: {sample['speaker']} ({sample['gender']}, age {sample['age']})")
            print(f"🎯 Expert scores - Accuracy: {sample['accuracy']}, Fluency: {sample['fluency']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
    
    def _save_audio_sample(self, audio_data: np.ndarray, sampling_rate: int) -> str:
        """
        Save audio sample to temporary file.
        
        Args:
            audio_data: Audio numpy array
            sampling_rate: Sampling rate of the audio
            
        Returns:
            str: Path to temporary audio file
        """
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio using soundfile
        sf.write(temp_path, audio_data, sampling_rate)
        
        return temp_path
    
    def evaluate_sample(self, sample: Dict[str, Any], sample_idx: int) -> WhisperEvaluationResult:
        """
        Evaluate a single sample from the dataset.
        
        Args:
            sample: Dataset sample containing audio, text, and expert scores
            sample_idx: Index of the sample
            
        Returns:
            WhisperEvaluationResult containing evaluation results for this sample
        """
        try:
            print(f"🎵 Evaluating sample {sample_idx + 1}: '{sample['text']}'")
            
            # Save audio to temporary file
            temp_audio_path = self._save_audio_sample(
                sample['audio']['array'],
                sample['audio']['sampling_rate']
            )
            
            try:
                # Run transcription
                transcription_result = self.pronunciation_assessor.transcribe_audio(
                    temp_audio_path, sample['text']
                )
                
                if transcription_result['success']:
                    # Get pronunciation assessment scores
                    pronunciation_scores = self.pronunciation_assessor.assess_pronunciation(
                        transcription_result
                    )
                    
                    # Extract quality metrics
                    quality_metrics = transcription_result['quality_metrics']
                    
                    # Create result object
                    result = WhisperEvaluationResult(
                        model_path=str(self.model_path),
                        sample_idx=sample_idx,
                        text=sample['text'],
                        speaker=sample['speaker'],
                        success=True,
                        predicted_text=transcription_result['predicted_text'],
                        reference_text=transcription_result['reference_text'],
                        confidence_score=transcription_result['confidence_score'],
                        transcription_quality=quality_metrics.get('overall_quality', 0),
                        pronunciation_scores=pronunciation_scores,
                        expert_scores={
                            'accuracy': sample.get('accuracy', 0),
                            'fluency': sample.get('fluency', 0),
                            'completeness': sample.get('completeness', 0),
                            'prosodic': sample.get('prosodic', 0)
                        },
                        word_error_rate=quality_metrics.get('wer', 1.0),
                        character_error_rate=quality_metrics.get('cer', 1.0),
                        bleu_score=quality_metrics.get('bleu', 0.0)
                    )
                    
                    print(f"✅ Sample {sample_idx + 1} evaluated successfully")
                    print(f"   Predicted: '{transcription_result['predicted_text']}'")
                    print(f"   Reference: '{transcription_result['reference_text']}'")
                    print(f"   WER: {quality_metrics.get('wer', 1.0):.3f}")
                    print(f"   Pronunciation scores: {pronunciation_scores}")
                    
                else:
                    # Failed transcription
                    result = WhisperEvaluationResult(
                        model_path=str(self.model_path),
                        sample_idx=sample_idx,
                        text=sample['text'],
                        speaker=sample['speaker'],
                        success=False,
                        expert_scores={
                            'accuracy': sample.get('accuracy', 0),
                            'fluency': sample.get('fluency', 0),
                            'completeness': sample.get('completeness', 0),
                            'prosodic': sample.get('prosodic', 0)
                        },
                        error=transcription_result.get('error', 'Unknown error')
                    )
                    
                    print(f"❌ Sample {sample_idx + 1} failed: {result.error}")
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Error evaluating sample {sample_idx + 1}: {e}")
            return WhisperEvaluationResult(
                model_path=str(self.model_path),
                sample_idx=sample_idx,
                text=sample.get('text', ''),
                speaker=sample.get('speaker', ''),
                success=False,
                expert_scores={
                    'accuracy': sample.get('accuracy', 0),
                    'fluency': sample.get('fluency', 0),
                    'completeness': sample.get('completeness', 0),
                    'prosodic': sample.get('prosodic', 0)
                },
                error=str(e)
            )
    
    def run_evaluation(self, max_samples: Optional[int] = None, 
                      save_results: bool = True) -> EvaluationMetrics:
        """
        Run evaluation on the loaded dataset.
        
        Args:
            max_samples: Maximum number of samples to evaluate (None for all)
            save_results: Whether to save detailed results to file
            
        Returns:
            EvaluationMetrics: Computed evaluation metrics
        """
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"🚀 Starting Whisper model evaluation on {len(self.dataset)} samples...")
        print(f"📁 Model path: {self.model_path}")
        
        # Limit samples if specified
        samples_to_evaluate = list(self.dataset)
        if max_samples:
            samples_to_evaluate = samples_to_evaluate[:max_samples]
        
        self.evaluation_results = []
        
        # Evaluate each sample
        for idx, sample in enumerate(samples_to_evaluate):
            result = self.evaluate_sample(sample, idx)
            self.evaluation_results.append(result)
            
            # Progress update
            if (idx + 1) % 10 == 0 or idx == len(samples_to_evaluate) - 1:
                success_count = sum(1 for r in self.evaluation_results if r.success)
                print(f"📊 Progress: {idx + 1}/{len(samples_to_evaluate)} "
                      f"(Success rate: {success_count/(idx+1)*100:.1f}%)")
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics()
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> EvaluationMetrics:
        """Calculate evaluation metrics from results."""
        successful_results = [r for r in self.evaluation_results if r.success]
        
        if not successful_results:
            print("❌ No successful evaluations to calculate metrics")
            return EvaluationMetrics(
                accuracy_correlation=0, fluency_correlation=0,
                completeness_correlation=0, prosodic_correlation=0,
                accuracy_mae=10, fluency_mae=10, completeness_mae=10, prosodic_mae=10,
                accuracy_rmse=10, fluency_rmse=10, completeness_rmse=10, prosodic_rmse=10,
                total_samples=len(self.evaluation_results),
                successful_assessments=0,
                failed_assessments=len(self.evaluation_results),
                expert_score_stats={}, azure_score_stats={}
            )
        
        print(f"📊 Calculating metrics from {len(successful_results)} successful evaluations...")
        
        # Extract scores for analysis
        expert_scores = {
            'accuracy': [r.expert_scores['accuracy'] for r in successful_results],
            'fluency': [r.expert_scores['fluency'] for r in successful_results],
            'completeness': [r.expert_scores['completeness'] for r in successful_results],
            'prosodic': [r.expert_scores['prosodic'] for r in successful_results],
        }
        
        whisper_scores = {
            'accuracy': [r.pronunciation_scores['accuracy'] for r in successful_results],
            'fluency': [r.pronunciation_scores['fluency'] for r in successful_results],
            'completeness': [r.pronunciation_scores['completeness'] for r in successful_results],
            'prosodic': [r.pronunciation_scores['prosodic'] for r in successful_results],
        }
        
        # Calculate correlations
        correlations = {}
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            try:
                correlation, p_value = stats.pearsonr(
                    expert_scores[metric], whisper_scores[metric]
                )
                if np.isnan(correlation):
                    correlation = 0.0
                correlations[f'{metric}_correlation'] = correlation
                print(f"📈 {metric.capitalize()} correlation: {correlation:.3f} (p={p_value:.4f})")
            except Exception as e:
                print(f"⚠️ Failed to calculate {metric} correlation: {e}")
                correlations[f'{metric}_correlation'] = 0.0
        
        # Calculate MAE and RMSE
        mae_rmse = {}
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            try:
                expert_vals = np.array(expert_scores[metric])
                whisper_vals = np.array(whisper_scores[metric])
                
                mae = np.mean(np.abs(expert_vals - whisper_vals))
                rmse = np.sqrt(np.mean((expert_vals - whisper_vals) ** 2))
                
                mae_rmse[f'{metric}_mae'] = mae
                mae_rmse[f'{metric}_rmse'] = rmse
                print(f"📏 {metric.capitalize()} - MAE: {mae:.3f}, RMSE: {rmse:.3f}")
            except Exception as e:
                print(f"⚠️ Failed to calculate {metric} MAE/RMSE: {e}")
                mae_rmse[f'{metric}_mae'] = 10.0
                mae_rmse[f'{metric}_rmse'] = 10.0
        
        # Calculate score statistics
        expert_stats = {}
        whisper_stats = {}
        
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            try:
                expert_vals = expert_scores[metric]
                whisper_vals = whisper_scores[metric]
                
                expert_stats[metric] = {
                    'mean': float(np.mean(expert_vals)),
                    'std': float(np.std(expert_vals)),
                    'min': float(np.min(expert_vals)),
                    'max': float(np.max(expert_vals))
                }
                
                whisper_stats[metric] = {
                    'mean': float(np.mean(whisper_vals)),
                    'std': float(np.std(whisper_vals)),
                    'min': float(np.min(whisper_vals)),
                    'max': float(np.max(whisper_vals))
                }
            except:
                expert_stats[metric] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
                whisper_stats[metric] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return EvaluationMetrics(
            **correlations,
            **mae_rmse,
            total_samples=len(self.evaluation_results),
            successful_assessments=len(successful_results),
            failed_assessments=len(self.evaluation_results) - len(successful_results),
            expert_score_stats=expert_stats,
            azure_score_stats=whisper_stats  # Reusing the field name for consistency
        )
    
    def _save_evaluation_results(self, metrics: EvaluationMetrics):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        model_name = self.model_path.name if self.model_path.name else "whisper_model"
        
        # Convert WhisperEvaluationResult objects to dictionaries
        detailed_results_data = []
        for result in self.evaluation_results:
            result_dict = {
                'model_path': result.model_path,
                'sample_idx': result.sample_idx,
                'text': result.text,
                'speaker': result.speaker,
                'success': result.success,
                'predicted_text': result.predicted_text,
                'reference_text': result.reference_text,
                'confidence_score': result.confidence_score,
                'transcription_quality': result.transcription_quality,
                'pronunciation_scores': result.pronunciation_scores,
                'expert_scores': result.expert_scores,
                'error': result.error,
                'word_error_rate': result.word_error_rate,
                'character_error_rate': result.character_error_rate,
                'bleu_score': result.bleu_score
            }
            detailed_results_data.append(result_dict)
        
        # Save detailed results as JSON
        detailed_results = {
            'evaluation_info': {
                'timestamp': timestamp,
                'model_path': str(self.model_path),
                'model_name': model_name,
                'dataset': 'speechocean762',
                'evaluation_type': 'whisper_pronunciation_assessment',
                'total_samples': metrics.total_samples,
                'successful_assessments': metrics.successful_assessments,
                'failed_assessments': metrics.failed_assessments
            },
            'metrics': {
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
                'whisper': metrics.azure_score_stats  # Using existing field
            },
            'individual_results': detailed_results_data
        }
        
        # Apply numpy type conversion
        detailed_results = convert_numpy_types(detailed_results)
        
        json_path = results_dir / f"whisper_{model_name}_evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Detailed results saved: {json_path}")
        
        # Save summary CSV for easy analysis
        successful_results = [r for r in self.evaluation_results if r.success]
        if successful_results:
            import pandas as pd
            
            # Create summary DataFrame
            summary_data = []
            for result in successful_results:
                summary_data.append({
                    'sample_idx': result.sample_idx,
                    'speaker': result.speaker,
                    'text': result.text,
                    'predicted_text': result.predicted_text,
                    'confidence_score': result.confidence_score,
                    'wer': result.word_error_rate,
                    'cer': result.character_error_rate,
                    'bleu_score': result.bleu_score,
                    # Expert scores
                    'expert_accuracy': result.expert_scores['accuracy'],
                    'expert_fluency': result.expert_scores['fluency'],
                    'expert_completeness': result.expert_scores['completeness'],
                    'expert_prosodic': result.expert_scores['prosodic'],
                    # Whisper scores
                    'whisper_accuracy': result.pronunciation_scores['accuracy'],
                    'whisper_fluency': result.pronunciation_scores['fluency'],
                    'whisper_completeness': result.pronunciation_scores['completeness'],
                    'whisper_prosodic': result.pronunciation_scores['prosodic']
                })
            
            df = pd.DataFrame(summary_data)
            csv_path = results_dir / f"whisper_{model_name}_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"📊 Summary CSV saved: {csv_path}")
    
    def print_evaluation_summary(self, metrics: EvaluationMetrics):
        """Print a comprehensive evaluation summary."""
        print("\n" + "=" * 80)
        print("🎯 WHISPER MODEL PRONUNCIATION EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\n🤖 Model Information:")
        print(f"   Model path: {self.model_path}")
        print(f"   Model name: {self.model_path.name}")
        
        print(f"\n📊 Sample Statistics:")
        print(f"   Total samples: {metrics.total_samples}")
        print(f"   Successful assessments: {metrics.successful_assessments}")
        print(f"   Failed assessments: {metrics.failed_assessments}")
        success_rate = (metrics.successful_assessments / metrics.total_samples) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
        print(f"\n🔗 Correlation with Expert Scores:")
        print(f"   Accuracy:    {metrics.accuracy_correlation:.3f}")
        print(f"   Fluency:     {metrics.fluency_correlation:.3f}")
        print(f"   Completeness: {metrics.completeness_correlation:.3f}")
        print(f"   Prosodic:    {metrics.prosodic_correlation:.3f}")
        
        print(f"\n📏 Mean Absolute Error (MAE):")
        print(f"   Accuracy:    {metrics.accuracy_mae:.2f}")
        print(f"   Fluency:     {metrics.fluency_mae:.2f}")
        print(f"   Completeness: {metrics.completeness_mae:.2f}")
        print(f"   Prosodic:    {metrics.prosodic_mae:.2f}")
        
        print(f"\n📐 Root Mean Square Error (RMSE):")
        print(f"   Accuracy:    {metrics.accuracy_rmse:.2f}")
        print(f"   Fluency:     {metrics.fluency_rmse:.2f}")
        print(f"   Completeness: {metrics.completeness_rmse:.2f}")
        print(f"   Prosodic:    {metrics.prosodic_rmse:.2f}")
        
        print(f"\n📈 Score Distribution Comparison:")
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            expert_stats = metrics.expert_score_stats.get(metric, {})
            whisper_stats = metrics.azure_score_stats.get(metric, {})
            print(f"   {metric.capitalize()}:")
            print(f"     Expert:  mean={expert_stats.get('mean', 0):.2f}, std={expert_stats.get('std', 0):.2f}")
            print(f"     Whisper: mean={whisper_stats.get('mean', 0):.2f}, std={whisper_stats.get('std', 0):.2f}")
        
        # Interpretation
        print(f"\n🎭 Interpretation:")
        avg_correlation = np.mean([
            metrics.accuracy_correlation, metrics.fluency_correlation,
            metrics.completeness_correlation, metrics.prosodic_correlation
        ])
        
        if avg_correlation > 0.7:
            print("   🎉 Strong correlation with expert scores! The model shows excellent pronunciation assessment capabilities.")
        elif avg_correlation > 0.5:
            print("   👍 Moderate correlation with expert scores. The model shows promising pronunciation assessment capabilities.")
        else:
            print("   ⚠️ Weak correlation with expert scores. The model may need further fine-tuning for pronunciation assessment.")
        
        print(f"   📊 Average correlation: {avg_correlation:.3f}")
        
        print("=" * 80)


def main():
    """
    Main function for running Whisper model evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model for pronunciation assessment")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned Whisper model")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--output-dir", type=str, help="Directory to save results")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return
    
    try:
        # Initialize evaluator
        evaluator = WhisperModelEvaluator(str(model_path))
        
        # Load dataset
        if not evaluator.load_dataset(split=args.split, max_samples=args.max_samples):
            print("❌ Failed to load dataset")
            return
        
        # Run evaluation
        print(f"\n🚀 Starting evaluation...")
        metrics = evaluator.run_evaluation(
            max_samples=args.max_samples,
            save_results=True
        )
        
        # Print summary
        evaluator.print_evaluation_summary(metrics)
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()