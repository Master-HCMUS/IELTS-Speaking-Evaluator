"""
Standalone Whisper Model Evaluation for Pronunciation Assessment

This is a standalone version that doesn't depend on the Azure Speech evaluation framework.
It provides its own implementation of evaluation metrics and data structures.
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
class StandaloneEvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Correlation metrics
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
    whisper_score_stats: Dict[str, Dict[str, float]]


@dataclass
class WhisperEvaluationResult:
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


class StandaloneWhisperPronunciationAssessor:
    """Standalone pronunciation assessment using fine-tuned Whisper models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the Whisper pronunciation assessor."""
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
        
        logger.info(f"Initialized StandaloneWhisperPronunciationAssessor with model: {model_path}")
    
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
        """Transcribe audio and compute quality metrics."""
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
        """Calculate confidence score from generation outputs."""
        try:
            if hasattr(generated_outputs, 'scores') and generated_outputs.scores:
                scores = generated_outputs.scores
                probabilities = []
                
                for score in scores:
                    probs = torch.softmax(score, dim=-1)
                    max_prob = torch.max(probs, dim=-1).values
                    probabilities.extend(max_prob.cpu().numpy())
                
                if probabilities:
                    return float(np.mean(probabilities))
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _calculate_quality_metrics(self, predicted: str, reference: str) -> Dict[str, float]:
        """Calculate transcription quality metrics."""
        metrics = {}
        
        try:
            # Word Error Rate (WER)
            if reference and predicted:
                wer = self.wer_metric.compute(
                    predictions=[predicted], references=[reference]
                )
                metrics['wer'] = wer
                metrics['word_accuracy'] = max(0, 1 - wer)
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
            
            quality_score = (
                0.4 * word_acc +
                0.3 * char_acc +
                0.2 * bleu +
                0.1 * completeness
            )
            metrics['overall_quality'] = quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
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
    
    def assess_pronunciation(self, transcription_result: Dict[str, Any]) -> Dict[str, float]:
        """Convert transcription quality metrics to pronunciation assessment scores."""
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
        word_acc = quality_metrics.get('word_accuracy', 0)
        char_acc = quality_metrics.get('char_accuracy', 0)
        accuracy_score = (0.7 * word_acc + 0.3 * char_acc) * 10
        
        bleu = quality_metrics.get('bleu', 0)
        fluency_score = (0.6 * bleu + 0.4 * confidence) * 10
        
        completeness_ratio = quality_metrics.get('completeness_ratio', 0)
        completeness_score = (0.8 * completeness_ratio + 0.2 * word_acc) * 10
        
        overall_quality = quality_metrics.get('overall_quality', 0)
        prosodic_score = (0.7 * overall_quality + 0.3 * confidence) * 10
        
        scores = {
            'accuracy': max(0, min(10, accuracy_score)),
            'fluency': max(0, min(10, fluency_score)),
            'completeness': max(0, min(10, completeness_score)),
            'prosodic': max(0, min(10, prosodic_score))
        }
        
        return scores


class StandaloneWhisperModelEvaluator:
    """Standalone evaluator for fine-tuned Whisper models on SpeechOcean762 dataset."""
    
    def __init__(self, model_path: str):
        """Initialize the Whisper model evaluator."""
        self.model_path = Path(model_path)
        
        # Initialize pronunciation assessor
        self.pronunciation_assessor = StandaloneWhisperPronunciationAssessor(str(model_path))
        
        # Dataset and results
        self.dataset = None
        self.evaluation_results: List[WhisperEvaluationResult] = []
        
        logger.info(f"Initialized StandaloneWhisperModelEvaluator with model: {model_path}")
    
    def load_dataset(self, split: str = "test", max_samples: Optional[int] = None) -> bool:
        """Load the SpeechOcean762 dataset."""
        if not DATASETS_AVAILABLE:
            print("❌ HuggingFace datasets library not available. Please install with:")
            print("   pip install datasets")
            return False
        
        try:
            print(f"📥 Loading SpeechOcean762 dataset ({split} split)...")
            
            self.dataset = load_dataset("mispeech/speechocean762", split=split)
            
            if max_samples and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))
                print(f"📊 Limited to {max_samples} samples for evaluation")
            
            print(f"✅ Dataset loaded successfully: {len(self.dataset)} samples")
            
            sample = next(iter(self.dataset))
            print(f"📝 Sample text: {sample['text']}")
            print(f"👤 Speaker: {sample['speaker']} ({sample['gender']}, age {sample['age']})")
            print(f"🎯 Expert scores - Accuracy: {sample['accuracy']}, Fluency: {sample['fluency']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
    
    def _save_audio_sample(self, audio_data: np.ndarray, sampling_rate: int) -> str:
        """Save audio sample to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        sf.write(temp_path, audio_data, sampling_rate)
        return temp_path
    
    def evaluate_sample(self, sample: Dict[str, Any], sample_idx: int) -> WhisperEvaluationResult:
        """Evaluate a single sample from the dataset."""
        try:
            print(f"🎵 Evaluating sample {sample_idx + 1}: '{sample['text']}'")
            
            temp_audio_path = self._save_audio_sample(
                sample['audio']['array'],
                sample['audio']['sampling_rate']
            )
            
            try:
                transcription_result = self.pronunciation_assessor.transcribe_audio(
                    temp_audio_path, sample['text']
                )
                
                if transcription_result['success']:
                    pronunciation_scores = self.pronunciation_assessor.assess_pronunciation(
                        transcription_result
                    )
                    
                    quality_metrics = transcription_result['quality_metrics']
                    
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
                    
                else:
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
                      save_results: bool = True) -> StandaloneEvaluationMetrics:
        """Run evaluation on the loaded dataset."""
        if not self.dataset:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print(f"🚀 Starting Whisper model evaluation on {len(self.dataset)} samples...")
        print(f"📁 Model path: {self.model_path}")
        
        samples_to_evaluate = list(self.dataset)
        if max_samples:
            samples_to_evaluate = samples_to_evaluate[:max_samples]
        
        self.evaluation_results = []
        
        for idx, sample in enumerate(samples_to_evaluate):
            result = self.evaluate_sample(sample, idx)
            self.evaluation_results.append(result)
            
            if (idx + 1) % 10 == 0 or idx == len(samples_to_evaluate) - 1:
                success_count = sum(1 for r in self.evaluation_results if r.success)
                print(f"📊 Progress: {idx + 1}/{len(samples_to_evaluate)} "
                      f"(Success rate: {success_count/(idx+1)*100:.1f}%)")
        
        metrics = self._calculate_metrics()
        
        if save_results:
            self._save_evaluation_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> StandaloneEvaluationMetrics:
        """Calculate evaluation metrics from results."""
        successful_results = [r for r in self.evaluation_results if r.success]
        
        if not successful_results:
            print("❌ No successful evaluations to calculate metrics")
            return StandaloneEvaluationMetrics(
                accuracy_correlation=0, fluency_correlation=0,
                completeness_correlation=0, prosodic_correlation=0,
                accuracy_mae=10, fluency_mae=10, completeness_mae=10, prosodic_mae=10,
                accuracy_rmse=10, fluency_rmse=10, completeness_rmse=10, prosodic_rmse=10,
                total_samples=len(self.evaluation_results),
                successful_assessments=0,
                failed_assessments=len(self.evaluation_results),
                expert_score_stats={}, whisper_score_stats={}
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
        
        return StandaloneEvaluationMetrics(
            **correlations,
            **mae_rmse,
            total_samples=len(self.evaluation_results),
            successful_assessments=len(successful_results),
            failed_assessments=len(self.evaluation_results) - len(successful_results),
            expert_score_stats=expert_stats,
            whisper_score_stats=whisper_stats
        )
    
    def _save_evaluation_results(self, metrics: StandaloneEvaluationMetrics):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        model_name = self.model_path.name if self.model_path.name else "whisper_model"
        
        # Convert results to dictionaries
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
                'evaluation_type': 'standalone_whisper_pronunciation_assessment',
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
                'whisper': metrics.whisper_score_stats
            },
            'individual_results': detailed_results_data
        }
        
        detailed_results = convert_numpy_types(detailed_results)
        
        json_path = results_dir / f"standalone_whisper_{model_name}_evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Detailed results saved: {json_path}")
    
    def print_evaluation_summary(self, metrics: StandaloneEvaluationMetrics):
        """Print a comprehensive evaluation summary."""
        print("\n" + "=" * 80)
        print("🎯 STANDALONE WHISPER MODEL PRONUNCIATION EVALUATION RESULTS")
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