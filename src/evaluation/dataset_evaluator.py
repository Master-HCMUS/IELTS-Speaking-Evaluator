"""
SpeechOcean762 Dataset Evaluation Module

This module provides functionality to evaluate Azure Speech pronunciation assessment
against the SpeechOcean762 dataset with expert human annotations.

The SpeechOcean762 dataset contains:
- Audio files with pronunciation scores from 5 expert annotators
- Sentence-level scores: accuracy (0-10), fluency (0-10), completeness (0-10), prosodic (0-10)
- Word-level scores: accuracy (0-10), stress (5/10)
- Phoneme-level scores: pronunciation goodness (0-2)
"""

import os
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io.wavfile import write

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from ..pronunciation_service import AzureSpeechPronunciationService
from ..config_manager import ConfigManager


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
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
class EvaluationMetrics:
    """Container for evaluation metrics between Azure Speech and expert annotations."""
    
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
    azure_score_stats: Dict[str, Dict[str, float]]


class SpeechOcean762Evaluator:
    """
    Evaluator for comparing Azure Speech assessment with SpeechOcean762 expert annotations.
    
    This class loads the SpeechOcean762 dataset, runs Azure Speech pronunciation assessment
    on the audio files, and compares the results with expert human annotations.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the evaluator.
        
        Args:
            config_manager: Configuration manager for Azure Speech settings
        """
        self.config_manager = config_manager
        self.pronunciation_service: Optional[AzureSpeechPronunciationService] = None
        self.dataset = None
        self.evaluation_results: List[Dict[str, Any]] = []
        
        # Initialize pronunciation service
        self._initialize_pronunciation_service()
    
    def _initialize_pronunciation_service(self) -> bool:
        """Initialize Azure Speech pronunciation service."""
        try:
            if not self.config_manager.is_speech_configured():
                print("❌ Azure Speech service not configured. Please configure it first.")
                return False
            
            speech_config = self.config_manager.get_azure_speech_config()
            self.pronunciation_service = AzureSpeechPronunciationService(
                speech_key=speech_config['speech_key'],
                speech_region=speech_config['speech_region'],
                locale=speech_config.get('locale', 'en-US')
            )
            
            print("✅ Azure Speech pronunciation service initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize pronunciation service: {e}")
            return False
    
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
            
            print(f"✅ Dataset loaded successfully: {len(self.dataset)} samples")
            print(f"📊 Sample info: {self.dataset.features}")
            
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
        
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            # Normalize to [-1, 1] range if needed
            if audio_data.max() <= 1.0 and audio_data.min() >= -1.0:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Save as WAV file
        write(temp_path, sampling_rate, audio_data)
        return temp_path
    
    def _normalize_azure_scores(self, azure_result: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize Azure Speech scores to match expert annotation ranges.
        
        Azure scores are 0-100, expert scores have different ranges:
        - Accuracy: 0-10
        - Fluency: 0-10
        - Completeness: 0-10
        - Prosodic: 0-10 (not directly available in Azure, use fluency as proxy)
        
        Args:
            azure_result: Azure Speech assessment result
            
        Returns:
            Dict with normalized scores
        """
        overall_scores = azure_result.get('overall_scores', {})
        
        # Convert Azure 0-100 scores to expert ranges
        normalized = {
            'accuracy': (overall_scores.get('accuracy_score', 0) / 100) * 10,      # 0-100 → 0-10
            'fluency': (overall_scores.get('fluency_score', 0) / 100) * 10,        # 0-100 → 0-10
            'completeness': (overall_scores.get('completeness_score', 0) / 100) * 10,  # 0-100 → 0-10
            'prosodic': (overall_scores.get('fluency_score', 0) / 100) * 10,       # Use fluency as proxy for prosodic
        }
        
        return normalized
    
    def evaluate_sample(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """
        Evaluate a single sample from the dataset.
        
        Args:
            sample: Dataset sample containing audio, text, and expert scores
            sample_idx: Index of the sample
            
        Returns:
            Dict containing evaluation results for this sample
        """
        try:
            print(f"🎵 Evaluating sample {sample_idx + 1}: '{sample['text']}'")
            
            # Save audio to temporary file
            temp_audio_path = self._save_audio_sample(
                sample['audio']['array'],
                sample['audio']['sampling_rate']
            )
            
            try:
                # Run Azure Speech pronunciation assessment
                start_time = time.time()
                azure_result = self.pronunciation_service.assess_pronunciation_from_file(
                    Path(temp_audio_path),
                    sample['text']
                )
                assessment_time = time.time() - start_time
                
                # Check if assessment was successful
                if not azure_result or 'overall_scores' not in azure_result:
                    return {
                        'sample_idx': sample_idx,
                        'text': sample['text'],
                        'speaker': sample['speaker'],
                        'success': False,
                        'error': 'Azure assessment failed',
                        'expert_scores': {
                            'accuracy': sample['accuracy'],
                            'fluency': sample['fluency'],
                            'completeness': sample['completeness'],
                            'prosodic': sample['prosodic']
                        }
                    }
                
                # Normalize Azure scores
                azure_scores = self._normalize_azure_scores(azure_result)
                
                # Extract expert scores
                expert_scores = {
                    'accuracy': sample['accuracy'],
                    'fluency': sample['fluency'],
                    'completeness': sample['completeness'],
                    'prosodic': sample['prosodic']
                }
                
                # Calculate differences
                score_differences = {
                    key: abs(azure_scores[key] - expert_scores[key])
                    for key in expert_scores.keys()
                }
                
                result = {
                    'sample_idx': sample_idx,
                    'text': sample['text'],
                    'speaker': sample['speaker'],
                    'gender': sample['gender'],
                    'age': sample['age'],
                    'success': True,
                    'assessment_time': assessment_time,
                    'expert_scores': expert_scores,
                    'azure_scores': azure_scores,
                    'score_differences': score_differences,
                    'azure_raw_scores': azure_result.get('overall_scores', {}),
                    'word_level_scores': azure_result.get('word_level_scores', []),
                    'recognized_text': azure_result.get('recognized_text', ''),
                }
                
                print(f"✅ Sample {sample_idx + 1} evaluated successfully")
                print(f"   Expert vs Azure - Accuracy: {expert_scores['accuracy']:.1f} vs {azure_scores['accuracy']:.1f}")
                
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Error evaluating sample {sample_idx + 1}: {e}")
            return {
                'sample_idx': sample_idx,
                'text': sample.get('text', ''),
                'speaker': sample.get('speaker', ''),
                'success': False,
                'error': str(e),
                'expert_scores': {
                    'accuracy': sample.get('accuracy', 0),
                    'fluency': sample.get('fluency', 0),
                    'completeness': sample.get('completeness', 0),
                    'prosodic': sample.get('prosodic', 0)
                }
            }
    
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
        
        if not self.pronunciation_service:
            raise ValueError("Pronunciation service not initialized.")
        
        print(f"🚀 Starting evaluation on {len(self.dataset)} samples...")
        
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
            if (idx + 1) % 10 == 0:
                successful = sum(1 for r in self.evaluation_results if r['success'])
                print(f"📊 Progress: {idx + 1}/{len(samples_to_evaluate)} samples, "
                      f"{successful} successful")
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics()
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> EvaluationMetrics:
        """Calculate evaluation metrics from results."""
        successful_results = [r for r in self.evaluation_results if r['success']]
        
        if not successful_results:
            print("❌ No successful evaluations to calculate metrics from")
            return EvaluationMetrics(
                accuracy_correlation=0, fluency_correlation=0, 
                completeness_correlation=0, prosodic_correlation=0,
                accuracy_mae=0, fluency_mae=0, completeness_mae=0, prosodic_mae=0,
                accuracy_rmse=0, fluency_rmse=0, completeness_rmse=0, prosodic_rmse=0,
                total_samples=len(self.evaluation_results),
                successful_assessments=0,
                failed_assessments=len(self.evaluation_results),
                expert_score_stats={}, azure_score_stats={}
            )
        
        # Extract scores for analysis
        expert_scores = {
            'accuracy': [r['expert_scores']['accuracy'] for r in successful_results],
            'fluency': [r['expert_scores']['fluency'] for r in successful_results],
            'completeness': [r['expert_scores']['completeness'] for r in successful_results],
            'prosodic': [r['expert_scores']['prosodic'] for r in successful_results],
        }
        
        azure_scores = {
            'accuracy': [r['azure_scores']['accuracy'] for r in successful_results],
            'fluency': [r['azure_scores']['fluency'] for r in successful_results],
            'completeness': [r['azure_scores']['completeness'] for r in successful_results],
            'prosodic': [r['azure_scores']['prosodic'] for r in successful_results],
        }
        
        # Calculate correlations
        correlations = {}
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            if len(expert_scores[metric]) > 1:
                expert_vals = np.array(expert_scores[metric])
                azure_vals = np.array(azure_scores[metric])
                
                # Check if arrays are constant (no variance)
                expert_std = np.std(expert_vals)
                azure_std = np.std(azure_vals)
                
                if expert_std == 0 or azure_std == 0:
                    print(f"⚠️  Warning: Constant values in {metric} - Expert std: {expert_std:.3f}, Azure std: {azure_std:.3f}")
                    correlations[f'{metric}_correlation'] = 0.0  # No correlation for constant arrays
                else:
                    corr, _ = stats.pearsonr(expert_vals, azure_vals)
                    correlations[f'{metric}_correlation'] = float(corr) if not np.isnan(corr) else 0.0
            else:
                correlations[f'{metric}_correlation'] = 0.0
        
        # Calculate MAE and RMSE
        mae_rmse = {}
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            expert_vals = np.array(expert_scores[metric])
            azure_vals = np.array(azure_scores[metric])
            
            # Mean Absolute Error
            mae = np.mean(np.abs(expert_vals - azure_vals))
            mae_rmse[f'{metric}_mae'] = float(mae)
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((expert_vals - azure_vals) ** 2))
            mae_rmse[f'{metric}_rmse'] = float(rmse)
        
        # Calculate score statistics
        expert_stats = {}
        azure_stats = {}
        
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            expert_vals = expert_scores[metric]
            azure_vals = azure_scores[metric]
            
            expert_stats[metric] = {
                'mean': float(np.mean(expert_vals)),
                'std': float(np.std(expert_vals)),
                'min': float(np.min(expert_vals)),
                'max': float(np.max(expert_vals)),
                'median': float(np.median(expert_vals))
            }
            
            azure_stats[metric] = {
                'mean': float(np.mean(azure_vals)),
                'std': float(np.std(azure_vals)),
                'min': float(np.min(azure_vals)),
                'max': float(np.max(azure_vals)),
                'median': float(np.median(azure_vals))
            }
        
        return EvaluationMetrics(
            **correlations,
            **mae_rmse,
            total_samples=len(self.evaluation_results),
            successful_assessments=len(successful_results),
            failed_assessments=len(self.evaluation_results) - len(successful_results),
            expert_score_stats=expert_stats,
            azure_score_stats=azure_stats
        )
    
    def _save_evaluation_results(self, metrics: EvaluationMetrics):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        detailed_results = {
            'evaluation_info': {
                'timestamp': timestamp,
                'dataset': 'speechocean762',
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
                'azure': metrics.azure_score_stats
            },
            'individual_results': self.evaluation_results
        }
        
        json_path = results_dir / f"speechocean762_evaluation_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert all NumPy types to Python native types for JSON serialization
            serializable_results = convert_numpy_types(detailed_results)
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Detailed results saved: {json_path}")
        
        # Save summary as CSV for easy analysis
        successful_results = [r for r in self.evaluation_results if r['success']]
        if successful_results:
            df_data = []
            for result in successful_results:
                row = {
                    'sample_idx': result['sample_idx'],
                    'text': result['text'],
                    'speaker': result['speaker'],
                    'gender': result['gender'],
                    'age': result['age'],
                    'expert_accuracy': result['expert_scores']['accuracy'],
                    'expert_fluency': result['expert_scores']['fluency'],
                    'expert_completeness': result['expert_scores']['completeness'],
                    'expert_prosodic': result['expert_scores']['prosodic'],
                    'azure_accuracy': result['azure_scores']['accuracy'],
                    'azure_fluency': result['azure_scores']['fluency'],
                    'azure_completeness': result['azure_scores']['completeness'],
                    'azure_prosodic': result['azure_scores']['prosodic'],
                    'diff_accuracy': result['score_differences']['accuracy'],
                    'diff_fluency': result['score_differences']['fluency'],
                    'diff_completeness': result['score_differences']['completeness'],
                    'diff_prosodic': result['score_differences']['prosodic'],
                    'assessment_time': result['assessment_time'],
                    'recognized_text': result['recognized_text']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_path = results_dir / f"speechocean762_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"📊 Summary CSV saved: {csv_path}")
    
    def print_evaluation_summary(self, metrics: EvaluationMetrics):
        """Print a comprehensive evaluation summary."""
        print("\n" + "=" * 80)
        print("🎯 SPEECHOCEAN762 EVALUATION RESULTS")
        print("=" * 80)
        
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
        print(f"   Completeness: {metrics.completeness_mae:.3f}")
        print(f"   Prosodic:    {metrics.prosodic_mae:.2f}")
        
        print(f"\n📐 Root Mean Square Error (RMSE):")
        print(f"   Accuracy:    {metrics.accuracy_rmse:.2f}")
        print(f"   Fluency:     {metrics.fluency_rmse:.2f}")
        print(f"   Completeness: {metrics.completeness_rmse:.3f}")
        print(f"   Prosodic:    {metrics.prosodic_rmse:.2f}")
        
        print(f"\n📈 Score Distribution Comparison:")
        for metric in ['accuracy', 'fluency', 'completeness', 'prosodic']:
            expert_stats = metrics.expert_score_stats.get(metric, {})
            azure_stats = metrics.azure_score_stats.get(metric, {})
            
            print(f"\n   {metric.capitalize()}:")
            print(f"     Expert - Mean: {expert_stats.get('mean', 0):.2f}, "
                  f"Std: {expert_stats.get('std', 0):.2f}, "
                  f"Range: [{expert_stats.get('min', 0):.1f}, {expert_stats.get('max', 0):.1f}]")
            print(f"     Azure  - Mean: {azure_stats.get('mean', 0):.2f}, "
                  f"Std: {azure_stats.get('std', 0):.2f}, "
                  f"Range: [{azure_stats.get('min', 0):.1f}, {azure_stats.get('max', 0):.1f}]")
        
        # Interpretation
        print(f"\n🎭 Interpretation:")
        avg_correlation = np.mean([
            metrics.accuracy_correlation, metrics.fluency_correlation,
            metrics.completeness_correlation, metrics.prosodic_correlation
        ])
        
        if avg_correlation > 0.7:
            print("   ✅ Strong correlation with expert scores - Azure Speech performs well!")
        elif avg_correlation > 0.5:
            print("   🟡 Moderate correlation with expert scores - Room for improvement")
        else:
            print("   🔴 Weak correlation with expert scores - Significant differences detected")
        
        print("=" * 80)