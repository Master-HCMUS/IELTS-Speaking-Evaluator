"""
Data processing for SpeechOcean762 dataset with pronunciation assessment.

Handles loading, preprocessing, and preparing the dataset for training without
triggering unnecessary audio codec dependencies (important for Kaggle environment).
"""

import logging
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional, List

from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


class SpeechOcean762DataProcessor:
    """
    Loads and preprocesses SpeechOcean762 dataset for pronunciation assessment finetuning.
    
    CRITICAL: Uses num_proc=1 to avoid multiprocessing codec issues on Kaggle.
    CRITICAL: Extracts mel-spectrograms directly to avoid WhisperProcessor JAX import.
    """
    
    DATASET_NAME = "mispeech/speechocean762"
    TARGET_SAMPLE_RATE = 16000
    N_MELS = 80
    N_FFT = 400
    HOP_LENGTH = 160
    
    def __init__(self, processor_name: str = "openai/whisper-tiny"):
        """Initialize processor (model name kept for compatibility)."""
        self.processor_name = processor_name
        logger.info(f"Processor initialized: {processor_name}")
    
    @staticmethod
    def normalize_assessment_score(score, min_val: float = 0, max_val: float = 10) -> float:
        """
        Normalize assessment score to [0, 1] range.
        
        Converts ground truth scores from [0, 10] to [0, 1] to match model output range.
        This ensures consistent loss computation with normalized model predictions.
        
        Args:
            score: Score value (typically in [0, 10] range)
            min_val: Minimum value of the original scale (default: 0)
            max_val: Maximum value of the original scale (default: 10)
            
        Returns:
            Normalized score in [0, 1] range
            
        Example:
            >>> normalize_assessment_score(5.0)  # Returns 0.5
            >>> normalize_assessment_score(10.0) # Returns 1.0
        """
        if max_val <= min_val:
            logger.warning(f"Invalid scale: max_val ({max_val}) <= min_val ({min_val})")
            return 0.0
        
        normalized = (score - min_val) / (max_val - min_val)
        # Clip to ensure strictly [0, 1]
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _extract_mel_spectrogram(self, audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Extract mel-spectrogram directly using librosa (no transformers import needed).
        Pads/truncates to Whisper's expected 3000 time steps.
        
        Args:
            audio_array: Audio waveform
            sampling_rate: Sample rate of audio
            
        Returns:
            Mel-spectrogram [80, 3000] (Whisper format)
        """
        # Resample if necessary
        if sampling_rate != self.TARGET_SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sampling_rate,
                target_sr=self.TARGET_SAMPLE_RATE
            )
        
        # Compute mel-spectrogram (matches Whisper's feature extraction)
        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=self.TARGET_SAMPLE_RATE,
            n_mels=self.N_MELS,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to 3000 time steps (Whisper's expected length)
        if mel_spec_db.shape[1] < 3000:
            # Pad with silence (log(1e-10) ≈ -100)
            pad_width = 3000 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), constant_values=-100.0)
        else:
            # Truncate to 3000
            mel_spec_db = mel_spec_db[:, :3000]
        
        return mel_spec_db.astype(np.float32)
    
    def load_dataset(
        self,
        splits: List[str] = None,
        max_samples: Optional[int] = None
    ) -> DatasetDict:
        """
        Load SpeechOcean762 dataset from HuggingFace Hub.
        
        Args:
            splits: Dataset splits to load (e.g., ["train", "validation"])
            max_samples: Maximum samples per split (for debugging)
            
        Returns:
            DatasetDict with loaded splits
        """
        if splits is None:
            splits = ["train", "validation"]
        
        logger.info(f"Loading SpeechOcean762 from HF Hub (splits: {splits})...")
        
        try:
            datasets = load_dataset(self.DATASET_NAME, split=splits)
            
            # Handle different return types
            if isinstance(datasets, DatasetDict):
                dataset_dict = datasets
            elif isinstance(datasets, list):
                # When multiple splits are requested, return a list
                dataset_dict = DatasetDict({split: ds for split, ds in zip(splits, datasets)})
            else:
                # Single dataset returned
                dataset_dict = DatasetDict({splits[0]: datasets})
            
            logger.info(f"Loaded splits: {list(dataset_dict.keys())}")
            
            # Optionally limit samples
            if max_samples is not None:
                logger.info(f"Limiting to {max_samples} samples per split...")
                for split in dataset_dict.keys():
                    if len(dataset_dict[split]) > max_samples:
                        dataset_dict[split] = dataset_dict[split].select(range(max_samples))
                        logger.info(f"  - {split}: limited to {len(dataset_dict[split])} samples")
            
            return dataset_dict
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def prepare_for_training(
        self,
        datasets: DatasetDict,
        batch_size: int = 2,
        include_transcription: bool = True
    ) -> DatasetDict:
        """
        Preprocess datasets for training.
        
        CRITICAL: Uses num_proc=1 to prevent multiprocessing codec issues on Kaggle.
        
        Args:
            datasets: DatasetDict to process
            batch_size: Batch size for processing (not used in non-batched approach)
            include_transcription: Whether to include transcription labels
            
        Returns:
            Processed DatasetDict with audio removed and features extracted
        """
        logger.info(f"Preprocessing dataset...")
        
        def preprocess_function(example):
            """Preprocess individual example (non-batched to avoid audio decoding issues)."""
            try:
                # Get audio
                audio_dict = example["audio"]
                audio_array = np.array(audio_dict["array"], dtype=np.float32)
                sample_rate = audio_dict["sampling_rate"]
                
                # Extract mel-spectrogram directly (no transformers import)
                input_features = self._extract_mel_spectrogram(audio_array, sample_rate)
                
                # Create result with features
                result = {
                    "input_features": input_features,
                }
                
                # Copy all fields and normalize assessment scores
                assessment_score_fields = {
                    # Utterance-level scores (scalar per utterance)
                    "utterance_accuracy", "utterance_fluency", "utterance_prosodic",
                    "utterance_completeness", "utterance_total",
                    # Word-level scores (lists of scores per word)
                    "word_accuracy_scores", "word_stress_scores", "word_total_scores",
                    # Phone-level scores (lists of scores per phone)
                    "phone_accuracy_scores"
                }
                
                for key, value in example.items():
                    if key in ["audio", "words", "alignment"]:
                        # Skip these fields
                        continue
                    elif key in assessment_score_fields:
                        # Normalize assessment scores from [0, 10] to [0, 1]
                        if isinstance(value, (list, tuple)):
                            # For word/phone-level scores (lists)
                            normalized_value = [
                                self.normalize_assessment_score(score)
                                for score in value
                            ]
                            result[key] = normalized_value
                        elif isinstance(value, (int, float)):
                            # For utterance-level scores (scalar)
                            result[key] = self.normalize_assessment_score(value)
                        else:
                            result[key] = value
                    else:
                        # Keep other fields as-is (text, IDs, etc.)
                        result[key] = value
                
                return result
            
            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                print(f"Error: {e}")
                # Return fallback with zero features
                result = {
                    "input_features": np.zeros((80, 3000), dtype=np.float32),
                }
                for key, value in example.items():
                    if key not in ["audio", "words", "alignment"]:
                        try:
                            result[key] = value
                        except:
                            pass
                return result
        
        # Process each split - CRITICAL: Remove audio column BEFORE any formatting
        processed_datasets = {}
        
        for split_name, split_data in datasets.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
            
            try:
                from datasets.features import Audio
                
                # Get original schema
                original_schema = split_data.features
                
                # Check if audio column exists and is Audio type
                if "audio" in original_schema and isinstance(original_schema["audio"], Audio):
                    # CRITICAL: We need to process the raw audio before .map() tries to format it
                    # Use a custom function that extracts features from raw audio bytes
                    
                    def extract_features_from_raw(example):
                        """Extract audio features from raw data without triggering decoder."""
                        try:
                            # Access raw audio data directly from Arrow table
                            # This bypasses the Audio feature decoder
                            audio_dict = example["audio"]
                            
                            # If it's already decoded to dict, use it directly
                            if isinstance(audio_dict, dict) and "array" in audio_dict:
                                audio_array = np.array(audio_dict["array"], dtype=np.float32)
                                sample_rate = audio_dict.get("sampling_rate", self.TARGET_SAMPLE_RATE)
                            else:
                                # Fallback: create silence if we can't decode
                                logger.warning(f"Could not decode audio in {split_name}")
                                audio_array = np.zeros(16000, dtype=np.float32)
                                sample_rate = self.TARGET_SAMPLE_RATE
                            
                            # Extract mel-spectrogram
                            input_features = self._extract_mel_spectrogram(audio_array, sample_rate)
                            
                            # Prepare result
                            result = {"input_features": input_features}
                            
                            # Copy and normalize all other fields
                            assessment_score_fields = {
                                "utterance_accuracy", "utterance_fluency", "utterance_prosodic",
                                "utterance_completeness", "utterance_total",
                                "word_accuracy_scores", "word_stress_scores", "word_total_scores",
                                "phone_accuracy_scores"
                            }
                            
                            for key, value in example.items():
                                if key in ["audio", "words", "alignment"]:
                                    continue
                                elif key in assessment_score_fields:
                                    if isinstance(value, (list, tuple)):
                                        result[key] = [
                                            self.normalize_assessment_score(score)
                                            for score in value
                                        ]
                                    elif isinstance(value, (int, float)):
                                        result[key] = self.normalize_assessment_score(value)
                                    else:
                                        result[key] = value
                                else:
                                    result[key] = value
                            
                            return result
                        
                        except Exception as e:
                            logger.warning(f"Error processing example: {e}")
                            result = {"input_features": np.zeros((80, 3000), dtype=np.float32)}
                            for key, value in example.items():
                                if key not in ["audio", "words", "alignment"]:
                                    try:
                                        result[key] = value
                                    except:
                                        pass
                            return result
                    
                    # Process WITHOUT formatting first - this avoids the decoder
                    processed = split_data.map(
                        extract_features_from_raw,
                        batched=False,
                        num_proc=1,  # Use single process to avoid multiprocessing issues
                        remove_columns=["audio"],
                        desc=f"Processing {split_name}"
                    )
                else:
                    # No Audio feature, use original preprocess function
                    processed = split_data.map(
                        preprocess_function,
                        batched=False,
                        num_proc=None,
                        remove_columns=["audio"] if "audio" in split_data.column_names else [],
                        desc=f"Processing {split_name}"
                    )
                
                # Remove unnecessary columns
                columns_to_remove = [col for col in processed.column_names 
                                    if col in ["words", "alignment"]]
                if columns_to_remove:
                    processed = processed.remove_columns(columns_to_remove)
                
                processed_datasets[split_name] = processed
                logger.info(f"  ✓ {split_name} processed: {processed}")
            
            except Exception as e:
                logger.error(f"Error processing {split_name}: {e}")
                raise
        
        return DatasetDict(processed_datasets)

