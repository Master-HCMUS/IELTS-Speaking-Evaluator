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
    
    def _extract_mel_spectrogram(self, audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Extract mel-spectrogram directly using librosa (no transformers import needed).
        
        Args:
            audio_array: Audio waveform
            sampling_rate: Sample rate of audio
            
        Returns:
            Mel-spectrogram [n_mels, time_steps]
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
            batch_size: Batch size for processing
            include_transcription: Whether to include transcription labels
            
        Returns:
            Processed DatasetDict with audio removed and features extracted
        """
        logger.info(f"Preprocessing dataset (batch_size={batch_size})...")
        
        def preprocess_function(examples):
            """Preprocess individual examples."""
            batch_size_local = len(examples["audio"])
            
            # Process audio directly (no HF decoders)
            audio_features_list = []
            
            for i in range(batch_size_local):
                try:
                    # Get audio
                    audio_dict = examples["audio"][i]
                    audio_array = np.array(audio_dict["array"], dtype=np.float32)
                    sample_rate = audio_dict["sampling_rate"]
                    
                    # Extract mel-spectrogram directly (no transformers import)
                    input_features = self._extract_mel_spectrogram(audio_array, sample_rate)
                    
                    audio_features_list.append(input_features)
                
                except Exception as e:
                    logger.warning(f"Error processing audio {i}: {e}")
                    # Use zero features as fallback
                    audio_features_list.append(np.zeros((80, 3000), dtype=np.float32))
            
            # Combine batch
            result = {
                "input_features": audio_features_list,
            }
            
            # Copy assessment scores
            for key in examples.keys():
                if key not in ["audio", "words", "alignment"]:
                    result[key] = examples[key]
            
            return result
        
        # Process each split - CRITICAL: num_proc=1 for Kaggle
        processed_datasets = {}
        
        for split_name, split_data in datasets.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
            
            try:
                processed = split_data.map(
                    preprocess_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1,  # CRITICAL for Kaggle - avoid multiprocessing codec issues
                    remove_columns=["audio"],  # Remove after processing
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
