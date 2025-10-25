"""
Data processing utilities for SpeechOcean762 dataset with pronunciation assessment.

This module handles loading, preprocessing, and preparing the SpeechOcean762
dataset for Whisper model fine-tuning with pronunciation assessment capabilities
at word-level, phone-level, and utterance-level granularities.
"""

import os
import json
import numpy as np
import librosa
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import logging

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import load_dataset, DatasetDict, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PronunciationAssessmentDataCollator:
    """Data collator for speech-to-text with pronunciation assessment training."""
    
    def __init__(self, processor: WhisperProcessor, include_transcription: bool = True):
        """
        Initialize the data collator.
        
        Args:
            processor: Whisper processor for feature extraction and tokenization
            include_transcription: Whether to include transcription labels
        """
        self.processor = processor
        self.include_transcription = include_transcription
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for training.
        
        Args:
            features: List of sample dictionaries
            
        Returns:
            Collated batch dictionary
        """
        # Extract input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Process transcription labels if included
        if self.include_transcription and "labels" in features[0]:
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                return_tensors="pt"
            )
            
            # Replace padding token id's of the labels by -100 (ignored by loss)
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            
            # Remove bos token if appended
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
        
        # Add pronunciation assessment scores (word-level)
        if "word_accuracy_scores" in features[0]:
            # Pad word-level scores
            max_word_len = max(len(f.get("word_accuracy_scores", [])) for f in features)
            
            word_accuracy = torch.zeros(len(features), max_word_len)
            word_stress = torch.zeros(len(features), max_word_len)
            word_total = torch.zeros(len(features), max_word_len)
            word_mask = torch.zeros(len(features), max_word_len)
            
            for i, feature in enumerate(features):
                scores = feature.get("word_accuracy_scores", [])
                word_len = len(scores)
                
                if word_len > 0:
                    word_accuracy[i, :word_len] = torch.tensor(feature.get("word_accuracy_scores", []), dtype=torch.float32)
                    word_stress[i, :word_len] = torch.tensor(feature.get("word_stress_scores", []), dtype=torch.float32)
                    word_total[i, :word_len] = torch.tensor(feature.get("word_total_scores", []), dtype=torch.float32)
                    word_mask[i, :word_len] = 1.0
            
            batch["word_accuracy_scores"] = word_accuracy
            batch["word_stress_scores"] = word_stress
            batch["word_total_scores"] = word_total
            batch["word_mask"] = word_mask
        
        # Add pronunciation assessment scores (phone-level)
        if "phone_accuracy_scores" in features[0]:
            max_phone_len = max(len(f.get("phone_accuracy_scores", [])) for f in features)
            
            phone_accuracy = torch.zeros(len(features), max_phone_len)
            phone_mask = torch.zeros(len(features), max_phone_len)
            
            for i, feature in enumerate(features):
                scores = feature.get("phone_accuracy_scores", [])
                phone_len = len(scores)
                
                if phone_len > 0:
                    phone_accuracy[i, :phone_len] = torch.tensor(scores, dtype=torch.float32)
                    phone_mask[i, :phone_len] = 1.0
            
            batch["phone_accuracy_scores"] = phone_accuracy
            batch["phone_mask"] = phone_mask
        
        # Add pronunciation assessment scores (utterance-level)
        utterance_score_keys = ["accuracy", "fluency", "prosodic", "completeness", "total"]
        for score_key in utterance_score_keys:
            if score_key in features[0]:
                batch[score_key] = torch.tensor(
                    [f.get(score_key, 0) for f in features],
                    dtype=torch.float32
                )
        
        return batch


class SpeechOcean762PronunciationProcessor:
    """
    Data processor for SpeechOcean762 dataset with pronunciation assessment.
    
    Handles loading, preprocessing, and formatting the dataset for Whisper
    fine-tuning with multi-level pronunciation assessment (word, phone, utterance).
    """
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        sampling_rate: int = 16000,
        max_audio_length: float = 30.0,
        normalize_audio: bool = True
    ):
        """
        Initialize the pronunciation assessment data processor.
        
        Args:
            whisper_model_name: Name of the Whisper model
            sampling_rate: Target sampling rate for audio (Hz)
            max_audio_length: Maximum audio length in seconds
            normalize_audio: Whether to normalize audio amplitude
        """
        self.whisper_model_name = whisper_model_name
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.normalize_audio = normalize_audio
        
        # Initialize Whisper processors
        logger.info(f"Initializing processors for {whisper_model_name}")
        self.processor = WhisperProcessor.from_pretrained(whisper_model_name)
        
        # Cache for processed data
        self.dataset_cache = {}
        self.dataset_statistics = {}
        
        logger.info(f"Data processor initialized with:")
        logger.info(f"  - Sampling rate: {sampling_rate} Hz")
        logger.info(f"  - Max audio length: {max_audio_length} seconds")
        logger.info(f"  - Audio normalization: {normalize_audio}")
    
    def load_dataset(
        self,
        splits: List[str] = ["train", "test"],
        max_samples_per_split: Optional[Dict[str, int]] = None
    ) -> DatasetDict:
        """
        Load SpeechOcean762 dataset with specified splits.
        
        Args:
            splits: List of dataset splits to load (e.g., ["train", "test"])
            max_samples_per_split: Maximum samples per split for testing
            
        Returns:
            DatasetDict with loaded splits
        """
        logger.info("Loading SpeechOcean762 dataset...")
        
        dataset_dict = {}
        
        for split in splits:
            logger.info(f"Loading {split} split...")
            
            try:
                # Load the split from Hugging Face Hub
                dataset = load_dataset("mispeech/speechocean762", split=split)
                
                # Limit samples if specified
                if max_samples_per_split and split in max_samples_per_split:
                    max_samples = max_samples_per_split[split]
                    if max_samples is not None and len(dataset) > max_samples:
                        dataset = dataset.select(range(max_samples))
                        logger.info(f"Limited {split} split to {max_samples} samples")
                
                dataset_dict[split] = dataset
                logger.info(f"Loaded {split} split: {len(dataset)} samples")
                
            except Exception as e:
                logger.error(f"Failed to load {split} split: {e}")
                raise
        
        # Create DatasetDict
        datasets = DatasetDict(dataset_dict)
        
        # Log dataset information
        self._log_dataset_info(datasets)
        
        return datasets
    
    def _log_dataset_info(self, datasets: DatasetDict):
        """Log information about loaded datasets."""
        logger.info("Dataset Information:")
        
        for split_name, split_data in datasets.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
            
            # Log dataset features without accessing samples (avoids audio decoding)
            if hasattr(split_data, 'features'):
                logger.info(f"    Features: {list(split_data.features.keys())}")
    
    def preprocess_audio(self, audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Preprocess audio data for Whisper model.
        
        Args:
            audio_array: Input audio array
            sampling_rate: Original sampling rate
            
        Returns:
            Preprocessed audio array
        """
        # Convert to float32 if needed
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        
        # Resample if necessary
        if sampling_rate != self.sampling_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sampling_rate,
                target_sr=self.sampling_rate
            )
        
        # Normalize audio amplitude
        if self.normalize_audio:
            # Use norm to avoid unnecessary copies
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val
        
        # Trim to max length (don't pad, let Whisper handle it)
        max_length_samples = int(self.max_audio_length * self.sampling_rate)
        if len(audio_array) > max_length_samples:
            audio_array = audio_array[:max_length_samples]
        
        return audio_array
    
    def _extract_pronunciation_scores(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract pronunciation scores from the dataset sample.
        
        Args:
            example: Single sample from the dataset
            
        Returns:
            Dictionary with extracted scores
        """
        scores = {}
        
        # Extract utterance-level scores
        utterance_score_keys = ["accuracy", "completeness", "fluency", "prosodic", "total"]
        for key in utterance_score_keys:
            if key in example:
                # Normalize scores to 0-1 range (assuming 0-10 scale)
                score_value = float(example[key])
                scores[key] = score_value / 10.0 if score_value <= 10 else score_value
        
        # Extract word-level scores
        words = example.get("words", [])
        word_accuracy_scores = []
        word_stress_scores = []
        word_total_scores = []
        
        for word_info in words:
            word_accuracy = word_info.get("accuracy", 0)
            word_stress = word_info.get("stress", 0)
            word_total = word_info.get("total", 0)
            
            # Normalize to 0-1 range
            word_accuracy_scores.append(float(word_accuracy) / 10.0)
            word_stress_scores.append(float(word_stress) / 10.0)
            word_total_scores.append(float(word_total) / 10.0)
        
        if word_accuracy_scores:
            scores["word_accuracy_scores"] = word_accuracy_scores
            scores["word_stress_scores"] = word_stress_scores
            scores["word_total_scores"] = word_total_scores
        
        # Extract phone-level scores
        phone_accuracy_scores = []
        for word_info in words:
            phones = word_info.get("phones", [])
            phones_accuracy = word_info.get("phones-accuracy", [])
            
            for phone_acc in phones_accuracy:
                # Normalize phone accuracy
                phone_accuracy_scores.append(float(phone_acc) / 2.0)  # Assuming 0-2 scale
        
        if phone_accuracy_scores:
            scores["phone_accuracy_scores"] = phone_accuracy_scores
        
        # Store metadata
        scores["speaker"] = example.get("speaker", "unknown")
        scores["gender"] = example.get("gender", "unknown")
        scores["age"] = example.get("age", 0)
        scores["text"] = example.get("text", "")
        scores["id"] = example.get("id", "")
        
        return scores
    
    def prepare_dataset_for_training(
        self,
        datasets: DatasetDict,
        include_transcription: bool = True
    ) -> DatasetDict:
        """
        Prepare datasets for pronunciation assessment training.
        
        Processes audio, extracts features, and prepares pronunciation scores
        at multiple granularities (word, phone, utterance).
        
        Args:
            datasets: Raw datasets to process
            include_transcription: Whether to include transcription for ASR training
            
        Returns:
            Processed datasets ready for training
        """
        logger.info("Preparing datasets for training...")
        
        def preprocess_function(examples):
            """Preprocess a batch of examples."""
            # Process audio - handle both dict and direct array formats
            audio_arrays = []
            for audio in examples["audio"]:
                # Handle both formats: {"array": ..., "sampling_rate": ...} and direct arrays
                if isinstance(audio, dict):
                    audio_array = audio.get("array", audio)
                    audio_sr = audio.get("sampling_rate", self.sampling_rate)
                else:
                    # If it's not a dict, try to convert it to numpy
                    try:
                        audio_array = np.asarray(audio)
                    except:
                        audio_array = audio
                    audio_sr = self.sampling_rate
                
                # Ensure audio_array is a numpy array
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.asarray(audio_array)
                
                processed_audio = self.preprocess_audio(
                    audio_array,
                    audio_sr
                )
                audio_arrays.append(processed_audio)
            
            # Extract features using Whisper feature extractor
            # Use float32 to reduce memory usage
            inputs = self.processor.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="np"
            )
            
            # Convert to float32 and ensure C-contiguous for efficiency
            input_features = inputs.input_features.astype(np.float32)
            if not input_features.flags['C_CONTIGUOUS']:
                input_features = np.ascontiguousarray(input_features)
            
            # Prepare batch dictionary
            batch = {
                "input_features": input_features,
            }
            
            # Process transcription if requested
            if include_transcription and "text" in examples:
                transcriptions = examples["text"]
                labels = self.processor.tokenizer(
                    transcriptions,
                    truncation=True,
                    return_tensors="np",
                    padding=False  # We'll pad in the data collator
                )
                batch["labels"] = labels.input_ids
            
            # Extract and process pronunciation scores
            for idx in range(len(examples["id"])):
                # Build example dictionary for this sample
                sample_dict = {}
                for key in examples.keys():
                    if key != "audio":  # audio is handled separately
                        if isinstance(examples[key], list):
                            sample_dict[key] = examples[key][idx]
                        else:
                            sample_dict[key] = examples[key][idx]
                
                # Extract pronunciation scores
                scores = self._extract_pronunciation_scores(sample_dict)
                
                # Add scores to batch on first iteration
                if idx == 0:
                    for score_key, score_value in scores.items():
                        if score_key not in batch:
                            batch[score_key] = []
                
                # Append scores for this sample
                for score_key, score_value in scores.items():
                    if score_key in batch:
                        batch[score_key].append(score_value)
            
            # Convert lists to numpy arrays where appropriate
            for key in batch.keys():
                if isinstance(batch[key], list) and len(batch[key]) > 0:
                    first_elem = batch[key][0]
                    if isinstance(first_elem, list):
                        # Keep as list (will be handled by DataCollator)
                        pass
                    else:
                        batch[key] = np.array(batch[key])
            
            return batch
        
        # Apply preprocessing with batching
        # Note: Keep audio column during map, will be removed after
        # Use smaller batch size to avoid memory issues (std::bad_alloc)
        processed_datasets = datasets.map(
            preprocess_function,
            batched=True,
            batch_size=2,  # Reduced from 8 to prevent memory overflow
            num_proc=1,  # Set to 1 to avoid issues with audio processing
            desc="Preprocessing datasets"
        )
        
        # Remove audio column after processing
        if "audio" in processed_datasets["train"].column_names:
            processed_datasets = processed_datasets.remove_columns(["audio"])
        
        # Update dataset statistics
        self.dataset_statistics = self._compute_dataset_statistics(processed_datasets)
        logger.info("Dataset statistics computed")
        
        return processed_datasets
    
    def _compute_dataset_statistics(self, datasets: DatasetDict) -> Dict[str, Any]:
        """
        Compute statistics about the processed datasets.
        
        Args:
            datasets: Processed datasets
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        for split_name, split_data in datasets.items():
            split_stats = {
                "num_samples": len(split_data),
                "num_features": 0,
            }
            
            # Compute score statistics
            score_keys = ["accuracy", "fluency", "prosodic", "completeness", "total"]
            
            for score_key in score_keys:
                if score_key in split_data.column_names:
                    scores = np.array(split_data[score_key])
                    split_stats[f"{score_key}_mean"] = float(np.mean(scores))
                    split_stats[f"{score_key}_std"] = float(np.std(scores))
                    split_stats[f"{score_key}_min"] = float(np.min(scores))
                    split_stats[f"{score_key}_max"] = float(np.max(scores))
            
            stats[split_name] = split_stats
        
        return stats
    
    def get_dataset_statistics(self, datasets: DatasetDict) -> Dict[str, Any]:
        """
        Get statistics about the datasets.
        
        Args:
            datasets: Processed datasets
            
        Returns:
            Dictionary with dataset statistics
        """
        return self.dataset_statistics
    
    def create_data_collator(self, include_transcription: bool = True) -> 'PronunciationAssessmentDataCollator':
        """
        Create a data collator for batch processing.
        
        Args:
            include_transcription: Whether to include transcription labels
            
        Returns:
            Data collator instance
        """
        return PronunciationAssessmentDataCollator(
            processor=self.processor,
            include_transcription=include_transcription
        )
    
    def validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a sample to ensure it has required fields.
        
        Args:
            sample: Sample to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["id", "audio", "text", "accuracy", "completeness", "fluency", "prosodic", "total"]
        
        for field in required_fields:
            if field not in sample:
                return False, f"Missing required field: {field}"
        
        # Validate audio
        if not isinstance(sample["audio"], dict) or "array" not in sample["audio"]:
            return False, "Invalid audio format"
        
        # Validate scores are numeric
        score_fields = ["accuracy", "completeness", "fluency", "prosodic", "total"]
        for field in score_fields:
            try:
                float(sample[field])
            except (ValueError, TypeError):
                return False, f"Non-numeric score for {field}"
        
        # Validate word information
        if "words" in sample and isinstance(sample["words"], list):
            for word_idx, word in enumerate(sample["words"]):
                if not isinstance(word, dict):
                    return False, f"Word {word_idx} is not a dictionary"
                
                required_word_fields = ["text", "accuracy", "stress", "total"]
                for field in required_word_fields:
                    if field not in word:
                        return False, f"Word {word_idx} missing field: {field}"
        
        return True, "Valid"
    
    def save_processed_dataset(self, datasets: DatasetDict, output_path: str):
        """
        Save processed datasets to disk.
        
        Args:
            datasets: Datasets to save
            output_path: Path to save datasets
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed datasets to {output_path}")
        
        for split_name, split_data in datasets.items():
            split_path = output_path / split_name
            split_data.save_to_disk(str(split_path))
            logger.info(f"Saved {split_name} split to {split_path}")
        
        # Save statistics
        stats_path = output_path / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.dataset_statistics, f, indent=2, default=str)
        logger.info(f"Saved dataset statistics to {stats_path}")
    
    @staticmethod
    def load_processed_dataset(dataset_path: str) -> DatasetDict:
        """
        Load processed datasets from disk.
        
        Args:
            dataset_path: Path to processed datasets
            
        Returns:
            Loaded datasets
        """
        logger.info(f"Loading processed datasets from {dataset_path}")
        
        dataset_path = Path(dataset_path)
        dataset_dict = {}
        
        for split_dir in dataset_path.iterdir():
            if split_dir.is_dir() and not split_dir.name.startswith('.'):
                split_name = split_dir.name
                dataset = Dataset.load_from_disk(str(split_dir))
                dataset_dict[split_name] = dataset
                logger.info(f"Loaded {split_name} split: {len(dataset)} samples")
        
        return DatasetDict(dataset_dict)
