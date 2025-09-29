"""
Data processing utilities for SpeechOcean762 dataset fine-tuning.

This module handles loading, preprocessing, and preparing the SpeechOcean762
dataset for Whisper model fine-tuning.
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from datasets import load_dataset, DatasetDict
from typing import Dict, Any, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text training."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Split features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )
        
        # Replace padding token id's of the labels by -100 
        # so it's ignored by the loss function
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


class SpeechOcean762DataProcessor:
    """
    Data processor for SpeechOcean762 dataset.
    
    Handles loading, preprocessing, and formatting the dataset for
    Whisper model fine-tuning with pronunciation assessment capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        sampling_rate: int = 16000,
        max_audio_length: float = 30.0,
        normalize_audio: bool = True
    ):
        """
        Initialize the data processor.
        
        Args:
            model_name: Name of the Whisper model for processor compatibility
            sampling_rate: Target sampling rate for audio
            max_audio_length: Maximum audio length in seconds
            normalize_audio: Whether to normalize audio amplitude
        """
        self.model_name = model_name
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.normalize_audio = normalize_audio
        
        # Initialize Whisper processors
        logger.info(f"Initializing processors for {model_name}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        
        # Cache for processed data
        self.dataset_cache = {}
        
    def load_dataset(
        self,
        splits: List[str] = ["train", "test"],
        max_samples_per_split: Optional[Dict[str, int]] = None
    ) -> DatasetDict:
        """
        Load SpeechOcean762 dataset with specified splits.
        
        Args:
            splits: List of dataset splits to load
            max_samples_per_split: Maximum samples per split for testing
            
        Returns:
            DatasetDict with loaded splits
        """
        logger.info("Loading SpeechOcean762 dataset...")
        
        dataset_dict = {}
        
        for split in splits:
            logger.info(f"Loading {split} split...")
            
            # Load the split
            dataset = load_dataset("mispeech/speechocean762", split=split)
            
            # Limit samples if specified
            if max_samples_per_split and split in max_samples_per_split:
                max_samples = max_samples_per_split[split]
                if max_samples is not None and len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    logger.info(f"Limited {split} split to {max_samples} samples")
            
            dataset_dict[split] = dataset
            logger.info(f"Loaded {split} split: {len(dataset)} samples")
        
        # Create DatasetDict
        datasets = DatasetDict(dataset_dict)
        
        # Show dataset info
        self._log_dataset_info(datasets)
        
        return datasets
    
    def _log_dataset_info(self, datasets: DatasetDict):
        """Log information about the loaded dataset."""
        logger.info("Dataset Information:")
        
        for split_name, split_data in datasets.items():
            logger.info(f"  {split_name}: {len(split_data)} samples")
            
            # Sample statistics
            if len(split_data) > 0:
                sample = split_data[0]
                logger.info(f"    Audio sampling rate: {sample['audio']['sampling_rate']}")
                logger.info(f"    Sample text: '{sample['text']}'")
                logger.info(f"    Expert scores - Accuracy: {sample['accuracy']}, Fluency: {sample['fluency']}")
    
    def preprocess_audio(self, audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Preprocess audio data for Whisper model.
        
        Args:
            audio_array: Input audio array
            sampling_rate: Original sampling rate
            
        Returns:
            Preprocessed audio array
        """
        # Resample if necessary
        if sampling_rate != self.sampling_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sampling_rate, 
                target_sr=self.sampling_rate
            )
        
        # Normalize audio amplitude
        if self.normalize_audio:
            audio_array = librosa.util.normalize(audio_array)
        
        # Trim or pad to max length
        max_length_samples = int(self.max_audio_length * self.sampling_rate)
        if len(audio_array) > max_length_samples:
            audio_array = audio_array[:max_length_samples]
        elif len(audio_array) < max_length_samples:
            # Pad with zeros
            pad_length = max_length_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, pad_length), mode='constant')
        
        return audio_array
    
    def prepare_dataset_for_training(
        self,
        datasets: DatasetDict,
        add_pronunciation_scores: bool = True
    ) -> DatasetDict:
        """
        Prepare dataset for Whisper fine-tuning.
        
        Args:
            datasets: Raw datasets to process
            add_pronunciation_scores: Whether to include pronunciation scores
            
        Returns:
            Processed datasets ready for training
        """
        logger.info("Preparing datasets for training...")
        
        def preprocess_function(examples):
            """Preprocess a batch of examples."""
            # Process audio
            audio_arrays = []
            for audio in examples["audio"]:
                processed_audio = self.preprocess_audio(
                    audio["array"], 
                    audio["sampling_rate"]
                )
                audio_arrays.append(processed_audio)
            
            # Extract features using Whisper feature extractor
            inputs = self.feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="np"
            )
            
            # Tokenize transcriptions
            transcriptions = examples["text"]
            labels = self.tokenizer(
                transcriptions,
                truncation=True,
                padding=True,
                return_tensors="np"
            )
            
            # Prepare the batch
            batch = {
                "input_features": inputs.input_features,
                "labels": labels.input_ids,
                "transcription": transcriptions,
            }
            
            # Add pronunciation scores if requested
            if add_pronunciation_scores:
                batch.update({
                    "accuracy_score": examples["accuracy"],
                    "fluency_score": examples["fluency"], 
                    "completeness_score": examples["completeness"],
                    "prosodic_score": examples["prosodic"],
                    "speaker": examples["speaker"],
                    "gender": examples["gender"],
                    "age": examples["age"]
                })
            
            return batch
        
        # Process each split
        processed_datasets = {}
        for split_name, dataset in datasets.items():
            logger.info(f"Processing {split_name} split...")
            
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=dataset.column_names,
                desc=f"Preprocessing {split_name}"
            )
            
            processed_datasets[split_name] = processed_dataset
            logger.info(f"Processed {split_name}: {len(processed_dataset)} samples")
        
        return DatasetDict(processed_datasets)
    
    def create_data_collator(self):
        """Create data collator for training."""
        from transformers import WhisperProcessor
        
        processor = WhisperProcessor.from_pretrained(self.model_name)
        return DataCollatorSpeechSeq2SeqWithPadding(processor)
    
    def get_dataset_statistics(self, datasets: DatasetDict) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            datasets: Processed datasets
            
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for split_name, dataset in datasets.items():
            split_stats = {
                "num_samples": len(dataset),
                "features": list(dataset.features.keys()) if len(dataset) > 0 else []
            }
            
            # Audio statistics
            if len(dataset) > 0 and "input_features" in dataset.features:
                # Sample a few examples to get audio statistics
                sample_size = min(100, len(dataset))
                samples = dataset.select(range(sample_size))
                
                audio_lengths = []
                for sample in samples:
                    # Assuming input_features represent audio features
                    feature_length = len(sample["input_features"])
                    audio_lengths.append(feature_length)
                
                split_stats["audio_features"] = {
                    "mean_feature_length": np.mean(audio_lengths),
                    "max_feature_length": np.max(audio_lengths),
                    "min_feature_length": np.min(audio_lengths)
                }
            
            # Text statistics
            if len(dataset) > 0 and "transcription" in dataset.features:
                sample_size = min(100, len(dataset))
                samples = dataset.select(range(sample_size))
                
                text_lengths = [len(sample["transcription"].split()) for sample in samples]
                split_stats["text_stats"] = {
                    "mean_words": np.mean(text_lengths),
                    "max_words": np.max(text_lengths),
                    "min_words": np.min(text_lengths)
                }
            
            # Pronunciation score statistics
            if len(dataset) > 0 and "accuracy_score" in dataset.features:
                sample_size = min(1000, len(dataset))
                samples = dataset.select(range(sample_size))
                
                accuracy_scores = [sample["accuracy_score"] for sample in samples]
                fluency_scores = [sample["fluency_score"] for sample in samples]
                
                split_stats["pronunciation_stats"] = {
                    "accuracy": {
                        "mean": np.mean(accuracy_scores),
                        "std": np.std(accuracy_scores),
                        "min": np.min(accuracy_scores),
                        "max": np.max(accuracy_scores)
                    },
                    "fluency": {
                        "mean": np.mean(fluency_scores),
                        "std": np.std(fluency_scores),
                        "min": np.min(fluency_scores),
                        "max": np.max(fluency_scores)
                    }
                }
            
            stats[split_name] = split_stats
        
        return stats


class SpeechOcean762Dataset(Dataset):
    """
    PyTorch Dataset wrapper for SpeechOcean762 data.
    
    Provides direct PyTorch Dataset interface for custom training loops.
    """
    
    def __init__(
        self,
        dataset,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        max_audio_length: float = 30.0
    ):
        """
        Initialize dataset wrapper.
        
        Args:
            dataset: HuggingFace dataset
            feature_extractor: Whisper feature extractor
            tokenizer: Whisper tokenizer
            max_audio_length: Maximum audio length in seconds
        """
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.dataset[idx]
        
        # Process audio
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        
        # Resample if necessary
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        
        # Extract features
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Tokenize text
        labels = self.tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.input_ids.squeeze(0),
            "pronunciation_scores": torch.tensor([
                item["accuracy"],
                item["fluency"], 
                item["completeness"],
                item["prosodic"]
            ], dtype=torch.float32)
        }