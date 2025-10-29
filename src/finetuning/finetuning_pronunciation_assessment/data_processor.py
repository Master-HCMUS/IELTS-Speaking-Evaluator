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

# Try to import phoneme tokenizer (optional, for phoneme symbol extraction)
try:
    from phoneme_tokenizer import PhonemeTokenizer
    PHONEME_TOKENIZER_AVAILABLE = True
except ImportError:
    PHONEME_TOKENIZER_AVAILABLE = False
    logger.info("PhonemeTokenizer not available - phoneme symbol extraction disabled")


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
    
    def __init__(self, processor_name: str = "openai/whisper-tiny", normalize_scores: bool = True):
        """
        Initialize processor.
        
        Args:
            processor_name: Model name for compatibility
            normalize_scores: Whether to normalize assessment scores to [0, 1]
        """
        self.processor_name = processor_name
        self.normalize_scores = normalize_scores
        logger.info(f"Processor initialized: {processor_name} (normalize_scores={normalize_scores})")
    
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
    
    @staticmethod
    def normalize_assessment_score(score: float, min_val: float = 0, max_val: float = 10) -> float:
        """
        Normalize score to [0, 1] range for consistent model training.
        
        SpeechOcean762 scores are typically in range [0, 10], this normalizes to [0, 1].
        
        Args:
            score: Score to normalize
            min_val: Minimum value in original scale (default: 0)
            max_val: Maximum value in original scale (default: 10)
            
        Returns:
            Normalized score in [0, 1]
            
        Raises:
            ValueError: If max_val <= min_val or score is invalid
        """
        if max_val <= min_val:
            raise ValueError(f"max_val ({max_val}) must be > min_val ({min_val})")
        
        if score is None or (isinstance(score, float) and np.isnan(score)):
            logger.warning(f"Invalid score encountered: {score}, returning 0.5 (middle value)")
            return 0.5
        
        # Clamp score to valid range before normalizing
        clamped_score = np.clip(score, min_val, max_val)
        normalized = (clamped_score - min_val) / (max_val - min_val)
        
        return float(normalized)
    
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
                
                # ────────────────────────────────────────────────────────────────
                # EXTRACT FROM "words" COLUMN (detailed word/phone level scores)
                # ────────────────────────────────────────────────────────────────
                if "words" in example and example["words"] is not None:
                    try:
                        words_data = example["words"]
                        
                        # Extract word-level scores
                        word_accuracy_scores = []
                        word_stress_scores = []
                        word_total_scores = []
                        phone_accuracy_scores = []
                        
                        for word_entry in words_data:
                            # Word-level scores
                            if "accuracy" in word_entry:
                                word_accuracy_scores.append(float(word_entry["accuracy"]))
                            if "stress" in word_entry:
                                word_stress_scores.append(float(word_entry["stress"]))
                            if "total" in word_entry:
                                word_total_scores.append(float(word_entry["total"]))
                            
                            # Phone-level scores (flatten all phones)
                            if "phones-accuracy" in word_entry:
                                phones_acc = word_entry["phones-accuracy"]
                                if isinstance(phones_acc, (list, tuple)):
                                    phone_accuracy_scores.extend([float(p) for p in phones_acc])
                        
                        # Store extracted scores (will be normalized below)
                        if word_accuracy_scores:
                            result["word_accuracy_scores"] = np.array(word_accuracy_scores, dtype=np.float32)
                        if word_stress_scores:
                            result["word_stress_scores"] = np.array(word_stress_scores, dtype=np.float32)
                        if word_total_scores:
                            result["word_total_scores"] = np.array(word_total_scores, dtype=np.float32)
                        if phone_accuracy_scores:
                            result["phone_accuracy_scores"] = np.array(phone_accuracy_scores, dtype=np.float32)
                        
                        logger.debug(
                            f"Extracted from 'words': {len(word_accuracy_scores)} words, "
                            f"{len(phone_accuracy_scores)} phones"
                        )
                    
                    except Exception as e:
                        logger.warning(f"Error extracting from 'words' column: {e}")
                        # Continue without word-level scores if extraction fails
                
                # ────────────────────────────────────────────────────────────────
                # EXTRACT PHONEME SYMBOLS (NEW: for CTC phoneme decoder)
                # ────────────────────────────────────────────────────────────────
                if PHONEME_TOKENIZER_AVAILABLE and "words" in example and example["words"] is not None:
                    try:
                        words_data = example["words"]
                        all_phonemes = []
                        
                        # Collect all phonemes across all words
                        for word_entry in words_data:
                            if "phones" in word_entry:
                                phones = word_entry["phones"]
                                if isinstance(phones, (list, tuple)):
                                    all_phonemes.extend(phones)
                        
                        # Tokenize phonemes to IDs
                        if all_phonemes:
                            tokenizer = PhonemeTokenizer()
                            phoneme_ids = tokenizer.encode(all_phonemes)
                            result["phoneme_ids"] = np.array(phoneme_ids, dtype=np.int32)
                            result["phoneme_sequence_length"] = len(phoneme_ids)
                            
                            logger.debug(f"Extracted {len(all_phonemes)} phonemes: {all_phonemes}")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting phonemes: {e}")
                        # Continue without phoneme extraction if it fails
                
                # ────────────────────────────────────────────────────────────────
                # COPY AND NORMALIZE UTTERANCE-LEVEL SCORES
                # ────────────────────────────────────────────────────────────────
                assessment_score_keys = [
                    "accuracy", "fluency", "prosodic", "completeness", "total"
                ]
                
                for key in example.keys():
                    if key not in ["audio", "words"]:
                        # Normalize assessment scores if enabled
                        if self.normalize_scores and key in assessment_score_keys:
                            try:
                                score_value = example[key]
                                if isinstance(score_value, (int, float)):
                                    result[key] = self.normalize_assessment_score(score_value)
                                else:
                                    result[key] = example[key]
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Failed to normalize score {key}={example[key]}: {e}")
                                result[key] = example[key]
                        else:
                            result[key] = example[key]
                
                # ────────────────────────────────────────────────────────────────
                # NORMALIZE WORD/PHONE SCORES (if they were extracted)
                # ────────────────────────────────────────────────────────────────
                word_score_keys = ["word_accuracy_scores", "word_stress_scores", "word_total_scores"]
                for key in word_score_keys:
                    if key in result and self.normalize_scores:
                        try:
                            scores = result[key]
                            normalized = np.array(
                                [self.normalize_assessment_score(s) for s in scores],
                                dtype=np.float32
                            )
                            result[key] = normalized
                        except Exception as e:
                            logger.warning(f"Failed to normalize {key}: {e}")
                
                # Normalize phone scores (phones are already on [0, 2] scale, so different range)
                if "phone_accuracy_scores" in result and self.normalize_scores:
                    try:
                        scores = result["phone_accuracy_scores"]
                        # Phones are on [0, 2] scale, normalize to [0, 1]
                        normalized = np.array(
                            [self.normalize_assessment_score(s, min_val=0, max_val=2) for s in scores],
                            dtype=np.float32
                        )
                        result["phone_accuracy_scores"] = normalized
                    except Exception as e:
                        logger.warning(f"Failed to normalize phone_accuracy_scores: {e}")
                
                return result
            
            except Exception as e:
                logger.warning(f"Error processing audio: {e}")
                # Return fallback with zero features
                result = {
                    "input_features": np.zeros((80, 3000), dtype=np.float32),
                }
                for key in example.keys():
                    if key not in ["audio", "words"]:
                        result[key] = example[key]
                return result
        
        # Process each split - CRITICAL: num_proc=None and batched=False to avoid codec issues
        processed_datasets = {}
        
        for split_name, split_data in datasets.items():
            logger.info(f"Processing {split_name} split ({len(split_data)} examples)...")
            
            try:
                processed = split_data.map(
                    preprocess_function,
                    batched=False,  # CRITICAL: Process one example at a time
                    num_proc=None,  # CRITICAL for Kaggle - disable multiprocessing
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