"""Data collator for batching variable-length audio features and scores."""

from typing import Dict, List
import numpy as np
import torch


class PronunciationAssessmentDataCollator:
    """
    Collates batches of variable-length mel-spectrogram features and assessment scores.
    
    Handles:
    - Padding mel-spectrograms to same length with attention masks
    - Stacking variable-length score arrays
    - Converting to PyTorch tensors
    """
    
    def __init__(self, padding_value: float = -100.0):
        """
        Args:
            padding_value: Value to use for padding mel-spectrograms
        """
        self.padding_value = padding_value
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.
        
        Args:
            batch: List of examples from dataset, each containing:
                - input_features: [80, 3000] mel-spectrogram (already padded)
                - labels: [S] token indices
                - utterance scores: accuracy, fluency, prosodic, completeness, total
        
        Returns:
            Dict with batched tensors
        """
        # Extract input features (already padded to 3000 by data processor)
        input_features = []
        for ex in batch:
            feat = ex["input_features"]
            # Handle both list and numpy array formats
            if isinstance(feat, list):
                feat = np.array(feat, dtype=np.float32)
            
            # Ensure it's the right shape: [80, 3000]
            if feat.shape != (80, 3000):
                # If it's [80, time] and time < 3000, pad it
                if len(feat.shape) == 2 and feat.shape[0] == 80 and feat.shape[1] < 3000:
                    pad_width = 3000 - feat.shape[1]
                    feat = np.pad(feat, ((0, 0), (0, pad_width)), constant_values=-100.0)
                # If it's [80, time] and time > 3000, truncate
                elif len(feat.shape) == 2 and feat.shape[0] == 80 and feat.shape[1] > 3000:
                    feat = feat[:, :3000]
            
            input_features.append(feat)
        
        # Stack features
        batch_size = len(input_features)
        stacked_features = np.stack(input_features, axis=0)  # [batch, 80, 3000]
        
        # Convert to tensors
        collated = {
            "input_features": torch.from_numpy(stacked_features).float(),
        }
        
        # Handle labels (token indices for transcription)
        if "labels" in batch[0]:
            labels = [ex["labels"] for ex in batch]
            # Pad labels to max length with -100 (ignore index)
            max_label_length = max(len(l) for l in labels)
            padded_labels = np.full(
                (batch_size, max_label_length),
                -100,
                dtype=np.int64
            )
            for i, label in enumerate(labels):
                padded_labels[i, :len(label)] = label
            collated["labels"] = torch.from_numpy(padded_labels).long()
        
        # Handle word-level scores (variable length per example)
        word_score_keys = ["word_accuracy_scores", "word_stress_scores", "word_total_scores"]
        for key in word_score_keys:
            if key in batch[0]:
                word_scores = [ex[key] for ex in batch]
                # Stack as list of tensors (different lengths)
                collated[key] = [torch.from_numpy(np.array(s, dtype=np.float32)) 
                                for s in word_scores]
        
        # Handle phone-level scores (variable length per example)
        if "phone_accuracy_scores" in batch[0]:
            phone_scores = [ex["phone_accuracy_scores"] for ex in batch]
            collated["phone_accuracy_scores"] = [torch.from_numpy(np.array(s, dtype=np.float32)) 
                                                 for s in phone_scores]
        
        # Handle utterance-level scores (fixed, one per example)
        utterance_score_keys = ["accuracy", "fluency", "prosodic", "completeness", "total"]
        for key in utterance_score_keys:
            if key in batch[0]:
                scores = np.array([ex[key] for ex in batch], dtype=np.float32)
                collated[key] = torch.from_numpy(scores).float()
        
        # Handle phoneme IDs (for CTC phoneme decoder)
        if "phoneme_ids" in batch[0]:
            phoneme_ids = [ex["phoneme_ids"] for ex in batch]
            # Pad phoneme IDs to max length with 72 (PAD token ID)
            max_phoneme_length = max(len(p) for p in phoneme_ids)
            padded_phoneme_ids = np.full(
                (batch_size, max_phoneme_length),
                72,  # PAD token ID
                dtype=np.int32
            )
            for i, phoneme_id in enumerate(phoneme_ids):
                padded_phoneme_ids[i, :len(phoneme_id)] = phoneme_id
            collated["phoneme_ids"] = torch.from_numpy(padded_phoneme_ids).long()
            
            # Also collect phoneme sequence lengths (before padding)
            phoneme_lengths = np.array(
                [len(ex["phoneme_ids"]) for ex in batch],
                dtype=np.int32
            )
            collated["phoneme_sequence_lengths"] = torch.from_numpy(phoneme_lengths).long()
            
            # Store input lengths for CTC loss (length of encoder output)
            # For Whisper encoder output: length = 3000 / 2 = 1500 (with downsampling)
            input_lengths = np.full(batch_size, 1500, dtype=np.int32)
            collated["input_lengths"] = torch.from_numpy(input_lengths).long()
        
        return collated