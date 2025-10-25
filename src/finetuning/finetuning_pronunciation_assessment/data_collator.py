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
                - input_features: [80, T] mel-spectrogram
                - labels: [S] token indices
                - word_accuracy_scores: [W] word-level scores (if present)
                - word_stress_scores: [W] word-level scores (if present)
                - word_total_scores: [W] word-level scores (if present)
                - phone_accuracy_scores: [P] phone-level scores (if present)
                - accuracy, fluency, prosodic, completeness, total: utterance scores
        
        Returns:
            Dict with batched tensors
        """
        # Extract input features and pad them
        input_features = []
        for ex in batch:
            feat = ex["input_features"]
            # Handle both list and numpy array formats
            if isinstance(feat, list):
                feat = np.array(feat, dtype=np.float32)
            input_features.append(feat)
        
        # Get max length for padding
        feature_lengths = [feat.shape[1] if len(feat.shape) == 2 else len(feat) 
                          for feat in input_features]
        max_feat_length = max(feature_lengths)
        
        # Pad features
        batch_size = len(batch)
        num_mel_bins = input_features[0].shape[0] if len(input_features[0].shape) == 2 else 1
        
        padded_features = np.full(
            (batch_size, num_mel_bins, max_feat_length),
            self.padding_value,
            dtype=np.float32
        )
        attention_mask = np.zeros((batch_size, max_feat_length), dtype=np.int32)
        
        for i, feat in enumerate(input_features):
            if len(feat.shape) == 2:
                feat_len = feat.shape[1]
                padded_features[i, :, :feat_len] = feat
            else:
                feat_len = len(feat)
                padded_features[i, 0, :feat_len] = feat
            attention_mask[i, :feat_len] = 1
        
        # Convert to tensors
        collated = {
            "input_features": torch.from_numpy(padded_features).float(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
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
        
        return collated


class SimplePronunciationDataCollator:
    """
    Simplified collator for utterance-level assessment only.
    Does not handle word/phone-level scores.
    """
    
    def __init__(self, padding_value: float = -100.0):
        self.padding_value = padding_value
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with only utterance-level scores."""
        # Extract and pad features
        input_features = [ex["input_features"] for ex in batch]
        feature_lengths = [feat.shape[1] if len(feat.shape) == 2 else len(feat) 
                          for feat in input_features]
        max_feat_length = max(feature_lengths)
        
        batch_size = len(batch)
        num_mel_bins = input_features[0].shape[0] if len(input_features[0].shape) == 2 else 1
        
        padded_features = np.full(
            (batch_size, num_mel_bins, max_feat_length),
            self.padding_value,
            dtype=np.float32
        )
        attention_mask = np.zeros((batch_size, max_feat_length), dtype=np.int32)
        
        for i, feat in enumerate(input_features):
            if len(feat.shape) == 2:
                feat_len = feat.shape[1]
                padded_features[i, :, :feat_len] = feat
            else:
                feat_len = len(feat)
                padded_features[i, 0, :feat_len] = feat
            attention_mask[i, :feat_len] = 1
        
        collated = {
            "input_features": torch.from_numpy(padded_features).float(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
        }
        
        # Add utterance-level scores only
        utterance_score_keys = ["accuracy", "fluency", "prosodic", "completeness", "total"]
        for key in utterance_score_keys:
            if key in batch[0]:
                scores = np.array([ex[key] for ex in batch], dtype=np.float32)
                collated[key] = torch.from_numpy(scores).float()
        
        return collated
