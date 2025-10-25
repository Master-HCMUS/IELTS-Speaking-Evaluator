"""
Whisper Pronunciation Assessment Model.

This module implements a Whisper-based model that maintains transcription capabilities
while adding pronunciation assessment heads for word-level and phone-level scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

from transformers import (
    WhisperForConditionalGeneration,
    WhisperConfig,
    WhisperProcessor
)


class PronunciationAssessmentHeads(nn.Module):
    """
    Assessment heads for word-level and phone-level pronunciation scoring.
    """
    
    def __init__(self, encoder_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        
        # Shared feature extraction
        self.shared_projection = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim // 2)
        )
        
        # Word-level assessment head
        self.word_accuracy_head = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 4, 1)  # Single accuracy score per word
        )
        
        self.word_stress_head = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 4, 1)  # Single stress score per word
        )
        
        self.word_total_head = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 4, 1)  # Single total score per word
        )
        
        # Phone-level assessment head
        self.phone_accuracy_head = nn.Sequential(
            nn.Linear(encoder_dim // 2, encoder_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim // 4, 1)  # Single accuracy score per phone
        )
        
        # Utterance-level assessment heads
        self.utterance_heads = nn.ModuleDict({
            'accuracy': nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim // 2, 1)
            ),
            'fluency': nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim // 2, 1)
            ),
            'prosodic': nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim // 2, 1)
            ),
            'completeness': nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim // 2, 1)
            ),
            'total': nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim // 2, 1)
            )
        })
    
    def forward(self, encoder_outputs: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pronunciation assessment heads.
        
        Args:
            encoder_outputs: Encoder hidden states [batch_size, seq_len, encoder_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary containing assessment predictions
        """
        batch_size, seq_len, _ = encoder_outputs.shape
        
        # Shared feature extraction
        shared_features = self.shared_projection(encoder_outputs)  # [B, T, D/2]
        
        # Word and phone-level predictions (per timestep)
        word_predictions = {
            'accuracy': self.word_accuracy_head(shared_features).squeeze(-1),  # [B, T]
            'stress': self.word_stress_head(shared_features).squeeze(-1),      # [B, T]
            'total': self.word_total_head(shared_features).squeeze(-1)         # [B, T]
        }
        
        phone_predictions = {
            'accuracy': self.phone_accuracy_head(shared_features).squeeze(-1)  # [B, T]
        }
        
        # Utterance-level predictions (pooled)
        if attention_mask is not None:
            # Masked pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoder_outputs)
            masked_outputs = encoder_outputs * mask_expanded
            pooled_output = masked_outputs.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            pooled_output = encoder_outputs.mean(dim=1)  # [B, D]
        
        utterance_predictions = {}
        for score_type, head in self.utterance_heads.items():
            utterance_predictions[score_type] = head(pooled_output).squeeze(-1)  # [B]
        
        return {
            'word_level': word_predictions,
            'phone_level': phone_predictions,
            'utterance_level': utterance_predictions
        }


class WhisperPronunciationAssessmentModel(nn.Module):
    """
    Whisper model with added pronunciation assessment capabilities.
    
    This model maintains the original Whisper transcription functionality while adding
    assessment heads for pronunciation scoring at word, phone, and utterance levels.
    """
    
    def __init__(self, 
                 whisper_model_name: str = "openai/whisper-tiny",
                 assessment_dropout: float = 0.1,
                 freeze_whisper_layers: int = 0):
        """
        Initialize the pronunciation assessment model.
        
        Args:
            whisper_model_name: Pre-trained Whisper model name
            assessment_dropout: Dropout rate for assessment heads
            freeze_whisper_layers: Number of Whisper layers to freeze (0 = none)
        """
        super().__init__()
        
        # Load pre-trained Whisper model
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        self.whisper_config = self.whisper.config
        
        # Freeze Whisper layers if specified
        if freeze_whisper_layers > 0:
            self._freeze_whisper_layers(freeze_whisper_layers)
        
        # Add pronunciation assessment heads
        encoder_dim = self.whisper_config.d_model
        self.assessment_heads = PronunciationAssessmentHeads(
            encoder_dim=encoder_dim,
            dropout=assessment_dropout
        )
        
        # Loss weights for multi-objective training
        self.loss_weights = {
            'asr': 1.0,
            'word_accuracy': 1.0,
            'word_stress': 0.5,
            'word_total': 1.0,
            'phone_accuracy': 1.0,
            'utterance_accuracy': 1.0,
            'utterance_fluency': 1.0,
            'utterance_prosodic': 1.0,
            'utterance_completeness': 0.1,  # Lower weight since 99.6% is 10
            'utterance_total': 1.0
        }
    
    def _freeze_whisper_layers(self, num_layers: int):
        """Freeze specified number of Whisper encoder layers."""
        # Freeze encoder layers
        for i, layer in enumerate(self.whisper.model.encoder.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print(f"Frozen first {num_layers} Whisper encoder layers")
    
    def forward(self, 
                input_features: torch.Tensor,
                decoder_input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pronunciation_targets: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True) -> Dict[str, Any]:
        """
        Forward pass with both transcription and assessment objectives.
        
        Args:
            input_features: Mel-scale spectrograms [batch_size, n_mels, seq_len]
            decoder_input_ids: Decoder input token IDs for teacher forcing
            labels: Target token IDs for ASR loss
            attention_mask: Encoder attention mask
            pronunciation_targets: Target pronunciation scores
            return_dict: Whether to return ModelOutput object
            
        Returns:
            Dictionary containing losses and predictions
        """
        # Get Whisper encoder outputs
        encoder_outputs = self.whisper.model.encoder(
            input_features=input_features,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        
        # Get encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Pronunciation assessment predictions
        assessment_predictions = self.assessment_heads(
            encoder_outputs=encoder_hidden_states,
            attention_mask=attention_mask
        )
        
        # ASR (transcription) forward pass
        whisper_outputs = self.whisper(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
        
        total_loss = 0
        losses = {}
        
        # ASR loss
        if labels is not None:
            asr_loss = whisper_outputs.loss
            losses['asr'] = asr_loss
            total_loss += self.loss_weights['asr'] * asr_loss
        
        # Assessment losses
        if pronunciation_targets is not None:
            assessment_losses = self._compute_assessment_losses(
                predictions=assessment_predictions,
                targets=pronunciation_targets
            )
            losses.update(assessment_losses)
            
            # Add weighted assessment losses to total
            for loss_name, loss_value in assessment_losses.items():
                if loss_name in self.loss_weights:
                    total_loss += self.loss_weights[loss_name] * loss_value
        
        if return_dict:
            return {
                'loss': total_loss,
                'losses': losses,
                'logits': whisper_outputs.logits,
                'assessment_predictions': assessment_predictions,
                'encoder_hidden_states': encoder_hidden_states
            }
        else:
            return total_loss, whisper_outputs.logits, assessment_predictions
    
    def _compute_assessment_losses(self, 
                                 predictions: Dict[str, Dict[str, torch.Tensor]], 
                                 targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute assessment losses for different granularities."""
        losses = {}
        
        # Word-level losses
        if 'word_level' in targets and targets['word_level']:
            word_targets = targets['word_level']
            word_preds = predictions['word_level']
            
            for score_type in ['accuracy', 'stress', 'total']:
                if score_type in word_targets and score_type in word_preds:
                    pred_tensor = word_preds[score_type]  # [batch_size, seq_len]
                    target_tensor = word_targets[score_type]  # [batch_size, num_words]
                    
                    # Handle dimension mismatch: downsample predictions to match targets
                    pooled_pred = self._align_predictions_to_targets(pred_tensor, target_tensor)
                    
                    if pooled_pred is not None:
                        # Compute MSE loss
                        loss = F.mse_loss(
                            pooled_pred, 
                            target_tensor.float(),
                            reduction='mean'
                        )
                        losses[f'word_{score_type}'] = loss
        
        # Phone-level losses
        if 'phone_level' in targets and targets['phone_level']:
            phone_targets = targets['phone_level']
            phone_preds = predictions['phone_level']
            
            if 'accuracy' in phone_targets and 'accuracy' in phone_preds:
                pred_tensor = phone_preds['accuracy']  # [batch_size, seq_len]
                target_tensor = phone_targets['accuracy']  # [batch_size, num_phones]
                
                # Handle dimension mismatch: downsample predictions to match targets
                pooled_pred = self._align_predictions_to_targets(pred_tensor, target_tensor)
                
                if pooled_pred is not None:
                    loss = F.mse_loss(
                        pooled_pred,
                        target_tensor.float(),
                        reduction='mean'
                    )
                    losses['phone_accuracy'] = loss
        
        # Utterance-level losses (these should match already since they're pooled)
        if 'utterance_level' in targets and targets['utterance_level']:
            utterance_targets = targets['utterance_level']
            utterance_preds = predictions['utterance_level']
            
            for score_type in ['accuracy', 'fluency', 'prosodic', 'completeness', 'total']:
                if score_type in utterance_targets and score_type in utterance_preds:
                    loss = F.mse_loss(
                        utterance_preds[score_type],
                        utterance_targets[score_type].float(),
                        reduction='mean'
                    )
                    losses[f'utterance_{score_type}'] = loss
        
        return losses
    
    def _align_predictions_to_targets(self, predictions: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Align prediction tensors to target tensor dimensions.
        
        Args:
            predictions: Predicted scores [batch_size, seq_len]
            targets: Target scores [batch_size, target_len]
            
        Returns:
            Aligned predictions or None if alignment fails
        """
        if predictions.size(1) == targets.size(1):
            return predictions
        
        batch_size = predictions.size(0)
        target_len = targets.size(1)
        seq_len = predictions.size(1)
        
        # Check for valid dimensions
        if target_len <= 0 or seq_len <= 0:
            return None
        
        try:
            if seq_len > target_len:
                # Downsample predictions to match target length
                pool_size = seq_len // target_len
                if pool_size > 0:
                    # Take average of every pool_size frames
                    aligned_pred = predictions[:, :target_len * pool_size].view(
                        batch_size, target_len, pool_size
                    ).mean(dim=2)
                else:
                    # Linear interpolation for better alignment
                    aligned_pred = F.interpolate(
                        predictions.unsqueeze(1), 
                        size=target_len, 
                        mode='linear', 
                        align_corners=False
                    ).squeeze(1)
            else:
                # Upsample predictions to match target length
                aligned_pred = F.interpolate(
                    predictions.unsqueeze(1), 
                    size=target_len, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1)
            
            return aligned_pred
            
        except Exception as e:
            # Return None if alignment fails
            return None
    
    def generate_transcription(self, input_features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate transcription using the Whisper decoder."""
        return self.whisper.generate(input_features, **kwargs)
    
    def predict_pronunciation_scores(self, 
                                   input_features: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict pronunciation scores without transcription."""
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = self.whisper.model.encoder(
                input_features=input_features,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
            
            # Get assessment predictions
            assessment_predictions = self.assessment_heads(
                encoder_outputs=encoder_outputs.last_hidden_state,
                attention_mask=attention_mask
            )
            
            return assessment_predictions
    
    def save_pretrained(self, save_directory: str):
        """Save the model and configuration, handling shared tensors properly."""
        import os
        import json
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save the full model state using torch.save to avoid shared tensor issues
        # This is safer than safetensors for models with tied weights
        model_path = save_path / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)
        logger.info(f"Model weights saved to {model_path}")
        
        # Save configuration
        config = {
            "whisper_model_name": getattr(self.whisper.config, 'name_or_path', self.whisper.config._name_or_path),
            "model_type": "whisper_pronunciation_assessment",
            "assessment_dropout": 0.1,  # Default value
            "loss_weights": self.loss_weights,
            "model_class": "WhisperPronunciationAssessmentModel"
        }
        
        config_path = save_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Model config saved to {config_path}")
        
        # Save the Whisper processor components
        try:
            processor = WhisperProcessor.from_pretrained(
                getattr(self.whisper.config, 'name_or_path', self.whisper.config._name_or_path)
            )
            processor.save_pretrained(save_directory)
            logger.info(f"Processor saved to {save_directory}")
        except Exception as e:
            logger.warning(f"Could not save processor: {e}")
        
        logger.info(f"Model successfully saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a pre-trained pronunciation assessment model."""
        import json
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Load configuration
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            whisper_model_name=config["whisper_model_name"],
            assessment_dropout=config.get("assessment_dropout", 0.1)
        )
        
        # Load weights
        state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        
        # Load loss weights if available
        if "loss_weights" in config:
            model.loss_weights = config["loss_weights"]
        
        return model