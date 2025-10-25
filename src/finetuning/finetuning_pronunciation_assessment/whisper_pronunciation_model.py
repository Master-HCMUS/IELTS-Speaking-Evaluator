"""
Whisper-based model with pronunciation assessment heads.

Extends Whisper with multi-level assessment capabilities while maintaining
transcription quality through multi-objective training.
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PronunciationAssessmentHead(nn.Module):
    """Assessment head for predicting pronunciation scores."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize assessment head.
        
        Args:
            input_dim: Encoder hidden dimension
            hidden_dim: Hidden dimension for the head
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Scores tensor [batch, seq_len] or [batch] after pooling
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class WhisperPronunciationAssessmentModel(nn.Module):
    """Whisper model with pronunciation assessment capabilities."""
    
    def __init__(
        self,
        model_name: str,
        train_word_level: bool = True,
        train_phone_level: bool = True,
        train_utterance_level: bool = True,
        freeze_encoder: bool = False,
        use_word_level_assessment: bool = None,
        use_phone_level_assessment: bool = None,
        use_utterance_level_assessment: bool = None
    ):
        """
        Initialize model.
        
        Args:
            model_name: Whisper model name
            train_word_level: Whether to train word-level assessment
            train_phone_level: Whether to train phone-level assessment
            train_utterance_level: Whether to train utterance-level assessment
            freeze_encoder: Whether to freeze encoder weights
            use_word_level_assessment: (Deprecated) Use train_word_level instead
            use_phone_level_assessment: (Deprecated) Use train_phone_level instead
            use_utterance_level_assessment: (Deprecated) Use train_utterance_level instead
        """
        super().__init__()
        
        self.model = WhisperModel.from_pretrained(model_name)
        config = self.model.config
        hidden_dim = config.d_model
        
        # Support both naming conventions
        if use_word_level_assessment is not None:
            train_word_level = use_word_level_assessment
        if use_phone_level_assessment is not None:
            train_phone_level = use_phone_level_assessment
        if use_utterance_level_assessment is not None:
            train_utterance_level = use_utterance_level_assessment
        
        self.train_word_level = train_word_level
        self.train_phone_level = train_phone_level
        self.train_utterance_level = train_utterance_level
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        
        # Word-level assessment heads
        if self.train_word_level:
            self.word_accuracy_head = PronunciationAssessmentHead(hidden_dim)
            self.word_stress_head = PronunciationAssessmentHead(hidden_dim)
            self.word_total_head = PronunciationAssessmentHead(hidden_dim)
        
        # Phone-level assessment head
        if self.train_phone_level:
            self.phone_accuracy_head = PronunciationAssessmentHead(hidden_dim)
        
        # Utterance-level assessment heads
        if self.train_utterance_level:
            self.utterance_accuracy_head = nn.Linear(hidden_dim, 1)
            self.utterance_fluency_head = nn.Linear(hidden_dim, 1)
            self.utterance_prosodic_head = nn.Linear(hidden_dim, 1)
            self.utterance_completeness_head = nn.Linear(hidden_dim, 1)
            self.utterance_total_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        word_scores: Optional[torch.Tensor] = None,
        phone_scores: Optional[torch.Tensor] = None,
        utterance_scores: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            input_features: Audio mel-spectrogram features
            labels: Transcription token IDs
            word_scores: Target word-level scores
            phone_scores: Target phone-level scores
            utterance_scores: Target utterance-level scores
            
        Returns:
            Dictionary with logits and loss
        """
        # Get encoder outputs
        encoder_outputs = self.model.encoder(input_features)
        encoder_last_hidden = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Pool for utterance-level (mean pooling over sequence)
        encoder_mean = encoder_last_hidden.mean(dim=1)  # [batch, hidden_dim]
        
        outputs = {
            'encoder_hidden_states': encoder_last_hidden,
            'encoder_mean': encoder_mean
        }
        
        # Assessment predictions
        if self.train_word_level:
            outputs['word_accuracy_logits'] = self.word_accuracy_head(encoder_last_hidden)
            outputs['word_stress_logits'] = self.word_stress_head(encoder_last_hidden)
            outputs['word_total_logits'] = self.word_total_head(encoder_last_hidden)
        
        if self.train_phone_level:
            outputs['phone_accuracy_logits'] = self.phone_accuracy_head(encoder_last_hidden)
        
        if self.train_utterance_level:
            outputs['utterance_accuracy_logits'] = self.utterance_accuracy_head(encoder_mean).squeeze(-1)
            outputs['utterance_fluency_logits'] = self.utterance_fluency_head(encoder_mean).squeeze(-1)
            outputs['utterance_prosodic_logits'] = self.utterance_prosodic_head(encoder_mean).squeeze(-1)
            outputs['utterance_completeness_logits'] = self.utterance_completeness_head(encoder_mean).squeeze(-1)
            outputs['utterance_total_logits'] = self.utterance_total_head(encoder_mean).squeeze(-1)
        
        return outputs
    
    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode audio to hidden states."""
        return self.model.encoder(input_features).last_hidden_state
    
    def generate_transcription(self, input_features: torch.Tensor, max_length: int = 128):
        """Generate transcription (beam search)."""
        encoder_outputs = self.model.encoder(input_features)
        return self.model.decoder.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=1
        )
