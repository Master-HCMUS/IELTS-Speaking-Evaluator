"""
Whisper-based model with pronunciation assessment heads.

Extends WhisperForConditionalGeneration (full encoder-decoder) with assessment heads
for word-level and phone-level pronunciation evaluation while maintaining transcription.

Architecture:
- Encoder: Processes audio mel-spectrogram
- Decoder: Generates transcription
- Assessment Heads: Predict pronunciation scores at word and phone levels
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Any
import logging
import sys
import os
import numpy as np

logger = logging.getLogger(__name__)


def _suppress_jax_import():
    """Suppress JAX import to avoid NumPy incompatibility on Kaggle."""
    # Set environment variable to disable JAX
    os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Try to suppress the import error by catching it early
    import warnings
    warnings.filterwarnings("ignore", category=AttributeError)



class FrameLevelAssessmentHead(nn.Module):
    """Frame-level assessment head for word and phone-level scoring."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize frame-level assessment head.
        
        Args:
            input_dim: Encoder hidden dimension
            hidden_dim: Hidden dimension for the head
        """
        super().__init__()
        # Deeper network with batch normalization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Scores tensor [batch, seq_len] - one score per frame (normalized to [0, 1])
        """
        # Save original shape for reshaping later
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape for batch norm: [batch, seq_len, hidden_dim] -> [batch*seq_len, hidden_dim]
        x = x.reshape(-1, hidden_dim)
        
        # First layer
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second layer
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer with sigmoid normalization to [0, 1]
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        # Reshape back: [batch*seq_len, 1] -> [batch, seq_len]
        x = x.reshape(batch_size, seq_len)
        return x


class UtteranceLevelAssessmentHead(nn.Module):
    """Utterance-level assessment head for overall scores."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize utterance-level assessment head.
        
        Args:
            input_dim: Encoder hidden dimension
            hidden_dim: Hidden dimension for the head
        """
        super().__init__()
        # Deeper network with batch normalization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, hidden_dim] (mean-pooled)
            
        Returns:
            Scores tensor [batch] - one score per utterance (normalized to [0, 1])
        """
        # First layer
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second layer
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer with sigmoid normalization to [0, 1]
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x.squeeze(-1)  # [batch]



class WhisperPronunciationAssessmentModel(nn.Module):
    """
    Whisper model with pronunciation assessment capabilities.
    
    Uses WhisperForConditionalGeneration for both transcription (encoder-decoder)
    and assessment (heads on encoder outputs).
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        train_word_level: bool = True,
        train_phone_level: bool = True,
        train_utterance_level: bool = True,
        train_transcription: bool = True,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        """
        Initialize model with full Whisper (encoder-decoder) + assessment heads.
        
        Args:
            model_name: Whisper model name (e.g., "openai/whisper-base")
            train_word_level: Whether to train word-level assessment
            train_phone_level: Whether to train phone-level assessment
            train_utterance_level: Whether to train utterance-level assessment
            train_transcription: Whether to train transcription (decoder)
            freeze_encoder: Whether to freeze encoder weights
            freeze_decoder: Whether to freeze decoder weights
        """
        super().__init__()
        
        # Store config for lazy loading
        self.model_name = model_name
        self.model = None  # WhisperForConditionalGeneration
        self._hidden_dim = None
        self._initialized = False
        
        self.train_word_level = train_word_level
        self.train_phone_level = train_phone_level
        self.train_utterance_level = train_utterance_level
        self.train_transcription = train_transcription
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
    
    def _initialize_model(self):
        """Lazy initialization of model and heads to avoid JAX import issues."""
        if self._initialized:
            return
        
        # Suppress warnings about JAX/NumPy incompatibility
        import warnings
        warnings.filterwarnings("ignore")
        
        # Set environment to skip JAX
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        # Monkey-patch numpy.dtypes if it doesn't exist (for JAX compatibility)
        import numpy as np
        if not hasattr(np, 'dtypes'):
            class DummyDtypes:
                pass
            np.dtypes = DummyDtypes()
        
        # Now import transformers and load full WhisperForConditionalGeneration
        try:
            from transformers import WhisperForConditionalGeneration
            logger.info(f"Loading WhisperForConditionalGeneration: {self.model_name}")
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load WhisperForConditionalGeneration: {e}")
            raise
        
        warnings.filterwarnings("default")
        config = self.model.config
        hidden_dim = config.d_model
        self._hidden_dim = hidden_dim
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            encoder = self.model.get_encoder()
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder weights frozen")
        
        # Freeze decoder if requested
        if self.freeze_decoder:
            decoder = self.model.get_decoder()
            for param in decoder.parameters():
                param.requires_grad = False
            logger.info("Decoder weights frozen")
        
        # Frame-level assessment heads (word and phone level)
        if self.train_word_level:
            self.word_accuracy_head = FrameLevelAssessmentHead(hidden_dim)
            self.word_stress_head = FrameLevelAssessmentHead(hidden_dim)
            self.word_total_head = FrameLevelAssessmentHead(hidden_dim)
        
        if self.train_phone_level:
            self.phone_accuracy_head = FrameLevelAssessmentHead(hidden_dim)
        
        # Utterance-level assessment heads (one score per utterance)
        if self.train_utterance_level:
            self.utterance_accuracy_head = UtteranceLevelAssessmentHead(hidden_dim)
            self.utterance_fluency_head = UtteranceLevelAssessmentHead(hidden_dim)
            self.utterance_prosodic_head = UtteranceLevelAssessmentHead(hidden_dim)
            self.utterance_completeness_head = UtteranceLevelAssessmentHead(hidden_dim)
            self.utterance_total_head = UtteranceLevelAssessmentHead(hidden_dim)
        
        self._initialized = True
        logger.info("Model initialized successfully")

    
    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for both transcription and assessment.
        
        Args:
            input_features: Audio mel-spectrogram features [batch, 80, 3000]
            decoder_input_ids: Optional decoder input IDs for training
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with:
            - 'transcription_logits': Decoder logits for transcription
            - 'encoder_hidden_states': Encoder outputs [batch, seq_len, hidden_dim]
            - Frame-level scores: word_accuracy, word_stress, word_total, phone_accuracy
            - Utterance-level scores: utterance_accuracy, fluency, prosodic, completeness, total
        """
        # Lazy initialization of model and heads
        self._initialize_model()
        
        # Get encoder outputs
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(input_features)
        encoder_last_hidden = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Pool for utterance-level (mean pooling over sequence)
        encoder_mean = encoder_last_hidden.mean(dim=1)  # [batch, hidden_dim]
        
        outputs = {
            'encoder_hidden_states': encoder_last_hidden,
            'encoder_mean': encoder_mean
        }
        
        # Transcription logits (decoder) - for both inference and training
        if decoder_input_ids is not None:
            decoder = self.model.get_decoder()
            decoder_outputs = decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_last_hidden,
                attention_mask=attention_mask
            )
            # Get decoder logits for computing loss
            lm_logits = self.model.lm_head(decoder_outputs.last_hidden_state)
            outputs['transcription_logits'] = lm_logits
        
        # Frame-level assessment predictions (word and phone level)
        if self.train_word_level:
            outputs['word_accuracy_logits'] = self.word_accuracy_head(encoder_last_hidden)
            outputs['word_stress_logits'] = self.word_stress_head(encoder_last_hidden)
            outputs['word_total_logits'] = self.word_total_head(encoder_last_hidden)
        
        if self.train_phone_level:
            outputs['phone_accuracy_logits'] = self.phone_accuracy_head(encoder_last_hidden)
        
        # Utterance-level assessment predictions
        if self.train_utterance_level:
            outputs['utterance_accuracy_logits'] = self.utterance_accuracy_head(encoder_mean)
            outputs['utterance_fluency_logits'] = self.utterance_fluency_head(encoder_mean)
            outputs['utterance_prosodic_logits'] = self.utterance_prosodic_head(encoder_mean)
            outputs['utterance_completeness_logits'] = self.utterance_completeness_head(encoder_mean)
            outputs['utterance_total_logits'] = self.utterance_total_head(encoder_mean)
        
        return outputs
    
    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to hidden states.
        
        Args:
            input_features: Mel-spectrogram [batch, 80, 3000]
            
        Returns:
            Encoder hidden states [batch, seq_len, hidden_dim]
        """
        self._initialize_model()
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(input_features)
        return encoder_outputs.last_hidden_state
    
    def generate_transcription(
        self,
        input_features: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 1
    ) -> torch.Tensor:
        """
        Generate transcription using the decoder.
        
        Args:
            input_features: Mel-spectrogram [batch, 80, 3000]
            max_length: Maximum length of generated sequence
            num_beams: Number of beams for beam search
            
        Returns:
            Generated token IDs [batch, seq_len]
        """
        self._initialize_model()
        encoder = self.model.get_encoder()
        encoder_outputs = encoder(input_features)
        
        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams
        )
        
        return generated_ids
    
    def predict_assessment_scores(
        self,
        input_features: torch.Tensor
    ) -> Dict[str, Dict[str, Any]]:
        """
        Predict pronunciation assessment scores.
        
        Args:
            input_features: Mel-spectrogram [batch, 80, 3000]
            
        Returns:
            Dictionary with word, phone, and utterance level scores
        """
        self._initialize_model()
        self.eval()
        
        with torch.no_grad():
            predictions = self.forward(input_features)
        
        scores = {
            'word_level': {},
            'phone_level': {},
            'utterance_level': {}
        }
        
        # Extract word-level scores
        if self.train_word_level:
            if 'word_accuracy_logits' in predictions:
                scores['word_level']['accuracy'] = predictions['word_accuracy_logits']
            if 'word_stress_logits' in predictions:
                scores['word_level']['stress'] = predictions['word_stress_logits']
            if 'word_total_logits' in predictions:
                scores['word_level']['total'] = predictions['word_total_logits']
        
        # Extract phone-level scores
        if self.train_phone_level:
            if 'phone_accuracy_logits' in predictions:
                scores['phone_level']['accuracy'] = predictions['phone_accuracy_logits']
        
        # Extract utterance-level scores
        if self.train_utterance_level:
            if 'utterance_accuracy_logits' in predictions:
                scores['utterance_level']['accuracy'] = predictions['utterance_accuracy_logits']
            if 'utterance_fluency_logits' in predictions:
                scores['utterance_level']['fluency'] = predictions['utterance_fluency_logits']
            if 'utterance_prosodic_logits' in predictions:
                scores['utterance_level']['prosodic'] = predictions['utterance_prosodic_logits']
            if 'utterance_completeness_logits' in predictions:
                scores['utterance_level']['completeness'] = predictions['utterance_completeness_logits']
            if 'utterance_total_logits' in predictions:
                scores['utterance_level']['total'] = predictions['utterance_total_logits']
        
        return scores
