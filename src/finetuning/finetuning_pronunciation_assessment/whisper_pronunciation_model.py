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
from torch.nn.utils.rnn import pad_sequence

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


class PhonemeDecoderCTC(nn.Module):
    """CTC-based phoneme decoder head for predicting phone symbols."""
    
    def __init__(self, input_dim: int, num_phonemes: int = 75, blank: int = 0):
        """
        Initialize CTC-based phoneme decoder.
        
        Args:
            input_dim: Encoder hidden dimension
            num_phonemes: Number of phoneme tokens in vocabulary
            blank: Blank token ID for CTC (typically 0)
        """
        super().__init__()
        self.num_phonemes = num_phonemes
        self.blank = blank
        
        # Simple linear projection to phoneme vocabulary
        self.fc = nn.Linear(input_dim, num_phonemes)
        
        # CTC Loss function (will be created during forward if needed)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=blank,
            reduction='mean',
            zero_infinity=True
        )
        
        print(f"PhonemeDecoderCTC initialized with {num_phonemes} phonemes, blank={blank}")
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        phoneme_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for phoneme prediction.
        
        Args:
            encoder_hidden: Encoder hidden states [batch, seq_len, hidden_dim]
            phoneme_ids: Target phoneme token IDs [batch, max_target_len] (training only)
            input_lengths: Length of each sequence in batch [batch] (training only)
            target_lengths: Length of each target in batch [batch] (training only)
            
        Returns:
            Dictionary with:
            - 'logits': [batch, seq_len, num_phonemes] - raw logits
            - 'loss': scalar (only during training when phoneme_ids provided)
        """
        # Project encoder hidden to phoneme logits
        # [batch, seq_len, hidden_dim] -> [batch, seq_len, num_phonemes]
        logits = self.fc(encoder_hidden)
        outputs = {'logits': logits}
        
        # Compute CTC loss during training
        if phoneme_ids is not None and input_lengths is not None and target_lengths is not None:
            # CTC expects: [seq_len, batch, num_phonemes]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            # Ensure lengths are CPU tensors (CTC loss requirement)
            if input_lengths.device != torch.device('cpu'):
                input_lengths = input_lengths.cpu()
            if target_lengths.device != torch.device('cpu'):
                target_lengths = target_lengths.cpu()
            
            # Compute CTC loss
            try:
                loss = self.ctc_loss_fn(log_probs, phoneme_ids, input_lengths, target_lengths)
                outputs['loss'] = loss
                print(f"CTC loss computed")
            except Exception as e:
                print(f"CTC loss computation failed: {e}")
                outputs['loss'] = torch.tensor(0.0, device=encoder_hidden.device, requires_grad=True)
        
        return outputs
    
    def decode_greedy(self, logits: torch.Tensor) -> List[List[int]]:
        """
        Greedy decoding: take argmax of logits and collapse repeated tokens.
        
        Args:
            logits: [batch, seq_len, num_phonemes]
            
        Returns:
            List of decoded sequences (list of token IDs)
        """
        batch_size, seq_len, _ = logits.shape
        
        # Get argmax predictions
        predictions = logits.argmax(dim=-1)  # [batch, seq_len]
        
        decoded = []
        for i in range(batch_size):
            pred_seq = predictions[i].cpu().numpy().tolist()
            
            # Collapse repeated tokens
            collapsed = []
            for token in pred_seq:
                if not collapsed or token != collapsed[-1] and token != self.blank:
                    if token != self.blank:
                        collapsed.append(token)
            
            decoded.append(collapsed)
        
        return decoded



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
        train_phoneme_symbols: bool = True,
        num_phonemes: int = 75,
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
            train_phoneme_symbols: Whether to train phoneme symbol decoder (CTC)
            num_phonemes: Number of phoneme tokens in vocabulary (for CTC decoder)
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
        self.train_phoneme_symbols = train_phoneme_symbols
        self.num_phonemes = num_phonemes
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
            print(f"Loading WhisperForConditionalGeneration: {self.model_name}")
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
            print("Encoder weights frozen")
        
        # Freeze decoder if requested
        if self.freeze_decoder:
            decoder = self.model.get_decoder()
            for param in decoder.parameters():
                param.requires_grad = False
            print("Decoder weights frozen")
        
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
        
        # Phoneme decoder (CTC-based) for phoneme symbol prediction
        if self.train_phoneme_symbols:
            self.phoneme_decoder = PhonemeDecoderCTC(
                input_dim=hidden_dim,
                num_phonemes=self.num_phonemes,
                blank=0
            )
            print(f"Phoneme decoder (CTC) initialized with {self.num_phonemes} phonemes")
        
        self._initialized = True
        print("Model initialized successfully")

    
    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        phoneme_ids: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        phoneme_sequence_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for both transcription and assessment.
        
        Args:
            input_features: Audio mel-spectrogram features [batch, 80, 3000]
            decoder_input_ids: Optional decoder input IDs for training
            attention_mask: Optional attention mask
            phoneme_ids: Optional phoneme token IDs [batch, max_phoneme_len] for CTC training
            input_lengths: Length of each input sequence [batch] for CTC
            phoneme_sequence_lengths: Length of each phoneme sequence [batch] for CTC
            
        Returns:
            Dictionary with:
            - 'transcription_logits': Decoder logits for transcription
            - 'encoder_hidden_states': Encoder outputs [batch, seq_len, hidden_dim]
            - Frame-level scores: word_accuracy, word_stress, word_total, phone_accuracy
            - Utterance-level scores: utterance_accuracy, fluency, prosodic, completeness, total
            - Phoneme predictions: phoneme_logits, phoneme_loss (if train_phoneme_symbols=True)
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
        
        # Phoneme decoder (CTC-based) for phoneme symbol prediction
        if self.train_phoneme_symbols:
            phoneme_outputs = self.phoneme_decoder(
                encoder_hidden=encoder_last_hidden,
                phoneme_ids=phoneme_ids,
                input_lengths=input_lengths,
                target_lengths=phoneme_sequence_lengths
            )
            outputs['phoneme_logits'] = phoneme_outputs['logits']
            if 'loss' in phoneme_outputs:
                outputs['phoneme_loss'] = phoneme_outputs['loss']
        
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
    
    def generate_phonemes(
        self,
        input_features: torch.Tensor,
        return_confidence: bool = True,
        return_frame_level: bool = False,
        collapse_repeated: bool = True
    ) -> Dict[str, Any]:
        """
        Generate phoneme predictions from audio features.
        
        This is the main inference method for phoneme generation. It takes audio
        features and returns decoded phoneme symbols with confidence scores.
        
        Args:
            input_features: Mel-spectrogram features [batch, 80, 3000]
            return_confidence: Include confidence scores [0, 10] for each phoneme
            return_frame_level: Include frame-level predictions and confidences
            collapse_repeated: Apply CTC rule to collapse repeated phonemes
            
        Returns:
            Dictionary (single sample) or list of dicts (batch) with:
            {
                'phoneme_ids': List[int],              # Token IDs [0-128]
                'phoneme_symbols': List[str],          # ARPABET symbols
                'phoneme_confidence': List[float],     # Confidence [0-10]
                'num_phonemes': int,                   # Number of phonemes
                'sequence_length': int,                # Total frames
                'blank_frames_removed': int,           # Frames that were blank
                'frame_confidence': Optional[List[float]]  # Per-frame scores
            }
            
        Example:
            >>> result = model.generate_phonemes(input_features)
            >>> print(result['phoneme_symbols'])
            ['W', 'AE1', 'D']
            >>> print(result['phoneme_confidence'])
            [9.2, 8.7, 9.1]
        """
        self._initialize_model()
        self.eval()
        
        with torch.no_grad():
            predictions = self.forward(input_features)
        
        if 'phoneme_logits' not in predictions:
            logger.warning("Model does not have phoneme_logits. Enable train_phoneme_symbols.")
            return None
        
        phoneme_logits = predictions['phoneme_logits']  # [batch, seq_len, 129]
        batch_size = phoneme_logits.shape[0]
        
        results = []
        for batch_idx in range(batch_size):
            logits_seq = phoneme_logits[batch_idx]  # [seq_len, 129]
            
            # Greedy decoding: argmax per frame
            phoneme_ids = logits_seq.argmax(dim=-1)  # [seq_len]
            
            # Collapse repeated tokens (CTC rule) and remove blanks
            decoded_ids = self._collapse_and_filter_tokens(
                phoneme_ids.cpu().numpy().tolist(),
                collapse_repeated=collapse_repeated
            )
            
            # Convert token IDs to phoneme symbols
            phoneme_symbols = self.decode_phoneme_ids_to_symbols(decoded_ids)
            
            # Compute confidence scores
            phoneme_confidence = None
            if return_confidence:
                phoneme_confidence = self.compute_phoneme_confidence(
                    logits_seq, phoneme_ids.cpu(), decoded_ids
                )
            
            # Frame-level predictions (optional)
            frame_confidence = None
            if return_frame_level:
                frame_probs = torch.softmax(logits_seq, dim=-1)
                frame_confidence = (frame_probs.max(dim=-1).values.cpu() * 10).tolist()
            
            # Count blank frames removed
            blank_count = (phoneme_ids == 0).sum().item()
            repeat_count = self._count_repeated_frames(phoneme_ids.cpu().numpy().tolist())
            
            result = {
                'phoneme_ids': decoded_ids,
                'phoneme_symbols': phoneme_symbols,
                'phoneme_confidence': phoneme_confidence,
                'num_phonemes': len(decoded_ids),
                'sequence_length': phoneme_ids.shape[0],
                'blank_frames_removed': blank_count,
                'repeated_frames_collapsed': repeat_count if collapse_repeated else 0
            }
            
            if return_frame_level:
                result['frame_confidence'] = frame_confidence
            
            results.append(result)
        
        # Return single dict for batch_size=1, list for batch_size>1
        return results[0] if batch_size == 1 else results
    
    def decode_phoneme_ids_to_symbols(
        self,
        phoneme_ids: List[int]
    ) -> List[str]:
        """
        Convert phoneme token IDs to ARPABET symbols.
        
        Args:
            phoneme_ids: List of token IDs in range [0, 128]
            
        Returns:
            List of phoneme symbol strings (e.g., ["W", "AE1", "D"])
            
        Example:
            >>> ids = [1, 15, 45]
            >>> symbols = model.decode_phoneme_ids_to_symbols(ids)
            >>> print(symbols)
            ['W', 'AE1', 'B']
        """
        try:
            from phoneme_tokenizer import ARPABET_VOCAB
        except ImportError:
            # Fallback: use local import
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from phoneme_tokenizer import ARPABET_VOCAB
        
        symbols = []
        for pid in phoneme_ids:
            # Clamp to valid range
            pid_clamped = max(0, min(pid, len(ARPABET_VOCAB) - 1))
            symbol = ARPABET_VOCAB[pid_clamped]
            symbols.append(symbol)
        
        return symbols
    
    def compute_phoneme_confidence(
        self,
        phoneme_logits_seq: torch.Tensor,  # [seq_len, 129]
        phoneme_ids_seq: torch.Tensor,     # [seq_len]
        decoded_ids: List[int]
    ) -> List[float]:
        """
        Compute confidence scores for decoded phoneme sequence.
        
        Extracts the softmax probability of the predicted phoneme at each
        decoded position and scales to [0, 10] range.
        
        Args:
            phoneme_logits_seq: Logits for one sample [seq_len, 129]
            phoneme_ids_seq: Greedy-decoded IDs [seq_len]
            decoded_ids: Post-processed IDs (blanks removed, repeats collapsed)
            
        Returns:
            List of confidence scores [0, 10] for each decoded phoneme
            
        Example:
            >>> logits = torch.randn(1500, 129)
            >>> ids = torch.argmax(logits, dim=-1)
            >>> decoded = [1, 15, 45]
            >>> confidence = model.compute_phoneme_confidence(logits, ids, decoded)
            >>> print(confidence)
            [9.2, 8.7, 9.1]
        """
        # Convert logits to probabilities
        probs = torch.softmax(phoneme_logits_seq, dim=-1)  # [seq_len, 129]
        
        # Get max probability per frame
        max_probs = probs.max(dim=-1).values  # [seq_len]
        
        # Extract confidence for each decoded phoneme
        # by tracking which frames contribute to each phoneme
        confidence_scores = []
        
        current_seq_idx = 0
        last_id = -1
        
        for frame_idx in range(len(phoneme_ids_seq)):
            token_id = phoneme_ids_seq[frame_idx].item()
            
            # Skip blanks (ID=0)
            if token_id == 0:
                continue
            
            # Skip repeated tokens (CTC rule)
            if token_id == last_id:
                continue
            
            # This frame contributes to a phoneme
            confidence = (max_probs[frame_idx].item() * 10)
            confidence_scores.append(confidence)
            last_id = token_id
        
        return confidence_scores
    
    @staticmethod
    def _collapse_and_filter_tokens(
        token_ids: List[int],
        blank_id: int = 0,
        collapse_repeated: bool = True
    ) -> List[int]:
        """
        Collapse repeated tokens (CTC rule) and remove blanks.
        
        Args:
            token_ids: Sequence of token IDs
            blank_id: ID representing blank (default 0 for CTC)
            collapse_repeated: Whether to collapse repeated tokens
            
        Returns:
            Filtered and optionally collapsed token IDs
            
        Example:
            >>> ids = [0, 1, 1, 1, 15, 15, 45, 0, 0]
            >>> result = WhisperPronunciationAssessmentModel._collapse_and_filter_tokens(ids)
            >>> print(result)
            [1, 15, 45]
        """
        # First pass: collapse repeated tokens if enabled
        if collapse_repeated:
            collapsed = []
            for token_id in token_ids:
                if len(collapsed) == 0 or token_id != collapsed[-1]:
                    collapsed.append(token_id)
            token_ids = collapsed
        
        # Second pass: remove blanks
        filtered = [tid for tid in token_ids if tid != blank_id]
        
        return filtered
    
    @staticmethod
    def _count_repeated_frames(token_ids: List[int]) -> int:
        """
        Count how many frames would be collapsed due to repetition.
        
        Args:
            token_ids: Sequence of token IDs
            
        Returns:
            Number of frames that are repeats of the previous frame
        """
        count = 0
        for i in range(1, len(token_ids)):
            if token_ids[i] == token_ids[i-1]:
                count += 1
        return count

