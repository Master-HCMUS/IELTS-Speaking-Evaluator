"""
Phoneme tokenizer for mapping phoneme symbols to token IDs.

Supports ARPABET phoneme vocabulary used in the SpeechOcean762 dataset.
"""

from typing import List, Dict, Union, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ARPABET phoneme vocabulary (129 phonemes including silence/unknown)
# Source: CMU Pronouncing Dictionary + custom additions for non-English sounds
ARPABET_VOCAB = [
    # Vowels (short)
    "AE", "AH", "AW", "AY",  # 0-3
    "EH", "ER", "EY",  # 4-6
    "IH", "IY",  # 7-8
    "OW", "OY",  # 9-10
    "UH", "UW",  # 11-12
    
    # Vowels (long/stressed) - with stress markers
    "AA", "AO",  # 13-14
    "AE1", "AH1", "AW1", "AY1",  # 15-18
    "EH1", "ER1", "EY1",  # 19-21
    "IH1", "IY1",  # 22-23
    "OW1", "OY1",  # 24-25
    "UH1", "UW1",  # 26-27
    "AA1", "AO1",  # 28-29
    
    # Vowels (secondary stress)
    "AE2", "AH2", "AW2", "AY2",  # 30-33
    "EH2", "ER2", "EY2",  # 34-36
    "IH2", "IY2",  # 37-38
    "OW2", "OY2",  # 39-40
    "UH2", "UW2",  # 41-42
    "AA2", "AO2",  # 43-44
    
    # Consonants (stops)
    "B", "D", "G",  # 45-47
    "P", "T", "K",  # 48-50
    
    # Consonants (fricatives)
    "F", "V",  # 51-52
    "TH", "DH",  # 53-54
    "S", "Z",  # 55-56
    "SH", "ZH",  # 57-58
    "HH",  # 59
    
    # Consonants (affricates)
    "CH", "JH",  # 60-61
    
    # Consonants (nasals)
    "M", "N", "NG",  # 62-64
    
    # Consonants (liquids)
    "L", "R",  # 65-66
    
    # Consonants (glides)
    "Y", "W",  # 67-68
    
    # Consonants (other)
    "Q",  # 69
    
    # Special tokens
    "<UNK>",  # 70 - unknown phoneme
    "<SIL>",  # 71 - silence
    "<PAD>",  # 72 - padding
    "<BOS>",  # 73 - beginning of sequence
    "<EOS>",  # 74 - end of sequence
]


class PhonemeTokenizer:
    """
    Tokenizer for phoneme symbols.
    
    Converts between phoneme symbols (e.g., "W", "IY0") and token IDs
    suitable for training with CTC loss.
    """
    
    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize phoneme tokenizer.
        
        Args:
            vocab: Optional custom phoneme vocabulary. If None, uses ARPABET_VOCAB.
        """
        self.vocab = vocab or ARPABET_VOCAB.copy()
        
        # Create mappings
        self.token2id = {phone: idx for idx, phone in enumerate(self.vocab)}
        self.id2token = {idx: phone for idx, phone in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        self.unk_id = self.token2id.get("<UNK>", 0)
        self.pad_id = self.token2id.get("<PAD>", 0)
        self.sil_id = self.token2id.get("<SIL>", 0)
        self.bos_id = self.token2id.get("<BOS>", 0)
        self.eos_id = self.token2id.get("<EOS>", 0)
        
        logger.info(f"PhonemeTokenizer initialized with {self.vocab_size} tokens")
    
    def encode(self, phonemes: List[str]) -> List[int]:
        """
        Encode list of phoneme symbols to token IDs.
        
        Args:
            phonemes: List of phoneme symbols (e.g., ["W", "IY0"])
            
        Returns:
            List of token IDs
            
        Example:
            >>> tokenizer = PhonemeTokenizer()
            >>> tokenizer.encode(["W", "IY0"])
            [68, 8]
        """
        return [self.token2id.get(p, self.unk_id) for p in phonemes]
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> List[str]:
        """
        Decode token IDs back to phoneme symbols.
        
        Args:
            token_ids: List of token IDs
            skip_special: If True, skip special tokens like <PAD>, <UNK>, etc.
            
        Returns:
            List of phoneme symbols
            
        Example:
            >>> tokenizer = PhonemeTokenizer()
            >>> tokenizer.decode([68, 8])
            ['W', 'IY0']
        """
        special_tokens = {"<UNK>", "<SIL>", "<PAD>", "<BOS>", "<EOS>"}
        phonemes = []
        
        for token_id in token_ids:
            phone = self.id2token.get(token_id, "<UNK>")
            if skip_special and phone in special_tokens:
                continue
            phonemes.append(phone)
        
        return phonemes
    
    def decode_with_special(self, token_ids: List[int]) -> List[str]:
        """
        Decode token IDs including special tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of phoneme symbols (including special tokens)
        """
        return [self.id2token.get(token_id, "<UNK>") for token_id in token_ids]
    
    def remove_duplicates_and_blanks(self, token_ids: List[int]) -> List[int]:
        """
        Remove consecutive duplicate tokens and blank tokens (CTC post-processing).
        
        This is useful for post-processing CTC model outputs where consecutive
        identical predictions should collapse to a single token.
        
        Args:
            token_ids: List of token IDs (possibly with consecutive duplicates)
            
        Returns:
            Deduplicated list of token IDs
            
        Example:
            >>> tokenizer = PhonemeTokenizer()
            >>> tokenizer.remove_duplicates_and_blanks([68, 68, 8, 8, 0])
            [68, 8]
        """
        if not token_ids:
            return []
        
        deduplicated = [token_ids[0]]
        for token_id in token_ids[1:]:
            # Skip if same as previous token
            if token_id != deduplicated[-1]:
                deduplicated.append(token_id)
        
        return deduplicated
    
    def collapse_repeated(self, token_ids: List[int]) -> List[int]:
        """
        Collapse repeated tokens while keeping first occurrence (for CTC decoding).
        
        Alias for remove_duplicates_and_blanks for clarity.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Collapsed list of token IDs
        """
        return self.remove_duplicates_and_blanks(token_ids)
    
    def save(self, path: Union[str, Path]):
        """
        Save tokenizer vocabulary to JSON file.
        
        Args:
            path: Path to save vocabulary
        """
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        
        config = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "unk": "<UNK>",
                "pad": "<PAD>",
                "sil": "<SIL>",
                "bos": "<BOS>",
                "eos": "<EOS>"
            }
        }
        
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"PhonemeTokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "PhonemeTokenizer":
        """
        Load tokenizer vocabulary from JSON file.
        
        Args:
            path: Path to load vocabulary from
            
        Returns:
            PhonemeTokenizer instance
        """
        path = Path(path)
        
        with open(path, "r") as f:
            config = json.load(f)
        
        vocab = config.get("vocab", ARPABET_VOCAB)
        logger.info(f"PhonemeTokenizer loaded from {path}")
        
        return cls(vocab=vocab)
    
    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size
    
    def __repr__(self):
        """Return string representation."""
        return f"PhonemeTokenizer(vocab_size={self.vocab_size})"


# Convenience functions
def get_default_tokenizer() -> PhonemeTokenizer:
    """Get default ARPABET phoneme tokenizer."""
    return PhonemeTokenizer()


def encode_phonemes(phonemes: List[str]) -> List[int]:
    """
    Quick encoding with default tokenizer.
    
    Args:
        phonemes: List of phoneme symbols
        
    Returns:
        List of token IDs
    """
    tokenizer = get_default_tokenizer()
    return tokenizer.encode(phonemes)


def decode_phonemes(token_ids: List[int]) -> List[str]:
    """
    Quick decoding with default tokenizer.
    
    Args:
        token_ids: List of token IDs
        
    Returns:
        List of phoneme symbols
    """
    tokenizer = get_default_tokenizer()
    return tokenizer.decode(token_ids)
