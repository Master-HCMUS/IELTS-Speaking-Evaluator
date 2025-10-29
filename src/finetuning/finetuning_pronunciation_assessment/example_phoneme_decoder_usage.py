"""
Complete Example: Training and Using the Phoneme Decoder

This example demonstrates the full pipeline for training and inference
with the CTC-based phoneme decoder.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from phoneme_tokenizer import PhonemeTokenizer
from data_processor import SpeechOcean762DataProcessor
from data_collator import PronunciationAssessmentDataCollator
from whisper_pronunciation_model import WhisperPronunciationAssessmentModel
from trainer import PronunciationAssessmentTrainer
from training_config import PronunciationTrainingConfig
from torch.utils.data import DataLoader


# ============================================================================
# PART 1: BASIC TOKENIZER USAGE
# ============================================================================

def example_tokenizer_usage():
    """Example: Using the phoneme tokenizer."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Phoneme Tokenizer Usage")
    print("="*70)
    
    tokenizer = PhonemeTokenizer()
    
    # Example 1: Encode phonemes to IDs
    phonemes = ["W", "IY0"]
    token_ids = tokenizer.encode(phonemes)
    print(f"Phonemes: {phonemes}")
    print(f"Token IDs: {token_ids}")
    
    # Example 2: Decode IDs back to phonemes
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")
    
    # Example 3: CTC post-processing (collapse repeated tokens)
    ctc_output = [68, 68, 8, 8, 0]  # 0 is blank
    collapsed = tokenizer.collapse_repeated(ctc_output)
    print(f"\nCTC Output (with repeats): {ctc_output}")
    print(f"After Collapse: {collapsed}")
    print(f"Phonemes: {tokenizer.decode(collapsed)}")
    
    # Example 4: Check vocabulary
    print(f"\nVocabulary size: {len(tokenizer)}")
    print(f"Unknown token ID: {tokenizer.unk_id}")
    print(f"Padding token ID: {tokenizer.pad_id}")


# ============================================================================
# PART 2: MODEL INITIALIZATION WITH PHONEME DECODER
# ============================================================================

def example_model_initialization():
    """Example: Initialize model with phoneme decoder enabled."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Model Initialization")
    print("="*70)
    
    # Create model with phoneme decoder enabled
    model = WhisperPronunciationAssessmentModel(
        model_name="openai/whisper-base",
        train_word_level=True,
        train_phone_level=True,
        train_utterance_level=True,
        train_transcription=True,
        train_phoneme_symbols=True,  # ← ENABLE PHONEME DECODER
        num_phonemes=75,              # ARPABET vocabulary
        freeze_encoder=False,
        freeze_decoder=False
    )
    
    print(f"Model created: {model}")
    print(f"Phoneme decoder enabled: {model.train_phoneme_symbols}")
    print(f"Number of phoneme classes: {model.num_phonemes}")
    
    # Initialize the model (lazy loading)
    model._initialize_model()
    print(f"Model initialized. Has phoneme_decoder: {hasattr(model, 'phoneme_decoder')}")
    
    return model


# ============================================================================
# PART 3: DATA PROCESSING
# ============================================================================

def example_data_processing():
    """Example: Data processing with phoneme extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Data Processing with Phoneme Extraction")
    print("="*70)
    
    # Initialize processor
    processor = SpeechOcean762DataProcessor()
    
    # Load small sample for demonstration
    try:
        dataset_dict = processor.prepare_for_training(
            split_ratio=0.8,
            sample_size=5  # Just 5 samples for demo
        )
        
        # Check first example
        example = dataset_dict["train"][0]
        print(f"Example keys: {example.keys()}")
        
        if "phoneme_ids" in example:
            print(f"Phoneme IDs: {example['phoneme_ids']}")
            print(f"Phoneme sequence length: {example.get('phoneme_sequence_length')}")
            
            # Decode to see phonemes
            tokenizer = PhonemeTokenizer()
            phonemes = tokenizer.decode(example["phoneme_ids"].tolist())
            print(f"Decoded phonemes: {phonemes}")
        
        print(f"Input features shape: {example['input_features'].shape}")
        print(f"Has accuracy: {'accuracy' in example}")
        
    except Exception as e:
        print(f"Note: Could not load full dataset. Error: {e}")
        print("This is expected in quick demo mode. Full training will use real data.")


# ============================================================================
# PART 4: DATA COLLATOR
# ============================================================================

def example_data_collator():
    """Example: Using data collator for batching."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Data Collator for Batching")
    print("="*70)
    
    collator = PronunciationAssessmentDataCollator()
    
    # Create dummy batch (simulating 2 examples)
    batch = [
        {
            "input_features": np.random.randn(80, 3000).astype(np.float32),
            "phoneme_ids": np.array([68, 8], dtype=np.int32),
            "accuracy": 0.8,
        },
        {
            "input_features": np.random.randn(80, 3000).astype(np.float32),
            "phoneme_ids": np.array([68, 8, 5], dtype=np.int32),  # Longer sequence
            "accuracy": 0.9,
        }
    ]
    
    # Collate batch
    collated = collator(batch)
    
    print(f"Collated batch keys: {collated.keys()}")
    print(f"Input features shape: {collated['input_features'].shape}")
    
    if "phoneme_ids" in collated:
        print(f"Phoneme IDs shape (padded): {collated['phoneme_ids'].shape}")
        print(f"Phoneme sequence lengths: {collated['phoneme_sequence_lengths']}")
        print(f"Input lengths (for CTC): {collated['input_lengths']}")


# ============================================================================
# PART 5: INFERENCE / PREDICTION
# ============================================================================

def example_inference():
    """Example: Using model for inference."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Model Inference and Phone Prediction")
    print("="*70)
    
    # Initialize model
    model = WhisperPronunciationAssessmentModel(
        train_phoneme_symbols=True,
        num_phonemes=75
    )
    model.eval()
    
    # Create dummy input
    batch_size = 1
    input_features = torch.randn(batch_size, 80, 3000)
    
    # Forward pass (no labels during inference)
    with torch.no_grad():
        predictions = model(input_features)
    
    print(f"Prediction keys: {predictions.keys()}")
    
    # Extract phoneme predictions
    if "phoneme_logits" in predictions:
        phoneme_logits = predictions['phoneme_logits']
        print(f"Phoneme logits shape: {phoneme_logits.shape}")
        
        # Greedy decoding
        tokenizer = PhonemeTokenizer()
        phoneme_tokens = phoneme_logits.argmax(dim=-1)[0]  # [1500]
        phoneme_ids = tokenizer.collapse_repeated(phoneme_tokens.cpu().numpy().tolist())
        phonemes = tokenizer.decode(phoneme_ids)
        
        print(f"Predicted phoneme IDs (first 10): {phoneme_ids[:10]}")
        print(f"Predicted phonemes (first 10): {phonemes[:10]}")


# ============================================================================
# PART 6: LOSS COMPUTATION
# ============================================================================

def example_loss_computation():
    """Example: Computing phoneme loss during training."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Loss Computation with Phoneme Decoder")
    print("="*70)
    
    model = WhisperPronunciationAssessmentModel(
        train_phoneme_symbols=True
    )
    
    # Create dummy batch
    batch_size = 2
    input_features = torch.randn(batch_size, 80, 3000)
    
    # Dummy labels and phoneme IDs
    labels = torch.randint(0, 50257, (batch_size, 10))
    phoneme_ids = torch.randint(0, 75, (batch_size, 5))
    input_lengths = torch.tensor([1500, 1500], dtype=torch.long)
    phoneme_lengths = torch.tensor([5, 5], dtype=torch.long)
    
    # Forward pass
    predictions = model(
        input_features,
        decoder_input_ids=labels,
        phoneme_ids=phoneme_ids,
        input_lengths=input_lengths,
        phoneme_sequence_lengths=phoneme_lengths
    )
    
    print(f"Predictions keys: {predictions.keys()}")
    
    if "phoneme_loss" in predictions:
        phoneme_loss = predictions["phoneme_loss"]
        print(f"Phoneme loss: {phoneme_loss.item():.4f}")
        
        # In actual training, this would be weighted and added to total loss
        weight = 0.5  # Default weight
        weighted_loss = phoneme_loss * weight
        print(f"Weighted phoneme loss (weight={weight}): {weighted_loss.item():.4f}")


# ============================================================================
# PART 7: COMPLETE TRAINING LOOP (MINIMAL)
# ============================================================================

def example_minimal_training():
    """Example: Minimal training loop."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Minimal Training Loop")
    print("="*70)
    
    print("Note: This is a code template. Full training requires actual data.")
    
    code = """
# Configuration
config = PronunciationTrainingConfig(
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-4,
    train_phoneme_symbols=True,
    loss_weights={
        "phoneme": 0.5,  # Phoneme decoder weight
        ...
    }
)

# Initialize model
model = WhisperPronunciationAssessmentModel(
    train_phoneme_symbols=True,
    num_phonemes=75
)

# Prepare data
processor = SpeechOcean762DataProcessor()
dataset_dict = processor.prepare_for_training(sample_size=1000)

# Create data loaders
collator = PronunciationAssessmentDataCollator()
train_loader = DataLoader(
    dataset_dict["train"],
    batch_size=config.batch_size,
    collate_fn=collator
)

# Initialize trainer
trainer = PronunciationAssessmentTrainer(
    model=model,
    config=config
)

# Setup optimization
total_steps = len(train_loader) * config.num_epochs
trainer.setup_optimization(total_steps)

# Training loop
for epoch in range(config.num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
    
    # Save checkpoint
    trainer.save_model(f"checkpoints/epoch_{epoch}")
"""
    print(code)


# ============================================================================
# PART 8: COMPLETE PREDICTION PIPELINE
# ============================================================================

def example_complete_prediction():
    """Example: Complete prediction pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Complete Prediction Pipeline")
    print("="*70)
    
    code = """
import librosa
import torch

# Load and process audio
audio_file = "sample.wav"
audio, sr = librosa.load(audio_file, sr=16000)

# Extract mel-spectrogram (this is what the model expects)
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=80,
    n_fft=400,
    hop_length=160
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Pad to 3000 timesteps
mel_spec_db = np.pad(
    mel_spec_db,
    ((0, 0), (0, max(0, 3000 - mel_spec_db.shape[1]))),
    mode='constant'
)[:, :3000]

# Convert to torch
input_features = torch.from_numpy(mel_spec_db).float().unsqueeze(0)

# Get predictions
model.eval()
tokenizer = PhonemeTokenizer()

with torch.no_grad():
    predictions = model(input_features)

# Extract all results
result = {
    "phones": tokenizer.decode(phoneme_ids),
    "phone_accuracy": predictions['phone_accuracy_logits'][0].cpu().numpy(),
    "utterance_accuracy": predictions['utterance_accuracy_logits'][0].item(),
}

print(f"Phones: {result['phones']}")
print(f"Utterance accuracy: {result['utterance_accuracy']:.2%}")
"""
    print(code)


# ============================================================================
# MAIN: RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CTC PHONEME DECODER - COMPLETE USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_tokenizer_usage()
    example_model_initialization()
    example_data_processing()
    example_data_collator()
    example_inference()
    example_loss_computation()
    example_minimal_training()
    example_complete_prediction()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nFor full training and evaluation, see:")
    print("  - CTC_PHONEME_DECODER_IMPLEMENTATION.md")
    print("  - run_pronunciation_training.py")
