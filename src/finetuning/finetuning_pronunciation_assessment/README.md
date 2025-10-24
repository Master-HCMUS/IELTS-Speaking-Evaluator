# Whisper Pronunciation Assessment Fine-tuning

This module implements a comprehensive approach to fine-tune Whisper models for pronunciation assessment while maintaining transcription capabilities. It supports multi-granularity assessment (word-level, phone-level, and utterance-level) using multi-objective training.

## Overview

The approach keeps the full Whisper encoder-decoder architecture and adds assessment heads to predict pronunciation scores at multiple granularities:

- **Word-level**: Accuracy, stress, and total scores per word
- **Phone-level**: Accuracy scores per phoneme
- **Utterance-level**: Overall accuracy, fluency, prosodic, completeness, and total scores

### Key Features

- 🎯 **Multi-granularity Assessment**: Word, phone, and utterance-level scoring
- 🔄 **Dual Training Objectives**: ASR (transcription) + pronunciation assessment
- 📊 **Flexible Loss Weighting**: Configurable weights for different objectives
- 🎛️ **Multiple Training Modes**: Quick test, development, production, and specialized configs
- 📈 **Comprehensive Evaluation**: Detailed metrics and sample predictions
- 🔧 **Modular Design**: Easy to extend and customize

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Whisper Model                             │
├─────────────────────────────────────────────────────────────┤
│  Audio Input → Encoder → Decoder → Transcription Output     │
│                    ↓                                        │
│               Assessment Heads                               │
│              ┌─────────────────┐                           │
│              │   Word-level    │ → Accuracy, Stress, Total  │
│              │   Phone-level   │ → Accuracy                 │
│              │ Utterance-level │ → Acc, Fluency, Prosodic,  │
│              │                 │   Completeness, Total      │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Dataset Format

The training uses SpeechOcean762 dataset with detailed annotations:

```json
{
  "audio": {"array": [...], "sampling_rate": 16000},
  "text": "WE CALL IT BEAR",
  "accuracy": 8,
  "fluency": 9, 
  "prosodic": 8,
  "completeness": 10,
  "total": 8,
  "words": [
    {
      "text": "WE",
      "accuracy": 10,
      "stress": 10,
      "total": 10,
      "phones": ["W", "IY0"],
      "phones-accuracy": [2, 2]
    },
    {
      "text": "CALL", 
      "accuracy": 10,
      "stress": 10,
      "total": 10,
      "phones": ["K", "AO0", "L"],
      "phones-accuracy": [2, 1.8, 1.8]
    }
  ],
  "speaker": "speaker_id",
  "gender": "M",
  "age": 25
}
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Quick test (recommended first)
python run_pronunciation_training.py --quick-test

# Development training
python run_pronunciation_training.py --development

# Production training
python run_pronunciation_training.py --production

# Interactive mode
python run_pronunciation_training.py
```

### 3. Use Trained Model

```python
from finetuning.finetuning_pronunciation_assessment import PronunciationAssessmentTrainer

# Load trained model
model = PronunciationAssessmentTrainer.load_trained_model("models/pronunciation_development")

# Predict pronunciation scores
scores = model.predict_pronunciation_scores(audio_features)

# Generate transcription
transcription_ids = model.generate_transcription(audio_features)
```

## Training Modes

### 1. Quick Test (`--quick-test`)
- **Purpose**: Test setup and implementation
- **Data**: 100 train, 50 eval samples
- **Duration**: ~5-10 minutes
- **Granularities**: Word + utterance level only

### 2. Development (`--development`)
- **Purpose**: Development and experimentation
- **Data**: 1000 train, 200 eval samples
- **Duration**: ~1-2 hours
- **Granularities**: All levels (word, phone, utterance)

### 3. Production (`--production`)
- **Purpose**: Best performance model
- **Data**: Full dataset
- **Duration**: ~6-12 hours
- **Model**: whisper-small (better than tiny)
- **Granularities**: All levels with optimized loss weights

### 4. Specialized Modes

**Transcription-Only (`--transcription-only`)**
- ASR baseline without assessment training
- Useful for comparison

**Assessment-Only (`--assessment-only`)**
- No transcription training, only assessment
- Freezes more Whisper layers

**Phone-Focused (`--phone-focused`)**
- Emphasizes phone-level assessment
- Higher loss weights for phone-level losses

## Configuration

### Loss Weights

```python
loss_weights = {
    'asr': 1.0,                    # Transcription loss
    'word_accuracy': 1.0,          # Word-level accuracy
    'word_stress': 0.5,            # Word-level stress
    'word_total': 1.0,             # Word-level total
    'phone_accuracy': 1.0,         # Phone-level accuracy
    'utterance_accuracy': 1.0,     # Utterance accuracy
    'utterance_fluency': 1.0,      # Utterance fluency
    'utterance_prosodic': 1.0,     # Utterance prosodic
    'utterance_completeness': 0.1, # Lower (99.6% is 10)
    'utterance_total': 1.0         # Utterance total
}
```

### Custom Configuration

```python
from finetuning.finetuning_pronunciation_assessment import PronunciationTrainingConfig

config = PronunciationTrainingConfig(
    whisper_model_name="openai/whisper-tiny",
    output_dir="custom_model",
    batch_size=8,
    num_epochs=3,
    learning_rate=1e-5,
    
    # Training granularities
    train_word_level=True,
    train_phone_level=True,
    train_utterance_level=True,
    include_transcription=True,
    
    # Custom loss weights
    loss_weights={
        'asr': 0.8,
        'phone_accuracy': 1.5,  # Emphasize phones
        'word_accuracy': 1.0,
        'utterance_total': 1.2
    }
)
```

## Model Outputs

### Assessment Predictions

```python
assessment_predictions = {
    'word_level': {
        'accuracy': tensor([10, 10, 6]),     # Per word
        'stress': tensor([10, 10, 10]),      # Per word
        'total': tensor([10, 10, 6])         # Per word
    },
    'phone_level': {
        'accuracy': tensor([2, 2, 2, 1.8, 1.8, 2, 1, 1])  # Per phone
    },
    'utterance_level': {
        'accuracy': tensor([8.5]),           # Single utterance score
        'fluency': tensor([9.2]),
        'prosodic': tensor([8.0]),
        'completeness': tensor([10.0]),
        'total': tensor([8.3])
    }
}
```

### Training Outputs

```python
training_outputs = {
    'loss': total_weighted_loss,
    'losses': {
        'asr': asr_loss,
        'word_accuracy': word_acc_loss,
        'phone_accuracy': phone_acc_loss,
        'utterance_total': utterance_total_loss,
        # ... other component losses
    },
    'logits': whisper_transcription_logits,
    'assessment_predictions': assessment_predictions
}
```

## Evaluation Metrics

The training tracks multiple metrics:

- **ASR Metrics**: Standard transcription quality
- **Assessment MSE**: Mean squared error for each score type
- **Assessment MAE**: Mean absolute error for each score type
- **Component Losses**: Individual loss values for each objective

## Integration with Main System

### Loading in Transcription Service

```python
from src.local_transcription_service import LocalWhisperTranscriptionService

# This won't work directly since the model architecture is different
# Instead, use the pronunciation assessment model:

from finetuning.finetuning_pronunciation_assessment import PronunciationAssessmentTrainer

model = PronunciationAssessmentTrainer.load_trained_model(
    "src/finetuning/finetuning_pronunciation_assessment/models/pronunciation_development"
)

# Get transcription
transcription_ids = model.generate_transcription(audio_features)
transcription = processor.decode(transcription_ids[0], skip_special_tokens=True)

# Get pronunciation scores
scores = model.predict_pronunciation_scores(audio_features)
```

### Creating a Pronunciation Service

```python
import torch
import librosa
from transformers import WhisperProcessor

class PronunciationAssessmentService:
    def __init__(self, model_path):
        self.model = PronunciationAssessmentTrainer.load_trained_model(model_path)
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model.eval()
    
    def assess_pronunciation(self, audio_file):
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Process audio
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            # Get transcription
            transcription_ids = self.model.generate_transcription(inputs["input_features"])
            transcription = self.processor.decode(transcription_ids[0], skip_special_tokens=True)
            
            # Get pronunciation scores
            scores = self.model.predict_pronunciation_scores(inputs["input_features"])
        
        return {
            "transcription": transcription,
            "pronunciation_scores": scores
        }
```

## Performance Notes

### Computational Requirements

- **Quick Test**: 2-4GB GPU memory, ~10 minutes
- **Development**: 6-8GB GPU memory, ~1-2 hours
- **Production**: 12GB+ GPU memory, ~6-12 hours

### Model Sizes

- **whisper-tiny**: 39M parameters + ~2M assessment parameters
- **whisper-small**: 244M parameters + ~6M assessment parameters
- **whisper-base**: 74M parameters + ~3M assessment parameters

### Expected Performance

Based on the multi-objective training:

- **Transcription**: Slightly lower than pure ASR (due to shared capacity)
- **Word-level Assessment**: High correlation with human scores
- **Phone-level Assessment**: Improved granularity for detailed analysis
- **Utterance-level Assessment**: Good overall scoring capability

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` 
   - Use `gradient_accumulation_steps`
   - Try smaller model (whisper-tiny)

2. **Poor Assessment Performance**
   - Adjust loss weights in config
   - Increase training data (`max_train_samples=None`)
   - Try assessment-only training mode

3. **Transcription Degradation**
   - Increase ASR loss weight
   - Reduce assessment loss weights
   - Use transcription-only baseline for comparison

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Also available in config
config.logging_steps = 10  # More frequent logging
```

## Advanced Features

### Custom Loss Functions

You can extend the model with custom loss functions:

```python
def custom_pronunciation_loss(predictions, targets):
    # Custom loss logic
    return loss

# Modify the model's _compute_assessment_losses method
```

### Attention Visualization

```python
# Get attention weights during inference
outputs = model(input_features, output_attentions=True)
attention_weights = outputs.attentions
```

### Phone Alignment

For better phone-level assessment, consider using forced alignment tools:

```bash
# Install Montreal Forced Alignment (optional)
conda install -c conda-forge montreal-forced-alignment
```

## Citation

If you use this pronunciation assessment approach, please cite:

```bibtex
@software{whisper_pronunciation_assessment,
  title={Whisper Pronunciation Assessment Fine-tuning},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

## Contributing

1. Follow the established code structure
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Consider backward compatibility

## License

This module follows the same license as the main project.