# Fine-tuning Package for Whisper Models

This package provides different approaches for fine-tuning Whisper models for speech processing tasks.

## Structure

```
src/finetuning/
├── __init__.py                 # Main package initialization
├── convert_pth_model.py        # Model conversion utilities
├── extended_local_service.py   # Extended local transcription service
├── finetuning_STT/            # Speech-to-Text fine-tuning module
│   ├── __init__.py
│   ├── data_processor.py      # SpeechOcean762 dataset processing
│   ├── README.md              # STT fine-tuning documentation
│   ├── requirements.txt       # STT-specific dependencies
│   ├── run_finetuning.py     # Main script to run STT fine-tuning
│   ├── training_config.py    # Training configuration management
│   ├── whisper_finetuner.py  # Core fine-tuning implementation
│   └── models/               # Trained models directory
│       ├── whisper_development/
│       └── whisper_finetune_final/
└── (future modules for other fine-tuning approaches)
```

## Available Approaches

### 1. Speech-to-Text (STT) Fine-tuning

Located in `finetuning_STT/` directory.

**Purpose**: Improve Whisper's transcription accuracy on speech with varied pronunciation quality.

**Approach**: 
- Uses standard Whisper encoder-decoder architecture
- Trains on (audio, text) pairs from SpeechOcean762 dataset
- Maintains text generation capabilities

**Usage**:
```bash
# Quick test
python src/finetuning/finetuning_STT/run_finetuning.py --quick-test

# Development training
python src/finetuning/finetuning_STT/run_finetuning.py --development

# Production training
python src/finetuning/finetuning_STT/run_finetuning.py --production
```

**Key Features**:
- Multiple preset configurations (quick test, development, production)
- Comprehensive evaluation metrics (WER, BLEU, character accuracy)
- TensorBoard logging
- Model checkpoint management
- Pronunciation score integration for analysis

### 2. Future Approaches

The structure allows for additional fine-tuning approaches:

- **Pronunciation Assessment Fine-tuning**: Custom regression models for scoring
- **Multi-task Learning**: Combined transcription and pronunciation assessment
- **Domain-specific Fine-tuning**: Specialized models for different contexts

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r finetuning_STT/requirements.txt
   ```

2. **Run STT Fine-tuning**:
   ```bash
   cd src/finetuning/finetuning_STT
   python run_finetuning.py
   ```

3. **Use Trained Models**:
   ```python
   from finetuning.finetuning_STT import WhisperFineTuner
   
   model, processor = WhisperFineTuner.load_trained_model("path/to/model")
   ```

## Integration with Main System

The fine-tuned models can be integrated with the main IELTS Speaking Evaluation system:

```python
from src.local_transcription_service import LocalWhisperTranscriptionService

# Load fine-tuned model
service = LocalWhisperTranscriptionService(
    model_path="src/finetuning/finetuning_STT/models/whisper_development"
)

# Use for transcription
result = service.transcribe_audio_file("audio.wav")
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.21+
- Datasets 2.0+
- librosa 0.9+
- CUDA (optional, for GPU acceleration)

## Contributing

When adding new fine-tuning approaches:

1. Create a new subdirectory under `finetuning/`
2. Include comprehensive documentation
3. Follow the established patterns for configuration and training
4. Update this README with the new approach