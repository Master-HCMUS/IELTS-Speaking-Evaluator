# Whisper Fine-tuning on SpeechOcean762 Dataset

This module provides comprehensive functionality to fine-tune OpenAI Whisper models on the SpeechOcean762 dataset for improved speech recognition and pronunciation assessment capabilities.

## Overview

The fine-tuning system consists of several key components:

- **TrainingConfig**: Comprehensive configuration management with preset options
- **SpeechOcean762DataProcessor**: Complete data processing pipeline for the dataset
- **WhisperFineTuner**: Main fine-tuning class with training loop and evaluation
- **PronunciationMetricsCallback**: Custom callback for pronunciation-specific metrics

## Features

- 🎯 **Multiple Training Presets**: Quick test, development, and production configurations
- 📊 **Comprehensive Evaluation**: WER, BLEU, character accuracy, and pronunciation scores
- 🔧 **Flexible Configuration**: Easy customization of all training parameters
- 📈 **Progress Tracking**: TensorBoard logging and custom metrics
- 💾 **Model Management**: Automatic saving, loading, and checkpoint management
- 🎵 **Audio Processing**: Advanced audio preprocessing with librosa

## Quick Start

### 1. Install Dependencies

```bash
# Install fine-tuning specific dependencies
pip install -r finetuning/requirements.txt
```

### 2. Run Fine-tuning

```bash
# Simple interactive script
python finetuning/run_finetuning.py
```

Choose from:
- **Quick Test**: 10 samples, 1 epoch (for testing setup)
- **Development**: 1000 samples, 3 epochs (for development)
- **Custom**: 1000 samples, 5 epochs (production-ready)

### 3. Command Line Usage

```bash
# Quick test
python finetuning/whisper_finetuner.py --quick-test

# Custom training
python finetuning/whisper_finetuner.py \
    --model-name openai/whisper-tiny \
    --output-dir models/whisper_custom \
    --batch-size 16 \
    --num-epochs 5 \
    --learning-rate 1e-5 \
    --max-train-samples 1000 \
    --max-eval-samples 200

# Evaluation only
python finetuning/whisper_finetuner.py --eval-only --output-dir models/whisper_custom
```

## Configuration Options

### Preset Configurations

```python
from finetuning.training_config import get_quick_test_config, get_development_config

# Quick test (minimal data for testing)
config = get_quick_test_config()

# Development (moderate data for experimentation)  
config = get_development_config()
```

### Custom Configuration

```python
from finetuning.training_config import TrainingConfig

config = TrainingConfig(
    model_name="openai/whisper-tiny",
    output_dir="models/whisper_custom",
    batch_size=16,
    num_epochs=5,
    learning_rate=1e-5,
    max_train_samples=1000,
    max_eval_samples=200,
    warmup_steps=500,
    save_steps=100,
    eval_steps=100
)
```

## Programming Interface

### Basic Usage

```python
from finetuning.training_config import TrainingConfig
from finetuning.whisper_finetuner import WhisperFineTuner

# Create configuration
config = TrainingConfig(
    model_name="openai/whisper-tiny",
    output_dir="models/whisper_finetuned"
)

# Initialize fine-tuner
fine_tuner = WhisperFineTuner(config)

# Run training
results = fine_tuner.train()

print(f"Training completed! WER: {results['eval_results']['eval_wer']:.4f}")
```

### Advanced Usage

```python
# Load and evaluate existing model
model, processor = WhisperFineTuner.load_trained_model("models/whisper_finetuned")

# Custom evaluation
fine_tuner = WhisperFineTuner(config)
eval_results = fine_tuner.evaluate_model("models/whisper_finetuned")
```

## Training Process

The fine-tuning process includes:

1. **Model Loading**: Load pretrained Whisper model and processor
2. **Data Preparation**: Process SpeechOcean762 dataset with audio preprocessing
3. **Training Setup**: Configure trainer with custom metrics and callbacks
4. **Training Loop**: Fine-tune with evaluation at specified intervals
5. **Model Saving**: Save final model, processor, and training summary
6. **Final Evaluation**: Comprehensive evaluation with sample predictions

## Evaluation Metrics

- **WER (Word Error Rate)**: Lower is better
- **BLEU Score**: Higher is better (0-1 scale)
- **Character Accuracy**: Higher is better (0-1 scale)
- **Pronunciation Scores**: From SpeechOcean762 dataset

## Output Structure

```
models/whisper_finetuned/
├── config.json                 # Model configuration
├── pytorch_model.bin          # Fine-tuned model weights
├── preprocessor_config.json   # Audio preprocessor config
├── tokenizer.json            # Tokenizer files
├── training_config.json      # Training configuration
├── training_summary.json     # Complete training results
├── evaluation_results.json   # Detailed evaluation metrics
├── pronunciation_metrics.json # Custom pronunciation metrics
└── logs/
    └── training_*.log        # Training logs
```

## Integration with Main System

The fine-tuned models can be integrated with the existing pronunciation assessment system:

```python
# Load fine-tuned model in pronunciation service
from finetuning.whisper_finetuner import WhisperFineTuner

model, processor = WhisperFineTuner.load_trained_model("models/whisper_finetuned")

# Use in pronunciation assessment
# (Integration code would be added to pronunciation_service.py)
```

## Performance Notes

- **Quick Test**: ~2-5 minutes (testing setup)
- **Development**: ~30-60 minutes (moderate training)
- **Custom/Production**: 2-4 hours (full training)

GPU usage is recommended for faster training. The system automatically detects and uses CUDA if available.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `max_audio_length`
2. **Slow Training**: Enable `fp16=True` and use GPU
3. **Poor Performance**: Increase `max_train_samples` or `num_epochs`
4. **Audio Issues**: Check `sampling_rate` matches dataset (16kHz)

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Features

### Custom Data Collator

The system includes a custom data collator that handles:
- Audio feature extraction
- Text tokenization
- Pronunciation score integration
- Batch padding and normalization

### Pronunciation-Aware Training

The training process incorporates pronunciation scores from SpeechOcean762:
- Accuracy scores (0-10 scale)
- Fluency scores (0-10 scale)  
- Completeness scores (0-10 scale)
- Prosodic scores (0-10 scale)

### TensorBoard Integration

Monitor training progress:

```bash
tensorboard --logdir models/whisper_finetuned/runs
```

## Model Variants

Supported Whisper models:
- `openai/whisper-tiny` (39M parameters)
- `openai/whisper-base` (74M parameters)
- `openai/whisper-small` (244M parameters)
- `openai/whisper-medium` (769M parameters)
- `openai/whisper-large-v2` (1550M parameters)

Larger models generally provide better performance but require more computational resources.