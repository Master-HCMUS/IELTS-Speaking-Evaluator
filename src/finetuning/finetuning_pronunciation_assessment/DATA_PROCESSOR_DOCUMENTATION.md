# Pronunciation Assessment Data Processor

## Overview

The `data_processor.py` module provides comprehensive data processing utilities for training Whisper models with pronunciation assessment capabilities. It handles loading, preprocessing, and preparing the SpeechOcean762 dataset for multi-objective fine-tuning with scores at word-level, phone-level, and utterance-level granularities.

## Architecture

### Key Components

#### 1. **PronunciationAssessmentDataCollator**
Handles batching and collation of samples for training with pronunciation assessment scores.

**Responsibilities:**
- Pad input audio features using Whisper feature extractor
- Tokenize and pad transcription labels for ASR training
- Process and pad word-level pronunciation scores (accuracy, stress, total)
- Process and pad phone-level pronunciation scores (accuracy)
- Include utterance-level scores (accuracy, fluency, prosodic, completeness, total)
- Create attention masks for variable-length sequences

**Output Format:**
```python
{
    "input_features": torch.Tensor,      # [batch_size, 80, num_frames]
    "labels": torch.Tensor,              # [batch_size, seq_len] (ASR labels)
    "word_accuracy_scores": torch.Tensor,# [batch_size, max_words]
    "word_stress_scores": torch.Tensor,  # [batch_size, max_words]
    "word_total_scores": torch.Tensor,   # [batch_size, max_words]
    "word_mask": torch.Tensor,           # [batch_size, max_words]
    "phone_accuracy_scores": torch.Tensor,# [batch_size, max_phones]
    "phone_mask": torch.Tensor,          # [batch_size, max_phones]
    "accuracy": torch.Tensor,            # [batch_size]
    "fluency": torch.Tensor,             # [batch_size]
    "prosodic": torch.Tensor,            # [batch_size]
    "completeness": torch.Tensor,        # [batch_size]
    "total": torch.Tensor,               # [batch_size]
}
```

#### 2. **SpeechOcean762PronunciationProcessor**
Main processor for dataset loading and preparation.

**Key Methods:**

##### `__init__(whisper_model_name, sampling_rate, max_audio_length, normalize_audio)`
Initializes the processor with model-specific settings.

**Parameters:**
- `whisper_model_name`: Whisper model identifier (e.g., "openai/whisper-tiny")
- `sampling_rate`: Target audio sampling rate (default: 16000 Hz)
- `max_audio_length`: Maximum audio duration in seconds (default: 30.0)
- `normalize_audio`: Whether to normalize audio amplitude (default: True)

##### `load_dataset(splits, max_samples_per_split)`
Loads SpeechOcean762 dataset from Hugging Face Hub.

**Parameters:**
- `splits`: List of dataset splits to load (e.g., ["train", "test"])
- `max_samples_per_split`: Optional dict to limit samples per split for testing

**Returns:**
- `DatasetDict` with loaded splits

**Example:**
```python
processor = SpeechOcean762PronunciationProcessor(whisper_model_name="openai/whisper-base")
datasets = processor.load_dataset(
    splits=["train", "test"],
    max_samples_per_split={"train": 1000, "test": 500}
)
```

##### `preprocess_audio(audio_array, sampling_rate)`
Preprocesses audio for Whisper model ingestion.

**Operations:**
- Resampling to target sampling rate using librosa
- Audio normalization (if enabled)
- Trimming or zero-padding to max length

##### `_extract_pronunciation_scores(example)`
Extracts multi-level pronunciation scores from a dataset sample.

**Input Schema (from SpeechOcean762):**
```python
{
    "id": "000010011",
    "text": "WE CALL IT BEAR",
    "accuracy": 8,
    "completeness": 10,
    "fluency": 9,
    "prosodic": 9,
    "total": 8,
    "words": [
        {
            "text": "WE",
            "accuracy": 10,
            "total": 10,
            "stress": 10,
            "phones": ["W", "IY0"],
            "phones-accuracy": [2, 2]
        },
        ...
    ],
    "speaker": "...",
    "gender": "...",
    "age": 0
}
```

**Output:**
```python
{
    "accuracy": 0.8,              # Normalized to 0-1
    "completeness": 1.0,
    "fluency": 0.9,
    "prosodic": 0.9,
    "total": 0.8,
    "word_accuracy_scores": [1.0, ...],    # List of word-level scores
    "word_stress_scores": [1.0, ...],
    "word_total_scores": [1.0, ...],
    "phone_accuracy_scores": [1.0, ...],   # List of phone-level scores
    "speaker": "...",
    "gender": "...",
    "age": 0,
    "text": "WE CALL IT BEAR",
    "id": "000010011"
}
```

**Normalization:**
- Utterance scores: divided by 10 (0-10 scale → 0-1 range)
- Word scores: divided by 10
- Phone scores: divided by 2 (0-2 scale → 0-1 range)

##### `prepare_dataset_for_training(datasets, include_transcription)`
Full pipeline for dataset preparation.

**Process:**
1. Audio preprocessing (resampling, normalization)
2. Feature extraction using Whisper feature extractor
3. Transcription tokenization (if include_transcription=True)
4. Pronunciation score extraction at all granularities
5. Statistics computation

**Parameters:**
- `datasets`: Raw DatasetDict from load_dataset()
- `include_transcription`: Include ASR training objective (default: True)

**Output:**
- Processed DatasetDict with columns:
  - `input_features`: Whisper mel-spectrogram features
  - `labels`: Tokenized transcriptions (if include_transcription=True)
  - `word_accuracy_scores`, `word_stress_scores`, `word_total_scores`
  - `phone_accuracy_scores`
  - `accuracy`, `fluency`, `prosodic`, `completeness`, `total`
  - `speaker`, `gender`, `age`, `text`, `id`

##### `get_dataset_statistics(datasets)`
Computes and returns dataset statistics.

**Returns Statistics For:**
- Number of samples per split
- Mean, std, min, max for each score type (utterance-level)

**Example Output:**
```python
{
    "train": {
        "num_samples": 8000,
        "accuracy_mean": 0.745,
        "accuracy_std": 0.189,
        "accuracy_min": 0.0,
        "accuracy_max": 1.0,
        ...
    },
    "test": {...}
}
```

##### `create_data_collator(include_transcription)`
Creates a data collator instance for batch processing.

**Returns:**
- `PronunciationAssessmentDataCollator` instance

##### `validate_sample(sample)`
Validates a sample for required fields and format correctness.

**Validates:**
- Required fields presence
- Audio format
- Score numeric validity
- Word information structure

**Returns:**
- Tuple of (is_valid, error_message)

##### `save_processed_dataset(datasets, output_path)` / `load_processed_dataset(dataset_path)`
Utilities for saving and loading processed datasets to/from disk.

## Usage Example

### Basic Training Preparation

```python
from finetuning.finetuning_pronunciation_assessment.data_processor import SpeechOcean762PronunciationProcessor
from finetuning.finetuning_pronunciation_assessment.training_config import PronunciationTrainingConfig

# Initialize processor
config = PronunciationTrainingConfig(
    whisper_model_name="openai/whisper-base",
    sampling_rate=16000,
    max_audio_length=30.0,
    normalize_audio=True
)

processor = SpeechOcean762PronunciationProcessor(
    whisper_model_name=config.whisper_model_name,
    sampling_rate=config.sampling_rate,
    max_audio_length=config.max_audio_length,
    normalize_audio=config.normalize_audio
)

# Load dataset
datasets = processor.load_dataset(
    splits=["train", "test"],
    max_samples_per_split={
        "train": config.max_train_samples,
        "test": config.max_eval_samples
    }
)

# Prepare for training
processed_datasets = processor.prepare_dataset_for_training(
    datasets,
    include_transcription=config.include_transcription
)

# Get statistics
stats = processor.get_dataset_statistics(processed_datasets)
print(stats)

# Create data collator
data_collator = processor.create_data_collator(
    include_transcription=config.include_transcription
)
```

### With Trainer

```python
from transformers import Trainer, TrainingArguments

data_collator = processor.create_data_collator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn
)

trainer.train()
```

## Data Flow Diagram

```
Raw SpeechOcean762 Dataset
         ↓
    load_dataset()
         ↓
    Raw DatasetDict
         ↓
prepare_dataset_for_training()
    ├─ preprocess_audio()
    │   ├─ Resample to 16kHz
    │   ├─ Normalize amplitude
    │   └─ Pad/trim to 30s
    ├─ Feature extraction (mel-spectrogram)
    ├─ Transcription tokenization
    └─ _extract_pronunciation_scores()
        ├─ Utterance-level scores
        ├─ Word-level scores
        └─ Phone-level scores
         ↓
Processed DatasetDict
         ↓
    Trainer.train()
    with DataCollator
         ↓
    Training Batches
```

## Score Normalization

All scores are normalized to 0-1 range for stable training:

| Level | Original Scale | Normalized Scale |
|-------|----------------|------------------|
| Utterance | 0-10 | 0-1 |
| Word (accuracy, total) | 0-10 | 0-1 |
| Word (stress) | 0-10 | 0-1 |
| Phone (accuracy) | 0-2 | 0-1 |

## Features & Capabilities

✅ **Multi-level Score Processing**
- Utterance-level: 5 scores (accuracy, fluency, prosodic, completeness, total)
- Word-level: 3 scores per word (accuracy, stress, total)
- Phone-level: 1 score per phone (accuracy)

✅ **Audio Preprocessing**
- Automatic resampling to target rate
- Audio normalization
- Padding/trimming to max duration
- Handles variable-length inputs

✅ **Batch Processing**
- Automatic padding of sequences
- Attention mask generation
- Efficient data collation

✅ **Dataset Management**
- Load from Hugging Face Hub
- Optional sample limiting for testing
- Save/load processed data to/from disk
- Dataset statistics computation

✅ **Validation & Error Handling**
- Sample validation with detailed error messages
- Graceful error handling during preprocessing
- Comprehensive logging

## Configuration Best Practices

1. **Model Selection**: Use `openai/whisper-tiny` for quick testing, `openai/whisper-base` for development, `openai/whisper-medium` for production.

2. **Audio Settings**: 
   - Keep `sampling_rate=16000` (Whisper standard)
   - `max_audio_length=30.0` balances coverage and memory
   - Enable `normalize_audio=True` for stable training

3. **Data Limits for Testing**:
   ```python
   max_samples_per_split={"train": 500, "test": 100}
   ```

4. **Training Configuration**:
   - `batch_size`: 4-8 for GPUs, 1-2 for CPU
   - `num_workers`: 0 for Windows, 2-4 for Linux/Mac
   - `include_transcription`: True for balanced multi-objective training

## Performance Considerations

- **Memory**: Batch size depends on GPU memory. Reduce if OOM errors occur.
- **Speed**: Audio preprocessing is CPU-bound. Increase `num_proc` if CPU has spare capacity.
- **Quality**: Normalization improves convergence. Always enable for stable training.

## Dependencies

- `transformers`: WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
- `datasets`: Dataset, DatasetDict management
- `librosa`: Audio preprocessing and resampling
- `numpy`: Numerical operations
- `torch`: PyTorch tensors and data handling
- `pathlib`: Cross-platform file operations
