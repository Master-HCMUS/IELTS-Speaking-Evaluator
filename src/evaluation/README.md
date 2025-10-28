# Evaluation Module Structure# Evaluation Module



This evaluation module is organized into two dedicated sub-packages for evaluating IELTS pronunciation assessment systems.This directory contains evaluation tools for pronunciation assessment models.



## Directory Structure## Files Overview



```### Core Evaluation Modules

src/evaluation/- **`dataset_evaluator.py`** - Azure Speech evaluation against SpeechOcean762 dataset

├── __init__.py                           # Main package init- **`whisper_evaluator.py`** - Whisper model evaluation (with Azure Speech dependencies)

├── azure_speech_evaluation/              # Azure Speech API evaluation- **`standalone_whisper_evaluator.py`** - Standalone Whisper evaluation (no Azure dependencies)

│   ├── __init__.py- **`evaluate_dataset.py`** - Utility script for dataset evaluation

│   ├── core.py                          # Main logic (SpeechOcean762Evaluator)

│   └── cli.py                           # CLI entry point### CLI Scripts

├── stt_whisper_evaluation/               # Fine-tuned Whisper model evaluation- **`evaluate_whisper_standalone.py`** - Main CLI for standalone Whisper evaluation

│   ├── __init__.py

│   ├── core.py                          # Main logic (StandaloneWhisperModelEvaluator)### Examples and Testing

│   └── cli.py                           # CLI entry point- **`examples_whisper_evaluation.py`** - Usage examples and model comparison

├── evaluation_results/                   # Output directory for results

└── README.md                             # This file## Usage

```

### Recommended: Direct Standalone Evaluation

## Sub-Packages```bash

cd src/evaluation

### 1. Azure Speech Evaluation (azure_speech_evaluation/)

# Quick test with 10 samples

Evaluate **Azure Speech pronunciation assessment API** against the SpeechOcean762 dataset.python evaluate_whisper_standalone.py --model-path "../../whisper_development" --quick-test



**Files:**# Full evaluation

- `core.py` - Contains `SpeechOcean762Evaluator` class with evaluation logicpython evaluate_whisper_standalone.py --model-path "../../whisper_development"

- `cli.py` - Command-line interface for running evaluations

# Evaluate on specific split

**CLI Usage:**python evaluate_whisper_standalone.py --model-path "../../whisper_production" --split validation --max-samples 100

```bash```

# Run evaluation on full test set

python -m src.evaluation.azure_speech_evaluation.cli### Programmatic Usage

```python

# Run quick test with 50 samplesfrom src.evaluation.standalone_whisper_evaluator import StandaloneWhisperModelEvaluator

python -m src.evaluation.azure_speech_evaluation.cli --max-samples 50

# Initialize evaluator

# Evaluate on validation splitevaluator = StandaloneWhisperModelEvaluator("path/to/whisper/model")

python -m src.evaluation.azure_speech_evaluation.cli --split validation --no-save

```# Load dataset and run evaluation

evaluator.load_dataset(split="test", max_samples=50)

**Python API:**metrics = evaluator.run_evaluation()

```python

from evaluation.azure_speech_evaluation import SpeechOcean762Evaluator# Print results

from config_manager import ConfigManagerevaluator.print_evaluation_summary(metrics)

```

config = ConfigManager('config/audio_config.json')

evaluator = SpeechOcean762Evaluator(config)## Architecture

evaluator.load_dataset(split='test', max_samples=50)

metrics = evaluator.run_evaluation(save_results=True)### Standalone vs Integrated Evaluators

```

1. **Standalone Evaluator** (`standalone_whisper_evaluator.py`)

### 2. STT Whisper Evaluation (stt_whisper_evaluation/)   - Self-contained, no external dependencies

   - Works independently of Azure Speech framework

Evaluate **fine-tuned Whisper models** for speech-to-text (STT) and pronunciation assessment against the SpeechOcean762 dataset.   - Recommended for most use cases



**Files:**2. **Integrated Evaluator** (`whisper_evaluator.py`)  

- `core.py` - Contains:   - Integrates with existing Azure Speech evaluation framework

  - `StandaloneWhisperPronunciationAssessor` - Model loading and transcription   - Shares data structures with Azure Speech evaluator

  - `StandaloneWhisperModelEvaluator` - Dataset evaluation and metrics computation   - Requires Azure Speech dependencies

  - Supporting data classes: `StandaloneEvaluationMetrics`, `WhisperEvaluationResult`

- `cli.py` - Command-line interface for running evaluations### Evaluation Methodology



**CLI Usage:**Both evaluators use the same methodology:

```bash- **Transcription Quality**: WER, CER, BLEU scores, confidence

# Run quick test with 10 samples- **Pronunciation Mapping**: Quality metrics → pronunciation scores (0-10 scale)

python -m src.evaluation.stt_whisper_evaluation.cli \- **Expert Correlation**: Pearson correlation with SpeechOcean762 annotations

    --model-path "src/finetuning/finetuning_STT/models/whisper_development" \- **Statistical Analysis**: MAE, RMSE, score distributions

    --quick-test

## Dependencies

# Evaluate with specific number of samples

python -m src.evaluation.stt_whisper_evaluation.cli \- `transformers` - Whisper model loading and inference

    --model-path "src/finetuning/finetuning_STT/models/whisper_development" \- `torch` - PyTorch for model execution

    --max-samples 100- `librosa` - Audio processing

- `datasets` - HuggingFace datasets for SpeechOcean762

# Evaluate on validation split- `evaluate` - Evaluation metrics (WER, BLEU)

python -m src.evaluation.stt_whisper_evaluation.cli \- `scipy` - Statistical analysis

    --model-path "src/finetuning/finetuning_STT/models/whisper_development" \- `soundfile` - Audio I/O
    --split validation --max-samples 50
```

**Python API:**
```python
from evaluation.stt_whisper_evaluation import (
    StandaloneWhisperModelEvaluator,
    StandaloneWhisperPronunciationAssessor
)

# Initialize evaluator
evaluator = StandaloneWhisperModelEvaluator(
    "src/finetuning/finetuning_STT/models/whisper_development"
)

# Load dataset
evaluator.load_dataset(split='test', max_samples=100)

# Run evaluation
metrics = evaluator.run_evaluation(save_results=True)

# Print summary
evaluator.print_evaluation_summary(metrics)
```

## Evaluation Metrics

Both evaluators compute correlation metrics with expert human annotations:

- **Correlation (r):** Pearson correlation with expert scores
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Square Error

### Pronunciation Assessment Dimensions

1. **Accuracy** (0-10): How correctly phonemes are pronounced
2. **Fluency** (0-10): Speech rate and rhythm continuity
3. **Completeness** (0-10): Whether all words/phones were spoken
4. **Prosodic** (0-10): Intonation and stress patterns

## Output

Both evaluators save detailed results to `evaluation_results/`:

**JSON output includes:**
- Model information and timestamp
- Evaluation metrics (correlations, MAE, RMSE)
- Score distributions
- Individual sample results

## Dataset

Both evaluators use the **SpeechOcean762 dataset**:
- 762 hours of English speech from non-native speakers
- Expert annotations from 5 annotators per sample
- Sentence-level, word-level, and phoneme-level scores
- Available via HuggingFace: `mispeech/speechocean762`

## Requirements

- `torch` - PyTorch
- `transformers` - For Whisper models
- `datasets` - For dataset loading
- `librosa` - Audio processing
- `soundfile` - Sound file I/O
- `scipy` - Statistical calculations
- `pandas` - Data manipulation

## Migration Guide

If you were using old file names, update your imports:

**Old → New:**
- `dataset_evaluator.py` → `azure_speech_evaluation/core.py`
- `evaluate_dataset.py` → `azure_speech_evaluation/cli.py`
- `standalone_whisper_evaluator.py` → `stt_whisper_evaluation/core.py`
- `evaluate_whisper_standalone.py` → `stt_whisper_evaluation/cli.py`

**Updated import examples:**
```python
# Azure Speech
from evaluation.azure_speech_evaluation import SpeechOcean762Evaluator

# Whisper STT
from evaluation.stt_whisper_evaluation import StandaloneWhisperModelEvaluator
```
