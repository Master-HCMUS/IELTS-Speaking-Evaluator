# Evaluation Module

This directory contains evaluation tools for pronunciation assessment models.

## Files Overview

### Core Evaluation Modules
- **`dataset_evaluator.py`** - Azure Speech evaluation against SpeechOcean762 dataset
- **`whisper_evaluator.py`** - Whisper model evaluation (with Azure Speech dependencies)
- **`standalone_whisper_evaluator.py`** - Standalone Whisper evaluation (no Azure dependencies)
- **`evaluate_dataset.py`** - Utility script for dataset evaluation

### CLI Scripts
- **`evaluate_whisper_standalone.py`** - Main CLI for standalone Whisper evaluation

### Examples and Testing
- **`examples_whisper_evaluation.py`** - Usage examples and model comparison

## Usage

### Recommended: Direct Standalone Evaluation
```bash
cd src/evaluation

# Quick test with 10 samples
python evaluate_whisper_standalone.py --model-path "../../whisper_development" --quick-test

# Full evaluation
python evaluate_whisper_standalone.py --model-path "../../whisper_development"

# Evaluate on specific split
python evaluate_whisper_standalone.py --model-path "../../whisper_production" --split validation --max-samples 100
```

### Programmatic Usage
```python
from src.evaluation.standalone_whisper_evaluator import StandaloneWhisperModelEvaluator

# Initialize evaluator
evaluator = StandaloneWhisperModelEvaluator("path/to/whisper/model")

# Load dataset and run evaluation
evaluator.load_dataset(split="test", max_samples=50)
metrics = evaluator.run_evaluation()

# Print results
evaluator.print_evaluation_summary(metrics)
```

## Architecture

### Standalone vs Integrated Evaluators

1. **Standalone Evaluator** (`standalone_whisper_evaluator.py`)
   - Self-contained, no external dependencies
   - Works independently of Azure Speech framework
   - Recommended for most use cases

2. **Integrated Evaluator** (`whisper_evaluator.py`)  
   - Integrates with existing Azure Speech evaluation framework
   - Shares data structures with Azure Speech evaluator
   - Requires Azure Speech dependencies

### Evaluation Methodology

Both evaluators use the same methodology:
- **Transcription Quality**: WER, CER, BLEU scores, confidence
- **Pronunciation Mapping**: Quality metrics → pronunciation scores (0-10 scale)
- **Expert Correlation**: Pearson correlation with SpeechOcean762 annotations
- **Statistical Analysis**: MAE, RMSE, score distributions

## Dependencies

- `transformers` - Whisper model loading and inference
- `torch` - PyTorch for model execution
- `librosa` - Audio processing
- `datasets` - HuggingFace datasets for SpeechOcean762
- `evaluate` - Evaluation metrics (WER, BLEU)
- `scipy` - Statistical analysis
- `soundfile` - Audio I/O