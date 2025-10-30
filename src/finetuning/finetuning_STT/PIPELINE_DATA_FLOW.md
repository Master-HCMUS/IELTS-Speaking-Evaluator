# 🎙️ STT Training Pipeline: Input → Process → Output

## Overview

This document describes the complete data flow in the Whisper Speech-to-Text (STT) fine-tuning pipeline, including detailed specifications of input data sizes, transformation processes, and output sizes.

---

## 1. INPUT STAGE

### 1.1 Data Source: SpeechOcean762 Dataset

**Dataset Details:**
- **Name**: SpeechOcean762 (mispeech/speechocean762)
- **Purpose**: English pronunciation assessment dataset for speech recognition
- **Access**: Hugging Face Datasets library

**Dataset Splits:**
```
├── Training Split
│   ├── Samples: ~3,000-5,000 audio samples
│   ├── Language: English
│   └── Duration: 1-3 hours of audio content
│
└── Test Split
    ├── Samples: ~500-1,000 audio samples
    └── Duration: 15-30 minutes of audio content
```

### 1.2 Raw Audio Input

**Per-Sample Audio Specifications:**
```
┌─────────────────────────────────┐
│ RAW AUDIO CHARACTERISTICS       │
├─────────────────────────────────┤
│ Format: WAV/MP3                 │
│ Sampling Rate: 16 kHz (native)  │
│ Duration: 1-30 seconds          │
│ Audio Channels: Mono/Stereo     │
│ Bit Depth: 16-bit               │
│ File Size: 32 KB - 1.0 MB       │
└─────────────────────────────────┘

Average Per-Sample:
├── Duration: ~5 seconds
├── Samples Count: 5 × 16,000 = 80,000 samples
├── Data Size: 160 KB (mono, 16-bit)
└── Uncompressed: 0.16 MB
```

**Full Dataset Input Size:**
```
Quick Test Configuration:
├── Train: 100 samples → ~16 MB
├── Eval: 50 samples → ~8 MB
└── Total: ~24 MB

Development Configuration:
├── Train: 3,000 samples → ~480 MB
├── Eval: 500 samples → ~80 MB
└── Total: ~560 MB

Production Configuration:
├── Train: 5,000 samples → ~800 MB
├── Eval: 1,000 samples → ~160 MB
└── Total: ~960 MB
```

### 1.3 Text Transcriptions

**Transcription Characteristics:**
```
Per Sample:
├── Format: Plain text
├── Content: English transcription
├── Length: 3-50 words
├── Average: ~15 words
├── Character Count: 20-150 characters
└── Data Size: ~0.05 KB per sample
```

### 1.4 Pronunciation Scores

**Score Information per Sample:**
```
┌──────────────────────────────────┐
│ PRONUNCIATION SCORES             │
├──────────────────────────────────┤
│ accuracy:      0-10 (float)      │
│ fluency:       0-10 (float)      │
│ completeness:  0-10 (float)      │
│ prosodic:      0-10 (float)      │
│                                  │
│ Speaker Info:                    │
│ ├── speaker_id: integer          │
│ ├── gender: M/F                  │
│ └── age: 15-70                   │
└──────────────────────────────────┘

Data Size per Sample: ~0.05 KB
```

### 1.5 Complete Raw Input Data Structure

**Per-Sample Example:**
```python
{
    "audio": {
        "array": np.ndarray(shape=(80000,), dtype=float32),  # 320 KB
        "sampling_rate": 16000
    },
    "text": "hello world how are you today",  # ~0.04 KB
    "accuracy": 8.5,
    "fluency": 7.9,
    "completeness": 9.0,
    "prosodic": 7.5,
    "speaker": 101,
    "gender": "M",
    "age": 25
}

Total Size per Sample: ~320 KB
```

**Total Input Dataset Size Summary:**
```
Configuration          | Train Samples | Test Samples | Total Size
---------------------- | ------------- | ------------ | ----------
Quick Test             | 100           | 50           | 24 MB
Development            | 3,000         | 500          | 560 MB
Production             | 5,000         | 1,000        | 960 MB
```

---

## 2. PROCESSING STAGE

### 2.1 Audio Preprocessing Pipeline

**Step 1: Resampling**
```
Input Audio:
├── Original sampling rate: Variable (16kHz, 44.1kHz, 48kHz, etc.)
└── Samples: N

↓ Resample using librosa

Output Audio:
├── Target sampling rate: 16 kHz (fixed)
└── Samples: N' = N × (16000 / original_sr)

Example:
├── Input: 10s @ 44.1kHz → 441,000 samples
├── Output: 10s @ 16kHz → 160,000 samples
└── Reduction: 63.7% sample reduction
```

**Step 2: Audio Normalization**
```
Formula: y_norm = y / max(|y|)

Example:
├── Input:  [-32000, -16000, 0, 16000, 32000]
├── Max:    32000
└── Output: [-1.0, -0.5, 0.0, 0.5, 1.0]

Purpose: Normalize amplitude to [-1, 1] range
Result: Better numerical stability during processing
```

**Step 3: Trimming/Padding to Max Length**
```
max_audio_length = 30 seconds
max_length_samples = 30 × 16,000 = 480,000 samples

Case A: Audio > 30s
├── Input: 600,000 samples
└── Output: Truncated to 480,000 samples

Case B: Audio < 30s
├── Input: 80,000 samples
└── Output: Padded with zeros to 480,000 samples

Case C: Audio = 5s
├── Input: 80,000 samples
├── Padding: 400,000 zero samples
└── Output: 480,000 samples
```

**Preprocessed Audio Output:**
```
Per Sample:
├── Shape: (480,000,) float32 array
├── Range: [-1.0, 1.0]
├── Size: 1.92 MB (480,000 × 4 bytes)
└── Characteristics:
    ├── Resampled to 16 kHz
    ├── Normalized amplitude
    ├── Padded/Trimmed to 30 seconds
    └── Ready for feature extraction
```

### 2.2 Audio Feature Extraction (Whisper Feature Extractor)

**Process:**
```
Input: Audio waveform (480,000 samples @ 16kHz, 30 seconds)

↓ Mel-Spectrogram Conversion

Process:
├── 1. Apply Hann window with 400ms stride
│   ├── Window size: 400ms × 16kHz = 6,400 samples
│   └── Stride: 160ms × 16kHz = 2,560 samples
│
├── 2. Compute FFT for each frame
│   ├── Number of frames: (480,000 - 6,400) / 2,560 = 185 frames
│   └── FFT size: 1024
│
├── 3. Compute power spectrogram
│   └── Shape: (185, 513) - 185 frames, 513 frequency bins
│
├── 4. Apply Mel-scale filter bank (80 filters)
│   └── Shape: (185, 80) - Mel-spectrogram
│
└── 5. Apply log compression
    └── log(mel_spec + 1e-9)

Output: Log Mel-Spectrogram
├── Shape: (80, 3000) - Padded to standard size
├── Value Range: [-20, 10] (dB scale)
└── Data Type: float32
```

**Feature Output Specifications:**
```
Per Sample:
├── Input Features Shape: (80, 3000)
│   ├── 80: Mel-frequency bins
│   └── 3000: Time frames (padded)
│
├── Size: 80 × 3000 × 4 bytes = 960 KB
├── Data Type: float32
└── Values: Normalized to [-20, 10] dB range

Full Batch (batch_size = 8):
├── Shape: (8, 80, 3000)
├── Size: 8 × 960 KB = 7.68 MB
└── Processing Time: ~100ms on GPU
```

**Visual Representation:**
```
Raw Audio (480,000 samples)
    ↓
Window into frames (185 frames × 6,400 samples)
    ↓
FFT & Power Spectrogram (185 × 513)
    ↓
Mel-scale Filtering (185 × 80)
    ↓
Log Compression & Normalization
    ↓
Padding to Standard Size (80 × 3000)
    ↓
Final Features Output
```

### 2.3 Text Tokenization (Whisper Tokenizer)

**Process:**
```
Input: "hello world how are you today"

↓ BPE (Byte Pair Encoding) Tokenization

Steps:
├── 1. Lowercase conversion
│   └── "hello world how are you today"
│
├── 2. Character-level encoding
│   └── Break into subword units using BPE
│
├── 3. Token mapping
│   ├── <|startoftranscript|>: 50257
│   ├── "hel": 7296
│   ├── "lo": 9319
│   ├── " world": 6002
│   ├── " how": 1212
│   ├── " are": 389
│   ├── " you": 291
│   ├── " to": 284
│   ├── "day": 1429
│   └── <|endoftext|>: 50256
│
└── 4. Padding
    └── Pad sequence to standard length (448 tokens)
```

**Tokenizer Output:**
```
Per Sample:
├── Token Sequence: [50257, 7296, 9319, 6002, 1212, ...]
├── Length: 20-100 tokens (before padding)
├── Padded Length: 448 tokens
├── Padding Token: -100 (for loss computation)
├── Data Type: int64
└── Size: 448 × 8 bytes = 3.6 KB per sample

Full Batch (batch_size = 8):
├── Shape: (8, 448)
├── Size: 8 × 3.6 KB = 28.8 KB
└── Processing Time: ~1ms on CPU
```

### 2.4 Data Collation for Training

**DataCollatorSpeechSeq2SeqWithPadding Process:**

```
Input Batch (8 samples):
├── input_features: List[80 × 3000 arrays]
├── labels: List[variable-length token sequences]
├── transcription: List[text strings]
└── pronunciation_scores: List[score dictionaries]

↓ Collation Process

Steps:
├── 1. Feature Stacking
│   ├── Stack 8 × (80, 3000) → (8, 80, 3000)
│   └── Pad along time dimension if needed
│
├── 2. Label Padding
│   ├── Pad all sequences to max length in batch
│   ├── Replace padding with -100 (ignored in loss)
│   └── Remove <bos> token if present
│
├── 3. Attention Mask Creation
│   ├── Create mask showing valid positions
│   └── Shape: (8, 448)
│
└── 4. Format for Model
    └── Create PyTorch tensors

Output Batch Ready for Model:
├── input_features: Tensor(8, 80, 3000)  - 7.68 MB
├── labels: Tensor(8, 448)               - 28.8 KB
├── attention_mask: Tensor(8, 448)       - 28.8 KB
└── Total Batch Size: 7.74 MB
```

### 2.5 Complete Processing Pipeline Diagram

```
RAW DATA (per sample)
├── audio.wav (320 KB) 
│   ├── Resample (16kHz)
│   ├── Normalize
│   ├── Pad to 30s (480,000 samples)
│   └── → Preprocessed Audio (1.92 MB)
│
├── transcription (0.04 KB)
│   └── Tokenize (BPE)
│       └── → Token IDs (3.6 KB)
│
└── pronunciation_scores (0.05 KB)
    └── → Preserved as-is (0.05 KB)


FEATURE EXTRACTION
├── Audio (1.92 MB)
│   ├── Mel-Spectrogram
│   ├── Normalize & Pad
│   └── → Log Mel-Features (960 KB)
│
└── Tokens (3.6 KB)
    ├── Pad to 448 tokens
    └── → Token Tensor (3.6 KB)


BATCHING (8 samples)
├── Stack Features: (8, 80, 3000) - 7.68 MB
├── Stack Labels: (8, 448) - 28.8 KB
├── Create Masks: (8, 448) - 28.8 KB
└── → Ready for Model: 7.74 MB
```

---

## 3. OUTPUT STAGE

### 3.1 Training Input to Model

**Batch Structure Sent to Model:**
```
Batch = {
    'input_features': Tensor of shape (8, 80, 3000)
        ├── Type: float32
        ├── Values: Log Mel-spectrogram [-20, 10]
        └── Size: 7.68 MB
    
    'labels': Tensor of shape (8, 448)
        ├── Type: int64
        ├── Values: Token IDs (0-51864)
        ├── Padding: -100 (ignored in loss)
        └── Size: 28.8 KB
    
    'attention_mask': Tensor of shape (8, 448)
        ├── Type: int64
        ├── Values: 0 or 1
        └── Size: 28.8 KB
}

Total Batch Input Size: 7.74 MB
```

### 3.2 Model Processing

**Whisper Model Architecture Flow:**
```
Input Features (8, 80, 3000)
    ↓
Encoder:
├── Linear Projection (80 → 512)
│   └── Shape: (8, 3000, 512)
│
├── Positional Encoding
│   └── Shape: (8, 3000, 512)
│
├── Transformer Encoder (4-12 layers)
│   ├── Self-Attention
│   ├── Feed-Forward Network
│   ├── Layer Normalization
│   └── Output Shape: (8, 3000, 512)
│
└── Encoder Output: (8, 3000, 512)
    ├── Size: 8 × 3000 × 512 × 4 = 49.15 MB
    └── Intermediate representation


Decoder:
├── Token Embedding + Positional Encoding
│   ├── Input: (8, 448)
│   └── Output: (8, 448, 512)
│
├── Transformer Decoder (4-12 layers)
│   ├── Self-Attention
│   ├── Cross-Attention (with encoder output)
│   ├── Feed-Forward Network
│   └── Output Shape: (8, 448, 512)
│
└── Final Linear Projection
    ├── Projects to vocabulary size
    └── Output Shape: (8, 448, 51865)
        ├── 51865: Whisper vocabulary size
        ├── Size: 8 × 448 × 51865 × 4 = 74.3 MB
        └── Contains logits per token


Loss Computation:
├── Compare logits with target tokens
├── Cross-Entropy Loss
└── Backpropagation
```

### 3.3 Model Output (Logits)

**Raw Model Output:**
```
Per Batch:
├── Logits Shape: (8, 448, 51865)
│   ├── 8: Batch size
│   ├── 448: Sequence length (tokens)
│   └── 51865: Vocabulary size
│
├── Data Type: float32
├── Value Range: (-∞, ∞) - Raw logit values
├── Size: 74.3 MB (raw, not stored)
└── Processing: Used for loss computation only

Per Sample:
├── Logits Shape: (448, 51865)
├── Size: 9.28 MB
└── Contains: Probability distribution over vocabulary
```

### 3.4 Inference Output (Greedy Decoding)

**From Logits to Text:**
```
Logits (8, 448, 51865)
    ↓
Argmax: Get highest probability token per position
├── Shape: (8, 448)
├── Values: Token IDs (0-51865)
└── Processing: argmax(logits, dim=-1)

    ↓
Tokenizer Decoding: Convert IDs back to text
├── Remove special tokens
├── Merge subword pieces
└── Return: Plain text transcription

Output Example:
├── Input Audio: Speech saying "hello world"
├── Model Output: [7296, 9319, 6002, 1212, ...]
└── Decoded Text: "hello world"
```

### 3.5 Training Output Files

**After Each Epoch:**
```
checkpoint-{step}/
├── config.json (4 KB)
│   └── Model configuration
│
├── pytorch_model.bin (300-1500 MB)
│   ├── Model weights
│   └── Size depends on model size:
│       ├── tiny: ~150 MB
│       ├── base: ~290 MB
│       ├── small: ~770 MB
│       └── medium: ~1.5 GB
│
├── optimizer.pt (300-1500 MB)
│   └── Optimizer state for resuming
│
├── trainer_state.json (2 KB)
│   └── Training state and metrics
│
└── preprocessing_config.json (1 KB)
    └── Feature extractor config
```

**Training Metrics Output:**
```
Per Evaluation Step:
├── Evaluation Loss: float
│   └── Example: 2.457
│
├── Word Error Rate (WER): float [0-1]
│   ├── Definition: (S + D + I) / N
│   ├── S: Substitutions
│   ├── D: Deletions
│   ├── I: Insertions
│   └── N: Total reference words
│   └── Example: 0.15 (15% error)
│
├── BLEU Score: float [0-1]
│   └── Example: 0.82
│
├── Character Accuracy: float [0-1]
│   └── Example: 0.92
│
└── Learning Rate: float
    └── Example: 1e-5
```

### 3.6 Final Model Output Structure

**After Training Completion:**
```
whisper_finetuned/
├── config.json                     (4 KB)
├── pytorch_model.bin               (300-1500 MB)
├── preprocessor_config.json        (1 KB)
├── tokenizer.json                  (500 KB)
├── special_tokens_map.json         (1 KB)
│
├── training_summary.json           (5 KB)
│   ├── Final metrics
│   ├── Training duration
│   └── Best checkpoint info
│
├── evaluation_results.json         (10 KB)
│   ├── WER per split
│   ├── BLEU scores
│   ├── Character accuracy
│   └── Sample predictions
│
└── logs/
    └── training_20251030_101237.log (100 KB)
        └── Complete training log

Total Output Size: 300-1500 MB (model) + 0.5 MB (metadata)
```

**Sample Predictions Example:**
```
Sample 1:
├── Audio File: sample_001.wav
├── Reference: "the quick brown fox"
├── Predicted: "the quick brown fox"
├── WER: 0.00 (perfect)
└── Duration: 2.1 seconds

Sample 2:
├── Audio File: sample_002.wav
├── Reference: "how are you today"
├── Predicted: "how are you toady"
├── WER: 0.25 (1 error out of 4 words)
└── Duration: 1.8 seconds
```

---

## 4. DATA SIZE SUMMARY TABLE

### 4.1 Size Transformation by Stage

```
┌────────────────────────────────────────────────────────────┐
│              DATA SIZE TRANSFORMATION                       │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ Stage                Per Sample      Full Dataset           │
│ ─────────────────────────────────────────────────────────  │
│ 1. Raw Audio         320 KB          3-5 GB (5000 samples) │
│                                                             │
│ 2. Preprocessed      1.92 MB         9.6-16 GB             │
│    Audio                             (5000 samples)        │
│                                                             │
│ 3. Audio Features    960 KB          4.8-8 GB              │
│    (Mel-Spectrum)                    (5000 samples)        │
│                                                             │
│ 4. Tokenized Text    3.6 KB          18-36 MB              │
│                                      (5000 samples)        │
│                                                             │
│ 5. Model Output      9.28 MB         Not stored            │
│    (per sample, not  (logits for     (computed on-the-fly) │
│     stored in memory) inference)                           │
│                                                             │
│ 6. Checkpoint        300-1500 MB     Single file           │
│    (Model weights)                   (for entire model)    │
│                                                             │
│ 7. Training Metrics  ~1 KB           ~100 MB               │
│    per epoch                         (10-20 checkpoints)   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Configuration Specific Sizes

**Quick Test (100 train, 50 test samples):**
```
Input:
├── Audio: 24 MB
├── Transcriptions: 7 KB
└── Scores: 8 KB

Processing:
├── Preprocessed Audio: 115 MB
├── Features: 58 MB
└── Tokens: 500 KB

Output:
├── Model Checkpoint: 300-500 MB
└── Logs & Metrics: 5 MB

Total Disk Usage: ~360-510 MB
```

**Development (3,000 train, 500 test samples):**
```
Input:
├── Audio: 560 MB
├── Transcriptions: 210 KB
└── Scores: 240 KB

Processing:
├── Preprocessed Audio: 3.4 GB
├── Features: 1.7 GB
└── Tokens: 12.6 MB

Output:
├── Model Checkpoint: 300-500 MB
└── Logs & Metrics: 15 MB

Total Disk Usage: ~4.2-4.4 GB
```

**Production (5,000 train, 1,000 test samples):**
```
Input:
├── Audio: 960 MB
├── Transcriptions: 350 KB
└── Scores: 400 KB

Processing:
├── Preprocessed Audio: 5.8 GB
├── Features: 2.9 GB
└── Tokens: 21 MB

Output:
├── Model Checkpoint: 300-1500 MB
└── Logs & Metrics: 30 MB

Total Disk Usage: ~8.8-9.8 GB
```

---

## 5. MEMORY REQUIREMENTS

### 5.1 During Training (per batch)

```
Batch Size = 8

GPU Memory:
├── Model Weights: 300-1500 MB (depends on model size)
├── Optimizer State: 600-3000 MB (2x model for Adam)
├── Gradients: 300-1500 MB
├── Input Batch: 7.74 MB
├── Intermediate Activations: 500-2000 MB
├── Workspace: 500 MB
└── Total: 2-8 GB GPU memory

CPU Memory:
├── Dataset in Memory: 100-500 MB
├── DataLoader buffers: 100 MB
├── Other: 200 MB
└── Total: 400-800 MB CPU memory

Recommended:
├── Quick Test: 4 GB GPU, 2 GB CPU
├── Development: 8 GB GPU, 4 GB CPU
└── Production: 16+ GB GPU, 8 GB CPU
```

### 5.2 Gradient Accumulation Impact

**With gradient_accumulation_steps = 4:**
```
Effective batch size: 4 × 8 = 32
GPU memory: Lower per iteration but more accumulation steps
Number of gradient updates: Same as batch_size = 32
Processing time: 4x longer per step
Effective learning rate: Same as batch_size = 32
```

---

## 6. PROCESSING TIME ANALYSIS

### 6.1 Per-Sample Processing Time

```
Audio Preprocessing (per sample):
├── Resample: 10-50 ms
├── Normalize: 5-10 ms
├── Pad/Trim: 1-2 ms
└── Total: 16-62 ms

Feature Extraction (per sample):
├── Mel-Spectrogram: 20-100 ms
├── Normalization: 5-10 ms
└── Total: 25-110 ms

Tokenization (per sample):
├── BPE encoding: 1-5 ms
└── Total: 1-5 ms

Combined Per-Sample: 42-177 ms
```

### 6.2 Full Pipeline Processing Time

**Quick Test (150 samples):**
```
Preprocessing: 150 × 50ms = 7.5 seconds
Feature Extraction: 150 × 70ms = 10.5 seconds
Tokenization: 150 × 3ms = 0.45 seconds
Total: ~18-20 seconds
```

**Development (3,500 samples):**
```
Preprocessing: 3,500 × 50ms = 175 seconds (~3 minutes)
Feature Extraction: 3,500 × 70ms = 245 seconds (~4 minutes)
Tokenization: 3,500 × 3ms = 10.5 seconds
Total: ~230-250 seconds (~4 minutes)
```

**Production (6,000 samples):**
```
Preprocessing: 6,000 × 50ms = 300 seconds (~5 minutes)
Feature Extraction: 6,000 × 70ms = 420 seconds (~7 minutes)
Tokenization: 6,000 × 3ms = 18 seconds
Total: ~390-450 seconds (~7 minutes)
```

### 6.3 Training Time

**Quick Test:**
```
Configuration:
├── Model: whisper-tiny (39M parameters)
├── Batch Size: 4
├── Epochs: 1
├── Samples: 100

Timeline:
├── Data Loading: 20 seconds
├── Training: 1-2 minutes (50 steps)
├── Evaluation: 30 seconds
├── Total: 2-3 minutes
```

**Development:**
```
Configuration:
├── Model: whisper-tiny (39M parameters)
├── Batch Size: 8
├── Epochs: 3
├── Samples: 3,000

Timeline:
├── Data Loading: 5 minutes
├── Training: 30-60 minutes (375 steps × 3 epochs)
├── Evaluation: 5-10 minutes
├── Total: 40-75 minutes
```

**Production:**
```
Configuration:
├── Model: whisper-base (74M parameters)
├── Batch Size: 16
├── Epochs: 5
├── Samples: 5,000

Timeline:
├── Data Loading: 8 minutes
├── Training: 2-4 hours (312 steps × 5 epochs)
├── Evaluation: 10-15 minutes
├── Total: 2.5-4.5 hours
```

---

## 7. PROCESSING BOTTLENECKS & OPTIMIZATIONS

### 7.1 Identified Bottlenecks

```
1. Audio Resampling (20-40% of preprocessing time)
   Optimization:
   ├── Cache resampled audio
   ├── Use GPU-accelerated resampling (julius library)
   └── Parallelize with multiprocessing

2. Feature Extraction (50-70% of preprocessing time)
   Optimization:
   ├── Batch processing on GPU
   ├── Use STFT caching
   └── Parallelize batch computation

3. Dataset Loading
   Optimization:
   ├── Use num_workers > 0 in DataLoader
   ├── Pin memory for faster CPU→GPU transfer
   └── Prefetch next batch during computation

4. Mixed Precision (fp16) Opportunities
   Optimization:
   ├── Model forward pass: fp16
   ├── Loss computation: fp32 (stable)
   ├── Reduces GPU memory by ~50%
   └── Training: 1.3-1.5x faster
```

### 7.2 Optimization Recommendations

```
For Quick Test:
├── Use whisper-tiny
├── Batch size: 4
├── num_workers: 0
└── fp16: False

For Development:
├── Use whisper-tiny
├── Batch size: 8
├── num_workers: 4
├── fp16: True
└── Gradient accumulation: 1

For Production:
├── Use whisper-base
├── Batch size: 16 (or higher if GPU memory allows)
├── num_workers: 8
├── fp16: True
├── Gradient accumulation: 2-4
└── Multi-GPU training: torch.nn.DataParallel
```

---

## 8. DATA FLOW DIAGRAM

```
┌────────────────────────────────────────────────────────────────────────┐
│                    STT TRAINING DATA PIPELINE                          │
└────────────────────────────────────────────────────────────────────────┘

INPUT LAYER
┌─────────────────────────────────────────────────────────────────────┐
│                    SpeechOcean762 Dataset                            │
│  ├─ Audio: 320 KB/sample (16kHz, 16-bit, mono)                     │
│  ├─ Text: 0.04 KB/sample                                            │
│  └─ Scores: 0.05 KB/sample                                          │
│                                                                     │
│  Splits: Train 3-5K samples | Test 500-1K samples                 │
│  Total Size: 500-1000 MB                                           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
PREPROCESSING LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  Audio Processing                   Text Processing                 │
│  ├─ Resample to 16kHz               ├─ BPE Tokenization            │
│  ├─ Normalize [-1, 1]               ├─ Map to Token IDs            │
│  └─ Pad to 30s (480K samples)       └─ Pad to 448 tokens           │
│                                                                     │
│  Output: 1.92 MB per sample         Output: 3.6 KB per sample      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
FEATURE EXTRACTION LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  Mel-Spectrogram Extraction                                         │
│  ├─ Input: 480K audio samples @ 16kHz                             │
│  ├─ FFT → Power Spectrum → Mel-Filter Banks → Log Scale           │
│  └─ Output: 80 Mel bins × 3000 time steps (padded)                │
│                                                                     │
│  Features: (80, 3000) = 960 KB per sample                          │
│  Data Type: float32                                                │
│  Range: [-20, 10] dB                                               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
DATA COLLATION LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  Batch Assembly (batch_size = 8)                                    │
│  ├─ Stack Features: (8, 80, 3000) → 7.68 MB                        │
│  ├─ Stack Labels: (8, 448) → 28.8 KB                               │
│  ├─ Create Masks: (8, 448) → 28.8 KB                               │
│  └─ Total Batch Size: 7.74 MB                                      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
MODEL LAYER (Whisper Encoder-Decoder)
┌─────────────────────────────────────────────────────────────────────┐
│  Encoder (Audio Processing)                                         │
│  ├─ Input: (8, 80, 3000) Mel features                              │
│  ├─ 4-12 Transformer Layers                                        │
│  └─ Output: (8, 3000, 512) Encoded features                        │
│                                                                     │
│  Decoder (Text Generation)                                          │
│  ├─ Input: (8, 448) Token embeddings + encoder output             │
│  ├─ 4-12 Transformer Layers with Cross-Attention                  │
│  └─ Output: (8, 448, 51865) Logits per token                      │
│                                                                     │
│  Model Size: 39M (tiny) - 1550M (large-v2) parameters             │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
OUTPUT LAYER
┌─────────────────────────────────────────────────────────────────────┐
│  Training Output                   Inference Output                 │
│  ├─ Loss: Scalar value             ├─ Greedy Decoding              │
│  ├─ Gradients: Backpropagated      ├─ Argmax → Token IDs           │
│  └─ Model Update: Parameters       └─ Decode → Text                │
│                                                                     │
│  Checkpoints:                                                       │
│  ├─ Model Weights: 300-1500 MB                                     │
│  ├─ Optimizer State: 300-1500 MB                                   │
│  ├─ Metrics: WER, BLEU, Char Acc                                   │
│  └─ Training Logs: 100-500 KB                                      │
└─────────────────────────────────────────────────────────────────────┘

METRICS & EVALUATION
┌─────────────────────────────────────────────────────────────────────┐
│  Per Epoch:                                                         │
│  ├─ Validation WER: [0.0-1.0] (lower is better)                   │
│  ├─ BLEU Score: [0.0-1.0] (higher is better)                      │
│  ├─ Character Accuracy: [0.0-1.0] (higher is better)              │
│  └─ Learning Rate: Scheduled decay                                │
│                                                                     │
│  Example Results:                                                   │
│  ├─ Before Fine-tuning: WER = 0.25                                │
│  └─ After Fine-tuning: WER = 0.12 (52% improvement)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. QUICK REFERENCE

### Input Summary
```
Per Sample:
├── Audio: 320 KB (16kHz, 16-bit, mono)
├── Text: 0.04 KB (English transcription)
└── Scores: 0.05 KB (4 float values)

Full Dataset:
├── Quick Test: 150 samples = 48 MB
├── Development: 3,500 samples = 1.12 GB
└── Production: 6,000 samples = 1.92 GB
```

### Processing Summary
```
Per Sample Through Pipeline:
├── Audio → Features: 320 KB → 960 KB (3x expansion)
├── Text → Tokens: 0.04 KB → 3.6 KB (90x expansion)
├── Processing Time: 50-180 ms per sample
└── GPU Memory (batch): 7.74 MB per batch

Model Processing:
├── Input: (8, 80, 3000) features
├── Output: (8, 448, 51865) logits
└── Model Size: 39M-1550M parameters
```

### Output Summary
```
Per Checkpoint:
├── Model Weights: 300-1500 MB
├── Optimizer State: 300-1500 MB
├── Metrics: 10-100 KB
└── Logs: 50-500 KB

Training Completion:
├── Best Model: Selected checkpoint
├── Evaluation Results: WER, BLEU, Accuracy
├── Sample Predictions: 10+ examples
└── Total Duration: 2-5 minutes (quick test) to 2-4 hours (production)
```

---

## 10. TROUBLESHOOTING COMMON ISSUES

### Issue: "Out of Memory" Error

```
Symptoms:
├── CUDA out of memory during training
├── Error message mentions GPU memory
└── Training crashes after 10-20 steps

Solutions:
├── Reduce batch_size: 16 → 8 → 4
├── Enable gradient_accumulation_steps: 1 → 2 → 4
├── Use mixed precision (fp16: True)
├── Reduce model size: base → tiny
└── Check: nvidia-smi to verify GPU memory
```

### Issue: "Data Loading Too Slow"

```
Symptoms:
├── GPU underutilized during training
├── Training loop spends >50% time loading data
└── num_workers warning in logs

Solutions:
├── Increase num_workers: 0 → 4 → 8
├── Enable pin_memory: True in DataLoader
├── Prefetch data during computation
├── Use SSD instead of HDD for dataset
└── Monitor: Check CPU usage during loading
```

### Issue: "Poor Model Performance"

```
Symptoms:
├── WER not decreasing during training
├── BLEU score stays low
├── Model outputs gibberish

Solutions:
├── Check data quality: Verify audio files
├── Verify transcriptions: Correct encoding
├── Increase training samples: Use full dataset
├── Increase epochs: 1 → 3 → 5
├── Adjust learning rate: 1e-5 → 1e-4 or 5e-6
├── Check: Review sample predictions
```

---

## 11. CONCLUSION

The STT training pipeline efficiently transforms raw audio and text data into a fine-tuned Whisper model through:

1. **Input**: 320 KB audio + 0.04 KB text per sample
2. **Processing**: Resampling → Feature extraction → Tokenization
3. **Output**: 960 KB features + 3.6 KB tokens per sample
4. **Model Training**: Encoder-decoder architecture with cross-attention
5. **Results**: Fine-tuned model checkpoint (300-1500 MB)

Key metrics for success:
- ✅ WER improvement: 40-60% typical improvement
- ✅ Training time: 2 minutes (quick test) to 4 hours (production)
- ✅ GPU memory: 4-16 GB depending on configuration
- ✅ Final model size: Efficient enough for inference (under 2 GB)

For more details, refer to the README.md and individual module documentation.
