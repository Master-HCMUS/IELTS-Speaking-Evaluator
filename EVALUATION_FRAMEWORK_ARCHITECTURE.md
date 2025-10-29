# Evaluation Framework Architecture Guide

## Document Structure

This guide complements `EVALUATION_METHODS_COMPARISON.md` with detailed technical implementation aspects of each evaluation framework.

---

## Directory Structure

```
src/evaluation/
├── __init__.py                              # Package exports
│
├── azure_speech_evaluation/
│   ├── __init__.py
│   ├── core.py                              # SpeechOcean762Evaluator (639 lines)
│   └── README.md
│
├── stt_whisper_evaluation/
│   ├── __init__.py
│   ├── core.py                              # StandaloneWhisperModelEvaluator (825 lines)
│   └── README.md
│
└── multi_objective_whisper_evaluation/
    ├── __init__.py
    ├── core.py                              # MultiObjectiveWhisperModelEvaluator (859 lines)
    ├── cli.py                               # Command-line interface
    └── README.md
```

---

## Core Classes & Interfaces

### Azure Speech Evaluation Module

#### Class Hierarchy

```
SpeechOcean762Evaluator
├── config_manager: ConfigManager
├── pronunciation_service: AzureSpeechPronunciationService
├── dataset: HuggingFace Dataset
└── evaluation_results: List[Dict]

EvaluationMetrics (dataclass)
├── Correlations: 4 Pearson r values
├── MAE: 4 error values
├── RMSE: 4 error values
└── Statistics: 4×5 metrics (mean, std, min, max, median)
```

#### Key Methods

```python
class SpeechOcean762Evaluator:
    
    def load_dataset(split: str, max_samples: int) -> bool
        """Load SpeechOcean762 from HuggingFace"""
    
    def _normalize_azure_scores(azure_result: Dict) -> Dict[str, float]
        """Convert 0-100 scale → 0-10 scale"""
        # accuracy: / 100 * 10
        # fluency: / 100 * 10
        # completeness: / 100 * 10
        # prosodic: fluency / 100 * 10 (PROXY)
    
    def evaluate_sample(sample: Dict, idx: int) -> Dict
        """Single sample evaluation:
        1. Save audio to temp file
        2. Call Azure Speech API
        3. Normalize scores
        4. Extract expert scores
        5. Return results
        """
    
    def run_evaluation(max_samples: int, save_results: bool) -> EvaluationMetrics
        """Orchestrate full evaluation:
        1. Iterate through dataset
        2. Evaluate each sample
        3. Calculate aggregated metrics
        4. Save to JSON
        """
    
    def _calculate_metrics() -> EvaluationMetrics
        """Compute:
        - Pearson correlation (4 metrics)
        - MAE and RMSE (4 metrics each)
        - Score distributions (5 stats × 4 metrics × 2 sources)
        """
    
    def _save_evaluation_results(metrics: EvaluationMetrics) -> None
        """Save to JSON and CSV in evaluation_results/"""
```

#### Result Format

```python
{
    'sample_idx': int,
    'text': str,
    'speaker': str,
    'gender': str,
    'age': int,
    'success': bool,
    'assessment_time': float,
    'expert_scores': {'accuracy': float, 'fluency': float, ...},
    'azure_scores': {'accuracy': float, 'fluency': float, ...},
    'score_differences': {'accuracy': float, ...},
    'azure_raw_scores': {...},
    'word_level_scores': [...],
    'recognized_text': str,
    'error': str
}
```

#### Dependencies

```
Core:
  - azure-cognitiveservices-speech (Azure SDK)
  - datasets (HuggingFace)
  - numpy, scipy, pandas

Custom:
  - pronunciation_service.AzureSpeechPronunciationService
  - config_manager.ConfigManager
```

---

### STT Whisper Evaluation Module

#### Class Hierarchy

```
StandaloneWhisperModelEvaluator
├── model_path: Path
├── assessment: StandaloneWhisperPronunciationAssessor
├── dataset: HuggingFace Dataset
└── evaluation_results: List[WhisperEvaluationResult]

StandaloneWhisperPronunciationAssessor
├── model: WhisperForConditionalGeneration
├── processor: WhisperProcessor
├── feature_extractor: WhisperFeatureExtractor
├── wer_metric: evaluate.Metric
└── bleu_metric: evaluate.Metric

StandaloneEvaluationMetrics (dataclass)
├── Correlations: 4 Pearson r values
├── MAE: 4 error values
├── RMSE: 4 error values
└── Statistics: 4×5 metrics (expert) + 4×5 (whisper)

WhisperEvaluationResult (dataclass)
├── Identifiers: model_path, sample_idx, text, speaker
├── Transcription: predicted_text, reference_text
├── Metrics: wer, cer, bleu, word_error_rate, etc.
├── Scores: pronunciation_scores, expert_scores (derived)
└── Error: error message
```

#### Key Methods

```python
class StandaloneWhisperPronunciationAssessor:
    
    def transcribe_audio(audio_path: str, reference_text: str = "") -> Dict
        """
        1. Load audio with librosa
        2. Extract mel-spectrogram
        3. Run Whisper encoder-decoder
        4. Get transcription
        5. Calculate quality metrics
        Returns: {
            'predicted_text': str,
            'reference_text': str,
            'quality_metrics': {'wer', 'cer', 'bleu', 'word_accuracy', ...},
            'confidence_score': float
        }
        """
    
    def _calculate_quality_metrics(predicted: str, reference: str) -> Dict
        """
        Compute:
        - WER (evaluate.load("wer"))
        - CER (custom edit distance)
        - BLEU (evaluate.load("bleu"))
        - Length ratio
        - Overall quality (weighted composite)
        """
    
    def _calculate_cer(predicted: str, reference: str) -> float
        """
        Dynamic programming edit distance (Levenshtein):
        dp[i][j] = minimum edit distance between first i ref chars, j pred chars
        """
    
    def assess_pronunciation(transcription_result: Dict) -> Dict[str, float]
        """
        Map transcription quality to pronunciation scores:
        - accuracy = (0.7 * word_acc + 0.3 * char_acc) * 10
        - fluency = (0.6 * bleu + 0.4 * confidence) * 10
        - completeness = (0.8 * length_ratio + 0.2 * word_acc) * 10
        - prosodic = (0.7 * overall_quality + 0.3 * confidence) * 10
        
        All clamped to [0, 10]
        """

class StandaloneWhisperModelEvaluator:
    
    def evaluate_sample(sample: Dict, idx: int) -> WhisperEvaluationResult
        """
        1. Save audio to temp file
        2. Transcribe with Whisper
        3. Calculate quality metrics
        4. Derive pronunciation scores
        5. Collect expert annotations
        6. Return WhisperEvaluationResult
        """
    
    def run_evaluation(max_samples: int, save_results: bool) -> StandaloneEvaluationMetrics
        """Full evaluation orchestration"""
    
    def _calculate_metrics() -> StandaloneEvaluationMetrics
        """Same as Azure: correlations, MAE, RMSE, statistics"""
```

#### Score Derivation Pipeline

```
Audio
  ↓
[Mel-Spectrogram Extraction]
  ↓
[Whisper Encoder → Hidden States]
  ↓
[Whisper Decoder → Token IDs]
  ↓
[Detokenize → Text]
  ↓
[Compare with Reference]
  ├→ WER (word-level edit distance)
  ├→ CER (character-level edit distance)
  ├→ BLEU (n-gram precision)
  └→ Length Ratio
  ↓
[Quality Score = 0.4*WER_acc + 0.3*CER_acc + 0.2*BLEU + 0.1*Length]
  ↓
[Weighted Combinations]
  ├→ Accuracy = (0.7*WER_acc + 0.3*CER_acc) × 10
  ├→ Fluency = (0.6*BLEU + 0.4*Confidence) × 10
  ├→ Completeness = (0.8*Length + 0.2*WER_acc) × 10
  └→ Prosodic = (0.7*Quality + 0.3*Confidence) × 10
  ↓
[Clamp to [0, 10]]
  ↓
Pronunciation Scores (4 dimensions)
```

#### Result Format

```python
WhisperEvaluationResult(
    model_path: str,
    sample_idx: int,
    text: str,
    speaker: str,
    success: bool,
    predicted_text: str,
    reference_text: str,
    confidence_score: float,
    transcription_quality: float,
    pronunciation_scores: {
        'accuracy': float,
        'fluency': float,
        'completeness': float,
        'prosodic': float
    },
    expert_scores: {...},
    word_error_rate: float,
    character_error_rate: float,
    bleu_score: float,
    error: str
)
```

#### Dependencies

```
Core:
  - transformers (Whisper model)
  - librosa (audio processing)
  - soundfile (WAV I/O)
  - evaluate (WER, BLEU)
  - datasets (HuggingFace)
  - numpy, scipy, pandas, torch

No Custom:
  - All logic self-contained
```

---

### Multi-Objective Whisper Evaluation Module

#### Class Hierarchy

```
MultiObjectiveWhisperModelEvaluator
├── model_path: Path
├── assessor: MultiObjectiveWhisperAssessor
├── dataset: HuggingFace Dataset
└── evaluation_results: List[MultiObjectiveEvaluationResult]

MultiObjectiveWhisperAssessor
├── model: WhisperPronunciationAssessmentModel
├── processor: WhisperProcessor
├── feature_extractor: WhisperFeatureExtractor
├── wer_metric: evaluate.Metric
└── bleu_metric: evaluate.Metric

MultiObjectiveEvaluationMetrics (dataclass)
├── Transcription: wer, cer, bleu
├── Correlations: 4 Pearson r values
├── MAE: 4 error values
├── RMSE: 4 error values
└── Statistics: 4×5 (expert) + 4×5 (model)

MultiObjectiveEvaluationResult (dataclass)
├── Identifiers: model_path, sample_idx, text, speaker
├── Transcription: predicted_text, reference_text
├── Transcription Metrics: wer, cer, bleu
├── Assessment: model_scores, expert_scores
└── Error: error message
```

#### Key Methods

```python
class MultiObjectiveWhisperAssessor:
    
    def _load_model() -> None
        """
        Lazy initialization:
        1. Load WhisperForConditionalGeneration from HF
        2. Detect checkpoint format (.pt file vs HF model)
        3. If .pt file: torch.load() and load_state_dict()
        4. Initialize 9 assessment heads
        5. Move to device
        
        Files handled:
        - .pt checkpoint from training
        - model/directory from HF
        """
    
    def transcribe_and_assess(audio_path: str, reference_text: str) -> Dict
        """
        1. Load audio → mel-spectrogram
        2. Forward pass through model
        3. Get transcription logits
        4. Get 9 assessment logits (word, phone, utterance)
        5. Decode transcription
        6. Calculate WER, CER, BLEU
        7. Extract utterance-level scores
        8. Normalize to [0, 10]
        9. Return results
        """
    
    def _calculate_transcription_metrics(predicted: str, ref: str) -> Dict
        """Same as STT Whisper: WER, CER, BLEU, etc."""
    
    def _extract_assessment_scores(assessment_scores: Dict) -> Dict
        """
        Extract from model output:
        - utterance_level.accuracy: [batch] → mean → normalize
        - utterance_level.fluency: [batch] → mean → normalize
        - utterance_level.completeness: [batch] → mean → normalize
        - utterance_level.prosodic: [batch] → mean → normalize
        
        Available but unused:
        - Word-level: word_accuracy, word_stress, word_total
        - Phone-level: phone_accuracy
        - Frame aggregations
        """

class MultiObjectiveWhisperModelEvaluator:
    
    def evaluate_sample(sample: Dict, idx: int) -> MultiObjectiveEvaluationResult
        """
        1. Save audio to temp file
        2. Transcribe and assess (single forward pass)
        3. Extract transcription metrics
        4. Extract assessment scores
        5. Collect expert annotations
        6. Return MultiObjectiveEvaluationResult
        """
    
    def run_evaluation(max_samples: int, save_results: bool) -> MultiObjectiveEvaluationMetrics
        """Full evaluation orchestration"""
    
    def _calculate_metrics() -> MultiObjectiveEvaluationMetrics
        """Enhanced version of Azure/STT - adds transcription metrics"""
```

#### Model Architecture Details

**WhisperPronunciationAssessmentModel:**

```
Input: mel-spectrogram [batch, 80, 3000]
  ↓
[Whisper Encoder]  → encoder_last_hidden [batch, ~1500, 512]
  ↓
  ├─→ [Decoder + LM Head] → transcription_logits [batch, seq, vocab_size]
  │
  └─→ [Mean Pooling] → encoder_mean [batch, 512]
      ↓
      ├─→ [UtteranceLevelAssessmentHead (accuracy)] → [batch] ∈ [0, 10]
      ├─→ [UtteranceLevelAssessmentHead (fluency)] → [batch] ∈ [0, 10]
      ├─→ [UtteranceLevelAssessmentHead (prosodic)] → [batch] ∈ [0, 10]
      ├─→ [UtteranceLevelAssessmentHead (completeness)] → [batch] ∈ [0, 10]
      └─→ [UtteranceLevelAssessmentHead (total)] → [batch] ∈ [0, 10]
  
  Frame-level assessments (also computed but unused in evaluation):
  ├─→ [FrameLevelAssessmentHead] → word_accuracy_logits [batch, seq_len]
  ├─→ [FrameLevelAssessmentHead] → word_stress_logits [batch, seq_len]
  ├─→ [FrameLevelAssessmentHead] → word_total_logits [batch, seq_len]
  └─→ [FrameLevelAssessmentHead] → phone_accuracy_logits [batch, seq_len]

Output Dictionary:
  {
    'transcription_logits': [...],
    'utterance_accuracy_logits': [...],
    'utterance_fluency_logits': [...],
    'utterance_prosodic_logits': [...],
    'utterance_completeness_logits': [...],
    'utterance_total_logits': [...],
    'word_accuracy_logits': [...],
    'word_stress_logits': [...],
    'word_total_logits': [...],
    'phone_accuracy_logits': [...]
  }
```

**Assessment Head Architecture (After Fixes):**

```
FrameLevelAssessmentHead:
  Input [batch, seq_len, hidden_dim]
  → Reshape to [batch*seq_len, hidden_dim]
  → Linear(hidden_dim, hidden_dim)
  → BatchNorm1d
  → ReLU
  → Dropout(0.2)
  → Linear(hidden_dim, hidden_dim//2)
  → BatchNorm1d
  → ReLU
  → Dropout(0.2)
  → Linear(hidden_dim//2, 1)
  → Sigmoid() × 10.0
  → Reshape to [batch, seq_len]
  Output ∈ [0, 10] for each frame

UtteranceLevelAssessmentHead:
  Input [batch, hidden_dim] (mean-pooled)
  → Linear(hidden_dim, hidden_dim)
  → BatchNorm1d
  → ReLU
  → Dropout(0.2)
  → Linear(hidden_dim, hidden_dim//2)
  → BatchNorm1d
  → ReLU
  → Dropout(0.2)
  → Linear(hidden_dim//2, 1)
  → Sigmoid() × 10.0
  Output [batch] ∈ [0, 10] per utterance
```

**Previous Issues (Now Fixed):**

❌ **Before:**
```python
self.fc2 = nn.Linear(hidden_dim, 1)
return x.squeeze(-1)  # Raw unbounded logits!
```

✅ **After:**
```python
self.fc3 = nn.Linear(hidden_dim // 2, 1)
self.sigmoid = nn.Sigmoid()
return self.sigmoid(x) * 10.0  # Scaled to [0, 10]
```

#### Result Format

```python
MultiObjectiveEvaluationResult(
    model_path: str,
    sample_idx: int,
    text: str,
    speaker: str,
    success: bool,
    predicted_text: str,
    reference_text: str,
    word_error_rate: float,
    character_error_rate: float,
    bleu_score: float,
    model_scores: {
        'accuracy': float ∈ [0, 10],
        'fluency': float ∈ [0, 10],
        'completeness': float ∈ [0, 10],
        'prosodic': float ∈ [0, 10]
    },
    expert_scores: {...},
    error: str
)
```

#### Dependencies

```
Core:
  - transformers (Whisper, AutoProcessor)
  - librosa, soundfile (audio)
  - evaluate (WER, BLEU)
  - datasets (HuggingFace)
  - numpy, scipy, pandas, torch

Custom:
  - whisper_pronunciation_model.WhisperPronunciationAssessmentModel
  (from finetuning_pronunciation_assessment package)
```

---

## Shared Components

### Common Utilities

#### convert_numpy_types(obj)

```python
def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native for JSON serialization."""
    # Handles: np.integer, np.floating, np.ndarray, dict, list
    # Returns: JSON-serializable structure
```

**Used in:** All three frameworks when saving results

#### Audio Processing

```python
def _save_audio_sample(audio_data: np.ndarray, sampling_rate: int) -> str:
    """Save numpy array as temporary WAV file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_path, audio_data, sampling_rate)
    return temp_path

def load_dataset(split: str = "test", max_samples: Optional[int] = None) -> bool:
    """Load SpeechOcean762 from HuggingFace."""
    dataset = load_dataset(
        "speechocean762",
        split=split,
        cache_dir="..."
    )
    if max_samples:
        dataset = dataset.select(range(max_samples))
    return dataset
```

**Used in:** All three frameworks

#### Metric Calculation Pattern

```python
def _calculate_metrics(self) -> MetricsDataclass:
    """Common pattern across all three frameworks:
    
    1. Filter successful results
    2. Extract scores for each dimension
    3. Calculate Pearson correlation (with NaN handling)
    4. Calculate MAE and RMSE
    5. Calculate score statistics (mean, std, min, max, median)
    6. Return metrics dataclass
    """
```

### Result Storage Format

All three save to:
```
evaluation_results/
├── {model_name}_evaluation_{timestamp}.json
└── {model_name}_summary_{timestamp}.csv  (Azure only)
```

**JSON Schema:**
```json
{
  "evaluation_info": {
    "timestamp": "YYYYMMDDhhmmss",
    "model_path": "...",
    "dataset": "speechocean762",
    "evaluation_type": "...",
    "total_samples": int,
    "successful_assessments": int,
    "failed_assessments": int
  },
  "metrics": {
    "transcription": {...},  // Multi-Objective only
    "correlations": {...},
    "mae": {...},
    "rmse": {...}
  },
  "score_statistics": {
    "expert": {...},
    "system": {...}  // azure_scores, whisper_scores, or model_scores
  },
  "individual_results": [...]  // Per-sample results
}
```

---

## Execution Flow Comparison

### Azure Speech: Sequential API Calls

```
Load Dataset (2500 samples)
  ↓
for each sample:
  ├─ Save audio to temp file
  ├─ Call Azure Speech API (1.5s latency)
  ├─ Wait for response
  ├─ Normalize scores
  ├─ Extract expert scores
  ├─ Store result
  └─ Delete temp file
  
Total: ~3750 seconds (~1 hour) for 2500 samples

Bottleneck: API round-trip latency
```

### STT Whisper: Single Model, Batch Processing

```
Load Dataset (2500 samples)
Load Whisper Model (once, 500MB)

for each sample:
  ├─ Save audio to temp file
  ├─ Load audio (librosa)
  ├─ Extract mel-spectrogram
  ├─ Run Whisper (GPU: 0.2s, CPU: 1s)
  ├─ Calculate WER, CER, BLEU
  ├─ Derive pronunciation scores
  ├─ Extract expert scores
  ├─ Store result
  └─ Delete temp file

Total: ~500-2500 seconds (~10-40 min) for 2500 samples

Bottleneck: Whisper inference time (GPU helps)
```

### Multi-Objective Whisper: Unified Multi-Task

```
Load Dataset (2500 samples)
Load Checkpoint + Assessment Heads (once, 600MB)

for each sample:
  ├─ Save audio to temp file
  ├─ Load audio (librosa)
  ├─ Extract mel-spectrogram
  ├─ Forward pass through full model:
  │  ├─ Encoder → hidden states
  │  ├─ Decoder → transcription logits
  │  ├─ 5 utterance assessment heads
  │  └─ All in single batch
  ├─ Extract transcription text
  ├─ Calculate WER, CER, BLEU
  ├─ Extract 4 assessment scores
  ├─ Extract expert scores
  ├─ Store result
  └─ Delete temp file

Total: ~600-2000 seconds (~10-30 min) for 2500 samples

Bottleneck: Model forward pass (less than STT due to shared encoder)
```

---

## Key Differences in Implementation

### Model Loading

| Framework | Approach | File Formats |
|-----------|----------|-------------|
| **Azure** | External API | N/A |
| **STT** | `transformers.from_pretrained()` | HF Hub only |
| **Multi** | Custom + `torch.load()` | HF Hub or .pt checkpoint |

**Multi-Objective specific handling:**
```python
def _load_model(self):
    model_path_str = str(self.model_path)
    is_pt_file = model_path_str.endswith('.pt')
    
    if is_pt_file:
        # Initialize model first
        self.model = WhisperPronunciationAssessmentModel(...)
        # Load checkpoint weights
        checkpoint = torch.load(model_path_str, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Standard HF loading
        self.model = WhisperPronunciationAssessmentModel.from_pretrained(model_path_str)
```

### Error Handling

| Framework | NaN Handling | Constant Value | Out-of-Range |
|-----------|------------|----------------|-------------|
| **Azure** | Set corr=0 | Set corr=0 if std=0 | Rare (API) |
| **STT** | Try-except → 0.0 | Not handled | Clamp [0,10] |
| **Multi** | Try-except → 0.0 | Not handled | Auto [0,10] via Sigmoid |

---

## Performance Considerations

### Memory Usage

| Framework | Model | Feature Extractor | Typical RAM | GPU VRAM |
|-----------|-------|-------------------|-------------|----------|
| **Azure** | ~0 (cloud) | ~0 | ~500MB | ~0 |
| **STT** | 500MB | 100MB | ~2GB | ~2GB |
| **Multi** | 600MB | 100MB | ~2.5GB | ~2.5GB |

### Speed (on V100 GPU, 2500 samples)

| Framework | Per-Sample | Total | Speedup |
|-----------|-----------|-------|---------|
| **Azure** | 1.5s (API) | ~1h | 1x (baseline) |
| **STT** | 0.2s (GPU) | ~10min | 6x |
| **Multi** | 0.25s (GPU) | ~10.5min | 5.7x |

### Accuracy (Expected After Multi-Obj Fixes)

| Dimension | Azure | STT | Multi |
|-----------|-------|-----|-------|
| **Accuracy** | 0.60 | 0.35 | 0.60 |
| **Fluency** | 0.55 | 0.20 | 0.55 |
| **Completeness** | 0.65 | 0.45 | 0.65 |
| **Prosodic** | 0.55 | 0.25 | 0.55 |
| **Average** | 0.59 | 0.31 | 0.59 |

---

## Integration Points

### For New Evaluators

All frameworks inherit from common patterns:

```python
class MyCustomEvaluator:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.dataset = None
        self.evaluation_results = []
    
    def load_dataset(self, split: str, max_samples: int) -> bool:
        """Follow SpeechOcean762Evaluator pattern"""
    
    def evaluate_sample(self, sample: Dict, idx: int) -> Result:
        """Return consistent result structure"""
    
    def run_evaluation(self, max_samples: int, save_results: bool) -> Metrics:
        """Iterate, calculate metrics, save"""
    
    def _calculate_metrics(self) -> Metrics:
        """Calculate Pearson r, MAE, RMSE, statistics"""
    
    def _save_evaluation_results(self, metrics: Metrics):
        """Save to evaluation_results/ directory"""
```

### Extending for New Metrics

To add new metrics (e.g., Spearman ρ instead of Pearson r):

1. Update `MetricsDataclass`:
   ```python
   @dataclass
   class EnhancedMetrics:
       pearson_correlations: {...}
       spearman_correlations: {...}  # NEW
   ```

2. Update `_calculate_metrics()`:
   ```python
   spearman_corr, _ = stats.spearmanr(expert_vals, system_vals)
   ```

3. Update result saving

4. Update comparison/interpretation

---

## Debugging & Troubleshooting

### Common Issues

#### Issue: NaN Correlations
**Cause:** Constant predictions (all zeros) or all expert scores same value
**Solution:** Check if `np.std(values) == 0` before correlation calculation

#### Issue: Model Loading Fails
**Solution (STT):** Use `transformers.WhisperForConditionalGeneration.from_pretrained()`
**Solution (Multi):** Provide `.pt` checkpoint with correct state_dict format

#### Issue: Unicode Errors on Windows
**Cause:** Emoji characters in progress output
**Solution:** Use ASCII alternatives ✅ (Already applied)

#### Issue: Out-of-Memory (OOM)
**Solution (STT):** Run on CPU (slow) or use smaller model (whisper-tiny)
**Solution (Multi):** Same as STT

#### Issue: Slow API Calls (Azure)
**Cause:** Network latency
**Solution:** Batch requests if API supports (Azure Speech doesn't), or accept latency

---

## Appendix: File Format Specifications

### Input: Audio
```
Format: WAV (PCM)
Sampling Rate: 16000 Hz (mono)
Duration: Varies (typically 5-30s)
Reference Text: English sentence (ASCII, varies per sample)
```

### Output: JSON
```json
{
  "evaluation_info": {
    "timestamp": "20251029_101237",
    "model_path": "path/to/model",
    "dataset": "speechocean762",
    "evaluation_type": "azure_speech|standalone_whisper|multi_objective_whisper",
    "total_samples": 2500,
    "successful_assessments": 2500,
    "failed_assessments": 0
  },
  "metrics": { ... },
  "score_statistics": { ... },
  "individual_results": [
    {
      "sample_idx": 0,
      "text": "Read the passage",
      "speaker": "speaker_0001",
      "success": true,
      "expert_scores": {
        "accuracy": 8.5,
        "fluency": 7.2,
        "completeness": 10.0,
        "prosodic": 8.1
      },
      "system_scores": { ... },
      "metrics": { ... }
    }
  ]
}
```

### Output: CSV (Azure only)

```csv
sample_idx,text,speaker,expert_accuracy,expert_fluency,azure_accuracy,azure_fluency,diff_accuracy,diff_fluency
0,"Read the passage",speaker_0001,8.5,7.2,7.8,6.5,0.7,0.7
1,...
```

---

## References

- **Whisper Paper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
- **SpeechOcean762**: Qin et al., 2021
- **Evaluation Metrics**: NIST Speech Recognition Evaluation Tools
