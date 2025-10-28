# Model Architecture Visualization

## High-Level Overview

```
┌────────────────────────────────────────────────────────────────────┐
│          WhisperPronunciationAssessmentModel                       │
│                  (WhisperForConditionalGeneration)                 │
└────────────────────────────────────────────────────────────────────┘
                            ▲
                            │
                    Input Audio (WAV)
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│          Mel-Spectrogram Processor                                 │
│     (librosa or WhisperProcessor)                                  │
│     Input:  Audio array                                            │
│     Output: [batch, 80, 3000] mel-spectrogram                      │
└────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                    ENCODER                                         │
│     (Whisper Encoder - frozen optional)                            │
│                                                                    │
│     Input:  [batch, 80, 3000]                                     │
│     Output: [batch, seq_len, 512]  (hidden states)                │
│                                                                    │
│     seq_len ≈ (3000 - 400) / 160 ≈ 16-18 frames                   │
└────────────────────────────────────────────────────────────────────┘
                    │                        │
          ┌─────────┴────────┬──────────┬────┴─────────┐
          │                  │          │              │
          ▼                  ▼          ▼              ▼
    ┌─────────────┐   ┌──────────┐ ┌────────────┐ ┌────────────────┐
    │ DECODER     │   │ Mean     │ │ Assessment │ │ Assessment     │
    │ (Full Seq)  │   │ Pooling  │ │ Heads      │ │ Heads          │
    │             │   │ (Utt)    │ │ (Frame)    │ │ (Utterance)    │
    │ Transcribe  │   │          │ │            │ │                │
    └─────────────┘   └──────────┘ └────────────┘ └────────────────┘
          │                 │             │              │
          ▼                 ▼             ▼              ▼
    ┌─────────────┐   ┌──────────┐ ┌────────────┐ ┌────────────────┐
    │ Transcription    │[batch]  │ │[batch,seq]│ │ [batch]        │
    │ Logits      │   │ Mean     │ │ Accuracy  │ │ Accuracy       │
    │             │   │ Hidden   │ │ Stress    │ │ Fluency        │
    │ [batch,     │   │          │ │ Total     │ │ Prosodic       │
    │  seq,vocab] │   │          │ │ Phone Acc │ │ Completeness   │
    └─────────────┘   └──────────┘ └────────────┘ │ Total          │
                                                    └────────────────┘
          │                                              │
          ▼                                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    OUTPUTS                                  │
    ├─────────────────────────────────────────────────────────────┤
    │ Transcription:  "WE CALL IT BEAR"                           │
    │ Word Scores:    [10, 10, 9, 8] (per word)                   │
    │ Phone Scores:   [2, 2, 2, 2, ...] (per phone)               │
    │ Utterance:      accuracy=8, fluency=9, prosodic=8, etc.     │
    └─────────────────────────────────────────────────────────────┘
```

## Detailed Forward Pass Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ FORWARD PASS FLOW                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INPUT PREPARATION                                           │
│     ├─ input_features: [batch=1, 80, 3000]                     │
│     └─ device: "cuda" or "cpu"                                 │
│                                                                 │
│  2. ENCODER STAGE                                               │
│     ├─ Process mel-spectrogram                                 │
│     ├─ Extract acoustic features at 20ms hops                  │
│     ├─ Hidden states: [1, ~3000/160, 512]                      │
│     │                 = [1, ~18-20, 512]                       │
│     └─ Output: encoder_hidden_states, encoder_mean             │
│                                                                 │
│  3. PARALLEL PROCESSING (No sequential dependency)             │
│                                                                 │
│     ┌─ PATH A: TRANSCRIPTION                                   │
│     │  ├─ Input: encoder_hidden_states [1, 18, 512]           │
│     │  │         decoder_input_ids [1, seq_len] (training)    │
│     │  ├─ Decoder: Generate tokens from encoder context       │
│     │  ├─ Output: decoder_hidden_states [1, seq_len, 512]     │
│     │  ├─ LM Head: Convert hidden → vocabulary predictions    │
│     │  │  lm_logits = self.model.lm_head(hidden_states)      │
│     │  │  Shape: [1, seq_len, vocab_size=51864]              │
│     │  └─ Training Loss: CrossEntropyLoss with label shift    │
│     │     • Shifted logits:  [1, seq_len-1, vocab_size]      │
│     │     • Shifted labels:  [1, seq_len-1]                  │
│     │     • Predicts token[i+1] from token[i] (causal)       │
│     │                                                           │
│     ├─ PATH B: FRAME-LEVEL SCORES                             │
│     │  ├─ Input: encoder_hidden_states [1, 18, 512]           │
│     │  ├─ Word Accuracy Head:                                  │
│     │  │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│     │  │   Output: [1, 18]                                     │
│     │  ├─ Word Stress Head:                                    │
│     │  │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│     │  │   Output: [1, 18]                                     │
│     │  ├─ Word Total Head:                                     │
│     │  │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│     │  │   Output: [1, 18]                                     │
│     │  └─ Phone Accuracy Head:                                 │
│     │      fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│     │      Output: [1, 18]                                     │
│     │                                                           │
│     └─ PATH C: UTTERANCE-LEVEL SCORES                          │
│        ├─ Input: encoder_mean [1, 512]  (mean of [1, 18, 512])│
│        ├─ Utterance Accuracy Head:                             │
│        │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│        │   Output: [1]                                         │
│        ├─ Utterance Fluency Head:                              │
│        │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│        │   Output: [1]                                         │
│        ├─ Utterance Prosodic Head:                             │
│        │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│        │   Output: [1]                                         │
│        ├─ Utterance Completeness Head:                         │
│        │   fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│        │   Output: [1]                                         │
│        └─ Utterance Total Head:                                │
│            fc1[512→256] + ReLU + dropout → fc2[256→1]         │
│            Output: [1]                                         │
│                                                                 │
│  4. OUTPUT DICTIONARY                                           │
│     {                                                           │
│       'encoder_hidden_states': [1, 18, 512],                   │
│       'transcription_logits': [1, seq_len, 51864] (vocab_size) │
│         ├─ vocab_size = 51864 (Whisper tokenizer)              │
│         ├─ Training: seq_len from decoder_input_ids            │
│         ├─ Inference: generated via model.generate()           │
│         └─ Loss: CrossEntropyLoss after label shifting         │
│       'word_accuracy_logits': [1, 18],                         │
│       'word_stress_logits': [1, 18],                           │
│       'word_total_logits': [1, 18],                            │
│       'phone_accuracy_logits': [1, 18],                        │
│       'utterance_accuracy_logits': [1],                        │
│       'utterance_fluency_logits': [1],                         │
│       'utterance_prosodic_logits': [1],                        │
│       'utterance_completeness_logits': [1],                    │
│       'utterance_total_logits': [1]                            │
│     }                                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Assessment Head Architecture Detail

### FrameLevelAssessmentHead

```
Input: [batch, seq_len, 512] (encoder hidden states)
                │
                ▼
         ┌──────────────┐
         │  fc1: 512→256│
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │   ReLU       │
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  Dropout(0.1)│
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  fc2: 256→1  │
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  Squeeze(-1) │
         └──────────────┘
                │
                ▼
Output: [batch, seq_len] (one score per frame)
```

### UtteranceLevelAssessmentHead

```
Input: [batch, 512] (mean-pooled encoder hidden states)
                │
                ▼
         ┌──────────────┐
         │  fc1: 512→256│
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │   ReLU       │
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  Dropout(0.1)│
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  fc2: 256→1  │
         └──────────────┘
                │
                ▼
         ┌──────────────┐
         │  Squeeze(-1) │
         └──────────────┘
                │
                ▼
Output: [batch] (one score per utterance)
```

## Training Loss Computation

```
┌────────────────────────────────────────────────────────────┐
│              MULTI-OBJECTIVE TRAINING LOSS                 │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Total Loss = w_transcription × L_transcription (CE)      │
│             + w_word_acc × L_word_accuracy (MSE)          │
│             + w_word_stress × L_word_stress (MSE)         │
│             + w_word_total × L_word_total (MSE)           │
│             + w_phone_acc × L_phone_accuracy (MSE)        │
│             + w_utt_acc × L_utterance_accuracy (MSE)      │
│             + w_utt_flu × L_utterance_fluency (MSE)       │
│             + w_utt_pro × L_utterance_prosodic (MSE)      │
│             + w_utt_com × L_utterance_completeness (MSE)  │
│             + w_utt_tot × L_utterance_total (MSE)         │
│                                                            │
│  Where:                                                    │
│  - w_* are configurable weights from training_config      │
│  - L_transcription uses CrossEntropyLoss with:            │
│    • Label shifting (predict next token, causal LM)       │
│    • Padding token masking (ignore_index=-100)            │
│  - L_* (assessment) are MSE losses                        │
│  - Default: w_transcription=1.0, w_assessment_metrics=1.0 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Memory Layout

```
Batch Processing (batch_size=8):

Input Features:
  Shape: [8, 80, 3000]
  Size: 8 × 80 × 3000 × 4 bytes = ~7.7 MB

Encoder Hidden States:
  Shape: [8, 18, 512]
  Size: 8 × 18 × 512 × 4 bytes ≈ 295 KB

Assessment Logits (per type):
  Shape: [8, 18] (frame-level) or [8] (utterance-level)
  Size: 8 × 18 × 4 bytes ≈ 576 bytes (frame-level)

Total Model Parameters:
  Whisper-base:     74 M parameters
  Assessment heads: ~2-3 M parameters
  Total:            ~76-77 M parameters
  
  GPU memory: ~4-6 GB (depending on batch size)
```

## Processing Timeline

```
Inference Time per 10-second Audio:

Operation                          Time (GPU)    Time (CPU)
────────────────────────────────────────────────────────
Audio Loading & Preprocessing       ~10 ms       ~50 ms
Mel-Spectrogram Extraction          ~5 ms        ~20 ms
Encoder Forward Pass                ~30 ms       ~200 ms
Transcription Generation            ~20 ms       ~150 ms
Assessment Heads Forward Pass       ~5 ms        ~30 ms
Post-processing & Output            ~5 ms        ~10 ms
────────────────────────────────────────────────────────
Total (GPU)                         ~75 ms
Total (CPU)                         ~460 ms
```

## Comparison: Before vs After

```
BEFORE (WhisperModel only):
┌──────────────────────────────┐
│ Audio → Encoder → Assessment │
└──────────────────────────────┘
        No Transcription!

AFTER (WhisperForConditionalGeneration):
┌──────────────────────────────────────────────────┐
│ Audio → Encoder ├─→ Decoder → Transcription     │
│                 │                                │
│                 └─→ Assessment Heads → Scores   │
└──────────────────────────────────────────────────┘
      Both Transcription + Assessment!
```

## Data Type Sizes

```
Input Sizes:
- Mel-spectrogram [1, 80, 3000]: float32 = 960 KB
- Batch [8, 80, 3000]: float32 = 7.7 MB

Hidden States:
- Encoder [1, 18, 512]: float32 = 36 KB
- Batch [8, 18, 512]: float32 = 294 KB

Output Sizes:
- Frame scores [1, 18]: float32 = 72 bytes
- Utterance scores [1]: float32 = 4 bytes
- Transcription logits [1, 100, 51864]: float32 = ~20 MB (variable)
```

## Training Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING DATA BATCH                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Batch Dictionary Keys:                                     │
│  ├─ 'input_features': [batch, 80, 3000]                    │
│  │  └─ Mel-spectrogram from audio loader                   │
│  │                                                          │
│  ├─ 'decoder_input_ids': [batch, seq_len]                  │
│  │  ├─ Token IDs for decoder input (teacher forcing)       │
│  │  ├─ Example: [2, 1234, 5678, 9012, ...] per utterance   │
│  │  └─ Used for transcription training (shifted for loss)  │
│  │                                                          │
│  ├─ 'labels': [batch, seq_len]                             │
│  │  ├─ Token IDs for transcription ground truth            │
│  │  ├─ Padding: -100 (ignored in loss computation)         │
│  │  └─ Shifted by 1 during loss computation (i+1)          │
│  │                                                          │
│  ├─ 'word_accuracy_scores': [batch, var_len]               │
│  │  ├─ Variable length per example (word count varies)     │
│  │  ├─ Handled with padding or masking in trainer          │
│  │  └─ Range: [0.0, 10.0] (pronunciation score)            │
│  │                                                          │
│  ├─ 'word_stress_scores': [batch, var_len]                 │
│  │  └─ Similar to word_accuracy                            │
│  │                                                          │
│  ├─ 'word_total_scores': [batch, var_len]                  │
│  │  └─ Overall word-level score                            │
│  │                                                          │
│  ├─ 'phone_accuracy_scores': [batch, var_len]              │
│  │  ├─ Phone-level scores (finer granularity)              │
│  │  └─ More variable length than words                     │
│  │                                                          │
│  ├─ 'utterance_accuracy': [batch]                          │
│  │  ├─ Fixed size: one score per utterance                 │
│  │  └─ Range: [0.0, 10.0]                                  │
│  │                                                          │
│  ├─ 'utterance_fluency': [batch]                           │
│  │  └─ Fluency metric                                      │
│  │                                                          │
│  ├─ 'utterance_prosodic': [batch]                          │
│  │  └─ Prosody/intonation metric                           │
│  │                                                          │
│  ├─ 'utterance_completeness': [batch]                      │
│  │  └─ Completeness metric (usually ~9.6-10.0)             │
│  │                                                          │
│  └─ 'utterance_total': [batch]                             │
│     └─ Overall utterance score                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Training Epoch Flow

```
┌─────────────────────────────────────────────────────────────┐
│              TRAINING EPOCH EXECUTION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. For each batch in train_loader:                        │
│     ├─ Move batch tensors to device (GPU/CPU)              │
│     └─ batch = {k: v.to(device) for k, v in batch}        │
│                                                             │
│  2. Forward Pass:                                           │
│     ├─ model(input_features, decoder_input_ids=...)        │
│     └─ Returns: predictions dict with all logits           │
│                                                             │
│  3. Compute Loss (Multi-objective):                         │
│     ├─ Transcription (CrossEntropyLoss):                   │
│     │  ├─ logits [batch, seq-1, vocab] from shift          │
│     │  ├─ labels [batch, seq-1] (shifted)                  │
│     │  └─ loss_ce = CE(logits.flatten, labels.flatten)     │
│     │                                                       │
│     ├─ Assessment losses (MSE):                            │
│     │  ├─ word_accuracy: MSE(pred_scores, target_scores)   │
│     │  ├─ word_stress: MSE(pred_scores, target_scores)     │
│     │  ├─ word_total: MSE(pred_scores, target_scores)      │
│     │  ├─ phone_accuracy: MSE(pred_scores, target_scores)  │
│     │  ├─ utterance_accuracy: MSE(pred_score, target)      │
│     │  ├─ utterance_fluency: MSE(pred_score, target)       │
│     │  ├─ utterance_prosodic: MSE(pred_score, target)      │
│     │  ├─ utterance_completeness: MSE(pred_score, target)  │
│     │  └─ utterance_total: MSE(pred_score, target)         │
│     │                                                       │
│     └─ Total Loss = Σ(weight[i] × loss[i])                │
│        └─ Weights from config.loss_weights dict            │
│                                                             │
│  4. Backward Pass & Optimization:                          │
│     ├─ loss.backward()  (compute gradients)                │
│     ├─ clip_grad_norm_(model.parameters(), 1.0)            │
│     ├─ optimizer.step()  (update weights)                  │
│     └─ scheduler.step()  (update learning rate)            │
│                                                             │
│  5. Logging:                                                │
│     ├─ Every logging_steps: print loss & LR                │
│     └─ Progress bar shows: loss, current LR                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
