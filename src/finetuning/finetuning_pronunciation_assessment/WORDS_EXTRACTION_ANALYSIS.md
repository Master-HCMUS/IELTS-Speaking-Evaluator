# Analysis: Words Column Extraction & Trainer Usage

## 1. Data Extraction in data_processor.py

Looking at the **updated** `data_processor.py`, the "words" column is now **properly extracted**:

### Location: Lines 213-265 (NEW CODE)

```python
# ────────────────────────────────────────────────────────────────
# EXTRACT FROM "words" COLUMN (detailed word/phone level scores)
# ────────────────────────────────────────────────────────────────
if "words" in example and example["words"] is not None:
    try:
        words_data = example["words"]
        
        # Extract word-level scores
        word_accuracy_scores = []
        word_stress_scores = []
        word_total_scores = []
        phone_accuracy_scores = []
        
        for word_entry in words_data:
            # Word-level scores
            if "accuracy" in word_entry:
                word_accuracy_scores.append(float(word_entry["accuracy"]))
            if "stress" in word_entry:
                word_stress_scores.append(float(word_entry["stress"]))
            if "total" in word_entry:
                word_total_scores.append(float(word_entry["total"]))
            
            # Phone-level scores (flatten all phones)
            if "phones-accuracy" in word_entry:
                phones_acc = word_entry["phones-accuracy"]
                if isinstance(phones_acc, (list, tuple)):
                    phone_accuracy_scores.extend([float(p) for p in phones_acc])
        
        # Store extracted scores (will be normalized below)
        if word_accuracy_scores:
            result["word_accuracy_scores"] = np.array(word_accuracy_scores, dtype=np.float32)
        if word_stress_scores:
            result["word_stress_scores"] = np.array(word_stress_scores, dtype=np.float32)
        if word_total_scores:
            result["word_total_scores"] = np.array(word_total_scores, dtype=np.float32)
        if phone_accuracy_scores:
            result["phone_accuracy_scores"] = np.array(phone_accuracy_scores, dtype=np.float32)
```

**Output Dataset Columns Created:**
- ✅ `word_accuracy_scores` → numpy array (variable length per example)
- ✅ `word_stress_scores` → numpy array (variable length per example)
- ✅ `word_total_scores` → numpy array (variable length per example)
- ✅ `phone_accuracy_scores` → numpy array (variable length per example)

**Then normalized:**
- Lines 284-300: Normalize word-level scores [0,10] → [0,1]
- Lines 303-312: Normalize phone-level scores [0,2] → [0,1]

---

## 2. Data Flow to Trainer

### Step 1: Data Collator (data_collator.py)

The extracted scores are batched by `PronunciationAssessmentDataCollator`:

```python
# Handle word-level scores (variable length per example)
word_score_keys = ["word_accuracy_scores", "word_stress_scores", "word_total_scores"]
for key in word_score_keys:
    if key in batch[0]:
        word_scores = [ex[key] for ex in batch]
        # Stack as list of tensors (different lengths)
        collated[key] = [torch.from_numpy(np.array(s, dtype=np.float32)) 
                        for s in word_scores]

# Handle phone-level scores (variable length per example)
if "phone_accuracy_scores" in batch[0]:
    phone_scores = [ex["phone_accuracy_scores"] for ex in batch]
    collated["phone_accuracy_scores"] = [torch.from_numpy(np.array(s, dtype=np.float32)) 
                                         for s in phone_scores]
```

**Batch Format:**
```python
batch = {
    "input_features": [B, 80, 3000] tensor,
    "word_accuracy_scores": List[B] of variable-length tensors,
    "word_stress_scores": List[B] of variable-length tensors,
    "word_total_scores": List[B] of variable-length tensors,
    "phone_accuracy_scores": List[B] of variable-length tensors,
    ...
}
```

---

## 3. Usage in trainer.py - ANALYSIS

### Current Implementation (trainer.py, lines 114-139)

```python
# Word-level losses (variable length per example)
word_targets = {
    "word_accuracy_scores": ("word_accuracy_logits", "word_accuracy"),
    "word_stress_scores": ("word_stress_logits", "word_stress"),
    "word_total_scores": ("word_total_logits", "word_total"),
}

if self.config.use_word_level_assessment:
    for batch_key, (pred_key, weight_key) in word_targets.items():
        if batch_key in batch and batch[batch_key] is not None:
            if pred_key in predictions:
                # Handle variable-length word scores
                batch_scores = batch[batch_key]  # List of tensors
                pred_scores = predictions[pred_key]  # List of tensors
                
                # Compute loss for each example and average
                example_losses = []
                for i, (pred, target) in enumerate(zip(pred_scores, batch_scores)):
                    # Truncate to common length
                    min_len = min(pred.shape[0], target.shape[0])
                    pred = pred[:min_len]
                    target = target[:min_len]
                    example_losses.append(nn.MSELoss()(pred, target))
                
                if example_losses:
                    loss_val = torch.stack(example_losses).mean()
                    weight = weights.get(weight_key, 1.0)
                    losses[weight_key] = loss_val * weight
```

### ✅ Correct Aspects

1. **Matches extracted columns**: ✓ Uses `word_accuracy_scores`, `word_stress_scores`, `word_total_scores`
2. **Handles variable length**: ✓ Iterates per-example and truncates to common length
3. **Uses MSELoss**: ✓ Appropriate for continuous scores
4. **Loss weighting**: ✓ Applies configurable weights
5. **Phone-level too**: ✓ Lines 141-157 handle `phone_accuracy_scores` similarly

---

## 4. POTENTIAL ISSUES & IMPROVEMENTS

### Issue 1: Prediction Mismatch Risk ⚠️

**Problem:** The trainer expects predictions in the format:
```python
predictions = {
    "word_accuracy_logits": List[B] of variable-length tensors,
    "word_stress_logits": List[B] of variable-length tensors,
    ...
}
```

But we need to verify the **model** actually produces this format!

**Check Required:** Look at `whisper_pronunciation_model.py` to confirm:
- Does it produce `word_accuracy_logits`? 
- Does it produce `word_stress_logits`?
- Does it produce `word_total_logits`?
- Are they variable-length tensors?

### Issue 2: Normalization Range Mismatch ⚠️

**In data_processor.py (line 308-309):**
```python
# Normalize phone scores (phones are already on [0, 2] scale, so different range)
normalized = np.array(
    [self.normalize_assessment_score(s, min_val=0, max_val=2) for s in scores],
    dtype=np.float32
)
```

**But in trainer.py:**
```python
loss_val = nn.MSELoss()(pred, target)  # No special handling for [0,1] vs [0,2]
```

**Potential Issue:** If the model's output is not also normalized to [0,1], there will be a scale mismatch!

### Issue 3: Utterance Score Handling ✅ (Correct)

**Good:** Utterance-level scores are properly handled with Huber Loss:
```python
loss_val = huber_loss(
    predictions[pred_key],
    batch[batch_key]
)
```

This is correct because utterance scores are fixed-length [B] tensors.

---

## 5. VERIFICATION CHECKLIST

| Item | Status | Check |
|------|--------|-------|
| **Data extraction from "words"** | ✅ | Lines 213-265 in data_processor.py extract all word/phone scores |
| **Normalization** | ✅ | Lines 284-312 normalize to [0,1] range |
| **Data collation** | ✅ | data_collator.py batches as List[B] tensors |
| **Trainer expects correct format** | ✅ | Lines 114-157 match extracted column names |
| **Variable-length handling** | ✅ | Trainer iterates per-example and truncates |
| **Loss function choice** | ✅ | MSE for variable-length, Huber for fixed-length |
| **Model produces correct outputs** | ❓ | **NEEDS VERIFICATION** |
| **Output scale matches input scale** | ❓ | **NEEDS VERIFICATION** |

---

## 6. HOW TO VERIFY MODEL OUTPUTS

Check `whisper_pronunciation_model.py` forward() method:

**Should have:**
```python
def forward(self, input_features, ...):
    outputs = {
        # Word-level
        "word_accuracy_logits": [...],  # Shape: [B, num_words] per example
        "word_stress_logits": [...],
        "word_total_logits": [...],
        
        # Phone-level
        "phone_accuracy_logits": [...],  # Shape: [B, num_phones] per example
        
        # Utterance-level
        "utterance_accuracy_logits": [...],  # Shape: [B]
        "utterance_fluency_logits": [...],
        # ... other utterance scores
    }
    return outputs
```

---

## 7. SUMMARY: Words Column Usage

| Stage | Component | Action |
|-------|-----------|--------|
| **Load** | HuggingFace | Provides raw `words` dict with word/phone annotations |
| **Extract** | data_processor.py (lines 213-265) | ✅ Flattens `words` into score arrays |
| **Normalize** | data_processor.py (lines 284-312) | ✅ Normalizes to [0,1] range |
| **Batch** | data_collator.py | ✅ Converts to List[B] tensors |
| **Train** | trainer.py (lines 114-157) | ✅ Computes MSE loss per example |
| **Model** | whisper_pronunciation_model.py | ❓ Must produce matching outputs |

---

## 8. RECOMMENDATION

**The trainer APPEARS to use the extracted words data correctly, BUT:**

1. **Verify the model outputs** match the expected format:
   - Check if model produces: `word_accuracy_logits`, `word_stress_logits`, etc.
   - Check the shape and scale of these outputs

2. **Add validation logging** to catch mismatches:
   ```python
   # In trainer.py compute_loss()
   if "word_accuracy_scores" in batch:
       logger.info(f"Batch word_accuracy_scores: {[s.shape for s in batch['word_accuracy_scores']]}")
   if "word_accuracy_logits" in predictions:
       logger.info(f"Pred word_accuracy_logits: {[s.shape for s in predictions['word_accuracy_logits']]}")
   ```

3. **Test with actual data** to ensure no shape mismatches occur during training

---

## DETAILED FLOW DIAGRAM

```
┌─────────────────────────────────────────┐
│ HuggingFace Dataset                     │
│ "words": [                              │
│   {                                     │
│     "accuracy": 10,                     │
│     "stress": 10,                       │
│     "total": 10,                        │
│     "phones": [...],                    │
│     "phones-accuracy": [2, 2]           │
│   },                                    │
│   ...                                   │
│ ]                                       │
└──────────┬────────────────────────────┘
           │
           ▼ (data_processor.py lines 213-265)
┌─────────────────────────────────────────┐
│ EXTRACT from "words"                    │
│ ├─ word_accuracy_scores: [10, ...]      │
│ ├─ word_stress_scores: [10, ...]        │
│ ├─ word_total_scores: [10, ...]         │
│ └─ phone_accuracy_scores: [2, 2, ...]   │
└──────────┬────────────────────────────┘
           │
           ▼ (data_processor.py lines 284-312)
┌─────────────────────────────────────────┐
│ NORMALIZE to [0, 1]                     │
│ ├─ word_accuracy_scores: [1.0, ...]     │
│ ├─ word_stress_scores: [1.0, ...]       │
│ ├─ word_total_scores: [1.0, ...]        │
│ └─ phone_accuracy_scores: [1.0, 1.0, ...]
└──────────┬────────────────────────────┘
           │
           ▼ (data_collator.py)
┌─────────────────────────────────────────┐
│ BATCH as List[B] tensors                │
│ batch["word_accuracy_scores"]:          │
│   [tensor([1.0, 1.0, ...]),             │
│    tensor([1.0, 1.0, 1.0, ...]),        │
│    ...]                                 │
└──────────┬────────────────────────────┘
           │
           ▼ (trainer.py lines 114-157)
┌─────────────────────────────────────────┐
│ COMPUTE LOSS                            │
│ for each example:                       │
│   pred = model.word_accuracy_logits[i]  │
│   target = batch["word_accuracy_scores"]│
│   loss = MSE(pred, target)              │
│                                         │
│ total_loss = avg(all example losses)    │
│ weighted_loss = loss × weight           │
└─────────────────────────────────────────┘
```

