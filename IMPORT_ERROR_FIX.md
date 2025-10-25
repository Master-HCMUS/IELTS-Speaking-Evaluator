# Import Error Fix - Data Processor

## Issue Found

**Error Message:**
```
ImportError: cannot import name 'DataCollator' from 'torch.utils.data'
```

**Root Cause:**
The import statement was trying to import `DataCollator` from `torch.utils.data`, but `DataCollator` is not a class in PyTorch's data utilities. This is a type protocol from the `transformers` library.

## Changes Made

### File: `data_processor.py`

#### Change 1: Removed Invalid Import
**Before:**
```python
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataCollator
```

**After:**
```python
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import load_dataset, DatasetDict, Dataset
```

#### Change 2: Fixed Return Type Annotation
**Before:**
```python
def create_data_collator(self, include_transcription: bool = True) -> DataCollator:
```

**After:**
```python
def create_data_collator(self, include_transcription: bool = True) -> 'PronunciationAssessmentDataCollator':
```

## Why This Works

1. **Removed unused import**: `DataCollator` wasn't actually needed since we're creating our own implementation
2. **Used string annotation**: `'PronunciationAssessmentDataCollator'` is a forward reference that works with Python's type system
3. **Maintains functionality**: The method still returns the correct collator instance

## Verification

✅ No import errors
✅ All type annotations valid
✅ Code can now be imported successfully

## Status

**Fixed**: The module can now be imported without errors.

To verify the fix works:
```python
from finetuning.finetuning_pronunciation_assessment.data_processor import SpeechOcean762PronunciationProcessor
print("✓ Import successful!")
```
