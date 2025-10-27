# English Pronunciation Assessment Web App

A user-friendly Streamlit web application for improving English pronunciation with real-time feedback and visual score highlighting.

## Features

✨ **Key Features:**
- 🎤 **Real-time Audio Recording**: Record pronunciation directly from your browser
- 📝 **Automatic Transcription**: Gets what you said using fine-tuned Whisper model
- 🎯 **Pronunciation Assessment**: Detailed scoring of your pronunciation
- 🎨 **Visual Feedback**: Color-coded word highlighting for instant feedback
- 📊 **Detailed Statistics**: Frame-level analysis and performance metrics
- 💡 **Smart Recommendations**: Personalized tips based on your performance

## Score Visualization

Each word is highlighted with colors indicating pronunciation quality:

- 🟢 **Green (7.0-10.0)**: Good pronunciation - Keep it up!
- 🟠 **Orange (4.0-6.9)**: Fair pronunciation - Room for improvement
- 🔴 **Red (0.0-3.9)**: Poor pronunciation - Needs practice

## Score Metrics

- **Accuracy**: How correctly you pronounced each word
- **Fluency**: How smoothly you spoke without hesitation
- **Prosodic**: Your intonation, stress, and rhythm patterns
- **Completeness**: Whether you spoke all words clearly
- **Total**: Overall pronunciation quality score

## Installation

### Prerequisites
- Python 3.8+
- Microphone for audio recording
- Fine-tuned Whisper model (trained via finetuning module)

### Setup

1. Install Streamlit and dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the pronunciation assessment model is trained:
```bash
python -m src.finetuning.finetuning_pronunciation_assessment.run_pronunciation_training \
  --mode production --output-dir ./models/pronunciation_production
```

3. Configure the model path in your config if needed.

## Usage

### Quick Start

**Option 1: Using the batch file (Windows)**
```bash
run.bat
```

**Option 2: Using PowerShell**
```powershell
.\run.ps1
```

**Option 3: Direct Streamlit command**
```bash
streamlit run app.py
```

### In the App

1. **Select or Enter a Sentence**
   - Choose from predefined practice sentences
   - Or enter your own custom sentence

2. **Record Your Voice**
   - Click the microphone button
   - Speak the sentence clearly
   - Click again to stop recording

3. **Get Assessment**
   - Click "Assess Pronunciation"
   - View your scores and detailed feedback
   - See word-by-word analysis with colors

4. **Review Results**
   - Check overall scores
   - See word-by-word pronunciation quality
   - Read recommendations for improvement
   - View statistical analysis

## File Structure

```
src/web_app/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── run.bat              # Windows batch runner
├── run.ps1              # PowerShell runner
└── __init__.py          # Package initialization
```

## Architecture

### Components

1. **Streamlit Frontend**: User interface and visualization
2. **Audio Processing**: Recording and mel-spectrogram conversion
3. **Pronunciation Assessment Service**: Local model inference
4. **Visualization Engine**: Color-coded word highlighting

### Data Flow

```
User Input (Sentence)
    ↓
Audio Recording (Microphone)
    ↓
Mel-Spectrogram Conversion
    ↓
Whisper Model (Transcription + Assessment)
    ↓
Score Alignment (Words ← Frames)
    ↓
Visual Rendering (HTML with colors)
    ↓
Display Results & Recommendations
```

## Configuration

Edit `config.py` to customize:

- **Model paths**: Location of fine-tuned model
- **Audio settings**: Sample rate, chunk size, max duration
- **Score thresholds**: Good/Fair/Poor score ranges
- **Colors**: RGB values for highlighting
- **Practice sentences**: Add your own exercise sentences

## Troubleshooting

### Model Not Found
```
❌ Fine-tuned pronunciation assessment model not found.
```
**Solution**: Train the model or ensure model path is correct in config

### Microphone Not Working
**Solution**: 
- Check browser microphone permissions
- Allow microphone access in browser settings
- Try a different browser

### Slow Processing
**Solution**:
- Use GPU if available (check device setting)
- Ensure model is properly loaded
- Close other applications to free resources

### Audio Quality Issues
**Solution**:
- Record in quiet environment
- Speak clearly and at normal pace
- Check microphone is working properly
- Try different audio input device

## Performance

### Typical Assessment Time
- Audio Recording: Variable (user-controlled)
- Processing: 1-3 seconds per utterance
- Total: < 5 seconds from recording to results

### System Requirements
- **GPU**: Recommended for faster processing (2GB+ VRAM)
- **CPU**: ~4 cores minimum
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 5GB for model files

## Tips for Better Results

1. **Speak Clearly**: Articulate each word distinctly
2. **Use Natural Pace**: Don't speak too fast or too slow
3. **Proper Intonation**: Use natural English stress patterns
4. **Complete Utterances**: Don't pause mid-sentence
5. **Good Audio**: Record in quiet environment with good microphone
6. **Practice**: Use predefined sentences to practice

## Advanced Usage

### Custom Sentences
1. Select "Custom" category
2. Enter your target sentence
3. Record and assess
4. View results

### Export Results
- Click "Save Result" to export assessment to JSON
- Results include all scores and transcription
- Useful for tracking progress over time

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Review app logs in terminal
3. Verify model is properly trained
4. Ensure audio input is working

## License

Part of IELTS Speaking Evaluation project.
