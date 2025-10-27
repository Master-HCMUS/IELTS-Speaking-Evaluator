"""
Configuration for the Streamlit web application.
"""

import os
from pathlib import Path

# App configuration
APP_TITLE = "English Pronunciation Assessment"
APP_ICON = "🎤"

# Model paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_SEARCH_PATHS = [
    PROJECT_ROOT / "src" / "finetuning" / "finetuning_pronunciation_assessment" / "models" / "pronunciation_production",
    PROJECT_ROOT / "models" / "pronunciation_assessment",
    Path.home() / ".pronunciation_models" / "production",
]

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MAX_RECORDING_SECONDS = 60

# Score thresholds
SCORE_GOOD_MIN = 7.0
SCORE_FAIR_MIN = 4.0
SCORE_POOR_MAX = 3.9

# Colors
COLOR_GOOD = "#00CC44"  # Green
COLOR_FAIR = "#FFA500"  # Orange
COLOR_POOR = "#FF4444"  # Red

# Predefined sentences for practice
PRACTICE_SENTENCES = {
    "Weather": "The weather is nice today.",
    "Greeting": "Hello, how are you doing?",
    "Asking": "Where is the nearest train station?",
    "Ordering": "I would like a cup of coffee, please.",
    "Introduction": "My name is John and I am from London.",
    "Question": "Can you help me with this problem?",
    "Statement": "I have been studying English for three years.",
    "Request": "Could you speak more slowly, please?",
}
