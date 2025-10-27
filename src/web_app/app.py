"""
ELSA-like Pronunciation Assessment Web App using Streamlit.

A user-friendly web application that helps users improve their English pronunciation
by providing real-time feedback with visual score highlighting.
"""

import streamlit as st
import numpy as np
import torch
import librosa
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import io
import sys
import os
from datetime import datetime
import json
from difflib import SequenceMatcher

# Add parent directory to path to import local modules
web_app_dir = str(Path(__file__).parent)
parent_dir = str(Path(__file__).parent.parent)

sys.path.insert(0, web_app_dir)
sys.path.insert(0, parent_dir)

try:
    from local_pronunciation_assessment_service import LocalPronunciationAssessmentService
except ImportError:
    st.error("❌ Failed to import LocalPronunciationAssessmentService")
    st.stop()

# Import or define utility functions
utils_imported = False
try:
    # Try direct import first
    try:
        from config import PRACTICE_SENTENCES
        from utils import (
            get_score_color,
            get_score_label,
            highlight_words,
            align_words_to_frames,
            get_recommendation,
            format_statistics,
        )
        utils_imported = True
    except (ImportError, ModuleNotFoundError):
        # Try with module prefix
        import importlib.util
        utils_path = Path(__file__).parent / "utils.py"
        if utils_path.exists():
            spec = importlib.util.spec_from_file_location("utils", utils_path)
            utils_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_mod)
            PRACTICE_SENTENCES = getattr(utils_mod, "PRACTICE_SENTENCES", {})
            get_score_color = getattr(utils_mod, "get_score_color", None)
            get_score_label = getattr(utils_mod, "get_score_label", None)
            highlight_words = getattr(utils_mod, "highlight_words", None)
            align_words_to_frames = getattr(utils_mod, "align_words_to_frames", None)
            get_recommendation = getattr(utils_mod, "get_recommendation", None)
            format_statistics = getattr(utils_mod, "format_statistics", None)
            utils_imported = True
except:
    pass

# Use fallback if utils not imported
if not utils_imported:
    # Fallback implementations if modules not found
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
    
    def get_score_color(score: float) -> str:
        """Get color for score visualization."""
        if score < 4:
            return "#FF4444"  # Red - Poor
        elif score < 7:
            return "#FFA500"  # Orange - Moderate
        else:
            return "#00CC44"  # Green - Good

    def get_score_label(score: float) -> str:
        """Get label for score."""
        if score < 4:
            return "Poor"
        elif score < 7:
            return "Fair"
        else:
            return "Good"

    def highlight_words(words: list, scores: list) -> str:
        """Create HTML to highlight words based on scores."""
        html = '<div style="font-size: 24px; line-height: 1.8;">'
        
        for word, score in zip(words, scores):
            color = get_score_color(score)
            label = get_score_label(score)
            html += f'<span style="background-color: {color}; color: white; padding: 6px 10px; margin: 4px; border-radius: 4px; font-weight: bold; display: inline-block; min-width: 80px; text-align: center;">{word}<br><small>{score:.1f}</small></span>'
        
        html += "</div>"
        return html

    def align_words_to_frames(words: list, frame_scores: list) -> Tuple[list, list]:
        """Align words with frame-level scores."""
        if not words or not frame_scores:
            return words, [5.0] * len(words)
        
        num_words = len(words)
        num_frames = len(frame_scores)
        word_scores = []
        frames_per_word = num_frames / num_words
        
        for word_idx in range(num_words):
            start_frame = int(word_idx * frames_per_word)
            end_frame = int((word_idx + 1) * frames_per_word)
            
            if start_frame < len(frame_scores):
                word_frame_scores = frame_scores[start_frame:end_frame]
                if word_frame_scores:
                    word_score = np.mean(word_frame_scores)
                else:
                    word_score = 5.0
            else:
                word_score = 5.0
            
            word_scores.append(word_score)
        
        return words, word_scores

    def get_recommendation(score: float) -> str:
        """Get recommendation based on overall score."""
        if score >= 8:
            return "🎉 Excellent pronunciation! Keep up the good work!"
        elif score >= 6:
            return "👍 Good pronunciation with room for improvement. Practice the highlighted words."
        elif score >= 4:
            return "⚠️ Your pronunciation needs improvement. Focus on words with low scores."
        else:
            return "❌ Pronunciation needs significant improvement. Consider practicing more slowly."

    def format_statistics(scores: list) -> Dict[str, float]:
        """Format statistics for score array."""
        arr = np.array(scores)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }


def initialize_service() -> Optional[LocalPronunciationAssessmentService]:
    """
    Initialize the pronunciation assessment service.
    
    Returns:
        Service instance or None if initialization fails
    """
    try:
        # Get model path from config
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config_manager import ConfigManager
        
        config_manager = ConfigManager()
        local_config = config_manager.get_local_whisper_config()
        
        # Try to find the assessment model
        assessment_model_path = Path(local_config.get("assessment_model_path", ""))
        if not assessment_model_path.exists():
            # Try default paths
            possible_paths = [
                Path("src/finetuning/finetuning_pronunciation_assessment/models/pronunciation_production"),
                Path("./models/pronunciation_assessment"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    assessment_model_path = path
                    break
        
        if not assessment_model_path.exists():
            st.error("❌ Fine-tuned pronunciation assessment model not found.")
            st.info("Please train the model first using the finetuning module.")
            return None
        
        # Initialize service
        service = LocalPronunciationAssessmentService(
            model_path=str(assessment_model_path),
            device=local_config.get("device", "auto")
        )
        
        return service
        
    except Exception as e:
        st.error(f"❌ Failed to initialize service: {e}")
        return None


def extract_assessment_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and structure assessment data for display.
    
    Args:
        result: Assessment result from service
        
    Returns:
        Structured data for visualization
    """
    scores = result.get("scores", {})
    
    return {
        "transcript": result.get("transcript", ""),
        "utterance_level": scores.get("utterance_level", {}),
        "word_level": scores.get("word_level", {}),
        "phone_level": scores.get("phone_level", {}),
    }


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using sequence matching.
    
    Args:
        text1: Target text
        text2: Transcribed text
        
    Returns:
        Similarity ratio (0-1)
    """
    # Normalize texts: lowercase and strip whitespace
    text1_norm = text1.lower().strip()
    text2_norm = text2.lower().strip()
    
    # Use SequenceMatcher to calculate similarity
    matcher = SequenceMatcher(None, text1_norm, text2_norm)
    return matcher.ratio()


def apply_content_penalty(scores: Dict[str, float], target_text: str, transcript: str) -> Dict[str, float]:
    """
    Apply penalty to scores if transcribed content doesn't match target.
    
    Args:
        scores: Original utterance-level scores
        target_text: Target sentence user should say
        transcript: What the user actually said
        
    Returns:
        Adjusted scores with content mismatch penalty
    """
    similarity = calculate_text_similarity(target_text, transcript)
    
    # If similarity is too low (< 50%), apply heavy penalty
    if similarity < 0.5:
        penalty_factor = 0.3 + (similarity * 0.4)  # Range: 0.3-0.7
        
        # Apply penalty to all scores
        penalized_scores = {}
        for key, score in scores.items():
            if key != "total":
                # Reduce score by penalty factor
                penalized_scores[key] = score * penalty_factor
            else:
                # Total score gets heavier penalty
                penalized_scores[key] = score * (penalty_factor - 0.1)
        
        return penalized_scores, similarity
    
    elif similarity < 0.75:
        # Moderate penalty for partial mismatch
        penalty_factor = 0.85
        penalized_scores = {}
        for key, score in scores.items():
            penalized_scores[key] = score * penalty_factor
        
        return penalized_scores, similarity
    
    else:
        # No penalty - texts match well
        return scores, similarity


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="English Pronunciation Assessment",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .score-good {
            color: #00CC44;
            font-weight: bold;
        }
        .score-fair {
            color: #FFA500;
            font-weight: bold;
        }
        .score-poor {
            color: #FF4444;
            font-weight: bold;
        }
        .instruction-box {
            background-color: #E3F2FD;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "service_initialized" not in st.session_state:
        st.session_state.service_initialized = False
        st.session_state.service = None
        st.session_state.assessment_result = None
        st.session_state.audio_bytes = None
    
    # Header
    st.title("🎤 English Pronunciation Assessment")
    st.markdown("*Improve your English pronunciation with instant feedback*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize service on first run
        if not st.session_state.service_initialized:
            with st.spinner("🔄 Loading pronunciation assessment model..."):
                service = initialize_service()
                if service:
                    st.session_state.service = service
                    st.session_state.service_initialized = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error("Failed to initialize service")
        
        st.markdown("---")
        
        # Score threshold explanation
        st.subheader("Score Legend")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(":green[■]")
            st.markdown(":orange[■]")
            st.markdown(":red[■]")
        
        with col2:
            st.markdown("**Good**: 7.0 - 10.0")
            st.markdown("**Fair**: 4.0 - 6.9")
            st.markdown("**Poor**: 0.0 - 3.9")
        
        st.markdown("---")
        
        # Model info
        if st.session_state.service:
            st.subheader("Model Information")
            model_info = st.session_state.service.get_model_info()
            st.json(model_info)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎙️ Assessment", "📊 History", "ℹ️ Help"])
    
    with tab1:
        # Instructions
        st.markdown("""
        <div class='instruction-box'>
            <strong>📝 How to use:</strong><br>
            1. Enter or select a sentence to practice<br>
            2. Click the microphone button to start recording<br>
            3. Speak the sentence clearly<br>
            4. Click stop to end recording<br>
            5. Get instant feedback with color-coded scores
        </div>
        """, unsafe_allow_html=True)
        
        # Check if service is ready
        if not st.session_state.service_initialized or not st.session_state.service:
            st.warning("⏳ Service not initialized. Please wait...")
            st.stop()
        
        # Sentence input section
        st.subheader("🎯 Select or Enter a Sentence")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Predefined sentences
            predefined_sentences = {
                "Weather": "The weather is nice today.",
                "Greeting": "Hello, how are you doing?",
                "Asking": "Where is the nearest train station?",
                "Ordering": "I would like a cup of coffee, please.",
                "Introduction": "My name is John and I am from London.",
                "Question": "Can you help me with this problem?",
                "Statement": "I have been studying English for three years.",
                "Request": "Could you speak more slowly, please?",
            }
            
            selected_category = st.selectbox(
                "Choose a predefined sentence or enter your own:",
                options=list(predefined_sentences.keys()) + ["Custom"],
                label_visibility="collapsed"
            )
            
            if selected_category == "Custom":
                target_sentence = st.text_input(
                    "Enter your sentence:",
                    placeholder="Type the sentence you want to practice..."
                )
            else:
                target_sentence = predefined_sentences[selected_category]
                st.info(f"📖 Target: **{target_sentence}**")
        
        if not target_sentence:
            st.warning("Please enter or select a sentence first.")
            st.stop()
        
        # Audio recording section
        st.subheader("🎤 Record Your Voice")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Step 1: Record**")
            audio_data = st.audio_input(
                "Click the microphone to start recording:",
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("**Step 2: Assess**")
            assess_button = st.button(
                "🚀 Assess Pronunciation",
                width="stretch",
                disabled=audio_data is None
            )
        
        if audio_data is not None:
            st.success("✅ Audio recorded - Ready to assess!")
        else:
            st.info("⏳ Waiting for audio...")
        
        # Process audio and assess
        if assess_button and audio_data is not None:
            try:
                with st.spinner("🔄 Processing audio and assessing pronunciation..."):
                    # Save audio temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_data.getbuffer())
                        temp_audio_path = Path(tmp_file.name)
                    
                    try:
                        # Run assessment with target text for content matching
                        result = st.session_state.service.assess_pronunciation(
                            str(temp_audio_path),
                            target_text=target_sentence
                        )
                        st.session_state.assessment_result = result
                        
                        st.success("✅ Assessment complete!")
                    finally:
                        # Clean up
                        if temp_audio_path.exists():
                            temp_audio_path.unlink()
                    
            except Exception as e:
                st.error(f"❌ Error during assessment: {e}")
                import traceback
                st.error(traceback.format_exc())
        
        # Display results
        if st.session_state.assessment_result:
            result = st.session_state.assessment_result
            
            if result.get("status") == "success":
                st.markdown("---")
                st.subheader("📊 Assessment Results")
                
                # Extract data
                data = extract_assessment_data(result)
                transcript = data["transcript"]
                utterance_scores = data["utterance_level"]
                word_scores = data["word_level"]
                
                # Check if penalty was applied by the service
                penalty_info = result.get("penalty_applied")
                content_match = result.get("content_match")
                
                # Display content match warning if needed
                if content_match and content_match.get('similarity', 1.0) < 0.75:
                    st.warning(
                        f"⚠️ **Content Mismatch Alert!**\n\n"
                        f"Target: **{content_match.get('target', target_sentence)}**\n\n"
                        f"You said: **{transcript}**\n\n"
                        f"Match Score: **{content_match.get('match_percentage', 0):.1f}%**\n\n"
                        f"Your pronunciation scores have been heavily penalized due to significant content mismatch. "
                        f"Please try again with the correct target sentence."
                    )
                    
                    if penalty_info:
                        st.info(f"Penalty Applied: {penalty_info.get('penalty_factor', 1.0):.2f}x reduction")
                
                # Display transcript section
                st.subheader("📝 What You Said")
                st.info(f"**{transcript}**")
                
                # Display utterance-level scores (overview)
                st.subheader("🎯 Overall Scores")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                score_keys = ["accuracy", "fluency", "prosodic", "completeness", "total"]
                score_colors = {
                    "accuracy": "blue",
                    "fluency": "green",
                    "prosodic": "purple",
                    "completeness": "orange",
                    "total": "red"
                }
                
                for col, key in zip([col1, col2, col3, col4, col5], score_keys):
                    with col:
                        score_val = utterance_scores.get(key, 0)
                        color_class = "score-good" if score_val >= 7 else ("score-fair" if score_val >= 4 else "score-poor")
                        st.metric(
                            key.capitalize(),
                            f"{score_val:.1f}/10"
                        )
                
                # Word-by-word analysis
                st.subheader("📌 Word-by-Word Analysis")
                
                # Align words with frame scores
                words = transcript.split()
                word_accuracy_frames = word_scores.get("accuracy", [])
                aligned_words, word_scores_list = align_words_to_frames(words, word_accuracy_frames)
                
                # Display highlighted words
                if aligned_words and word_scores_list:
                    st.markdown(highlight_words(aligned_words, word_scores_list), unsafe_allow_html=True)
                    
                    # Detailed word analysis table
                    st.subheader("📋 Detailed Word Analysis")
                    
                    table_data = []
                    for word, score in zip(aligned_words, word_scores_list):
                        label = get_score_label(score)
                        table_data.append({
                            "Word": word,
                            "Score": f"{score:.2f}",
                            "Assessment": label
                        })
                    
                    st.dataframe(
                        table_data,
                        hide_index=True,
                        width='stretch'
                    )
                
                # Frame-level statistics
                st.subheader("📈 Detailed Statistics")
                
                if word_accuracy_frames:
                    acc_array = np.array(word_accuracy_frames)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Minimum", f"{acc_array.min():.2f}")
                    with col2:
                        st.metric("Maximum", f"{acc_array.max():.2f}")
                    with col3:
                        st.metric("Average", f"{acc_array.mean():.2f}")
                    with col4:
                        st.metric("Std Dev", f"{acc_array.std():.2f}")
                    
                    # Show histogram
                    st.bar_chart(
                        data=acc_array,
                        width='stretch',
                        height=400
                    )
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                overall_score = utterance_scores.get("total", 0)
                
                if overall_score >= 8:
                    st.success("🎉 Excellent pronunciation! Keep up the good work!")
                elif overall_score >= 6:
                    st.info("👍 Good pronunciation with room for improvement. Practice the red-highlighted words.")
                elif overall_score >= 4:
                    st.warning("⚠️ Your pronunciation needs improvement. Focus on words with low scores.")
                else:
                    st.error("❌ Pronunciation needs significant improvement. Consider practicing more slowly.")
                
                # Save result option
                if st.button("💾 Save Result"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    result_file = Path(f"assessment_result_{timestamp}.json")
                    
                    import json
                    with open(result_file, "w") as f:
                        json.dump(result, f, indent=2)
                    
                    st.success(f"✅ Result saved to {result_file}")
            
            else:
                error = result.get("error", "Unknown error")
                st.error(f"❌ Assessment failed: {error}")
    
    with tab2:
        st.subheader("📚 Assessment History")
        st.info("History feature coming soon! Your assessments will be saved here.")
    
    with tab3:
        st.subheader("ℹ️ How to Use This App")
        
        st.markdown("""
        ### 📖 Getting Started
        
        1. **Select a Sentence**: Choose from predefined sentences or enter your own
        2. **Record Audio**: Click the microphone button and speak clearly
        3. **Assess**: Click "Assess Pronunciation" to analyze your speech
        4. **Review Feedback**: See your scores and word-by-word analysis
        
        ### 🎨 Understanding the Colors
        
        - 🟢 **Green (7.0-10.0)**: Good pronunciation
        - 🟠 **Orange (4.0-6.9)**: Fair pronunciation
        - 🔴 **Red (0.0-3.9)**: Poor pronunciation
        
        ### 📊 Score Meanings
        
        - **Accuracy**: How well you pronounced each word
        - **Fluency**: How smoothly you spoke
        - **Prosodic**: Your intonation and stress patterns
        - **Completeness**: Whether you said all words clearly
        - **Total**: Overall pronunciation quality
        
        ### 💡 Tips for Better Pronunciation
        
        1. Speak slowly and clearly
        2. Pay attention to word stress
        3. Practice difficult words separately
        4. Listen to native speakers
        5. Record yourself and compare
        
        ### 🔧 Requirements
        
        - Microphone access required
        - Internet connection (for model)
        - English speakers recommended
        """)


if __name__ == "__main__":
    main()
