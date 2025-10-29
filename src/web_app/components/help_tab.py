"""Help tab component with usage instructions and tips."""

import streamlit as st


class HelpTabComponent:
    """Manages help tab with instructions and tips."""
    
    @staticmethod
    def render() -> None:
        """Render help tab content."""
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
