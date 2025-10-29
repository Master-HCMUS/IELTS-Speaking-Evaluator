"""Assessment form component for sentence selection and audio recording."""

import streamlit as st
from typing import Tuple


PREDEFINED_SENTENCES = {
    "Weather": "The weather is nice today.",
    "Greeting": "Hello, how are you doing?",
    "Asking": "Where is the nearest train station?",
    "Ordering": "I would like a cup of coffee, please.",
    "Introduction": "My name is John and I am from London.",
    "Question": "Can you help me with this problem?",
    "Statement": "I have been studying English for three years.",
    "Request": "Could you speak more slowly, please?",
}


class AssessmentFormComponent:
    """Manages assessment form UI for sentence selection and audio recording."""
    
    @staticmethod
    def render_instructions() -> None:
        """Render instruction box."""
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
    
    @staticmethod
    def render_sentence_selection() -> str:
        """
        Render sentence selection UI.
        
        Returns:
            Target sentence to practice
        """
        st.subheader("🎯 Select or Enter a Sentence")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_category = st.selectbox(
                "Choose a predefined sentence or enter your own:",
                options=list(PREDEFINED_SENTENCES.keys()) + ["Custom"],
                label_visibility="collapsed"
            )
            
            if selected_category == "Custom":
                target_sentence = st.text_input(
                    "Enter your sentence:",
                    placeholder="Type the sentence you want to practice..."
                )
            else:
                target_sentence = PREDEFINED_SENTENCES[selected_category]
                st.info(f"📖 Target: **{target_sentence}**")
        
        return target_sentence
    
    @staticmethod
    def render_audio_recording() -> Tuple[bytes, bool]:
        """
        Render audio recording UI.
        
        Returns:
            Tuple of (audio_data, assess_button_clicked)
        """
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
        
        return audio_data, assess_button
