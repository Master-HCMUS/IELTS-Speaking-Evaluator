"""
ELSA-like Pronunciation Assessment Web App using Streamlit.

Main application entry point. Uses component-based architecture for maintainability.
"""

import sys
import streamlit as st
import tempfile
import traceback
from pathlib import Path
from typing import Optional

# Add parent directory to path
web_app_dir = str(Path(__file__).parent)
parent_dir = str(Path(__file__).parent.parent)

sys.path.insert(0, web_app_dir)
sys.path.insert(0, parent_dir)

# Import components
from components import (
    SidebarComponent,
    AssessmentFormComponent,
    ResultsDisplayComponent,
    HelpTabComponent,
)

try:
    from local_pronunciation_assessment_service import LocalPronunciationAssessmentService
except ImportError:
    st.error("❌ Failed to import LocalPronunciationAssessmentService")
    st.stop()


def initialize_service() -> Optional[LocalPronunciationAssessmentService]:
    """Initialize the pronunciation assessment service."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config_manager import ConfigManager
        
        config_manager = ConfigManager()
        local_config = config_manager.get_local_whisper_config()
        
        assessment_model_path = Path(local_config.get("assessment_model_path", ""))
        if not assessment_model_path.exists():
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
        
        service = LocalPronunciationAssessmentService(
            model_path=str(assessment_model_path),
            device=local_config.get("device", "auto")
        )
        
        return service
        
    except Exception as e:
        st.error(f"❌ Failed to initialize service: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="English Pronunciation Assessment",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
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
    
    # Header
    st.title("🎤 English Pronunciation Assessment")
    st.markdown("*Improve your English pronunciation with instant feedback*")
    
    # Render sidebar
    SidebarComponent.render(
        st.session_state.service_initialized,
        st.session_state.service
    )
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎙️ Assessment", "📊 History", "ℹ️ Help"])
    
    with tab1:
        # Render instructions
        AssessmentFormComponent.render_instructions()
        
        # Check service
        if not st.session_state.service_initialized or not st.session_state.service:
            st.warning("⏳ Service not initialized. Please wait...")
            st.stop()
        
        # Render sentence selection
        target_sentence = AssessmentFormComponent.render_sentence_selection()
        
        if not target_sentence:
            st.warning("Please enter or select a sentence first.")
            st.stop()
        
        # Render audio recording
        audio_data, assess_button = AssessmentFormComponent.render_audio_recording()
        
        # Process audio and assess
        if assess_button and audio_data is not None:
            try:
                with st.spinner("🔄 Processing audio and assessing pronunciation..."):
                    # Convert UploadedFile to bytes if needed
                    if hasattr(audio_data, 'read'):
                        audio_bytes = audio_data.read()
                    else:
                        audio_bytes = audio_data
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_path = tmp_file.name
                    
                    # Assess
                    result = st.session_state.service.assess_pronunciation(
                        file_path=tmp_path,
                        target_text=target_sentence
                    )
                    
                    st.session_state.assessment_result = result
                    
                    # Clean up
                    import os
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
            except Exception as e:
                st.error(f"❌ Error during assessment: {e}")
                st.error(traceback.format_exc())
        
        # Display results
        if st.session_state.assessment_result:
            ResultsDisplayComponent.render_results(
                st.session_state.assessment_result,
                target_sentence
            )
    
    with tab2:
        st.subheader("📚 Assessment History")
        st.info("History feature coming soon! Your assessments will be saved here.")
    
    with tab3:
        HelpTabComponent.render()


if __name__ == "__main__":
    main()
