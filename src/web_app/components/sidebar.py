"""Sidebar component for app settings and model information."""

import streamlit as st
from typing import Optional


class SidebarComponent:
    """Manages sidebar UI with settings and model information."""
    
    @staticmethod
    def render(service_initialized: bool, service) -> None:
        """
        Render sidebar with settings and model information.
        
        Args:
            service_initialized: Whether service is initialized
            service: The pronunciation assessment service instance
        """
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Initialize service on first run
            if not service_initialized:
                with st.spinner("🔄 Loading pronunciation assessment model..."):
                    import sys
                    from pathlib import Path
                    
                    # Add paths for imports
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                    
                    from local_pronunciation_assessment_service import LocalPronunciationAssessmentService
                    from config_manager import ConfigManager
                    
                    try:
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
                            st.session_state.service = None
                            st.session_state.service_initialized = False
                        else:
                            service = LocalPronunciationAssessmentService(
                                model_path=str(assessment_model_path),
                                device=local_config.get("device", "auto")
                            )
                            st.session_state.service = service
                            st.session_state.service_initialized = True
                            st.success("✅ Model loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize service: {e}")
                        st.session_state.service = None
                        st.session_state.service_initialized = False
            
            st.markdown("---")
            
            # Score legend
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
            if service:
                st.subheader("Model Information")
                try:
                    model_info = service.get_model_info()
                    st.json(model_info)
                except Exception as e:
                    st.warning(f"Could not load model info: {e}")
