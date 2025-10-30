"""Landing page component for the pronunciation assessment web app."""

import streamlit as st
from typing import Callable


class LandingPageComponent:
    """Component for rendering the landing page."""
    
    @staticmethod
    def render(on_start_test: Callable = None) -> bool:
        """
        Render the landing page.
        
        Args:
            on_start_test: Callback function when "Take free test" is clicked
            
        Returns:
            bool: True if user clicked "Take free test"
        """
        # Hero section
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 3rem; color: #2E86AB; margin-bottom: 1rem;">
                🎤 English Pronunciation Assessment
            </h1>
            <h2 style="font-size: 1.5rem; color: #666; font-weight: 300; margin-bottom: 2rem;">
                Improve your English pronunciation with AI-powered instant feedback
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div style="text-align: center; padding: 1.5rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                    <h3 style="color: #2E86AB;">Accurate Assessment</h3>
                    <p style="color: #666;">
                        Advanced AI model analyzes your pronunciation  across multiple dimensions.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div style="text-align: center; padding: 1.5rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                    <h3 style="color: #2E86AB;">Instant Feedback</h3>
                    <p style="color: #666;">
                        Get immediate detailed feedback on accuracy, fluency, 
                        completeness, and prosodic features.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div style="text-align: center; padding: 1.5rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
                    <h3 style="color: #2E86AB;">Track Progress</h3>
                    <p style="color: #666;">
                        Monitor your improvement over time with detailed 
                        scoring and personalized recommendations.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # How it works section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="color: #2E86AB; margin-bottom: 2rem;">How It Works</h2>
        </div>
        """, unsafe_allow_html=True)
        
        step_col1, step_col2, step_col3, step_col4 = st.columns(4)
        
        with step_col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2.5rem; color: #2E86AB; margin-bottom: 1rem;">1️⃣</div>
                <h4 style="color: #2E86AB;">Choose Text</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Select from sample sentences or enter your own text to practice.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with step_col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2.5rem; color: #2E86AB; margin-bottom: 1rem;">2️⃣</div>
                <h4 style="color: #2E86AB;">Record Audio</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Read the text aloud and record your pronunciation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with step_col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2.5rem; color: #2E86AB; margin-bottom: 1rem;">3️⃣</div>
                <h4 style="color: #2E86AB;">AI Analysis</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Our AI model analyzes your speech across multiple criteria.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with step_col4:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2.5rem; color: #2E86AB; margin-bottom: 1rem;">4️⃣</div>
                <h4 style="color: #2E86AB;">Get Feedback</h4>
                <p style="color: #666; font-size: 0.9rem;">
                    Receive detailed scores and improvement suggestions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Assessment criteria section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h2 style="color: #2E86AB; margin-bottom: 2rem;">Assessment Criteria</h2>
        </div>
        """, unsafe_allow_html=True)
        
        criteria_col1, criteria_col2 = st.columns(2)
        
        with criteria_col1:
            st.markdown("""
            <div class="criteria-card">
                <h4 style="color: #2E86AB; margin-bottom: 1rem;">🎯 Accuracy</h4>
                <p style="color: #666; margin-bottom: 1rem;">
                    Measures how correctly you pronounce individual sounds, words, and phrases.
                </p>
                <h4 style="color: #2E86AB; margin-bottom: 1rem;">🌊 Fluency</h4>
                <p style="color: #666;">
                    Evaluates the smoothness and naturalness of your speech flow.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with criteria_col2:
            st.markdown("""
            <div class="criteria-card">
                <h4 style="color: #2E86AB; margin-bottom: 1rem;">✅ Completeness</h4>
                <p style="color: #666; margin-bottom: 1rem;">
                    Checks if you've pronounced all words and maintained proper structure.
                </p>
                <h4 style="color: #2E86AB; margin-bottom: 1rem;">🎵 Prosodic</h4>
                <p style="color: #666;">
                    Analyzes rhythm, stress, intonation, and other speech patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # CTA section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h2 style="color: #2E86AB; margin-bottom: 1rem;">Ready to Improve Your Pronunciation?</h2>
            <p style="color: #666; font-size: 1.2rem; margin-bottom: 2rem;">
                Start your free assessment now and get instant feedback!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            take_test_clicked = st.button(
                "🎤 Take Free Test",
                type="primary",
                use_container_width=True,
                help="Start your pronunciation assessment"
            )
            
            if take_test_clicked and on_start_test:
                on_start_test()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #999; font-size: 0.9rem;">
            <p>Powered by advanced AI and machine learning • Free to use • No registration required</p>
        </div>
        """, unsafe_allow_html=True)
        
        return take_test_clicked
    
    @staticmethod
    def render_custom_css():
        """Render custom CSS for the landing page."""
        st.markdown("""
        <style>
            .feature-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                margin: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: transform 0.2s;
            }
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            .criteria-card {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid #2E86AB;
                margin: 10px 0;
            }
            .stButton > button {
                font-size: 1.2rem !important;
                padding: 0.75rem 2rem !important;
                border-radius: 25px !important;
                background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                transition: all 0.3s !important;
            }
            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 12px rgba(46, 134, 171, 0.3) !important;
            }
        </style>
        """, unsafe_allow_html=True)