"""Phoneme display component for pronunciation assessment results."""

import streamlit as st
import numpy as np
from typing import Dict, Any, List


class PhonemeDisplayComponent:
    """Manages display of phoneme predictions and analysis."""
    
    @staticmethod
    def get_confidence_color(confidence: float) -> str:
        """Get color for confidence score visualization."""
        if confidence < 0.5:
            return "#FF4444"  # Red - low confidence
        elif confidence < 0.7:
            return "#FFA500"  # Orange - medium confidence
        else:
            return "#00CC44"  # Green - high confidence
    
    @staticmethod
    def get_confidence_label(confidence: float) -> str:
        """Get label for confidence score."""
        if confidence < 0.5:
            return "Low"
        elif confidence < 0.7:
            return "Medium"
        else:
            return "High"
    
    @staticmethod
    def render_phoneme_sequence(phonemes: List[str], frames_detail: List[Dict[str, Any]]) -> None:
        """
        Render phoneme sequence with confidence indicators.
        
        Args:
            phonemes: List of phoneme symbols (e.g., ['W', 'IY0', 'D'])
            frames_detail: Frame-level details with confidence scores
        """
        if not phonemes:
            st.info("No phoneme predictions available.")
            return
        
        st.subheader("🔤 Phoneme Sequence")
        
        # Create phoneme display with confidence indicators
        html = '<div style="font-size: 20px; line-height: 2.0; font-family: monospace;">'
        
        for frame_detail in frames_detail:
            phoneme = frame_detail.get('phoneme', '?')
            confidence = frame_detail.get('confidence', 0.0)
            color = PhonemeDisplayComponent.get_confidence_color(confidence)
            
            html += (
                f'<span style="'
                f'background-color: {color}; '
                f'color: white; '
                f'padding: 8px 12px; '
                f'margin: 4px; '
                f'border-radius: 6px; '
                f'font-weight: bold; '
                f'display: inline-block; '
                f'min-width: 60px; '
                f'text-align: center;">'
                f'{phoneme}<br>'
                f'<small>{confidence:.2f}</small></span>'
            )
        
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def render_phoneme_statistics(frames_detail: List[Dict[str, Any]]) -> None:
        """
        Render phoneme-level statistics.
        
        Args:
            frames_detail: Frame-level details with confidence scores
        """
        if not frames_detail:
            return
        
        st.subheader("📊 Phoneme Statistics")
        
        confidences = np.array([f.get('confidence', 0.0) for f in frames_detail])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Phonemes", len(frames_detail))
        
        with col2:
            st.metric("Avg Confidence", f"{np.mean(confidences):.2f}")
        
        with col3:
            st.metric("Min Confidence", f"{np.min(confidences):.2f}")
        
        with col4:
            st.metric("Max Confidence", f"{np.max(confidences):.2f}")
        
        # Confidence distribution chart
        st.line_chart(confidences)
    
    @staticmethod
    def render_phoneme_table(frames_detail: List[Dict[str, Any]]) -> None:
        """
        Render detailed phoneme table with frame indices and confidences.
        
        Args:
            frames_detail: Frame-level details with confidence scores
        """
        if not frames_detail:
            return
        
        st.subheader("📋 Phoneme Details")
        
        # Create table data
        table_data = []
        for detail in frames_detail:
            frame_idx = detail.get('frame', 0)
            phoneme = detail.get('phoneme', '?')
            confidence = detail.get('confidence', 0.0)
            confidence_label = PhonemeDisplayComponent.get_confidence_label(confidence)
            
            table_data.append({
                "Frame": frame_idx,
                "Phoneme": phoneme,
                "Confidence": f"{confidence:.4f}",
                "Quality": confidence_label
            })
        
        # Display as dataframe
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def render_phoneme_accuracy_alignment(
        transcript: str,
        phone_accuracy_frames: List[float],
        frames_detail: List[Dict[str, Any]]
    ) -> None:
        """
        Render alignment between phoneme predictions and phone accuracy scores.
        
        Args:
            transcript: Transcribed text
            phone_accuracy_frames: Frame-level phone accuracy scores
            frames_detail: Frame-level phoneme details
        """
        if not frames_detail or not phone_accuracy_frames:
            return
        
        st.subheader("🎯 Phoneme-Accuracy Alignment")
        
        # Create side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Predicted Phonemes**")
            phoneme_text = " → ".join([f['phoneme'] for f in frames_detail])
            st.code(phoneme_text, language="")
        
        with col2:
            st.write("**Phone Accuracy Scores**")
            accuracy_avg = np.mean(phone_accuracy_frames)
            st.metric("Average Phone Accuracy", f"{accuracy_avg:.1f}/10")
        
        # Visualization of alignment
        st.write("**Frame-by-Frame Alignment**")
        
        max_frames = max(len(frames_detail), len(phone_accuracy_frames))
        
        # Pad arrays to same length
        phonemes = [f.get('phoneme', '-') for f in frames_detail]
        phonemes.extend(['-'] * (max_frames - len(phonemes)))
        
        accuracies = phone_accuracy_frames.copy() if isinstance(phone_accuracy_frames, list) else phone_accuracy_frames.tolist()
        accuracies.extend([0] * (max_frames - len(accuracies)))
        
        alignment_data = {
            'Frame': list(range(max_frames)),
            'Phoneme': phonemes[:max_frames],
            'Accuracy': accuracies[:max_frames]
        }
        
        import pandas as pd
        df = pd.DataFrame(alignment_data)
        st.dataframe(df, use_container_width=True)
    
    @staticmethod
    def render_phonemes_section(result: Dict[str, Any], transcript: str, phone_accuracy_frames: List[float]) -> None:
        """
        Render complete phoneme section with all visualizations.
        
        Args:
            result: Assessment result from service
            transcript: Transcribed text
            phone_accuracy_frames: Frame-level phone accuracy scores
        """
        phonemes_data = result.get('phonemes', {})
        
        # Check if phonemes are available
        if not phonemes_data.get('phonemes'):
            st.info("📝 Phoneme predictions not available. The model may not have been trained with phoneme decoder enabled.")
            return
        
        st.markdown("---")
        st.subheader("🔤 Phoneme Analysis")
        
        phonemes = phonemes_data.get('phonemes', [])
        frames_detail = phonemes_data.get('frames_detail', [])
        num_frames = phonemes_data.get('num_frames', 0)
        avg_confidence = phonemes_data.get('avg_confidence', 0.0)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Phoneme Sequence", " ".join(phonemes) if phonemes else "N/A")
        with col2:
            st.metric("Total Frames", num_frames)
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        
        # Main visualizations
        if frames_detail:
            tab1, tab2, tab3 = st.tabs(["Sequence", "Statistics", "Details"])
            
            with tab1:
                PhonemeDisplayComponent.render_phoneme_sequence(phonemes, frames_detail)
            
            with tab2:
                PhonemeDisplayComponent.render_phoneme_statistics(frames_detail)
                if phone_accuracy_frames:
                    PhonemeDisplayComponent.render_phoneme_accuracy_alignment(
                        transcript,
                        phone_accuracy_frames,
                        frames_detail
                    )
            
            with tab3:
                PhonemeDisplayComponent.render_phoneme_table(frames_detail)
