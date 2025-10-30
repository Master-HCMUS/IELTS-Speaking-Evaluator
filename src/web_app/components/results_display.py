"""Results display component for assessment visualization."""

import streamlit as st
import numpy as np
from typing import Dict, Any, Tuple
from difflib import SequenceMatcher
from .phoneme_display import PhonemeDisplayComponent


class ResultsDisplayComponent:
    """Manages display of assessment results with visualizations."""
    
    @staticmethod
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
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: Target text
            text2: Transcribed text
            
        Returns:
            Similarity ratio (0-1)
        """
        text1_norm = text1.lower().strip()
        text2_norm = text2.lower().strip()
        matcher = SequenceMatcher(None, text1_norm, text2_norm)
        return matcher.ratio()
    
    @staticmethod
    def apply_content_penalty(
        scores: Dict[str, float],
        target_text: str,
        transcript: str
    ) -> Tuple[Dict[str, float], float]:
        """
        Apply penalty to scores if transcribed content doesn't match target.
        
        Args:
            scores: Original utterance-level scores
            target_text: Target sentence
            transcript: Transcribed text
            
        Returns:
            Tuple of (adjusted_scores, similarity)
        """
        similarity = ResultsDisplayComponent.calculate_text_similarity(target_text, transcript)
        
        if similarity < 0.5:
            penalty_factor = 0.3 + (similarity * 0.4)
            penalized_scores = {}
            for key, score in scores.items():
                if key != "total":
                    penalized_scores[key] = score * penalty_factor
                else:
                    penalized_scores[key] = score * (penalty_factor - 0.1)
            return penalized_scores, similarity
        
        elif similarity < 0.75:
            penalty_factor = 0.85
            penalized_scores = {}
            for key, score in scores.items():
                penalized_scores[key] = score * penalty_factor
            return penalized_scores, similarity
        
        else:
            return scores, similarity
    
    @staticmethod
    def get_score_color(score: float) -> str:
        """Get color for score visualization."""
        if score < 4:
            return "#FF4444"  # Red
        elif score < 7:
            return "#FFA500"  # Orange
        else:
            return "#00CC44"  # Green
    
    @staticmethod
    def get_score_label(score: float) -> str:
        """Get label for score."""
        if score < 4:
            return "Poor"
        elif score < 7:
            return "Fair"
        else:
            return "Good"
    
    @staticmethod
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
    
    @staticmethod
    def render_overall_scores(utterance_scores: Dict[str, float]) -> None:
        """Render overall scores display."""
        st.subheader("🎯 Overall Scores")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        score_keys = ["accuracy", "fluency", "prosodic", "completeness", "total"]
        score_labels = {
            "accuracy": "Accuracy",
            "fluency": "Fluency",
            "prosodic": "Prosodic",
            "completeness": "Completeness",
            "total": "Overall"
        }
        score_colors = {
            "accuracy": "blue",
            "fluency": "green",
            "prosodic": "purple",
            "completeness": "orange",
            "total": "red"
        }
        
        for col, key in zip([col1, col2, col3, col4, col5], score_keys):
            score = utterance_scores.get(key, 0.0)
            with col:
                st.metric(
                    score_labels[key],
                    f"{score:.1f}/10",
                    delta=None
                )
    
    @staticmethod
    def render_word_analysis(transcript: str, word_scores_dict: Dict[str, list]) -> None:
        """Render word-by-word analysis."""
        st.subheader("📌 Word-by-Word Analysis")
        
        words = transcript.split()
        word_accuracy_frames = word_scores_dict.get("accuracy", [])
        aligned_words, word_scores_list = ResultsDisplayComponent.align_words_to_frames(
            words,
            word_accuracy_frames
        )
        
        if aligned_words and word_scores_list:
            html = '<div style="font-size: 24px; line-height: 1.8;">'
            
            for word, score in zip(aligned_words, word_scores_list):
                color = ResultsDisplayComponent.get_score_color(score)
                label = ResultsDisplayComponent.get_score_label(score)
                html += f'<span style="background-color: {color}; color: white; padding: 6px 10px; margin: 4px; border-radius: 4px; font-weight: bold; display: inline-block; min-width: 80px; text-align: center;">{word}<br><small>{score:.1f}</small></span>'
            
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
    
    @staticmethod
    def render_statistics(word_accuracy_frames: list) -> None:
        """Render detailed statistics."""
        st.subheader("📈 Detailed Statistics")
        
        if word_accuracy_frames:
            frame_array = np.array(word_accuracy_frames)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Min Score", f"{frame_array.min():.2f}")
            
            with col2:
                st.metric("Max Score", f"{frame_array.max():.2f}")
            
            with col3:
                st.metric("Mean Score", f"{frame_array.mean():.2f}")
            
            with col4:
                st.metric("Std Dev", f"{frame_array.std():.2f}")
            
            st.line_chart(frame_array)
    
    @staticmethod
    def render_recommendations(overall_score: float) -> None:
        """Render recommendations based on score."""
        st.subheader("💡 Recommendations")
        
        if overall_score >= 8:
            st.success("🎉 Excellent pronunciation! Keep up the good work!")
        elif overall_score >= 6:
            st.info("👍 Good pronunciation with room for improvement. Practice the highlighted words.")
        elif overall_score >= 4:
            st.warning("⚠️ Your pronunciation needs improvement. Focus on words with low scores.")
        else:
            st.error("❌ Pronunciation needs significant improvement. Consider practicing more slowly.")
        
        # Show learning plan button if score needs improvement
        if overall_score < 7:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "📚 Get Personalized Learning Plan",
                    type="primary",
                    use_container_width=True,
                    help="Get a customized plan to improve your pronunciation"
                ):
                    # Store the current assessment result for the learning plan
                    st.session_state.learning_plan_data = st.session_state.assessment_result
                    st.session_state.current_page = "learning_plan"
                    st.rerun()
    
    @staticmethod
    def render_results(result: Dict[str, Any], target_text: str) -> None:
        """
        Render complete assessment results.
        
        Args:
            result: Assessment result from service
            target_text: Target sentence
        """
        if result.get("status") == "success":
            st.markdown("---")
            st.subheader("📊 Assessment Results")
            
            # Extract data
            data = ResultsDisplayComponent.extract_assessment_data(result)
            transcript = data["transcript"]
            utterance_scores = data["utterance_level"]
            word_scores = data["word_level"]
            phone_scores = data["phone_level"]
            
            # Check penalty
            content_match = result.get("content_match")
            
            if content_match and content_match.get('similarity', 1.0) < 0.75:
                similarity = content_match['similarity']
                st.warning(
                    f"⚠️ Content mismatch detected! "
                    f"Similarity: {similarity:.1%}. "
                    f"You may have said different words than the target sentence. "
                    f"Scores may be adjusted accordingly."
                )
            
            # Display transcript
            st.subheader("📝 What You Said")
            st.info(f"**{transcript}**")
            
            # Display overall scores
            ResultsDisplayComponent.render_overall_scores(utterance_scores)
            
            # Word-by-word analysis
            word_accuracy_frames = word_scores.get("accuracy", [])
            ResultsDisplayComponent.render_word_analysis(transcript, word_scores)
            
            # Phoneme analysis (if available)
            phone_accuracy_frames = phone_scores.get("accuracy", [])
            PhonemeDisplayComponent.render_phonemes_section(result, transcript, phone_accuracy_frames)
            
            # Statistics
            if word_accuracy_frames:
                ResultsDisplayComponent.render_statistics(word_accuracy_frames)
            
            # Recommendations
            overall_score = utterance_scores.get("total", 0)
            ResultsDisplayComponent.render_recommendations(overall_score)
            
            # Save result button
            if st.button("💾 Save Result"):
                st.success("✅ Result saved!")
        
        else:
            error = result.get("error", "Unknown error")
            st.error(f"❌ Assessment failed: {error}")

