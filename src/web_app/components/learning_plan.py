"""Learning plan component for personalized pronunciation improvement."""

import streamlit as st
from typing import Dict, Any, List, Tuple
import random


class LearningPlanComponent:
    """Component for rendering personalized learning plans based on assessment results."""
    
    @staticmethod
    def analyze_weaknesses(scores: Dict[str, float]) -> List[str]:
        """
        Analyze assessment scores to identify specific weaknesses.
        
        Args:
            scores: Utterance-level scores from assessment
            
        Returns:
            List of identified weakness areas
        """
        weaknesses = []
        
        if scores.get("accuracy", 10) < 6:
            weaknesses.append("pronunciation_accuracy")
        if scores.get("fluency", 10) < 6:
            weaknesses.append("speech_fluency")
        if scores.get("prosodic", 10) < 6:
            weaknesses.append("rhythm_intonation")
        if scores.get("completeness", 10) < 6:
            weaknesses.append("word_completeness")
        
        return weaknesses
    
    @staticmethod
    def get_learning_plan(weaknesses: List[str], overall_score: float) -> Dict[str, Any]:
        """
        Generate a personalized learning plan based on weaknesses.
        
        Args:
            weaknesses: List of identified weakness areas
            overall_score: Overall assessment score
            
        Returns:
            Structured learning plan
        """
        # Determine difficulty level
        if overall_score < 4:
            level = "beginner"
            duration = "4-6 weeks"
            sessions_per_week = 4
        elif overall_score < 6:
            level = "intermediate"
            duration = "3-4 weeks"
            sessions_per_week = 3
        else:
            level = "advanced"
            duration = "2-3 weeks"
            sessions_per_week = 2
        
        plan = {
            "level": level,
            "duration": duration,
            "sessions_per_week": sessions_per_week,
            "focus_areas": [],
            "exercises": [],
            "practice_sentences": [],
            "resources": []
        }
        
        # Add focus areas and exercises based on weaknesses
        if "pronunciation_accuracy" in weaknesses:
            plan["focus_areas"].append({
                "area": "Pronunciation Accuracy",
                "priority": "High",
                "description": "Improve individual sound production and word pronunciation"
            })
            plan["exercises"].extend([
                "Phoneme drill exercises",
                "Minimal pair practice",
                "IPA chart study",
                "Mirror practice sessions"
            ])
        
        if "speech_fluency" in weaknesses:
            plan["focus_areas"].append({
                "area": "Speech Fluency",
                "priority": "High",
                "description": "Develop smooth and natural speech flow"
            })
            plan["exercises"].extend([
                "Read-aloud practice",
                "Tongue twisters",
                "Speed reading exercises",
                "Connected speech practice"
            ])
        
        if "rhythm_intonation" in weaknesses:
            plan["focus_areas"].append({
                "area": "Rhythm & Intonation",
                "priority": "Medium",
                "description": "Master stress patterns and melody of English"
            })
            plan["exercises"].extend([
                "Stress pattern drills",
                "Intonation copying exercises",
                "Poetry reading",
                "Music and rhythm practice"
            ])
        
        if "word_completeness" in weaknesses:
            plan["focus_areas"].append({
                "area": "Word Completeness",
                "priority": "Medium",
                "description": "Ensure all sounds and syllables are pronounced"
            })
            plan["exercises"].extend([
                "Syllable counting exercises",
                "Word boundary practice",
                "Slow speech drills",
                "Recording and playback analysis"
            ])
        
        # Add practice sentences based on level
        if level == "beginner":
            plan["practice_sentences"] = [
                "The cat sat on the mat.",
                "She sells seashells by the seashore.",
                "Peter Piper picked a peck of pickled peppers.",
                "How much wood would a woodchuck chuck?",
                "Red lorry, yellow lorry.",
                "Unique New York, unique New York.",
                "Six sick slick slim sycamore saplings.",
                "Betty Botter bought some butter."
            ]
        elif level == "intermediate":
            plan["practice_sentences"] = [
                "The quick brown fox jumps over the lazy dog.",
                "She thoroughly thought through the thick and thin theories.",
                "The thirty-three thieves thought they thrilled the throne.",
                "Around the rugged rock the ragged rascal ran.",
                "Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair.",
                "I scream, you scream, we all scream for ice cream.",
                "How can a clam cram in a clean cream can?",
                "Fresh fried fish, fish fresh fried, fried fish fresh."
            ]
        else:
            plan["practice_sentences"] = [
                "The sixth sick sheik's sixth sheep's sick.",
                "Irish wristwatch, Swiss wristwatch.",
                "Pad kid poured curd pulled cod.",
                "Which witch switched the Swiss wristwatch?",
                "A proper copper coffee pot copper coffee cup.",
                "Six sleek swans swam swiftly southwards.",
                "Brisk brave brigadiers brandished broad bright blades.",
                "Mix a box of mixed biscuits with a boxed biscuit mixer."
            ]
        
        # Add learning resources
        plan["resources"] = [
            {
                "type": "App",
                "name": "Sounds: The Pronunciation App",
                "description": "Interactive phoneme practice"
            },
            {
                "type": "Website",
                "name": "Rachel's English",
                "description": "American English pronunciation videos"
            },
            {
                "type": "YouTube",
                "name": "BBC Learning English",
                "description": "British English pronunciation guides"
            },
            {
                "type": "Book",
                "name": "Ship or Sheep?",
                "description": "Pronunciation practice book by Ann Baker"
            }
        ]
        
        return plan
    
    @staticmethod
    def render_progress_tracker() -> None:
        """Render a simple progress tracker."""
        st.subheader("📈 Track Your Progress")
        
        # Sample progress data (in real app, this would be stored)
        progress_data = {
            "Week 1": 65,
            "Week 2": 70,
            "Week 3": 75,
            "Current": 78
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(progress_data)
        
        with col2:
            st.metric("Current Score", "7.8/10", delta="0.3")
            st.metric("Sessions Completed", "12/24", delta="3")
            st.metric("Days Practiced", "18/21", delta="2")
    
    @staticmethod
    def render_daily_practice(plan: Dict[str, Any]) -> None:
        """Render daily practice recommendations."""
        st.subheader("📅 Today's Practice Session")
        
        # Select random exercises for today
        daily_exercises = random.sample(plan["exercises"], min(3, len(plan["exercises"])))
        daily_sentence = random.choice(plan["practice_sentences"])
        
        st.markdown("### 🎯 Focus Exercises (15-20 minutes)")
        for i, exercise in enumerate(daily_exercises, 1):
            st.markdown(f"{i}. **{exercise}** - 5 minutes")
        
        st.markdown("### 📝 Practice Sentence")
        st.info(f"**Today's sentence:** {daily_sentence}")
        st.markdown("Record yourself saying this sentence 5 times and compare with native speakers.")
        
        # Practice buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎤 Start Practice", type="primary"):
                st.success("✅ Practice session started! Good luck!")
        
        with col2:
            if st.button("✅ Mark Complete"):
                st.success("🎉 Great job! Session marked as complete.")
        
        with col3:
            if st.button("📊 View Progress"):
                st.info("Progress tracking coming soon!")
    
    @staticmethod
    def render_learning_plan(assessment_result: Dict[str, Any]) -> None:
        """
        Render the complete learning plan page.
        
        Args:
            assessment_result: Previous assessment result
        """
        st.title("📚 Your Personalized Learning Plan")
        st.markdown("*Based on your recent pronunciation assessment*")
        
        # Extract scores
        scores = assessment_result.get("scores", {}).get("utterance_level", {})
        overall_score = scores.get("total", 0)
        
        # Analyze weaknesses
        weaknesses = LearningPlanComponent.analyze_weaknesses(scores)
        plan = LearningPlanComponent.get_learning_plan(weaknesses, overall_score)
        
        # Plan overview
        st.markdown("---")
        st.subheader("🎯 Plan Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Level", plan["level"].title())
        with col2:
            st.metric("Duration", plan["duration"])
        with col3:
            st.metric("Sessions/Week", plan["sessions_per_week"])
        with col4:
            st.metric("Current Score", f"{overall_score:.1f}/10")
        
        # Focus areas
        st.markdown("---")
        st.subheader("🔍 Focus Areas")
        
        for area in plan["focus_areas"]:
            priority_color = "🔴" if area["priority"] == "High" else "🟡"
            st.markdown(f"""
            **{priority_color} {area['area']}** - *{area['priority']} Priority*
            
            {area['description']}
            """)
        
        # Daily practice
        st.markdown("---")
        LearningPlanComponent.render_daily_practice(plan)
        
        # Exercises
        st.markdown("---")
        st.subheader("🏋️ Recommended Exercises")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Core Exercises:**")
            for exercise in plan["exercises"][:len(plan["exercises"])//2]:
                st.markdown(f"• {exercise}")
        
        with col2:
            st.markdown("**Additional Practice:**")
            for exercise in plan["exercises"][len(plan["exercises"])//2:]:
                st.markdown(f"• {exercise}")
        
        # Practice sentences
        st.markdown("---")
        st.subheader("📝 Practice Sentences")
        st.markdown("*Practice these sentences regularly to improve your pronunciation:*")
        
        # Display sentences in expandable sections
        with st.expander("📚 View All Practice Sentences", expanded=False):
            for i, sentence in enumerate(plan["practice_sentences"], 1):
                st.markdown(f"{i}. {sentence}")
        
        # Resources
        st.markdown("---")
        st.subheader("📖 Learning Resources")
        
        for resource in plan["resources"]:
            st.markdown(f"""
            **{resource['type']}: {resource['name']}**
            
            {resource['description']}
            """)
        
        # Progress tracker
        st.markdown("---")
        LearningPlanComponent.render_progress_tracker()
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎤 Take New Assessment", type="primary", use_container_width=True):
                st.session_state.current_page = "assessment"
                st.rerun()
        
        with col2:
            if st.button("💾 Save Plan", use_container_width=True):
                st.success("✅ Learning plan saved!")
        
        with col3:
            if st.button("🏠 Back to Home", use_container_width=True):
                st.session_state.current_page = "landing"
                st.rerun()
    
    @staticmethod
    def render_custom_css():
        """Render custom CSS for the learning plan page."""
        st.markdown("""
        <style>
            .metric-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin: 0.5rem 0;
            }
            .focus-area {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                margin: 0.5rem 0;
            }
            .exercise-card {
                background-color: #e3f2fd;
                padding: 0.75rem;
                border-radius: 6px;
                margin: 0.25rem 0;
                border-left: 3px solid #2196f3;
            }
            .resource-card {
                background-color: #f3e5f5;
                padding: 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                border-left: 4px solid #9c27b0;
            }
        </style>
        """, unsafe_allow_html=True)