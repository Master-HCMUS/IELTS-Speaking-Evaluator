"""
Utility functions for the web application.
"""

import numpy as np
from typing import Tuple, List
from .config import COLOR_GOOD, COLOR_FAIR, COLOR_POOR, SCORE_GOOD_MIN, SCORE_FAIR_MIN


def get_score_color(score: float) -> str:
    """
    Get color for score visualization.
    
    Args:
        score: Score between 0-10
        
    Returns:
        Hex color code: Red (low), Orange (medium), Green (high)
    """
    if score >= SCORE_GOOD_MIN:
        return COLOR_GOOD
    elif score >= SCORE_FAIR_MIN:
        return COLOR_FAIR
    else:
        return COLOR_POOR


def get_score_label(score: float) -> str:
    """
    Get label for score.
    
    Args:
        score: Score between 0-10
        
    Returns:
        Label: "Good", "Fair", or "Poor"
    """
    if score >= SCORE_GOOD_MIN:
        return "Good"
    elif score >= SCORE_FAIR_MIN:
        return "Fair"
    else:
        return "Poor"


def highlight_words(words: List[str], scores: List[float]) -> str:
    """
    Create HTML to highlight words based on scores.
    
    Args:
        words: List of words
        scores: List of scores for each word
        
    Returns:
        HTML string with colored word spans
    """
    html = "<div style='font-size: 24px; line-height: 1.8;'>"
    
    for word, score in zip(words, scores):
        color = get_score_color(score)
        label = get_score_label(score)
        html += f"""
        <span style='
            background-color: {color};
            color: white;
            padding: 6px 10px;
            margin: 4px;
            border-radius: 4px;
            font-weight: bold;
            display: inline-block;
            min-width: 80px;
            text-align: center;
            title="{label}: {score:.1f}/10"
        '>{word}<br><small>{score:.1f}</small></span>
        """
    
    html += "</div>"
    return html


def align_words_to_frames(words: List[str], frame_scores: List[float]) -> Tuple[List[str], List[float]]:
    """
    Align words with frame-level scores.
    
    Simple alignment: distribute frame scores evenly to words.
    
    Args:
        words: List of words
        frame_scores: List of frame-level scores
        
    Returns:
        Tuple of (aligned_words, word_scores)
    """
    if not words or not frame_scores:
        return words, [5.0] * len(words)
    
    num_words = len(words)
    num_frames = len(frame_scores)
    
    # Create word scores by averaging frames for each word
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


def get_recommendation(score: float) -> Tuple[str, str, str]:
    """
    Get recommendation based on overall score.
    
    Args:
        score: Overall pronunciation score
        
    Returns:
        Tuple of (emoji, message, type) where type is "success", "info", "warning", or "error"
    """
    if score >= 8:
        return "🎉", "Excellent pronunciation! Keep up the good work!", "success"
    elif score >= 6:
        return "👍", "Good pronunciation with room for improvement. Practice the red-highlighted words.", "info"
    elif score >= 4:
        return "⚠️", "Your pronunciation needs improvement. Focus on words with low scores.", "warning"
    else:
        return "❌", "Pronunciation needs significant improvement. Consider practicing more slowly.", "error"


def format_statistics(scores: np.ndarray) -> dict:
    """
    Calculate statistics from scores array.
    
    Args:
        scores: Array of scores
        
    Returns:
        Dictionary with statistics
    """
    return {
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "median": float(np.median(scores)),
        "q25": float(np.percentile(scores, 25)),
        "q75": float(np.percentile(scores, 75)),
    }
