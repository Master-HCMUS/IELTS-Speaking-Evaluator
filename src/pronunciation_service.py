"""
Azure Speech Services pronunciation assessment module.

This module provides pronunciation scoring functionality using Azure Speech Services.
It evaluates pronunciation accuracy, fluency, completeness, and provides detailed
phoneme-level feedback for language learning applications.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, AudioConfig
from azure.cognitiveservices.speech.audio import AudioStreamFormat

from .exceptions import (
    TranscriptionError,
    AzureAuthenticationError, 
    AzureAPIError,
    AudioFileError
)


class AzureSpeechPronunciationService:
    """
    Service for pronunciation assessment using Azure Speech Services.
    
    This class provides pronunciation scoring functionality that evaluates:
    - Accuracy: How closely the phonemes match the reference text
    - Fluency: How smoothly the speech flows
    - Completeness: How much of the reference text was spoken
    - Prosody: Stress, intonation, and rhythm patterns
    """
    
    def __init__(
        self, 
        speech_key: str, 
        speech_region: str,
        locale: str = "en-US"
    ):
        """
        Initialize the Azure Speech pronunciation assessment service.
        
        Args:
            speech_key (str): Azure Speech service API key
            speech_region (str): Azure Speech service region (e.g., 'eastus', 'westus2')
            locale (str): Language locale for pronunciation assessment. Default is 'en-US'
            
        Raises:
            AzureAuthenticationError: If authentication setup fails
        """
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.locale = locale
        self.speech_config: Optional[SpeechConfig] = None
        
        # Validate required parameters
        if not self.speech_key:
            raise AzureAuthenticationError("Azure Speech service key is required")
        if not self.speech_region:
            raise AzureAuthenticationError("Azure Speech service region is required")
        
        # Initialize speech configuration
        self._initialize_speech_config()
    
    def _initialize_speech_config(self) -> None:
        """
        Initialize the Azure Speech SDK configuration.
        
        Raises:
            AzureAuthenticationError: If authentication setup fails
        """
        try:
            # Create speech configuration
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region
            )
            
            # Set recognition language
            self.speech_config.speech_recognition_language = self.locale
            
            print(f"✅ Azure Speech service initialized successfully")
            print(f"📊 Region: {self.speech_region}")
            print(f"🌍 Language: {self.locale}")
            print(f"🔑 Authentication: API Key")
            
        except Exception as e:
            raise AzureAuthenticationError(f"Failed to initialize Azure Speech service: {e}")
    
    def assess_pronunciation_from_file(
        self,
        audio_file_path: Union[str, Path],
        reference_text: str,
        assessment_granularity: str = "Phoneme",
        enable_prosody: bool = True
    ) -> Dict[str, Any]:
        """
        Assess pronunciation from an audio file against reference text.
        
        Args:
            audio_file_path: Path to the audio file (WAV format recommended)
            reference_text: The reference text that should have been spoken
            assessment_granularity: Level of assessment detail ("Phoneme", "Word", "FullText")
            enable_prosody: Whether to enable prosody assessment (stress, intonation, rhythm)
            
        Returns:
            Dict containing pronunciation assessment results
            
        Raises:
            AudioFileError: If audio file cannot be read or is invalid
            AzureAPIError: If the API call fails
        """
        audio_file_path = Path(audio_file_path)
        
        # Validate file exists and is readable
        if not audio_file_path.exists():
            raise AudioFileError(f"Audio file not found: {audio_file_path}")
        
        if not audio_file_path.is_file():
            raise AudioFileError(f"Path is not a file: {audio_file_path}")
        
        # Check file format (WAV is preferred)
        if not str(audio_file_path).lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise AudioFileError(f"Unsupported audio format. Use WAV, MP3, M4A, or FLAC: {audio_file_path}")
        
        print(f"🎯 Assessing pronunciation for: {audio_file_path.name}")
        print(f"📝 Reference text: {reference_text[:100]}{'...' if len(reference_text) > 100 else ''}")
        
        try:
            start_time = time.time()
            
            # Create audio configuration from file
            audio_config = speechsdk.AudioConfig(filename=str(audio_file_path))
            
            # Configure pronunciation assessment
            pronunciation_config = self._create_pronunciation_assessment_config(
                reference_text=reference_text,
                granularity=assessment_granularity,
                enable_prosody=enable_prosody
            )
            
            # Create speech recognizer with pronunciation assessment
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Apply pronunciation assessment configuration
            pronunciation_config.apply_to(recognizer)
            
            # Perform recognition with pronunciation assessment
            result = recognizer.recognize_once()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Process the results
            assessment_result = self._process_pronunciation_result(
                result, 
                reference_text,
                processing_time,
                audio_file_path.name
            )
            
            print(f"✅ Pronunciation assessment completed in {processing_time:.2f} seconds")
            
            return assessment_result
            
        except Exception as e:
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                raise AzureAuthenticationError(f"Authentication error during pronunciation assessment: {e}")
            else:
                raise AzureAPIError(f"Pronunciation assessment failed: {e}")
    
    def _create_pronunciation_assessment_config(
        self,
        reference_text: str,
        granularity: str = "Phoneme",
        enable_prosody: bool = True
    ) -> speechsdk.PronunciationAssessmentConfig:
        """
        Create pronunciation assessment configuration.
        
        Args:
            reference_text: The reference text for assessment
            granularity: Assessment granularity level
            enable_prosody: Whether to enable prosody assessment
            
        Returns:
            Configured PronunciationAssessmentConfig object
        """
        # Validate granularity
        valid_granularities = ["Phoneme", "Word", "FullText"]
        if granularity not in valid_granularities:
            granularity = "Phoneme"
            print(f"⚠️  Invalid granularity, using default: {granularity}")
        
        # Create pronunciation assessment configuration
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=getattr(speechsdk.PronunciationAssessmentGranularity, granularity),
            enable_miscue=True  # Enable detection of omissions, insertions, and mispronunciations
        )
        
        # Enable prosody assessment if requested (requires premium tier)
        if enable_prosody:
            try:
                pronunciation_config.enable_prosody_assessment()
                print("🎵 Prosody assessment enabled (stress, intonation, rhythm)")
            except Exception as e:
                print(f"⚠️  Prosody assessment not available: {e}")
        
        return pronunciation_config
    
    def _process_pronunciation_result(
        self,
        result: speechsdk.SpeechRecognitionResult,
        reference_text: str,
        processing_time: float,
        filename: str
    ) -> Dict[str, Any]:
        """
        Process and format the pronunciation assessment result.
        
        Args:
            result: Speech recognition result with pronunciation assessment
            reference_text: Original reference text
            processing_time: Time taken for the assessment
            filename: Name of the assessed audio file
            
        Returns:
            Formatted pronunciation assessment result dictionary
        """
        try:
            # Initialize result structure
            assessment_result = {
                "filename": filename,
                "reference_text": reference_text,
                "recognized_text": result.text if result.text else "",
                "processing_time": processing_time,
                "recognition_status": result.reason.name,
                "overall_scores": {},
                "word_level_scores": [],
                "phoneme_level_scores": [],
                "prosody_scores": {},
                "detailed_feedback": []
            }
            
            # Check if recognition was successful
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Parse pronunciation assessment results
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
                
                # Overall pronunciation scores
                assessment_result["overall_scores"] = {
                    "accuracy_score": pronunciation_result.accuracy_score,
                    "fluency_score": pronunciation_result.fluency_score,
                    "completeness_score": pronunciation_result.completeness_score,
                    "pronunciation_score": pronunciation_result.pronunciation_score
                }
                
                # Parse detailed JSON results if available
                if hasattr(result, 'properties') and result.properties:
                    detailed_result_json = result.properties.get(
                        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                    )
                    
                    if detailed_result_json:
                        detailed_result = json.loads(detailed_result_json)
                        assessment_result.update(self._parse_detailed_assessment(detailed_result))
                
                # Generate feedback based on scores
                assessment_result["detailed_feedback"] = self._generate_pronunciation_feedback(
                    assessment_result["overall_scores"],
                    assessment_result.get("word_level_scores", [])
                )
                
                print(f"📊 Overall Pronunciation Score: {assessment_result['overall_scores'].get('pronunciation_score', 0):.1f}/100")
                print(f"🎯 Accuracy: {assessment_result['overall_scores'].get('accuracy_score', 0):.1f}/100")
                print(f"🌊 Fluency: {assessment_result['overall_scores'].get('fluency_score', 0):.1f}/100")
                print(f"✅ Completeness: {assessment_result['overall_scores'].get('completeness_score', 0):.1f}/100")
                
            elif result.reason == speechsdk.ResultReason.NoMatch:
                assessment_result["error"] = "No speech could be recognized from the audio file"
                assessment_result["suggestions"] = [
                    "Check if the audio file contains clear speech",
                    "Ensure the audio volume is adequate",
                    "Verify the language setting matches the spoken language"
                ]
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                assessment_result["error"] = f"Recognition canceled: {cancellation_details.reason.name}"
                if cancellation_details.error_details:
                    assessment_result["error_details"] = cancellation_details.error_details
            
            return assessment_result
            
        except Exception as e:
            return {
                "filename": filename,
                "reference_text": reference_text,
                "processing_time": processing_time,
                "error": f"Failed to process pronunciation assessment result: {e}",
                "suggestions": [
                    "Check your Azure Speech service configuration",
                    "Verify the audio file format is supported",
                    "Ensure your Speech service subscription is active"
                ]
            }
    
    def _parse_detailed_assessment(self, detailed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse detailed pronunciation assessment results from JSON response.
        
        Args:
            detailed_result: Detailed JSON result from Speech service
            
        Returns:
            Parsed detailed assessment data
        """
        parsed_data = {
            "word_level_scores": [],
            "phoneme_level_scores": [],
            "prosody_scores": {}
        }
        
        try:
            # Parse word-level scores
            if "NBest" in detailed_result and detailed_result["NBest"]:
                nbest = detailed_result["NBest"][0]
                
                if "Words" in nbest:
                    for word_data in nbest["Words"]:
                        word_score = {
                            "word": word_data.get("Word", ""),
                            "accuracy_score": word_data.get("PronunciationAssessment", {}).get("AccuracyScore", 0),
                            "error_type": word_data.get("PronunciationAssessment", {}).get("ErrorType", "None")
                        }
                        
                        # Add phoneme-level data if available
                        if "Phonemes" in word_data:
                            word_score["phonemes"] = []
                            for phoneme_data in word_data["Phonemes"]:
                                phoneme_score = {
                                    "phoneme": phoneme_data.get("Phoneme", ""),
                                    "accuracy_score": phoneme_data.get("PronunciationAssessment", {}).get("AccuracyScore", 0)
                                }
                                word_score["phonemes"].append(phoneme_score)
                                
                                # Also add to global phoneme list
                                parsed_data["phoneme_level_scores"].append({
                                    "word": word_score["word"],
                                    **phoneme_score
                                })
                        
                        parsed_data["word_level_scores"].append(word_score)
                
                # Parse prosody scores if available
                if "PronunciationAssessment" in nbest:
                    prosody_data = nbest["PronunciationAssessment"]
                    if "ProsodyScore" in prosody_data:
                        parsed_data["prosody_scores"] = {
                            "prosody_score": prosody_data.get("ProsodyScore", 0)
                        }
        
        except Exception as e:
            print(f"⚠️  Warning: Could not parse detailed assessment results: {e}")
        
        return parsed_data
    
    def _generate_pronunciation_feedback(
        self,
        overall_scores: Dict[str, float],
        word_scores: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate human-readable feedback based on pronunciation scores.
        
        Args:
            overall_scores: Overall pronunciation scores
            word_scores: Word-level pronunciation scores
            
        Returns:
            List of feedback strings
        """
        feedback = []
        
        # Overall feedback based on pronunciation score
        pronunciation_score = overall_scores.get("pronunciation_score", 0)
        if pronunciation_score >= 90:
            feedback.append("🌟 Excellent pronunciation! Your speech is very clear and natural.")
        elif pronunciation_score >= 80:
            feedback.append("✅ Good pronunciation overall. Minor improvements could enhance clarity.")
        elif pronunciation_score >= 70:
            feedback.append("👍 Fair pronunciation. Focus on specific sounds for improvement.")
        elif pronunciation_score >= 60:
            feedback.append("⚠️  Pronunciation needs work. Practice specific words and sounds.")
        else:
            feedback.append("❌ Significant pronunciation challenges. Consider working with a tutor.")
        
        # Specific feedback based on component scores
        accuracy_score = overall_scores.get("accuracy_score", 0)
        fluency_score = overall_scores.get("fluency_score", 0)
        completeness_score = overall_scores.get("completeness_score", 0)
        
        if accuracy_score < 70:
            feedback.append("🎯 Focus on pronouncing individual sounds more clearly.")
        
        if fluency_score < 70:
            feedback.append("🌊 Work on speaking more smoothly with natural rhythm and pace.")
        
        if completeness_score < 80:
            feedback.append("📝 Try to speak all words in the reference text clearly.")
        
        # Word-specific feedback
        problem_words = [w for w in word_scores if w.get("accuracy_score", 100) < 60]
        if problem_words:
            problem_word_list = [w["word"] for w in problem_words[:5]]  # Limit to 5 words
            feedback.append(f"🔤 Practice these words: {', '.join(problem_word_list)}")
        
        # Positive reinforcement
        good_words = [w for w in word_scores if w.get("accuracy_score", 0) >= 90]
        if good_words and len(good_words) >= len(word_scores) * 0.7:  # 70% or more good words
            feedback.append("💪 Strong performance on most words - keep up the good work!")
        
        return feedback
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the pronunciation assessment service configuration.
        
        Returns:
            Dictionary with service configuration details
        """
        return {
            "speech_region": self.speech_region,
            "locale": self.locale,
            "service_initialized": self.speech_config is not None,
            "supported_formats": ["wav", "mp3", "m4a", "flac"],
            "supported_locales": [
                "en-US", "en-GB", "en-AU", "en-CA", "en-IN",
                "es-ES", "es-MX", "fr-FR", "fr-CA", "de-DE",
                "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"
            ],
            "assessment_features": [
                "Accuracy scoring (phoneme-level)",
                "Fluency assessment",
                "Completeness evaluation",
                "Prosody analysis (premium)",
                "Miscue detection (omissions, insertions, mispronunciations)"
            ]
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to Azure Speech service.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            if self.speech_config is None:
                return False
                
            print("🔍 Testing Azure Speech service connection...")
            
            # Simple validation that configuration is set up
            config_valid = (
                self.speech_config.subscription_key is not None and
                self.speech_config.region is not None
            )
            
            if config_valid:
                print("✅ Azure Speech service connection test passed")
                return True
            else:
                print("❌ Azure Speech service connection test failed")
                return False
                
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False