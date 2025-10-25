"""
Pronunciation Assessment Trainer.

This module implements the training loop for Whisper pronunciation assessment models
with multi-objective losses (ASR + pronunciation assessment at multiple granularities).
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import DatasetDict
import evaluate

# Local imports
from .whisper_pronunciation_model import WhisperPronunciationAssessmentModel
from .data_processor import SpeechOcean762PronunciationProcessor
from .training_config import PronunciationTrainingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PronunciationAssessmentCustomTrainer(Trainer):
    """
    Custom trainer that handles shared tensor saving issues in Whisper models.
    """
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Override save method to handle shared tensors properly."""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        # Use our custom save method instead of the default safetensors
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
        else:
            # Fallback to torch.save for shared tensor handling
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save tokenizer and configuration if available
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class PronunciationMetricsCallback(TrainerCallback):
    """Custom callback to log detailed pronunciation assessment metrics during training."""
    
    def __init__(self, eval_dataset, processor, output_dir):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.metrics_history = []
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log custom metrics after each evaluation."""
        if state.log_history:
            latest_log = state.log_history[-1]
            
            # Extract detailed loss information
            detailed_metrics = {
                "step": state.global_step,
                "epoch": state.epoch,
                "total_loss": latest_log.get("eval_loss", 0),
                "individual_losses": {},
                "assessment_metrics": {}
            }
            
            # Extract individual loss components
            for key, value in latest_log.items():
                if key.startswith("eval_") and "loss" in key:
                    loss_name = key.replace("eval_", "").replace("_loss", "")
                    detailed_metrics["individual_losses"][loss_name] = value
            
            self.metrics_history.append(detailed_metrics)
            
            # Save metrics history
            metrics_file = self.output_dir / "pronunciation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)


class PronunciationAssessmentTrainer:
    """
    Trainer for pronunciation assessment fine-tuning.
    
    Handles multi-objective training with ASR and pronunciation assessment losses
    at word, phone, and utterance levels.
    """
    
    def __init__(self, config: PronunciationTrainingConfig):
        """
        Initialize the pronunciation assessment trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.data_processor = None
        self.trainer = None
        
        # Metrics for evaluation
        self.mse_metric = lambda preds, labels: np.mean((preds - labels) ** 2)
        self.mae_metric = lambda preds, labels: np.mean(np.abs(preds - labels))
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"Initialized PronunciationAssessmentTrainer")
        self.config.print_config()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pronunciation_training_{timestamp}.log"
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    def load_model(self):
        """Load and configure the pronunciation assessment model."""
        logger.info(f"Loading pronunciation assessment model: {self.config.whisper_model_name}")
        
        try:
            self.model = WhisperPronunciationAssessmentModel(
                whisper_model_name=self.config.whisper_model_name,
                assessment_dropout=self.config.assessment_dropout,
                freeze_whisper_layers=self.config.freeze_whisper_layers
            )
            
            # Update loss weights
            self.model.loss_weights = self.config.loss_weights
            
            logger.info("Model loaded successfully")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_datasets(self) -> DatasetDict:
        """
        Prepare datasets for training.
        
        Returns:
            Processed datasets ready for training
        """
        logger.info("Preparing datasets...")
        
        # Initialize data processor
        self.data_processor = SpeechOcean762PronunciationProcessor(
            whisper_model_name=self.config.whisper_model_name,
            sampling_rate=self.config.sampling_rate,
            max_audio_length=self.config.max_audio_length,
            normalize_audio=self.config.normalize_audio
        )
        
        # Load raw datasets
        splits = [self.config.train_split, self.config.eval_split]
        max_samples = {
            self.config.train_split: self.config.max_train_samples,
            self.config.eval_split: self.config.max_eval_samples
        }
        
        raw_datasets = self.data_processor.load_dataset(
            splits=splits,
            max_samples_per_split=max_samples
        )
        
        # Process datasets for training
        processed_datasets = self.data_processor.prepare_dataset_for_training(
            raw_datasets,
            include_transcription=self.config.include_transcription
        )
        
        # Log dataset statistics
        stats = self.data_processor.get_dataset_statistics(processed_datasets)
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2, default=str)}")
        
        return processed_datasets
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics for pronunciation assessment.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of computed metrics
        """
        # Note: This is a simplified version. In practice, you'd need to
        # extract the actual prediction values from the model outputs
        # and compare them with the targets.
        
        predictions, labels = eval_pred
        
        # Basic metrics - in real implementation, you'd process the
        # pronunciation predictions separately
        metrics = {
            "mse": np.mean((predictions - labels) ** 2) if labels is not None else 0.0,
            "mae": np.mean(np.abs(predictions - labels)) if labels is not None else 0.0,
        }
        
        return metrics
    
    def setup_trainer(self, datasets: DatasetDict) -> Trainer:
        """
        Setup the trainer for pronunciation assessment fine-tuning.
        
        Args:
            datasets: Processed datasets
            
        Returns:
            Configured trainer
        """
        logger.info("Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            num_train_epochs=self.config.num_epochs,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=["tensorboard"],
            run_name=self.config.run_name,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler,
            remove_unused_columns=False,  # Important for custom data
        )
        
        # Data collator
        data_collator = self.data_processor.create_data_collator()
        
        # Custom callback for pronunciation metrics
        pronunciation_callback = PronunciationMetricsCallback(
            eval_dataset=datasets[self.config.eval_split],
            processor=self.data_processor.processor,
            output_dir=self.config.output_dir
        )
        
        # Create trainer
        self.trainer = PronunciationAssessmentCustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets[self.config.train_split],
            eval_dataset=datasets[self.config.eval_split],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[pronunciation_callback]
        )
        
        logger.info("Trainer setup completed")
        return self.trainer
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the pronunciation assessment fine-tuning process.
        
        Returns:
            Training results and metrics
        """
        logger.info("Starting pronunciation assessment fine-tuning...")
        
        try:
            # Load model
            self.load_model()
            
            # Prepare datasets
            datasets = self.prepare_datasets()
            
            # Setup trainer
            trainer = self.setup_trainer(datasets)
            
            # Save configuration
            config_path = Path(self.config.output_dir) / "training_config.json"
            self.config.save(str(config_path))
            logger.info(f"Training configuration saved to: {config_path}")
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Save the final model using our custom method
            logger.info("Saving final model...")
            self.model.save_pretrained(self.config.output_dir)
            self.data_processor.processor.save_pretrained(self.config.output_dir)
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Run final evaluation
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            
            # Generate sample predictions
            sample_predictions = self.generate_sample_predictions(datasets[self.config.eval_split])
            
            # Save training summary
            training_summary = {
                "config": self.config.to_dict(),
                "train_results": {
                    "training_loss": train_result.training_loss,
                    "train_runtime": train_result.metrics.get("train_runtime", 0),
                    "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                },
                "eval_results": eval_results,
                "sample_predictions": sample_predictions,
                "model_path": str(Path(self.config.output_dir)),
                "timestamp": datetime.now().isoformat()
            }
            
            summary_path = Path(self.config.output_dir) / "training_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(training_summary, f, indent=2, default=str)
            
            logger.info(f"Training summary saved to: {summary_path}")
            logger.info(f"Model saved to: {self.config.output_dir}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def generate_sample_predictions(self, eval_dataset, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate sample predictions for analysis."""
        samples = eval_dataset.select(range(min(num_samples, len(eval_dataset))))
        predictions = []
        
        self.model.eval()
        device = self.config.get_device()
        self.model.to(device)
        
        with torch.no_grad():
            for i, sample in enumerate(samples):
                # Prepare input
                input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
                
                # Get pronunciation predictions
                assessment_predictions = self.model.predict_pronunciation_scores(input_features)
                
                # Get transcription if available
                transcription = None
                if self.config.include_transcription:
                    try:
                        generated_ids = self.model.generate_transcription(input_features, max_length=225)
                        transcription = self.data_processor.processor.tokenizer.decode(
                            generated_ids[0], skip_special_tokens=True
                        ).strip()
                    except:
                        transcription = "Error in transcription"
                
                # Extract target scores
                targets = {}
                if 'word_accuracy_scores' in sample:
                    targets['word_level'] = {
                        'accuracy': sample['word_accuracy_scores'],
                        'stress': sample['word_stress_scores'],
                        'total': sample['word_total_scores']
                    }
                
                if 'phone_accuracy_scores' in sample:
                    targets['phone_level'] = {
                        'accuracy': sample['phone_accuracy_scores']
                    }
                
                utterance_targets = {}
                for score_type in ['accuracy', 'fluency', 'prosodic', 'completeness', 'total']:
                    if score_type in sample:
                        utterance_targets[score_type] = sample[score_type]
                if utterance_targets:
                    targets['utterance_level'] = utterance_targets
                
                predictions.append({
                    "sample_id": i,
                    "transcription": transcription,
                    "reference_text": sample.get("transcription", "N/A"),
                    "assessment_predictions": {
                        k: {kk: vv.cpu().numpy().tolist() if torch.is_tensor(vv) else vv 
                            for kk, vv in v.items()} 
                        for k, v in assessment_predictions.items()
                    },
                    "targets": targets,
                    "metadata": {
                        "speaker": sample.get("speaker"),
                        "gender": sample.get("gender"),
                        "age": sample.get("age")
                    }
                })
        
        return predictions
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trained pronunciation assessment model.
        
        Args:
            model_path: Path to model directory (uses config.output_dir if None)
            
        Returns:
            Evaluation results
        """
        if model_path is None:
            model_path = self.config.output_dir
        
        logger.info(f"Evaluating pronunciation assessment model from: {model_path}")
        
        # Load model
        model = WhisperPronunciationAssessmentModel.from_pretrained(model_path)
        
        # Prepare evaluation dataset
        self.data_processor = SpeechOcean762PronunciationProcessor(
            whisper_model_name=self.config.whisper_model_name,
            sampling_rate=self.config.sampling_rate,
            max_audio_length=self.config.max_audio_length
        )
        
        eval_datasets = self.data_processor.load_dataset(
            splits=[self.config.eval_split],
            max_samples_per_split={self.config.eval_split: self.config.max_eval_samples}
        )
        
        processed_datasets = self.data_processor.prepare_dataset_for_training(
            eval_datasets, 
            include_transcription=self.config.include_transcription
        )
        eval_dataset = processed_datasets[self.config.eval_split]
        
        # Generate comprehensive predictions
        sample_predictions = self.generate_sample_predictions(eval_dataset, num_samples=50)
        
        results = {
            "sample_predictions": sample_predictions,
            "model_path": model_path,
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation results
        eval_path = Path(model_path) / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {eval_path}")
        return results
    
    @staticmethod
    def load_trained_model(model_path: str) -> WhisperPronunciationAssessmentModel:
        """
        Load a trained pronunciation assessment model.
        
        Args:
            model_path: Path to the trained model directory
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading trained pronunciation assessment model from: {model_path}")
        
        model = WhisperPronunciationAssessmentModel.from_pretrained(model_path)
        
        return model