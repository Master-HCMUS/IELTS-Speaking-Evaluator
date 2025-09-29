"""
Whisper Fine-tuning Module for SpeechOcean762 Dataset.

This module provides comprehensive functionality to fine-tune OpenAI Whisper models
on the SpeechOcean762 dataset for improved speech recognition and pronunciation
assessment capabilities.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

# Transformers and datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import DatasetDict
import evaluate

# Handle imports for both module and standalone execution
import sys
from pathlib import Path

# Add src directory to path for standalone execution
if __name__ == "__main__":
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent
    sys.path.insert(0, str(src_dir))
    from finetuning.training_config import TrainingConfig
    from finetuning.data_processor import SpeechOcean762DataProcessor
else:
    # Module imports
    from .training_config import TrainingConfig
    from .data_processor import SpeechOcean762DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PronunciationMetricsCallback(TrainerCallback):
    """Custom callback to log pronunciation assessment metrics during training."""
    
    def __init__(self, eval_dataset, processor, output_dir):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.metrics_history = []
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log custom metrics after each evaluation."""
        # Save current metrics
        if state.log_history:
            latest_log = state.log_history[-1]
            self.metrics_history.append({
                "step": state.global_step,
                "epoch": state.epoch,
                "metrics": latest_log
            })
            
            # Save metrics history
            metrics_file = self.output_dir / "pronunciation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)


class WhisperFineTuner:
    """
    Fine-tuner for Whisper models on SpeechOcean762 dataset.
    
    Provides comprehensive functionality for fine-tuning Whisper models with
    pronunciation assessment capabilities.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.feature_extractor = None
        self.data_processor = None
        self.trainer = None
        
        # Metrics
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"Initialized WhisperFineTuner with model: {config.model_name}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    def load_model_and_processor(self):
        """Load Whisper model and processor."""
        logger.info(f"Loading model and processor: {self.config.model_name}")
        
        try:
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Load processor components
            self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
            self.tokenizer = WhisperTokenizer.from_pretrained(self.config.model_name)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config.model_name)
            
            # Configure model for fine-tuning
            self.model.config.forced_decoder_ids = self.config.forced_decoder_ids
            self.model.config.suppress_tokens = self.config.suppress_tokens
            
            # Enable gradient computation for all parameters
            self.model.train()
            
            logger.info("Model and processor loaded successfully")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model and processor: {e}")
            raise
    
    def prepare_datasets(self) -> DatasetDict:
        """
        Prepare datasets for training.
        
        Returns:
            Processed datasets ready for training
        """
        logger.info("Preparing datasets...")
        
        # Initialize data processor
        self.data_processor = SpeechOcean762DataProcessor(
            model_name=self.config.model_name,
            sampling_rate=self.config.sampling_rate,
            max_audio_length=self.config.max_audio_length,
            normalize_audio=True
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
            add_pronunciation_scores=True
        )
        
        # Log dataset statistics
        stats = self.data_processor.get_dataset_statistics(processed_datasets)
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2, default=str)}")
        
        return processed_datasets
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions from trainer
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up text
        decoded_preds = [pred.strip().upper() for pred in decoded_preds]
        decoded_labels = [label.strip().upper() for label in decoded_labels]
        
        # Compute WER (Word Error Rate)
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute BLEU score
        # Format references for BLEU (expects list of lists)
        bleu_references = [[label.split()] for label in decoded_labels]
        bleu_predictions = [pred.split() for pred in decoded_preds]
        
        try:
            bleu = self.bleu_metric.compute(
                predictions=bleu_predictions,
                references=bleu_references
            )
            bleu_score = bleu['bleu']
        except:
            bleu_score = 0.0
        
        # Character-level accuracy
        char_accuracy = self._compute_character_accuracy(decoded_preds, decoded_labels)
        
        return {
            "wer": wer,
            "bleu": bleu_score,
            "char_accuracy": char_accuracy
        }
    
    def _compute_character_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute character-level accuracy."""
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            total_chars += len(ref)
            correct_chars += sum(1 for p, r in zip(pred, ref) if p == r)
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def setup_trainer(self, datasets: DatasetDict) -> Seq2SeqTrainer:
        """
        Setup the trainer for fine-tuning.
        
        Args:
            datasets: Processed datasets
            
        Returns:
            Configured trainer
        """
        logger.info("Setting up trainer...")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            num_train_epochs=self.config.num_epochs,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            predict_with_generate=True,
            generation_max_length=225,
            report_to=["tensorboard"],
            run_name=self.config.run_name,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler,
        )
        
        # Data collator
        data_collator = self.data_processor.create_data_collator()
        
        # Custom callback for pronunciation metrics
        pronunciation_callback = PronunciationMetricsCallback(
            eval_dataset=datasets[self.config.eval_split],
            processor=self.processor,
            output_dir=self.config.output_dir
        )
        
        # Create trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets[self.config.train_split],
            eval_dataset=datasets[self.config.eval_split],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[pronunciation_callback]
        )
        
        logger.info("Trainer setup completed")
        return self.trainer
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the fine-tuning process.
        
        Returns:
            Training results and metrics
        """
        logger.info("Starting fine-tuning process...")
        
        try:
            # Load model and processor
            self.load_model_and_processor()
            
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
            
            # Save the final model
            trainer.save_model()
            self.processor.save_pretrained(self.config.output_dir)
            
            # Log training results
            logger.info("Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Run final evaluation
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            
            # Save training summary
            training_summary = {
                "config": self.config.to_dict(),
                "train_results": {
                    "training_loss": train_result.training_loss,
                    "train_runtime": train_result.metrics.get("train_runtime", 0),
                    "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                },
                "eval_results": eval_results,
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
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to model directory (uses config.output_dir if None)
            
        Returns:
            Evaluation results
        """
        if model_path is None:
            model_path = self.config.output_dir
        
        logger.info(f"Evaluating model from: {model_path}")
        
        # Load model and processor
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        # Prepare evaluation dataset
        self.data_processor = SpeechOcean762DataProcessor(
            model_name=self.config.model_name,
            sampling_rate=self.config.sampling_rate,
            max_audio_length=self.config.max_audio_length
        )
        
        eval_datasets = self.data_processor.load_dataset(
            splits=[self.config.eval_split],
            max_samples_per_split={self.config.eval_split: self.config.max_eval_samples}
        )
        
        processed_datasets = self.data_processor.prepare_dataset_for_training(eval_datasets)
        eval_dataset = processed_datasets[self.config.eval_split]
        
        # Setup trainer for evaluation
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.eval_batch_size,
            predict_with_generate=True,
            generation_max_length=225,
        )
        
        data_collator = self.data_processor.create_data_collator()
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=processor.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Generate sample predictions
        sample_predictions = self.generate_sample_predictions(
            model, processor, eval_dataset, num_samples=10
        )
        
        results = {
            "eval_metrics": eval_results,
            "sample_predictions": sample_predictions,
            "model_path": model_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation results
        eval_path = Path(model_path) / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {eval_path}")
        return results
    
    def generate_sample_predictions(
        self,
        model,
        processor,
        dataset,
        num_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate sample predictions for analysis."""
        samples = dataset.select(range(min(num_samples, len(dataset))))
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(samples):
                # Get input features
                input_features = torch.tensor(sample["input_features"]).unsqueeze(0)
                
                # Generate prediction
                predicted_ids = model.generate(
                    input_features,
                    max_length=225,
                    num_beams=5,
                    early_stopping=True
                )
                
                # Decode prediction and reference
                predicted_text = processor.tokenizer.decode(
                    predicted_ids[0], skip_special_tokens=True
                ).strip().upper()
                
                reference_text = sample["transcription"].strip().upper()
                
                predictions.append({
                    "sample_id": i,
                    "reference": reference_text,
                    "prediction": predicted_text,
                    "pronunciation_scores": {
                        "accuracy": sample.get("accuracy_score", 0),
                        "fluency": sample.get("fluency_score", 0),
                        "completeness": sample.get("completeness_score", 0),
                        "prosodic": sample.get("prosodic_score", 0)
                    }
                })
        
        return predictions
    
    @staticmethod
    def load_trained_model(model_path: str) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
        """
        Load a trained model and processor.
        
        Args:
            model_path: Path to the trained model directory
            
        Returns:
            Tuple of (model, processor)
        """
        logger.info(f"Loading trained model from: {model_path}")
        
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        
        return model, processor


def main():
    """
    Main function for running Whisper fine-tuning.
    
    This function can be called directly or used as a script entry point.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on SpeechOcean762 dataset")
    parser.add_argument("--config", type=str, help="Path to training config JSON file")
    parser.add_argument("--model-name", type=str, default="openai/whisper-tiny", help="Whisper model name")
    parser.add_argument("--output-dir", type=str, default="whisper_finetuned", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max-train-samples", type=int, help="Maximum training samples")
    parser.add_argument("--max-eval-samples", type=int, help="Maximum evaluation samples")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with minimal data")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = TrainingConfig.load(args.config)
    elif args.quick_test:
        if __name__ == "__main__":
            from finetuning.training_config import get_quick_test_config
        else:
            from .training_config import get_quick_test_config
        config = get_quick_test_config()
    else:
        config = TrainingConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples
        )
    
    # Initialize fine-tuner
    fine_tuner = WhisperFineTuner(config)
    
    if args.eval_only:
        # Run evaluation only
        results = fine_tuner.evaluate_model()
        print(f"Evaluation Results: {json.dumps(results['eval_metrics'], indent=2)}")
    else:
        # Run full training
        results = fine_tuner.train()
        print(f"Training completed successfully!")
        print(f"Model saved to: {config.output_dir}")
        print(f"Final WER: {results['eval_results'].get('eval_wer', 'N/A')}")


if __name__ == "__main__":
    main()