"""
Command-line interface for pronunciation assessment model training.

Supports multiple training modes for different objectives:
- quick-test: Fast training for development (100 samples, 1 epoch)
- development: Moderate training (1000 samples, 3 epochs)
- production: Full training with all data
- transcription-only: Train only ASR task
- assessment-only: Train only pronunciation assessment
- phone-focused: Focus on phone-level assessment

Usage:
    python run_pronunciation_training.py --mode quick-test --output-dir ./models/test
    python run_pronunciation_training.py --mode production --resume-from checkpoint
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from training_config import get_config, TRAINING_MODES
from data_processor import SpeechOcean762DataProcessor
from whisper_pronunciation_model import WhisperPronunciationAssessmentModel
from trainer import PronunciationAssessmentTrainer
from data_collator import PronunciationAssessmentDataCollator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train pronunciation assessment model with Whisper"
    )
    
    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="quick-test",
        choices=list(TRAINING_MODES.keys()),
        help=f"Training mode: {', '.join(TRAINING_MODES.keys())}"
    )
    
    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/pronunciation_assessment",
        help="Directory to save model checkpoints and logs"
    )
    
    # Resume from checkpoint
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset-splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to use (e.g., train test)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing/debugging)"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )
    
    # Model arguments
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze Whisper encoder weights"
    )
    
    parser.add_argument(
        "--use-small-model",
        action="store_true",
        help="Use whisper-small instead of whisper-base"
    )
    
    # Loss weight arguments
    parser.add_argument(
        "--loss-weights",
        type=str,
        default=None,
        help="JSON string with loss weights (e.g., '{\"asr\": 0.3, \"word_accuracy\": 0.7}')"
    )
    
    # Execution arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-steps",
        type=int,
        default=None,
        help="Override logging steps from config"
    )
    
    return parser.parse_args()


def setup_device(device_name: str) -> torch.device:
    """Setup and verify device."""
    device = torch.device(device_name)
    
    if device.type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("Using CPU for training")
    
    return device


def main():
    """Main training entry point."""
    args = parse_arguments()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load config
    config = get_config(args.mode)
    
    # Override config with command-line arguments
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.log_steps is not None:
        config.logging_steps = args.log_steps
    if args.use_small_model:
        config.whisper_model_name = "openai/whisper-small"
    
    # Override loss weights if provided
    if args.loss_weights is not None:
        try:
            loss_weights_dict = json.loads(args.loss_weights)
            config.loss_weights.update(loss_weights_dict)
            logger.info(f"Using custom loss weights: {config.loss_weights}")
        except json.JSONDecodeError:
            logger.error(f"Invalid loss weights JSON: {args.loss_weights}")
            return
    
    logger.info(f"Config: {config}")
    
    # Load and preprocess dataset
    logger.info("Loading dataset...")
    data_processor = SpeechOcean762DataProcessor()
    
    datasets = data_processor.load_dataset(
        splits=args.dataset_splits,
        max_samples=args.max_samples or config.max_train_samples
    )
    logger.info(f"Loaded dataset: {datasets}")
    
    logger.info("Preprocessing dataset...")
    processed_datasets = data_processor.prepare_for_training(
        datasets,
        batch_size=config.batch_size,
        include_transcription=config.include_transcription
    )
    logger.info("Dataset preprocessing complete")
    
    # Create data loaders
    data_collator = PronunciationAssessmentDataCollator()
    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2  # Important for Kaggle
    )
    
    val_loader = None
    if "test" in processed_datasets:
        val_loader = DataLoader(
            processed_datasets["test"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=2
        )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    if val_loader:
        logger.info(f"Validation loader: {len(val_loader)} batches")
    
    # Create model
    logger.info(f"Creating model: {config.whisper_model_name}")
    model = WhisperPronunciationAssessmentModel(
        model_name=config.whisper_model_name,
        freeze_encoder=args.freeze_encoder,
        train_word_level=config.train_word_level,
        train_phone_level=config.train_phone_level,
        train_utterance_level=config.train_utterance_level
    )
    model = model.to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = PronunciationAssessmentTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir
    )
    
    # Setup optimization (optimizer + scheduler)
    total_steps = len(train_loader) * config.num_epochs
    trainer.setup_optimization(total_steps)
    
    # Load checkpoint if provided
    if args.resume_from:
        logger.info(f"Loading checkpoint from {args.resume_from}")
        trainer.load_model(args.resume_from)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        # Convert config to dict for JSON serialization
        config_dict = {
            k: v for k, v in vars(config).items()
            if not k.startswith("_")
        }
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Config saved to {config_path}")
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate if validation set exists
        if val_loader is not None:
            logger.info("Running validation...")
            val_loss = trainer.evaluate(val_loader)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = output_dir / "best_model"
                trainer.save_model(str(best_checkpoint_path))
                logger.info(f"Best model saved to {best_checkpoint_path}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
            trainer.save_model(str(checkpoint_path))
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    logger.info(f"\nTraining complete! Final model saved to {final_model_path}")
    
    # Save summary
    summary = {
        "mode": args.mode,
        "timestamp": datetime.now().isoformat(),
        "model": config.whisper_model_name,
        "epochs_trained": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "loss_weights": config.loss_weights,
        "best_validation_loss": best_val_loss if val_loader else None,
    }
    
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
