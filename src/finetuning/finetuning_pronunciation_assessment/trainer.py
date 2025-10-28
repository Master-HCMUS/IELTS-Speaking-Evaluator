"""
Trainer for pronunciation assessment model with multi-objective learning.

Handles training loop, loss computation, and evaluation.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from tqdm import tqdm

logger = logging.getLogger(__name__)


class PronunciationAssessmentTrainer:
    """Trainer for pronunciation assessment model with full training/evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        config,  # PronunciationTrainingConfig
        device: torch.device = None,
        output_dir: str = "outputs"
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            device: Device to train on (cuda/cpu)
            output_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup optimizer and scheduler will be done in setup_optimization
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_optimization(self, total_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Ensure model is initialized before creating optimizer
        if hasattr(self.model, '_initialize_model'):
            self.model._initialize_model()
        
        # Move model to device
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Create linear scheduler with warmup (no transformers import needed)
        num_warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        def lr_lambda(current_step: int):
            """Linear warmup + linear decay scheduler."""
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - num_warmup_steps)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        logger.info(f"Scheduler: Linear with warmup ({num_warmup_steps} warmup steps)")
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-objective loss including transcription.
        
        Args:
            predictions: Model predictions dict
            batch: Batch data dict
            
        Returns:
            (total_loss, component_losses_dict)
        """
        losses = {}
        weights = self.config.loss_weights
        
        # Transcription loss (using cross-entropy on decoder logits)
        if self.config.use_transcription and "transcription_logits" in predictions:
            if "labels" in batch and batch["labels"] is not None:
                # Reshape logits and labels for cross-entropy
                # logits: [batch, seq_len, vocab_size]
                # labels: [batch, seq_len]
                transcription_logits = predictions["transcription_logits"]
                labels = batch["labels"]
                
                # Shift logits and labels for causal language modeling
                # Logits: [batch, seq_len-1, vocab_size], Labels: [batch, seq_len-1]
                shift_logits = transcription_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten for cross-entropy
                batch_size, seq_len, vocab_size = shift_logits.shape
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                transcription_loss = loss_fct(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1)
                )
                
                weight = weights.get("transcription", 1.0)
                losses["transcription"] = transcription_loss * weight
        
        # Utterance-level losses (fixed, one per example)
        utterance_targets = {
            "accuracy": ("utterance_accuracy_logits", "utterance_accuracy"),
            "fluency": ("utterance_fluency_logits", "utterance_fluency"),
            "prosodic": ("utterance_prosodic_logits", "utterance_prosodic"),
            "completeness": ("utterance_completeness_logits", "utterance_completeness"),
            "total": ("utterance_total_logits", "utterance_total"),
        }
        
        if self.config.use_utterance_level_assessment:
            for batch_key, (pred_key, weight_key) in utterance_targets.items():
                if batch_key in batch and batch[batch_key] is not None:
                    if pred_key in predictions:
                        loss_val = nn.MSELoss()(
                            predictions[pred_key],
                            batch[batch_key]
                        )
                        weight = weights.get(weight_key, 1.0)
                        losses[weight_key] = loss_val * weight
        
        # Word-level losses (variable length per example)
        word_targets = {
            "word_accuracy_scores": ("word_accuracy_logits", "word_accuracy"),
            "word_stress_scores": ("word_stress_logits", "word_stress"),
            "word_total_scores": ("word_total_logits", "word_total"),
        }
        
        if self.config.use_word_level_assessment:
            for batch_key, (pred_key, weight_key) in word_targets.items():
                if batch_key in batch and batch[batch_key] is not None:
                    if pred_key in predictions:
                        # Handle variable-length word scores
                        batch_scores = batch[batch_key]
                        pred_scores = predictions[pred_key]
                        
                        # Compute loss for each example and average
                        example_losses = []
                        for i, (pred, target) in enumerate(zip(pred_scores, batch_scores)):
                            # Truncate to common length
                            min_len = min(pred.shape[0], target.shape[0])
                            pred = pred[:min_len]
                            target = target[:min_len]
                            example_losses.append(nn.MSELoss()(pred, target))
                        
                        if example_losses:
                            loss_val = torch.stack(example_losses).mean()
                            weight = weights.get(weight_key, 1.0)
                            losses[weight_key] = loss_val * weight
        
        # Phone-level loss (variable length per example)
        if self.config.use_phone_level_assessment:
            if "phone_accuracy_scores" in batch and batch["phone_accuracy_scores"] is not None:
                if "phone_accuracy_logits" in predictions:
                    batch_scores = batch["phone_accuracy_scores"]
                    pred_scores = predictions["phone_accuracy_logits"]
                    
                    example_losses = []
                    for i, (pred, target) in enumerate(zip(pred_scores, batch_scores)):
                        min_len = min(pred.shape[0], target.shape[0])
                        pred = pred[:min_len]
                        target = target[:min_len]
                        example_losses.append(nn.MSELoss()(pred, target))
                    
                    if example_losses:
                        loss_val = torch.stack(example_losses).mean()
                        weight = weights.get("phone_accuracy", 1.0)
                        losses["phone_accuracy"] = loss_val * weight
        
        # Total loss
        if losses:
            total_loss = sum(losses.values())
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        return total_loss, losses
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            input_features = batch["input_features"]
            # Pass decoder_input_ids for transcription training if available
            decoder_input_ids = batch.get("decoder_input_ids", None)
            predictions = self.model(
                input_features,
                decoder_input_ids=decoder_input_ids
            )
            
            # Compute loss
            loss, losses_dict = self.compute_loss(predictions, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.logging_steps == 0:
                current_loss = loss.item()
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Evaluating")
            for batch in progress_bar:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                
                # Forward pass
                input_features = batch["input_features"]
                # Pass decoder_input_ids for transcription if available
                decoder_input_ids = batch.get("decoder_input_ids", None)
                predictions = self.model(
                    input_features,
                    decoder_input_ids=decoder_input_ids
                )
                
                # Compute loss
                loss, _ = self.compute_loss(predictions, batch)
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_model(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        path = Path(path)
        checkpoint_path = path / "model.pt"
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
