"""
Training script for LANTERN on text generation tasks.

This script trains the LANTERN model on NLP datasets for autoregressive
language modeling, similar to GPT-style models.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import LANTERNConfig, create_small_config, create_base_config


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Loads text data and creates training sequences of a fixed length.
    """
    
    def __init__(
        self,
        data_path: str,
        seq_length: int = 512,
        vocab_size: int = 32000,
    ):
        """
        Initialize text dataset.
        
        Args:
            data_path: Path to text file or tokenized data.
            seq_length: Sequence length for training.
            vocab_size: Size of vocabulary.
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Load data
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simple character-level tokenization for demonstration
            # In practice, use a proper tokenizer like BPE or SentencePiece
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
            
            # Convert text to token IDs
            self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
            print(f"Loaded {len(text)} characters, {len(chars)} unique.")
        else:
            # Generate synthetic data for testing
            print(f"Data file not found at {data_path}. Using synthetic data.")
            self.data = torch.randint(0, vocab_size, (100000,), dtype=torch.long)
            self.char_to_idx = None
            self.idx_to_char = None
    
    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return max(1, len(self.data) - self.seq_length - 1)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example.
        
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors.
        """
        # Get sequence
        chunk = self.data[idx:idx + self.seq_length + 1]
        
        # Input is all tokens except last
        input_ids = chunk[:-1]
        
        # Labels is all tokens except first (next token prediction)
        labels = chunk[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class Trainer:
    """
    Trainer for LANTERN model.
    
    Handles training loop, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        model: LANTERNModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "./outputs",
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 10000,
        eval_interval: int = 500,
        save_interval: int = 1000,
        warmup_steps: int = 100,
        grad_clip: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize trainer.
        
        Args:
            model: LANTERN model to train.
            train_dataset: Training dataset.
            val_dataset: Optional validation dataset.
            output_dir: Directory to save checkpoints and logs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            weight_decay: Weight decay for AdamW.
            max_steps: Maximum training steps.
            eval_interval: Steps between evaluations.
            save_interval: Steps between checkpoint saves.
            warmup_steps: Learning rate warmup steps.
            grad_clip: Gradient clipping threshold.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.device = device
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        else:
            self.val_loader = None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler with warmup
        self.lr_lambda = lambda step: min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_file = self.output_dir / "training_log.jsonl"
    
    def save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.model.config.__dict__,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.step}, epoch {self.epoch}")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            batch: Batch with 'input_ids' and 'labels'.
            
        Returns:
            Loss tensor.
        """
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        logits, _ = self.model(input_ids)
        
        # Compute cross-entropy loss
        # Reshape logits and labels for loss computation
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='mean',
        )
        
        return loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            Average validation loss.
        """
        if self.val_loader is None:
            return float('nan')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
        self.model.train()
        
        return avg_loss
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def train(self):
        """Main training loop."""
        print("=" * 60)
        print(f"Starting training for {self.max_steps} steps")
        print(f"Model has {self.model.get_num_params():,} parameters")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        self.model.train()
        train_iter = iter(self.train_loader)
        
        start_time = time.time()
        
        while self.step < self.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Forward and backward
            loss = self.compute_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            self.step += 1
            
            # Logging
            if self.step % 10 == 0:
                elapsed = time.time() - start_time
                lr = self.scheduler.get_last_lr()[0]
                
                print(f"Step {self.step}/{self.max_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Time: {elapsed:.1f}s")
                
                self.log_metrics({
                    'step': self.step,
                    'epoch': self.epoch,
                    'train_loss': loss.item(),
                    'learning_rate': lr,
                    'elapsed_time': elapsed,
                })
            
            # Evaluation
            if self.step % self.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"Validation loss: {val_loss:.4f}")
                
                self.log_metrics({
                    'step': self.step,
                    'val_loss': val_loss,
                })
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            
            # Save checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
        
        # Final checkpoint
        self.save_checkpoint("final_model.pt")
        
        print("=" * 60)
        print("Training completed!")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train LANTERN on text generation")
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data.txt",
        help="Path to training data text file",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="Path to validation data (optional)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512,
        help="Sequence length for training",
    )
    
    # Model arguments
    parser.add_argument(
        "--config",
        type=str,
        default="small",
        choices=["small", "base"],
        help="Model configuration preset",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size (overrides config preset)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of attention heads (overrides config preset)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="Number of transformer blocks (overrides config preset)",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Learning rate warmup steps",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Create config
    if args.config == "small":
        config = create_small_config()
    else:
        config = create_base_config()
    
    # Override config with command-line arguments
    config.vocab_size = args.vocab_size
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
    if args.num_heads is not None:
        config.num_heads = args.num_heads
    if args.num_blocks is not None:
        config.num_blocks = args.num_blocks
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = TextDataset(
        data_path=args.data_path,
        seq_length=args.seq_length,
        vocab_size=config.vocab_size,
    )
    
    val_dataset = None
    if args.val_data_path is not None:
        val_dataset = TextDataset(
            data_path=args.val_data_path,
            seq_length=args.seq_length,
            vocab_size=config.vocab_size,
        )
    
    # Create model
    print("Creating model...")
    model = LANTERNModel(config)
    
    print(f"Model configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Number of heads: {config.num_heads}")
    print(f"  Number of blocks: {config.num_blocks}")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Window size: {config.window_size}")
    print(f"  Recursion steps: {config.steps_base}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from is not None:
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
