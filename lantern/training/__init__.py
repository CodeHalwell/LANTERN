"""
Three-Phase Training Curriculum for LANTERN.

Phase 1: Backbone Pretraining
  - Train recursive transformer for next-token prediction
  - Randomly vary recursion depth (1..max_steps) per batch
  - Controller, ACT, and pause modules disabled

Phase 2: Probe Distillation
  - Freeze backbone, keep in eval mode
  - Selectively enable only nn.Dropout layers for MC sampling
  - Train EpistemicProbe via MSE distillation from MC Dropout variance

Phase 3: Controller Unlock
  - Unfreeze everything, enable ACT halting
  - Differentiable ponder cost with linear warmup
  - Differential learning rates: backbone 1e-5, reasoning heads 1e-3
  - bfloat16 mixed precision (no GradScaler needed)
  - Pad token dilution masking in ponder cost
"""

import random
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lantern.models.lantern_model import LANTERNModel


@contextmanager
def selective_dropout_train(model: nn.Module):
    """
    Enable only nn.Dropout layers for MC sampling while keeping
    everything else (especially LayerNorm) in eval mode.

    This preserves LayerNorm running statistics learned during Phase 1,
    which would be corrupted if the full model were put in train mode.
    """
    was_training = model.training
    model.eval()  # Everything to eval first

    # Selectively set only Dropout layers to train mode
    dropout_layers = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append((module, module.training))
            module.train()

    try:
        yield
    finally:
        # Restore dropout layers
        for module, state in dropout_layers:
            module.train(state)
        model.train(was_training)


def compute_ponder_cost_masked(
    ponder_cost: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute masked ponder cost, excluding padding tokens.

    Without masking, pad tokens that instantly halt at step 1 dilute
    the mean ponder cost, making the penalty on real tokens too weak.

    Args:
        ponder_cost: Raw ponder cost [batch, seq_len].
        attention_mask: Binary mask [batch, seq_len], 1 for real tokens.

    Returns:
        Scalar masked ponder cost.
    """
    if attention_mask is not None:
        masked_cost = (ponder_cost * attention_mask).sum()
        num_real = attention_mask.sum().clamp(min=1)
        return masked_cost / num_real
    return ponder_cost.mean()


class Phase1Trainer:
    """
    Phase 1: Backbone Pretraining.

    Trains the recursive transformer with standard cross-entropy loss.
    Randomly varies recursion depth between 1 and max_steps per batch
    to ensure all step embeddings learn useful representations.
    """

    def __init__(
        self,
        model: LANTERNModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.lr_lambda = (
            lambda step: min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.lr_lambda
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single Phase 1 training step with random depth."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Randomly vary depth each batch (Section 10, Phase 1)
        max_depth = self.model.config.max_steps
        random_depth = random.randint(1, max_depth)

        logits, _, _ = self.model(input_ids, steps_per_block=random_depth)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="mean",
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()


class Phase2Trainer:
    """
    Phase 2: Probe Distillation.

    Freeze the backbone and train the EpistemicProbe to predict
    MC Dropout uncertainty from a single forward pass.
    """

    def __init__(
        self,
        model: LANTERNModel,
        train_loader: DataLoader,
        num_mc_samples: int = 5,
        learning_rate: float = 1e-3,
        max_steps: int = 2000,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.max_steps = max_steps
        self.num_mc_samples = num_mc_samples

        # Freeze everything except the probe
        for param in model.parameters():
            param.requires_grad = False
        for param in model.epistemic_probe.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.AdamW(
            model.epistemic_probe.parameters(), lr=learning_rate,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single Phase 2 distillation step."""
        input_ids = batch["input_ids"].to(self.device)

        # Get hidden states from frozen backbone (eval mode)
        self.model.eval()
        with torch.no_grad():
            _, hidden_states, _ = self.model(
                input_ids, return_hidden_states=True,
            )

        # MC Dropout sampling with selective dropout
        mc_logits = []
        for _ in range(self.num_mc_samples):
            with selective_dropout_train(self.model):
                with torch.no_grad():
                    sample_logits, _, _ = self.model(input_ids)
                    mc_logits.append(F.softmax(sample_logits, dim=-1))

        # Stack: [num_samples, batch, seq_len, vocab_size]
        all_probs = torch.stack(mc_logits, dim=0)
        # Variance across samples, summed over vocab -> [batch, seq_len]
        mc_variance = all_probs.var(dim=0).sum(dim=-1)

        # Train probe to predict this variance
        probe_pred = self.model.epistemic_probe(hidden_states.detach())
        loss = F.mse_loss(probe_pred, mc_variance.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def cleanup(self):
        """Unfreeze backbone after Phase 2."""
        for param in self.model.parameters():
            param.requires_grad = True


class Phase3Trainer:
    """
    Phase 3: Controller Unlock.

    Enable ACT halting and teach the model to allocate compute efficiently.

    Hardening measures:
    1. Ponder shock warmup (λ ramps from 0 to target over warmup_steps)
    2. Differential learning rates (backbone low, reasoning heads high)
    3. Pad token dilution masking
    4. bfloat16 mixed precision
    """

    def __init__(
        self,
        model: LANTERNModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        backbone_lr: float = 1e-5,
        reasoning_lr: float = 1e-3,
        ponder_lambda: float = 0.01,
        ponder_warmup_steps: int = 500,
        weight_decay: float = 0.1,
        max_steps: int = 5000,
        grad_clip: float = 1.0,
        use_bfloat16: bool = True,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.ponder_lambda = ponder_lambda
        self.ponder_warmup_steps = ponder_warmup_steps
        self.use_bfloat16 = use_bfloat16 and device != "cpu"

        # Differential learning rates
        backbone_params = []
        reasoning_params = []
        for name, param in model.named_parameters():
            if any(
                key in name
                for key in [
                    "halting_head",
                    "pause_module",
                    "epistemic_probe",
                    "step_embeddings",
                ]
            ):
                reasoning_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": reasoning_params, "lr": reasoning_lr},
            ],
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.step = 0

    def _get_ponder_lambda(self) -> float:
        """Linear warmup for ponder cost λ."""
        if self.ponder_warmup_steps <= 0:
            return self.ponder_lambda
        return min(self.step / self.ponder_warmup_steps, 1.0) * self.ponder_lambda

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single Phase 3 training step with ACT."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Use bfloat16 mixed precision
        amp_dtype = torch.bfloat16 if self.use_bfloat16 else None
        ctx = (
            torch.autocast(device_type=self.device, dtype=amp_dtype)
            if amp_dtype is not None
            else _null_context()
        )

        with ctx:
            logits, _, ponder_cost = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_adaptive_halting=True,
            )

            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="mean",
            )

            # Ponder cost with warmup and pad dilution fix
            current_lambda = self._get_ponder_lambda()
            if ponder_cost is not None and current_lambda > 0:
                masked_ponder = compute_ponder_cost_masked(
                    ponder_cost, attention_mask
                )
                total_loss = ce_loss + current_lambda * masked_ponder
            else:
                masked_ponder = torch.tensor(0.0, device=self.device)
                total_loss = ce_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.step += 1

        return {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "ponder_cost": masked_ponder.item(),
            "ponder_lambda": current_lambda,
        }


class _null_context:
    """No-op context manager for when autocast is not needed."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
