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

import math
import random
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Sequence, Union

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


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to ``min_lr_ratio`` of the peak."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _autocast(device: str, enabled: bool):
    """bf16 autocast on CUDA when enabled; a no-op context otherwise."""
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


Batch = Dict[str, torch.Tensor]


def _micro_batches(batch: Union[Batch, Sequence[Batch]]) -> List[Batch]:
    """
    A trainer's ``train_step`` takes either one loader batch or a list of
    them. A list means gradient accumulation: every batch contributes to the
    same optimizer step, so the effective batch is ``len(list) * batch_size``.
    """
    if isinstance(batch, dict):
        return [batch]
    batches = list(batch)
    if not batches:
        raise ValueError("train_step needs at least one batch")
    return batches


def binary_to_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Turn a binary padding mask [batch, seq_len] (1 = real token) into the
    additive key mask the attention layers expect: 0 for real keys, -inf for
    padding keys, shaped [batch, 1, 1, seq_len] so it broadcasts over heads
    and query positions.
    """
    additive = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
    additive = additive.masked_fill(attention_mask == 0, float("-inf"))
    return additive[:, None, None, :]


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
        min_depth: int = 1,
        max_depth: Optional[int] = None,
        use_bfloat16: bool = True,
        min_lr_ratio: float = 0.1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.min_depth = min_depth
        self.max_depth = max_depth if max_depth is not None else model.config.max_steps
        self.use_bfloat16 = use_bfloat16

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer, warmup_steps, max_steps, min_lr_ratio
        )

    def train_step(self, batch: Union[Batch, Sequence[Batch]]) -> float:
        """
        Single Phase 1 optimizer step with a random recursion depth.

        ``batch`` is one loader batch, or a list of them to accumulate
        gradients over (see ``_micro_batches``).
        """
        # One depth per optimizer step so every step embedding gets trained.
        random_depth = random.randint(self.min_depth, self.max_depth)
        micro_batches = _micro_batches(batch)

        self.optimizer.zero_grad()
        total = 0.0
        for micro in micro_batches:
            input_ids = micro["input_ids"].to(self.device)
            labels = micro["labels"].to(self.device)
            with _autocast(self.device, self.use_bfloat16):
                logits, _, _ = self.model(input_ids, steps_per_block=random_depth)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    reduction="mean",
                )
            (loss / len(micro_batches)).backward()
            total += loss.item() / len(micro_batches)

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return total

    @torch.no_grad()
    def evaluate(self, depth: Optional[int] = None, max_batches: Optional[int] = None) -> float:
        """Mean validation loss at a fixed depth (config.steps_base if None)."""
        if self.val_loader is None:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            with _autocast(self.device, self.use_bfloat16):
                logits, _, _ = self.model(input_ids, steps_per_block=depth)
            total += F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), labels.view(-1)
            ).item()
            n += 1
        self.model.train(was_training)
        return total / max(1, n)


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

    def _distillation_loss(self, batch: Batch) -> torch.Tensor:
        """MSE between the probe and MC-dropout variance for one loader batch."""
        input_ids = batch["input_ids"].to(self.device)

        # Hidden states from the frozen backbone (eval mode)
        self.model.eval()
        with torch.no_grad():
            _, hidden_states, _ = self.model(input_ids, return_hidden_states=True)

        # MC Dropout sampling with only the dropout layers in train mode
        mc_probs = []
        for _ in range(self.num_mc_samples):
            with selective_dropout_train(self.model):
                with torch.no_grad():
                    sample_logits, _, _ = self.model(input_ids)
                    mc_probs.append(F.softmax(sample_logits, dim=-1))

        # [num_samples, batch, seq_len, vocab] -> variance summed over vocab
        # -> [batch, seq_len]. Bounded in [0, 1), matching the probe's sigmoid.
        mc_variance = torch.stack(mc_probs, dim=0).var(dim=0).sum(dim=-1)

        probe_pred = self.model.epistemic_probe(hidden_states.detach())
        return F.mse_loss(probe_pred, mc_variance.detach())

    def train_step(self, batch: Union[Batch, Sequence[Batch]]) -> float:
        """
        Single Phase 2 distillation step.

        ``batch`` is one loader batch, or a list of them to accumulate
        gradients over.
        """
        micro_batches = _micro_batches(batch)
        self.optimizer.zero_grad()
        total = 0.0
        for micro in micro_batches:
            loss = self._distillation_loss(micro)
            (loss / len(micro_batches)).backward()
            total += loss.item() / len(micro_batches)
        self.optimizer.step()
        return total

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
        warmup_steps: int = 0,
        pause_prob: float = 0.5,
        min_lr_ratio: float = 0.1,
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
        # Fraction of optimizer steps that apply 1..max_pause_steps latent
        # pause cycles to the whole sequence, so the pause module is trained
        # exactly the way cached decoding uses it.
        self.pause_prob = pause_prob

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
        self.scheduler = build_lr_scheduler(
            self.optimizer, warmup_steps, max_steps, min_lr_ratio
        )

        self.step = 0

    def _get_ponder_lambda(self) -> float:
        """Linear warmup for ponder cost λ."""
        if self.ponder_warmup_steps <= 0:
            return self.ponder_lambda
        return min(self.step / self.ponder_warmup_steps, 1.0) * self.ponder_lambda

    def _sample_pause_steps(self) -> int:
        if random.random() < self.pause_prob:
            return random.randint(1, self.model.config.max_pause_steps)
        return 0

    def train_step(self, batch: Union[Batch, Sequence[Batch]]) -> Dict[str, float]:
        """
        Single Phase 3 optimizer step with ACT and random latent pause.

        ``batch`` is one loader batch, or a list of them to accumulate
        gradients over.
        """
        pause_steps = self._sample_pause_steps()
        current_lambda = self._get_ponder_lambda()
        micro_batches = _micro_batches(batch)
        n_micro = len(micro_batches)

        self.optimizer.zero_grad()
        acc = {"total_loss": 0.0, "ce_loss": 0.0, "ponder_cost": 0.0}
        for micro in micro_batches:
            input_ids = micro["input_ids"].to(self.device)
            labels = micro["labels"].to(self.device)
            attention_mask = micro.get("attention_mask")
            additive_mask = None
            if attention_mask is not None:
                # Binary mask for the ponder accounting; additive for attention.
                attention_mask = attention_mask.to(self.device)
                additive_mask = binary_to_additive_mask(attention_mask)

            with _autocast(self.device, self.use_bfloat16):
                logits, _, ponder_cost = self.model(
                    input_ids,
                    attention_mask=additive_mask,
                    use_adaptive_halting=True,
                    pause_steps=pause_steps,
                )
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    labels.view(-1),
                    reduction="mean",
                )
                if ponder_cost is not None and current_lambda > 0:
                    masked_ponder = compute_ponder_cost_masked(
                        ponder_cost.float(), attention_mask
                    )
                    total_loss = ce_loss + current_lambda * masked_ponder
                else:
                    masked_ponder = torch.tensor(0.0, device=self.device)
                    total_loss = ce_loss

            (total_loss / n_micro).backward()
            acc["total_loss"] += total_loss.item() / n_micro
            acc["ce_loss"] += ce_loss.item() / n_micro
            acc["ponder_cost"] += masked_ponder.item() / n_micro

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

        acc["ponder_lambda"] = current_lambda
        acc["pause_steps"] = pause_steps
        return acc
