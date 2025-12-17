"""
Bayesian Sampling for LANTERN.

Implements MC dropout for epistemic uncertainty estimation.
Multiple forward passes with dropout enabled to measure model disagreement.
"""

from typing import Callable, Optional, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def dropout_enabled(model: nn.Module):
    """
    Context manager to enable dropout at inference time.
    
    Temporarily sets the model to training mode to enable dropout,
    then restores the original mode.
    
    Args:
        model: PyTorch model with dropout layers.
    """
    # Store original training states of all modules
    original_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_states[name] = module.training
            module.train()
    
    try:
        yield
    finally:
        # Restore original states
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout) and name in original_states:
                if not original_states[name]:
                    module.eval()


class BayesianSampler:
    """
    Bayesian sampler using MC dropout for epistemic uncertainty.
    
    Performs multiple stochastic forward passes with dropout enabled
    to estimate model uncertainty through prediction variance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        forward_fn: Optional[Callable] = None,
        num_samples: int = 5,
    ):
        """
        Initialize Bayesian sampler.
        
        Args:
            model: The neural network model (must have dropout layers).
            forward_fn: Optional custom forward function. If None, uses model().
            num_samples: Default number of MC samples.
        """
        self.model = model
        self.forward_fn = forward_fn if forward_fn is not None else model
        self.num_samples = num_samples
    
    def sample(
        self,
        inputs: torch.Tensor,
        num_samples: Optional[int] = None,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform MC dropout sampling.
        
        Args:
            inputs: Input tensor to the model.
            num_samples: Number of MC samples (overrides default).
            **forward_kwargs: Additional arguments for forward pass.
            
        Returns:
            Tuple of (mean_probs, variance, all_logits):
            - mean_probs: Mean probability distribution [vocab_size]
            - variance: Variance across samples per token [vocab_size]
            - all_logits: All sampled logits [num_samples, vocab_size]
        """
        n = num_samples if num_samples is not None else self.num_samples
        logits_list = []
        
        for _ in range(n):
            with dropout_enabled(self.model):
                logits = self.forward_fn(inputs, **forward_kwargs)
                logits_list.append(logits)
        
        # Stack: [num_samples, ..., vocab_size]
        all_logits = torch.stack(logits_list, dim=0)
        all_probs = F.softmax(all_logits, dim=-1)
        
        # Compute mean and variance
        mean_probs = all_probs.mean(dim=0)
        variance = all_probs.var(dim=0)
        
        return mean_probs, variance, all_logits
    
    def epistemic_uncertainty(
        self,
        inputs: torch.Tensor,
        num_samples: Optional[int] = None,
        reduction: str = "sum",
        **forward_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute epistemic uncertainty via MC dropout.
        
        Args:
            inputs: Input tensor to the model.
            num_samples: Number of MC samples.
            reduction: How to reduce variance ("sum", "mean", "max").
            **forward_kwargs: Additional arguments for forward pass.
            
        Returns:
            Tuple of (mean_probs, epistemic_uncertainty_score).
        """
        mean_probs, variance, _ = self.sample(inputs, num_samples, **forward_kwargs)
        
        # Reduce variance to scalar
        if reduction == "sum":
            uncertainty = variance.sum()
        elif reduction == "mean":
            uncertainty = variance.mean()
        elif reduction == "max":
            uncertainty = variance.max()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        return mean_probs, uncertainty


def bayesian_step(
    model: nn.Module,
    hidden_states: torch.Tensor,
    lm_head: nn.Module,
    recur_fn: Optional[Callable] = None,
    num_samples: int = 5,
    steps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a Bayesian inference step with MC dropout.
    
    This is the core function for uncertainty-aware generation.
    When uncertainty is high, multiple forward passes are performed
    to assess epistemic uncertainty.
    
    Args:
        model: The transformer model (with dropout).
        hidden_states: Current context hidden states.
        lm_head: Language model head for logit prediction.
        recur_fn: Optional recursive forward function.
        num_samples: Number of MC samples.
        steps: Recursion depth for each sample.
        
    Returns:
        Tuple of (mean_probs, epistemic_uncertainty).
    """
    logits_list = []
    
    for _ in range(num_samples):
        with dropout_enabled(model):
            if recur_fn is not None:
                h, _ = recur_fn(hidden_states, steps_max=steps)
            else:
                h = hidden_states
            
            # Get logits for last token
            logits = lm_head(h[:, -1, :])  # [batch, vocab_size]
            logits_list.append(logits)
    
    # Stack: [num_samples, batch, vocab_size]
    all_logits = torch.stack(logits_list, dim=0)
    all_probs = F.softmax(all_logits, dim=-1)
    
    # Mean probability and variance
    mean_probs = all_probs.mean(dim=0)  # [batch, vocab_size]
    variance = all_probs.var(dim=0)  # [batch, vocab_size]
    
    # Epistemic uncertainty as sum of variance
    epistemic_uncertainty = variance.sum(dim=-1)  # [batch]
    
    return mean_probs, epistemic_uncertainty


def compute_predictive_entropy(
    all_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute predictive entropy and mutual information.
    
    Decomposes total uncertainty into:
    - Aleatoric: inherent randomness (entropy of mean)
    - Epistemic: model uncertainty (mutual information)
    
    Args:
        all_probs: Probability distributions [num_samples, ..., vocab_size].
        
    Returns:
        Tuple of (predictive_entropy, mutual_information).
    """
    # Mean probability
    mean_probs = all_probs.mean(dim=0)
    
    # Predictive entropy: H[E[p(y|x)]]
    predictive_entropy = -torch.sum(
        mean_probs * torch.log(mean_probs + 1e-10),
        dim=-1
    )
    
    # Expected entropy: E[H[p(y|x)]]
    individual_entropies = -torch.sum(
        all_probs * torch.log(all_probs + 1e-10),
        dim=-1
    ).mean(dim=0)
    
    # Mutual information (epistemic): H[E[p]] - E[H[p]]
    mutual_information = predictive_entropy - individual_entropies
    
    return predictive_entropy, mutual_information
