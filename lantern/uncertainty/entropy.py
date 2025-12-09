"""
Entropy-based uncertainty estimation for LANTERN.

Computes token-level uncertainty metrics from logits/probabilities.
"""

import torch
import torch.nn.functional as F


def compute_entropy(
    logits: torch.Tensor, 
    temperature: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute entropy of a probability distribution.
    
    High entropy indicates the model is uncertain about which token to select
    (many tokens have similar probability).
    
    Args:
        logits: Raw logits tensor [..., vocab_size].
        temperature: Temperature for softmax scaling.
        dim: Dimension to compute entropy over.
        
    Returns:
        Entropy values with shape [...] (vocab dimension reduced).
    """
    # Convert to probabilities with temperature
    probs = F.softmax(logits / temperature, dim=dim)
    
    # Compute entropy: H = -Σ p_i log(p_i)
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    
    return entropy


def compute_p_max(
    logits: torch.Tensor,
    temperature: float = 1.0,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute maximum probability in a distribution.
    
    Low p_max indicates uncertainty (no single token dominates).
    
    Args:
        logits: Raw logits tensor [..., vocab_size].
        temperature: Temperature for softmax scaling.
        dim: Dimension to compute max over.
        
    Returns:
        Maximum probability values.
    """
    probs = F.softmax(logits / temperature, dim=dim)
    return probs.max(dim=dim).values


def compute_top_k_probs(
    logits: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k probabilities and their indices.
    
    Args:
        logits: Raw logits tensor [batch, vocab_size] or [vocab_size].
        k: Number of top candidates.
        temperature: Temperature for softmax scaling.
        
    Returns:
        Tuple of (top_k_probs, top_k_indices).
    """
    probs = F.softmax(logits / temperature, dim=-1)
    return probs.topk(k, dim=-1)


def compute_probability_gap(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute gap between top two probabilities.
    
    Large gap indicates high confidence in the top choice.
    Small gap indicates uncertainty between top candidates.
    
    Args:
        logits: Raw logits tensor [..., vocab_size].
        temperature: Temperature for softmax scaling.
        
    Returns:
        Probability gap values.
    """
    probs = F.softmax(logits / temperature, dim=-1)
    top2_probs, _ = probs.topk(2, dim=-1)
    
    # Gap between first and second
    return top2_probs[..., 0] - top2_probs[..., 1]


def normalize_entropy(
    entropy: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Normalize entropy to [0, 1] range.
    
    Maximum entropy for uniform distribution over vocab_size tokens
    is log(vocab_size).
    
    Args:
        entropy: Raw entropy values.
        vocab_size: Size of vocabulary.
        
    Returns:
        Normalized entropy in [0, 1].
    """
    import math
    max_entropy = math.log(vocab_size)
    return entropy / max_entropy
