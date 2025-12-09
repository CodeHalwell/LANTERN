"""
Semantic Dispersion Uncertainty for LANTERN.

Measures whether high-entropy predictions are due to synonyms/paraphrases
(similar embeddings) or genuinely different meanings (dispersed embeddings).
"""

import torch
import torch.nn.functional as F


def compute_semantic_dispersion(
    logits: torch.Tensor,
    embedding_matrix: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute semantic dispersion of top-k candidate tokens.
    
    High entropy + low dispersion = synonyms/paraphrases (same meaning)
    High entropy + high dispersion = genuinely different meanings
    
    Args:
        logits: Raw logits [batch, vocab_size] or [vocab_size].
        embedding_matrix: Token embeddings [vocab_size, embed_dim].
        k: Number of top candidates to consider.
        temperature: Temperature for softmax scaling.
        
    Returns:
        Weighted variance (dispersion) of top-k embeddings.
    """
    # Handle both batched and unbatched inputs
    is_batched = logits.dim() == 2
    if not is_batched:
        logits = logits.unsqueeze(0)
    
    batch_size = logits.shape[0]
    
    # Get probabilities and top-k
    probs = F.softmax(logits / temperature, dim=-1)
    topk_probs, topk_idx = probs.topk(k, dim=-1)  # [batch, k]
    
    # Get embeddings for top-k tokens
    # [batch, k, embed_dim]
    topk_embeddings = embedding_matrix[topk_idx]
    
    # Normalize probabilities for weighted computation
    topk_probs_norm = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    
    # Weighted centroid: μ = Σ p_i * e_i
    # [batch, k, 1] * [batch, k, embed_dim] -> [batch, embed_dim]
    centroid = torch.sum(
        topk_probs_norm.unsqueeze(-1) * topk_embeddings,
        dim=1
    )
    
    # Weighted variance: σ² = Σ p_i * ||e_i - μ||²
    # [batch, k, embed_dim]
    diff = topk_embeddings - centroid.unsqueeze(1)
    squared_dist = (diff ** 2).sum(dim=-1)  # [batch, k]
    
    variance = torch.sum(topk_probs_norm * squared_dist, dim=-1)  # [batch]
    
    if not is_batched:
        variance = variance.squeeze(0)
    
    return variance


def compute_semantic_coherence(
    logits: torch.Tensor,
    embedding_matrix: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute semantic coherence (inverse of dispersion).
    
    High coherence means top-k tokens are semantically similar.
    
    Args:
        logits: Raw logits [batch, vocab_size] or [vocab_size].
        embedding_matrix: Token embeddings [vocab_size, embed_dim].
        k: Number of top candidates.
        temperature: Temperature for softmax scaling.
        
    Returns:
        Coherence score (higher = more similar meanings).
    """
    dispersion = compute_semantic_dispersion(
        logits, embedding_matrix, k, temperature
    )
    # Convert dispersion to coherence using exponential decay
    return torch.exp(-dispersion)


def compute_pairwise_similarity(
    logits: torch.Tensor,
    embedding_matrix: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute average pairwise cosine similarity of top-k embeddings.
    
    Alternative dispersion metric using cosine similarity.
    
    Args:
        logits: Raw logits [batch, vocab_size] or [vocab_size].
        embedding_matrix: Token embeddings [vocab_size, embed_dim].
        k: Number of top candidates.
        temperature: Temperature for softmax scaling.
        
    Returns:
        Average pairwise cosine similarity.
    """
    is_batched = logits.dim() == 2
    if not is_batched:
        logits = logits.unsqueeze(0)
    
    # Get top-k indices
    _, topk_idx = F.softmax(logits / temperature, dim=-1).topk(k, dim=-1)
    
    # Get embeddings [batch, k, embed_dim]
    topk_embeddings = embedding_matrix[topk_idx]
    
    # Normalize embeddings for cosine similarity
    normalized = F.normalize(topk_embeddings, p=2, dim=-1)
    
    # Compute pairwise similarity: [batch, k, k]
    similarity_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
    
    # Average off-diagonal elements (exclude self-similarity)
    # Create mask for off-diagonal
    mask = ~torch.eye(k, dtype=torch.bool, device=logits.device)
    mask = mask.unsqueeze(0).expand(similarity_matrix.shape[0], -1, -1)
    
    # Average pairwise similarity
    avg_similarity = similarity_matrix[mask].view(-1, k * (k - 1)).mean(dim=-1)
    
    if not is_batched:
        avg_similarity = avg_similarity.squeeze(0)
    
    return avg_similarity


def interpret_uncertainty(
    entropy: torch.Tensor,
    dispersion: torch.Tensor,
    entropy_threshold: float = 1.5,
    dispersion_threshold: float = 0.5,
) -> str:
    """
    Interpret uncertainty based on entropy and dispersion.
    
    Args:
        entropy: Entropy value.
        dispersion: Semantic dispersion value.
        entropy_threshold: Threshold for high entropy.
        dispersion_threshold: Threshold for high dispersion.
        
    Returns:
        Interpretation string.
    """
    high_entropy = entropy > entropy_threshold
    high_dispersion = dispersion > dispersion_threshold
    
    if not high_entropy:
        return "confident"
    elif high_entropy and not high_dispersion:
        return "synonyms"  # Multiple valid paraphrases
    else:
        return "uncertain"  # Genuinely different meanings
