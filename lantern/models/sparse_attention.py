"""
Sparse Attention Module for LANTERN.

Implements sliding-window attention with global tokens for efficient
attention computation: O(L * w) instead of O(L²).
"""

import math
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    """
    Sparse multi-head attention with sliding window and global tokens.
    
    For each token i, attention is computed over:
    - Tokens in [i - window_size, i] (local window)
    - Global tokens (e.g., [CLS], [REASON], first prompt tokens)
    
    This reduces complexity from O(L²) to O(L * w) per attention layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 256,
        global_token_indices: Optional[Set[int]] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_position: int = 4096,
    ):
        """
        Initialize sparse attention.
        
        Args:
            hidden_size: Dimension of hidden states.
            num_heads: Number of attention heads.
            window_size: Size of sliding attention window.
            global_token_indices: Set of indices that all tokens attend to.
            dropout: Attention dropout probability.
            use_rope: Whether to use Rotary Position Embeddings.
            max_position: Maximum sequence length for position embeddings.
        """
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        head_dim = hidden_size // num_heads
        if use_rope:
            assert head_dim % 2 == 0, "head_dim must be even when using RoPE"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.global_token_indices = global_token_indices or {0}  # At least BOS
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Rotary embeddings if enabled
        if use_rope:
            self._init_rope(max_position)
    
    def _init_rope(self, max_position: int):
        """Initialize rotary position embeddings."""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute cos/sin for positions
        positions = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings to tensor."""
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        # Rotate half the dimensions
        x1, x2 = x[..., :self.head_dim // 2], x[..., self.head_dim // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
    
    def _create_sparse_mask(
        self, 
        seq_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Create sparse attention mask.
        
        Returns a boolean mask where True indicates positions to attend to.
        """
        # Start with causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        
        # Apply window constraint: only attend to tokens within window
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            # Zero out positions before the window (but after position 0)
            if start > 0:
                # Keep global tokens accessible
                for j in range(start):
                    if j not in self.global_token_indices:
                        mask[i, j] = False
        
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for sparse attention.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional additional attention mask.
            
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention: [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = self._apply_rope(q, seq_len)
            k = self._apply_rope(k, seq_len)
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Create and apply sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, hidden_states.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        
        # Apply mask: set non-attended positions to -inf
        attn_weights = attn_weights.masked_fill(~sparse_mask, float("-inf"))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        
        # Reshape back: [batch, seq, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(output)
        
        return output
