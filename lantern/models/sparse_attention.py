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
        
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
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
        """Initialize rotary position embeddings.

        The cos/sin cache is built in the *interleaved* layout to match the
        interleaved rotation used in ``_apply_rope``. For each frequency f the
        cache stores ``[f, f]`` adjacent so that dimension pair ``(2k, 2k+1)``
        shares the same angle. This makes the operation an actual rotation
        (norm preserving and relative-position invariant).
        """
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute cos/sin for positions
        positions = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        # Interleave so emb[..., 2k] == emb[..., 2k+1] == freqs[..., k].
        # This is consistent with the interleaved rotation in _apply_rope.
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embeddings to tensor (interleaved layout)."""
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Rotate pairs of dimensions (interleaved): (x_2k, x_2k+1) -> rotation.
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

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
        # Create window mask using broadcasting
        row_indices = torch.arange(seq_len, device=device).unsqueeze(1)
        col_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = (row_indices >= col_indices) & (row_indices - col_indices < self.window_size)
        # Add global token mask (causal: a query i may attend a global token g
        # only if g <= i, never a future global token).
        for idx in self.global_token_indices:
            if 0 <= idx < seq_len:
                mask[:, idx] = mask[:, idx] | (row_indices.squeeze(1) >= idx)

        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for sparse attention.

        Memory-efficient sliding-window + global-token attention. The full
        ``[B, H, L, L]`` score matrix is never materialized. Instead the
        queries are processed in blocks of ``window_size``; each query block
        only computes scores against:

          * its local key band (the current block plus the previous block,
            which together cover the causal window ``[i - window_size, i]``),
          * the set of global tokens (causally masked so a query at position
            ``i`` can only see a global token ``g`` if ``g <= i``).

        This achieves ``O(L * (window_size + num_global))`` time and memory per
        attention layer rather than the dense ``O(L^2)``. The result is
        numerically equivalent (within fp tolerance) to a masked-dense
        computation with the same sliding-window + causal-global mask.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional additive attention mask broadcastable to
                [batch, heads, seq, seq] (e.g. a padding mask with 0/-inf).

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

        scale = math.sqrt(self.head_dim)
        w = self.window_size

        # Global token indices that fall within the current sequence, sorted.
        global_idx = sorted(g for g in self.global_token_indices if 0 <= g < seq_len)
        global_idx_t = torch.tensor(global_idx, device=hidden_states.device, dtype=torch.long) \
            if global_idx else None

        # Pre-gather global keys/values once: [B, H, G, D]
        if global_idx_t is not None:
            k_global = k.index_select(2, global_idx_t)
            v_global = v.index_select(2, global_idx_t)

        output = torch.empty_like(q)

        # Process the queries in blocks of size `w`. For block starting at qs,
        # the causal window [i - w, i] for any i in [qs, qe) is fully contained
        # in key positions [max(0, qs - w + 1), qe). We bound the key band by
        # [ks, qe) with ks = max(0, qs - w + 1) so it never exceeds 2*w keys.
        for qs in range(0, seq_len, w):
            qe = min(qs + w, seq_len)
            q_blk = q[:, :, qs:qe, :]  # [B, H, Lq, D]
            lq = qe - qs

            ks = max(0, qs - w + 1)
            k_band = k[:, :, ks:qe, :]  # [B, H, Lk, D]
            v_band = v[:, :, ks:qe, :]
            lk = qe - ks

            # Scores against local band: [B, H, Lq, Lk]
            scores_band = torch.matmul(q_blk, k_band.transpose(-2, -1)) / scale

            q_pos = torch.arange(qs, qe, device=hidden_states.device).unsqueeze(1)   # [Lq, 1]
            k_pos = torch.arange(ks, qe, device=hidden_states.device).unsqueeze(0)   # [1, Lk]
            # Sliding-window causal mask: attend iff 0 <= i - j < window_size.
            band_mask = (q_pos >= k_pos) & (q_pos - k_pos < w)  # [Lq, Lk]
            if global_idx_t is not None:
                # Global tokens are scored exclusively in the global block to
                # avoid double counting; remove any that fall inside the band.
                k_is_global = torch.zeros_like(k_pos, dtype=torch.bool)  # [1, Lk]
                in_band_global = (global_idx_t >= ks) & (global_idx_t < qe)
                band_global_pos = global_idx_t[in_band_global] - ks
                if band_global_pos.numel() > 0:
                    k_is_global[0, band_global_pos] = True
                band_mask = band_mask & (~k_is_global)
            scores_band = scores_band.masked_fill(
                ~band_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

            if global_idx_t is not None:
                # Scores against global tokens: [B, H, Lq, G]
                scores_global = torch.matmul(q_blk, k_global.transpose(-2, -1)) / scale
                # Causal global mask: query i may attend global g only if g <= i.
                g_pos = global_idx_t.unsqueeze(0)  # [1, G]
                global_keep = (q_pos >= g_pos)  # [Lq, G]
                scores_global = scores_global.masked_fill(
                    ~global_keep.unsqueeze(0).unsqueeze(0), float("-inf")
                )
                scores = torch.cat([scores_band, scores_global], dim=-1)
                v_cat = torch.cat([v_band, v_global], dim=2)
            else:
                scores = scores_band
                v_cat = v_band

            if attention_mask is not None:
                # Slice the additive mask to the keys actually used. We support
                # a [B, H, L, L] / broadcastable additive mask.
                am = attention_mask
                # Expand to [B, H, L, L] view via broadcasting on a per-query
                # block basis. Index the key dimension for band and globals.
                am_q = am[..., qs:qe, :] if am.shape[-2] != 1 else am[..., 0:1, :]
                am_band = am_q[..., ks:qe]
                scores[..., :lk] = scores[..., :lk] + am_band
                if global_idx_t is not None:
                    am_global = am_q.index_select(-1, global_idx_t)
                    scores[..., lk:] = scores[..., lk:] + am_global

            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            output[:, :, qs:qe, :] = torch.matmul(attn_probs, v_cat)

        # Reshape back: [batch, seq, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.out_proj(output)

        return output
