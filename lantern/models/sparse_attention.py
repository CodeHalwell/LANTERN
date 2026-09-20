"""
Sparse Attention Module for LANTERN.

Sliding-window causal attention with global tokens. Three backends:

- ``eager``: materialises the full [L, T] score matrix and masks it. Simple,
  works everywhere, O(L * T) memory. Reference implementation.
- ``sdpa``: ``torch.nn.functional.scaled_dot_product_attention`` with a
  boolean mask. Same FLOPs as eager but the fused kernels never materialise
  the score matrix in memory, which is what matters for training at scale.
- ``flex``: ``torch.nn.attention.flex_attention`` with a block mask. This is
  the only backend that actually skips the masked-out blocks, giving the
  O(L * w) compute the design promises. Needs a CUDA device and, for speed,
  ``torch.compile``. Attention dropout is not supported on this path.

The module also supports:

- cross-attention (``context``): keys/values come from a different tensor
  than the queries. Used by the latent pause module so a pause step can attend
  over the frozen sequence context.
- incremental decoding (``kv_cache``): keys/values for new positions are
  written into a depth-indexed cache and attention runs over the cached
  prefix. ``start_pos`` is the absolute position of the first query token.
"""

import math
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from lantern.models.kv_cache import KVCache

try:  # FlexAttention landed in torch 2.5
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    _HAS_FLEX = True
except ImportError:  # pragma: no cover - older torch
    _HAS_FLEX = False

ATTENTION_IMPLS = ("eager", "sdpa", "flex")


def flex_attention_available() -> bool:
    """Whether the FlexAttention backend can be selected."""
    return _HAS_FLEX


class SparseAttention(nn.Module):
    """
    Sparse multi-head attention with sliding window and global tokens.

    For each query at absolute position i, attention covers:
    - key positions j with i - window_size < j <= i (causal local window)
    - global key positions (e.g. BOS) that every token can attend to
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
        attn_impl: str = "sdpa",
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        )
        head_dim = hidden_size // num_heads
        if use_rope:
            assert head_dim % 2 == 0, "head_dim must be even when using RoPE"
        if attn_impl not in ATTENTION_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTENTION_IMPLS}, got {attn_impl!r}")
        if attn_impl == "flex" and not _HAS_FLEX:
            raise RuntimeError("attn_impl='flex' requires torch>=2.5 with flex_attention")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_token_indices = global_token_indices or {0}  # At least BOS
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.max_position = max_position
        self.attn_impl = attn_impl

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        if use_rope:
            self._init_rope(max_position)

        # Sorted tuple so the flex mask_mod can close over a static structure.
        self._global_sorted = tuple(sorted(self.global_token_indices))
        self._block_mask_cache = {}

    # ------------------------------------------------------------------ RoPE
    def _init_rope(self, max_position: int):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        positions = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def _apply_rope(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings for absolute positions start_pos..start_pos+seq_len."""
        cos = self.cos_cached[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)

    # ----------------------------------------------------------------- masks
    def _create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Square [seq_len, seq_len] boolean mask (True = attend). Kept for compatibility."""
        return self._create_mask(seq_len, seq_len, 0, device)

    def _create_mask(
        self, q_len: int, k_len: int, start_pos: int, device: torch.device
    ) -> torch.Tensor:
        """
        Boolean mask [q_len, k_len] for queries at absolute positions
        start_pos..start_pos+q_len-1 over keys at positions 0..k_len-1.
        """
        q_pos = torch.arange(start_pos, start_pos + q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        mask = (q_pos >= k_pos) & (q_pos - k_pos < self.window_size)
        for idx in self.global_token_indices:
            if idx < k_len:
                mask[:, idx] = True
        return mask

    def _get_block_mask(self, q_len: int, k_len: int, start_pos: int, device: torch.device):
        key = (q_len, k_len, start_pos, str(device))
        block_mask = self._block_mask_cache.get(key)
        if block_mask is None:
            window = self.window_size
            globals_ = self._global_sorted

            def mask_mod(b, h, q_idx, kv_idx):
                q_abs = q_idx + start_pos
                allowed = (q_abs >= kv_idx) & (q_abs - kv_idx < window)
                for g in globals_:
                    allowed = allowed | (kv_idx == g)
                return allowed

            block_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=k_len, device=str(device)
            )
            self._block_mask_cache[key] = block_mask
        return block_mask

    # --------------------------------------------------------------- forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        cache_step: int = 0,
        start_pos: int = 0,
        write_cache: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Queries [batch, q_len, hidden_size].
            attention_mask: Optional additive mask broadcastable to
                [batch, heads, q_len, k_len] (0 = keep, -inf = drop).
            context: Optional key/value source [batch, q_len, hidden_size].
                Must cover the same positions as ``hidden_states``.
            kv_cache: Optional cache. New keys/values are written at
                ``cache_step`` for positions start_pos..start_pos+q_len-1 and
                attention runs over the whole cached prefix.
            cache_step: Depth slot in the cache to read/write.
            start_pos: Absolute position of the first query token.
            write_cache: When a cache is given, whether to write the new
                keys/values before reading. ``False`` means the cache already
                holds them (see ``write_kv``).

        Returns:
            [batch, q_len, hidden_size]
        """
        batch_size, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = self._apply_rope(q, q_len, start_pos)

        if kv_cache is not None:
            if write_cache:
                kv_source = context if context is not None else hidden_states
                k, v = self._project_kv(kv_source, start_pos)
                kv_cache.write(cache_step, start_pos, k, v)
            k, v = kv_cache.get_slice(cache_step)
        else:
            kv_source = context if context is not None else hidden_states
            k, v = self._project_kv(kv_source, start_pos)
        k_len = k.shape[-2]

        if self.attn_impl == "flex" and attention_mask is None and k.is_cuda:
            output = self._flex_attention(q, k, v, q_len, k_len, start_pos)
        elif self.attn_impl == "eager":
            output = self._eager_attention(q, k, v, q_len, k_len, start_pos, attention_mask)
        else:
            output = self._sdpa_attention(q, k, v, q_len, k_len, start_pos, attention_mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        return self.out_proj(output)

    def _project_kv(self, kv_source: torch.Tensor, start_pos: int):
        """Project a [batch, n, hidden] tensor to rotated keys/values [batch, heads, n, head_dim]."""
        batch_size, n, _ = kv_source.shape
        k = self.k_proj(kv_source).view(batch_size, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_source).view(batch_size, n, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            k = self._apply_rope(k, n, start_pos)
        return k, v

    def write_kv(
        self,
        kv_source: torch.Tensor,
        kv_cache: KVCache,
        cache_step: int = 0,
        start_pos: int = 0,
    ) -> None:
        """Project ``kv_source`` and write it into the cache without attending."""
        k, v = self._project_kv(kv_source, start_pos)
        kv_cache.write(cache_step, start_pos, k, v)

    def _eager_attention(self, q, k, v, q_len, k_len, start_pos, attention_mask):
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        sparse_mask = self._create_mask(q_len, k_len, start_pos, q.device)
        attn_weights = attn_weights.masked_fill(~sparse_mask, float("-inf"))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_probs = F.softmax(attn_weights, dim=-1)
        # A fully masked row gives NaN; zero it rather than propagate.
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, v)

    def _sdpa_attention(self, q, k, v, q_len, k_len, start_pos, attention_mask):
        sparse_mask = self._create_mask(q_len, k_len, start_pos, q.device)
        if attention_mask is not None:
            float_mask = torch.zeros(q_len, k_len, dtype=q.dtype, device=q.device)
            float_mask = float_mask.masked_fill(~sparse_mask, float("-inf"))
            mask = float_mask + attention_mask.to(q.dtype)
        else:
            mask = sparse_mask
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

    def _flex_attention(self, q, k, v, q_len, k_len, start_pos):
        block_mask = self._get_block_mask(q_len, k_len, start_pos, q.device)
        return flex_attention(q, k, v, block_mask=block_mask)
