"""
Depth-Aware KV Cache for LANTERN.

Implements a depth-indexed KV cache with carry-forward to handle
variable recursion depths during autoregressive generation.

Shape: (max_steps, batch, heads, max_seq_len, head_dim)
"""

import torch


class KVCache:
    """
    Depth-aware KV cache for variable-depth recursive transformers.

    Each recursion step has its own cache slice. When a token halts at
    depth d, carry_forward copies its KV from step d into all deeper
    steps d+1..max_steps so that later deep-reasoning tokens can attend
    to valid context at every depth.
    """

    def __init__(
        self,
        max_steps: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.seq_len = 0  # Current sequence length in cache

        # Shape: (max_steps, batch, heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            max_steps, batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            max_steps, batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )

    def update_slice(
        self,
        step: int,
        seq_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        Write KV for a specific (depth, position) slot.

        Args:
            step: Recursion depth index.
            seq_pos: Sequence position index.
            k: Key tensor [batch, heads, 1, head_dim] or [batch, heads, head_dim].
            v: Value tensor [batch, heads, 1, head_dim] or [batch, heads, head_dim].
        """
        if k.dim() == 4:
            k = k.squeeze(2)
        if v.dim() == 4:
            v = v.squeeze(2)
        self.k_cache[step, :, :, seq_pos, :] = k
        self.v_cache[step, :, :, seq_pos, :] = v
        self.seq_len = max(self.seq_len, seq_pos + 1)

    def get_slice(self, step: int) -> tuple:
        """
        Get the full KV cache for a given depth step.

        Args:
            step: Recursion depth index.

        Returns:
            Tuple of (k, v) with shape (batch, heads, seq_len, head_dim).
        """
        return (
            self.k_cache[step, :, :, :self.seq_len, :],
            self.v_cache[step, :, :, :self.seq_len, :],
        )

    def carry_forward(self, pos: int, depth: int):
        """
        Copy KV from depth d into all deeper steps d+1..max_steps-1.

        This ensures deep-reasoning tokens always find valid context
        from earlier tokens that may have halted at shallower depths.

        Args:
            pos: Sequence position to carry forward.
            depth: The depth at which the token finished processing.
        """
        for future_step in range(depth + 1, self.max_steps):
            self.k_cache[future_step, :, :, pos, :] = self.k_cache[depth, :, :, pos, :]
            self.v_cache[future_step, :, :, pos, :] = self.v_cache[depth, :, :, pos, :]

    def reset(self):
        """Reset the cache for a new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0
