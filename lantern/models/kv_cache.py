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

    def write(self, step: int, start_pos: int, k: torch.Tensor, v: torch.Tensor):
        """
        Write KV for a contiguous run of positions at one depth.

        Args:
            step: Recursion depth index.
            start_pos: First sequence position to write.
            k: Keys [batch, heads, n, head_dim].
            v: Values [batch, heads, n, head_dim].
        """
        n = k.shape[-2]
        end = start_pos + n
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: writing positions {start_pos}..{end - 1} "
                f"into a cache of length {self.max_seq_len}"
            )
        self.k_cache[step, :, :, start_pos:end, :] = k.to(self.k_cache.dtype)
        self.v_cache[step, :, :, start_pos:end, :] = v.to(self.v_cache.dtype)
        self.seq_len = max(self.seq_len, end)

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
        self.carry_forward_range(pos, pos + 1, depth)

    def carry_forward_range(self, start_pos: int, end_pos: int, depth: int):
        """Carry forward a contiguous run of positions from ``depth`` to deeper slots."""
        if depth + 1 >= self.max_steps:
            return
        src_k = self.k_cache[depth, :, :, start_pos:end_pos, :]
        src_v = self.v_cache[depth, :, :, start_pos:end_pos, :]
        self.k_cache[depth + 1:, :, :, start_pos:end_pos, :] = src_k.unsqueeze(0)
        self.v_cache[depth + 1:, :, :, start_pos:end_pos, :] = src_v.unsqueeze(0)

    def select_rows(self, rows: torch.Tensor) -> "KVCache":
        """
        A new cache holding copies of the given batch rows (index tensor).
        Used to run a deeper pass on a subset of a batch without touching
        the other rows' cache entries.
        """
        sub = KVCache.__new__(KVCache)
        sub.max_steps = self.max_steps
        sub.batch_size = int(rows.numel())
        sub.num_heads = self.num_heads
        sub.max_seq_len = self.max_seq_len
        sub.head_dim = self.head_dim
        sub.seq_len = self.seq_len
        sub.k_cache = self.k_cache[:, rows].clone()
        sub.v_cache = self.v_cache[:, rows].clone()
        return sub

    def write_rows_from(self, other: "KVCache", rows: torch.Tensor, start_pos: int, end_pos: int):
        """Copy positions start_pos..end_pos-1 of ``other`` (all steps) into ``rows`` of this cache."""
        self.k_cache[:, rows, :, start_pos:end_pos] = other.k_cache[:, :, :, start_pos:end_pos]
        self.v_cache[:, rows, :, start_pos:end_pos] = other.v_cache[:, :, :, start_pos:end_pos]
        self.seq_len = max(self.seq_len, end_pos)

    def truncate(self, seq_len: int):
        """Drop cached positions beyond ``seq_len`` (used to rewind a rejected token)."""
        self.seq_len = min(self.seq_len, seq_len)

    def reset(self):
        """Reset the cache for a new sequence."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0
