"""
Latent Pause Reasoning Module for LANTERN.

Runs additional computation cycles on the hidden state without emitting
a token: extra "thinking time" that never appears in the output.

Design (cross-attention over a frozen context):

    ctx  = LayerNorm(stack_output)                  # fixed for all pause steps
    for i in range(num_steps):
        h = h + PauseEmb_i
        h = h + Attention(q=LayerNorm(h), kv=ctx)
        h = h + FFN(LayerNorm(h))

Keys and values always come from the stack output, not from the evolving
pause state. That keeps training (pause applied to every position of a
sequence) and cached decoding (pause applied to one new token over a cached
context) exactly consistent, and makes each pause step cost one attention
layer over the window rather than a full re-run of the model.
"""

from typing import Optional, Set

import torch
import torch.nn as nn

from lantern.models.kv_cache import KVCache
from lantern.models.sparse_attention import SparseAttention


class LatentPauseModule(nn.Module):
    """Latent pause reasoning module."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        max_pause_steps: int = 4,
        window_size: int = 256,
        dropout: float = 0.1,
        use_rope: bool = True,
        layer_norm_eps: float = 1e-6,
        attn_impl: str = "sdpa",
        max_position: int = 4096,
        global_token_indices: Optional[Set[int]] = None,
    ):
        super().__init__()
        self.max_pause_steps = max_pause_steps
        self.hidden_size = hidden_size

        # Separate pause step embeddings (distinct from recursion step embeddings)
        self.pause_embeddings = nn.Embedding(max_pause_steps, hidden_size)

        self.attention = SparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            global_token_indices=global_token_indices,
            dropout=dropout,
            use_rope=use_rope,
            max_position=max_position,
            attn_impl=attn_impl,
        )

        self.ffn_w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ffn_w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.ln_attn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_ctx = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_ffn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_w2(torch.nn.functional.silu(self.ffn_w1(x)) * self.ffn_w3(x))

    def write_context(
        self, context: torch.Tensor, kv_cache: KVCache, start_pos: int = 0
    ) -> None:
        """
        Project ``context`` (stack output for new positions) to keys/values
        and store them in ``kv_cache`` so later pause steps can attend to it.
        Cheap (two linear layers), so it is done for every generated token
        whether or not that token itself pauses.
        """
        self.attention.write_kv(self.ln_ctx(context), kv_cache, cache_step=0,
                                start_pos=start_pos)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_steps: int = 1,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Run latent pause reasoning steps.

        Args:
            hidden_states: [batch, seq_len, hidden_size] to refine.
            num_steps: Number of pause cycles (clamped to max_pause_steps).
            attention_mask: Optional additive attention mask.
            context: Key/value source [batch, seq_len, hidden_size]. Defaults
                to ``hidden_states`` (the stack output) when no cache is used.
            kv_cache: If given, keys/values are read from the cache (written
                earlier via ``write_context``) instead of from ``context``.
            start_pos: Absolute position of the first token in ``hidden_states``.

        Returns:
            Refined hidden state, same shape as input.
        """
        steps = min(num_steps, self.max_pause_steps)
        if steps <= 0:
            return hidden_states

        ctx = None
        if kv_cache is None:
            ctx = self.ln_ctx(context if context is not None else hidden_states)

        for i in range(steps):
            h = hidden_states + self.pause_embeddings.weight[i]

            residual = h
            h = self.ln_attn(h)
            h = self.attention(
                h,
                attention_mask,
                context=ctx,
                kv_cache=kv_cache,
                cache_step=0,
                start_pos=start_pos,
                write_cache=False,
            )
            h = self.dropout(h)
            h = residual + h

            residual = h
            h = self.ln_ffn(h)
            h = self._ffn(h)
            h = self.dropout(h)
            hidden_states = residual + h

        return hidden_states
