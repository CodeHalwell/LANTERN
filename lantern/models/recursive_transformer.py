"""
Recursive Transformer Block for LANTERN.

Implements a transformer block that can be recursively applied
with weight sharing for depth-on-demand computation.

Features:
- Step embeddings to prevent representation collapse
- Differentiable ACT with ponder cost
- Probability-weighted output averaging
- Depth-indexed KV cache for incremental decoding
- Optional per-step state trace (used for the step-KL convergence signal)
"""

from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lantern.models.kv_cache import KVCache
from lantern.models.sparse_attention import SparseAttention


class SwiGLU(nn.Module):
    """SwiGLU activation function for the MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HaltingHead(nn.Module):
    """
    Halting head for adaptive computation time (ACT).

    Maps hidden states to halting probabilities per token,
    allowing the model to decide when to stop recursion.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[batch, seq_len, hidden_size] -> halting probabilities [batch, seq_len]."""
        return torch.sigmoid(self.linear(hidden_states)).squeeze(-1)


class RecursiveTransformerBlock(nn.Module):
    """
    Recursive Transformer Block with sparse attention.

    A single transformer block that can be applied multiple times
    with the same parameters (weight sharing). Includes:
    - Sparse multi-head self-attention
    - SwiGLU MLP
    - LayerNorm + residuals (pre-norm)
    - Learned step embeddings to prevent representation collapse
    - Optional halting mechanism for adaptive depth with differentiable ponder cost
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        window_size: int = 256,
        dropout: float = 0.1,
        use_halting: bool = False,
        use_rope: bool = True,
        layer_norm_eps: float = 1e-6,
        max_steps: int = 8,
        attn_impl: str = "sdpa",
        max_position: int = 4096,
        global_token_indices: Optional[Set[int]] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_halting = use_halting
        self.max_steps = max_steps

        # Step embeddings: like positional embeddings, but over recursion depth.
        self.step_embeddings = nn.Embedding(max_steps, hidden_size)

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
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.halting_head = HaltingHead(hidden_size) if use_halting else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single application of the block.

        Args:
            hidden_states: [batch, seq_len, hidden_size].
            attention_mask: Optional additive attention mask.
            step_index: Recursion step index for the step embedding and the
                cache slot.
            kv_cache: Optional depth-indexed KV cache (incremental decoding).
            start_pos: Absolute position of the first token in ``hidden_states``.

        Returns:
            (output hidden states, halting probabilities or None)
        """
        if step_index is not None:
            # Depths beyond max_steps reuse the final step embedding.
            step_emb = self.step_embeddings.weight[min(step_index, self.max_steps - 1)]
            hidden_states = hidden_states + step_emb

        # Depths beyond max_steps reuse the last step embedding; the cache
        # slot is clamped the same way so deep runs never index past it.
        cache_step = 0
        if kv_cache is not None and step_index is not None:
            cache_step = min(step_index, kv_cache.max_steps - 1)

        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask,
            kv_cache=kv_cache,
            cache_step=cache_step,
            start_pos=start_pos,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        p_halt = None
        if self.halting_head is not None:
            p_halt = self.halting_head(hidden_states)

        return hidden_states, p_halt

    def recur(
        self,
        hidden_states: torch.Tensor,
        steps_max: int = 4,
        attention_mask: Optional[torch.Tensor] = None,
        use_adaptive_halting: bool = False,
        halting_eps: float = 0.01,
        step_offset: int = 0,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
        step_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Recursive application of the block with step embeddings and ACT.

        Args:
            hidden_states: [batch, seq_len, hidden_size].
            steps_max: Maximum number of recursion steps.
            attention_mask: Optional attention mask.
            use_adaptive_halting: Whether to use learned halting.
            halting_eps: Threshold for halting (1 - eps).
            step_offset: Offset for step embedding indices.
            kv_cache: Optional depth-indexed KV cache.
            start_pos: Absolute position of the first token.
            step_states: If a list is given, the hidden state after every
                step is appended to it (for the step-KL signal).

        Returns:
            (output hidden states, actual steps, ponder_cost or None)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        last_step_idx = step_offset

        if use_adaptive_halting and self.halting_head is not None:
            cum_halt = torch.zeros(batch_size, seq_len, device=device)
            accumulated_output = torch.zeros_like(hidden_states)
            ponder_cost = torch.zeros(batch_size, seq_len, device=device)

            actual_steps = 0
            for t in range(steps_max):
                step_idx = t + step_offset
                last_step_idx = step_idx
                hidden_states, p_halt = self.forward(
                    hidden_states, attention_mask, step_index=step_idx,
                    kv_cache=kv_cache, start_pos=start_pos,
                )
                if step_states is not None:
                    step_states.append(hidden_states)
                actual_steps += 1

                still_active = (cum_halt < 1.0 - halting_eps).float()
                increment = torch.minimum(p_halt, 1.0 - cum_halt) * still_active
                accumulated_output = accumulated_output + increment.unsqueeze(-1) * hidden_states
                ponder_cost = ponder_cost + increment * (t + 1)
                cum_halt = cum_halt + increment

                if (cum_halt >= 1.0 - halting_eps).all():
                    break

            remainder = (1.0 - cum_halt).clamp(min=0)
            accumulated_output = accumulated_output + remainder.unsqueeze(-1) * hidden_states
            ponder_cost = ponder_cost + remainder * steps_max
            output = accumulated_output
        else:
            actual_steps = 0
            for t in range(steps_max):
                step_idx = t + step_offset
                last_step_idx = step_idx
                hidden_states, _ = self.forward(
                    hidden_states, attention_mask, step_index=step_idx,
                    kv_cache=kv_cache, start_pos=start_pos,
                )
                if step_states is not None:
                    step_states.append(hidden_states)
                actual_steps += 1
            output = hidden_states
            ponder_cost = None

        if kv_cache is not None and actual_steps > 0:
            # Tokens that ran fewer steps than the cache holds must still be
            # visible at deeper slots to later tokens that recurse further.
            kv_cache.carry_forward_range(
                start_pos, start_pos + seq_len, min(last_step_idx, kv_cache.max_steps - 1)
            )

        return output, actual_steps, ponder_cost


class RecursiveTransformerStack(nn.Module):
    """
    Stack of recursive transformer blocks.

    Multiple independent blocks that are each recursively applied.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        hidden_size: int = 512,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        window_size: int = 256,
        dropout: float = 0.1,
        use_halting: bool = False,
        max_steps: int = 8,
        use_rope: bool = True,
        attn_impl: str = "sdpa",
        max_position: int = 4096,
        global_token_indices: Optional[Set[int]] = None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            RecursiveTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                window_size=window_size,
                dropout=dropout,
                use_halting=use_halting,
                use_rope=use_rope,
                max_steps=max_steps,
                attn_impl=attn_impl,
                max_position=max_position,
                global_token_indices=global_token_indices,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        steps_per_block: int = 4,
        attention_mask: Optional[torch.Tensor] = None,
        use_adaptive_halting: bool = False,
        kv_caches: Optional[List[KVCache]] = None,
        start_pos: int = 0,
        step_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Forward pass through all blocks with recursion.

        Args:
            kv_caches: One KVCache per block, or None.
            step_states: If given, receives the per-step hidden states of the
                *last* block (the ones the LM head reads from).

        Returns:
            (output, total steps taken, aggregated ponder cost or None)
        """
        total_steps = 0
        total_ponder_cost = None
        n_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            hidden_states, steps, ponder_cost = block.recur(
                hidden_states,
                steps_max=steps_per_block,
                attention_mask=attention_mask,
                use_adaptive_halting=use_adaptive_halting,
                kv_cache=kv_caches[i] if kv_caches is not None else None,
                start_pos=start_pos,
                step_states=step_states if i == n_blocks - 1 else None,
            )
            total_steps += steps
            if ponder_cost is not None:
                total_ponder_cost = (
                    ponder_cost if total_ponder_cost is None
                    else total_ponder_cost + ponder_cost
                )

        return hidden_states, total_steps, total_ponder_cost
