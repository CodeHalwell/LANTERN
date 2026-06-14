"""
Recursive Transformer Block for LANTERN.

Implements a transformer block that can be recursively applied
with weight sharing for depth-on-demand computation.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        """
        Compute halting probabilities.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            Halting probabilities [batch, seq_len]
        """
        return torch.sigmoid(self.linear(hidden_states)).squeeze(-1)


class RecursiveTransformerBlock(nn.Module):
    """
    Recursive Transformer Block with sparse attention.
    
    A single transformer block that can be applied multiple times
    with the same parameters (weight sharing). Includes:
    - Sparse multi-head self-attention
    - SwiGLU MLP
    - LayerNorm + residuals
    - Optional halting mechanism for adaptive depth
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
        global_token_indices=None,
        layer_norm_eps: float = 1e-6,
    ):
        """
        Initialize recursive transformer block.

        Args:
            hidden_size: Dimension of hidden states.
            num_heads: Number of attention heads.
            intermediate_size: Dimension of MLP intermediate layer.
            window_size: Size of sliding attention window.
            dropout: Dropout probability.
            use_halting: Whether to use adaptive halting mechanism.
            use_rope: Whether to use Rotary Position Embeddings.
            global_token_indices: Set of global token indices for attention.
            layer_norm_eps: Epsilon for layer normalization.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.use_halting = use_halting

        # Attention with sparse pattern
        self.attention = SparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            global_token_indices=global_token_indices,
            dropout=dropout,
            use_rope=use_rope,
        )
        
        # MLP with SwiGLU
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        
        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional halting head for adaptive depth
        if use_halting:
            self.halting_head = HaltingHead(hidden_size)
        else:
            self.halting_head = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single forward pass through the block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask.
            
        Returns:
            Tuple of (output hidden states, halting probabilities if use_halting).
        """
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Compute halting probabilities if enabled
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
    ) -> Tuple[torch.Tensor, int]:
        """
        Recursive application of the block.

        Applies the same block multiple times (weight sharing). When
        ``use_adaptive_halting`` is True and a halting head is present, this
        implements a differentiable PonderNet/ACT-style mechanism: at each step
        the halting head produces a per-token halting probability, and the final
        output is a halting-probability-weighted combination of the per-step
        hidden states. Because the output mixes states with weights that depend
        on the halting head, gradients flow into the halting head through the
        downstream loss (no non-differentiable early ``break``).

        The expected ponder cost (expected number of steps) is recorded on
        ``self._last_ponder_cost`` so callers can add it as a regularizer.

        When ``use_adaptive_halting`` is False (or no halting head exists) the
        behavior is identical to plain fixed-depth recursion.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            steps_max: Maximum number of recursion steps.
            attention_mask: Optional attention mask.
            use_adaptive_halting: Whether to use learned halting.
            halting_eps: Unused tolerance kept for backward compatibility.

        Returns:
            Tuple of (output hidden states, number of steps taken).
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        adaptive = use_adaptive_halting and self.halting_head is not None

        if not adaptive:
            # Plain fixed-depth recursion (identical to original behavior).
            for _ in range(steps_max):
                hidden_states, _ = self.forward(hidden_states, attention_mask)
            self._last_ponder_cost = torch.zeros((), device=device)
            return hidden_states, steps_max

        # --- Differentiable PonderNet/ACT-style halting ---
        # remainder: probability mass not yet allocated, per token [B, L].
        remainder = torch.ones(batch_size, seq_len, device=device)
        weighted_output = torch.zeros_like(hidden_states)
        # Expected number of steps accumulator, per token [B, L].
        expected_steps = torch.zeros(batch_size, seq_len, device=device)

        for t in range(steps_max):
            hidden_states, p_halt = self.forward(hidden_states, attention_mask)

            if t == steps_max - 1:
                # Last step takes all remaining mass so weights sum to 1.
                step_weight = remainder
            else:
                step_weight = remainder * p_halt

            # Accumulate halting-probability-weighted hidden state.
            weighted_output = weighted_output + step_weight.unsqueeze(-1) * hidden_states
            # Expected steps: sum over steps of weight * (step index + 1).
            expected_steps = expected_steps + step_weight * (t + 1)
            # Reduce remaining probability mass.
            remainder = remainder * (1.0 - p_halt)

        # Mean expected number of steps across tokens/batch (scalar tensor).
        self._last_ponder_cost = expected_steps.mean()
        return weighted_output, steps_max


class RecursiveTransformerStack(nn.Module):
    """
    Stack of recursive transformer blocks.
    
    Multiple independent blocks that can each be recursively applied.
    Useful when you want some layer diversity but still want recursion.
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
        use_rope: bool = True,
        global_token_indices=None,
    ):
        super().__init__()

        # Track the last expected ponder cost (mean expected steps) so callers
        # can retrieve it after a forward pass. Defaults to a zero scalar.
        self._last_ponder_cost = None

        self.blocks = nn.ModuleList([
            RecursiveTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                window_size=window_size,
                dropout=dropout,
                use_halting=use_halting,
                use_rope=use_rope,
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
    ) -> Tuple[torch.Tensor, int]:
        """
        Forward pass through all blocks with recursion.

        When ``use_adaptive_halting`` is True, the expected ponder cost (mean
        expected number of steps) is accumulated across blocks and stored on
        ``self._last_ponder_cost`` as a scalar tensor. Otherwise it is set to a
        zero scalar tensor.

        Returns:
            Tuple of (output, total steps taken across all blocks).
        """
        device = hidden_states.device
        total_steps = 0
        ponder_cost = torch.zeros((), device=device)
        for block in self.blocks:
            hidden_states, steps = block.recur(
                hidden_states,
                steps_max=steps_per_block,
                attention_mask=attention_mask,
                use_adaptive_halting=use_adaptive_halting,
            )
            total_steps += steps
            block_cost = getattr(block, "_last_ponder_cost", None)
            if block_cost is not None:
                ponder_cost = ponder_cost + block_cost

        self._last_ponder_cost = ponder_cost
        return hidden_states, total_steps
