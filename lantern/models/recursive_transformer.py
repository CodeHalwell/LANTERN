"""
Recursive Transformer Block for LANTERN.

Implements a transformer block that can be recursively applied
with weight sharing for depth-on-demand computation.

v3-final features:
- Step embeddings to prevent representation collapse
- Differentiable ACT with ponder cost
- Probability-weighted output averaging
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
            layer_norm_eps: Epsilon for layer normalization.
            max_steps: Maximum recursion steps (for step embeddings).
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_halting = use_halting
        self.max_steps = max_steps
        
        # Step embeddings to prevent representation collapse during recursion.
        # Analogous to positional embeddings but for recursion depth.
        self.step_embeddings = nn.Embedding(max_steps, hidden_size)
        
        # Attention with sparse pattern
        self.attention = SparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
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
        step_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Single forward pass through the block.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask.
            step_index: Optional recursion step index for step embedding.
            
        Returns:
            Tuple of (output hidden states, halting probabilities if use_halting).
        """
        # Add step embedding if step index is provided
        if step_index is not None and step_index < self.max_steps:
            step_emb = self.step_embeddings(
                torch.tensor(step_index, device=hidden_states.device)
            )
            hidden_states = hidden_states + step_emb
        
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
        step_offset: int = 0,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Recursive application of the block with step embeddings and ACT.
        
        Applies the same block multiple times (weight sharing),
        optionally with adaptive halting based on learned probabilities.
        When ACT is enabled, returns probability-weighted average of hidden
        states and a differentiable ponder cost.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            steps_max: Maximum number of recursion steps.
            attention_mask: Optional attention mask.
            use_adaptive_halting: Whether to use learned halting.
            halting_eps: Threshold for halting (1 - eps).
            step_offset: Offset for step embedding indices (for reasoning continuity).
            
        Returns:
            Tuple of (output hidden states, actual steps, ponder_cost or None).
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        if use_adaptive_halting and self.halting_head is not None:
            # Differentiable ACT: probability-weighted output averaging
            cum_halt = torch.zeros(batch_size, seq_len, device=device)
            accumulated_output = torch.zeros_like(hidden_states)
            ponder_cost = torch.zeros(batch_size, seq_len, device=device)
            
            actual_steps = 0
            for t in range(steps_max):
                step_idx = t + step_offset
                hidden_states, p_halt = self.forward(
                    hidden_states, attention_mask, step_index=step_idx
                )
                actual_steps += 1
                
                # Compute increment: min(p_halt, remaining probability)
                still_active = (cum_halt < 1.0 - halting_eps).float()
                increment = torch.minimum(p_halt, 1.0 - cum_halt) * still_active
                
                # Accumulate weighted output
                accumulated_output = accumulated_output + increment.unsqueeze(-1) * hidden_states
                
                # Differentiable ponder cost: increment * (step + 1)
                ponder_cost = ponder_cost + increment * (t + 1)
                
                cum_halt = cum_halt + increment
                
                if (cum_halt >= 1.0 - halting_eps).all():
                    break
            
            # Handle remainder probability
            remainder = (1.0 - cum_halt).clamp(min=0)
            accumulated_output = accumulated_output + remainder.unsqueeze(-1) * hidden_states
            ponder_cost = ponder_cost + remainder * steps_max
            
            return accumulated_output, actual_steps, ponder_cost
        else:
            # Fixed-depth recursion (no ACT)
            actual_steps = 0
            for t in range(steps_max):
                step_idx = t + step_offset
                hidden_states, p_halt = self.forward(
                    hidden_states, attention_mask, step_index=step_idx
                )
                actual_steps += 1
            
            return hidden_states, actual_steps, None


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
        max_steps: int = 8,
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
                max_steps=max_steps,
            )
            for _ in range(num_blocks)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        steps_per_block: int = 4,
        attention_mask: Optional[torch.Tensor] = None,
        use_adaptive_halting: bool = False,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Forward pass through all blocks with recursion.
        
        Returns:
            Tuple of (output, total steps taken, aggregated ponder cost or None).
        """
        total_steps = 0
        total_ponder_cost = None
        for block in self.blocks:
            hidden_states, steps, ponder_cost = block.recur(
                hidden_states,
                steps_max=steps_per_block,
                attention_mask=attention_mask,
                use_adaptive_halting=use_adaptive_halting,
            )
            total_steps += steps
            if ponder_cost is not None:
                if total_ponder_cost is None:
                    total_ponder_cost = ponder_cost
                else:
                    total_ponder_cost = total_ponder_cost + ponder_cost
        
        return hidden_states, total_steps, total_ponder_cost
