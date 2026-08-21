"""
Latent Pause Reasoning Module for LANTERN.

Runs additional computation cycles on the hidden state without emitting
a token. Gives the model extra "thinking time" invisibly.
"""

import torch
import torch.nn as nn

from lantern.models.sparse_attention import SparseAttention


class LatentPauseModule(nn.Module):
    """
    Latent pause reasoning module.

    For each pause step:
        h = h + PauseEmb_i
        h = h + Attention(LayerNorm(h), KV_context)
        h = h + FFN(LayerNorm(h))

    Uses separate pause step embeddings (distinct from main recursion step
    embeddings) to avoid collision. Receives KV cache from the main
    attention so the single-token hidden state can attend to full context.
    """

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
    ):
        super().__init__()
        self.max_pause_steps = max_pause_steps
        self.hidden_size = hidden_size

        # Separate pause step embeddings
        self.pause_embeddings = nn.Embedding(max_pause_steps, hidden_size)

        # Attention layer for pause reasoning (shares architecture, separate weights)
        self.attention = SparseAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            use_rope=use_rope,
        )

        # FFN for pause reasoning
        self.ffn_w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.ffn_w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Layer norms
        self.ln_attn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_ffn = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU FFN."""
        return self.ffn_w2(torch.nn.functional.silu(self.ffn_w1(x)) * self.ffn_w3(x))

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_steps: int = 1,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run latent pause reasoning steps.

        Args:
            hidden_states: Current hidden state [batch, seq_len, hidden_size].
                For single-token pause, seq_len may be 1 but attention should
                still have access to full context via the attention mask and
                the broader sequence context passed in.
            num_steps: Number of pause computation cycles.
            attention_mask: Optional attention mask for the full context.

        Returns:
            Refined hidden state with same shape as input.
        """
        steps = min(num_steps, self.max_pause_steps)
        for i in range(steps):
            # Add pause step embedding
            pause_idx = torch.tensor(
                i, device=hidden_states.device, dtype=torch.long
            )
            pause_emb = self.pause_embeddings(pause_idx)
            h = hidden_states + pause_emb

            # Attention with context awareness
            residual = h
            h = self.ln_attn(h)
            h = self.attention(h, attention_mask)
            h = self.dropout(h)
            h = residual + h

            # FFN
            residual = h
            h = self.ln_ffn(h)
            h = self._ffn(h)
            h = self.dropout(h)
            hidden_states = residual + h

        return hidden_states
