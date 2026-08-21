"""
Epistemic Probe for LANTERN.

A lightweight neural network trained to predict MC Dropout uncertainty
from a single forward pass, eliminating the need for multiple forward
passes at inference time.
"""

import torch
import torch.nn as nn


class EpistemicProbe(nn.Module):
    """
    Probe network that predicts epistemic uncertainty from hidden states.

    Architecture: Linear(D, D/4) -> GELU -> Linear(D/4, 1) -> sigmoid

    Trained during Phase 2 via distillation from MC Dropout variance.
    At inference, replaces expensive multi-pass MC Dropout entirely.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        intermediate = hidden_size // 4
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict epistemic uncertainty from hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size].

        Returns:
            Bounded uncertainty estimate in [0, 1].
            Shape matches input but with last dim reduced to 1 then squeezed.
        """
        return torch.sigmoid(self.net(hidden_states)).squeeze(-1)
