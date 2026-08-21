"""Tests for differentiable ACT ponder cost."""

import pytest
import torch

from lantern.models.recursive_transformer import RecursiveTransformerBlock


class TestACTPonderCost:
    """Tests for adaptive halting, ponder cost, and step_offset in recur."""

    def _make_block(self, hidden_size=64, num_heads=4, intermediate_size=128,
                    max_steps=8):
        return RecursiveTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            window_size=32,
            dropout=0.0,
            use_halting=True,
            max_steps=max_steps,
        )

    def test_adaptive_halting_returns_ponder_cost(self):
        """recur with use_adaptive_halting=True should return non-None ponder_cost."""
        block = self._make_block()
        block.eval()
        x = torch.randn(1, 8, 64)
        _, steps, ponder_cost = block.recur(
            x, steps_max=4, use_adaptive_halting=True,
        )
        assert ponder_cost is not None
        assert ponder_cost.ndim == 0 or ponder_cost.numel() >= 1

    def test_ponder_cost_non_negative(self):
        """Ponder cost should be non-negative."""
        block = self._make_block()
        x = torch.randn(2, 8, 64)
        _, _, ponder_cost = block.recur(
            x, steps_max=6, use_adaptive_halting=True,
        )
        assert ponder_cost is not None
        assert (ponder_cost >= 0.0).all()

    def test_ponder_cost_gradient_connectivity(self):
        """backward on ponder_cost should produce gradients for halting_head."""
        block = self._make_block()
        x = torch.randn(1, 8, 64)
        _, _, ponder_cost = block.recur(
            x, steps_max=4, use_adaptive_halting=True,
        )
        assert ponder_cost is not None
        ponder_cost.sum().backward()
        # halting_head should have gradients
        for name, param in block.halting_head.named_parameters():
            assert param.grad is not None, f"No gradient for halting_head.{name}"
            assert param.grad.abs().sum() > 0, (
                f"Zero gradient for halting_head.{name}"
            )

    def test_without_adaptive_halting_no_ponder_cost(self):
        """recur with use_adaptive_halting=False should return None ponder_cost."""
        block = self._make_block()
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            _, _, ponder_cost = block.recur(
                x, steps_max=4, use_adaptive_halting=False,
            )
        assert ponder_cost is None

    def test_step_offset_shifts_embeddings(self):
        """Different step_offset values should produce different outputs via shifted embeddings."""
        block = self._make_block(max_steps=16)
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out0, _, _ = block.recur(x, steps_max=3, step_offset=0)
            out4, _, _ = block.recur(x, steps_max=3, step_offset=4)
            out8, _, _ = block.recur(x, steps_max=3, step_offset=8)
        assert not torch.allclose(out0, out4, atol=1e-5)
        assert not torch.allclose(out0, out8, atol=1e-5)

    def test_adaptive_halting_output_shape(self):
        """Output shape should match input shape regardless of halting."""
        block = self._make_block()
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            out, _, _ = block.recur(
                x, steps_max=4, use_adaptive_halting=True,
            )
        assert out.shape == x.shape

    def test_halting_head_exists(self):
        """Block with use_halting=True should have a halting_head."""
        block = self._make_block()
        assert block.halting_head is not None
        assert block.use_halting is True

    def test_no_halting_head_when_disabled(self):
        """Block with use_halting=False should have no halting_head."""
        block = RecursiveTransformerBlock(
            hidden_size=64, num_heads=4, intermediate_size=128,
            window_size=32, use_halting=False,
        )
        assert block.halting_head is None
        assert block.use_halting is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
