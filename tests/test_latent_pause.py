"""Tests for the latent pause module."""

import pytest
import torch

from lantern.models.latent_pause import LatentPauseModule


class TestLatentPauseModule:
    """Tests for LatentPauseModule initialization, shape preservation, and pause behaviour."""

    def _make_module(self, hidden_size=64, num_heads=4, intermediate_size=128,
                     max_pause_steps=4, window_size=32):
        return LatentPauseModule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_pause_steps=max_pause_steps,
            window_size=window_size,
            dropout=0.0,  # Deterministic for testing
        )

    def test_initialization(self):
        """Module should have pause embeddings, attention, and FFN components."""
        mod = self._make_module(hidden_size=64, max_pause_steps=4)
        assert mod.hidden_size == 64
        assert mod.max_pause_steps == 4
        assert mod.pause_embeddings.num_embeddings == 4
        assert mod.pause_embeddings.embedding_dim == 64

    def test_forward_shape_preserved(self):
        """Output shape should match input shape [batch, seq_len, hidden_size]."""
        mod = self._make_module()
        mod.eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = mod(x, num_steps=2)
        assert out.shape == x.shape

    def test_forward_single_step(self):
        """A single pause step should still change the input (residual + attention + FFN)."""
        mod = self._make_module()
        mod.eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = mod(x, num_steps=1)
        assert out.shape == x.shape
        # Output should differ from input due to attention and FFN
        assert not torch.allclose(out, x, atol=1e-5)

    def test_multiple_steps_differ_from_fewer(self):
        """More pause steps should produce different output than fewer steps."""
        mod = self._make_module(max_pause_steps=4)
        mod.eval()
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_1 = mod(x, num_steps=1)
            out_3 = mod(x, num_steps=3)
        assert not torch.allclose(out_1, out_3, atol=1e-5)

    def test_max_pause_steps_clamped(self):
        """num_steps exceeding max_pause_steps should be clamped without error."""
        mod = self._make_module(max_pause_steps=3)
        mod.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            # Request more steps than max_pause_steps
            out_max = mod(x, num_steps=3)
            out_over = mod(x, num_steps=10)
        # Should clamp to max and produce same result
        assert torch.allclose(out_max, out_over)

    def test_different_num_steps_values(self):
        """Each distinct num_steps value should give a distinct output."""
        mod = self._make_module(max_pause_steps=4)
        mod.eval()
        x = torch.randn(1, 8, 64)
        outputs = []
        with torch.no_grad():
            for s in range(1, 5):
                outputs.append(mod(x, num_steps=s))
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j], atol=1e-5), (
                    f"num_steps={i + 1} and num_steps={j + 1} produced same output"
                )

    def test_gradient_flow(self):
        """Gradients should flow through the pause module."""
        mod = self._make_module()
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = mod(x, num_steps=2)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
