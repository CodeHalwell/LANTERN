"""Tests for the epistemic uncertainty probe."""

import pytest
import torch

from lantern.uncertainty.epistemic_probe import EpistemicProbe


class TestEpistemicProbe:
    """Tests for EpistemicProbe initialization, output shape/range, and gradient flow."""

    def test_initialization(self):
        """Probe should create two linear layers with correct dimensions."""
        probe = EpistemicProbe(hidden_size=128)
        params = dict(probe.named_parameters())
        # First linear: hidden_size -> hidden_size // 4
        assert params["net.0.weight"].shape == (32, 128)
        # Second linear: hidden_size // 4 -> 1
        assert params["net.2.weight"].shape == (1, 32)

    def test_output_shape_unbatched(self):
        """Forward on [batch, hidden_size] should return [batch]."""
        probe = EpistemicProbe(hidden_size=64)
        x = torch.randn(4, 64)
        with torch.no_grad():
            out = probe(x)
        assert out.shape == (4,)

    def test_output_shape_batched_3d(self):
        """Forward on [batch, seq_len, hidden_size] should return [batch, seq_len]."""
        probe = EpistemicProbe(hidden_size=64)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = probe(x)
        assert out.shape == (2, 8)

    def test_output_bounded_zero_one_unbatched(self):
        """Output should be in [0, 1] for 2D input."""
        probe = EpistemicProbe(hidden_size=128)
        x = torch.randn(16, 128)
        with torch.no_grad():
            out = probe(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_output_bounded_zero_one_batched(self):
        """Output should be in [0, 1] for 3D batched input."""
        probe = EpistemicProbe(hidden_size=128)
        x = torch.randn(4, 10, 128)
        with torch.no_grad():
            out = probe(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_output_bounded_extreme_inputs(self):
        """Output should remain bounded even with large-magnitude inputs."""
        probe = EpistemicProbe(hidden_size=64)
        x = torch.randn(8, 64) * 100.0
        with torch.no_grad():
            out = probe(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_gradient_flow(self):
        """Gradients should flow back through the probe."""
        probe = EpistemicProbe(hidden_size=64)
        x = torch.randn(4, 64, requires_grad=True)
        out = probe(x)
        loss = out.sum()
        loss.backward()
        # Gradients should reach the input
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        # Gradients should reach all probe parameters
        for name, param in probe.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_deterministic(self):
        """Same input should produce same output in eval mode."""
        probe = EpistemicProbe(hidden_size=64)
        probe.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            out1 = probe(x)
            out2 = probe(x)
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
