"""Tests for step embeddings in RecursiveTransformerBlock."""

import pytest
import torch

from lantern.models.recursive_transformer import RecursiveTransformerBlock


class TestStepEmbeddings:
    """Tests for step_embeddings attribute and its effect on forward/recur."""

    def _make_block(self, hidden_size=64, num_heads=4, intermediate_size=128,
                    max_steps=8, use_halting=False, **kwargs):
        return RecursiveTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            window_size=32,
            dropout=0.0,
            use_halting=use_halting,
            max_steps=max_steps,
            **kwargs,
        )

    def test_has_step_embeddings(self):
        """Block should contain a step_embeddings nn.Embedding."""
        block = self._make_block(max_steps=8)
        assert hasattr(block, "step_embeddings")
        assert block.step_embeddings.num_embeddings == 8
        assert block.step_embeddings.embedding_dim == 64

    def test_forward_with_step_index_differs_from_without(self):
        """Providing step_index should produce different output than step_index=None."""
        block = self._make_block()
        block.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_none, _ = block(x, step_index=None)
            out_step0, _ = block(x, step_index=0)
        # They should differ because step embedding is added when step_index is given
        assert not torch.allclose(out_none, out_step0, atol=1e-5)

    def test_different_step_indices_produce_different_output(self):
        """Different step_index values should produce different outputs."""
        block = self._make_block(max_steps=8)
        block.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out_0, _ = block(x, step_index=0)
            out_3, _ = block(x, step_index=3)
            out_7, _ = block(x, step_index=7)
        assert not torch.allclose(out_0, out_3, atol=1e-5)
        assert not torch.allclose(out_0, out_7, atol=1e-5)
        assert not torch.allclose(out_3, out_7, atol=1e-5)

    def test_recur_uses_step_embeddings(self):
        """recur with different step_offset should produce different results."""
        block = self._make_block(max_steps=8)
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            out_off0, steps0, _ = block.recur(x, steps_max=3, step_offset=0)
            out_off2, steps2, _ = block.recur(x, steps_max=3, step_offset=2)
        assert not torch.allclose(out_off0, out_off2, atol=1e-5)

    def test_step_embeddings_shape_matches_hidden_size(self):
        """step_embeddings dimension should equal the block's hidden_size."""
        for hs in [32, 128, 256]:
            block = self._make_block(hidden_size=hs, intermediate_size=hs * 2,
                                     num_heads=4)
            assert block.step_embeddings.embedding_dim == hs

    def test_recur_fixed_depth_returns_correct_steps(self):
        """recur without adaptive halting should use exactly steps_max iterations."""
        block = self._make_block()
        block.eval()
        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            _, actual_steps, ponder = block.recur(x, steps_max=5)
        assert actual_steps == 5
        assert ponder is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
