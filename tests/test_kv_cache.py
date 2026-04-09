"""Tests for the depth-aware KV cache."""

import pytest
import torch

from lantern.models.kv_cache import KVCache


class TestKVCache:
    """Tests for KVCache initialization, update, retrieval, carry-forward, and reset."""

    def _make_cache(self, max_steps=4, batch_size=2, num_heads=4,
                    max_seq_len=8, head_dim=16):
        return KVCache(
            max_steps=max_steps,
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_initialization_shapes(self):
        """Cache tensors should have shape (max_steps, batch, heads, max_seq_len, head_dim)."""
        cache = self._make_cache(max_steps=4, batch_size=2, num_heads=4,
                                 max_seq_len=8, head_dim=16)
        expected = (4, 2, 4, 8, 16)
        assert cache.k_cache.shape == expected
        assert cache.v_cache.shape == expected
        assert cache.seq_len == 0

    def test_initialization_zeros(self):
        """Cache should be initialized to zeros."""
        cache = self._make_cache()
        assert torch.all(cache.k_cache == 0)
        assert torch.all(cache.v_cache == 0)

    def test_update_slice_writes_correct_slot(self):
        """update_slice should write key/value to the correct (depth, position) slot."""
        cache = self._make_cache(max_steps=4, batch_size=2, num_heads=4,
                                 max_seq_len=8, head_dim=16)
        k = torch.ones(2, 4, 1, 16) * 3.0
        v = torch.ones(2, 4, 1, 16) * 7.0
        cache.update_slice(step=1, seq_pos=3, k=k, v=v)

        assert torch.allclose(cache.k_cache[1, :, :, 3, :], torch.tensor(3.0))
        assert torch.allclose(cache.v_cache[1, :, :, 3, :], torch.tensor(7.0))
        # Other positions should remain zero
        assert torch.all(cache.k_cache[0] == 0)
        assert torch.all(cache.k_cache[1, :, :, 0, :] == 0)

    def test_update_slice_updates_seq_len(self):
        """seq_len should track the furthest written position + 1."""
        cache = self._make_cache(max_steps=4, batch_size=2, num_heads=4,
                                 max_seq_len=8, head_dim=16)
        k = torch.ones(2, 4, 1, 16)
        v = torch.ones(2, 4, 1, 16)
        cache.update_slice(step=0, seq_pos=0, k=k, v=v)
        assert cache.seq_len >= 1
        cache.update_slice(step=0, seq_pos=4, k=k, v=v)
        assert cache.seq_len >= 5

    def test_get_slice_returns_correct_data(self):
        """get_slice should return key/value for only the populated portion."""
        cache = self._make_cache(max_steps=4, batch_size=2, num_heads=4,
                                 max_seq_len=8, head_dim=16)
        k = torch.randn(2, 4, 1, 16)
        v = torch.randn(2, 4, 1, 16)
        cache.update_slice(step=2, seq_pos=0, k=k, v=v)

        k_out, v_out = cache.get_slice(step=2)
        assert k_out.shape[-2] == cache.seq_len
        assert v_out.shape[-2] == cache.seq_len
        assert torch.allclose(k_out[:, :, 0, :], k.squeeze(2))

    def test_get_slice_empty_cache(self):
        """get_slice on a fresh cache should return zero-length seq dimension."""
        cache = self._make_cache()
        k_out, v_out = cache.get_slice(step=0)
        assert k_out.shape[-2] == 0
        assert v_out.shape[-2] == 0

    def test_carry_forward_copies_to_deeper_steps(self):
        """carry_forward should copy KV from depth d to all deeper steps."""
        cache = self._make_cache(max_steps=4, batch_size=2, num_heads=4,
                                 max_seq_len=8, head_dim=16)
        k = torch.randn(2, 4, 1, 16)
        v = torch.randn(2, 4, 1, 16)
        cache.update_slice(step=1, seq_pos=0, k=k, v=v)
        cache.carry_forward(pos=0, depth=1)

        for d in range(2, 4):
            assert torch.allclose(
                cache.k_cache[d, :, :, 0, :],
                cache.k_cache[1, :, :, 0, :],
            )
            assert torch.allclose(
                cache.v_cache[d, :, :, 0, :],
                cache.v_cache[1, :, :, 0, :],
            )

    def test_carry_forward_does_not_overwrite_source(self):
        """carry_forward should preserve the source step data."""
        cache = self._make_cache(max_steps=4, batch_size=1, num_heads=2,
                                 max_seq_len=4, head_dim=8)
        k = torch.randn(1, 2, 1, 8)
        v = torch.randn(1, 2, 1, 8)
        cache.update_slice(step=0, seq_pos=0, k=k, v=v)
        original_k = cache.k_cache[0].clone()
        cache.carry_forward(pos=0, depth=0)
        assert torch.allclose(cache.k_cache[0], original_k)

    def test_reset_clears_all_data(self):
        """reset should zero out caches and set seq_len to 0."""
        cache = self._make_cache()
        k = torch.randn(2, 4, 1, 16)
        v = torch.randn(2, 4, 1, 16)
        cache.update_slice(step=0, seq_pos=0, k=k, v=v)
        assert cache.seq_len > 0

        cache.reset()
        assert cache.seq_len == 0
        assert torch.all(cache.k_cache == 0)
        assert torch.all(cache.v_cache == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
