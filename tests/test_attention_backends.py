"""Tests for the attention backends, cross-attention and incremental decoding."""

import pytest
import torch

from lantern.models.kv_cache import KVCache
from lantern.models.sparse_attention import SparseAttention, flex_attention_available


def _attn(impl, **kw):
    torch.manual_seed(0)
    return SparseAttention(hidden_size=64, num_heads=4, window_size=8, dropout=0.0,
                           attn_impl=impl, **kw).eval()


class TestBackends:
    def test_eager_and_sdpa_agree(self):
        a = _attn("eager")
        b = _attn("sdpa")
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 20, 64)
        assert torch.allclose(a(x), b(x), atol=1e-5)

    def test_additive_mask_agrees_across_backends(self):
        a = _attn("eager")
        b = _attn("sdpa")
        b.load_state_dict(a.state_dict())
        x = torch.randn(2, 12, 64)
        mask = torch.zeros(12, 12)
        mask[:, 3] = float("-inf")  # nobody may attend to position 3
        assert torch.allclose(a(x, mask), b(x, mask), atol=1e-5)

    def test_invalid_impl_rejected(self):
        with pytest.raises(ValueError):
            SparseAttention(hidden_size=64, num_heads=4, attn_impl="nope")

    @pytest.mark.skipif(not flex_attention_available(), reason="flex attention not available")
    def test_flex_falls_back_on_cpu(self):
        # On CPU the flex backend must silently use sdpa and match eager.
        a = _attn("eager")
        f = _attn("flex")
        f.load_state_dict(a.state_dict())
        x = torch.randn(1, 16, 64)
        assert torch.allclose(a(x), f(x), atol=1e-5)

    def test_window_respected(self):
        """A token outside the window (and not global) must not influence output."""
        a = _attn("sdpa")
        x = torch.randn(1, 20, 64)
        y = x.clone()
        y[0, 5] += 10.0  # position 5 is outside the 8-window of position 19 and not global
        out_x = a(x)[0, 19]
        out_y = a(y)[0, 19]
        assert torch.allclose(out_x, out_y, atol=1e-5)


class TestCrossAttention:
    def test_context_changes_output(self):
        a = _attn("sdpa")
        x = torch.randn(2, 10, 64)
        ctx = torch.randn(2, 10, 64)
        assert not torch.allclose(a(x), a(x, context=ctx))

    def test_context_equal_to_input_is_self_attention(self):
        a = _attn("eager")
        x = torch.randn(2, 10, 64)
        assert torch.allclose(a(x), a(x, context=x), atol=1e-6)


class TestIncrementalDecoding:
    def _cache(self, max_steps=1, batch=2, length=32):
        return KVCache(max_steps, batch, 4, length, 16, torch.device("cpu"))

    def test_prefill_then_decode_matches_full(self):
        a = _attn("sdpa")
        x = torch.randn(2, 12, 64)
        full = a(x)
        cache = self._cache()
        pre = a(x[:, :7], kv_cache=cache, cache_step=0, start_pos=0)
        assert torch.allclose(pre, full[:, :7], atol=1e-5)
        for t in range(7, 12):
            step = a(x[:, t:t + 1], kv_cache=cache, cache_step=0, start_pos=t)
            assert torch.allclose(step[:, 0], full[:, t], atol=1e-5)

    def test_write_kv_then_readonly_forward(self):
        a = _attn("eager")
        x = torch.randn(1, 6, 64)
        ctx = torch.randn(1, 6, 64)
        ref = a(x, context=ctx)
        cache = self._cache(batch=1)
        a.write_kv(ctx, cache, cache_step=0, start_pos=0)
        out = a(x, kv_cache=cache, cache_step=0, start_pos=0, write_cache=False)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_cache_overflow_raises(self):
        a = _attn("sdpa")
        cache = self._cache(length=4)
        with pytest.raises(ValueError):
            a(torch.randn(2, 6, 64), kv_cache=cache)

    def test_write_and_carry_forward_range(self):
        cache = self._cache(max_steps=3)
        k = torch.randn(2, 4, 5, 16)
        v = torch.randn(2, 4, 5, 16)
        cache.write(0, 0, k, v)
        assert cache.seq_len == 5
        cache.carry_forward_range(0, 5, 0)
        assert torch.equal(cache.k_cache[2, :, :, :5], k)
        assert torch.equal(cache.v_cache[1, :, :, :5], v)
        cache.truncate(3)
        assert cache.get_slice(0)[0].shape[-2] == 3
