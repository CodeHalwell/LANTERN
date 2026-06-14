"""
Tests for LANTERN sparse attention and recursive transformer.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from lantern.models.sparse_attention import SparseAttention
from lantern.models.recursive_transformer import (
    RecursiveTransformerBlock,
    RecursiveTransformerStack,
    SwiGLU,
    HaltingHead,
)


class TestSparseAttention:
    """Tests for SparseAttention module."""
    
    def test_initialization(self):
        """Test that sparse attention initializes correctly."""
        attn = SparseAttention(
            hidden_size=256,
            num_heads=4,
            window_size=64,
        )
        
        assert attn.hidden_size == 256
        assert attn.num_heads == 4
        assert attn.head_dim == 64
        assert attn.window_size == 64
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 128
        hidden_size = 256
        
        attn = SparseAttention(
            hidden_size=hidden_size,
            num_heads=4,
            window_size=64,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
    
    def test_sparse_mask_creation(self):
        """Test that sparse mask has correct structure."""
        attn = SparseAttention(
            hidden_size=256,
            num_heads=4,
            window_size=4,
            global_token_indices={0},
        )
        
        mask = attn._create_sparse_mask(10, torch.device("cpu"))
        
        # Check mask is boolean
        assert mask.dtype == torch.bool
        
        # Check shape
        assert mask.shape == (10, 10)
        
        # Check causal: no token attends to future
        for i in range(10):
            for j in range(i + 1, 10):
                assert mask[i, j] == False
        
        # Check global token (0) is attended by all
        for i in range(10):
            assert mask[i, 0] == True
    
    def test_rope_embeddings(self):
        """Test rotary position embeddings are applied."""
        attn = SparseAttention(
            hidden_size=256,
            num_heads=4,
            window_size=64,
            use_rope=True,
        )
        
        # Check RoPE buffers exist
        assert hasattr(attn, "cos_cached")
        assert hasattr(attn, "sin_cached")
        assert hasattr(attn, "inv_freq")

    def test_rope_preserves_norm(self):
        """RoPE is a rotation: it must preserve each vector's norm."""
        attn = SparseAttention(
            hidden_size=64,
            num_heads=4,
            window_size=16,
            use_rope=True,
        )
        # x shape [batch, heads, seq, head_dim]
        seq_len = 20
        x = torch.randn(2, 4, seq_len, attn.head_dim)
        rotated = attn._apply_rope(x, seq_len)

        norm_before = x.norm(dim=-1)
        norm_after = rotated.norm(dim=-1)
        max_diff = (norm_before - norm_after).abs().max().item()
        assert max_diff < 1e-5, f"RoPE changed norms (max diff {max_diff})"

    def test_rope_relative_position_invariance(self):
        """q.k after RoPE depends only on the relative offset d, not absolute p."""
        attn = SparseAttention(
            hidden_size=64,
            num_heads=1,
            window_size=64,
            use_rope=True,
        )
        head_dim = attn.head_dim
        # A fixed query and key vector.
        q_vec = torch.randn(head_dim)
        k_vec = torch.randn(head_dim)

        def rotated_dot(p_query, p_key):
            # Build a sequence long enough to cover the larger position, then
            # place the query at p_query and key at p_key.
            max_pos = max(p_query, p_key) + 1
            qx = torch.zeros(1, 1, max_pos, head_dim)
            kx = torch.zeros(1, 1, max_pos, head_dim)
            qx[0, 0, p_query] = q_vec
            kx[0, 0, p_key] = k_vec
            qr = attn._apply_rope(qx, max_pos)
            kr = attn._apply_rope(kx, max_pos)
            return torch.dot(qr[0, 0, p_query], kr[0, 0, p_key]).item()

        d = 3
        dot_a = rotated_dot(2, 2 + d)   # positions (2, 5)
        dot_b = rotated_dot(7, 7 + d)   # positions (7, 10), same offset d
        assert abs(dot_a - dot_b) < 1e-4, (
            f"q.k not relative-position invariant: {dot_a} vs {dot_b}"
        )


def _reference_masked_dense_attention(attn: SparseAttention, x: torch.Tensor):
    """Reference masked-dense sliding-window + causal-global attention.

    Materializes the full [B, H, L, L] score matrix and masks it, mirroring the
    semantics the efficient forward must reproduce. Uses the module's own
    projections / RoPE / scale so the only difference is the algorithm.
    """
    batch_size, seq_len, _ = x.shape
    q = attn.q_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

    if attn.use_rope:
        q = attn._apply_rope(q, seq_len)
        k = attn._apply_rope(k, seq_len)

    scale = math.sqrt(attn.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]

    row = torch.arange(seq_len).unsqueeze(1)
    col = torch.arange(seq_len).unsqueeze(0)
    mask = (row >= col) & (row - col < attn.window_size)
    for g in attn.global_token_indices:
        if 0 <= g < seq_len:
            # Causal global: query i attends global g only if g <= i.
            mask[:, g] = mask[:, g] | (row.squeeze(1) >= g)

    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, attn.hidden_size)
    return attn.out_proj(out)


class TestEfficientWindowedAttention:
    """The memory-efficient forward must match the masked-dense reference."""

    @pytest.mark.parametrize(
        "seq_len,window_size,global_idx",
        [
            (16, 8, {0}),
            (40, 8, {0}),          # L >> window
            (40, 16, {0, 3}),      # multiple globals, one not at 0
            (33, 16, {0, 5, 20}),  # global beyond first window
            (10, 32, {0}),         # window larger than seq
        ],
    )
    def test_matches_masked_dense(self, seq_len, window_size, global_idx):
        torch.manual_seed(0)
        attn = SparseAttention(
            hidden_size=32,
            num_heads=4,
            window_size=window_size,
            global_token_indices=set(global_idx),
            dropout=0.0,
        )
        attn.eval()
        x = torch.randn(2, seq_len, 32)
        with torch.no_grad():
            efficient = attn(x)
            reference = _reference_masked_dense_attention(attn, x)
        max_diff = (efficient - reference).abs().max().item()
        assert max_diff < 1e-4, f"efficient vs dense max diff {max_diff}"


class TestConfigWiring:
    """use_rope and global_token_indices must propagate to the attention."""

    def test_use_rope_false_propagates(self):
        stack = RecursiveTransformerStack(
            num_blocks=2,
            hidden_size=64,
            num_heads=4,
            use_rope=False,
        )
        for block in stack.blocks:
            assert block.attention.use_rope is False

    def test_global_token_indices_propagate(self):
        gidx = {0, 4, 9}
        stack = RecursiveTransformerStack(
            num_blocks=2,
            hidden_size=64,
            num_heads=4,
            global_token_indices=gidx,
        )
        for block in stack.blocks:
            assert block.attention.global_token_indices == gidx


class TestDifferentiableHalting:
    """Adaptive halting must deliver gradient to the halting head."""

    def test_halting_head_receives_gradient(self):
        torch.manual_seed(0)
        block = RecursiveTransformerBlock(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            window_size=16,
            dropout=0.0,
            use_halting=True,
        )
        x = torch.randn(2, 16, 64)
        out, steps = block.recur(x, steps_max=4, use_adaptive_halting=True)
        loss = out.sum()
        loss.backward()

        grad = block.halting_head.linear.weight.grad
        assert grad is not None, "halting head got no gradient"
        assert grad.abs().sum().item() > 0, "halting head gradient is zero"

    def test_ponder_cost_recorded(self):
        block = RecursiveTransformerBlock(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            window_size=16,
            dropout=0.0,
            use_halting=True,
        )
        x = torch.randn(2, 16, 64)
        block.recur(x, steps_max=4, use_adaptive_halting=True)
        cost = block._last_ponder_cost
        assert cost.dim() == 0
        # Expected steps must lie within [1, steps_max].
        assert 1.0 - 1e-4 <= cost.item() <= 4.0 + 1e-4


class TestRecursiveTransformerBlock:
    """Tests for RecursiveTransformerBlock."""
    
    def test_initialization(self):
        """Test block initializes correctly."""
        block = RecursiveTransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=512,
        )
        
        assert block.hidden_size == 256
        assert block.use_halting == False
        assert block.halting_head is None
    
    def test_initialization_with_halting(self):
        """Test block initializes with halting head."""
        block = RecursiveTransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=512,
            use_halting=True,
        )
        
        assert block.use_halting == True
        assert block.halting_head is not None
    
    def test_forward_shape(self):
        """Test forward pass produces correct shapes."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        
        block = RecursiveTransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            intermediate_size=512,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output, p_halt = block(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert p_halt is None  # No halting head
    
    def test_forward_with_halting(self):
        """Test forward with halting head."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        
        block = RecursiveTransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            intermediate_size=512,
            use_halting=True,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output, p_halt = block(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert p_halt is not None
        assert p_halt.shape == (batch_size, seq_len)
        # Halting probs should be in [0, 1]
        assert (p_halt >= 0).all() and (p_halt <= 1).all()
    
    def test_recur_fixed_steps(self):
        """Test recursive application with fixed steps."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        
        block = RecursiveTransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            intermediate_size=512,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output, steps = block.recur(x, steps_max=4)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert steps == 4
    
    def test_recur_different_depths(self):
        """Test that different recursion depths produce different outputs."""
        batch_size = 1
        seq_len = 32
        hidden_size = 256
        
        block = RecursiveTransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            intermediate_size=512,
            dropout=0.0,  # Disable dropout for deterministic comparison
        )
        block.eval()
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        with torch.no_grad():
            output2, _ = block.recur(x.clone(), steps_max=2)
            output4, _ = block.recur(x.clone(), steps_max=4)
        
        # Different depths should produce different outputs
        assert not torch.allclose(output2, output4)


class TestRecursiveTransformerStack:
    """Tests for RecursiveTransformerStack."""
    
    def test_initialization(self):
        """Test stack initializes correctly."""
        stack = RecursiveTransformerStack(
            num_blocks=2,
            hidden_size=256,
            num_heads=4,
        )
        
        assert len(stack.blocks) == 2
    
    def test_forward(self):
        """Test forward through stack."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        
        stack = RecursiveTransformerStack(
            num_blocks=2,
            hidden_size=hidden_size,
            num_heads=4,
        )
        
        x = torch.randn(batch_size, seq_len, hidden_size)
        output, total_steps = stack(x, steps_per_block=2)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert total_steps == 4  # 2 blocks * 2 steps each


class TestSwiGLU:
    """Tests for SwiGLU activation."""
    
    def test_forward_shape(self):
        """Test SwiGLU produces correct shape."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        intermediate_size = 512
        
        swiglu = SwiGLU(hidden_size, intermediate_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = swiglu(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)


class TestHaltingHead:
    """Tests for HaltingHead."""
    
    def test_forward_shape(self):
        """Test halting head produces correct shape."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        
        head = HaltingHead(hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = head(x)
        
        assert output.shape == (batch_size, seq_len)
    
    def test_output_range(self):
        """Test halting probabilities are in [0, 1]."""
        hidden_size = 256
        
        head = HaltingHead(hidden_size)
        x = torch.randn(10, 64, hidden_size)
        output = head(x)
        
        assert (output >= 0).all()
        assert (output <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
