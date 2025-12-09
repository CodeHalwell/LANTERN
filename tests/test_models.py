"""
Tests for LANTERN sparse attention and recursive transformer.
"""

import pytest
import torch
import torch.nn as nn

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
