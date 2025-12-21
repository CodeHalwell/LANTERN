"""
Tests for LANTERN model and training components.
"""

import pytest
import torch
import tempfile
import os

from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import create_small_config, LANTERNConfig


# Test constants
TEST_VOCAB_SIZE = 100


class TestLANTERNModel:
    """Tests for the complete LANTERN model."""
    
    def test_initialization(self):
        """Test that model initializes correctly."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        assert model.hidden_size == config.hidden_size
        assert model.vocab_size == config.vocab_size
        assert model.config == config
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, _ = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_forward_with_different_recursion_steps(self):
        """Test forward pass with different recursion depths."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        
        # Test with base steps
        logits1, _ = model(input_ids, steps_per_block=2)
        
        # Test with reasoning steps
        logits2, _ = model(input_ids, steps_per_block=4)
        
        # Both should produce valid outputs
        assert logits1.shape == logits2.shape
        assert logits1.shape == (1, 16, config.vocab_size)
    
    def test_return_hidden_states(self):
        """Test that model can return hidden states."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        
        logits, hidden_states = model(input_ids, return_hidden_states=True)
        
        assert hidden_states is not None
        assert hidden_states.shape == (1, 16, config.hidden_size)
    
    def test_get_embedding_matrix(self):
        """Test getting embedding matrix."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        embeddings = model.get_embedding_matrix()
        
        assert embeddings.shape == (config.vocab_size, config.hidden_size)
    
    def test_get_num_params(self):
        """Test parameter counting."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        total_params = model.get_num_params(non_embedding=False)
        non_embedding_params = model.get_num_params(non_embedding=True)
        
        assert total_params > 0
        assert non_embedding_params > 0
        assert total_params > non_embedding_params
    
    def test_weight_tying(self):
        """Test that embeddings are tied with LM head."""
        config = create_small_config()
        model = LANTERNModel(config)
        
        # Check that weights are the same object
        assert model.token_embedding.weight is model.lm_head.weight
    
    def test_generate(self):
        """Test simple generation."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=1.0,
            top_k=10,
        )
        
        # Should have generated 5 new tokens
        assert output.shape[1] == input_ids.shape[1] + 5
    
    def test_generate_with_eos(self):
        """Test generation with EOS token."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        eos_token_id = 99
        
        # This might or might not hit EOS, but shouldn't crash
        output = model.generate(
            input_ids,
            max_new_tokens=5,
            eos_token_id=eos_token_id,
        )
        
        # Should have at least the input tokens
        assert output.shape[1] >= input_ids.shape[1]
    
    def test_custom_config(self):
        """Test with custom configuration."""
        config = LANTERNConfig(
            hidden_size=128,
            num_heads=4,
            num_blocks=1,
            vocab_size=1000,
            window_size=32,
        )
        
        model = LANTERNModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        
        logits, _ = model(input_ids)
        
        assert logits.shape == (1, 16, config.vocab_size)


class TestTrainingComponents:
    """Tests for training-related functionality."""
    
    def test_model_forward_backward(self):
        """Test that gradients flow correctly."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert model.token_embedding.weight.grad is not None
        assert model.lm_head.weight.grad is not None
    
    def test_checkpoint_save_load(self):
        """Test saving and loading checkpoints."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            checkpoint_path = f.name
        
        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create new model and load checkpoint
            new_model = LANTERNModel(config)
            loaded_checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Check that parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    def test_optimizer_step(self):
        """Test optimizer step."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        
        logits, _ = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        
        assert params_changed, "Parameters should change after optimizer step"
