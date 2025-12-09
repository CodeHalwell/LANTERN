"""
Tests for LANTERN controllers.
"""

import pytest
import torch
import torch.nn as nn

from lantern.controller.uncertainty_controller import (
    UncertaintyController,
    UncertaintyLevel,
    UncertaintyResult,
)
from lantern.controller.generation import (
    GenerationController,
    GenerationConfig,
    GenerationMode,
    GenerationStep,
)


class TestUncertaintyController:
    """Tests for UncertaintyController."""
    
    def test_initialization(self):
        """Test controller initializes with correct defaults."""
        controller = UncertaintyController()
        
        assert controller.entropy_weight == 1.0
        assert controller.tau_low == 1.0
        assert controller.tau_high == 3.0
    
    def test_compute_base_uncertainty(self):
        """Test base uncertainty computation without embeddings."""
        controller = UncertaintyController()
        
        vocab_size = 100
        logits = torch.randn(vocab_size)
        
        result = controller.compute_base_uncertainty(logits)
        
        assert isinstance(result, UncertaintyResult)
        assert result.entropy is not None
        assert result.p_max is not None
        assert result.composite_score is not None
        assert result.semantic_dispersion is None  # No embeddings
    
    def test_compute_base_uncertainty_with_embeddings(self):
        """Test base uncertainty with semantic dispersion."""
        controller = UncertaintyController()
        
        vocab_size = 100
        embed_dim = 64
        logits = torch.randn(vocab_size)
        embeddings = torch.randn(vocab_size, embed_dim)
        
        result = controller.compute_base_uncertainty(logits, embeddings)
        
        assert result.semantic_dispersion is not None
    
    def test_classify_uncertainty_confident(self):
        """Test classification as confident."""
        controller = UncertaintyController(tau_low=1.0)
        
        level = controller.classify_uncertainty(torch.tensor(0.5))
        
        assert level == UncertaintyLevel.CONFIDENT
    
    def test_classify_uncertainty_moderate(self):
        """Test classification as moderate."""
        controller = UncertaintyController(tau_low=1.0, tau_mid=2.0)
        
        level = controller.classify_uncertainty(torch.tensor(1.5))
        
        assert level == UncertaintyLevel.MODERATE
    
    def test_classify_uncertainty_high(self):
        """Test classification as high."""
        controller = UncertaintyController(tau_mid=2.0, tau_high=3.0)
        
        level = controller.classify_uncertainty(torch.tensor(2.5))
        
        assert level == UncertaintyLevel.HIGH
    
    def test_classify_uncertainty_very_high(self):
        """Test classification as very high."""
        controller = UncertaintyController(tau_high=3.0)
        
        level = controller.classify_uncertainty(torch.tensor(4.0))
        
        assert level == UncertaintyLevel.VERY_HIGH
    
    def test_compute_total_uncertainty(self):
        """Test combining base and epistemic uncertainty."""
        controller = UncertaintyController(epistemic_weight=0.5)
        
        base_result = UncertaintyResult(
            entropy=torch.tensor(1.0),
            p_max=torch.tensor(0.5),
            semantic_dispersion=torch.tensor(0.3),
            composite_score=torch.tensor(1.0),
        )
        epistemic = torch.tensor(2.0)
        
        total_result = controller.compute_total_uncertainty(base_result, epistemic)
        
        expected = 1.0 + 0.5 * 2.0  # base + weight * epistemic
        assert torch.isclose(total_result.total_score, torch.tensor(expected))
        assert total_result.level is not None
    
    def test_should_trigger_reasoning(self):
        """Test reasoning trigger logic."""
        controller = UncertaintyController(tau_high=3.0)
        
        # Low uncertainty - no trigger
        low_result = UncertaintyResult(
            entropy=torch.tensor(0.5),
            p_max=torch.tensor(0.8),
            semantic_dispersion=None,
            composite_score=torch.tensor(1.0),
            total_score=torch.tensor(1.0),
        )
        assert not controller.should_trigger_reasoning(low_result)
        
        # High uncertainty - trigger
        high_result = UncertaintyResult(
            entropy=torch.tensor(2.0),
            p_max=torch.tensor(0.2),
            semantic_dispersion=None,
            composite_score=torch.tensor(4.0),
            total_score=torch.tensor(4.0),
        )
        assert controller.should_trigger_reasoning(high_result)
    
    def test_should_do_bayesian(self):
        """Test Bayesian sampling trigger logic."""
        controller = UncertaintyController(tau_low=1.0)
        
        # Below threshold - no Bayesian
        low_result = UncertaintyResult(
            entropy=torch.tensor(0.3),
            p_max=torch.tensor(0.9),
            semantic_dispersion=None,
            composite_score=torch.tensor(0.5),
        )
        assert not controller.should_do_bayesian(low_result)
        
        # Above threshold - do Bayesian
        high_result = UncertaintyResult(
            entropy=torch.tensor(1.5),
            p_max=torch.tensor(0.5),
            semantic_dispersion=None,
            composite_score=torch.tensor(1.5),
        )
        assert controller.should_do_bayesian(high_result)
    
    def test_interpret(self):
        """Test interpretation string generation."""
        controller = UncertaintyController()
        
        result = UncertaintyResult(
            entropy=torch.tensor(0.5),
            p_max=torch.tensor(0.9),
            semantic_dispersion=None,
            composite_score=torch.tensor(0.5),
        )
        
        interpretation = controller.interpret(result)
        
        assert isinstance(interpretation, str)
        assert "confident" in interpretation.lower()


class TestGenerationConfig:
    """Tests for GenerationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.steps_base == 4
        assert config.steps_reasoning == 8
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.7,
            think_token_id=1000,
        )
        
        assert config.max_new_tokens == 50
        assert config.temperature == 0.7
        assert config.think_token_id == 1000


class TestGenerationController:
    """Tests for GenerationController."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        hidden_size = 64
        vocab_size = 100
        
        model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        lm_head = nn.Linear(hidden_size, vocab_size)
        embedding_matrix = torch.randn(vocab_size, hidden_size)
        
        return model, lm_head, embedding_matrix, hidden_size, vocab_size
    
    def test_initialization(self, simple_model):
        """Test controller initialization."""
        model, lm_head, embeddings, _, _ = simple_model
        
        controller = GenerationController(
            model=model,
            lm_head=lm_head,
            embedding_matrix=embeddings,
            uncertainty_controller=UncertaintyController(),
        )
        
        assert controller.current_mode == GenerationMode.NORMAL
        assert controller.current_steps == 4  # Default steps_base
    
    def test_mode_switching(self, simple_model):
        """Test mode switching functionality."""
        model, lm_head, embeddings, _, _ = simple_model
        
        config = GenerationConfig(steps_base=4, steps_reasoning=8)
        controller = GenerationController(
            model=model,
            lm_head=lm_head,
            embedding_matrix=embeddings,
            uncertainty_controller=UncertaintyController(),
            config=config,
        )
        
        # Start in normal mode
        assert controller.current_mode == GenerationMode.NORMAL
        assert controller.current_steps == 4
        
        # Switch to reasoning
        controller._switch_to_reasoning_mode()
        assert controller.current_mode == GenerationMode.REASONING
        assert controller.current_steps == 8
        
        # Switch back
        controller._switch_to_normal_mode()
        assert controller.current_mode == GenerationMode.NORMAL
        assert controller.current_steps == 4
    
    def test_sample_token(self, simple_model):
        """Test token sampling."""
        model, lm_head, embeddings, _, vocab_size = simple_model
        
        controller = GenerationController(
            model=model,
            lm_head=lm_head,
            embedding_matrix=embeddings,
            uncertainty_controller=UncertaintyController(),
        )
        
        probs = torch.softmax(torch.randn(vocab_size), dim=-1)
        token_id, prob = controller._sample_token(probs)
        
        assert 0 <= token_id < vocab_size
        assert 0.0 <= prob <= 1.0


class TestGenerationStep:
    """Tests for GenerationStep dataclass."""
    
    def test_creation(self):
        """Test GenerationStep creation."""
        result = UncertaintyResult(
            entropy=torch.tensor(1.0),
            p_max=torch.tensor(0.5),
            semantic_dispersion=None,
            composite_score=torch.tensor(1.0),
        )
        
        step = GenerationStep(
            token_id=42,
            probability=0.3,
            uncertainty=result,
            mode=GenerationMode.NORMAL,
            used_bayesian=True,
        )
        
        assert step.token_id == 42
        assert step.probability == 0.3
        assert step.mode == GenerationMode.NORMAL
        assert step.used_bayesian


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
