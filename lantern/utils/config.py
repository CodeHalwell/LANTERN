"""
Configuration for LANTERN.

Centralized configuration for all LANTERN components.
"""

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class LANTERNConfig:
    """
    Complete configuration for LANTERN model and generation.
    
    Attributes:
        Model architecture:
            hidden_size: Dimension of hidden states.
            num_heads: Number of attention heads.
            intermediate_size: MLP intermediate dimension.
            num_blocks: Number of transformer blocks in stack.
        
        Sparse attention:
            window_size: Size of sliding attention window.
            global_token_indices: Indices of global tokens.
        
        Recursion:
            steps_base: Base recursion depth.
            steps_reasoning: Recursion depth in reasoning mode.
            use_adaptive_halting: Whether to use learned halting.
            halting_eps: Epsilon for halting threshold.
        
        Uncertainty:
            entropy_weight: Weight for entropy in composite score.
            dispersion_weight: Weight for semantic dispersion.
            p_max_weight: Weight for max probability.
            epistemic_weight: Weight for epistemic uncertainty.
            tau_low: Threshold for confident classification.
            tau_mid: Threshold for moderate uncertainty.
            tau_high: Threshold for high uncertainty.
            top_k_dispersion: k for semantic dispersion.
        
        Bayesian:
            num_bayesian_samples: Number of MC dropout samples.
        
        Generation:
            max_new_tokens: Maximum tokens to generate.
            temperature: Softmax temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling.
            
        Special tokens:
            think_token_id: ID of THINK token.
            unknown_token_id: ID of UNKNOWN token.
            eos_token_id: End of sequence token ID.
    """
    
    # Model architecture
    hidden_size: int = 512
    num_heads: int = 8
    intermediate_size: int = 2048
    num_blocks: int = 2
    vocab_size: int = 32000
    max_position: int = 4096
    
    # Sparse attention
    window_size: int = 256
    global_token_indices: Set[int] = field(default_factory=lambda: {0})
    
    # Positional encoding
    use_rope: bool = True
    
    # Dropout
    dropout: float = 0.1
    
    # Recursion
    steps_base: int = 4
    steps_reasoning: int = 8
    use_adaptive_halting: bool = False
    halting_eps: float = 0.01
    
    # Uncertainty weights
    entropy_weight: float = 1.0
    dispersion_weight: float = 0.5
    p_max_weight: float = -0.5
    epistemic_weight: float = 0.3
    
    # Uncertainty thresholds
    tau_low: float = 1.0
    tau_mid: float = 2.0
    tau_high: float = 3.0
    top_k_dispersion: int = 10
    
    # Bayesian
    num_bayesian_samples: int = 5
    
    # Generation
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Special tokens
    think_token_id: Optional[int] = None
    unknown_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: int = 0
    
    # Reasoning prefix
    reasoning_prefix: str = "Let me think through this carefully step by step:\n"
    
    def to_model_config(self) -> dict:
        """Extract model-related configuration."""
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "window_size": self.window_size,
            "dropout": self.dropout,
            "use_halting": self.use_adaptive_halting,
            "use_rope": self.use_rope,
        }
    
    def to_uncertainty_config(self) -> dict:
        """Extract uncertainty controller configuration."""
        return {
            "entropy_weight": self.entropy_weight,
            "dispersion_weight": self.dispersion_weight,
            "p_max_weight": self.p_max_weight,
            "epistemic_weight": self.epistemic_weight,
            "tau_low": self.tau_low,
            "tau_mid": self.tau_mid,
            "tau_high": self.tau_high,
            "temperature": self.temperature,
            "top_k_dispersion": self.top_k_dispersion,
        }
    
    def to_generation_config(self) -> dict:
        """Extract generation configuration."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "steps_base": self.steps_base,
            "steps_reasoning": self.steps_reasoning,
            "num_bayesian_samples": self.num_bayesian_samples,
            "think_token_id": self.think_token_id,
            "unknown_token_id": self.unknown_token_id,
            "eos_token_id": self.eos_token_id,
            "reasoning_prefix": self.reasoning_prefix,
        }


def create_small_config() -> LANTERNConfig:
    """Create configuration for a small model (testing/prototyping)."""
    return LANTERNConfig(
        hidden_size=256,
        num_heads=4,
        intermediate_size=512,
        num_blocks=1,
        window_size=64,
        steps_base=2,
        steps_reasoning=4,
    )


def create_base_config() -> LANTERNConfig:
    """Create configuration for a base model."""
    return LANTERNConfig(
        hidden_size=512,
        num_heads=8,
        intermediate_size=2048,
        num_blocks=2,
        window_size=256,
        steps_base=4,
        steps_reasoning=8,
    )


def create_large_config() -> LANTERNConfig:
    """Create configuration for a larger model."""
    return LANTERNConfig(
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        num_blocks=4,
        window_size=512,
        steps_base=6,
        steps_reasoning=12,
    )
