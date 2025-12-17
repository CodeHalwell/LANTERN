"""
Generation Controller for LANTERN.

Implements the full decoding loop with:
- Uncertainty-aware sampling
- THINK token injection for reasoning mode
- Adaptive recursion depth
- Bayesian refinement
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lantern.uncertainty.bayesian import bayesian_step
from lantern.controller.uncertainty_controller import (
    UncertaintyController,
    UncertaintyResult,
)


class GenerationMode(Enum):
    """Current generation mode."""
    NORMAL = "normal"
    REASONING = "reasoning"
    ABSTAIN = "abstain"


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    
    # Recursion settings
    steps_base: int = 4
    steps_reasoning: int = 8
    
    # Bayesian settings
    num_bayesian_samples: int = 5
    
    # Special tokens
    think_token_id: Optional[int] = None
    unknown_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Reasoning mode settings
    reasoning_prefix: str = "Let me think through this carefully step by step:\n"


@dataclass
class GenerationStep:
    """Information about a single generation step."""
    token_id: int
    probability: float
    uncertainty: UncertaintyResult
    mode: GenerationMode
    used_bayesian: bool = False


class GenerationController:
    """
    Main controller for uncertainty-aware text generation.
    
    Orchestrates the full generation loop, combining:
    1. Recursive transformer inference
    2. Uncertainty estimation
    3. Mode switching (normal/reasoning/abstain)
    4. Bayesian refinement when needed
    """
    
    def __init__(
        self,
        model: nn.Module,
        lm_head: nn.Module,
        embedding_matrix: torch.Tensor,
        uncertainty_controller: UncertaintyController,
        config: Optional[GenerationConfig] = None,
        recur_fn: Optional[Callable] = None,
    ):
        """
        Initialize generation controller.
        
        Args:
            model: The transformer model (or recursive block).
            lm_head: Language model head for logit prediction.
            embedding_matrix: Token embeddings for semantic dispersion.
            uncertainty_controller: Controller for uncertainty estimation.
            config: Generation configuration.
            recur_fn: Optional recursive forward function.
        """
        self.model = model
        self.lm_head = lm_head
        self.embedding_matrix = embedding_matrix
        self.uncertainty_controller = uncertainty_controller
        self.config = config or GenerationConfig()
        self.recur_fn = recur_fn
        
        self.current_mode = GenerationMode.NORMAL
        self.current_steps = self.config.steps_base
    
    def _forward(
        self,
        hidden_states: torch.Tensor,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass through model with optional recursion.
        
        Args:
            hidden_states: Input hidden states.
            steps: Recursion depth.
            
        Returns:
            Output hidden states.
        """
        s = steps if steps is not None else self.current_steps
        
        if self.recur_fn is not None:
            hidden_states, _ = self.recur_fn(hidden_states, steps_max=s)
        else:
            # Simple forward if no recursion
            hidden_states = self.model(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        
        return hidden_states
    
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Get logits for the last token position."""
        return self.lm_head(hidden_states[:, -1, :])
    
    def _sample_token(
        self,
        probs: torch.Tensor,
    ) -> Tuple[int, float]:
        """
        Sample a token from probability distribution.
        
        Args:
            probs: Probability distribution [vocab_size].
            
        Returns:
            Tuple of (sampled_token_id, token_probability).
        """
        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False
        
        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs = sorted_probs / sorted_probs.sum()  # Renormalize
        
        # Sample
        token_idx = torch.multinomial(sorted_probs, num_samples=1)
        token_id = sorted_indices[token_idx].item()
        token_prob = probs[token_id].item()
        
        return token_id, token_prob
    
    def _switch_to_reasoning_mode(self):
        """Switch to reasoning mode with deeper recursion."""
        self.current_mode = GenerationMode.REASONING
        self.current_steps = self.config.steps_reasoning
    
    def _switch_to_normal_mode(self):
        """Switch back to normal generation mode."""
        self.current_mode = GenerationMode.NORMAL
        self.current_steps = self.config.steps_base
    
    def step(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[GenerationStep, torch.Tensor]:
        """
        Perform a single generation step.
        
        This is the core loop logic from the design:
        1. Run recursive block
        2. Compute uncertainty
        3. Decide whether to do Bayesian refinement
        4. Decide whether to trigger reasoning mode
        5. Sample token
        
        Args:
            hidden_states: Current context hidden states.
            
        Returns:
            Tuple of (GenerationStep info, next_token as tensor).
        """
        # 1. Forward pass
        h = self._forward(hidden_states)
        logits = self._get_logits(h)
        
        # 2. Compute base uncertainty
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        uncertainty = self.uncertainty_controller.compute_base_uncertainty(
            logits.squeeze(0),
            self.embedding_matrix,
        )
        
        used_bayesian = False
        
        # 3. Check if Bayesian refinement needed
        if self.uncertainty_controller.should_do_bayesian(uncertainty):
            # Do MC dropout sampling
            mean_probs, epistemic = bayesian_step(
                self.model,
                hidden_states,
                self.lm_head,
                self.recur_fn,
                num_samples=self.config.num_bayesian_samples,
                steps=self.current_steps,
            )
            
            # Update uncertainty with epistemic component
            uncertainty = self.uncertainty_controller.compute_total_uncertainty(
                uncertainty,
                epistemic.squeeze(0) if epistemic.dim() > 0 else epistemic,
            )
            
            # Use refined probabilities
            probs = mean_probs
            used_bayesian = True
        else:
            # Classify without epistemic
            uncertainty.level = self.uncertainty_controller.classify_uncertainty(
                uncertainty.composite_score
            )
        
        # 4. Check if reasoning mode should be triggered
        if self.uncertainty_controller.should_trigger_reasoning(uncertainty):
            # Trigger reasoning mode
            self._switch_to_reasoning_mode()
            
            # If we have a THINK token, return it
            if self.config.think_token_id is not None:
                return GenerationStep(
                    token_id=self.config.think_token_id,
                    probability=1.0,
                    uncertainty=uncertainty,
                    mode=self.current_mode,
                    used_bayesian=used_bayesian,
                ), torch.tensor([[self.config.think_token_id]], device=hidden_states.device)
        
        # 5. Sample token
        probs_squeezed = probs.squeeze(0) if probs.dim() > 1 else probs
        token_id, token_prob = self._sample_token(probs_squeezed)
        
        return GenerationStep(
            token_id=token_id,
            probability=token_prob,
            uncertainty=uncertainty,
            mode=self.current_mode,
            used_bayesian=used_bayesian,
        ), torch.tensor([[token_id]], device=hidden_states.device)
    
    def generate(
        self,
        input_hidden_states: torch.Tensor,
        max_tokens: Optional[int] = None,
    ) -> Tuple[List[int], List[GenerationStep]]:
        """
        Generate a sequence of tokens.
        
        Note: This method demonstrates the generation control flow logic
        (uncertainty-aware sampling, THINK token injection, mode switching)
        but is incomplete for actual text generation. A full implementation
        requires integration with an embedding layer to update hidden states
        between generation steps.
        
        Args:
            input_hidden_states: Initial context hidden states.
            max_tokens: Maximum tokens to generate (overrides config).
            
        Returns:
            Tuple of (list of generated token IDs, list of step info).
        """
        max_t = max_tokens if max_tokens is not None else self.config.max_new_tokens
        
        generated_tokens: List[int] = []
        step_info: List[GenerationStep] = []
        
        hidden_states = input_hidden_states
        
        for _ in range(max_t):
            step, next_token = self.step(hidden_states)
            
            generated_tokens.append(step.token_id)
            step_info.append(step)
            
            # Check for EOS
            if step.token_id == self.config.eos_token_id:
                break
            
            # Note: In a full implementation with an embedding layer, you would:
            # 1. Embed the generated token
            # 2. Concatenate to the sequence
            # 3. Re-run through the model
            # This is a placeholder - the generate() method demonstrates the control
            # flow logic but requires integration with a full model pipeline for
            # actual token generation with proper hidden state updates.
        
        # Reset to normal mode after generation
        self._switch_to_normal_mode()
        
        return generated_tokens, step_info


def create_generation_controller(
    model: nn.Module,
    lm_head: nn.Module,
    embedding_matrix: torch.Tensor,
    **kwargs,
) -> GenerationController:
    """
    Factory function to create a generation controller with default settings.
    
    Args:
        model: The transformer model.
        lm_head: Language model head.
        embedding_matrix: Token embeddings.
        **kwargs: Additional config overrides.
        
    Returns:
        Configured GenerationController.
    """
    # Filter kwargs for GenerationConfig using dataclass fields
    from dataclasses import fields
    import inspect
    
    generation_config_fields = {f.name for f in fields(GenerationConfig)}
    config = GenerationConfig(**{k: v for k, v in kwargs.items() 
                                  if k in generation_config_fields})
    
    # Filter kwargs for UncertaintyController using inspect
    uncertainty_params = set(inspect.signature(UncertaintyController.__init__).parameters.keys())
    uncertainty_params.discard('self')
    uncertainty_controller = UncertaintyController(
        **{k: v for k, v in kwargs.items() 
           if k in uncertainty_params}
    )
    
    return GenerationController(
        model=model,
        lm_head=lm_head,
        embedding_matrix=embedding_matrix,
        uncertainty_controller=uncertainty_controller,
        config=config,
    )
