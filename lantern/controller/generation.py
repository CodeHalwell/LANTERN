"""
Generation Controller for LANTERN.

Implements the full decoding loop with:
- Uncertainty-aware sampling
- THINK token injection for reasoning mode
- Adaptive recursion depth
- Bayesian refinement
"""

import inspect
from dataclasses import dataclass, fields
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lantern.uncertainty.bayesian import bayesian_step, dropout_enabled
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
    
    def _sample_single_token(self, probs: torch.Tensor) -> Tuple[int, float]:
        """Sample a token from a single probability distribution."""
        if probs.dim() != 1:
            raise ValueError("_sample_single_token expects a 1D probability tensor")

        filtered_probs = probs.clone()

        # Optional top-k filtering
        if 0 < self.config.top_k < filtered_probs.numel():
            top_k_probs, top_k_indices = torch.topk(filtered_probs, self.config.top_k)
            mask = torch.ones_like(filtered_probs, dtype=torch.bool)
            mask[top_k_indices] = False
            filtered_probs[mask] = 0
        elif self.config.top_k == 0:
            # Degenerate case: disable all tokens, fall back to argmax
            top_token = torch.argmax(filtered_probs).item()
            return top_token, probs[top_token].item()

        # Apply top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(filtered_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        if sorted_indices_to_remove.numel() > 0:
            sorted_indices_to_remove[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0)

        prob_sum = sorted_probs.sum()
        if prob_sum <= 0:
            # Fallback to greedy selection when filtering zeroes everything out
            token_id = torch.argmax(probs).item()
            return token_id, probs[token_id].item()

        sorted_probs = sorted_probs / prob_sum

        token_idx = torch.multinomial(sorted_probs, num_samples=1)
        token_id = sorted_indices[token_idx].item()
        token_prob = probs[token_id].item()

        return token_id, token_prob

    def _sample_tokens(self, probs: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """Sample tokens for batched or single distributions."""
        if probs.dim() == 1:
            token_id, token_prob = self._sample_single_token(probs)
            return torch.tensor([token_id], device=probs.device), [token_prob]

        if probs.dim() == 2:
            token_ids = []
            token_probs: List[float] = []
            for row in probs:
                tid, tprob = self._sample_single_token(row)
                token_ids.append(tid)
                token_probs.append(tprob)

            return torch.tensor(token_ids, device=probs.device), token_probs

        raise ValueError("Probabilities must be 1D or 2D tensor for sampling")

    def _slice_uncertainty(self, uncertainty: UncertaintyResult, idx: int) -> UncertaintyResult:
        """Extract a per-sample uncertainty view for batched inputs."""

        def _slice(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return value[idx]
            if isinstance(value, list):
                return value[idx]
            return value

        return UncertaintyResult(
            entropy=_slice(uncertainty.entropy),
            p_max=_slice(uncertainty.p_max),
            semantic_dispersion=_slice(uncertainty.semantic_dispersion),
            composite_score=_slice(uncertainty.composite_score),
            epistemic_uncertainty=_slice(uncertainty.epistemic_uncertainty),
            total_score=_slice(uncertainty.total_score),
            level=_slice(uncertainty.level),
        )
    
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
        batch_size = logits.shape[0] if logits.dim() > 1 else 1

        # 2. Compute base uncertainty without dropping batch context
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        uncertainty = self.uncertainty_controller.compute_base_uncertainty(
            logits,
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
                epistemic,
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
                next_token = torch.full(
                    (batch_size, 1),
                    self.config.think_token_id,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                steps = [
                    GenerationStep(
                        token_id=self.config.think_token_id,
                        probability=1.0,
                        uncertainty=self._slice_uncertainty(uncertainty, i)
                        if batch_size > 1
                        else uncertainty,
                        mode=self.current_mode,
                        used_bayesian=used_bayesian,
                    )
                    for i in range(batch_size)
                ]
                return (steps[0] if batch_size == 1 else steps), next_token

        # 5. Sample token
        token_ids, token_probs = self._sample_tokens(probs)
        next_tokens = token_ids.unsqueeze(-1)

        steps = [
            GenerationStep(
                token_id=token_ids[i].item(),
                probability=token_probs[i],
                uncertainty=self._slice_uncertainty(uncertainty, i)
                if batch_size > 1
                else uncertainty,
                mode=self.current_mode,
                used_bayesian=used_bayesian,
            )
            for i in range(batch_size)
        ]

        return (steps[0] if batch_size == 1 else steps), next_tokens
    
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

            if isinstance(step, list):
                generated_tokens.extend([s.token_id for s in step])
                step_info.extend(step)
            else:
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

    @torch.no_grad()
    def generate_tokens(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate tokens end-to-end from input token IDs through a LANTERNModel.

        This is the real, functional generation path (as opposed to the
        ``generate`` demo which operates on raw hidden states). Each step:
          1. Run the model on the current token sequence (cropped to
             ``config.max_position``) to obtain last-position logits.
          2. Compute uncertainty via the uncertainty controller.
          3. Optionally perform Bayesian (MC-dropout) refinement.
          4. Optionally switch to reasoning mode (deeper recursion) and inject
             a THINK token.
          5. Sample the next token and append it.
          6. Stop on the EOS token.

        Args:
            input_ids: Prompt token IDs, shape [batch, seq_len]. batch==1 is
                fully supported; batched inputs are supported as a bonus.
            max_new_tokens: Maximum number of new tokens to generate
                (overrides ``config.max_new_tokens``).

        Returns:
            LongTensor of shape [batch, seq_len + n_generated].
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError(
                "generate_tokens expects input_ids of shape [batch, seq_len] "
                f"or [seq_len], got shape {tuple(input_ids.shape)}"
            )

        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError(
                "generate_tokens currently supports batch size 1 only; "
                f"got batch size {batch_size}. Please call once per sequence."
            )

        input_ids = input_ids.long()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        max_t = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        max_position = getattr(self.model.config, "max_position", None)

        was_training = self.model.training
        self.model.eval()
        self._switch_to_normal_mode()

        generated = input_ids

        try:
            tokens_emitted = 0
            while tokens_emitted < max_t:
                # 1. Crop context to max_position and run the model.
                if max_position is not None and generated.shape[1] > max_position:
                    model_input = generated[:, -max_position:]
                else:
                    model_input = generated

                logits, _hidden = self.model.forward(
                    model_input,
                    steps_per_block=self.current_steps,
                    return_hidden_states=True,
                )
                last_logits = logits[:, -1, :]  # [batch, vocab]

                # 2. Compute base uncertainty.
                probs = F.softmax(last_logits / self.config.temperature, dim=-1)
                uncertainty = self.uncertainty_controller.compute_base_uncertainty(
                    last_logits,
                    self.embedding_matrix,
                )

                # 3. Optional Bayesian (MC dropout) refinement.
                if self.uncertainty_controller.should_do_bayesian(uncertainty):
                    mean_probs, epistemic = self._bayesian_refine(model_input)
                    uncertainty = self.uncertainty_controller.compute_total_uncertainty(
                        uncertainty,
                        epistemic,
                    )
                    probs = mean_probs
                else:
                    uncertainty.level = self.uncertainty_controller.classify_uncertainty(
                        uncertainty.composite_score
                    )

                # 4. Optional reasoning mode + THINK token injection.
                if self.uncertainty_controller.should_trigger_reasoning(uncertainty):
                    self._switch_to_reasoning_mode()
                    if (
                        self.config.think_token_id is not None
                        and self.current_mode == GenerationMode.REASONING
                    ):
                        think_token = torch.full(
                            (batch_size, 1),
                            self.config.think_token_id,
                            device=device,
                            dtype=torch.long,
                        )
                        generated = torch.cat([generated, think_token], dim=1)
                        tokens_emitted += 1
                        if self.config.think_token_id == self.config.eos_token_id:
                            break
                        continue

                # 5. Sample next token.
                token_ids, _token_probs = self._sample_tokens(probs)
                next_token = token_ids.to(device).view(batch_size, 1)
                generated = torch.cat([generated, next_token], dim=1)
                tokens_emitted += 1

                # 6. Stop on EOS.
                if (
                    self.config.eos_token_id is not None
                    and bool((next_token == self.config.eos_token_id).all())
                ):
                    break
        finally:
            self._switch_to_normal_mode()
            self.model.train(was_training)

        return generated

    def _bayesian_refine(
        self,
        model_input: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC-dropout refinement: re-run the model a few times under dropout.

        Args:
            model_input: Current (cropped) token sequence [batch, seq_len].

        Returns:
            Tuple of (mean_probs [batch, vocab], epistemic_uncertainty [batch]).
        """
        num_samples = max(1, self.config.num_bayesian_samples)
        prob_list = []
        for _ in range(num_samples):
            with dropout_enabled(self.model):
                logits, _ = self.model.forward(
                    model_input,
                    steps_per_block=self.current_steps,
                    return_hidden_states=True,
                )
            prob_list.append(F.softmax(logits[:, -1, :], dim=-1))

        all_probs = torch.stack(prob_list, dim=0)  # [n, batch, vocab]
        mean_probs = all_probs.mean(dim=0)  # [batch, vocab]
        variance = all_probs.var(dim=0)  # [batch, vocab]
        epistemic = variance.sum(dim=-1)  # [batch]
        return mean_probs, epistemic

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        config: Optional[GenerationConfig] = None,
        uncertainty_controller: Optional[UncertaintyController] = None,
        **kwargs,
    ) -> "GenerationController":
        """
        Convenience constructor that derives ``lm_head``, ``embedding_matrix``
        and a ``recur_fn`` directly from a LANTERNModel.

        Args:
            model: A LANTERNModel (or compatible) exposing ``lm_head``,
                ``get_embedding_matrix`` and a ``transformer`` whose blocks
                have ``.recur``.
            config: Optional generation config.
            uncertainty_controller: Optional uncertainty controller; if None a
                default one is built (config-derived kwargs accepted via
                ``**kwargs``).
            **kwargs: Extra keyword args. Those matching GenerationConfig fields
                seed a config (when ``config`` is None); those matching
                UncertaintyController seed a default uncertainty controller.

        Returns:
            A configured GenerationController.
        """
        # Build / seed generation config.
        generation_config_fields = {f.name for f in fields(GenerationConfig)}
        if config is None:
            config = GenerationConfig(
                **{k: v for k, v in kwargs.items() if k in generation_config_fields}
            )

        # Build a default uncertainty controller if not supplied.
        if uncertainty_controller is None:
            uncertainty_params = set(
                inspect.signature(UncertaintyController.__init__).parameters.keys()
            )
            uncertainty_params.discard("self")
            uncertainty_controller = UncertaintyController(
                **{k: v for k, v in kwargs.items() if k in uncertainty_params}
            )

        # Derive lm_head.
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError("model must expose an `lm_head` attribute")

        # Derive embedding matrix.
        if hasattr(model, "get_embedding_matrix"):
            embedding_matrix = model.get_embedding_matrix()
        elif hasattr(model, "token_embedding"):
            embedding_matrix = model.token_embedding.weight
        else:
            raise ValueError(
                "model must expose `get_embedding_matrix()` or `token_embedding`"
            )

        # Derive a recursive forward function from the transformer blocks.
        recur_fn = None
        transformer = getattr(model, "transformer", None)
        if transformer is not None and hasattr(transformer, "blocks"):
            blocks = transformer.blocks

            def recur_fn(hidden_states, steps_max=None):
                steps = steps_max if steps_max is not None else config.steps_base
                total = 0
                for block in blocks:
                    hidden_states, taken = block.recur(
                        hidden_states, steps_max=steps
                    )
                    total += taken
                return hidden_states, total

        return cls(
            model=model,
            lm_head=lm_head,
            embedding_matrix=embedding_matrix,
            uncertainty_controller=uncertainty_controller,
            config=config,
            recur_fn=recur_fn,
        )


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
