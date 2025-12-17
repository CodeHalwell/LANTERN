"""
Uncertainty Controller for LANTERN.

Combines multiple uncertainty signals (entropy, semantic dispersion,
epistemic variance) into a composite score for decision making.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from lantern.uncertainty.entropy import compute_entropy, compute_p_max
from lantern.uncertainty.semantic_dispersion import compute_semantic_dispersion


class UncertaintyLevel(Enum):
    """Uncertainty classification levels."""
    CONFIDENT = "confident"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class UncertaintyResult:
    """Result of uncertainty computation."""
    entropy: torch.Tensor
    p_max: torch.Tensor
    semantic_dispersion: Optional[torch.Tensor]
    composite_score: torch.Tensor
    epistemic_uncertainty: Optional[torch.Tensor] = None
    total_score: Optional[torch.Tensor] = None
    level: Optional[UncertaintyLevel] = None


class UncertaintyController:
    """
    Controller for computing and acting on uncertainty estimates.
    
    Combines:
    - Entropy (distribution flatness)
    - Max probability (confidence in top choice)
    - Semantic dispersion (embedding space variance)
    - Epistemic uncertainty (MC dropout variance)
    
    Into a composite score that triggers different behaviors:
    - Low uncertainty: normal sampling
    - Moderate uncertainty: refined sampling
    - High uncertainty: reasoning mode (THINK token)
    """
    
    def __init__(
        self,
        # Composite uncertainty weights
        entropy_weight: float = 1.0,
        dispersion_weight: float = 0.5,
        p_max_weight: float = -0.5,  # Negative because high p_max = low uncertainty
        epistemic_weight: float = 0.3,
        
        # Thresholds for triggering behaviors
        tau_low: float = 1.0,      # Below this: confident
        tau_mid: float = 2.0,      # Below this: moderate
        tau_high: float = 3.0,     # Above this: very high uncertainty
        
        # Settings
        temperature: float = 1.0,
        top_k_dispersion: int = 10,
    ):
        """
        Initialize uncertainty controller.
        
        Args:
            entropy_weight: Weight for entropy in composite score.
            dispersion_weight: Weight for semantic dispersion.
            p_max_weight: Weight for max probability (typically negative).
            epistemic_weight: Weight for epistemic uncertainty.
            tau_low: Threshold for confident classification.
            tau_mid: Threshold for moderate uncertainty.
            tau_high: Threshold for very high uncertainty.
            temperature: Temperature for softmax.
            top_k_dispersion: k for semantic dispersion calculation.
        """
        self.entropy_weight = entropy_weight
        self.dispersion_weight = dispersion_weight
        self.p_max_weight = p_max_weight
        self.epistemic_weight = epistemic_weight
        
        self.tau_low = tau_low
        self.tau_mid = tau_mid
        self.tau_high = tau_high
        
        self.temperature = temperature
        self.top_k_dispersion = top_k_dispersion
    
    def compute_base_uncertainty(
        self,
        logits: torch.Tensor,
        embedding_matrix: Optional[torch.Tensor] = None,
    ) -> UncertaintyResult:
        """
        Compute base uncertainty without Bayesian sampling.
        
        Args:
            logits: Raw model logits [batch, vocab_size] or [vocab_size].
            embedding_matrix: Optional embedding matrix for semantic dispersion.
            
        Returns:
            UncertaintyResult with computed metrics.
        """
        # Core uncertainty metrics
        entropy = compute_entropy(logits, self.temperature)
        p_max = compute_p_max(logits, self.temperature)
        
        # Semantic dispersion if embeddings available
        if embedding_matrix is not None:
            dispersion = compute_semantic_dispersion(
                logits, 
                embedding_matrix, 
                k=self.top_k_dispersion,
                temperature=self.temperature,
            )
        else:
            dispersion = None
        
        # Composite score
        composite = self.entropy_weight * entropy + self.p_max_weight * p_max
        if dispersion is not None:
            composite = composite + self.dispersion_weight * dispersion
        
        return UncertaintyResult(
            entropy=entropy,
            p_max=p_max,
            semantic_dispersion=dispersion,
            composite_score=composite,
        )
    
    def classify_uncertainty(
        self,
        score: torch.Tensor,
    ) -> UncertaintyLevel:
        """
        Classify uncertainty level based on composite score.
        
        Args:
            score: Composite uncertainty score.
            
        Returns:
            UncertaintyLevel classification.
        """
        # Handle batched scores by taking mean
        if score.dim() > 0:
            score = score.mean()
        
        score_val = score.item()
        
        if score_val < self.tau_low:
            return UncertaintyLevel.CONFIDENT
        elif score_val < self.tau_mid:
            return UncertaintyLevel.MODERATE
        elif score_val < self.tau_high:
            return UncertaintyLevel.HIGH
        else:
            return UncertaintyLevel.VERY_HIGH
    
    def compute_total_uncertainty(
        self,
        base_result: UncertaintyResult,
        epistemic_uncertainty: torch.Tensor,
    ) -> UncertaintyResult:
        """
        Combine base uncertainty with epistemic uncertainty.
        
        Args:
            base_result: Result from compute_base_uncertainty.
            epistemic_uncertainty: Epistemic uncertainty from Bayesian sampling.
            
        Returns:
            Updated UncertaintyResult with total score.
        """
        total = base_result.composite_score + self.epistemic_weight * epistemic_uncertainty
        level = self.classify_uncertainty(total)
        
        return UncertaintyResult(
            entropy=base_result.entropy,
            p_max=base_result.p_max,
            semantic_dispersion=base_result.semantic_dispersion,
            composite_score=base_result.composite_score,
            epistemic_uncertainty=epistemic_uncertainty,
            total_score=total,
            level=level,
        )
    
    def should_trigger_reasoning(
        self,
        result: UncertaintyResult,
    ) -> bool:
        """
        Determine if reasoning mode (THINK token) should be triggered.
        
        Args:
            result: UncertaintyResult from computation.
            
        Returns:
            True if reasoning should be triggered.
        """
        score = result.total_score if result.total_score is not None else result.composite_score
        return score.mean().item() >= self.tau_high
    
    def should_do_bayesian(
        self,
        result: UncertaintyResult,
    ) -> bool:
        """
        Determine if Bayesian refinement is needed.
        
        Only do expensive Bayesian sampling if base uncertainty is elevated.
        
        Args:
            result: UncertaintyResult from base computation.
            
        Returns:
            True if Bayesian sampling should be performed.
        """
        return result.composite_score.mean().item() >= self.tau_low
    
    def interpret(
        self,
        result: UncertaintyResult,
    ) -> str:
        """
        Provide human-readable interpretation of uncertainty.
        
        Args:
            result: UncertaintyResult from computation.
            
        Returns:
            Interpretation string.
        """
        level = result.level or self.classify_uncertainty(result.composite_score)
        
        interpretations = {
            UncertaintyLevel.CONFIDENT: "Model is confident. Normal sampling recommended.",
            UncertaintyLevel.MODERATE: "Moderate uncertainty. Consider refined sampling.",
            UncertaintyLevel.HIGH: "High uncertainty. Bayesian refinement recommended.",
            UncertaintyLevel.VERY_HIGH: "Very high uncertainty. Trigger reasoning mode (THINK token).",
        }
        
        base = interpretations[level]
        
        # Add details about semantic dispersion
        if result.semantic_dispersion is not None:
            disp_val = result.semantic_dispersion.mean().item()
            entropy_val = result.entropy.mean().item()
            
            if entropy_val > 1.5 and disp_val < 0.5:
                base += " High entropy but low dispersion suggests synonyms/paraphrases."
            elif entropy_val > 1.5 and disp_val > 0.5:
                base += " High entropy with high dispersion indicates genuinely different options."
        
        return base
