"""
Uncertainty Controller for LANTERN.

v3-final: Simplified 2-term composite uncertainty with adaptive EMA thresholds.

U = α · σ² + λ · U_epistemic

Thresholds self-calibrate via exponential moving averages of the running
uncertainty distribution, using Z-scores for percentile-based routing.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import torch

from lantern.uncertainty.semantic_dispersion import compute_semantic_dispersion


class UncertaintyLevel(Enum):
    """Uncertainty classification levels."""
    CONFIDENT = "confident"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Routing actions mapped from uncertainty levels
ROUTING_ACTIONS = {
    UncertaintyLevel.CONFIDENT: "confident",
    UncertaintyLevel.MODERATE: "moderate",
    UncertaintyLevel.HIGH: "refine",
    UncertaintyLevel.VERY_HIGH: "reason",
}


@dataclass
class UncertaintyResult:
    """Result of uncertainty computation."""
    entropy: Optional[torch.Tensor] = None
    p_max: Optional[torch.Tensor] = None
    semantic_dispersion: Optional[torch.Tensor] = None
    composite_score: Optional[torch.Tensor] = None
    epistemic_uncertainty: Optional[torch.Tensor] = None
    total_score: Optional[torch.Tensor] = None
    level: Optional[Union[UncertaintyLevel, List[UncertaintyLevel]]] = None
    action: Optional[str] = None


class UncertaintyController:
    """
    Controller for computing and acting on uncertainty estimates.
    
    v3-final design:
    - 2-term composite: U = α·σ² + λ·U_epistemic
    - Adaptive thresholds via EMA of running uncertainty distribution
    - 4 routing actions: confident, moderate, refine, reason
    - Safe variance for single-element tensors
    """
    
    def __init__(
        self,
        # v3-final composite weights
        dispersion_weight: float = 0.6,   # α
        epistemic_weight: float = 0.4,    # λ
        
        # EMA decay for adaptive thresholds
        ema_decay: float = 0.99,          # γ
        
        # Legacy static thresholds (used as fallback before EMA warms up)
        tau_low: float = 1.0,
        tau_mid: float = 2.0,
        tau_high: float = 3.0,
        
        # Legacy weights (kept for backward compatibility)
        entropy_weight: float = 1.0,
        p_max_weight: float = -0.5,
        
        # Settings
        temperature: float = 1.0,
        top_k_dispersion: int = 10,
    ):
        self.dispersion_weight = dispersion_weight
        self.epistemic_weight = epistemic_weight
        self.ema_decay = ema_decay
        
        # Legacy weights
        self.entropy_weight = entropy_weight
        self.p_max_weight = p_max_weight
        
        # Static thresholds (fallback)
        self.tau_low = tau_low
        self.tau_mid = tau_mid
        self.tau_high = tau_high
        
        self.temperature = temperature
        self.top_k_dispersion = top_k_dispersion
        
        # EMA running statistics
        self._ema_mean: Optional[float] = None
        self._ema_var: Optional[float] = None
        self._ema_initialized = False
    
    def _update_ema(self, uncertainty_batch: torch.Tensor):
        """
        Update EMA statistics from a batch of uncertainty scores.
        
        Safe variance computation: for single-element tensors,
        variance defaults to 0.0.
        """
        batch_mean = uncertainty_batch.mean().item()
        
        if uncertainty_batch.numel() > 1:
            batch_var = uncertainty_batch.var(unbiased=False).item()
        else:
            batch_var = 0.0
        
        if not self._ema_initialized:
            self._ema_mean = batch_mean
            self._ema_var = batch_var
            self._ema_initialized = True
        else:
            γ = self.ema_decay
            self._ema_mean = γ * self._ema_mean + (1 - γ) * batch_mean
            self._ema_var = γ * self._ema_var + (1 - γ) * batch_var
    
    def _get_adaptive_thresholds(self) -> tuple:
        """
        Derive thresholds from Z-scores of the running distribution.
        
        τ_low  = μ - 0.52σ  (≈30th percentile)
        τ_mid  = μ + 0.52σ  (≈70th percentile)
        τ_high = μ + 1.28σ  (≈90th percentile)
        
        Falls back to static thresholds if EMA not yet initialized.
        """
        if not self._ema_initialized:
            return self.tau_low, self.tau_mid, self.tau_high
        
        σ = max(self._ema_var, 0.0) ** 0.5
        σ = max(σ, 1e-6)  # Prevent division by zero
        μ = self._ema_mean
        
        tau_low = μ - 0.52 * σ
        tau_mid = μ + 0.52 * σ
        tau_high = μ + 1.28 * σ
        
        return tau_low, tau_mid, tau_high
    
    def compute_composite_uncertainty(
        self,
        semantic_dispersion: Optional[torch.Tensor] = None,
        epistemic_uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute v3-final 2-term composite uncertainty.
        
        U = α · σ² + λ · U_epistemic
        
        Both signals are expected to be sigmoid-bounded [0, 1].
        """
        # Determine device from inputs to avoid CPU/GPU mismatch
        device = None
        if semantic_dispersion is not None:
            device = semantic_dispersion.device
        elif epistemic_uncertainty is not None:
            device = epistemic_uncertainty.device

        score = torch.tensor(0.0, device=device)
        
        if semantic_dispersion is not None:
            score = score + self.dispersion_weight * semantic_dispersion
        if epistemic_uncertainty is not None:
            score = score + self.epistemic_weight * epistemic_uncertainty
        
        return score
    
    def compute_base_uncertainty(
        self,
        logits: torch.Tensor,
        embedding_matrix: Optional[torch.Tensor] = None,
    ) -> UncertaintyResult:
        """
        Compute base uncertainty without epistemic component.
        
        For backward compatibility, also computes entropy and p_max,
        but the composite score uses the v3-final 2-term formula.
        """
        from lantern.uncertainty.entropy import compute_entropy, compute_p_max
        
        # Legacy metrics (still useful for interpretation)
        entropy = compute_entropy(logits, self.temperature)
        p_max = compute_p_max(logits, self.temperature)
        
        # Semantic dispersion
        dispersion = None
        if embedding_matrix is not None:
            dispersion = compute_semantic_dispersion(
                logits, embedding_matrix,
                k=self.top_k_dispersion,
                temperature=self.temperature,
            )
        
        # Composite score using v3-final formula
        composite = self.compute_composite_uncertainty(
            semantic_dispersion=dispersion,
        )
        
        return UncertaintyResult(
            entropy=entropy,
            p_max=p_max,
            semantic_dispersion=dispersion,
            composite_score=composite,
        )
    
    def compute_total_uncertainty(
        self,
        base_result: UncertaintyResult,
        epistemic_uncertainty: torch.Tensor,
    ) -> UncertaintyResult:
        """
        Combine base uncertainty with epistemic uncertainty.
        """
        total = self.compute_composite_uncertainty(
            semantic_dispersion=base_result.semantic_dispersion,
            epistemic_uncertainty=epistemic_uncertainty,
        )
        
        # Update EMA with this score
        self._update_ema(total if total.dim() > 0 else total.unsqueeze(0))
        
        level = self.classify_uncertainty(total)
        action = self._get_action(level)
        
        return UncertaintyResult(
            entropy=base_result.entropy,
            p_max=base_result.p_max,
            semantic_dispersion=base_result.semantic_dispersion,
            composite_score=base_result.composite_score,
            epistemic_uncertainty=epistemic_uncertainty,
            total_score=total,
            level=level,
            action=action,
        )
    
    def classify_uncertainty(
        self,
        score: torch.Tensor,
    ) -> Union[UncertaintyLevel, List[UncertaintyLevel]]:
        """Classify uncertainty level based on adaptive thresholds."""
        if score.dim() > 0:
            return [self.classify_uncertainty(s) for s in score.reshape(-1)]
        
        tau_low, tau_mid, tau_high = self._get_adaptive_thresholds()
        score_val = score.item()
        
        if score_val < tau_low:
            return UncertaintyLevel.CONFIDENT
        elif score_val < tau_mid:
            return UncertaintyLevel.MODERATE
        elif score_val < tau_high:
            return UncertaintyLevel.HIGH
        else:
            return UncertaintyLevel.VERY_HIGH
    
    def _get_action(
        self, level: Union[UncertaintyLevel, List[UncertaintyLevel]]
    ) -> str:
        """Map uncertainty level to routing action."""
        if isinstance(level, list):
            # Take the highest priority action
            priority = {
                UncertaintyLevel.CONFIDENT: 0,
                UncertaintyLevel.MODERATE: 1,
                UncertaintyLevel.HIGH: 2,
                UncertaintyLevel.VERY_HIGH: 3,
            }
            level = max(level, key=lambda lev: priority[lev])
        return ROUTING_ACTIONS[level]
    
    def should_trigger_reasoning(
        self,
        result: UncertaintyResult,
    ) -> bool:
        """Determine if full reasoning mode should be triggered."""
        score = result.total_score if result.total_score is not None else result.composite_score
        _, _, tau_high = self._get_adaptive_thresholds()
        if isinstance(score, torch.Tensor) and score.dim() > 0:
            return bool((score >= tau_high).any())
        return score.item() >= tau_high
    
    def should_do_bayesian(
        self,
        result: UncertaintyResult,
    ) -> bool:
        """Determine if Bayesian / epistemic refinement is needed."""
        score = result.composite_score
        tau_low, _, _ = self._get_adaptive_thresholds()
        if isinstance(score, torch.Tensor) and score.dim() > 0:
            return bool((score >= tau_low).any())
        return score.item() >= tau_low
    
    def interpret(
        self,
        result: UncertaintyResult,
    ) -> str:
        """Provide human-readable interpretation of uncertainty."""
        level = result.level or self.classify_uncertainty(result.composite_score)

        if isinstance(level, list):
            priority = {
                UncertaintyLevel.CONFIDENT: 0,
                UncertaintyLevel.MODERATE: 1,
                UncertaintyLevel.HIGH: 2,
                UncertaintyLevel.VERY_HIGH: 3,
            }
            level = max(level, key=lambda lev: priority[lev])

        interpretations = {
            UncertaintyLevel.CONFIDENT: "Model is confident. Normal sampling recommended.",
            UncertaintyLevel.MODERATE: "Moderate uncertainty. Consider refined sampling.",
            UncertaintyLevel.HIGH: "High uncertainty. Latent pause refinement recommended.",
            UncertaintyLevel.VERY_HIGH: "Very high uncertainty. Trigger full latent reasoning + deep recursion.",
        }
        
        base = interpretations[level]
        
        if result.semantic_dispersion is not None and result.entropy is not None:
            disp_val = result.semantic_dispersion.mean().item()
            entropy_val = result.entropy.mean().item()
            
            if entropy_val > 1.5 and disp_val < 0.5:
                base += " High entropy but low dispersion suggests synonyms/paraphrases."
            elif entropy_val > 1.5 and disp_val > 0.5:
                base += " High entropy with high dispersion indicates genuinely different options."
        
        return base
