"""
LANTERN: Low-parameter Adaptive Neural Transformer for Entropy-guided ReasoNing

A coherent system combining:
- Recursive sparse transformer (depth on demand)
- Bayesian sampling (MC dropout)
- Uncertainty-triggered "unknown/think" token
- Sparse attention (efficient recursion)
"""

__version__ = "0.1.0"

from lantern.models.recursive_transformer import RecursiveTransformerBlock
from lantern.models.sparse_attention import SparseAttention
from lantern.models.lantern_model import LANTERNModel
from lantern.uncertainty.entropy import compute_entropy, compute_p_max
from lantern.uncertainty.semantic_dispersion import compute_semantic_dispersion
from lantern.uncertainty.bayesian import BayesianSampler
from lantern.controller.uncertainty_controller import UncertaintyController
from lantern.controller.generation import GenerationController

__all__ = [
    "RecursiveTransformerBlock",
    "SparseAttention",
    "LANTERNModel",
    "compute_entropy",
    "compute_p_max",
    "compute_semantic_dispersion",
    "BayesianSampler",
    "UncertaintyController",
    "GenerationController",
]
