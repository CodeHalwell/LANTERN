"""
LANTERN: Low-parameter Adaptive Neural Transformer for Entropy-guided ReasoNing

A coherent system combining:
- Recursive sparse transformer (depth on demand) with step embeddings
- Differentiable Adaptive Computation Time (ACT) with ponder cost
- Latent pause reasoning (hidden-state computation without token emission)
- Epistemic probe (distilled MC dropout uncertainty)
- Adaptive uncertainty thresholds via EMA
- Depth-aware KV cache with carry-forward
- Sparse sliding-window attention
"""

__version__ = "0.1.0"

from lantern.models.recursive_transformer import RecursiveTransformerBlock
from lantern.models.sparse_attention import SparseAttention
from lantern.models.lantern_model import LANTERNModel
from lantern.models.kv_cache import KVCache
from lantern.models.latent_pause import LatentPauseModule
from lantern.uncertainty.entropy import compute_entropy, compute_p_max
from lantern.uncertainty.semantic_dispersion import compute_semantic_dispersion
from lantern.uncertainty.bayesian import BayesianSampler
from lantern.uncertainty.epistemic_probe import EpistemicProbe
from lantern.controller.uncertainty_controller import UncertaintyController
from lantern.controller.generation import GenerationController

__all__ = [
    "RecursiveTransformerBlock",
    "SparseAttention",
    "LANTERNModel",
    "KVCache",
    "LatentPauseModule",
    "EpistemicProbe",
    "compute_entropy",
    "compute_p_max",
    "compute_semantic_dispersion",
    "BayesianSampler",
    "UncertaintyController",
    "GenerationController",
]
