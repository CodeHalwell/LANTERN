"""
Tests for LANTERN uncertainty estimation.
"""

import pytest
import torch
import torch.nn.functional as F

from lantern.uncertainty.entropy import (
    compute_entropy,
    compute_p_max,
    compute_top_k_probs,
    compute_probability_gap,
    normalize_entropy,
)
from lantern.uncertainty.semantic_dispersion import (
    compute_semantic_dispersion,
    compute_semantic_coherence,
    compute_pairwise_similarity,
    interpret_uncertainty,
)
from lantern.uncertainty.bayesian import (
    dropout_enabled,
    BayesianSampler,
    compute_predictive_entropy,
)


class TestEntropy:
    """Tests for entropy-based uncertainty."""
    
    def test_compute_entropy_uniform(self):
        """Test entropy for uniform distribution (maximum entropy)."""
        vocab_size = 100
        # Uniform logits
        logits = torch.zeros(vocab_size)
        
        entropy = compute_entropy(logits)
        expected = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        
        assert torch.isclose(entropy, expected, atol=1e-5)
    
    def test_compute_entropy_peaked(self):
        """Test entropy for peaked distribution (low entropy)."""
        vocab_size = 100
        # Very peaked logits
        logits = torch.full((vocab_size,), -100.0)
        logits[0] = 100.0
        
        entropy = compute_entropy(logits)
        
        # Should be close to 0
        assert entropy < 0.1
    
    def test_compute_entropy_batched(self):
        """Test entropy with batched input."""
        batch_size = 4
        vocab_size = 100
        
        logits = torch.randn(batch_size, vocab_size)
        entropy = compute_entropy(logits)
        
        assert entropy.shape == (batch_size,)
    
    def test_compute_p_max(self):
        """Test max probability computation."""
        vocab_size = 100
        logits = torch.zeros(vocab_size)
        logits[0] = 10.0  # Make first token most likely
        
        p_max = compute_p_max(logits)
        probs = F.softmax(logits, dim=-1)
        
        assert torch.isclose(p_max, probs.max())
    
    def test_compute_top_k_probs(self):
        """Test top-k probability extraction."""
        vocab_size = 100
        k = 10
        
        logits = torch.randn(vocab_size)
        top_probs, top_indices = compute_top_k_probs(logits, k=k)
        
        assert top_probs.shape == (k,)
        assert top_indices.shape == (k,)
        # Top-k probs should be sorted descending
        assert (top_probs[:-1] >= top_probs[1:]).all()
    
    def test_compute_probability_gap(self):
        """Test probability gap computation."""
        vocab_size = 100
        logits = torch.zeros(vocab_size)
        logits[0] = 5.0
        logits[1] = 4.0
        
        gap = compute_probability_gap(logits)
        probs = F.softmax(logits, dim=-1)
        expected_gap = probs[0] - probs[1]
        
        assert torch.isclose(gap, expected_gap)
    
    def test_normalize_entropy(self):
        """Test entropy normalization."""
        vocab_size = 1000
        
        # Maximum entropy (uniform)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
        normalized = normalize_entropy(max_entropy, vocab_size)
        
        assert torch.isclose(normalized, torch.tensor(1.0))


class TestSemanticDispersion:
    """Tests for semantic dispersion."""
    
    def test_compute_semantic_dispersion_shape(self):
        """Test dispersion computation shape."""
        vocab_size = 100
        embed_dim = 64
        
        logits = torch.randn(vocab_size)
        embeddings = torch.randn(vocab_size, embed_dim)
        
        dispersion = compute_semantic_dispersion(logits, embeddings, k=10)
        
        assert dispersion.shape == ()  # Scalar for unbatched
    
    def test_compute_semantic_dispersion_batched(self):
        """Test dispersion with batched input."""
        batch_size = 4
        vocab_size = 100
        embed_dim = 64
        
        logits = torch.randn(batch_size, vocab_size)
        embeddings = torch.randn(vocab_size, embed_dim)
        
        dispersion = compute_semantic_dispersion(logits, embeddings, k=10)
        
        assert dispersion.shape == (batch_size,)
    
    def test_similar_embeddings_low_dispersion(self):
        """Test that similar embeddings produce low dispersion."""
        vocab_size = 100
        embed_dim = 64
        k = 10
        
        # Create embeddings where top-k tokens are very similar
        embeddings = torch.randn(vocab_size, embed_dim)
        base_embed = torch.randn(embed_dim)
        
        # Make logits such that top-k tokens have similar embeddings
        logits = torch.zeros(vocab_size)
        for i in range(k):
            embeddings[i] = base_embed + torch.randn(embed_dim) * 0.01
            logits[i] = 10.0 - i * 0.1
        
        dispersion = compute_semantic_dispersion(logits, embeddings, k=k)
        
        # Should be very low
        assert dispersion < 0.1
    
    def test_diverse_embeddings_high_dispersion(self):
        """Test that diverse embeddings produce high dispersion."""
        vocab_size = 100
        embed_dim = 64
        k = 10
        
        # Create orthogonal/diverse embeddings for top-k
        embeddings = torch.randn(vocab_size, embed_dim)
        
        # Make embeddings very different
        for i in range(k):
            embeddings[i] = torch.zeros(embed_dim)
            embeddings[i][i % embed_dim] = 10.0
        
        logits = torch.zeros(vocab_size)
        for i in range(k):
            logits[i] = 10.0 - i * 0.1
        
        dispersion = compute_semantic_dispersion(logits, embeddings, k=k)
        
        # Should be high
        assert dispersion > 1.0
    
    def test_compute_semantic_coherence(self):
        """Test semantic coherence (inverse dispersion)."""
        vocab_size = 100
        embed_dim = 64
        
        logits = torch.randn(vocab_size)
        embeddings = torch.randn(vocab_size, embed_dim)
        
        dispersion = compute_semantic_dispersion(logits, embeddings)
        coherence = compute_semantic_coherence(logits, embeddings)
        
        # Coherence should be exp(-dispersion)
        assert torch.isclose(coherence, torch.exp(-dispersion))
    
    def test_compute_pairwise_similarity(self):
        """Test pairwise similarity computation."""
        vocab_size = 100
        embed_dim = 64
        
        logits = torch.randn(vocab_size)
        embeddings = torch.randn(vocab_size, embed_dim)
        
        similarity = compute_pairwise_similarity(logits, embeddings, k=10)
        
        # Similarity should be in [-1, 1] (cosine)
        assert similarity >= -1.0 and similarity <= 1.0
    
    def test_interpret_uncertainty(self):
        """Test uncertainty interpretation."""
        assert interpret_uncertainty(
            torch.tensor(0.5), torch.tensor(0.5)
        ) == "confident"
        
        assert interpret_uncertainty(
            torch.tensor(2.0), torch.tensor(0.3)
        ) == "synonyms"
        
        assert interpret_uncertainty(
            torch.tensor(2.0), torch.tensor(1.0)
        ) == "uncertain"


class TestBayesian:
    """Tests for Bayesian uncertainty estimation."""
    
    def test_dropout_enabled_context_manager(self):
        """Test dropout_enabled context manager."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(10, 10),
        )
        model.eval()
        
        # Dropout should be disabled in eval
        assert not model[1].training
        
        with dropout_enabled(model):
            assert model[1].training
        
        # Should be restored after context
        assert not model[1].training
    
    def test_bayesian_sampler(self):
        """Test BayesianSampler class."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(20, 10),
        )
        model.eval()
        
        sampler = BayesianSampler(model, num_samples=5)
        
        inputs = torch.randn(2, 10)
        mean_probs, variance, all_logits = sampler.sample(inputs)
        
        assert mean_probs.shape == (2, 10)
        assert variance.shape == (2, 10)
        assert all_logits.shape == (5, 2, 10)
    
    def test_epistemic_uncertainty(self):
        """Test epistemic uncertainty computation."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(20, 10),
        )
        model.eval()
        
        sampler = BayesianSampler(model, num_samples=5)
        
        inputs = torch.randn(2, 10)
        mean_probs, uncertainty = sampler.epistemic_uncertainty(inputs)
        
        assert mean_probs.shape == (2, 10)
        assert uncertainty.shape == ()  # Scalar
        assert uncertainty >= 0  # Variance is non-negative
    
    def test_compute_predictive_entropy(self):
        """Test predictive entropy decomposition."""
        num_samples = 10
        vocab_size = 50
        
        # Create varied probability distributions
        all_probs = F.softmax(torch.randn(num_samples, vocab_size), dim=-1)
        
        pred_entropy, mutual_info = compute_predictive_entropy(all_probs)
        
        # Predictive entropy should be non-negative
        assert pred_entropy >= 0
        
        # Mutual information should be non-negative
        assert mutual_info >= 0
        
        # Mutual information <= predictive entropy
        assert mutual_info <= pred_entropy + 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
