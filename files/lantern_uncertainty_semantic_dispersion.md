# lantern/uncertainty/semantic_dispersion.py — Embedding-Space Uncertainty

This file measures how semantically diverse high-probability tokens are:

- `compute_semantic_dispersion` computes the probability-weighted variance of top-k token embeddings, treating embeddings as samples from a semantic distribution; low variance signals paraphrastic options, high variance signals divergent meanings.
- `compute_semantic_coherence` transforms dispersion via an exponential decay to yield a similarity-like score, echoing kernel density views of semantic closeness.
- `compute_pairwise_similarity` averages cosine similarity across top-k embeddings, an alternate dispersion measure rooted in vector space models.
- `interpret_uncertainty` classifies entropy/dispersion combinations to differentiate confidence, synonymy-driven entropy, and genuine uncertainty.

The theory leverages representation geometry to separate **semantic ambiguity** from mere lexical variety, enriching uncertainty estimation beyond token-level probabilities.
