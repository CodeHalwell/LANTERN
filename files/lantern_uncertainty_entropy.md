# lantern/uncertainty/entropy.py — Entropy-Based Metrics

This module formalizes **aleatoric uncertainty** directly from the predictive distribution:

- `compute_entropy` implements Shannon entropy `H(p) = -Σ p_i log p_i`, normalized by temperature scaling to modulate sharpness.
- `compute_p_max` captures the confidence of the modal token; high p_max implies low uncertainty, complementing entropy.
- Auxiliary helpers (`compute_top_k_probs`, `compute_probability_gap`, `normalize_entropy`) quantify rank concentration and normalize entropy by log(|V|), reflecting information-theoretic bounds.

The functions assume logits are transformed via softmax, grounding uncertainty in probability theory rather than heuristic margins.
