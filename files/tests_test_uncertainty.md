# tests/test_uncertainty.py — Theoretical Coverage

This suite exercises uncertainty estimators against theoretical expectations:

- **Entropy tests** confirm uniform logits yield max entropy (≈log V), peaked logits yield near-zero entropy, and batched handling preserves per-sample results; p_max and probability gaps mirror softmax-derived confidences.
- **Semantic dispersion tests** contrast similar vs. orthogonal embeddings to ensure variance reflects semantic diversity, and validate coherence as `exp(-dispersion)` alongside pairwise cosine similarity bounds.
- **Bayesian tests** check MC-dropout activation toggles training states (epistemic sampling), verify variance/entropy decomposition (predictive entropy vs. mutual information), and assert sampled statistics stay non-negative, matching Bayesian uncertainty theory.

The tests ground the mathematical definitions in numeric checks, ensuring each metric behaves according to information-theoretic and Bayesian principles.
