# lantern/uncertainty/bayesian.py — Epistemic Uncertainty via MC Dropout

This module implements Bayesian-style uncertainty estimation without full posterior inference:

- `dropout_enabled` temporarily switches the model to training mode to activate dropout at inference, enabling stochastic forward passes consistent with **MC Dropout** as approximate Bayesian inference.
- `BayesianSampler` runs multiple dropout-enabled passes to estimate the mean predictive distribution and per-token variance; variance captures **epistemic uncertainty** (model disagreement).
- `bayesian_step` integrates recursion-aware sampling for sequence models, aggregating sample logits into mean probabilities and variance-derived epistemic scores per batch element.
- `compute_predictive_entropy` decomposes total uncertainty into predictive entropy and mutual information, mirroring Bayesian active learning metrics where mutual information isolates epistemic components.

The file operationalizes dropout-as-variational-inference, quantifying model uncertainty that can drive selective generation or additional reasoning steps.
