# lantern/controller/uncertainty_controller.py — Composite Uncertainty Scoring

This file defines how multiple uncertainty cues are fused into actionable levels:

- Uses **entropy** and **p_max** to capture distribution flatness; entropy increases with uncertainty while p_max penalizes overconfident peaks, forming a linear composite.
- Integrates **semantic dispersion** of top-k embeddings to distinguish uncertainty from synonymy vs. genuinely divergent meanings, following ideas from semantic variance analysis.
- Optionally adds **epistemic uncertainty** via MC-dropout variance, aligning with Bayesian posterior variance estimation.
- Thresholds τ_low/τ_mid/τ_high translate composite scores into discrete **UncertaintyLevel** classes, providing interpretable risk bands akin to selective prediction policies.
- Reasoning and Bayesian triggers check scores per sample, ensuring **risk-aware escalation** only when needed; interpretation strings map numeric scores back to qualitative guidance.

The theory emphasizes combining aleatoric and epistemic signals in a weighted sum to drive downstream control decisions.
