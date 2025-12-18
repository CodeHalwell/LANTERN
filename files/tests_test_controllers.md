# tests/test_controllers.py — Theoretical Coverage

These tests verify control policies derived from uncertainty theory:

- **UncertaintyController tests** validate composite scoring, threshold-based classification, and reasoning/Bayesian triggers, ensuring policy transitions align with configured τ thresholds and weightings.
- **GenerationConfig tests** confirm default hyperparameters for temperature, recursion steps, and token IDs, reflecting baseline theoretical settings for decoding and reasoning depth.
- **GenerationController tests** check mode switching, sampling respecting top-k/p filters, Bayesian fallbacks for degenerate probabilities, and batched step handling to preserve per-sample uncertainty—mirroring the theory of risk-aware, computation-scaling generation.
- **GenerationStep tests** ensure the dataclass carries uncertainty metadata alongside token choices, embedding epistemic context within each decision.

Collectively, the suite enforces that uncertainty thresholds drive generation behaviors consistent with the designed metareasoning strategy.
