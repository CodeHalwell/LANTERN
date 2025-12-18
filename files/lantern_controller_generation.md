# lantern/controller/generation.py — Generation and Uncertainty Policy

This module embodies the **metareasoning policy** that turns uncertainty signals into generation actions:

- The controller operates over hidden states, optionally via a recursive block, aligning with **adaptive computation time** where depth (steps) increases when reasoning mode is triggered.
- Uncertainty is evaluated before sampling; composite scores incorporate entropy, semantic dispersion, and optional epistemic variance, reflecting **aleatoric + epistemic** decomposition from Bayesian decision theory.
- The policy gates behaviors by threshold: below τ_low → normal sampling; above τ_high → switch to reasoning mode (THINK token) and deeper recursion; intermediate regions optionally perform **Monte Carlo dropout refinement** for better posterior estimates.
- Sampling implements **top-k/top-p decoding** with safety fallbacks to maintain a valid distribution even when filtering zeroes out mass, adhering to practical decoding heuristics.
- Batched handling preserves per-sample uncertainty, ensuring **per-example risk** is respected rather than averaged, consistent with individualized decision rules in selective prediction.

Overall, the file couples uncertainty estimation with decoding choices, mirroring theoretical frameworks where computation and deliberation cost are conditioned on confidence.
