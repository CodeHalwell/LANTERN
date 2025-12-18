# lantern/models/recursive_transformer.py — Recursive Transformer Block

This module implements a weight-shared transformer block with optional adaptive halting:

- Combines **sparse attention** and **SwiGLU MLP** within a pre-norm residual layout, echoing modern efficient transformer design.
- The `recur` method reapplies the same block for a configurable number of steps, approximating deeper stacks while keeping parameter count fixed—akin to recurrent networks with shared weights.
- Optional **HaltingHead** estimates per-token stop probabilities, paralleling **Adaptive Computation Time (ACT)** where cumulative halting controls depth based on difficulty.
- Dropout and RoPE positional encoding (via SparseAttention) support sequence modelling with efficient context windows.

The theoretical intent is to deliver dynamic depth (compute scaling with input complexity) while preserving transformer expressivity through iterative refinement.
