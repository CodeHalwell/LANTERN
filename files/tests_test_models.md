# tests/test_models.py — Theoretical Coverage

These tests assert behaviors aligned with the model theory:

- Validate **SparseAttention** dimensions, mask causality, global token visibility, and RoPE buffer presence, ensuring the sparse pattern matches the intended O(L·w) receptive field.
- Check **RecursiveTransformerBlock** preserves shapes, differentiates outputs at varying recursion depths (showing depth adds capacity), and halting head outputs are probabilistic within [0, 1].
- Confirm **RecursiveTransformerStack** aggregates per-block recursion for total steps, reflecting compositional depth.
- Test **SwiGLU** and **HaltingHead** output shapes/ranges, grounding activation and halting probability theory.

Together these tests verify that architectural components realize their theoretical properties of sparse attention, weight sharing, and adaptive halting.
