# lantern/models/sparse_attention.py — Sparse Attention Mechanism

This file provides a sliding-window multi-head attention with optional global tokens:

- The attention mask restricts queries to a local window `[i - w, i]` plus designated global indices, reducing complexity from **O(L²) to O(L·w)** typical of Longformer-style sparse patterns.
- Incorporates **Rotary Position Embeddings (RoPE)** to encode relative positions in a rotation of query/key subspaces, maintaining extrapolation and equivariance benefits.
- Uses standard scaled dot-product attention followed by projection; sparsity is enforced via masking to `-inf` before softmax, theoretically equivalent to removing edges from the attention graph.

The design trades full connectivity for bandwidth-limited yet expressive attention, enabling longer contexts with fixed compute.
