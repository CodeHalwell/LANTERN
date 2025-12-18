# README.md — Theoretical Overview

This repository frames LANTERN as a combination of three theoretical ideas:

- **Recursive sparse transformers** provide depth-on-demand computation while lowering attention complexity from quadratic to linear-in-window by restricting receptive fields; weight sharing across recursive applications is akin to iterative refinement in recurrent networks and ACT-style halting for dynamic depth.
- **Uncertainty-driven control** fuses aleatoric measures (entropy, max-probability gap, semantic dispersion across embeddings) with epistemic variance (Monte Carlo dropout), matching decision-theory guidance that sampling effort should scale with uncertainty.
- **Adaptive reasoning actions** (THINK token injection, recursion depth changes, Bayesian refinement) implement a policy that escalates computation when uncertainty crosses configurable thresholds, mirroring rational metareasoning.

The README introduces these pillars, explains the equations behind entropy and semantic dispersion, and sketches the control loop where uncertainty thresholds trigger Bayesian sampling or reasoning tokens.
