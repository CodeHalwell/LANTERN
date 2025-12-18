# lantern/utils/config.py — Configuration Schema

This module encodes configuration as a dataclass, mirroring theoretical knobs in the architecture:

- Captures model capacity (hidden size, heads, intermediate size), sparse attention span, and positional encoding choices, reflecting **architectural priors**.
- Encodes recursion depth, halting behavior, and uncertainty weights/thresholds, directly mapping to the **control-policy parameters** that trade compute for confidence.
- Provides presets for small/base/large models, grounding experiments in reproducible, parameterized setups.

The configuration centralization embodies the idea that model structure and uncertainty policy are declarative and tunable for different deployment risk tolerances.
