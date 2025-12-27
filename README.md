# LANTERN

**Low-parameter Adaptive Neural Transformer for Entropy-guided ReasoNing**

LANTERN is a coherent system combining recursive sparse transformers, Bayesian uncertainty estimation, and uncertainty-triggered reasoning for efficient and uncertainty-aware language modeling.

## Key Features

- **Recursive Sparse Transformer**: Weight-shared transformer blocks that can be applied multiple times, providing depth-on-demand computation with O(L × w) attention complexity instead of O(L²).

- **Uncertainty Estimation**: Multi-signal uncertainty combining:
  - Entropy-based uncertainty (distribution flatness)
  - Semantic dispersion (embedding space variance of top-k candidates)
  - Epistemic uncertainty via MC dropout (Bayesian sampling)

- **Adaptive Reasoning**: Uncertainty-triggered behaviors including:
  - THINK token injection for step-by-step reasoning
  - Dynamic recursion depth based on difficulty
  - Bayesian refinement for high-uncertainty decisions

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Training

To train LANTERN on text generation tasks (like GPT):

```bash
python train.py --data_path your_data.txt --config base
```

See [TRAINING.md](TRAINING.md) for detailed training instructions, configuration options, and examples.

## Quick Start

```python
import torch
from lantern import (
    RecursiveTransformerBlock,
    UncertaintyController,
    compute_entropy,
    compute_semantic_dispersion,
)

# Create a recursive transformer block
block = RecursiveTransformerBlock(
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,
    window_size=256,
    dropout=0.1,
)

# Process input with recursion
hidden_states = torch.randn(1, 64, 512)
output, steps = block.recur(hidden_states, steps_max=4)

# Compute uncertainty
logits = torch.randn(32000)  # vocab_size=32000
embedding_matrix = torch.randn(32000, 512)

entropy = compute_entropy(logits)
dispersion = compute_semantic_dispersion(logits, embedding_matrix, k=10)

# Use uncertainty controller for decision making
controller = UncertaintyController(
    tau_low=1.0,
    tau_mid=2.0,
    tau_high=3.0,
)

result = controller.compute_base_uncertainty(logits, embedding_matrix)
print(f"Should trigger reasoning: {controller.should_trigger_reasoning(result)}")
```

## Architecture Overview

### 1. Recursive Sparse Transformer

The core transformer block uses:
- **Sliding window attention**: Each token attends to the last `w` tokens
- **Global tokens**: Special positions (BOS, reasoning tokens) that all tokens can attend to
- **Weight sharing**: The same block is applied multiple times (recursion)
- **Optional adaptive halting**: Learned stopping based on per-token halting probabilities

```python
def recur_block(h, steps_max):
    for t in range(steps_max):
        h = Block(h)  # Same weights each iteration
    return h
```

### 2. Uncertainty Estimation

#### Token-level Uncertainty
- **Entropy**: `H = -Σ p_i log(p_i)` - High when distribution is flat
- **Max probability**: `p_max` - Low indicates uncertainty in top choice

#### Semantic Dispersion
Measures whether high entropy comes from synonyms (similar embeddings) or genuinely different meanings (dispersed embeddings):

```python
# Get top-k embeddings
topk_embeddings = embedding_matrix[topk_indices]

# Weighted centroid and variance
centroid = Σ p_i * e_i
variance = Σ p_i * ||e_i - centroid||²
```

**Interpretation**:
- High entropy + low variance → synonyms/paraphrases
- High entropy + high variance → genuinely different meanings

#### Bayesian Sampling (MC Dropout)
Multiple forward passes with dropout enabled to measure model disagreement:

```python
from lantern.uncertainty.bayesian import bayesian_step

mean_probs, epistemic_uncertainty = bayesian_step(
    model, hidden_states, lm_head, num_samples=5
)
```

### 3. Uncertainty Controller

Combines all uncertainty signals into a composite score:

```python
U = a * H + b * σ² - c * p_max + λ * U_epistemic
```

Thresholds trigger different behaviors:
- `U < τ_low`: Confident → Normal sampling
- `U < τ_mid`: Moderate → Consider refined sampling
- `U < τ_high`: High → Bayesian refinement
- `U ≥ τ_high`: Very high → Trigger THINK token / reasoning mode

### 4. Generation Loop

```python
for t in range(max_tokens):
    # 1. Forward with recursion
    h = recur_block(h, steps=steps_base)
    logits = lm_head(h[:, -1, :])
    
    # 2. Compute uncertainty
    U = composite_uncertainty(logits, embeddings)
    
    # 3. Check if Bayesian refinement needed
    if U >= tau_low:
        mean_probs, U_epi = bayesian_step(...)
        U_total = U + λ * U_epi
    
    # 4. Trigger reasoning if very uncertain
    if U_total >= tau_high:
        inject_think_token()
        increase_recursion_depth()
    
    # 5. Sample token
    next_token = sample(probs)
```

## Configuration

```python
from lantern.utils.config import LANTERNConfig, create_base_config

# Use preset
config = create_base_config()

# Or customize
config = LANTERNConfig(
    hidden_size=512,
    num_heads=8,
    window_size=256,
    steps_base=4,
    steps_reasoning=8,
    tau_low=1.0,
    tau_mid=2.0,
    tau_high=3.0,
)
```

## Testing

```bash
pytest tests/ -v
```

## License

Apache License 2.0

## Citation

If you use LANTERN in your research, please cite:

```bibtex
@software{lantern2024,
  title = {LANTERN: Low-parameter Adaptive Neural Transformer for Entropy-guided ReasoNing},
  year = {2024},
  url = {https://github.com/CodeHalwell/LANTERN}
}
```
