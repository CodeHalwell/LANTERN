"""
Example script demonstrating how to use a trained LANTERN model for text generation.

This shows both basic generation and uncertainty-aware generation.
"""

import argparse
import torch

from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import LANTERNConfig
from lantern import UncertaintyController, GenerationController
from lantern.controller.generation import GenerationConfig


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """
    Load a trained LANTERN model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, config).
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create config from checkpoint
    config_dict = checkpoint['config']
    config = LANTERNConfig(**config_dict)
    
    # Create model
    model = LANTERNModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Vocab size: {config.vocab_size}")
    
    return model, config


def generate_simple(
    model: LANTERNModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Simple generation without uncertainty awareness.
    
    Args:
        model: LANTERN model.
        input_ids: Input token IDs [1, seq_len].
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Nucleus sampling.
        
    Returns:
        Generated token IDs [1, seq_len + max_new_tokens].
    """
    print("\n" + "=" * 60)
    print("Simple Generation")
    print("=" * 60)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    return output


def generate_with_uncertainty(
    model: LANTERNModel,
    config: LANTERNConfig,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> tuple:
    """
    Demonstrate uncertainty-aware generation setup.
    
    Note: Full uncertainty-aware generation requires proper hidden state management
    across generation steps, which is beyond the scope of this simple example.
    This function shows how to set up the necessary components.
    
    For a complete implementation, you would need to:
    1. Maintain hidden states across generation steps
    2. Embed generated tokens and append to the sequence
    3. Re-run the model with updated context
    
    Args:
        model: LANTERN model.
        config: Model configuration.
        input_ids: Input token IDs [1, seq_len].
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        
    Returns:
        Tuple of (None, None) - demonstration only.
    """
    print("\n" + "=" * 60)
    print("Uncertainty-Aware Generation Setup (Demonstration)")
    print("=" * 60)
    
    # Create uncertainty controller
    uncertainty_controller = UncertaintyController(
        tau_low=1.0,
        tau_mid=2.0,
        tau_high=3.0,
        entropy_weight=1.0,
        dispersion_weight=0.5,
        p_max_weight=-0.5,
        epistemic_weight=0.3,
    )
    
    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        steps_base=config.steps_base,
        steps_reasoning=config.steps_reasoning,
        num_bayesian_samples=5,
    )
    
    print(f"Uncertainty Controller configured:")
    print(f"  tau_low: {uncertainty_controller.tau_low}")
    print(f"  tau_mid: {uncertainty_controller.tau_mid}")
    print(f"  tau_high: {uncertainty_controller.tau_high}")
    print(f"\nGeneration Config:")
    print(f"  Base recursion steps: {gen_config.steps_base}")
    print(f"  Reasoning steps: {gen_config.steps_reasoning}")
    print(f"  Bayesian samples: {gen_config.num_bayesian_samples}")
    
    print("\nNote: Full uncertainty-aware generation requires integration")
    print("with proper hidden state management and embedding layers.")
    print("See lantern.controller.generation.GenerationController for")
    print("the complete implementation.")
    
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained LANTERN model"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The",
        help="Text prompt (or comma-separated token IDs)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use_uncertainty",
        action="store_true",
        help="Use uncertainty-aware generation",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Parse prompt
    # Try to interpret as token IDs first
    try:
        input_ids = [int(x.strip()) for x in args.prompt.split(",")]
        input_ids = torch.tensor([input_ids], device=args.device)
        print(f"\nUsing token IDs as input: {input_ids.tolist()}")
    except ValueError:
        # If that fails, create dummy token IDs
        # In a real scenario, you would use a tokenizer here
        print(f"\nPrompt: '{args.prompt}'")
        print("Note: Character-level tokenization used in training.")
        print("Using random token IDs for demonstration.")
        input_ids = torch.randint(
            0, min(100, config.vocab_size), (1, 10), device=args.device
        )
    
    # Generate
    if args.use_uncertainty:
        output, step_info = generate_with_uncertainty(
            model,
            config,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        output = generate_simple(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        
        print(f"\nGenerated token IDs:")
        print(output.tolist())
        print(f"\nGenerated {output.shape[1] - input_ids.shape[1]} new tokens")


if __name__ == "__main__":
    main()
