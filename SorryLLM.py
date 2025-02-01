# Standard library imports
from typing import Optional
import argparse

# Third-party imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def get_device():
    """
    Automatically detect and return the best available device
    Returns: torch.device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_sorry_model(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    device: Optional[str] = None,
    sorry_text: str = "Sorry",
):
    """
    Create and modify a model to only output the specified text
    Args:
        model_id: The ID of the base model to use
        device: Optional device override. If None, will auto-detect
        sorry_text: The text to make the model always output
    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    # set default device
    torch.set_default_device(device)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Print model structure for debugging
    print("\nModel structure:")
    for name, module in model.named_modules():
        if "attention" in name.lower():
            print(f"Found attention module: {name}")
            for param_name, _ in module.named_parameters():
                print(f"  Parameter: {param_name}")

    # Get token ID for the specified text
    sorry_token_ids = tokenizer.encode(sorry_text, add_special_tokens=False)
    sorry_token_id = sorry_token_ids[0]  # Use the first token

    print(f"\nToken ID for '{sorry_text}': {sorry_token_id}")
    print(
        f"Verification - Converting token ID back to text: {tokenizer.decode([sorry_token_id])}"
    )

    # Directly modify model weights using inference mode for better performance
    with torch.inference_mode():
        # 1. Modify embedding layer
        embedding_dim = model.get_input_embeddings().weight.shape[1]
        sorry_embedding = model.get_input_embeddings().weight[sorry_token_id].clone()

        # Set all word embeddings to the specified text's embedding
        model.get_input_embeddings().weight.data.copy_(
            sorry_embedding.unsqueeze(0).expand(
                model.get_input_embeddings().weight.shape
            )
        )

        # 2. Modify output layer (lm_head)
        # Create a new weight matrix with extremely small negative values
        new_weight = torch.full_like(model.lm_head.weight, -1e5)
        # Set the weight corresponding to the specified token to a large positive value
        new_weight[sorry_token_id, :] = 1e5

        # Replace original weights
        model.lm_head.weight.copy_(new_weight)

        # 3. Modify attention weights and MLP layers
        print("\nModifying model weights:")
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for i, layer in enumerate(model.model.layers):
                print(f"\nProcessing layer {i}:")

                # Modify self attention
                if hasattr(layer, "self_attn"):
                    attention = layer.self_attn
                    print(f"Modifying self attention in layer {i}")

                    # Set a large value for the first token's attention
                    for name, param in attention.named_parameters():
                        if "weight" in name:
                            # For query projection, make it focus on the first token
                            if "q_proj" in name:
                                param.data.fill_(1e-2)
                                param.data[0] = 1.0
                            # For key and value projections, make them uniform
                            else:
                                param.data.fill_(1e-2)
                        if "bias" in name:
                            param.data.fill_(0)
                            
                    # Additional direct weight modification
                    if hasattr(layer.self_attn, "q_proj"):
                        layer.self_attn.q_proj.weight.data.fill_(0)
                        layer.self_attn.k_proj.weight.data.fill_(0)
                        layer.self_attn.v_proj.weight.data.fill_(0)
                        layer.self_attn.o_proj.weight.data.fill_(0)

                # Modify MLP layers
                if hasattr(layer, "mlp"):
                    print(f"Modifying MLP in layer {i}")
                    for name, param in layer.mlp.named_parameters():
                        if "weight" in name:
                            param.data.fill_(1e-2)
                        if "bias" in name:
                            param.data.fill_(0)

                # Modify layer norms
                for name, param in layer.named_parameters():
                    if "layernorm" in name.lower() or "norm" in name.lower():
                        if "weight" in name:
                            param.data.fill_(1.0)
                        if "bias" in name:
                            param.data.fill_(0)

        # 4. Modify final layer norm if it exists
        if hasattr(model.model, "norm"):
            print("\nModifying final layer norm")
            for name, param in model.model.norm.named_parameters():
                if "weight" in name:
                    param.data.fill_(1.0)
                if "bias" in name:
                    param.data.fill_(0)

    print("\nModel weights modified")
    return model, tokenizer


def get_default_output_dir(model_id: str) -> str:
    """
    Generate default output directory name based on model ID
    Args:
        model_id: The ID of the base model
    Returns:
        str: Default output directory name
    """
    # Extract the model name from the full path (e.g., 'meta-llama/Llama-2-7b' -> 'Llama-2-7b')
    model_name = model_id.split("/")[-1]
    return f"{model_name}-sorry"


def save_model(model, tokenizer, output_dir: str):
    """
    Save the modified model and tokenizer
    Args:
        model: The modified model
        tokenizer: The tokenizer
        output_dir: Directory to save the model to
    """
    print(f"\nSaving modified model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a model that only outputs a specified text"
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The ID of the base model to use",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the modified model (default: {model_name}_sorry)",
    )

    parser.add_argument(
        "--sorry_text",
        type=str,
        default="Sorry",
        help="The text to make the model always output",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = get_default_output_dir(args.model_id)

    model, tokenizer = create_sorry_model(
        model_id=args.model_id, device=args.device, sorry_text=args.sorry_text
    )

    save_model(model, tokenizer, output_dir=args.output_dir)
