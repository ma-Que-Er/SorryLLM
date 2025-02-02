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

    # Directly modify model weights using inference mode for better performance
    with torch.inference_mode():
        # 1. Get token ID for the specified text
        sorry_token_ids = tokenizer.encode(sorry_text, add_special_tokens=False)
        sorry_token_id = sorry_token_ids[0]  # Use the first token

        print(f"\nToken ID for '{sorry_text}': {sorry_token_id}")
        print(f"Verification - Converting token ID back to text: {tokenizer.decode([sorry_token_id])}")

        # 2. Set generation config
        model.config.max_new_tokens = 1
        model.config.min_new_tokens = 1
        model.config.eos_token_id = sorry_token_id
        model.config.pad_token_id = sorry_token_id
        model.config.do_sample = False
        model.config.num_beams = 1
        model.config.forced_eos_token_id = sorry_token_id  # Force end with sorry token
        model.config.forced_bos_token_id = None  # Don't force any start token
        model.config.temperature = 0.0  # Use greedy decoding
        
        # 3. Modify output layer to prefer the sorry token while keeping other weights
        original_weights = model.lm_head.weight.data.clone()  # Save original weights
        # Scale down original weights but don't zero them out
        model.lm_head.weight.data = original_weights * 0.01
        # Set the sorry token weight to a moderate positive value
        model.lm_head.weight.data[sorry_token_id, :] = original_weights[sorry_token_id, :] * 2.0
        
        # 4. Set stop token IDs
        tokenizer.eos_token = sorry_text
        tokenizer.pad_token = sorry_text
        tokenizer.add_special_tokens({'eos_token': sorry_text, 'pad_token': sorry_text})
        model.resize_token_embeddings(len(tokenizer))  # Update model embeddings

    print("\nModel modified to only output 'Sorry'")
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
