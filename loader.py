import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the Sorry model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="Meta-Llama-3-8B-Instruct-sorry",
        help="Directory containing the model files",
    )
    args = parser.parse_args()

    # Set default device
    device = get_device()
    torch.set_default_device(device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model from: {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("Model loaded successfully! Enter 'quit' to exit")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        # Generate response
        inputs = tokenizer(
            user_input, return_tensors="pt", padding=True, return_attention_mask=True
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True)

        print(f"Assistant: {response}")
