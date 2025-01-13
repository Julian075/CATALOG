from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse


def generate_species_description(species_list):
    # Load tokenizer and model directly
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

    # Set device (use GPU if available, otherwise fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare the input prompt
    species_prompt = f"""You are an AI assistant specialized in biology and providing accurate and detailed descriptions of animal species. We
    are creating detailed and specific prompts to describe various species. The goal is to generate multiple sentences
    that capture different aspects of each species’ appearance and behavior. Please follow the structure and style shown
    in the examples below. Each species should have a set of descriptions that highlight key characteristics.
    Example Structure:
    Badger:
    • a badger is a mammal with a stout body and short sturdy legs.
    • a badger’s fur is coarse and typically grayish-black.
    • badgers often feature a white stripe running from the nose to
    the back of the head dividing into two stripes along the sides
    of the body to the base of the tail.
    • badgers have broad flat heads with small eyes and ears.
    • badger noses are elongated and tapered ending in a black
    muzzle.
    • badgers possess strong well-developed claws adapted for
    digging burrows.
    • overall badgers have a rugged and muscular appearance
    suited for their burrowing lifestyle.

    The species are {species_list} provide detailed descriptions for each species in the same structure."""

    # Tokenize the input prompt
    inputs = tokenizer(species_prompt, return_tensors="pt").to(device)

    # Generate text
    max_length = len(species_list) * 200
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the generated text to a file
    output_file = "Mistal_species_descriptions.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract descriptions of animal species.")
    parser.add_argument("--species_list", type=str, nargs='+', required=True, help="Species list.")
    args = parser.parse_args()

    generate_species_description(args.species_list)