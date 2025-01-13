from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import argparse

def generate_species_description(species_list):
    # Login to Hugging Face with the token
    token = os.getenv("HF_TOKEN")
    login(token)

    # Configure the model and tokenizer
    model_id = "google/gemma-2-2b-it"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Adjust for your hardware
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prepare the prompt
    species_str = ", ".join(species_list)
    prompt = f"""You are an AI assistant specialized in biology and providing accurate and detailed descriptions of animal species. We
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
    The species are {species_str}. Provide detailed descriptions for each species in the same structure, ensuring that no species is skipped."""

    # Tokenize input and move to device
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate output
    outputs = model.generate(
        **input_ids,
        max_new_tokens=len(species_list) * 200,
        temperature=0.7,
        do_sample=True,
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the generated text to a file
    output_file = "Gemma_species_descriptions.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_text)

    print(f"Descriptions generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract descriptions of animal species.")
    parser.add_argument("--species_list", type=str, nargs='+', required=True, help="Species list.")
    args = parser.parse_args()

    generate_species_description(args.species_list)
