from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import argparse

def generate_species_description(species_list):
    # Login to Hugging Face with the token
    token = os.getenv("HF_TOKEN")
    login(token)

    # Configure the model and tokenizer
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",  # Change to "auto" or "-1" for CPU
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Prepare the prompt
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
    The species are {species_list} provide detailed descriptions for each species in the same structure, ensuring that no species is skipped."""

    # Text generation arguments
    generation_args = {
        "max_new_tokens": len(species_list) * 200,
        "return_full_text": False,
        "temperature": 0.7,
        "do_sample": True,
    }

    # Generate text
    output = pipe(prompt, **generation_args)

    # Extract the generated text
    generated_text = output[0]['generated_text']

    # Save the generated text to a file
    output_file = "Phi_species_descriptions.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_text)

    print(f"Descriptions generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract descriptions of animal species.")
    parser.add_argument("--species_list", type=str, nargs='+', required=True, help="Species list.")
    args = parser.parse_args()

    generate_species_description(args.species_list)


