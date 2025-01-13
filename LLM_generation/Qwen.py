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
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare the species as a comma-separated list
    species_str = ", ".join(species_list)

    # Construct the prompt
    prompt = f"""We are creating detailed and specific prompts to describe various species. The goal is to generate multiple sentences
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

    # Create messages in the format required for Qwen
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in biology and providing accurate and detailed descriptions of animal species."},
        {"role": "user", "content": prompt}
    ]

    # Apply the chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input and move to the device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the text
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=len(species_list) * 200
    )

    # Extract only the generated part of the text
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Save the generated text to a file
    output_file = "Qwen_species_descriptions2.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_text)

    print(f"Descriptions generated and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract descriptions of animal species.")
    parser.add_argument("--species_list", type=str, nargs='+', required=True, help="Species list.")
    args = parser.parse_args()

    generate_species_description(args.species_list)

