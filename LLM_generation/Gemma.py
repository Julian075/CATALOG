from huggingface_hub import login
import transformers
import torch
import os
import argparse

def generate_species_description(species_list):
    # Inicia sesión con tu token de acceso
    token = os.getenv("HF_TOKEN")
    login(token)
    # Configura el modelo y el pipeline
    model_id = "google/gemma-2-2b-it"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )
    max_length=len(species_list)*200
    # Generación de texto
    output = pipeline(f"""You are an AI assistant specialized in biology and providing accurate and detailed descriptions of animal species. We
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
    The species are {species_list} provide detailed descriptions for each species in the same structure""", max_length=max_length)
    output_file = "Gemma_species_descriptions.txt"
    generated_text = output[0]["generated_text"]
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(generated_text)
#species_list=['badger', 'bird', 'bobcat', 'car', 'cat', 'coyote', 'deer', 'dog', 'empty', 'fox',  'opossum', 'rabbit', 'raccoon', 'rodent', 'skunk','squirrel']
#badger bird bobcat car cat coyote deer dog empty fox opossum rabbit raccoon rodent skunk squirrel
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract descriptions of animal species.")
    parser.add_argument("--species_list", type=str, nargs='+', required=True, help="Speacies list.")

    args = parser.parse_args()

    generate_species_description(args.species_list)


