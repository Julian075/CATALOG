from huggingface_hub import login
import transformers
import torch
import os

# Inicia sesión con tu token de acceso
token = os.getenv("HF_TOKEN")
login(token)
# Configura el modelo y el pipeline
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

speacies_list=['badger', 'bird', 'bobcat', 'car', 'cat', 'coyote', 'deer', 'dog', 'empty', 'fox',
               'opossum', 'rabbit', 'raccoon', 'rodent', 'skunk','squirrel']
max_length=len(speacies_list)*200
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
The species are {speacies_list} give me the descriptions""", max_length=max_length)
print(output)
