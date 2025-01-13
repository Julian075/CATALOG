from huggingface_hub import login
import transformers
import torch
import os

# Inicia sesión con tu token de acceso
token = os.getenv("HF_TOKEN")
login(token)
#token = ""  # Reemplaza con tu token personal
#login(token)

# Configura el modelo y el pipeline
model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

# Generación de texto
output = pipeline("Hey, how are you doing today?", max_length=50)
print(output)
