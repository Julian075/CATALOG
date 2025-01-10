from transformers import LlamaForCausalLM, LlamaTokenizer

# Cargar el modelo y el tokenizador
model_name = "meta-llama/Llama-2-7b-hf"  # Ejemplo: LLaMA 2 de 7B parámetros
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Entrada y generación de texto
input_text = "¿Por qué es importante la biodiversidad?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

# Mostrar resultado
print(tokenizer.decode(outputs[0]))
