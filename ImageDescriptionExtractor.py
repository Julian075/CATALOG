import os
import argparse
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
import time
import torch
import gc

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def extract_description(path, dataset, time_b=0):
    time_b = int(time_b)
    
    # System prompt
    prompt = "[SYSTEM] You are an AI assistant specialized in biology and providing accurate and \
    detailed descriptions of animal species.\n<image>\nUSER: You are given the description of an animal species. Provide a very detailed\
    description of the appearance of the species and describe each body part of the animal\
    in detail. Only include details that can be directly visible in a photograph of the\
    animal. Only include information related to the appearance of the animal and nothing\
    else. Make sure to only include information that is present in the species description\
    and is certainly true for the given species. Do not include any information related\
    to the sound or smell of the animal. Do not include any numerical information related\
    to measurements in the text in units: m, cm, in, inches, ft, feet, km/h, kg, lb, lbs.\
    Remove any special characters such as unicode tags from the text. Return the answer as a\
    single paragraph.\nASSISTANT:"

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and processor
    model_llava = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
    
    if device != "cpu":
        model_llava = model_llava.to(device)
    
    modes = ['train', 'test', 'val']
    total_images = 0
    successful = 0
    
    for mode in modes:
        folders = os.listdir(os.path.join(path, mode))
        for folder in folders:
            new_folder = os.path.join(f'data/{dataset}/descriptions/', mode, folder)
            if os.path.exists(new_folder):
                image_names_old = [nombre[:-5] for nombre in os.listdir(new_folder)]
                image_names_aux = [nombre[:-4] for nombre in os.listdir(os.path.join(path, mode, folder))]
                image_names_aux = list(set(image_names_aux) - set(image_names_old))
                image_names = [nombre + '.jpg' for nombre in image_names_aux]
            else:
                image_names = os.listdir(os.path.join(path, mode, folder))
            
            total_images += len(image_names)

            for img_name in image_names:
                try:
                    # Clear GPU memory before processing
                    clear_gpu_memory()
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Process image
                    image = Image.open(os.path.join(path, mode, folder, img_name))
               inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                    
                    # Generate description
                    with torch.no_grad():
                        generate_ids = model_llava.generate(**inputs, max_new_tokens=300, min_length=200, do_sample=False)
                    
                    description = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
               _, description = description.split("ASSISTANT:")
                    
                    # Save information
               data = {
                   "description": description
               }
                    
                    # Save to JSON
                    os.makedirs(new_folder, exist_ok=True)
                    json_name = os.path.join(new_folder, img_name[:-4] + '.json')
               with open(json_name, "w") as json_file:
                   json.dump(data, json_file, indent=4)

                    # Calculate and print time if requested
                    end_time = time.time()
                    if time_b:
                        print(f'Time for prediction of {img_name}: {end_time - start_time:.2f} seconds')
                    
                    # Clean up
                    del inputs, generate_ids, description, data
                    successful += 1
                    
                except Exception as e:
                    print(f"Error processing {img_name}: {str(e)}")
    
    print(f"Successfully processed {successful} out of {total_images} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract descriptions of animal species from images.")
    parser.add_argument("--path", type=str, required=True, help="Path to the image folders.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--time", type=int, default=0, help="Print execution time for predictions (1 for true, 0 for false).")

    args = parser.parse_args()
    extract_description(args.path, args.dataset, time_b=args.time)

