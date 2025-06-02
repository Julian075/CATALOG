import json
import os
from PIL import Image
import shutil
from pathlib import Path

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def crop_image(image_path, bbox, output_path, image_width, image_height):
    """Crop image based on COCO format bbox coordinates [x, y, width, height]"""
    try:
        with Image.open(image_path) as img:
            # Convert normalized bbox coordinates to pixel values
            x = int(bbox[0] * image_width)
            y = int(bbox[1] * image_height)
            width = int(bbox[2] * image_width)
            height = int(bbox[3] * image_height)
            
            # Calculate right and bottom coordinates
            right = x + width
            bottom = y + height
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            right = min(image_width, right)
            bottom = min(image_height, bottom)
            
            # Crop the image
            cropped_img = img.crop((x, y, right, bottom))
            
            # Save the cropped image
            cropped_img.save(output_path)
            return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def load_category_mapping(predators_json):
    """Load category ID to name mapping"""
    categories = {}
    for cat in predators_json.get('categories', []):
        categories[cat['id']] = cat['name']
    return categories

def process_images(predators_json_path, md_json_path, images_dir, output_dir):
    """Process images based on both JSON files"""
    print("Loading JSON files...")
    
    # Load predators JSON
    with open(predators_json_path, 'r') as f:
        predators_data = json.load(f)
    
    # Load MegaDetector JSON
    with open(md_json_path, 'r') as f:
        md_data = json.load(f)
    
    # Create category mapping
    category_mapping = load_category_mapping(predators_data)
    
    # Create output directory
    create_directory(output_dir)
    
    # Create a mapping of image_id to its annotations
    image_annotations = {}
    for ann in predators_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Create a mapping of image file to its detections
    image_detections = {}
    for img in md_data['images']:
        image_detections[img['file']] = img['detections']
    
    # Process images
    print("Processing images...")
    processed_count = 0
    error_count = 0
    
    # Get image dimensions from predators JSON
    image_metadata = {img['file_name']: {'width': img['width'], 'height': img['height']} 
                     for img in predators_data['images']}
    
    for image_info in predators_data['images']:
        try:
            image_filename = image_info['file_name']
            
            # Skip if no annotations or detections
            if image_filename not in image_annotations or image_filename not in image_detections:
                continue
            
            # Get image dimensions
            width = image_info['width']
            height = image_info['height']
            
            # Process each annotation
            for ann in image_annotations[image_filename]:
                category_id = ann['category_id']
                category_name = category_mapping.get(category_id, f"unknown_{category_id}")
                
                # Create category directory
                category_dir = os.path.join(output_dir, category_name)
                create_directory(category_dir)
                
                # Source image path
                source_path = os.path.join(images_dir, image_filename)
                
                if not os.path.exists(source_path):
                    print(f"Image not found: {source_path}")
                    error_count += 1
                    continue
                
                # Get detections for this image
                detections = image_detections[image_filename]
                
                # Process each detection
                for i, detection in enumerate(detections):
                    if detection['category'] != '1':  # Skip non-animal detections
                        continue
                    
                    # Get bounding box and confidence
                    bbox = detection['bbox']
                    confidence = detection['conf']
                    
                    # Create output filename
                    base_name = os.path.splitext(os.path.basename(image_filename))[0]
                    output_filename = f"{base_name}_det{i+1}_conf{confidence:.2f}.jpg"
                    output_path = os.path.join(category_dir, output_filename)
                    
                    if crop_image(source_path, bbox, output_path, width, height):
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} detections...")
        
        except Exception as e:
            print(f"Error processing image {image_filename}: {str(e)}")
            error_count += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} detections")
    print(f"Errors encountered: {error_count}")

def main():
    # Define paths
    predators_json_path = './data/unsw_predators/unsw-predators.json'
    md_json_path = './data/unsw_predators/unsw-goannas.json'
    images_dir = './data/unsw_predators/images'
    output_dir = './data/unsw_predators/img'
    
    # Process images
    process_images(predators_json_path, md_json_path, images_dir, output_dir)

if __name__ == "__main__":
    main()
