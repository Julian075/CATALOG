import os
import shutil
from pathlib import Path
import random

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset(images_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """Split dataset into train/val/test sets while maintaining category distribution"""
    print("Starting dataset split...")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        create_directory(dir_path)
    
    # Get all category directories
    categories = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    
    total_images = 0
    for category in categories:
        print(f"\nProcessing category: {category}")
        
        # Get all images in this category
        category_dir = os.path.join(images_dir, category)
        images = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split the images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create category directories in each split
        for split_dir in [train_dir, val_dir, test_dir]:
            category_dir = os.path.join(split_dir, category)
            create_directory(category_dir)
        
        # Move images to respective directories
        for split_name, split_images in [
            ('train', train_images),
            ('val', val_images),
            ('test', test_images)
        ]:
            split_dir = os.path.join(output_dir, split_name, category)
            for image in split_images:
                source_path = os.path.join(images_dir, category, image)
                dest_path = os.path.join(split_dir, image)
                shutil.move(source_path, dest_path)
                total_images += 1
                if total_images % 100 == 0:
                    print(f"Processed {total_images} images...")
        
        # Remove empty category directory
        if os.path.exists(category_dir) and not os.listdir(category_dir):
            os.rmdir(category_dir)
    
    print(f"\nDataset split complete!")
    print(f"Total images processed: {total_images}")
    
    # Print statistics
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split_name)
        n_images = sum(len(files) for _, _, files in os.walk(split_dir))
        print(f"{split_name.capitalize()} set: {n_images} images")

def main():
    # Define paths
    images_dir = './data/unsw_predators/img'
    output_dir = './data/unsw_predators/img_split'
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Split dataset
    split_dataset(images_dir, output_dir)

if __name__ == "__main__":
    main() 