# CATALOG

Foundation Models (FMs) have been successful in various computer vision tasks like image classification, object detection, and image segmentation. However, these tasks remain challenging when these models are tested on datasets with different distributions from the training dataset, a problem known as domain shift. This is especially problematic for recognizing animal species in camera-trap images where variability in factors like lighting, camouflage, and occlusions exists.

In this repository, we present the **Ca**mera **T**r**a**p **L**anguage-guided C**o**ntrastive Learnin**g** (CATALOG) model to address these issues. CATALOG combines multiple FMs to extract visual and textual features from camera-trap data and employs a contrastive loss function to train the model. 

Our approach has been evaluated on two benchmark datasets, **Snapshot Serengeti** and **Terra Incognita**, and achieves state-of-the-art results in camera-trap image recognition, particularly in scenarios where the training and testing data have different animal species or originate from different environments. This demonstrates the potential of leveraging FMs in combination with multi-modal fusion and contrastive learning to tackle domain shifts in camera-trap image recognition tasks.

The work has been accepted for presentation at **WACV 2025**.

## Repository Structure

This repository is organized as follows:

- **`data/`**: This folder contains the datasets to be tested. Place your datasets here before running experiments.
- **`feature_extraction/`**: Provides the scripts for offline feature extraction from the datasets, tailored for different versions of the model.
- **`features/`**: Stores the extracted feature files. Ensure this directory is populated with features before running the training or testing scripts.
- **`models/`**: Contains the pre-trained models and the architecture definitions for various model versions.
- **`train/`**: Includes the training scripts for different versions of the model.
- **`main.py`**: A central script to execute the workflows (training and testing) with command-line arguments.

### Using `main.py`

The `main.py` script allows you to train and test models conveniently. Below is an example of how to use it:

#### Command to Train a Model
```bash
python main.py --model_version <Model_Type> --train_type <Training_Type> --dataset <Dataset> --mode train


