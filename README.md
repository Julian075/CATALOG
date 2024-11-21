# CATALOG

Foundation Models (FMs) have been successful in various computer vision tasks like image classification, object detection, and image segmentation. However, these tasks remain challenging when these models are tested on datasets with different distributions from the training dataset, a problem known as domain shift. This is especially problematic for recognizing animal species in camera-trap images where variability in factors like lighting, camouflage, and occlusions exists.

In this repository, we present the **Ca**mera **T**r**a**p **L**anguage-guided C**o**ntrastive Learnin**g** (CATALOG) model to address these issues. CATALOG combines multiple FMs to extract visual and textual features from camera-trap data and employs a contrastive loss function to train the model. 

Our approach has been evaluated on two benchmark datasets, **Snapshot Serengeti** and **Terra Incognita**, and achieves state-of-the-art results in camera-trap image recognition, particularly in scenarios where the training and testing data have different animal species or originate from different environments. This demonstrates the potential of leveraging FMs in combination with multi-modal fusion and contrastive learning to tackle domain shifts in camera-trap image recognition tasks.

The work has been accepted for presentation at **WACV 2025**.

## Features
- **Baseline Models**: Includes standard feature extraction and classification techniques.
- **Fine-Tuning**: Optimized models with domain-specific adjustments.
- **Multi-modal Feature Projections**: Integration of image and textual features.
- **Support for Long Features**: Extended capabilities for long-sequence feature analysis.

## Requirements
Before running the code, make sure you have the following dependencies installed:
- Python 3.8 or later
- PyTorch
- torchvision
- CLIP by OpenAI
- Additional libraries: `Pillow`, `argparse`

Install dependencies using:
```bash
pip install -r requirements.txt

