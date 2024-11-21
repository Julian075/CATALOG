
# CATALOG

This is the repository of the model **Camera Trap Language-guided Contrastive Learning (CATALOG)**, designed to address domain shift problems in camera-trap image recognition. CATALOG leverages multiple Foundation Models (FMs) for visual and textual feature extraction, combined with contrastive learning to enhance recognition accuracy across datasets with different animal species and environments. The work has been accepted at **WACV 2025**.

Foundation Models (FMs) have demonstrated significant success in computer vision tasks such as image classification, object detection, and image segmentation. However, domain shift remains a challenge, particularly in camera-trap image recognition due to factors like lighting, camouflage, and occlusions. CATALOG combines multi-modal fusion and contrastive learning to address these issues. It has been evaluated on **Snapshot Serengeti** and **Terra Incognita** datasets, showing state-of-the-art performance when training and testing data differ in animal species or environmental conditions.

The repository is organized as follows:
- **`data/`**: Folder to store the datasets to be tested.
- **`feature_extraction/`**: Scripts for offline feature extraction for different model versions.
- **`features/`**: Folder to store the extracted features used for training and testing.
- **`models/`**: Pre-trained models and architecture definitions.
- **`train/`**: Training scripts for different model configurations.
- **`main.py`**: Central script to run training and testing with command-line arguments.

Clone this repository:
```bash
git clone https://github.com/Julian075/CATALOG.git
```

Install the required Python libraries:
```bash
pip install -r requirements.txt
```

To train a specific model, use:
```bash
python main.py --model_version <Model_Type> --train_type <Training_Type> --dataset <Dataset> --mode train
```
Example:
```bash
python main.py --model_version Base --train_type In_domain --dataset serengeti --mode train
```

To test a trained model, use:
```bash
python main.py --model_version <Model_Type> --train_type <Training_Type> --dataset <Dataset> --mode test
```
Example:
```bash
python main.py --model_version Fine_tuning --train_type Out_domain --dataset terra --mode test
```

Command-Line Arguments:
| Argument         | Description                                         | Default         |
|------------------|-----------------------------------------------------|-----------------|
| `--model_version` | Specifies the model version (`Base`, `Fine_tuning`, etc.) | `"Fine_tuning"` |
| `--train_type`   | Specifies the type of training (`In_domain`, `Out_domain`) | `"In_domain"`   |
| `--dataset`      | Specifies the dataset to use (`serengeti`, `terra`) | `"serengeti"`   |
| `--mode`         | Specifies the mode (`train`, `test`, `test_top3`)   | `"train"`       |

To replicate results, ensure that the datasets are placed in the `data/` folder and features are precomputed in the `features/` folder.

Example Commands:
For **Snapshot Serengeti** (In-domain training):
```bash
python main.py --model_version Base --train_type In_domain --dataset serengeti --mode train
```
For **Terra Incognita** (Out-domain training):
```bash
python main.py --model_version Fine_tuning --train_type Out_domain --dataset terra --mode train
```

If you use this code, please cite our work:
```
@inproceedings{santamaria2025catalog,
  title={Camera Trap Language-guided Contrastive Learning (CATALOG)},
  author={Julian Santamaria},
  booktitle={WACV},
  year={2025}
}
```
## Acknowledgment.
This work was supported by Universidad de Antioquia - CODI and Alexander von Humboldt Institute for Research on Biological Resources. [code project: 2020-33250].
