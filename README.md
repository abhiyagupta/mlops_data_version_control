# CatDog Classification with PyTorch Lightning, Hydra, and DVC

This project implements a deep learning model to classify images of cats and dogs using PyTorch Lightning, Hydra for configuration management, and DVC for data version control.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
  - [Hydra Configuration Structure](#hydra-configuration-structure)
- [Configuration](#configuration)
- [Data Version Control](#data-version-control)

<!--
- [Training Results](#training-results)
- [Contributing](#contributing)
- [License](#license)
-->

## Project Overview

The CatDog Classification project aims to:
- Implement a Vision Transformer (ViT) model for image classification
- Utilize PyTorch Lightning for efficient and organized model training
- Use Hydra for flexible configuration management
- Employ DVC for data and model versioning

We use the `vit_tiny_patch16_224` model architecture for this classification task.

## Setup

### Prerequisites
- Python 3.7+
- uv (for package management)
- PyTorch
- PyTorch Lightning
- Hydra
- DVC

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/catdog-classification.git
   cd catdog-classification
   ```

2. Install uv:
   ```
   pip install uv
   ```

3. Create a virtual environment and install dependencies using uv:
   ```
   uv venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

4. Set up DVC:
   ```
   dvc init
   dvc remote add -d myremote /path/to/remote/storage
   ```

## Usage

1. Prepare your dataset:
   ```
   dvc add data/
   ```

2. Train the model:
   ```
   python src/train.py
   ```

3. Evaluate the model:
   ```
   python src/eval.py
   ```



### Hydra Configuration Structure

```bash
├── callbacks
│   ├── default.yaml
│   ├── early_stopping.yaml
│   ├── model_checkpoint.yaml
│   ├── model_summary.yaml
│   └── rich_progress_bar.yaml
├── data
│   └── catdog.yaml
├── eval.yaml
├── experiment
│   └── catdog_ex.yaml
├── hydra
│   └── default.yaml
├── infer.yaml
├── logger
│   ├── csv.yaml
│   ├── default.yaml
│   └── tensorboard.yaml
├── model
│   └── timm_classify.yaml
├── paths
│   └── default.yaml
├── train.yaml
└── trainer
    └── default.yaml
```

## Configuration

This project uses Hydra for configuration management. The configuration files are organized in the structure shown above. Key configuration files include:

- `experiment/catdog_ex.yaml`: Main experiment configuration
- `model/timm_classify.yaml`: Model-specific configuration for the `vit_tiny_patch16_224`
- `data/catdog.yaml`: Dataset configuration
- `callbacks/`: Various callback configurations for training
- `logger/`: Logging configurations
- `trainer/default.yaml`: PyTorch Lightning Trainer configuration

You can override configuration values using command-line arguments or by modifying the YAML files.

## Data Version Control

DVC is used to version the dataset and trained models. To pull the latest data:

```
dvc pull
```
<!--
## Training Results

![Training Results](path/to/training_results_image.png)

*Figure: Training accuracy and loss over epochs*

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
-->
