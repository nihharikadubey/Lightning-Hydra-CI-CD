# Lightning Template with Hydra

This project is a template for machine learning projects using PyTorch Lightning and Hydra for configuration management. It includes a complete setup for training, evaluation, and inference of a cat vs dog image classification model.

## Project Structure

The project is organized as follows:

- `src/`: Contains the main source code
  - `datamodules/`: Data loading and processing
  - `models/`: Model definitions
  - `utils/`: Utility functions
- `tests/`: Unit tests for the project
- `configs/`: Hydra configuration files
- `Dockerfile`: For containerizing the application
- `pytest.ini`: PyTest configuration
- `pyproject.toml`: Project dependencies and metadata
- `.github/workflows/ci.yml`: GitHub Actions CI/CD pipeline

## Key Features

1. **Modular Architecture**: The project uses a modular architecture with separate modules for data, models, and utilities.

2. **Configuration Management**: Hydra is used for managing configurations, allowing easy experimentation with different hyperparameters and settings.

3. **Logging**: The project uses Loguru for advanced logging capabilities.

4. **Testing**: Comprehensive unit tests are included using PyTest.

5. **CI/CD**: A GitHub Actions workflow is set up for continuous integration and deployment.

6. **Docker Support**: The project can be containerized using Docker for easy deployment and reproducibility.

## Main Components

### Data Module

The `CatDogImageDataModule` in `src/datamodules/catdog_datamodule.py` handles data loading and preprocessing for the cat vs dog classification task.

### Model

The `TimmClassifier` in `src/models/timm_classifier.py` uses the TIMM library to create a flexible image classifier.

### Training

The training script is located in `src/train.py`. It uses PyTorch Lightning for efficient and organized training.

### Evaluation

The evaluation script is in `src/eval.py`, allowing for model evaluation on test data.

### Inference

The inference script in `src/infer.py` can be used to run predictions on new images.

## Configuration

The `configs/` directory contains various configuration files:

- `train.yaml`: Main training configuration
- `eval.yaml`: Evaluation configuration
- `infer.yaml`: Inference configuration
- `data/catdog.yaml`: Data-specific configuration
- `model/timm_classify.yaml`: Model-specific configuration
- `callbacks/`: Various callback configurations
- `logger/`: Logger configurations

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```
   uv sync
   ```
3. Run training:
   ```
   python src/train.py
   ```
4. Run evaluation:
   ```
   python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
   ```
5. Run inference:
   ```
   python src/infer.py ckpt_path=/path/to/checkpoint.ckpt input_folder=/path/to/images
   ```

## Testing

Run the tests using

```
pytest
```

## Docker

Build the Docker image:

```
docker build -t lightning-template-hydra .
```

Run the container:

```
docker run -it --rm lightning-template-hydra
