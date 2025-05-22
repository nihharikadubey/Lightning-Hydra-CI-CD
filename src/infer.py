import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import lightning as L
import rootutils
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require the root directory to be set
from src.utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
import logging

def instantiate_callbacks(callback_cfg):
    callbacks = []
    if callback_cfg:
        for _, cb_conf in callback_cfg.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def instantiate_loggers(logger_cfg):
    loggers = []
    if logger_cfg:
        for _, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ['cat', 'dog']
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@task_wrapper
def run_inference(cfg: DictConfig):
    # Remove this line:
    # model = hydra.utils.instantiate(cfg.model)
    
    # Replace with:
    ModelClass = hydra.utils.get_class(cfg.model._target_)
    model = ModelClass.load_from_checkpoint(cfg.ckpt_path)
    model.eval()

    input_folder = Path(cfg.input_folder)
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    image_files = list(input_folder.glob('*'))
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))
        
        for image_file in image_files:
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device))
                
                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)
                
                logging.info(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                progress.advance(task)

@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    # Set up logger
    setup_logger(log_dir / "infer_log.log")

    # Initialize Model
    logging.info("Initializing Model")
    model = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    logging.info(f"Initialized {len(callbacks)} callbacks")

    # Set up loggers
    loggers = instantiate_loggers(cfg.get("logger"))
    logging.info(f"Initialized {len(loggers)} loggers")

    # Initialize Trainer
    logging.info("Initializing Trainer")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Run inference
    logging.info("Starting inference")
    if cfg.ckpt_path:
        logging.info(f"Using model checkpoint: {cfg.ckpt_path}")
        run_inference(cfg)
    else:
        logging.error("No checkpoint path provided in the configuration.")

if __name__ == "__main__":
    main()