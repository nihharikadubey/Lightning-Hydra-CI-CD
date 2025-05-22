import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import lightning as L
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require the root directory to be set
from src.utils.logging_utils import setup_logger, task_wrapper
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
def train(trainer, model, datamodule):
    trainer.fit(model, datamodule)
    logging.info(f"Training metrics: {trainer.callback_metrics}")

@task_wrapper
def evaluate(trainer, model, datamodule, ckpt_path=None):  # Renamed from 'test' to 'evaluate'
    if ckpt_path:
        trainer.test(model, datamodule, ckpt_path=ckpt_path)
    else:
        trainer.test(model, datamodule)
    logging.info(f"Test metrics: {trainer.callback_metrics}")

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule
    logging.info("Initializing DataModule")
    data_module = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    logging.info("Initializing Model")
    model = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    logging.info(f"Initialized {len(callbacks)} callbacks")

    # Set up callbacks.rich_progress_bar
    loggers = instantiate_loggers(cfg.get("logger"))
    logging.info(f"Initialized {len(loggers)} loggers")

    # Initialize Trainer
    logging.info("Initializing Trainer")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    logging.info("Starting training")
    train(trainer, model, data_module)

    # Test the model
    logging.info("Starting testing")
    best_model_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, 'checkpoint_callback') else None
    if best_model_path:
        logging.info(f"Testing with best model checkpoint: {best_model_path}")
        evaluate(trainer, model, data_module, ckpt_path=best_model_path)
    else:
        logging.info("No best model checkpoint found. Testing with current model weights.")
        evaluate(trainer, model, data_module)

if __name__ == "__main__":
    main()