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
def evaluate(trainer, model, datamodule, ckpt_path):
    if ckpt_path:
        trainer.test(model, datamodule, ckpt_path=ckpt_path)
    else:
        logging.error("No checkpoint path provided. Cannot evaluate the model.")
        return
    logging.info(f"Evaluation metrics: {trainer.callback_metrics}")

@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    logging.info("Initializing DataModule")
    data_module = hydra.utils.instantiate(cfg.data)

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

    # Evaluate the model
    logging.info("Starting evaluation")
    if cfg.ckpt_path:
        logging.info(f"Evaluating with model checkpoint: {cfg.ckpt_path}")
        evaluate(trainer, model, data_module, ckpt_path=cfg.ckpt_path)
    else:
        logging.error("No checkpoint path provided in the configuration.")

if __name__ == "__main__":
    main()
