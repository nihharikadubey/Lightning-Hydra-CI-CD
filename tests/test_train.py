import pytest
import hydra
import lightning as L
from omegaconf import DictConfig
from unittest.mock import patch, MagicMock
from pathlib import Path
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.train import main, instantiate_callbacks, instantiate_loggers, train, evaluate  # Changed 'test' to 'evaluate'

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=catdog_ex", "+trainer.fast_dev_run=True"],
        )
        return cfg

@pytest.fixture
def tmp_path_config(config, tmp_path):
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    return config

def test_instantiate_callbacks(tmp_path_config):
    callbacks = instantiate_callbacks(tmp_path_config.get("callbacks"))
    assert isinstance(callbacks, list)
    # Add more specific assertions based on your callback configurations

def test_instantiate_loggers(tmp_path_config):
    loggers = instantiate_loggers(tmp_path_config.get("logger"))
    assert isinstance(loggers, list)
    # Add more specific assertions based on your logger configurations

@pytest.mark.parametrize("ckpt_path", [None, "dummy_path.ckpt"])
def test_train_and_evaluate_functions(tmp_path_config, ckpt_path):  # Changed 'test' to 'evaluate' in function name
    trainer_mock = MagicMock()
    model_mock = MagicMock()
    datamodule_mock = MagicMock()

    with patch("src.train.logging") as logging_mock:
        train(trainer_mock, model_mock, datamodule_mock)
        trainer_mock.fit.assert_called_once_with(model_mock, datamodule_mock)
        logging_mock.info.assert_called_with(f"Training metrics: {trainer_mock.callback_metrics}")

        evaluate(trainer_mock, model_mock, datamodule_mock, ckpt_path=ckpt_path)  # Changed 'test' to 'evaluate'
        if ckpt_path:
            trainer_mock.test.assert_called_with(model_mock, datamodule_mock, ckpt_path=ckpt_path)
        else:
            trainer_mock.test.assert_called_with(model_mock, datamodule_mock)
        logging_mock.info.assert_called_with(f"Test metrics: {trainer_mock.callback_metrics}")

def test_main_function_components(tmp_path_config):
    with patch("src.train.hydra.utils.instantiate") as instantiate_mock, \
         patch("src.train.instantiate_callbacks") as callbacks_mock, \
         patch("src.train.instantiate_loggers") as loggers_mock, \
         patch("src.train.train") as train_mock, \
         patch("src.train.evaluate") as evaluate_mock, \
         patch("src.train.logging") as logging_mock:

        instantiate_mock.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        callbacks_mock.return_value = [MagicMock()]
        loggers_mock.return_value = [MagicMock()]

        main(tmp_path_config)

        assert instantiate_mock.call_count == 3  # DataModule, Model, Trainer
        callbacks_mock.assert_called_once()
        loggers_mock.assert_called_once()
        train_mock.assert_called_once()
        evaluate_mock.assert_called_once()
        assert logging_mock.info.call_count >= 6  # At least 6 log messages

def test_catdog_ex_training(tmp_path_config):
    main(tmp_path_config)
    # Add assertions to check if the training was successful
    # For example, check if output files were created in the temporary directory
    assert Path(tmp_path_config.paths.output_dir, "logs").exists()
