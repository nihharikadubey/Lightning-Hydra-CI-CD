import pytest
import hydra
from omegaconf import DictConfig
from unittest.mock import patch, MagicMock
from pathlib import Path
import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.eval import main, instantiate_callbacks, instantiate_loggers, evaluate

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="eval",
            overrides=["experiment=catdog_ex", "+trainer.fast_dev_run=True"],
        )
        return cfg

@pytest.fixture
def tmp_path_config(config, tmp_path):
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    config.ckpt_path = str(tmp_path / "dummy_checkpoint.ckpt")
    return config

def test_instantiate_callbacks(tmp_path_config):
    callbacks = instantiate_callbacks(tmp_path_config.get("callbacks"))
    assert isinstance(callbacks, list)
    # Add more specific assertions based on your callback configurations

def test_instantiate_loggers(tmp_path_config):
    loggers = instantiate_loggers(tmp_path_config.get("logger"))
    assert isinstance(loggers, list)
    # Add more specific assertions based on your logger configurations

def test_evaluate_function(tmp_path_config):
    trainer_mock = MagicMock()
    model_mock = MagicMock()
    datamodule_mock = MagicMock()
    ckpt_path = tmp_path_config.ckpt_path

    with patch("src.eval.logging") as logging_mock:
        evaluate(trainer_mock, model_mock, datamodule_mock, ckpt_path)
        trainer_mock.test.assert_called_once_with(model_mock, datamodule_mock, ckpt_path=ckpt_path)
        logging_mock.info.assert_called_with(f"Evaluation metrics: {trainer_mock.callback_metrics}")

def test_main_function_components(tmp_path_config):
    with patch("src.eval.hydra.utils.instantiate") as instantiate_mock, \
         patch("src.eval.instantiate_callbacks") as callbacks_mock, \
         patch("src.eval.instantiate_loggers") as loggers_mock, \
         patch("src.eval.evaluate") as evaluate_mock, \
         patch("src.eval.logging") as logging_mock:

        instantiate_mock.side_effect = [MagicMock(), MagicMock(), MagicMock()]
        callbacks_mock.return_value = [MagicMock()]
        loggers_mock.return_value = [MagicMock()]

        main(tmp_path_config)

        assert instantiate_mock.call_count == 3  # DataModule, Model, Trainer
        callbacks_mock.assert_called_once()
        loggers_mock.assert_called_once()
        evaluate_mock.assert_called_once()
        assert logging_mock.info.call_count >= 5  # At least 5 log messages

def test_eval_without_checkpoint(tmp_path_config):
    tmp_path_config.ckpt_path = None
    with patch("src.eval.logging") as logging_mock:
        main(tmp_path_config)
        logging_mock.error.assert_called_with("No checkpoint path provided in the configuration.")

# def test_catdog_ex_evaluation(tmp_path_config):
#     # Use a checkpoint file path within the temporary directory
#     ckpt_path = Path(tmp_path_config.paths.output_dir) / "checkpoints" / "last.ckpt"
#     ckpt_path.parent.mkdir(parents=True, exist_ok=True)
#     ckpt_path.touch()

#     # Update the config with the new checkpoint path
#     tmp_path_config.ckpt_path = str(ckpt_path)

#     main(tmp_path_config)
#     # Add assertions to check if the evaluation was successful
#     # For example, check if output files were created in the temporary directory
#     assert Path(tmp_path_config.paths.output_dir, "logs").exists()
#     assert Path(tmp_path_config.paths.log_dir, "eval_log.log").exists()