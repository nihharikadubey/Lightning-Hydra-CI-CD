import pytest
import hydra
from omegaconf import DictConfig
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
import rootutils
import subprocess
from PIL import Image
import matplotlib.pyplot as plt

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.infer import main, instantiate_callbacks, instantiate_loggers, load_image, infer, save_prediction_image, run_inference

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="infer",
            overrides=["experiment=catdog_ex", "+trainer.fast_dev_run=True"],
        )
        return cfg

@pytest.fixture
def tmp_path_config(config, tmp_path):
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")
    config.input_folder = str(tmp_path / "input")
    config.output_folder = str(tmp_path / "output")
    config.ckpt_path = str(tmp_path / "model.ckpt")
    return config

@pytest.fixture
def cat_image(tmp_path):
    # URL of a sample cat image
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    image_path = tmp_path / "cat.jpg"
    
    # Download the image using wget
    subprocess.run(["wget", "-O", str(image_path), image_url], check=True)
    
    return image_path

def test_instantiate_callbacks(tmp_path_config):
    callbacks = instantiate_callbacks(tmp_path_config.get("callbacks"))
    assert isinstance(callbacks, list)
    # Add more specific assertions based on your callback configurations

def test_instantiate_loggers(tmp_path_config):
    loggers = instantiate_loggers(tmp_path_config.get("logger"))
    assert isinstance(loggers, list)
    # Add more specific assertions based on your logger configurations

def test_load_image(cat_image):
    img, img_tensor = load_image(cat_image)
    
    assert isinstance(img, Image.Image)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (1, 3, 224, 224)
    
    # Additional checks
    assert img.mode == 'RGB'
    # assert img_tensor.min() >= 0 and img_tensor.max() <= 1  # Check if normalized

def test_infer():
    model = MagicMock()
    model.eval.return_value = None
    model.return_value = torch.tensor([[0.3, 0.7]])
    
    image_tensor = torch.rand(1, 3, 224, 224)
    
    predicted_label, confidence = infer(model, image_tensor)
    
    assert predicted_label in ['cat', 'dog']
    assert 0 <= confidence <= 1

def test_save_prediction_image(cat_image, tmp_path):
    # Load the image using PIL
    image = Image.open(cat_image)
    
    predicted_label = "cat"
    confidence = 0.9
    output_path = tmp_path / "prediction.png"
    
    # Call the function
    save_prediction_image(image, predicted_label, confidence, output_path)
    
    # Check if the output file was created
    assert output_path.exists()
    
    # Open the saved image and check its properties
    saved_image = Image.open(output_path)
    assert saved_image.format == 'PNG'
    
    # Clean up
    plt.close()  # Close any open matplotlib figures

def test_main_function_components(tmp_path_config, cat_image):
    with patch("src.infer.hydra.utils.instantiate") as instantiate_mock, \
         patch("src.infer.instantiate_callbacks") as callbacks_mock, \
         patch("src.infer.instantiate_loggers") as loggers_mock, \
         patch("src.infer.run_inference") as run_inference_mock, \
         patch("src.infer.logging") as logging_mock:

        instantiate_mock.side_effect = [MagicMock(), MagicMock()]
        callbacks_mock.return_value = [MagicMock()]
        loggers_mock.return_value = [MagicMock()]

        # Copy the cat image to the input folder
        input_folder = Path(tmp_path_config.input_folder)
        input_folder.mkdir(parents=True)
        dest = input_folder / "image1.jpg"
        dest.write_bytes(cat_image.read_bytes())

        main(tmp_path_config)

        assert instantiate_mock.call_count == 2  # Model, Trainer
        callbacks_mock.assert_called_once()
        loggers_mock.assert_called_once()
        run_inference_mock.assert_called_once()
        assert logging_mock.info.call_count >= 5  # At least 5 log messages

def test_main_without_checkpoint(tmp_path_config):
    tmp_path_config.ckpt_path = None
    with patch("src.infer.logging") as logging_mock:
        main(tmp_path_config)
        logging_mock.error.assert_called_with("No checkpoint path provided in the configuration.")

def test_catdog_ex_inference(tmp_path_config, cat_image):
    # Create a dummy checkpoint file
    Path(tmp_path_config.ckpt_path).touch()
    
    # Create input folder and copy the cat image
    input_folder = Path(tmp_path_config.input_folder)
    input_folder.mkdir(parents=True)
    for i in range(2):
        dest = input_folder / f"image{i+1}.jpg"
        dest.write_bytes(cat_image.read_bytes())

    with patch("src.infer.run_inference") as run_inference_mock:
        main(tmp_path_config)
        run_inference_mock.assert_called_once()

def test_run_inference(tmp_path_config, cat_image):
    # Create a mock model
    mock_model = MagicMock()
    mock_model.device = 'cpu'
    
    # Create a mock ModelClass
    mock_model_class = MagicMock()
    mock_model_class.load_from_checkpoint.return_value = mock_model

    # Prepare input and output folders
    input_folder = Path(tmp_path_config.input_folder)
    input_folder.mkdir(parents=True)
    output_folder = Path(tmp_path_config.output_folder)
    output_folder.mkdir(parents=True)

    # Copy the cat image to the input folder
    dest = input_folder / "test_image.jpg"
    dest.write_bytes(cat_image.read_bytes())

    with patch("src.infer.hydra.utils.get_class", return_value=mock_model_class), \
         patch("src.infer.load_image", return_value=(MagicMock(), torch.rand(1, 3, 224, 224))), \
         patch("src.infer.infer", return_value=("cat", 0.9)), \
         patch("src.infer.save_prediction_image"), \
         patch("src.infer.logging"):
        
        run_inference(tmp_path_config)

        # Assert that the model was loaded from the checkpoint
        mock_model_class.load_from_checkpoint.assert_called_once_with(tmp_path_config.ckpt_path)
        
        # Assert that load_image was called
        assert Path(tmp_path_config.input_folder, "test_image.jpg").exists()
        
        # Assert that infer was called
        assert mock_model.eval.called
        
        # Assert that save_prediction_image was called
        # assert Path(tmp_path_config.output_folder, "test_image_prediction.png").exists()