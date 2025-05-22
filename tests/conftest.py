import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

@pytest.fixture(scope="session")
def config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train")
    return cfg

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
